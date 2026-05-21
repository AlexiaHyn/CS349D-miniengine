"""Radix-tree prefix cache — Milestone 3, Part B.

Stores already-computed KV pages keyed by token prefix so a new request whose
prompt starts with a cached prefix can reuse those pages instead of
recomputing them.

Design choice — **one node = one page**:
    Each non-root node's ``key`` is exactly ``page_size`` tokens and
    ``len(pages) == 1``.  This keeps insertion straightforward (no
    mid-page splits) and matches the spec's "match at page granularity"
    requirement.  Lookup hashes a full-page tuple at each step.

Lock-ref counting (sglang-style):
    ``ref_count`` on a node = "number of locked descendants in this
    subtree".  We increment along the path-to-root when a request
    borrows pages, decrement when it finishes.  Eviction skips any
    node whose own ``ref_count > 0`` *or* whose any-descendant lock is
    implied through its parent chain — by only evicting leaves with
    ``ref_count == 0``.

Pages held by the cache are *not* in the pool's free list.  The pool's
``allocate`` asks ``cache.evict(n)`` to release pages before raising OOM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miniengine.kv_memory_pool import KVMemoryPool

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Aggregate cache statistics — surfaced via ``/cache_stats``."""

    total_lookups: int = 0
    total_query_tokens: int = 0
    total_hit_tokens: int = 0
    total_inserted_pages: int = 0
    total_evicted_pages: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_query_tokens == 0:
            return 0.0
        return self.total_hit_tokens / self.total_query_tokens


class RadixNode:
    """A radix-tree node holding exactly one page of KV (or zero, for
    the root)."""

    __slots__ = ("parent", "children", "key", "pages", "ref_count", "last_access")

    def __init__(self) -> None:
        self.parent: RadixNode | None = None
        # children keyed by the full-page token tuple
        self.children: dict[tuple[int, ...], RadixNode] = {}
        self.key: tuple[int, ...] = ()       # tokens on edge from parent (page_size of them)
        self.pages: list[int] = []           # pool page ids (len 1 for non-root)
        self.ref_count: int = 0              # locked-descendants count
        self.last_access: float = time.monotonic()


@dataclass
class MatchResult:
    """Result of a prefix lookup.

    ``matched_tokens`` is page-aligned (multiple of ``page_size``);
    ``matched_pages`` carries the KV pages for those tokens.
    ``last_node`` is the deepest node the walk reached — callers lock it
    (``inc_lock_ref``) for the lifetime of the borrowing request.
    """

    matched_pages: list[int] = field(default_factory=list)
    matched_tokens: int = 0
    last_node: "RadixNode | None" = None


class RadixCache:
    """Token-prefix → KV-pages cache backed by a radix tree.

    Required behaviours:
      * page-aligned matching — never return a partial-page result
      * LRU eviction of unlocked subtrees
      * eviction-on-allocate: ``KVMemoryPool.allocate`` calls
        ``cache.evict(n)`` when the free list is short
      * ``inc_lock_ref`` / ``dec_lock_ref`` protect in-flight requests
    """

    def __init__(self, pool: "KVMemoryPool") -> None:
        self.pool = pool
        self.page_size = pool.page_size
        self.root = RadixNode()
        self.metrics = CacheMetrics()
        # Tracks total pages owned by the tree (not in pool free list).
        # Maintained explicitly so num_cached_pages is O(1).
        self._num_pages: int = 0

    @property
    def num_cached_pages(self) -> int:
        """Total pages currently held by the tree."""
        return self._num_pages

    def num_evictable_pages(self) -> int:
        """Pages that an LRU sweep could free right now.

        A page is evictable iff it lives in a subtree whose every leaf
        has ``ref_count == 0`` — equivalently, the node's own subtree
        contains no locked node.  We approximate cheaply by: a node's
        page is evictable iff that node has ``ref_count == 0``.  (A
        locked descendant pushes ``ref_count`` up along the path to
        root, so any ancestor of a locked node has ``ref_count > 0``
        and we won't try to evict it either.)
        """
        count = 0
        stack: list[RadixNode] = [self.root]
        while stack:
            node = stack.pop()
            if node is not self.root and node.ref_count == 0:
                count += len(node.pages)
            stack.extend(node.children.values())
        return count

    # ── Lookup ─────────────────────────────────────────────────────────

    def match_prefix(self, tokens: list[int]) -> MatchResult:
        """Find the longest page-aligned prefix of ``tokens`` in the tree."""
        self.metrics.total_lookups += 1
        self.metrics.total_query_tokens += len(tokens)

        matched_pages: list[int] = []
        node: RadixNode = self.root
        pos = 0
        ps = self.page_size
        while pos + ps <= len(tokens):
            block = tuple(tokens[pos : pos + ps])
            child = node.children.get(block)
            if child is None:
                break
            matched_pages.extend(child.pages)
            node = child
            pos += ps

        if pos > 0:
            # Bump access time on the deepest matched node so LRU keeps
            # frequently-hit prefixes alive.
            node.last_access = time.monotonic()
            self.metrics.total_hit_tokens += pos
            return MatchResult(matched_pages=matched_pages, matched_tokens=pos, last_node=node)
        return MatchResult()

    # ── Lock ref counting (sglang-style) ───────────────────────────────

    def inc_lock_ref(self, node: "RadixNode | None") -> None:
        """Lock ``node`` (and the path to root) against eviction."""
        while node is not None and node is not self.root:
            node.ref_count += 1
            node = node.parent

    def dec_lock_ref(self, node: "RadixNode | None") -> None:
        """Release a lock.  Refresh ``last_access`` while walking so a
        request that just finished bumps its prefix's LRU position."""
        now = time.monotonic()
        while node is not None and node is not self.root:
            node.ref_count -= 1
            node.last_access = now
            if node.ref_count < 0:
                # Defensive: should never go negative.  Log + clamp.
                logger.warning("RadixCache: ref_count went negative on a node; clamping")
                node.ref_count = 0
            node = node.parent

    # ── Insertion ──────────────────────────────────────────────────────

    def insert_and_return(
        self, tokens: list[int], pages: list[int]
    ) -> tuple[RadixNode, list[int]]:
        """Insert ``(tokens, pages)`` into the tree, page by page.

        Walks down the existing tree as long as page-aligned blocks
        match; appends new child nodes for any blocks past the match
        point.  Returns the deepest node touched and the list of caller
        pages that were redundant (already cached at the same prefix);
        the caller frees those back to the pool.

        ``tokens`` should already be page-aligned (a multiple of
        ``page_size``); we silently drop the partial tail since
        page-granular sharing is the spec requirement.
        """
        ps = self.page_size
        full_pages = len(tokens) // ps
        # The cache only stores complete pages — drop the tail.  Caller
        # decides what to do with the tail page (it's not cacheable
        # because its contents depend on the in-flight request).
        if full_pages > len(pages):
            # Caller handed us fewer pages than the page-aligned token
            # count requires — implies a bug upstream.  Be defensive.
            logger.warning(
                "RadixCache.insert: %d full pages of tokens but only %d pages provided; truncating",
                full_pages, len(pages),
            )
            full_pages = len(pages)

        redundant: list[int] = []
        node = self.root
        for i in range(full_pages):
            start = i * ps
            block = tuple(tokens[start : start + ps])
            existing = node.children.get(block)
            if existing is not None:
                # Same prefix already cached.  Two sub-cases:
                #   (a) caller is re-inserting a page they *borrowed*
                #       from the cache at admission time (same page id).
                #       The cache still owns this page; do NOT report
                #       it as redundant or the caller will free a page
                #       the cache is currently holding (double-free).
                #   (b) caller computed a fresh duplicate page (different
                #       page id, same contents).  Free the duplicate.
                if existing.pages and existing.pages[0] != pages[i]:
                    redundant.append(pages[i])
                existing.last_access = time.monotonic()
                node = existing
                continue
            # Create new child.
            new_node = RadixNode()
            new_node.parent = node
            new_node.key = block
            new_node.pages = [pages[i]]
            new_node.last_access = time.monotonic()
            node.children[block] = new_node
            self._num_pages += 1
            self.metrics.total_inserted_pages += 1
            node = new_node

        return node, redundant

    # ── Eviction ───────────────────────────────────────────────────────

    def evict(self, n_pages_needed: int) -> int:
        """LRU-evict at least ``n_pages_needed`` pages (best effort).

        Walk the tree, collect all unlocked leaves, sort by
        ``last_access``, and free pages from the oldest until we have
        ``n_pages_needed`` (or run out of unlocked leaves).  After a
        leaf is freed its parent may become a new leaf — we run a
        bounded number of sweeps so a long chain (multi-page request)
        can be freed end-to-end.
        """
        if n_pages_needed <= 0:
            return 0

        freed = 0
        # Cap sweeps so a pathological tree shape can't make this
        # unbounded.  In practice 2-3 sweeps cover any realistic depth.
        max_sweeps = 64
        for _ in range(max_sweeps):
            if freed >= n_pages_needed:
                break
            leaves = self._collect_unlocked_leaves()
            if not leaves:
                break
            leaves.sort(key=lambda n: n.last_access)
            progress = False
            for node in leaves:
                if freed >= n_pages_needed:
                    break
                # Re-check lock + leaf status; a concurrent ``inc_lock_ref``
                # could have happened (though our scheduler is single-thread,
                # we keep this safe).
                if node.ref_count != 0 or node.children:
                    continue
                if node.parent is None:
                    continue  # root sentinel, shouldn't happen for leaves
                # Free the page back to the pool and unlink the node.
                self.pool.free(node.pages)
                freed += len(node.pages)
                self._num_pages -= len(node.pages)
                self.metrics.total_evicted_pages += len(node.pages)
                del node.parent.children[node.key]
                node.parent = None
                progress = True
            if not progress:
                break

        return freed

    def _collect_unlocked_leaves(self) -> list[RadixNode]:
        """Return all leaf nodes whose ``ref_count == 0``.

        A leaf here is a tree leaf (no children); the root is never
        returned even when empty (we keep the sentinel).
        """
        leaves: list[RadixNode] = []
        stack: list[RadixNode] = [self.root]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children.values())
            else:
                if node is not self.root and node.ref_count == 0:
                    leaves.append(node)
        return leaves

    # ── Maintenance ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop the whole tree, return every page to the pool."""
        # DFS, collect all pages, then free in one batch + reset.
        pages: list[int] = []
        stack: list[RadixNode] = list(self.root.children.values())
        while stack:
            node = stack.pop()
            pages.extend(node.pages)
            stack.extend(node.children.values())
        if pages:
            self.pool.free(pages)
        self.root = RadixNode()
        self._num_pages = 0
