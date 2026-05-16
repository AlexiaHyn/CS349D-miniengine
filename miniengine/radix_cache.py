"""Radix-tree prefix cache — Milestone 3, Part B.

Stores already-computed KV pages keyed by token prefix so a new request whose
prompt starts with a cached prefix can reuse those pages instead of
recomputing them.  This file is the **skeleton** — fill in the methods marked
``TODO``.

The data structure is a radix tree whose nodes own KV pages from the
``KVMemoryPool``.  Pages held by the cache are *not* in the pool's free list;
they return there only when the cache evicts them (LRU) or when an in-flight
insert chooses to free a redundant duplicate.

Performance counters in ``CacheMetrics`` are read by the ``/cache_stats``
endpoint and by the scheduler's per-prefill-batch INFO log line.  Update them
inside your implementation so those observability hooks light up.
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
    """A radix-tree node.  Design hints — adjust to fit your implementation:

    * ``parent`` / ``children`` form the tree.
    * ``key`` carries the tokens on the edge from the parent.
    * ``pages`` are the KV pages corresponding to those tokens; for safety
      against partial-page sharing, keep ``len(key)`` a multiple of
      ``page_size`` and ``len(pages) == len(key) // page_size``.
    * ``ref_count`` should reflect "number of locked leaves in this
      subtree" so eviction can check a single field.  Manipulated by
      ``inc_lock_ref`` / ``dec_lock_ref``.
    * ``last_access`` drives LRU.
    """

    __slots__ = ("parent", "children", "key", "pages", "ref_count", "last_access")

    def __init__(self) -> None:
        self.parent: RadixNode | None = None
        self.children: dict = {}
        self.key: list[int] = []
        self.pages: list[int] = []
        self.ref_count: int = 0
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
    last_node: RadixNode | None = None


class RadixCache:
    """Token-prefix → KV-pages cache backed by a radix tree.

    Required behaviours (see milestone 3 doc):
      * page-aligned matching — never return a partial-page result
      * LRU eviction of unlocked subtrees
      * eviction-on-allocate: ``KVMemoryPool.allocate`` should call
        ``cache.evict(n)`` when the free list is short
      * ``inc_lock_ref`` / ``dec_lock_ref`` protect in-flight requests
        (same names as sglang's radix cache).
    """

    def __init__(self, pool: "KVMemoryPool") -> None:
        self.pool = pool
        self.page_size = pool.page_size
        self.root = RadixNode()
        self.metrics = CacheMetrics()

    @property
    def num_cached_pages(self) -> int:
        """Total pages currently held by the tree."""
        raise NotImplementedError("TODO: track and report")

    def num_evictable_pages(self) -> int:
        """Pages that an LRU sweep could free right now."""
        raise NotImplementedError("TODO: walk the tree, count unlocked pages")

    # ── Lookup ─────────────────────────────────────────────────────────

    def match_prefix(self, tokens: list[int]) -> MatchResult:
        """Find the longest page-aligned prefix of ``tokens`` in the tree.

        Update ``metrics.total_lookups`` / ``total_query_tokens`` /
        ``total_hit_tokens`` so the perf counters are accurate.
        """
        raise NotImplementedError("TODO: walk the tree page by page")

    # ── Lock ref counting (sglang-style) ───────────────────────────────

    def inc_lock_ref(self, node: RadixNode | None) -> None:
        """Lock ``node`` (and the path to root) against eviction."""
        raise NotImplementedError("TODO: increment ref_count along the path")

    def dec_lock_ref(self, node: RadixNode | None) -> None:
        """Release a lock.  Refresh ``last_access`` while walking."""
        raise NotImplementedError("TODO: decrement ref_count along the path")

    # ── Insertion ──────────────────────────────────────────────────────

    def insert_and_return(
        self, tokens: list[int], pages: list[int]
    ) -> tuple[RadixNode, list[int]]:
        """Insert (tokens, pages) into the tree.

        Returns ``(leaf_node, redundant_pages)``: ``redundant_pages`` are
        pages the caller handed in that were duplicates of pages already
        cached at the same prefix — the caller should return them to the
        pool.  Update ``metrics.total_inserted_pages`` to reflect what
        actually got added.
        """
        raise NotImplementedError("TODO: walk down, split at page boundaries")

    # ── Eviction ───────────────────────────────────────────────────────

    def evict(self, n_pages_needed: int) -> int:
        """LRU-evict at least ``n_pages_needed`` pages (best effort).

        Return the number actually freed.  Bump
        ``metrics.total_evicted_pages``.  Never touch a locked node.
        """
        raise NotImplementedError("TODO: walk leaves, free oldest until N freed")

    # ── Maintenance ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop the whole tree, return every page to the pool."""
        raise NotImplementedError("TODO: walk + return_pages(...) + reset root")
