"""Pre-allocated paged KV cache memory pool — Milestone 2, Part A.

The pool owns a fixed amount of GPU memory, divided into equal-size
**pages**.  Each page holds the KV state for `page_size` tokens for one
layer.  Requests acquire pages as their KV grows and return them when
they finish; the cache itself never reallocates.

Storage layout
--------------
Per layer we keep two tensors

    K, V  :  (num_pages, page_size, num_kv_heads, head_dim)

i.e. page-major.  This shape was chosen because:

* `flash_attn_varlen_func` (and `flash_attn_with_kvcache`) accepts a
  *block table* whose entries are page indices into a tensor with this
  exact layout (`(num_pages, page_size, num_kv_heads, head_dim)`),
  so no reshape/transpose at the kernel boundary.
* `view(-1, num_kv_heads, head_dim)` flattens the first two dims into a
  global slot space, so a per-token `slot_mapping` (`page_idx * page_size
  + offset`) writes directly into the right cell with `index_copy_`.

Page 0 is reserved as a graph-padding scratch sink.  Under-full
CUDA-graph batches address this page at distinct offsets so their
throwaway K/V writes never clobber a real request's KV.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


class KVMemoryPool:
    """Pre-allocated paged KV cache pool.

    Args:
        num_pages:    Total pages in the pool (capacity).  Page 0 is
                      reserved as a CUDA-graph scratch sink, so the
                      number of pages actually handed out is
                      ``num_pages - 1``.
        page_size:    Tokens per page.  Tunable knob — exposed as
                      ``--page-size`` on the CLI.
        num_layers:   Number of transformer layers.
        num_kv_heads: KV heads per layer (GQA).
        head_dim:     Per-head dimension.
        dtype:        KV dtype (typically bfloat16).
        device:       e.g. ``"cuda"``.
    """

    SCRATCH_PAGE = 0  # never handed out — graph padding writes here

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        if num_pages < 2:
            raise ValueError(
                f"KV pool needs at least 2 pages (1 scratch + 1 usable), got {num_pages}"
            )

        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        shape = (num_pages, page_size, num_kv_heads, head_dim)
        self.k_caches = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self.v_caches = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]

        # Free list: every page except the reserved scratch page.
        self._free: list[int] = list(range(1, num_pages))

        # Milestone-3 Part B: optional radix cache the pool consults
        # when the free list is short.  Set via ``attach_cache``.
        self._cache = None  # type: ignore[var-annotated]

        elem = torch.tensor([], dtype=dtype).element_size()
        bytes_total = (
            2 * num_layers * num_pages * page_size * num_kv_heads * head_dim * elem
        )
        logger.info(
            "KVMemoryPool: %d pages × %d tokens (%.2f GB total, %d usable pages)",
            num_pages,
            page_size,
            bytes_total / 1e9,
            len(self._free),
        )

    # ── Allocation API ─────────────────────────────────────────────────

    def attach_cache(self, cache) -> None:
        """Wire a ``RadixCache`` so ``allocate`` can evict cached pages.

        Setting to ``None`` disables eviction-on-allocate.
        """
        self._cache = cache

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve ``num_pages`` pages and return their indices.

        If the free list is short, asks the attached ``RadixCache``
        (if any) to evict unlocked pages first.  Raises
        ``RuntimeError`` only when even after eviction we can't satisfy
        the request.
        """
        if num_pages < 0:
            raise ValueError(f"num_pages must be non-negative, got {num_pages}")
        if num_pages == 0:
            return []
        if len(self._free) < num_pages:
            # Ask the cache to drop LRU pages back to us.
            if self._cache is not None:
                deficit = num_pages - len(self._free)
                self._cache.evict(deficit)
        if len(self._free) < num_pages:
            raise RuntimeError(
                f"KV pool exhausted: requested {num_pages}, "
                f"have {len(self._free)} free pages"
            )
        out = self._free[:num_pages]
        self._free = self._free[num_pages:]
        return out

    def free(self, page_indices: list[int]) -> None:
        """Return the listed pages to the free pool."""
        if not page_indices:
            return
        self._free.extend(page_indices)

    def pages_needed(self, seq_len: int) -> int:
        """How many pages are required to store ``seq_len`` tokens."""
        if seq_len <= 0:
            return 0
        return math.ceil(seq_len / self.page_size)

    # ── Introspection ──────────────────────────────────────────────────

    @property
    def num_free(self) -> int:
        """Pages currently available for allocation (excludes scratch)."""
        return len(self._free)

    @property
    def num_evictable(self) -> int:
        """Pages the attached cache could evict on demand (0 if none)."""
        if self._cache is None:
            return 0
        return self._cache.num_evictable_pages()

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer ``(K, V)`` cache tensors.

        The attention path holds references to these and indexes into
        them via per-request page tables.  The shape is stable for the
        whole lifetime of the pool: no reallocation, no resizing, no
        swapping the tensors out after construction.
        """
        return list(zip(self.k_caches, self.v_caches))

    # ── Construction helpers ───────────────────────────────────────────

    @classmethod
    def from_budget(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        bytes_budget: int,
    ) -> "KVMemoryPool":
        """Convenience: derive ``num_pages`` from a memory budget.

        Picks the largest ``num_pages`` whose total K+V allocation fits
        in ``bytes_budget``.  A floor of 16 pages is enforced so the
        pool is always usable; if the budget is below that the caller
        should bump it (or surface an OOM later).
        """
        elem = torch.tensor([], dtype=dtype).element_size()
        bytes_per_page = (
            2 * num_layers * num_kv_heads * page_size * head_dim * elem
        )
        num_pages = max(16, bytes_budget // bytes_per_page)
        return cls(
            num_pages=int(num_pages),
            page_size=page_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
