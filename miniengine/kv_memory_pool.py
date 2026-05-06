"""Pre-allocated paged KV cache memory pool — Milestone 2, Part A.

This is a SKELETON. Implement the methods below.

The pool owns a fixed amount of GPU memory, divided into equal-size
**pages**. Each page holds the KV state for `page_size` tokens for one
layer. Requests acquire pages as their KV grows and return them when
they finish; the cache itself never reallocates.

Storage layout (page-major vs token-major, contiguous K+V vs separate,
shape conventions, etc.) is YOUR design decision — pick something and
document the tradeoffs.

Design choices
--------------

Storage layout (per layer):
    K, V: shape (num_pages, page_size, num_kv_heads, head_dim)

  Page-major (pages on dim 0) makes index gather across requests cheap
  via index_select(dim=0, ...) and keeps each page contiguous in memory.
  K and V are kept as separate tensors so attention kernels (and our
  SDPA fallback) can consume them directly.

Free-list:
    Plain Python list used as a LIFO stack. allocate()/free() are O(N)
    in the size of the request — small (a request needs ceil(L/P) pages,
    typically ≤ 100). Pages are returned to the tail of the free list,
    so recently freed pages get reused first (cache-friendly).

Page-table representation:
    Per-request `list[int]` of page indices, owned by the engine /
    request, NOT by the pool. The pool only owns the slabs and the free
    list.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PagedKVMeta:
    """Per-request page-table + length, stored on Request.kv_cache.

    The pool is not aware of these — the engine builds them per request
    and uses them when reading/writing KV through the pool.
    """

    page_indices: list[int]
    length: int  # number of valid tokens currently in the cache


class KVMemoryPool:
    """Pre-allocated paged KV cache pool.

    Args:
        num_pages:    Total pages in the pool (capacity).
        page_size:    Tokens per page. Tunable knob — exposed as
                      `--page-size` on the CLI. Smaller = less
                      fragmentation, bigger page tables; larger = the
                      opposite.
        num_layers:   Number of transformer layers.
        num_kv_heads: KV heads per layer (GQA).
        head_dim:     Per-head dimension.
        dtype:        KV dtype (typically bfloat16).
        device:       e.g. "cuda".
    """

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
        if num_pages <= 0:
            raise ValueError(f"num_pages must be positive, got {num_pages}")
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")

        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Per-layer (K, V) tensors. Page-major: (num_pages, page_size,
        # num_kv_heads, head_dim). Allocated up front; never resized.
        self._kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_layers):
            k = torch.zeros(
                num_pages, page_size, num_kv_heads, head_dim,
                dtype=dtype, device=device,
            )
            v = torch.zeros(
                num_pages, page_size, num_kv_heads, head_dim,
                dtype=dtype, device=device,
            )
            self._kv_caches.append((k, v))

        # Free list (LIFO). Reverse so we hand out pages 0,1,2,… first.
        self._free: list[int] = list(reversed(range(num_pages)))

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve `num_pages` pages and return their indices.

        Raises if the pool cannot satisfy the request.
        """
        if num_pages < 0:
            raise ValueError(f"Cannot allocate negative pages: {num_pages}")
        if num_pages > len(self._free):
            raise RuntimeError(
                f"KV pool out of pages: requested {num_pages}, "
                f"have {len(self._free)} free / {self.num_pages} total"
            )
        out = [self._free.pop() for _ in range(num_pages)]
        return out

    def free(self, page_indices: list[int]) -> None:
        """Return the listed pages to the free pool."""
        # Append to the tail — recently freed pages are reused soonest.
        self._free.extend(page_indices)

    def pages_needed(self, seq_len: int) -> int:
        """How many pages are required to store `seq_len` tokens."""
        if seq_len <= 0:
            return 0
        return (seq_len + self.page_size - 1) // self.page_size

    @property
    def num_free(self) -> int:
        """Pages currently available for allocation."""
        return len(self._free)

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (K, V) cache tensors.

        Stable for the life of the pool — attention reads/writes via
        per-request page tables index into these tensors.
        """
        return self._kv_caches


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
    ) -> KVMemoryPool:
        """Derive `num_pages` from a memory budget.

        bytes_budget is split equally across all layers, then across K
        and V. `bytes_per_page` covers ONE of (K, V) for ONE layer.
        """
        if bytes_budget <= 0:
            raise ValueError(f"bytes_budget must be positive, got {bytes_budget}")

        elem_size = torch.tensor([], dtype=dtype).element_size()
        bytes_per_page_one_tensor = (
            page_size * num_kv_heads * head_dim * elem_size
        )
        # 2 = K + V per layer.
        bytes_per_page_total = bytes_per_page_one_tensor * num_layers * 2
        num_pages = max(1, bytes_budget // bytes_per_page_total)

        return cls(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )


    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"KVMemoryPool(num_pages={self.num_pages}, page_size={self.page_size}, "
            f"num_layers={self.num_layers}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, free={self.num_free})"
        )
