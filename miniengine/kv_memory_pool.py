"""Pre-allocated paged KV cache memory pool — Milestone 2, Part A.

This is a SKELETON. Implement the methods below.

The pool owns a fixed amount of GPU memory, divided into equal-size
**pages**. Each page holds the KV state for `page_size` tokens for one
layer. Requests acquire pages as their KV grows and return them when
they finish; the cache itself never reallocates.

Storage layout (page-major vs token-major, contiguous K+V vs separate,
shape conventions, etc.) is YOUR design decision — pick something and
document the tradeoffs.
"""

from __future__ import annotations

import torch


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
        raise NotImplementedError

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve `num_pages` pages and return their indices.

        Raises if the pool cannot satisfy the request.
        """
        raise NotImplementedError

    def free(self, page_indices: list[int]) -> None:
        """Return the listed pages to the free pool."""
        raise NotImplementedError

    def pages_needed(self, seq_len: int) -> int:
        """How many pages are required to store `seq_len` tokens."""
        raise NotImplementedError

    @property
    def num_free(self) -> int:
        """Pages currently available for allocation."""
        raise NotImplementedError

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (K, V) cache tensors.

        The attention path holds references to these and indexes into
        them via per-request page tables. The exact shape is up to your
        design — but it must be STABLE: no reallocation, no resizing,
        no swapping out the tensors after construction.
        """
        raise NotImplementedError

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
        """Convenience: derive `num_pages` from a memory budget."""
        raise NotImplementedError
