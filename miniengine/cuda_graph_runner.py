"""Manual CUDA graph capture for paged decode — Milestone 2 extra credit.

CUDA graphs replay a pre-recorded sequence of GPU ops in one launch,
removing per-op Python and driver overhead. They impose strict
invariants:

  - Stable tensor identities: reuse the same input objects every step;
    copy fresh data into them.
  - No CPU↔GPU sync inside the captured region (no .item(), no Python
    branches on tensor values, no lazy growth).
  - Fixed shapes per graph: capture one graph per bucket batch size and
    round live batches up to the nearest bucket.

We capture only the model forward — sampling stays outside the graph
since per-request top-k/top-p with multinomial + .item() would break
capture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from miniengine.core import Request
from miniengine.kv_memory_pool import PagedKVMeta
from miniengine.model import PagedAttnCtx
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


@dataclass
class _BucketGraph:
    batch_size: int
    max_pages: int
    graph: torch.cuda.CUDAGraph
    # Persistent input buffers (we copy_ into these every replay).
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    slot_mapping: torch.Tensor
    page_table: torch.Tensor
    lengths_after: torch.Tensor
    cu_seqlens_q: torch.Tensor
    # Persistent output buffer.
    logits: torch.Tensor


class CudaGraphRunner:
    """Captures one graph per (bucket_batch_size) for paged decode replay.

    `try_replay(requests)` returns sampled token_ids if it could route
    the call through a captured graph, else None (caller falls back to
    the eager paged_decode path).
    """

    def __init__(
        self,
        engine,
        bucket_batch_sizes: list[int],
        max_pages_per_request: int = 1024,
    ) -> None:
        self.engine = engine
        self.bucket_batch_sizes = sorted(set(bucket_batch_sizes))
        self.max_pages = max_pages_per_request
        self._graphs: dict[int, _BucketGraph] = {}
        self._captured = False

    # ── Capture ─────────────────────────────────────────────────────────

    def capture(self) -> None:
        """Capture all bucket graphs. Must be called after model is ready."""
        if self._captured:
            return
        if not torch.cuda.is_available():
            logger.warning("CUDA not available — skipping graph capture")
            return

        engine = self.engine
        device = engine.device
        # Warm up dynamo / kernels so first capture doesn't include init.
        for _ in range(3):
            self._dummy_forward(self.bucket_batch_sizes[0])

        for bs in self.bucket_batch_sizes:
            self._graphs[bs] = self._capture_one(bs)
            logger.info("Captured CUDA graph for batch_size=%d", bs)
        self._captured = True

    def _dummy_forward(self, batch_size: int) -> None:
        """Run a no-op decode to warm up before capture."""
        engine = self.engine
        pool = engine.kv_pool
        if pool is None or pool.num_free < batch_size:
            return
        # Allocate one page per dummy request, run a forward, free.
        page_indices = pool.allocate(batch_size)
        device = engine.device

        input_ids = torch.zeros(1, batch_size, dtype=torch.long, device=device)
        position_ids = torch.zeros(1, batch_size, dtype=torch.long, device=device)
        slot_mapping = torch.tensor(
            [pi * pool.page_size for pi in page_indices],
            dtype=torch.long, device=device,
        )
        page_table = torch.tensor(
            [[pi] for pi in page_indices], dtype=torch.int32, device=device
        )
        lengths_after = torch.ones(batch_size, dtype=torch.int32, device=device)
        cu_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device)

        ctx = PagedAttnCtx(
            kv_caches=pool.kv_caches,
            page_size=pool.page_size,
            page_indices=page_table,
            lengths_after=lengths_after,
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_q,
            is_decode=True,
        )
        with torch.inference_mode():
            engine.model(input_ids, position_ids, paged_ctx=ctx)
        torch.cuda.synchronize()
        pool.free(page_indices)

    def _capture_one(self, batch_size: int) -> _BucketGraph:
        engine = self.engine
        pool = engine.kv_pool
        device = engine.device
        max_pages = self.max_pages

        input_ids = torch.zeros(1, batch_size, dtype=torch.long, device=device)
        position_ids = torch.zeros(1, batch_size, dtype=torch.long, device=device)
        slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=device)
        page_table = torch.zeros(batch_size, max_pages, dtype=torch.int32, device=device)
        lengths_after = torch.ones(batch_size, dtype=torch.int32, device=device)
        cu_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device)

        # First run outside the graph to allocate workspaces.
        ctx = PagedAttnCtx(
            kv_caches=pool.kv_caches,
            page_size=pool.page_size,
            page_indices=page_table,
            lengths_after=lengths_after,
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_q,
            is_decode=True,
        )
        with torch.inference_mode():
            logits, _ = engine.model(input_ids, position_ids, paged_ctx=ctx)
        torch.cuda.synchronize()

        # Pre-allocate the output buffer that the graph will fill.
        logits_buf = torch.empty_like(logits)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.inference_mode():
                out_logits, _ = engine.model(input_ids, position_ids, paged_ctx=ctx)
                logits_buf.copy_(out_logits)

        return _BucketGraph(
            batch_size=batch_size,
            max_pages=max_pages,
            graph=graph,
            input_ids=input_ids,
            position_ids=position_ids,
            slot_mapping=slot_mapping,
            page_table=page_table,
            lengths_after=lengths_after,
            cu_seqlens_q=cu_q,
            logits=logits_buf,
        )

    # ── Replay ──────────────────────────────────────────────────────────

    def try_replay(self, requests: list[Request]) -> list[int] | None:
        """Replay a captured graph if the live batch fits one."""
        if not self._captured or not requests:
            return None
        live = len(requests)
        bucket = next((b for b in self.bucket_batch_sizes if b >= live), None)
        if bucket is None:
            return None
        bg = self._graphs.get(bucket)
        if bg is None:
            return None

        engine = self.engine
        pool = engine.kv_pool
        page_size = pool.page_size

        # Build inputs into the persistent buffers. Pad up to `bucket`
        # by replicating the first request's tensors (their outputs go
        # unused).
        input_ids_list: list[int] = []
        position_ids_list: list[int] = []
        slot_mapping_list: list[int] = []
        lengths_after = []
        page_indices_list: list[list[int]] = []

        for req in requests:
            meta: PagedKVMeta = req.kv_cache  # type: ignore[assignment]
            old_len = meta.length
            new_len = old_len + 1
            engine._allocate_pages_for(req, new_len)
            meta.length = new_len
            page_local = old_len // page_size
            offset = old_len % page_size
            slot = meta.page_indices[page_local] * page_size + offset
            slot_mapping_list.append(slot)
            input_ids_list.append(req.output_ids[-1])
            position_ids_list.append(old_len)
            lengths_after.append(new_len)
            page_indices_list.append(list(meta.page_indices))

        # Pad to bucket size.
        pad = bucket - live
        if pad > 0:
            input_ids_list.extend([input_ids_list[0]] * pad)
            position_ids_list.extend([position_ids_list[0]] * pad)
            slot_mapping_list.extend([slot_mapping_list[0]] * pad)
            lengths_after.extend([lengths_after[0]] * pad)
            page_indices_list.extend([page_indices_list[0]] * pad)

        # Copy into persistent buffers.
        bg.input_ids.copy_(
            torch.tensor([input_ids_list], dtype=torch.long, device=engine.device)
        )
        bg.position_ids.copy_(
            torch.tensor([position_ids_list], dtype=torch.long, device=engine.device)
        )
        bg.slot_mapping.copy_(
            torch.tensor(slot_mapping_list, dtype=torch.long, device=engine.device)
        )
        bg.lengths_after.copy_(
            torch.tensor(lengths_after, dtype=torch.int32, device=engine.device)
        )
        # Page table — write only the prefix of each row, leave the
        # rest at whatever the buffer holds (those entries are masked
        # by lengths_after anyway).
        bg.page_table.zero_()
        for i, pt in enumerate(page_indices_list):
            n = min(len(pt), bg.max_pages)
            bg.page_table[i, :n] = torch.tensor(
                pt[:n], dtype=torch.int32, device=engine.device
            )

        bg.graph.replay()

        # Sample only the live entries; padded ones are discarded.
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            tok = sample_token(
                bg.logits[:, i, :], req.sampling_params, req.output_ids
            )
            token_ids.append(tok)
        return token_ids
