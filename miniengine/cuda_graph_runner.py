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

This runner targets the flash-attn paged-kv decode path (see
`Engine.paged_decode`): persistent input buffers map 1:1 to the
`PagedMeta(phase="decode")` fields plus `input_ids` / `position_ids`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from miniengine.core import Request
from miniengine.model import PagedMeta
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


@dataclass
class _BucketGraph:
    batch_size: int
    max_pages: int
    graph: torch.cuda.CUDAGraph
    # Persistent input buffers (we copy_ into these every replay).
    input_ids: torch.Tensor      # (B, 1) long
    position_ids: torch.Tensor   # (B, 1) long
    block_table: torch.Tensor    # (B, max_pages) int32
    cache_seqlens: torch.Tensor  # (B,) int32
    # Persistent output buffer.
    logits: torch.Tensor         # (B, 1, vocab) — same shape as model output


class CudaGraphRunner:
    """Captures one CUDA graph per bucket batch size for paged decode replay.

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
        """Capture all bucket graphs. Must be called after the model is ready."""
        if self._captured:
            return
        if not torch.cuda.is_available():
            logger.warning("CUDA not available — skipping graph capture")
            return

        engine = self.engine
        pool = engine.kv_pool
        if pool is None:
            raise RuntimeError("CudaGraphRunner.capture() requires a paged KV pool")

        device = engine.device
        page_size = pool.page_size

        # Pre-extend the RoPE cache to cover the maximum decode position
        # the graph will ever encounter — the lazy-grow path in the
        # forward calls .item() and would graph-break.
        max_decode_pos = self.max_pages * page_size
        engine.model.model.rotary_emb.extend_to(max_decode_pos)
        logger.info(
            "RoPE cache pre-warmed to %d positions (max_pages=%d, page_size=%d)",
            max_decode_pos, self.max_pages, page_size,
        )

        # Warm up dynamo / kernels at EVERY bucket batch size before
        # capture, not just the first. Otherwise dynamo's first-time
        # trace for buckets 2+ runs *during* capture and ends up baked
        # into the recorded graph, leaving compile uplift on the table.
        # Spec: "compile, run a few warmup forwards so dynamo settles,
        # then capture."
        for bs in self.bucket_batch_sizes:
            for _ in range(2):
                self._dummy_forward(bs)
        torch.cuda.synchronize()

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
        device = engine.device
        page_size = pool.page_size

        page_indices = pool.allocate(batch_size)
        try:
            input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            block_table = torch.tensor(
                [[pi] for pi in page_indices], dtype=torch.int32, device=device
            )
            cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

            meta = PagedMeta(
                phase="decode",
                page_size=page_size,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
            )
            with torch.inference_mode():
                engine.model(
                    input_ids, position_ids,
                    kv_caches=pool.kv_caches, paged_meta=meta,
                )
            torch.cuda.synchronize()
        finally:
            pool.free(page_indices)

    def _capture_one(self, batch_size: int) -> _BucketGraph:
        engine = self.engine
        pool = engine.kv_pool
        device = engine.device
        max_pages = self.max_pages

        # Persistent input/output buffers — same identities every replay.
        input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        block_table = torch.zeros(
            batch_size, max_pages, dtype=torch.int32, device=device
        )
        cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

        meta = PagedMeta(
            phase="decode",
            page_size=pool.page_size,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
        )

        # First run outside the graph to allocate workspaces.
        with torch.inference_mode():
            logits, _ = engine.model(
                input_ids, position_ids,
                kv_caches=pool.kv_caches, paged_meta=meta,
            )
        torch.cuda.synchronize()

        logits_buf = torch.empty_like(logits)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            out_logits, _ = engine.model(
                input_ids, position_ids,
                kv_caches=pool.kv_caches, paged_meta=meta,
            )
            logits_buf.copy_(out_logits)

        return _BucketGraph(
            batch_size=batch_size,
            max_pages=max_pages,
            graph=graph,
            input_ids=input_ids,
            position_ids=position_ids,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            logits=logits_buf,
        )

    # ── Replay ──────────────────────────────────────────────────────────

    def try_replay(self, requests: list[Request]) -> list[int] | None:
        """Replay a captured graph if the live batch fits a bucket."""
        if not self._captured or not requests:
            return None
        live = len(requests)
        bucket = next((b for b in self.bucket_batch_sizes if b >= live), None)
        if bucket is None:
            return None
        bg = self._graphs.get(bucket)
        if bg is None:
            return None

        # Fall back to eager if any request's page table exceeds the
        # captured size.
        for req in requests:
            if len(req.page_indices) > bg.max_pages:
                return None

        engine = self.engine
        device = engine.device

        # Build the live half of each input list, then pad up to bucket.
        # Padded entries reuse row-0's tensors; their outputs are discarded.
        input_ids_list: list[int] = []
        position_ids_list: list[int] = []
        cache_seqlens_list: list[int] = []
        page_lists: list[list[int]] = []
        for req in requests:
            input_ids_list.append(req.output_ids[-1])
            position_ids_list.append(req.seq_len)
            cache_seqlens_list.append(req.seq_len)
            page_lists.append(req.page_indices)

        pad = bucket - live
        if pad > 0:
            input_ids_list.extend([input_ids_list[0]] * pad)
            position_ids_list.extend([position_ids_list[0]] * pad)
            cache_seqlens_list.extend([cache_seqlens_list[0]] * pad)
            page_lists.extend([page_lists[0]] * pad)

        # Copy into persistent buffers (no shape changes — graph stays valid).
        bg.input_ids.copy_(
            torch.tensor(input_ids_list, dtype=torch.long, device=device).unsqueeze(1)
        )
        bg.position_ids.copy_(
            torch.tensor(position_ids_list, dtype=torch.long, device=device).unsqueeze(1)
        )
        bg.cache_seqlens.copy_(
            torch.tensor(cache_seqlens_list, dtype=torch.int32, device=device)
        )
        bg.block_table.zero_()
        for i, pt in enumerate(page_lists):
            n = min(len(pt), bg.max_pages)
            bg.block_table[i, :n] = torch.tensor(
                pt[:n], dtype=torch.int32, device=device
            )

        bg.graph.replay()

        # Bookkeeping: kernel just wrote at slot=cache_seqlens, advance.
        for req in requests:
            req.seq_len += 1

        # Sample only the live entries (padded outputs are discarded).
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            tok = sample_token(
                bg.logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
            )
            token_ids.append(tok)
        return token_ids
