"""
CLI entry point — launch the MiniEngine server.

Usage:
    # Milestone 1 baseline
    python -m miniengine --model Qwen/Qwen3-8B --mode batched

    # Milestone 2: paged + torch.compile
    python -m miniengine --model Qwen/Qwen3-8B \\
        --mode paged --mem-fraction-static 0.85 \\
        --page-size 32 --torch-compile

    # Plus extra-credit CUDA graphs
    python -m miniengine --model Qwen/Qwen3-8B \\
        --mode paged --mem-fraction-static 0.85 \\
        --page-size 32 --torch-compile \\
        --cuda-graph --cuda-graph-batch-sizes 1,2,4,8,16,32
"""

from __future__ import annotations

import argparse
import logging

import torch
import uvicorn

from miniengine.engine import Engine, ATTENTION_BACKENDS
from miniengine.scheduler import Scheduler
from miniengine import server as srv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="miniengine",
        description="Minimal LLM serving engine",
    )
    p.add_argument(
        "--model", type=str, required=True, help="HuggingFace model id or local path"
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--device", type=str, default="cuda", help="Device to load model on")
    p.add_argument(
        "--max-running",
        type=int,
        default=16,
        help="Max concurrent requests in the scheduler",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="paged",
        choices=["baseline", "batched", "paged"],
        help="Scheduling mode: baseline (one request at a time), "
        "batched (iteration-level batching, milestone 1), "
        "or paged (paged KV + flash_attn varlen, milestone 2)",
    )

    # ── Milestone-2 paged-mode flags ───────────────────────────────────
    p.add_argument(
        "--page-size",
        type=int,
        default=256,
        help="paged: tokens per KV page.  flashinfer (default backend) "
        "works with any page size, including small ones (16, 32).  "
        "flash_attn 2.x typically requires page_size %% 256 == 0.",
    )
    p.add_argument(
        "--attention-backend",
        type=str,
        default="flashinfer",
        choices=list(ATTENTION_BACKENDS),
        help="paged: attention kernel backend.  flashinfer (default) "
        "supports any page size and has a tensor-core decode kernel; "
        "flash_attn varlen is battle-tested but requires "
        "page_size %% 256 == 0 in standard builds.",
    )
    p.add_argument(
        "--flashinfer-workspace-mb",
        type=int,
        default=128,
        help="paged: scratch buffer size for flashinfer wrappers (MB)",
    )
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help="paged: fraction of total GPU memory pre-allocated for static "
        "tensors (model weights + KV pool).  Pool capacity is derived from "
        "this; e.g. 0.85 leaves ~15%% for activations.",
    )
    p.add_argument(
        "--kv-pool-gb",
        type=float,
        default=None,
        help="paged: explicit KV-pool size in GB (overrides --mem-fraction-static)",
    )
    p.add_argument(
        "--activation-reserve-gb",
        type=float,
        default=4.0,
        help="paged: fallback VRAM held back from the pool for forward "
        "activations when neither --mem-fraction-static nor --kv-pool-gb "
        "is set",
    )
    p.add_argument(
        "--max-position",
        type=int,
        default=16384,
        help="paged: max RoPE position; cos/sin tables are sized at engine init",
    )

    # ── Milestone-2 accelerator flags (additive) ───────────────────────
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile on per-layer MLP and RMSNorm sub-modules",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture decode CUDA graphs at fixed batch sizes (requires --mode paged)",
    )
    p.add_argument(
        "--cuda-graph-batch-sizes",
        default="1,2,4,8,16,32",
        help="paged: comma-separated bucket batch sizes to capture",
    )
    p.add_argument(
        "--cuda-graph-max-pages",
        type=int,
        default=32,
        help="paged: max KV pages per sequence the captured graph supports",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("miniengine")

    dtype = getattr(torch, args.dtype)
    logger.info(
        "Initializing engine  model=%s  dtype=%s  mode=%s",
        args.model,
        args.dtype,
        args.mode,
    )

    bs_list = (
        [int(x) for x in args.cuda_graph_batch_sizes.split(",") if x.strip()]
        if args.cuda_graph
        else None
    )

    engine = Engine(
        model_path=args.model,
        dtype=dtype,
        device=args.device,
        mode=args.mode,
        torch_compile=args.torch_compile,
        cuda_graph=args.cuda_graph,
        page_size=args.page_size,
        mem_fraction_static=args.mem_fraction_static,
        kv_pool_gb=args.kv_pool_gb,
        activation_reserve_gb=args.activation_reserve_gb,
        max_position=args.max_position,
        cuda_graph_batch_sizes=bs_list,
        cuda_graph_max_pages=args.cuda_graph_max_pages,
        attention_backend=args.attention_backend,
        flashinfer_workspace_mb=args.flashinfer_workspace_mb,
    )
    sched = Scheduler(engine=engine, max_running=args.max_running, mode=args.mode)

    # Wire up the server module globals
    srv.engine = engine
    srv.scheduler = sched
    srv.model_id = args.model

    # Start scheduler background thread
    sched.start()

    logger.info("Starting server on %s:%d", args.host, args.port)
    try:
        uvicorn.run(srv.app, host=args.host, port=args.port, log_level="info")
    finally:
        sched.stop()


if __name__ == "__main__":
    main()
