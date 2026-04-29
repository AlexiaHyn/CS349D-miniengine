# Milestone 2: Paged KV Cache + Compile/Graph Acceleration

## Objective

As many of you observed in Milestone 1, a major performance bottleneck
is the **dynamic allocation and manipulation of KV caches** — every
decode step grows tensors via `torch.cat`, fragmenting memory,
breaking pointer stability, and capping the batch size we can
sustain.

In this milestone you replace the dynamic cache with a **pre-allocated
paged KV memory pool**, then build the rest of the engine on top of
it. A pool gives you:

- predictable, up-front memory budgeting,
- stable tensor identities across the whole run,
- a foundation for **PagedAttention**, which reads and writes KV
  through per-request page tables, and for **CUDA graph capture**,
  which fundamentally requires stable pointers.

With paging in place, you'll layer on `torch.compile` to reduce
small-op launch overhead, and (extra credit) CUDA graphs to remove
that overhead almost entirely.

## Part A: Pre-allocated paged KV memory pool

Beyond per-step allocation overhead, the milestone-1 batched
implementation can also **OOM under high concurrency** — every
active request grows its own KV tensor, with no global cap on total
KV usage. Pre-allocating the pool fixes the KV budget up front, so
capacity is bounded by the pool size and admission either succeeds
or is rejected deterministically.

The single starter file for this milestone is
[`miniengine/kv_memory_pool.py`](../miniengine/kv_memory_pool.py),
which gives you method signatures only. Everything else
(`engine.py`, `model.py`, `scheduler.py`, etc.) you extend yourself,
building on your milestone-1 code or the provided reference code.

Your pool should:

- Pre-allocate per-layer K and V tensors that hold a fixed number of
  **pages**, each storing `page_size` tokens of one layer's KV.
- Maintain a free list so requests can acquire pages as their KV
  grows and return them on completion.
- Expose a per-layer view that Part B will index into via per-request
  **page tables**.

Make `page_size` an **adjustable CLI parameter** (e.g.,
`--page-size`, default 32). Smaller pages waste less KV at the tail
of each sequence; larger pages keep page tables small. The effect on
raw decode speed is usually minor.

Storage layout, free-list data structure, and page-table
representation are your design choices.

## Part B: PagedAttention integration

PagedAttention is the attention computation that reads and writes KV
state through **page tables** instead of a contiguous per-request KV
tensor. It applies to **both prefill and decode** — neither phase
allocates fresh KV memory; both index into the pool you built.

- **Prefill** writes each prompt token's freshly computed K/V into
  the pages assigned to that request.
- **Decode** appends one new token's K/V to the request's current
  (partially full) page each step; attention reads the full
  per-request KV by gathering through the page table.

`flash_attn_varlen_func` and `flash_attn_with_kvcache` from
[`flash-attn`](https://github.com/Dao-AILab/flash-attention) already
understand page tables and slot mappings — you may use them, or roll
your own. Either way, expect attention to be the trickiest piece:
correctness depends on the pages being addressed exactly the way the
kernel expects.

**Batched, packed prefill is required.** A naive batched prefill
pads all prompts to the same length and wastes matmul on padding.
With paging in place, flatten N prompts into one packed sequence and
run a single forward pass with `flash_attn_varlen_func` — no
padding, no waste.

**Target.** Paged attention + packed batched prefill should deliver
**at least 2× throughput** over your milestone-1 batched baseline,
and often more at higher concurrency. Sweep input/output lengths and
concurrency levels and report a setting that highlights the gain.


## Part C: Reducing launch overhead

On modern GPUs, **excessive invocation of small operations is itself
a performance bottleneck** — each tiny kernel launch makes the GPU
sit idle while the CPU dispatches the next one, and as GPUs get
faster this overhead becomes a larger fraction of total time. Decode
is especially affected, since each step does very little compute per
kernel. Two techniques attack this:

- **Kernel fusion** (`torch.compile`) merges chains of small ops
  into one generated kernel — fewer launches, more arithmetic per
  launch.
- **Execution-context caching** (CUDA graphs) records a sequence of
  GPU operations once, then replays it in a single launch —
  eliminating per-op Python and driver overhead almost entirely.

### Required: `torch.compile`

Find places where `torch.compile` fits and demonstrate a measurable speedup. 
Note that wrapping the **whole model** often does *not* yield a gain — dynamic 
shapes (variable batch sizes, growing KV) and Python-level branching trigger
recompiles or fall back to eager. Pick a sub-region with stable
shapes and minimal branching (a single transformer block, the MLP,
a fused norm/RoPE path, etc.) and report the resulting throughput delta.

**Target.** **10% improvement** on both throughput and
latency over the paged baseline.

### Extra credit: manual CUDA graphs

CUDA graphs typically yield a larger speedup than `torch.compile`
alone, but they impose strict invariants on the captured region:

- **Stable tensor identities** — reuse the same input objects every
  step; copy fresh data into them.
- **No CPU↔GPU sync** inside the captured region — no `.item()`,
  no Python branching on tensor values, no lazy tensor growth.
- **Fixed shapes per graph** — typically capture one graph per
  bucket batch size and round live batches up to the nearest bucket.

Build a `CudaGraphRunner` that captures the paged-decode forward at
a set of bucket sizes and replays at runtime. Keep sampling
**outside** the graph (per-request top-k/top-p with `multinomial`
and `.item()` would break capture). Stack with `torch.compile`:
compile, run a few warmup forwards so dynamo settles, then capture.

**Target.** **20% improvement** on both throughput and
latency over `paged + torch.compile`. Report the measured speedup
and the tradeoffs you encountered.

## Running

```bash
# Milestone 1 baseline (for comparison)
python -m miniengine --model Qwen/Qwen3-8B --mode batched

# Milestone 2: paged + torch.compile
python -m miniengine --model Qwen/Qwen3-8B \
    --mode paged --mem-fraction-static 0.85 \
    --page-size 32 --torch-compile

# Plus extra-credit CUDA graphs
python -m miniengine --model Qwen/Qwen3-8B \
    --mode paged --mem-fraction-static 0.85 \
    --page-size 32 --torch-compile \
    --cuda-graph --cuda-graph-batch-sizes 1,2,4,8,16,32
```

## Deliverables

### Required server CLI flags

Your engine must accept the following flags:

| Flag | Description |
|------|-------------|
| `--mode paged` | Selects the paged engine path (in addition to the milestone-1 modes). |
| `--mem-fraction-static` | Fraction of total GPU memory pre-allocated for static tensors (model weights + KV cache pool). The pool capacity is derived from this. |
| `--page-size` | Tokens per page in the KV pool. |
| `--torch-compile` | Enables `torch.compile` on your chosen sub-region. |

Optional (extra credit):

| Flag | Description |
|------|-------------|
| `--cuda-graph` | Enables manual CUDA graph capture for paged decode. |
| `--cuda-graph-batch-sizes` | Comma-separated bucket sizes to capture (e.g., `1,2,4,8,16,32`). |

### Report

Submit a **PDF report to Gradescope**. Run all benchmarks on an
**L4 GPU** with **`Qwen/Qwen3-8B`** so results are comparable across
submissions. The report must include screenshots of the terminal
output for:

1. **Accuracy.** `bench_accuracy` on MMLU and/or GSM8K showing your
   paged engine (with and without `--torch-compile`) matches the
   milestone-1 baseline within noise.

2. **Throughput.** `bench_serving` for at least:
   - milestone-1 `batched`,
   - milestone-2 `paged` (≥ 2× throughput target),
   - `paged + torch.compile` (≥ 10% over `paged`),
   - `paged + torch.compile + cuda-graph` if you did the extra
     credit (≥ 20% over `paged + torch.compile`).

   Each screenshot should show TTFT p50/p99, TPOT p50/p99, and
   generation throughput.

3. **Page-size comparison.** `bench_serving` at two `--page-size`
   values (e.g., 16 vs 128) with a short comment.

The report should also walk through:

- **Design and implementation.** Your choices for the KV pool
  (storage layout, free-list, page-table representation), how
  prefill and decode plug into the pool, where you applied
  `torch.compile`, and (if attempted) how you structured CUDA graph
  capture.
- **Source of performance benefit.** For each optimization, explain
  *why* it helps: which bottleneck it removes (allocation overhead,
  padding waste, kernel launches, etc.) and how that maps to the
  numbers you measured.