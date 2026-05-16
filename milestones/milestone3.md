# Milestone 3: Chunked Prefill + Radix Prefix Cache

## Objective

Two orthogonal optimizations on top of the milestone-2 paged engine:

1. **Chunked prefill** bounds the per-step activation-memory footprint of the prefill forward, so long prompts and high concurrency don't OOM the activation buffers (attention scores, MLP intermediates, etc.).
2. **Radix prefix cache** eliminates redundant prefill work for requests that share token prefixes — RAG with retrieved context, multi-turn chat that re-sends history, few-shot evaluation with the same exemplar block, etc.

Both layer on the milestone-2 paged KV pool. Implement Part A first, then Part B on top of it.

## Part A: Chunked Prefill

Milestone 2's prefill processes every admitted request's entire prompt in one packed varlen forward — the per-step q-token total can run into the tens of thousands and the resulting activation tensors can outsize the GPU's free memory, causing a CUDA OOM. Chunking caps per-step q-tokens at a configurable budget so that activation memory scales with the chunk size, not with the prompt length.

Beyond OOM avoidance, chunking also unlocks a memory-pressure backstop you'll need for Part B: when concurrent long prompts can't all fit at full length, the scheduler can admit them one chunk at a time and rely on cache eviction / completion to recover pages incrementally.

Required CLI:

| Flag | Description |
|------|-------------|
| `--prefill-chunk-size N` | Per-step prefill token budget. `0` disables chunking (milestone-2 single-shot path). |

### Target

- **OOM avoidance.** Find a `bench_serving` configuration (large `--input-len`, high `--concurrencies`, or both) that OOMs with `--prefill-chunk-size 0` and succeeds with a sensible chunk size on your GPU. Show the failure mode of the unchunked run (CUDA OOM in server log, HTTP 5xx / stream error on the client) and the success of the chunked run.
- **No regression.** Pick a chunk size and an input length such that chunking *actually fires* — e.g. `--prefill-chunk-size 512` with `--input-len 4096` (≈ 8 chunks per request) — and confirm the chunked path matches the unchunked path within noise on throughput, TTFT p50/p99, TPOT (run `bench_serving` at conc 1 / 4 / 16) and on accuracy (`bench_accuracy` MMLU).  Pick an input length large enough that chunking runs but small enough that the unchunked baseline still fits in memory — there's no point comparing against a baseline that OOMs.  **In your report, explicitly state your reasoning for the chunk size and input length you chose**: too small a chunk size pays kernel-launch overhead per chunk; too large defeats the OOM-avoidance purpose; the relationship between `chunk_size`, `input_len`, `concurrency`, and the GPU's free activation memory is exactly what you're trying to characterize, so make those trade-offs explicit.

## Part B: Radix Prefix Cache

Add a token-prefix → KV-pages cache so requests with overlapping prompts share already-computed KV. A skeleton with the data-structure invariants and method stubs is in [`miniengine/radix_cache.py`](../miniengine/radix_cache.py); fill it in and wire it into the engine, scheduler, and pool.

The cache must:
- match at **page granularity** — partial-page sharing isn't safe with our attention metadata,
- match against **prompt + generation** so multi-turn requests hit on the prior turn's response,
- **coexist with the pool's free list** — pages held by the cache aren't free; the pool's `allocate` asks the cache to evict LRU pages before raising OOM,
- **pin in-flight matches** so an active request's borrowed pages can't be evicted underneath it.

Provided scaffolding (already in the repo):
- `/cache_stats` HTTP endpoint (server-wide hit-rate / eviction counters).
- `usage.cache_hit_tokens` in every `/v1/chat/completions` response.
- `benchmark/bench_cache.py` with two shared-prefix workloads (`shared`, `multiturn`).
- `Request.cache_hit_tokens` field for per-request accounting.

You will need to add yourself: the CLI flag (`--disable-radix-cache`), all of `radix_cache.py`'s method bodies, and the engine/scheduler/pool plumbing.

Required CLI:

| Flag | Description |
|------|-------------|
| `--disable-radix-cache` | Disable the cache (cache is on by default). |

### Target

Use `bench_cache.py` for the two workloads below, comparing cache-on vs `--disable-radix-cache`.

- **`shared` workload.** K groups × N questions, tunable `--shared-prefix-len` and `--question-len`.  **Target: at least 2× throughput improvement and at least 2× TTFT reduction** at a prefix length where the cache should clearly help (e.g. `--shared-prefix-len 2000`).  Sweep prefix length to show the speedup scales with it.
- **`multiturn` workload.** N sessions × M turns.  Per-turn hit rate should climb across turns; the per-turn breakdown bench_cache emits should show 0% on turn 0 and steadily rising rates after.  **Target: at least 50% throughput improvement and at least 50% TTFT reduction** aggregated across turns.  The cached prefix only short-circuits prefill, so on multiturn the throughput win is bounded by how much of each request is prefill versus newly-generated tokens — pick `--turns-per-session` (and the per-turn generation length) so the cumulative cached prefix dominates.
- **No regression.** On `bench_serving` default (WildChat-style, low prefix sharing), the cache should be within noise of `--disable-radix-cache` on throughput / TTFT / TPOT at conc 1 / 4 / 16.

## Bonus: Retraction (advanced)

Even with chunked prefill and a radix cache, the engine can still OOM during *decode*. Decode-time page growth is incremental and unpredictable — you don't know upfront how many tokens a request will generate, and admission can't reserve the worst case without crushing concurrency. If running requests keep growing and none complete fast enough, the pool drains; the next decode step fails to allocate; the request dies.

**Retraction** is the standard mitigation. When the pool can't satisfy the next decode allocation, the scheduler picks a victim from the active set, frees all of its KV pages back to the pool, and pushes it back onto the waiting queue. The remaining requests proceed; the retracted request will be re-admitted (and re-prefilled) when capacity returns. Picking the youngest / largest-remaining-work victim is a reasonable starting heuristic.

To claim the bonus:
1. Construct a workload that triggers the decode-time OOM on your engine without retraction.
2. Implement retraction and show the same workload completes successfully.
3. Describe your victim-selection policy and any edge cases (in-flight chunked-prefill request, pinned cache pages, etc.).

Include screenshots of both the failing baseline and the successful retraction run.

## Running

```bash
# Milestone-2 baseline (for comparison)
python -m miniengine --model Qwen/Qwen3-8B --mode paged --mem-fraction-static 0.85 --page-size 32

# Milestone-3 (Part A + Part B; Part B is on by default)
python -m miniengine --model Qwen/Qwen3-8B --mode paged --mem-fraction-static 0.85 --page-size 32 --prefill-chunk-size 512
```

## Deliverables

### Required server CLI flags

| Flag | Description |
|------|-------------|
| `--prefill-chunk-size N` | Per-step prefill token budget (`0` disables chunking). |
| `--disable-radix-cache` | Disable the prefix cache (cache is on by default). |

### Report

Submit a **PDF report to Gradescope**. Run all benchmarks on an **L4 GPU** with **`Qwen/Qwen3-8B`** so results are comparable across submissions. Include terminal screenshots for the items below and tabulate the numbers.

**Part A — Chunked Prefill**
1. A `bench_serving` invocation that OOMs at `--prefill-chunk-size 0` and succeeds with a chunked setting. Show both server-side errors and client outcome.
2. `bench_accuracy` MMLU at `--prefill-chunk-size 0` vs the chunked setting — accuracy within noise.
3. `bench_serving` default workload at conc 1 / 4 / 16, `--prefill-chunk-size 0` vs chunked — TTFT p50/p99, TPOT p50/p99, throughput within noise.

**Part B — Radix Prefix Cache**
4. `bench_cache.py --workload shared` at one fixed config (e.g. 10 groups × 10 questions, `--shared-prefix-len 2000`, `--concurrency 4`), cache-on vs `--disable-radix-cache`.  Report wall time, throughput, TTFT p50/p99, and hit rate.  **Hit at least 2× throughput improvement and at least 2× TTFT reduction.**
5. `bench_cache.py --workload shared` sweep over `--shared-prefix-len` ∈ {200, 500, 2000, 4000}, cache-on.  Show speedup scales with prefix length.
6. `bench_cache.py --workload multiturn` cache-on vs off, including the per-turn breakdown.  **At a long enough conversation, hit at least 50% throughput improvement and at least 50% TTFT reduction aggregated across turns.**
7. `bench_serving` default workload at conc 1 / 4 / 16, `--disable-radix-cache` vs default — within noise.

**Bonus — Retraction** (optional)
8. The OOM-triggering workload at default settings (no retraction).
9. The same workload after retraction is implemented — completion + design notes.
