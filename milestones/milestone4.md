# Milestone 4: Advanced Features (Choose One)

## Objective

Milestones 1–3 built the core serving engine: continuous batching, a paged KV pool with PagedAttention, `torch.compile` + CUDA graphs, chunked prefill, and a radix prefix cache. Everything from here on is **advanced** — features that real production inference stacks ship and that each open a different research/engineering direction.

For milestone 4 you pick **one** of the three tracks below and take it end-to-end: design, implementation, and a quantitative demonstration on `bench_serving` / `bench_cache` (or a workload you justify yourself, for Track 3). Each track has a clearly defined "full credit" bar and one or more bonus opportunities.

| Track | Theme | Reference |
|-------|-------|-----------|
| **1** | HiCache: extend the radix cache with a CPU-tier KV store | <https://www.lmsys.org/blog/2025-09-10-sglang-hicache/> |
| **2** | Speculative decoding with a draft model | <https://arxiv.org/abs/2211.17192> |
| **3** | Your own proposal (scheduling, kernels, batching, …) | — |

All tracks should layer on top of your milestone-3 engine. Don't disable chunked prefill or the radix cache to make a track easier — the comparison baseline is **milestone-3 default**.

---

## Track 1 — HiCache: Hierarchical KV Cache (GPU + CPU)

The milestone-3 radix cache lives entirely in HBM. Once the GPU KV pool is full, LRU eviction frees pages back to the free list and the cached prefix is **gone** — the next request that would have hit on it has to re-prefill from scratch. On long-horizon workloads (deep multi-turn, many concurrent sessions, large RAG corpora) HBM-only caching falls off a cliff: the working set exceeds capacity and the hit rate collapses.

**HiCache** (SGLang's hierarchical cache) adds a CPU-memory tier underneath HBM. Evicted GPU pages are demoted to a CPU-side pool instead of being freed; a hit against a CPU-resident prefix triggers an H2D copy that re-promotes the pages before the request runs. CPU DRAM is 10–100× larger than HBM on the same machine, so the effective cache capacity grows by the same factor.

For this milestone you implement the **GPU + CPU** tiers only.

### What to build

Extend `miniengine/radix_cache.py` (or add a sibling `hicache.py`) into a `HiRadixTree` that:

1. **Tracks tier per node.** Each radix node's pages live in either the **GPU pool** or a new **CPU pool** (a pre-allocated pinned-memory tensor of shape `(num_cpu_pages, page_size, num_kv_heads, head_dim)` per layer; allocate at startup, sized by a new `--cpu-cache-size-gb` flag).
2. **Demotes on GPU eviction.** When the GPU pool needs free pages, instead of dropping the LRU radix entry on the floor, copy its pages **D2H** into the CPU pool and update the node to point at the new CPU tier. Free the GPU pages back to the pool.
3. **Promotes on hit.** On a lookup that matches a CPU-resident prefix, allocate fresh GPU pages, copy **H2D** from the CPU pool, repoint the node at GPU, and free the CPU slots. The request then proceeds as if it had been a normal GPU-cache hit.
4. **Evicts the CPU tier on overflow.** The CPU pool also has a capacity. When it fills, evict LRU CPU entries (drop them entirely — there is no lower tier).
5. **Stays correct under concurrency.** A request that triggers a promotion holds a lock on the involved pages for the duration of its prefill (same `inc_lock_ref` / `dec_lock_ref` pattern as milestone 3).

Use **non-blocking copies on a dedicated CUDA stream** so the demote/promote traffic overlaps with model execution when possible. Pinned host memory is required for async H2D.

### Required CLI

| Flag | Description |
|------|-------------|
| `--cpu-cache-size-gb N` | Size of the CPU KV tier in GiB. `0` disables HiCache (radix cache stays GPU-only). |
| `--hicache-overlap` *(optional)* | Use a dedicated CUDA stream for async demote/promote. Default off — implement first as a blocking copy, then turn this on for the perf-bonus runs. |

### Target

You will use `bench_cache.py --workload multiturn` as the harness — it's the workload that most cleanly stresses cache distance.

1. **Demonstrate the GPU-only cliff.** With HiCache **off** (i.e. milestone-3 baseline), find a `--num-sessions` / `--turns-per-session` configuration where the per-turn hit-rate breakdown starts strong on early turns and then **drops sharply** as the working set exceeds GPU pool capacity. The collapse should be obvious in the per-turn table (e.g. ≥70% on turn 1, <20% by the last turn). Document the GPU pool size (`mem-fraction-static`) so the reader can see the capacity wall.
2. **Restore the hit rate with HiCache.** Re-run the same workload with HiCache enabled and a CPU pool sized at ≥10× the GPU pool. The per-turn hit rate must **stay high across all turns** (no collapse). Report the new per-turn table side by side with (1).
3. **End-to-end completion.** The workload must still finish correctly (token-level sanity, no hangs, no OOM in either tier).

Full credit requires (1)+(2)+(3) on `Qwen/Qwen3-8B`/L4. End-to-end throughput and TTFT numbers should be reported but are **not** required to improve over the GPU-only baseline — async H2D and CPU PCIe bandwidth bound how much wall-time the rehydration saves.

### Bonus points

- Implement `--hicache-overlap` (dedicated stream, pinned memory, async copies) and show the overlap is real (e.g. NSight timeline or measured promote-time-hidden ratio).
- Show a **real end-to-end win** — > 20% throughput *or* > 20% TTFT improvement on at least one configuration of the multiturn workload, with HiCache vs milestone-3 default.

---

## Track 2 — Speculative Decoding

Standard autoregressive decode does **one target-model forward per generated token** — under low concurrency that's the entire critical path of TPOT, and the GPU is severely underutilized because the forward is a single-token matmul. Speculative decoding ([Leviathan et al., 2022](https://arxiv.org/abs/2211.17192)) addresses both: a small **draft model** generates a candidate sequence of `K` tokens (cheap, K serial forwards on a small model), then the target model verifies all `K+1` positions in **one batched forward**. Tokens that match the target's argmax (or pass the rejection-sampling test) are accepted; the first mismatch is replaced with the target's own token and the rest are discarded. The expected number of tokens advanced per target-forward is the **accept length**.

When accept length > 1, you spend fewer target-model forwards per generated token, and TPOT drops accordingly — most visibly under **low concurrency**, where the target forward is launch-bound rather than compute-bound, so the extra batched verification work is nearly free.

### What to build

Wire a draft model into the decode loop:

1. **Load both models.** Target = `Qwen/Qwen3-8B` (the milestone model). Draft = [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) — same tokenizer / vocab as the target.
2. **Draft K tokens.** From the current state of an active request, run the draft model autoregressively for `K` steps to produce a candidate continuation. Keep the draft's own KV cache so successive draft phases are cheap.
3. **Verify in one target forward.** Feed the `K` candidate tokens (plus the last accepted token) to the target model in a single forward pass; for each position compare against the target's sampled / argmax token. Accept the longest matching prefix, replace the first mismatch with the target's token, and discard the rest. **Roll back** the draft's KV to match.
4. **Stay correct.** With greedy sampling, the output stream must be **token-identical** to the non-speculative path on the same prompts. (With temperature > 0, use the standard rejection-sampling formulation — accept rate is the guarantee, not bitwise identity.)
5. **Integrate cleanly with the rest of the engine.** Paged KV for the target stays as-is; the draft can use a simpler contiguous KV (it's tiny) or its own small page pool. If speculative decoding interacts badly with chunked prefill or the radix cache, document the interaction — the speculative path only needs to run during **decode**.

### Required CLI

| Flag | Description |
|------|-------------|
| `--speculative-draft-model NAME` | HF id of the draft model (e.g. `Qwen/Qwen3-0.6B`). Empty disables speculative decoding. |
| `--speculative-num-draft-tokens K` | Tokens drafted per target verification (typical 3–7). |

### Target

Use `bench_serving` (default WildChat-style workload) and `bench_accuracy` on MMLU. For full credit:

1. **Functional pipeline.** Speculative decoding produces correct outputs — MMLU accuracy on `Qwen/Qwen3-8B` matches the non-speculative baseline within noise (±1 pp).
2. **Accept length > 2.** Report the **mean accept length** across a `bench_serving` run at conc 1. ≥ 2.0 is the bar. Show how accept length varies with `K`.
3. **Fewer target forwards.** Instrument and report the number of target-model forward passes per generated token. With non-speculative decoding this is 1.0; with speculation it should be `1 / mean_accept_length` (well below 1.0).

### Bonus

Pick **one** for extra credit:

- **(a) Tree verification.** Instead of drafting a single linear sequence of `K` tokens, draft a **tree** of candidate branches (top-k at each step, depth `d`), verify all branches in one batched target forward with a properly masked attention pattern, and accept the longest valid root-to-leaf path. Demonstrate a **higher mean accept length** than the linear-draft baseline at comparable target-forward cost.
- **(b) EAGLE-3.** Replace the standalone small draft model with an [EAGLE-3](https://arxiv.org/abs/2503.01840)-style draft head that consumes the target model's last-layer hidden states. A public EAGLE-3 head matched to the milestone target is available at [`AngelSlim/Qwen3-8B_eagle3`](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3) — load this rather than training one yourself. Compare accept length and TPOT against the small-draft-model baseline.
- **(c) Real end-to-end win.** Achieve a **≥20% TPOT reduction** at conc 1 on a `bench_serving` configuration you choose. Show numbers vs the milestone-3 baseline and explain why your chosen `K` is sweet-spot for that configuration.

---

## Track 3 — Your Own Proposal

The third option is open-ended. As we discussed in class, a new scheduling algorithm targeted at a specific scenario is a natural fit (e.g. SLO-aware priority scheduling, prefill / decode separation, fairness across tenants, long-output prioritization, prefix-cache-aware admission, …). Kernel work or a different batching strategy is equally valid — pick something that actually addresses a measurable bottleneck in this engine. We're also open to proposals that don't fall into any of these buckets — quantization, structured-output / constrained decoding, a different attention variant, request-level retries with cached state, multi-LoRA serving, anything else you can make a quantitative case for. If in doubt, pitch it.

If you take this track you **must** deliver all three of:

1. **A strong quantitative argument that the scenario matters.** Don't optimize a strawman. Show — with a workload, a trace, a paper, or production data — that the scenario you're targeting is common / expensive / important enough to be worth a system-level change. "Sometimes users send long prompts" is not enough; "in this trace from X, 30% of requests have prompts > 8k tokens and they account for 70% of TTFT p99" is the bar.
2. **A clear performance benefit.** ≥ **20%** improvement on either throughput **or** latency (p50 or p99 — your choice, but state which) on the scenario you're targeting. Compare against the milestone-3 default.
3. **A regression analysis on other scenarios.** Run your modified engine on the *non-target* scenarios that your optimization might hurt — and on `bench_serving` default — and show that the regression is **minor and acceptable**. "My new scheduling policy speeds up workload X by 30% and only slows down workload Y by 4%, which is within run-to-run noise" is the kind of statement we're looking for.

Required CLI for Track 3 is whatever your design needs — pick flag names consistent with the existing ones (`--<thing>`, `--disable-<thing>`, `--<thing>-size N`, etc.) and document each.

---

## Running

```bash
# Track 1 — HiCache (multiturn cliff demo)
python -m miniengine --model Qwen/Qwen3-8B --mode paged \
    --mem-fraction-static 0.85 --page-size 32 \
    --prefill-chunk-size 512 --cpu-cache-size-gb 32

# Track 2 — Speculative decoding
python -m miniengine --model Qwen/Qwen3-8B --mode paged \
    --mem-fraction-static 0.85 --page-size 32 \
    --prefill-chunk-size 512 \
    --speculative-draft-model Qwen/Qwen3-0.6B \
    --speculative-num-draft-tokens 5
```

## Deliverables

### Report

Submit a **PDF report to Gradescope**. Run all benchmarks on an **L4 GPU** with **`Qwen/Qwen3-8B`** as the target/serving model so results are comparable across submissions. Use `Qwen/Qwen3-0.6B` for the Track 2 draft. Include terminal screenshots and tabulate the numbers.

The report should cover, regardless of track:

- **Design.** What did you build? Where in the engine does it live? What invariants does it rely on (e.g. for HiCache, pinned memory + stream; for spec-dec, vocab parity between draft and target; for Track 3, the scheduler/kernel/etc. surface you modified)?
- **Correctness.** How do you know it didn't break the engine? MMLU within noise is the cheapest evidence; token-identity smoke tests are stronger when applicable (e.g. greedy speculative decoding).
- **Quantitative evaluation.** The track-specific targets above — per-turn hit-rate tables (Track 1), accept length + target-forward count (Track 2), or the scenario-importance / improvement / regression triple (Track 3).
- **What didn't work, and why.** This is the most valuable part of the report at this point in the course. If you tried tree verification and the accept-length gain didn't justify the verification cost — say so. If your scheduling policy helped one workload and tanked another — show it.

### Per-track quick-reference

| Track | Full-credit bar | Bonus |
|-------|-----------------|-------|
| 1. HiCache | Show the GPU-only hit-rate cliff on multiturn; show HiCache restores per-turn hit rate. | async overlap; real throughput/TTFT win. |
| 2. Speculative decoding | Functional pipeline, accept length > 2, target-forward count down. | One of: tree verification (higher accept length), EAGLE-3 head, or ≥20% TPOT reduction. |
| 3. Your proposal | Scenario justification + ≥20% improvement on target + regression analysis on non-target. | — (the bar is already higher; depth and rigor of the analysis count.) |
