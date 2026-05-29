# Milestone 4 Track 2 — Bonus (c) Continuation Guide

This README hands off the in-flight bonus attempt to a teammate.
**Full-credit is done.** What's left is **Bonus (c): ≥20% TPOT reduction at
conc 1** — the part we got blocked on.

---

## Current status — full-credit done, bonus (c) blocked

### What's already passed ✅ (Track 2 full-credit, results in this dir)

| Requirement | Result | Source |
|---|---|---|
| Functional pipeline | ok=8/8 across all K | `spec_k{3,5,7}.log` |
| MMLU accuracy ±1pp | **61.5% = 61.5% baseline (0pp diff)** | `spec_mmlu.log` |
| Mean accept length ≥ 2.0 | K=3: **2.49**, K=5: **2.79**, K=7: **2.94** | `spec_k{3,5,7}.log` + `/spec_stats` |
| Target-fwd-per-tok < 1.0 | K=3: **0.40**, K=5: **0.36**, K=7: **0.34** | same |

### What's NOT passed — Bonus (c) target ❌

> "Achieve **≥20% TPOT reduction** at conc 1 on a `bench_serving` configuration."

Same workload (input 1024 / output 256 / conc 1 / 8 reqs):

| Config | TPOT p50 (ms) | vs baseline | source |
|---|---|---|---|
| **non-spec baseline** | **69.2** | — | `nospec_baseline.log` |
| spec K=3 | 76.9 | **−11.1% (slower)** | `spec_k3.log` |
| spec K=5 | 94.0 | −35.8% | `spec_k5.log` |
| spec K=7 | 114.1 | −64.9% | `spec_k7.log` |

**Bar to clear: TPOT < 69.2 × 0.8 = `55.4 ms`.**
None of our K values currently hit it — spec decode is *slower* than non-spec
on L4 + Qwen3-8B.

---

## Why we lost — root cause (timing decomposition)

Solving for D = draft-step time, V = target-verify time from the three
K runs (`TPOT × accept_length = D·K + V`):

```
K=3:  76.9 × 2.49 ≈ 191 ms / round
K=5:  94.0 × 2.79 ≈ 262 ms / round
K=7: 114.1 × 2.94 ≈ 335 ms / round
```

Solving any two equations gives **D ≈ 36 ms / draft step, V ≈ 83 ms / verify**.

Two specific bottlenecks:

1. **Draft is 4–7× slower than it should be (36 ms vs expected 5–10 ms).**
   Root cause located in `model.py`: the dense KV path does
   ```python
   k = torch.cat([kv_cache[0], k], dim=2)
   v = torch.cat([kv_cache[1], v], dim=2)
   ```
   per layer per draft step. On a 1024-token prompt that's ~2 MB alloc + copy
   per layer × 28 layers ≈ **~60 ms of just memory traffic** — accounts for
   most of the 36 ms (with some hidden by overlap).
2. **Target verify is slower than single-token decode (83 vs 69 ms).**
   `target_verify` goes through the **chunked-prefill** kwargs path
   (`_fi_kwargs_chunked_prefill`), which has no CUDA graph.
   Single-token decode (the non-spec path) does have a `graph_runner`. So
   verify pays full Python + kernel-launch overhead while decode doesn't.

### What that implies (the math for the fix)

If draft drops from 36 ms → 10 ms (graph or in-place fix removes cat):
```
K=3: 3×10 + 83 = 113 ms / round; ÷2.49 accept → 45.4 ms/token
     vs baseline 69.2 ms → -34% TPOT  ✅ clears bonus by a wide margin
```
If verify also drops 83 → 35 ms (via reusing decode graph path):
```
K=3: 30 + 35 = 65 ms / round; ÷2.49 → 26 ms/token → -62% TPOT
```

So **fixing draft alone is enough** for Bonus (c).

---

## What we tried — and where it broke

**commit `839838c` ("Speed up draft path: in-place KV buffer …")**

Added an in-place KV buffer path to `Attention.forward`: when the caller
passes `kv_buf=(K_buf, V_buf), kv_buf_pos, kv_buf_len`, the layer scatters
new K/V into the pre-allocated buffer at `[pos, pos+seq_len)` and attention
reads `K_buf[:, :, :len+seq_len, :]` as a view — no `torch.cat`, no copy.

Wired through `TransformerModel` (per-layer `kv_bufs` list) and `CausalLM`
(transparent via `**paged_kwargs`). `Engine.draft_prefill` allocates the
buffer; `draft_decode` does index-write + len++.

**The bug:** `draft_prefill` runs the prompt (seq_len = e.g. 1024) through
the in-place path with `kv_buf_pos=0, kv_buf_len=0`. That triggers **1
AssertionError** on the very first warm request. We did NOT capture the
traceback — GCP billing was disabled mid-debug and SSH dropped.

**Suspected cause:** with `kv_buf` set and `seq_len > 1`, the `is_causal`
guard in `model.py:Attention.forward` needs `is_causal=True`, and the
GQA expand step happens *after* the buffer slice — the shapes / strides
of `K_buf[:, :, :seq_len, :]` (a view) interacting with the `expand` and
`reshape` is probably what blew up.  Could also be a Qwen3 q_norm / k_norm
shape issue with the view.

---

## Recommended fix (safe, minimum surface)

**Keep prefill on the legacy `torch.cat` path; only use in-place for decode.**
Prefill runs once per request — even if it's slower per step, it's amortized.
Decode runs hundreds of times — that's where the win is.

Concretely in `engine.py`, replace `draft_prefill` with:

```python
@torch.inference_mode()
def draft_prefill(self, request: Request) -> torch.Tensor:
    """Prefill the draft model via the legacy cat path, then copy KV into
    a pre-allocated in-place buffer for subsequent draft_decode steps.

    This avoids the prefill-path bug in commit 839838c while still giving
    decode the no-cat speedup it needs for Bonus (c).
    """
    assert self.draft_model is not None
    prompt_len = len(request.input_ids)
    ids = _to_long(request.input_ids, self.device).unsqueeze(0)
    pos = torch.arange(prompt_len, device=self.device).unsqueeze(0)

    # Legacy dense prefill — produces list[(k, v)] with cat, but only runs once.
    logits, kv_list = self.draft_model(ids, pos, kv_caches=None)

    # Copy into a pre-allocated buffer so subsequent draft_decode can do
    # in-place writes (no per-step torch.cat).
    kv = self._alloc_draft_kv_bufs(prompt_len)
    for i, (k, v) in enumerate(kv_list):
        kv["bufs"][i][0][:, :, :prompt_len, :].copy_(k)
        kv["bufs"][i][1][:, :, :prompt_len, :].copy_(v)
    kv["len"] = prompt_len
    request.draft_kv = kv
    return logits[0, -1]
```

`draft_decode` and `rollback_draft_kv` already in 839838c are correct —
they only ever do `seq_len=1` writes, which the in-place path handles
fine.

If the fix above still asserts, fallback is even safer: **only patch
`Attention.forward`'s decode path** (`kv_buf is not None and seq_len == 1`)
and leave prefill (with or without kv_buf) on the old cat path. One line
guard in `model.py`:

```python
if kv_buf is not None and hidden.shape[1] == 1:
    # in-place decode write
    ...
else:
    # legacy cat path (also handles in-place-prefill safely)
    ...
```

---

## How to verify on a fresh L4 VM

### 0. Environment

L4 24GB (Qwen3-8B bf16 ~21 GB → ~3 GB headroom for KV + activation).
Software (same as our setup):

- `python3` + `torch==2.7.0+cu126` + `flash-attn==2.8.3` + `flashinfer-python`
- `ninja` on PATH (`~/.local/bin/ninja`) — flashinfer JIT needs it
- HF_TOKEN (yours; ours was redacted)

### 1. Clone and pick the right commit

```bash
cd ~
git clone <repo-url>
cd CS349D-miniengine

# Either: pick up where we left off and apply the safer fix above
git checkout main          # at 839838c — the broken patch
# … edit engine.py:draft_prefill as shown above …

# Or: revert to the last-known-good and reapply by hand
git checkout 5ce55fd       # last known good (full-credit numbers came from here)
```

### 2. Smoke test (verify token-identity vs non-spec)

Before any benchmark, confirm spec decode still passes MMLU within ±1pp.
This catches any KV-sync bug instantly.

```bash
# Launch spec server (K=5 is fine for the smoke test)
cat > /tmp/launch_spec.sh << 'EOF'
#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
export HF_TOKEN=YOUR_HF_TOKEN
cd ~/CS349D-miniengine
exec python3 -m miniengine \
    --model Qwen/Qwen3-8B --mode paged \
    --mem-fraction-static 0.93 --page-size 32 --prefill-chunk-size 512 \
    --speculative-draft-model Qwen/Qwen3-0.6B \
    --speculative-num-draft-tokens 5 --port 8000
EOF
chmod +x /tmp/launch_spec.sh
nohup setsid bash /tmp/launch_spec.sh > /tmp/srv.log 2>&1 < /dev/null & disown

# Wait until ready
until curl -s -m 5 http://localhost:8000/spec_stats | grep -q '"enabled":true'; do
    sleep 3
done
echo "READY"

# Warm JIT
curl -s -m 200 -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"Tell me about the moon."}],"max_tokens":32}'

# Check NO AssertionError
grep -c "AssertionError" /tmp/srv.log   # MUST be 0
```

### 3. Bonus (c) bench

Server config above (K=3 recommended — we showed it's the sweet spot).
Change `--speculative-num-draft-tokens 5` to `3` in the launch script.

```bash
cd ~/CS349D-miniengine
timeout 600 python3 -m benchmark.bench_serving \
    --model Qwen/Qwen3-8B \
    --base-url http://localhost:8000 \
    --input-len 1024 --output-len 256 \
    --concurrencies 1 --num-requests 8 \
    | tee /tmp/spec_k3_fixed.log

# Read the TPOT_p50 from the table at the bottom and /spec_stats summary
curl -s http://localhost:8000/spec_stats
```

**Pass criterion: TPOT_p50 < 55.4 ms** (= baseline 69.2 ms × 0.8 for the
20% reduction).

If passed, save:
```bash
cp /tmp/spec_k3_fixed.log ~/CS349D-miniengine/milestone4_results/
```
and commit.

### 4. Stop the VM

```bash
gcloud compute instances stop <VM_NAME> --zone <ZONE>
```

---

## Files of interest

| Path | What it does |
|---|---|
| `miniengine/spec_decode.py` | `SpeculativeDecoder` (greedy verify + accept + KV rollback) |
| `miniengine/engine.py:1241–1287` | `draft_prefill` / `draft_decode` / `rollback_draft_kv` — the path to fix |
| `miniengine/model.py:Attention.forward` | The `kv_buf` in-place path (in 839838c) — check `is_causal` + GQA expand interaction with view slices |
| `miniengine/scheduler.py:_step_spec` | conc=1 orchestration |
| `miniengine/server.py` `/spec_stats` | live accept length / fwd-per-tok |
| `milestone4_results/*.log` | All numbers we already have |

---

## Decision tree for the teammate

1. **Apply the safer `draft_prefill` fix above** → smoke test (Step 2) → if MMLU clean, run Step 3.
2. If Step 3 shows `TPOT_p50 < 55.4 ms` → bonus claimed, update the report.
3. If still > 55.4 ms (unlikely given the math) → next biggest lever is
   moving target_verify to the decode-graph path; that's a bigger change
   (touch `engine.target_verify` + reuse `CudaGraphRunner` with `--cuda-graph`).
4. If the AssertionError reappears with the safer fix → grab the
   traceback (`grep -B3 -A10 AssertionError /tmp/srv.log`) and look at
   `attention.forward` view/expand interaction.

Honest expectation: the safer fix should clear bonus (c) on the first
try. The numbers are: draft 36 → ~10 ms drops K=3 round time enough
that any reasonable accept length puts TPOT well under 55 ms.

Good luck.
