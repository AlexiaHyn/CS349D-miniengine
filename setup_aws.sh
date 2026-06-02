#!/usr/bin/env bash
# setup_aws.sh — one-shot bonus (c) bench on a fresh AWS g6.xlarge
#
# What it does, end to end:
#   1. install deps (flash-attn, flashinfer, etc.) on top of the AWS Deep
#      Learning OSS PyTorch AMI
#   2. clone / pull miniengine (you already cloned if you're running this
#      from inside the repo)
#   3. launch the spec decode server (Qwen3-8B target + Qwen3-0.6B draft,
#      K=3 — the sweet spot we identified on L4)
#   4. warm JIT
#   5. run the bonus (c) bench  (bench_serving conc=1 input1024 output256)
#      and the non-spec baseline at the same config for a fair compare
#   6. print the verdict against the 55.4 ms TPOT bonus bar
#
# Usage:
#   export HF_TOKEN=<your HF token>
#   cd ~/CS349D-miniengine && bash setup_aws.sh
#
# Idempotent: re-running skips already-installed pieces.

set -euo pipefail
SCRIPT_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )
cd "$SCRIPT_DIR"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: export HF_TOKEN=<your HuggingFace token> before running."
    exit 1
fi

LOG_DIR="$SCRIPT_DIR/milestone4_results"
mkdir -p "$LOG_DIR"

# ── 1. deps ────────────────────────────────────────────────────────────
echo "[setup] checking deps..."
python3 - <<'PY' || NEED_INSTALL=1
import importlib, sys
for m in ("torch", "flash_attn", "flashinfer", "transformers", "fastapi",
          "aiohttp", "safetensors", "datasets"):
    importlib.import_module(m)
print("all deps present")
PY

if [[ "${NEED_INSTALL:-0}" == "1" ]]; then
    echo "[setup] installing missing deps..."
    pip install --user --quiet ninja
    # flash-attn needs --no-build-isolation if torch is already installed
    pip install --user --quiet \
        flashinfer-python \
        flash-attn --no-build-isolation \
        transformers safetensors huggingface_hub \
        aiohttp fastapi uvicorn datasets
fi

# put ninja on PATH (flashinfer JIT needs it)
export PATH="$HOME/.local/bin:$PATH"
which ninja >/dev/null || { echo "ERROR: ninja not on PATH"; exit 1; }

# ── 2. quick env sanity ────────────────────────────────────────────────
echo "[setup] sanity check..."
python3 - <<'PY'
import torch, flash_attn, flashinfer
print(f"torch    : {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"flash_attn: {flash_attn.__version__}")
print(f"flashinfer: {flashinfer.__version__}")
print(f"gpu      : {torch.cuda.get_device_name(0)} {torch.cuda.mem_get_info()[1]/1e9:.1f}GB")
PY

# ── 3. launch spec decode server (K=3) ─────────────────────────────────
echo "[setup] killing any old server..."
pkill -9 -f "python3 -m miniengine" || true
sleep 3

cat > /tmp/launch_spec_k3.sh <<EOF
#!/bin/bash
export PATH="\$HOME/.local/bin:\$PATH"
export HF_TOKEN=$HF_TOKEN
cd $SCRIPT_DIR
exec python3 -m miniengine \\
    --model Qwen/Qwen3-8B \\
    --mode paged \\
    --mem-fraction-static 0.93 \\
    --page-size 32 \\
    --prefill-chunk-size 512 \\
    --speculative-draft-model Qwen/Qwen3-0.6B \\
    --speculative-num-draft-tokens 3 \\
    --port 8000
EOF
chmod +x /tmp/launch_spec_k3.sh
nohup setsid bash /tmp/launch_spec_k3.sh > /tmp/srv_spec_k3.log 2>&1 < /dev/null &
disown

echo "[setup] waiting for server to load target + draft model..."
for i in {1..120}; do
    if curl -sf -m 4 http://localhost:8000/spec_stats | grep -q '"enabled":true'; then
        echo "[setup] server ready after ${i}0s"
        break
    fi
    sleep 10
done

curl -sf -m 4 http://localhost:8000/spec_stats | grep -q '"k":3' || {
    echo "ERROR: server did not come up. tail of /tmp/srv_spec_k3.log:"
    tail -40 /tmp/srv_spec_k3.log
    exit 1
}

# ── 4. warm JIT ────────────────────────────────────────────────────────
echo "[setup] warming flashinfer JIT (one-time ~30s)..."
curl -s -m 200 -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"Tell me about the moon."}],"max_tokens":32}' \
    > /dev/null
echo "[setup] checking for AssertionError in server log..."
if grep -q "AssertionError" /tmp/srv_spec_k3.log; then
    echo "ERROR: AssertionError in /tmp/srv_spec_k3.log — see context:"
    grep -B3 -A20 "AssertionError" /tmp/srv_spec_k3.log | head -50
    exit 1
fi
echo "[setup] warm OK, no assertions"

# ── 5. bonus (c) bench: spec K=3 ───────────────────────────────────────
echo "[bench] running spec K=3 bench (this is the bonus (c) candidate)..."
timeout 600 python3 -m benchmark.bench_serving \
    --model Qwen/Qwen3-8B --base-url http://localhost:8000 \
    --input-len 1024 --output-len 256 \
    --concurrencies 1 --num-requests 8 \
    2>&1 | tee "$LOG_DIR/spec_k3_aws.log"

echo "[stats] /spec_stats after bench:"
curl -s http://localhost:8000/spec_stats | tee "$LOG_DIR/spec_k3_aws_stats.json"
echo

# ── 6. non-spec baseline at the SAME config (fair compare) ─────────────
echo "[setup] stopping spec server, starting non-spec baseline..."
pkill -9 -f "python3 -m miniengine" || true
sleep 5

cat > /tmp/launch_nospec.sh <<EOF
#!/bin/bash
export PATH="\$HOME/.local/bin:\$PATH"
export HF_TOKEN=$HF_TOKEN
cd $SCRIPT_DIR
exec python3 -m miniengine \\
    --model Qwen/Qwen3-8B \\
    --mode paged \\
    --mem-fraction-static 0.93 \\
    --page-size 32 \\
    --prefill-chunk-size 512 \\
    --port 8000
EOF
chmod +x /tmp/launch_nospec.sh
nohup setsid bash /tmp/launch_nospec.sh > /tmp/srv_nospec.log 2>&1 < /dev/null &
disown

echo "[setup] waiting for non-spec server..."
for i in {1..60}; do
    if curl -sf -m 4 http://localhost:8000/cache_stats >/dev/null; then
        echo "[setup] non-spec server ready after ${i}0s"
        break
    fi
    sleep 10
done

curl -s -m 200 -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"hi"}],"max_tokens":16}' \
    > /dev/null

echo "[bench] running non-spec baseline..."
timeout 600 python3 -m benchmark.bench_serving \
    --model Qwen/Qwen3-8B --base-url http://localhost:8000 \
    --input-len 1024 --output-len 256 \
    --concurrencies 1 --num-requests 8 \
    2>&1 | tee "$LOG_DIR/nospec_baseline_aws.log"

# ── 7. verdict ────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════════"
echo "                          BONUS (c) VERDICT"
echo "════════════════════════════════════════════════════════════════════"
NOSPEC_TPOT=$(grep -E "^\s+1\s+[0-9]" "$LOG_DIR/nospec_baseline_aws.log" | tail -1 | awk '{print $6}')
SPEC_TPOT=$(grep -E "^\s+1\s+[0-9]" "$LOG_DIR/spec_k3_aws.log" | tail -1 | awk '{print $6}')
echo "non-spec baseline TPOT p50 : ${NOSPEC_TPOT} ms"
echo "spec K=3 TPOT p50          : ${SPEC_TPOT} ms"
python3 - <<PY
base = float("$NOSPEC_TPOT")
spec = float("$SPEC_TPOT")
red  = 100 * (base - spec) / base
print(f"TPOT reduction             : {red:+.1f}%")
bar = base * 0.8
print(f"bonus (c) bar (-20%)       : {bar:.1f} ms")
verdict = "PASS ✅" if spec < bar else "FAIL ❌"
print(f"verdict                    : {verdict}")
PY
echo "════════════════════════════════════════════════════════════════════"

pkill -9 -f "python3 -m miniengine" || true
echo
echo "[done] logs are in $LOG_DIR/"
echo "[done] don't forget to STOP THE INSTANCE:"
echo "       aws ec2 stop-instances --instance-ids <id> --region us-east-1"
