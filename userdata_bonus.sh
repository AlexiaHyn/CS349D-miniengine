#!/bin/bash
# EC2 user-data: runs autonomously on first boot as root.
# Goal: install deps, clone repo, run Track-2 bonus (c) K=3 bench + non-spec
# baseline, write results to console output (readable via
# `aws ec2 get-console-output --instance-id <id> --latest`), then shut down.
#
# No SSH needed. The only things this needs from outside are:
#   - HF_TOKEN (baked in below)
#   - AWS_DEFAULT_REGION (set to us-east-1)

set -ex
exec > >(tee /var/log/userdata.log) 2>&1
echo "=========================="
echo "USERDATA START $(date -u)"
echo "=========================="

# Auto-shutdown bumper: regardless of exit code, shut down 35min from now.
# Gives plenty of slack for the longest valid run (~25min) but won't leave
# the instance running indefinitely if the script fails or hangs.
shutdown -h +35 || true

export HF_TOKEN=REDACTED_HF_TOKEN
export AWS_DEFAULT_REGION=us-east-1

# ── 1. Wait for nvidia driver + filesystem ────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls /opt/pytorch/bin/python3

# ── 2. Activate the DLAMI pytorch venv as ubuntu ──────────────────────
sudo -i -u ubuntu bash <<'UBUNTU_EOF'
set -ex
source /opt/pytorch/bin/activate
which python3
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ── 3. Install missing deps ──────────────────────────────────────────
pip install --quiet ninja transformers safetensors huggingface_hub datasets flashinfer-python
pip install --quiet flash-attn --no-build-isolation
python3 -c "import flash_attn, flashinfer; print('flash_attn', flash_attn.__version__, 'flashinfer', flashinfer.__version__)"

# ── 4. Clone repo at latest commit (must include 0895943 dtype fix) ──
cd ~
rm -rf CS349D-miniengine
git clone https://github.com/AlexiaHyn/CS349D-miniengine.git
cd CS349D-miniengine
echo "REPO HEAD: $(git log --oneline -1)"

export HF_TOKEN=REDACTED_HF_TOKEN

# ── 5. Launch K=3 spec server in background ──────────────────────────
nohup python3 -m miniengine \
    --model Qwen/Qwen3-8B --mode paged \
    --mem-fraction-static 0.93 --page-size 32 --prefill-chunk-size 512 \
    --speculative-draft-model Qwen/Qwen3-0.6B \
    --speculative-num-draft-tokens 3 --port 8000 \
    > /tmp/srv_k3.log 2>&1 &
SPEC_PID=$!
echo "spec server PID: $SPEC_PID"

# Wait for server ready: model dl ~2min + safetensors load ~5-8min +
# draft model + JIT ~3min → total can be 10-15min on cold start.
# Poll for up to 25 min to be safe.
for i in {1..150}; do
    if curl -sf -m 4 http://localhost:8000/spec_stats 2>/dev/null | grep -q '"enabled":true'; then
        echo "spec server ready after $((i*10))s"
        break
    fi
    sleep 10
done
curl -sf -m 4 http://localhost:8000/spec_stats | grep -q '"k":3' || {
    echo "FATAL: spec server did not come up"
    tail -50 /tmp/srv_k3.log
    exit 1
}

# Warm JIT (first request compiles flashinfer kernels)
echo "warming JIT..."
curl -s -m 300 -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"Tell me about the moon."}],"max_tokens":32}' > /dev/null
echo "warm done"
if grep -q "AssertionError\|RuntimeError" /tmp/srv_k3.log; then
    echo "FATAL: errors in server log"
    grep -E "AssertionError|RuntimeError" /tmp/srv_k3.log | head -5
    exit 1
fi

# ── 6. Run K=3 spec bench ────────────────────────────────────────────
echo "===== K=3 spec bench ====="
timeout 600 python3 -m benchmark.bench_serving \
    --model Qwen/Qwen3-8B --base-url http://localhost:8000 \
    --input-len 1024 --output-len 256 \
    --concurrencies 1 --num-requests 8 \
    2>&1 | tee /tmp/spec_k3_aws.log
echo "===== /spec_stats final ====="
curl -s http://localhost:8000/spec_stats

# Capture key numbers
SPEC_TPOT=$(grep -E "^\s+1\s+[0-9]" /tmp/spec_k3_aws.log | tail -1 | awk '{print $6}')
SPEC_ACCEPT=$(curl -s http://localhost:8000/spec_stats | python3 -c "import json,sys; print(json.load(sys.stdin)['mean_accept_length'])")
echo "SPEC_TPOT_P50=${SPEC_TPOT}"
echo "SPEC_ACCEPT_LEN=${SPEC_ACCEPT}"

# ── 7. Stop spec server, launch non-spec baseline ────────────────────
kill -9 $SPEC_PID || true
sleep 5

nohup python3 -m miniengine \
    --model Qwen/Qwen3-8B --mode paged \
    --mem-fraction-static 0.93 --page-size 32 --prefill-chunk-size 512 \
    --port 8000 \
    > /tmp/srv_nospec.log 2>&1 &
NOSPEC_PID=$!
for i in {1..150}; do
    if curl -sf -m 4 http://localhost:8000/cache_stats >/dev/null 2>&1; then
        echo "nospec server ready after $((i*10))s"
        break
    fi
    sleep 10
done

curl -s -m 200 -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"hi"}],"max_tokens":16}' > /dev/null

echo "===== non-spec baseline bench ====="
timeout 600 python3 -m benchmark.bench_serving \
    --model Qwen/Qwen3-8B --base-url http://localhost:8000 \
    --input-len 1024 --output-len 256 \
    --concurrencies 1 --num-requests 8 \
    2>&1 | tee /tmp/nospec_aws.log

NOSPEC_TPOT=$(grep -E "^\s+1\s+[0-9]" /tmp/nospec_aws.log | tail -1 | awk '{print $6}')
echo "NOSPEC_TPOT_P50=${NOSPEC_TPOT}"

# ── 8. Verdict ───────────────────────────────────────────────────────
echo
echo "================================================================="
echo "                       BONUS (c) VERDICT"
echo "================================================================="
echo "non-spec baseline TPOT p50 : ${NOSPEC_TPOT} ms"
echo "spec K=3 TPOT p50          : ${SPEC_TPOT} ms"
echo "spec K=3 mean accept       : ${SPEC_ACCEPT}"
python3 - <<PY
base = float("$NOSPEC_TPOT")
spec = float("$SPEC_TPOT")
red  = 100 * (base - spec) / base
bar  = base * 0.8
verdict = "PASS" if spec < bar else "FAIL"
print(f"TPOT reduction             : {red:+.1f}%")
print(f"bonus (c) bar (-20%)       : {bar:.1f} ms")
print(f"verdict                    : {verdict}")
PY
echo "================================================================="

# Dump bench logs into console for retrieval via get-console-output
echo "===== FULL spec_k3 bench log ====="
cat /tmp/spec_k3_aws.log
echo "===== FULL nospec bench log ====="
cat /tmp/nospec_aws.log

UBUNTU_EOF

echo "=========================="
echo "USERDATA DONE $(date -u)"
echo "=========================="

# Auto-shutdown so we don't pay for idle GPU
# Comment this out for debugging
shutdown -h +2
