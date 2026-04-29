# MiniEngine

A minimal LLM serving engine for educational purposes, inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang).

The engine serves an OpenAI-compatible API and is structured so that the **scheduler** is the primary optimization target across a series of assignments.

## Architecture

```
┌──────────┐     ┌─────────────┐     ┌───────────┐
│  Client   │────▶│  HTTP Server │────▶│ Scheduler  │
│ (curl/py) │◀────│  (FastAPI)   │◀────│  (Thread)  │
└──────────┘     └─────────────┘     └─────┬─────┘
                                           │
                                     ┌─────▼─────┐
                                     │   Engine   │
                                     │ (HF Model) │
                                     └───────────┘
```

| Module | File | Role |
|--------|------|------|
| **core** | `miniengine/core.py` | Data structures: `Request`, `SamplingParams`, `TokenOutput` |
| **model** | `miniengine/model.py` | Bare-bone Qwen3 in pure PyTorch (GQA + RoPE + SwiGLU); loads HF safetensors directly |
| **engine** | `miniengine/engine.py` | Wraps `model.CausalLM` — tokenize, prefill, per-request and batched decode |
| **sampler** | `miniengine/sampler.py` | Top-k / top-p / temperature / repetition-penalty sampling |
| **scheduler** | `miniengine/scheduler.py` | Request scheduling loop (**optimization target**) |
| **server** | `miniengine/server.py` | OpenAI-compatible FastAPI server with SSE streaming |

## Quick Start

```bash
# Install
pip install -e .

# Launch (downloads model on first run)
python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507

# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
    "stream": true
  }'
```

## Benchmarks

Two benchmarks are provided. Both send requests to a **running** MiniEngine server.

### Serving Benchmark (`bench_serving.py`)

Measures throughput and latency under varying concurrency. Prompts are
sampled from [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M)
and truncated/padded to the target input length.

```bash
# Default: 1024 in / 512 out, randomness 0.5, sweep concurrency 1→32
python -m benchmark.bench_serving

# Custom lengths and randomness
python -m benchmark.bench_serving --input-len 512 --output-len 256 --randomness 0.5

# Specific concurrency levels
python -m benchmark.bench_serving --concurrencies 1,4,16 --num-requests 50
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input-len` | 1024 | Target input length (tokens) |
| `--output-len` | 512 | Target output length (tokens) |
| `--randomness` | 0.5 | Length randomness: `1.0` = all requests use exact target lengths; `0.0` = uniform random from `[1, target]`; `0.5` = uniform from `[target/2, target]` |
| `--concurrencies` | 1,2,4,8,16,32 | Concurrency levels to sweep |
| `--num-requests` | auto: `max(conc*2, 8)` per level | Total requests per concurrency level |

**Reports per concurrency level:** TTFT p50/p99, completion latency p50/p99, TPOT p50/p99, generation throughput (tok/s).

### Accuracy Benchmark (`bench_accuracy.py`)

Evaluates model correctness on MMLU (multiple choice) or GSM8K (math).

```bash
# MMLU — 200 questions, fast (short outputs)
python -m benchmark.bench_accuracy --dataset mmlu --num-samples 200

# GSM8K — 100 math word problems
python -m benchmark.bench_accuracy --dataset gsm8k --num-samples 100

# Filter MMLU by subject
python -m benchmark.bench_accuracy --dataset mmlu --subjects abstract_algebra,anatomy
```