"""
Serving benchmark — throughput & latency under varying concurrency.

Sends streaming requests to a running MiniEngine server, sweeping
concurrency from 1 to 32 (configurable).  Prompts are sampled from
WildChat and truncated / padded to the target input length.

Usage:
    # Start the server first, then:
    python -m benchmark.bench_serving
    python -m benchmark.bench_serving --input-len 512 --output-len 256 --randomness 0.5
    python -m benchmark.bench_serving --concurrencies 1,2,4,8,16,32

Reports per concurrency level:
    - TTFT          p50 / p99  (time to first token)
    - Completion    p50 / p99  (end-to-end request latency)
    - Generation throughput    (output tokens / s)

Requires: aiohttp, numpy, datasets, transformers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp
import numpy as np


# ── Per-request metrics ────────────────────────────────────────────────


@dataclass
class RequestMetrics:
    input_len: int
    target_output_len: int
    start_time: float = 0.0
    first_token_time: float | None = None
    end_time: float | None = None
    num_output_tokens: int = 0
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.start_time

    @property
    def completion_latency(self) -> float | None:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def tpot(self) -> float | None:
        if self.end_time is None or self.first_token_time is None:
            return None
        if self.num_output_tokens <= 1:
            return None
        return (self.end_time - self.first_token_time) / (self.num_output_tokens - 1)


# ── Prompt generation ──────────────────────────────────────────────────


def load_wildchat_prompts(num_prompts: int) -> list[str]:
    """Load user messages from WildChat as a prompt pool."""
    from datasets import load_dataset

    print("  Loading WildChat prompts...", flush=True)
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    prompts: list[str] = []
    for row in ds:
        conv = row.get("conversation", [])
        if not conv or conv[0].get("role") != "user":
            continue
        text = conv[0]["content"].strip()
        if 20 < len(text) < 50000:  # filter very short/long
            prompts.append(text)
        if len(prompts) >= num_prompts * 5:  # collect a pool to sample from
            break

    random.shuffle(prompts)
    print(f"  Loaded {len(prompts)} candidate prompts", flush=True)
    return prompts


def prepare_requests(
    prompts: list[str],
    tokenizer,
    num_requests: int,
    input_len: int,
    output_len: int,
    randomness: float,
) -> list[dict]:
    """
    Prepare requests with controlled input/output lengths.

    randomness:
        1.0 = all requests use exactly (input_len, output_len)
        0.0 = uniform random from [1, input_len] and [1, output_len]
        0.5 = uniform random from [input_len/2, input_len] etc.
    """
    def tokenize(content: str) -> list[int]:
        msgs = [{"role": "user", "content": content}]
        try:
            out = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            out = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
            )
        if hasattr(out, "keys") and "input_ids" in out:
            out = out["input_ids"]
        return out

    # Measure tokens-per-filler once so padding can jump directly to target.
    filler = " The quick brown fox jumps over the lazy dog."
    base_len = len(tokenize(""))
    tokens_per_filler = max(1, len(tokenize(filler)) - base_len)

    requests = []
    for i in range(num_requests):
        # Compute this request's target lengths
        if randomness >= 1.0:
            req_input_len = input_len
            req_output_len = output_len
        else:
            lo_frac = randomness  # e.g. 0.5 → sample from [50%, 100%] of target
            min_in = max(1, int(input_len * lo_frac))
            min_out = max(1, int(output_len * lo_frac))
            req_input_len = random.randint(min_in, input_len)
            req_output_len = random.randint(min_out, output_len)

        # Pick a prompt and truncate/pad to target input length
        raw_prompt = prompts[i % len(prompts)]
        messages = [{"role": "user", "content": raw_prompt}]
        ids = tokenize(raw_prompt)

        if len(ids) > req_input_len:
            # Truncate: decode back to text from truncated ids
            # Keep the chat template structure by truncating user content
            truncated_ids = ids[:req_input_len]
            truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
            messages = [{"role": "user", "content": truncated_text}]
            actual_input_len = req_input_len
        elif len(ids) < req_input_len:
            # Pad by estimating how many fillers we need, then calibrate.
            padded = raw_prompt
            deficit = req_input_len - len(ids)
            n = max(1, -(-deficit // tokens_per_filler))  # ceil div
            padded += filler * n
            cur_ids = tokenize(padded)
            # Calibrate: at most a couple of iterations.
            for _ in range(4):
                if len(cur_ids) >= req_input_len:
                    break
                extra = max(1, -(-(req_input_len - len(cur_ids)) // tokens_per_filler))
                padded += filler * extra
                cur_ids = tokenize(padded)
            messages = [{"role": "user", "content": padded}]
            actual_input_len = len(cur_ids)
        else:
            actual_input_len = len(ids)

        requests.append({
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": req_output_len,
            "input_len": actual_input_len,
            "output_len": req_output_len,
        })

    return requests


# ── HTTP client ─────────────────────────────────────────────────────────


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    request: dict,
) -> RequestMetrics:
    metrics = RequestMetrics(
        input_len=request["input_len"],
        target_output_len=request["output_len"],
    )
    payload = {
        "model": "default",
        "messages": request["messages"],
        "max_tokens": request["max_tokens"],
        "temperature": 0,
        "stream": True,
    }

    metrics.start_time = time.perf_counter()
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions", json=payload,
        ) as resp:
            if resp.status != 200:
                metrics.error = f"HTTP {resp.status}"
                metrics.end_time = time.perf_counter()
                return metrics

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    metrics.num_output_tokens += 1
                    if metrics.first_token_time is None:
                        metrics.first_token_time = time.perf_counter()

    except Exception as e:
        metrics.error = str(e)

    metrics.end_time = time.perf_counter()
    return metrics


async def run_at_concurrency(
    base_url: str,
    requests: list[dict],
    concurrency: int,
) -> list[RequestMetrics]:
    """Run all requests with fixed concurrency using a worker pool."""
    results: list[RequestMetrics] = []
    queue: asyncio.Queue = asyncio.Queue()
    for r in requests:
        await queue.put(r)

    async def worker(session: aiohttp.ClientSession):
        while True:
            try:
                req = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            m = await send_request(session, base_url, req)
            results.append(m)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600),
    ) as session:
        workers = [asyncio.create_task(worker(session)) for _ in range(concurrency)]
        await asyncio.gather(*workers)

    return results


# ── Reporting ───────────────────────────────────────────────────────────


def pct(arr, p):
    return np.percentile(arr, p) * 1000 if len(arr) > 0 else float("nan")


def mean_ms(arr):
    return np.mean(arr) * 1000 if len(arr) > 0 else float("nan")


def print_single_result(conc, ok, total_time, ttfts, completions, tpots, total_out):
    gen_tps = total_out / total_time if total_time > 0 else 0
    print(
        f"  conc={conc:<3}  "
        f"TTFT p50={pct(ttfts,50):>7.0f}ms  p99={pct(ttfts,99):>7.0f}ms  "
        f"Compl p50={pct(completions,50):>7.0f}ms  p99={pct(completions,99):>7.0f}ms  "
        f"GenTok/s={gen_tps:>7.0f}  "
        f"ok={ok}",
        flush=True,
    )


def print_summary_table(all_results: dict[int, list[RequestMetrics]]):
    print(f"\n{'=' * 100}")
    print(
        f"{'Conc':>5}  {'TTFT_p50':>9}  {'TTFT_p99':>9}  "
        f"{'Compl_p50':>9}  {'Compl_p99':>9}  "
        f"{'TPOT_p50':>9}  {'TPOT_p99':>9}  "
        f"{'GenTok/s':>9}  {'OK':>4}"
    )
    print(" " * 7 + "(ms)" + " " * 7 + "(ms)" + " " * 8 + "(ms)" + " " * 8 + "(ms)" + " " * 8 + "(ms)" + " " * 8 + "(ms)")
    print(f"{'-' * 100}")

    for conc in sorted(all_results.keys()):
        results = all_results[conc]
        ok = [r for r in results if r.error is None]
        if not ok:
            print(f"{conc:>5}  ALL FAILED")
            continue

        ttfts = np.array([r.ttft for r in ok if r.ttft is not None])
        completions = np.array([r.completion_latency for r in ok if r.completion_latency is not None])
        tpots = np.array([r.tpot for r in ok if r.tpot is not None])
        total_out = sum(r.num_output_tokens for r in ok)
        total_time = max(r.end_time for r in ok) - min(r.start_time for r in ok)
        gen_tps = total_out / total_time if total_time > 0 else 0

        print(
            f"{conc:>5}  "
            f"{pct(ttfts,50):>9.0f}  {pct(ttfts,99):>9.0f}  "
            f"{pct(completions,50):>9.0f}  {pct(completions,99):>9.0f}  "
            f"{pct(tpots,50):>9.1f}  {pct(tpots,99):>9.1f}  "
            f"{gen_tps:>9.0f}  {len(ok):>4}"
        )

    print(f"{'=' * 100}")


# ── Main ────────────────────────────────────────────────────────────────


async def async_main(args, requests_pool, concurrencies):
    base_url = args.base_url
    all_results: dict[int, list[RequestMetrics]] = {}

    for conc in concurrencies:
        # Use a fresh copy of requests each time
        reqs = requests_pool[:args.num_requests]
        print(f"\n  Running concurrency={conc} ({len(reqs)} requests)...", flush=True)

        t0 = time.perf_counter()
        results = await run_at_concurrency(base_url, reqs, conc)
        total_time = time.perf_counter() - t0

        all_results[conc] = results

        ok = [r for r in results if r.error is None]
        ttfts = np.array([r.ttft for r in ok if r.ttft is not None])
        completions = np.array([r.completion_latency for r in ok if r.completion_latency is not None])
        tpots = np.array([r.tpot for r in ok if r.tpot is not None])
        total_out = sum(r.num_output_tokens for r in ok)

        print_single_result(conc, len(ok), total_time, ttfts, completions, tpots, total_out)

    print_summary_table(all_results)


def main():
    p = argparse.ArgumentParser(description="Serving benchmark (throughput + latency)")
    p.add_argument("--base-url", type=str, default="http://localhost:8000")
    p.add_argument("--model", type=str, default=None,
                   help="HuggingFace model id (for tokenizer). Auto-detected from server if not set.")
    p.add_argument("--num-requests", type=int, default=100)
    p.add_argument("--input-len", type=int, default=1024,
                   help="Target input length in tokens")
    p.add_argument("--output-len", type=int, default=1024,
                   help="Target output length in tokens")
    p.add_argument("--randomness", type=float, default=1.0,
                   help="Length randomness: 1.0=fixed, 0.0=uniform [1,target], 0.5=[target/2,target]")
    p.add_argument("--concurrencies", type=str, default="1,2,4,8,16,32",
                   help="Comma-separated concurrency levels to sweep")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    concurrencies = [int(x) for x in args.concurrencies.split(",")]

    # Auto-detect model from server
    model_id = args.model
    if model_id is None:
        import requests as req_lib
        try:
            resp = req_lib.get(f"{args.base_url}/v1/models", timeout=5)
            model_id = resp.json()["data"][0]["id"]
            print(f"  Auto-detected model: {model_id}", flush=True)
        except Exception:
            print("  ERROR: Could not detect model from server. Use --model to specify.", flush=True)
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Serving Benchmark")
    print(f"  Server       : {args.base_url}")
    print(f"  Model        : {model_id}")
    print(f"  Requests     : {args.num_requests}")
    print(f"  Input len    : {args.input_len}")
    print(f"  Output len   : {args.output_len}")
    print(f"  Randomness   : {args.randomness}")
    print(f"  Concurrencies: {concurrencies}")
    print(f"{'=' * 60}")

    # Load tokenizer
    print("\n  Loading tokenizer...", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load prompts from WildChat
    prompts = load_wildchat_prompts(args.num_requests)

    # Prepare requests with controlled lengths
    print(f"  Preparing {args.num_requests} requests...", flush=True)
    requests_pool = prepare_requests(
        prompts, tokenizer, args.num_requests,
        args.input_len, args.output_len, args.randomness,
    )
    actual_in = [r["input_len"] for r in requests_pool]
    actual_out = [r["output_len"] for r in requests_pool]
    print(
        f"  Input lengths:  min={min(actual_in)}, max={max(actual_in)}, mean={sum(actual_in)/len(actual_in):.0f}\n"
        f"  Output lengths: min={min(actual_out)}, max={max(actual_out)}, mean={sum(actual_out)/len(actual_out):.0f}",
        flush=True,
    )
    del tokenizer

    asyncio.run(async_main(args, requests_pool, concurrencies))


if __name__ == "__main__":
    main()
