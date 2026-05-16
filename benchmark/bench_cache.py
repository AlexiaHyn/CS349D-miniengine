"""
Cache effectiveness benchmark with two workloads.

Workloads
---------
``shared``     K independent groups, each with a long synthetic
               shared prefix and N short unique tails.  Tunable knobs:
               ``--num-groups``, ``--questions-per-group``,
               ``--shared-prefix-len`` (per-group prefix tokens),
               ``--question-len`` (per-request unique-tail tokens).
               Same K-groups topology that subject-grouped 5-shot
               MMLU exercises in production — we use filler-based
               synthetic content so the bench has no dataset
               dependency and the prefix/tail lengths are knobs you
               can sweep.

``multiturn``  N independent sessions; each session has M sequential
               turns of a back-and-forth conversation.  Turn k's prompt
               embeds the prior turns' user messages *and* assistant
               responses, so the cumulative prefix grows turn by turn
               and subsequent turns hit on it.  Inspired by SGLang's
               benchmark/hicache/bench_multiturn.py, deliberately
               simplified — single fixed-length question per turn, no
               session arrival rate, no synthetic dataset.

Usage
-----
    # Run against a server.  Cache hit rate is parsed from each
    # response's ``usage.cache_hit_tokens``.
    python -m benchmark.bench_cache --workload shared    --num-requests 100
    python -m benchmark.bench_cache --workload multiturn --num-sessions 16 --turns-per-session 5

Validation pattern: run twice — once against ``--base-url`` of a server
launched with ``--disable-radix-cache``, once against one with the
cache on — and compare the "Wall time / Throughput / TTFT / Hit rate"
lines.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp


# ── Sample bookkeeping ─────────────────────────────────────────────────


@dataclass
class Sample:
    workload: str
    turn: int            # 0 for shared; 0..M-1 for multiturn
    session_id: int      # request index for shared; session index for multiturn
    prompt_tokens: int
    cache_hit_tokens: int
    completion_tokens: int
    start_time: float
    first_token_time: float | None
    end_time: float

    @property
    def ttft(self) -> float:
        return (self.first_token_time or self.end_time) - self.start_time

    @property
    def latency(self) -> float:
        return self.end_time - self.start_time

    @property
    def hit_rate(self) -> float:
        return self.cache_hit_tokens / max(1, self.prompt_tokens)


# ── HTTP helper (streaming so we can measure TTFT) ─────────────────────


async def stream_chat(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: list[dict],
    max_tokens: int,
) -> tuple[str, dict, float, float, float]:
    """POST /v1/chat/completions with stream=true.

    Returns (full_text, usage, t_start, t_first_token, t_end).
    """
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    parts: list[str] = []
    usage: dict = {}
    t_first: float | None = None
    t_start = time.perf_counter()
    async with session.post(
        f"{base_url}/v1/chat/completions", json=payload
    ) as resp:
        async for line in resp.content:
            s = line.decode().strip()
            if not s.startswith("data: "):
                continue
            body = s[6:]
            if body == "[DONE]":
                break
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                continue
            if "usage" in data:
                usage = data["usage"]
            delta = data["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                parts.append(content)
    t_end = time.perf_counter()
    return "".join(parts), usage, t_start, (t_first or t_end), t_end


# ── Workload: shared prefix ────────────────────────────────────────────

# A neutral filler sentence; repeating it hits a target token count
# without producing nonsense.  All workload prompts ultimately route
# through Qwen3's chat template, so the leading tokens vary only in
# repetition count.
FILLER_LINE = (
    "The fox is a small omnivorous mammal belonging to several genera. "
)


@dataclass
class PromptSpec:
    """One built prompt waiting to be fired at the server."""

    group_id: int      # integer index, used as the cache-breakdown key
    group_label: str   # display string for the per-group breakdown
    prompt: str        # the full user-message content


def make_synthetic_prefix(group_id: int, target_tokens: int) -> str:
    """Build a per-group synthetic prefix of roughly ``target_tokens``.

    The leading bytes encode ``group_id`` so two groups have distinct
    first pages — the radix cache then treats them as separate root
    children and there's no cross-group pollution.  Filler repeats of
    ``FILLER_LINE`` (~15 tokens each on Qwen3) bring the prefix up to
    the requested length; we err on the high side so the prefix is
    safely page-aligned at small ``--page-size``.
    """
    header = f"Group {group_id} context: "
    repeats = max(1, target_tokens // 13 + 1)
    return header + (FILLER_LINE * repeats)


def make_synthetic_question(question_id: int, target_tokens: int) -> str:
    """Build a synthetic question that's *unique per* ``question_id``.

    The leading bytes encode ``question_id`` so every request's unique
    tail starts with a different token sequence — the radix cache can
    never spuriously match one request's tail against another's.
    ``target_tokens`` controls how much filler trails the unique
    header; 0 (or ≤ 10) gives just the bare ID-encoded question
    (~10 tokens), larger values append filler.
    """
    base = f"Question {question_id}: please answer briefly."
    if target_tokens <= 10:
        return base
    needed = target_tokens - 10  # rough length of base in tokens
    repeats = max(0, needed // 15)
    if repeats == 0:
        return base
    return base + " " + (FILLER_LINE * repeats)


def build_synthetic_prompts(
    num_groups: int,
    questions_per_group: int,
    shared_prefix_len: int,
    question_len: int,
) -> list[PromptSpec]:
    out: list[PromptSpec] = []
    for g in range(num_groups):
        prefix = make_synthetic_prefix(g, shared_prefix_len)
        for i in range(questions_per_group):
            # Globally unique question id — guarantees every request's
            # unique tail differs from every other's, so the cache
            # only hits via the shared group prefix.
            q_id = g * questions_per_group + i
            question = make_synthetic_question(q_id, question_len)
            out.append(
                PromptSpec(
                    group_id=g,
                    group_label=f"group_{g}",
                    prompt=prefix + "\n\n" + question,
                )
            )
    return out


async def workload_shared(
    base_url: str,
    prompts: list[PromptSpec],
    concurrency: int,
    max_tokens: int,
) -> list[Sample]:
    """Run a list of pre-built prompt specs against the server.

    Worker-pool dispatches in FIFO order; ``prompts`` is expected to
    be ordered group-by-group so a group's cache primes before the
    next one begins (under serial dispatch).
    """
    queue: asyncio.Queue = asyncio.Queue()
    for i, spec in enumerate(prompts):
        queue.put_nowait((i, spec))
    samples: list[Sample] = []

    async def worker(session):
        while True:
            try:
                i, spec = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            messages = [{"role": "user", "content": spec.prompt}]
            _text, usage, t_start, t_first, t_end = await stream_chat(
                session, base_url, messages, max_tokens
            )
            samples.append(
                Sample(
                    workload="shared",
                    # Repurpose ``turn`` as group id so the report's
                    # per-group breakdown code path works unchanged.
                    turn=spec.group_id,
                    session_id=i,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    cache_hit_tokens=usage.get("cache_hit_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    start_time=t_start,
                    first_token_time=t_first,
                    end_time=t_end,
                )
            )

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300),
    ) as session:
        workers = [asyncio.create_task(worker(session)) for _ in range(concurrency)]
        await asyncio.gather(*workers)
    return samples


# ── Workload: multi-turn ───────────────────────────────────────────────

# Sample user-turn texts — short and varied so each one tokenizes
# differently, but cheap to generate.
USER_TURNS = [
    "Tell me a fun fact about the moon.",
    "Roughly how big is it?",
    "What is it made of?",
    "Has anyone landed on it?",
    "What is the dark side like?",
    "How long does it take to orbit Earth?",
    "Why do we see phases?",
    "What is the tidal effect?",
    "Could humans live there?",
    "Does it have a magnetic field?",
]


def session_system_prompt(session_id: int) -> str:
    """A per-session system prompt that's long enough to fill a page.

    Includes the session id so each session has a unique prefix and
    won't cross-contaminate cache hits between sessions.
    """
    return (
        f"You are a helpful encyclopedic assistant.  Session {session_id}. "
        f"Be concise.  Answer the user's question directly and stop. "
    ) * 4  # ~80–110 tokens at the Qwen3 tokenizer


async def workload_multiturn(
    base_url: str,
    num_sessions: int,
    turns_per_session: int,
    concurrency: int,
    max_tokens: int,
) -> list[Sample]:
    """N sessions of M turns each.  Sessions run concurrently; turns
    within a session are sequential because turn k's prompt embeds the
    assistant response from turn k-1.
    """
    queue: asyncio.Queue = asyncio.Queue()
    for sid in range(num_sessions):
        queue.put_nowait(sid)
    samples: list[Sample] = []
    rng = random.Random(42)

    async def session_runner(session):
        while True:
            try:
                sid = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            messages: list[dict] = [
                {"role": "system", "content": session_system_prompt(sid)}
            ]
            for turn in range(turns_per_session):
                # Pick a varied but reproducible question.
                user_msg = USER_TURNS[
                    (rng.randrange(len(USER_TURNS)) + turn) % len(USER_TURNS)
                ]
                messages.append({"role": "user", "content": user_msg})
                text, usage, t_start, t_first, t_end = await stream_chat(
                    session, base_url, messages, max_tokens
                )
                samples.append(
                    Sample(
                        workload="multiturn",
                        turn=turn,
                        session_id=sid,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        cache_hit_tokens=usage.get("cache_hit_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        start_time=t_start,
                        first_token_time=t_first,
                        end_time=t_end,
                    )
                )
                # Feed the assistant response into the conversation so
                # turn k+1's prompt embeds it — this is the lever that
                # makes prefix cache hits accumulate across turns.
                messages.append({"role": "assistant", "content": text})

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300),
    ) as session:
        workers = [
            asyncio.create_task(session_runner(session)) for _ in range(concurrency)
        ]
        await asyncio.gather(*workers)
    return samples


# ── Reporting ──────────────────────────────────────────────────────────


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(int(p * len(xs)), len(xs) - 1)]


def report(samples: list[Sample], workload: str, wall_time: float) -> None:
    if not samples:
        print("No samples!")
        return
    total_prompt = sum(s.prompt_tokens for s in samples)
    total_hit = sum(s.cache_hit_tokens for s in samples)
    total_gen = sum(s.completion_tokens for s in samples)
    hit_rate = total_hit / max(1, total_prompt)
    ttfts = [s.ttft for s in samples]
    lats = [s.latency for s in samples]

    print(f"\n{'=' * 64}")
    print(f"  Workload          : {workload}")
    print(f"  Requests          : {len(samples)}")
    print(f"  Wall time         : {wall_time:.2f} s")
    print(f"  Throughput        : {len(samples) / wall_time:6.2f} req/s")
    print(f"  Gen tok/s         : {total_gen / wall_time:6.0f}")
    print(f"  TTFT p50 / p99    : {statistics.median(ttfts)*1000:5.0f} ms / "
          f"{percentile(ttfts, 0.99)*1000:5.0f} ms")
    print(f"  Latency p50 / p99 : {statistics.median(lats)*1000:5.0f} ms / "
          f"{percentile(lats, 0.99)*1000:5.0f} ms")
    print(f"  Prompt tokens     : {total_prompt}")
    print(f"  Cache-hit tokens  : {total_hit}")
    print(f"  Cache hit rate    : {hit_rate * 100:.1f}%")
    print(f"{'=' * 64}")

    if workload == "multiturn":
        # Show per-turn cache-hit rate so we can see the curve climb.
        by_turn: dict[int, list[Sample]] = {}
        for s in samples:
            by_turn.setdefault(s.turn, []).append(s)
        print("\n  Per-turn breakdown:")
        print("    turn   N   prompt_tok    hit_tok   hit_rate   TTFT_p50")
        for turn in sorted(by_turn):
            ss = by_turn[turn]
            tp = sum(s.prompt_tokens for s in ss)
            th = sum(s.cache_hit_tokens for s in ss)
            mid_ttft = statistics.median([s.ttft for s in ss])
            print(
                f"    {turn:>4}  {len(ss):>3}  {tp:>10}  {th:>9}  "
                f"{100 * th / max(1, tp):>7.1f}%  {mid_ttft*1000:>7.0f} ms"
            )

    if workload == "shared":
        by_grp: dict[int, list[Sample]] = {}
        for s in samples:
            by_grp.setdefault(s.turn, []).append(s)
        # Recover the group label (from PromptSpec) — passed via the
        # group_labels mapping if any; otherwise the integer id is fine.
        labels = getattr(report, "_group_labels", None) or {}
        if len(by_grp) > 1:
            print("\n  Per-group breakdown:")
            print(
                "     grp  label                          N   prompt_tok    "
                "hit_tok   hit_rate   TTFT_p50"
            )
            for g in sorted(by_grp):
                ss = sorted(by_grp[g], key=lambda s: s.session_id)
                tp = sum(s.prompt_tokens for s in ss)
                th = sum(s.cache_hit_tokens for s in ss)
                mid_ttft = statistics.median([s.ttft for s in ss])
                label = labels.get(g, f"group_{g}")
                print(
                    f"    {g:>4}  {label[:28]:<28}  {len(ss):>3}  {tp:>10}  "
                    f"{th:>9}  {100 * th / max(1, tp):>7.1f}%  "
                    f"{mid_ttft*1000:>7.0f} ms"
                )


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cache effectiveness benchmark (shared prefix / multi-turn)",
    )
    p.add_argument("--base-url", default="http://localhost:8001")
    p.add_argument(
        "--workload",
        choices=["shared", "multiturn"],
        required=True,
        help="shared: K groups, each with a synthetic shared prefix "
        "and N unique tails.  multiturn: many sessions, each with a "
        "growing conversation.",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max generation length per request.  Lower values shift "
        "the bench toward prefill-dominated (where cache wins are "
        "biggest).",
    )
    # shared-specific
    p.add_argument(
        "--num-groups",
        type=int,
        default=10,
        help="shared workload: number of independent groups (each "
        "with its own shared prefix).",
    )
    p.add_argument(
        "--questions-per-group",
        type=int,
        default=10,
        help="shared workload: questions per group, all sharing that "
        "group's prefix.",
    )
    p.add_argument(
        "--shared-prefix-len",
        type=int,
        default=500,
        help="shared workload: target tokens in each group's prefix.",
    )
    p.add_argument(
        "--question-len",
        type=int,
        default=0,
        help="shared workload: target tokens per unique question tail "
        "(0 = use the bare canned question, ~5–10 tokens).",
    )
    # multiturn-specific
    p.add_argument("--num-sessions", type=int, default=16)
    p.add_argument("--turns-per-session", type=int, default=5)
    args = p.parse_args()

    print(f"  Server      : {args.base_url}")
    print(f"  Workload    : {args.workload}")
    if args.workload == "shared":
        print(f"  Groups      : {args.num_groups}")
        print(f"  Qs/group    : {args.questions_per_group}")
        print(f"  Prefix len  : ~{args.shared_prefix_len} tokens")
        print(f"  Question len: ~{args.question_len or '<bare>'} tokens")
    else:  # multiturn
        print(f"  Sessions    : {args.num_sessions}")
        print(f"  Turns       : {args.turns_per_session}")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Max tokens  : {args.max_tokens}")

    t0 = time.perf_counter()
    if args.workload == "shared":
        prompts = build_synthetic_prompts(
            args.num_groups,
            args.questions_per_group,
            args.shared_prefix_len,
            args.question_len,
        )
        report._group_labels = {p.group_id: p.group_label for p in prompts}  # type: ignore[attr-defined]
        samples = asyncio.run(
            workload_shared(
                args.base_url,
                prompts,
                args.concurrency,
                args.max_tokens,
            )
        )
    else:  # multiturn
        report._group_labels = {}  # type: ignore[attr-defined]
        samples = asyncio.run(
            workload_multiturn(
                args.base_url,
                args.num_sessions,
                args.turns_per_session,
                args.concurrency,
                args.max_tokens,
            )
        )
    wall_time = time.perf_counter() - t0
    report(samples, args.workload, wall_time)


if __name__ == "__main__":
    main()
