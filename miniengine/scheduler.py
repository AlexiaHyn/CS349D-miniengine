"""
Request scheduler — the core orchestrator of the serving engine.

The scheduler sits between the HTTP server and the model engine:

    Server  ──add_request()──▶  Scheduler  ──prefill/decode──▶  Engine
      ▲                            │
      └─── token_queue (stream) ◄──┘

It runs in a background thread, repeatedly calling step() which:
  1. Admits waiting requests and prefills them  (WAITING → RUNNING)
  2. Runs one decode step on every running request
  3. Retires finished requests                    (RUNNING → FINISHED)

"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from miniengine.core import Request, RequestStatus, TokenOutput
from miniengine.engine import Engine

logger = logging.getLogger(__name__)


class Scheduler:
    """
    FCFS scheduler with two modes:

      baseline : process one request to completion before the next.
      batched  : iteration-level batching — admit + prefill many requests,
                 then advance all running requests by one token in a
                 single batched forward pass.  New requests can join the
                 batch the same step they finish prefill.

    Public API (thread-safe):
        add_request(req)   — enqueue a new request
        start()            — launch the background scheduling loop
        stop()             — gracefully shut down
    """

    def __init__(self, engine: Engine, max_running: int = 16, mode: str = "paged"):
        self.engine = engine
        self.max_running = max_running
        self.mode = mode

        # Queues
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        # Thread control
        self._lock = threading.Lock()
        self._running_flag = False
        self._thread: threading.Thread | None = None

        # Stats
        self.total_finished: int = 0
        self.total_generated_tokens: int = 0

    # ── Public API (thread-safe) ────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """Enqueue a request for scheduling."""
        with self._lock:
            self.waiting.append(request)
            logger.info(
                "Enqueued request %s  (prompt_len=%d, waiting=%d)",
                request.request_id,
                request.num_input_tokens,
                len(self.waiting),
            )

    def start(self) -> None:
        """Start the scheduler loop in a background daemon thread."""
        self._running_flag = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread to join."""
        self._running_flag = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped")

    # ── Main loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running_flag:
            has_work = bool(self.waiting) or bool(self.running)
            if not has_work:
                time.sleep(0.005)  # idle sleep to avoid busy-waiting
                continue
            try:
                self.step()
            except Exception:
                logger.exception("Scheduler step failed")

    # ── Scheduling step ─────────────────────────────────────────────────

    def step(self) -> list[Request]:
        """
        One scheduling iteration.  Behaviour depends on self.mode.

        Returns list of requests that finished in this step.
        """
        if self.mode == "baseline":
            return self._step_baseline()
        if self.mode == "paged":
            return self._step_paged()
        return self._step_batched()

    def _step_baseline(self) -> list[Request]:
        """One request to completion per step. Maximally naive."""
        finished: list[Request] = []

        with self._lock:
            if not self.waiting:
                return finished
            req = self.waiting.popleft()

        req.status = RequestStatus.RUNNING
        token_id = self.engine.prefill(req)
        req.output_ids.append(token_id)
        self._stream_token(req, token_id)

        while not self._check_finished(req, token_id):
            token_id = self.engine.decode_step(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

        self._finish_request(req, finished)
        return finished

    def _step_batched(self) -> list[Request]:
        """
        Iteration-level batched step:
          Phase 1 — admit waiting requests and prefill them (per-request).
          Phase 2 — batched decode: one token for every running request.
        Newly prefilled requests join the decode batch in the same step.
        """
        finished: list[Request] = []

        # ── Phase 1: admit + prefill ────────────────────────────────────
        with self._lock:
            to_prefill: list[Request] = []
            while (
                self.waiting and len(self.running) + len(to_prefill) < self.max_running
            ):
                to_prefill.append(self.waiting.popleft())

        for req in to_prefill:
            req.status = RequestStatus.RUNNING
            token_id = self.engine.prefill(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)
            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                self.running.append(req)

        # ── Phase 2: batched decode ─────────────────────────────────────
        if self.running:
            token_ids = self.engine.batched_decode(self.running)
            still_running: list[Request] = []
            for req, token_id in zip(self.running, token_ids):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    still_running.append(req)
            self.running = still_running

        return finished

    def _step_paged(self) -> list[Request]:
        """
        Paged-mode step — same shape as ``_step_batched`` but:

          - admission is gated by KV-pool page availability,
          - prefill is varlen-batched (one packed forward for all admissions),
          - decode is paged + flash_attn (with optional CUDA graph),
          - finishing a request frees its KV pages via
            ``engine.free_paged_state``.
        """
        finished: list[Request] = []
        pool = self.engine.pool
        assert pool is not None, "scheduler in paged mode but engine has no KV pool"

        # ── Phase 1: admit + batched paged prefill ──────────────────────
        # Gate admission on the CUMULATIVE page reservation of the whole
        # batch, not per-request — otherwise N requests each individually
        # "fitting" can collectively exceed the pool and OOM mid-prefill,
        # corrupting in-flight state.  Reserve prompt pages + 1 decode-
        # headroom page each.  A cache hit (radix prefix match) can only
        # *reduce* the real allocation, so this estimate is a safe upper
        # bound; the pool's eviction-on-allocate recovers cached pages
        # when the free list runs short.
        with self._lock:
            to_prefill: list[Request] = []
            reserved = 0
            while (
                self.waiting
                and len(self.running) + len(to_prefill) < self.max_running
            ):
                req = self.waiting[0]
                need = pool.pages_needed(len(req.input_ids)) + 1
                # Cached pages held by the radix cache are evictable, so
                # count them as available alongside the free list.
                evictable = getattr(pool, "num_evictable", 0)
                if reserved + need > pool.num_free + evictable:
                    break  # batch is full; wait for pages to free
                reserved += need
                to_prefill.append(self.waiting.popleft())

        if to_prefill:
            for req in to_prefill:
                req.status = RequestStatus.RUNNING
            try:
                token_ids = self.engine.paged_batched_prefill(to_prefill)
            except RuntimeError as _prefill_err:
                # Pool exhausted despite the admission estimate (e.g.
                # eviction couldn't free enough because pages are pinned
                # by other in-flight requests).  Roll the whole batch back
                # to the waiting queue rather than leaving half-built
                # state in `running`; they retry next step.
                #
                # Use logger.exception so the FULL traceback is printed —
                # the plain warning message above hid the real error type.
                logger.exception(
                    "paged prefill failed (batch=%d): %s",
                    len(to_prefill), _prefill_err,
                )
                for req in to_prefill:
                    self.engine.free_paged_state(req)
                    req.status = RequestStatus.WAITING
                with self._lock:
                    for req in reversed(to_prefill):
                        self.waiting.appendleft(req)
                token_ids = None

            if token_ids is not None:
                for req, token_id in zip(to_prefill, token_ids):
                    req.output_ids.append(token_id)
                    self._stream_token(req, token_id)
                    if self._check_finished(req, token_id):
                        self._finish_request(req, finished)
                    else:
                        self.running.append(req)

        # ── Phase 2: paged batched decode (with retraction on OOM) ──────
        # Decode grows each running request's KV by one page when it
        # crosses a page boundary.  Page growth is unpredictable, so the
        # pool can drain mid-run.  Rather than crash, RETRACT a victim:
        # free its KV pages, reset it to a fresh prompt, and push it back
        # onto the waiting queue.  It will be re-admitted (and re-
        # prefilled) once capacity returns.  Victim = youngest running
        # request (last admitted, least sunk work) — a simple, effective
        # heuristic.
        while self.running and not self.engine.grow_decode_pages(self.running):
            victim = self.running.pop()  # youngest (appended last)
            self._retract_request(victim)
            logger.warning(
                "Decode OOM: retracted request %s (running=%d, waiting=%d)",
                victim.request_id,
                len(self.running),
                len(self.waiting),
            )

        if self.running:
            token_ids = self.engine.paged_batched_decode(self.running)
            still_running: list[Request] = []
            for req, token_id in zip(self.running, token_ids):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    still_running.append(req)
            self.running = still_running

        return finished

    def _retract_request(self, req: Request) -> None:
        """Evict an in-flight request back to the waiting queue.

        Frees all its KV pages (and drops any radix-cache borrow lock via
        ``free_paged_state``), discards the tokens generated so far, and
        re-enqueues it at the FRONT of the waiting queue so it's re-
        admitted first when capacity returns.  Re-prefill recomputes the
        prompt KV from scratch; partial generation is lost (acceptable —
        retraction is a rare backstop under memory pressure).
        """
        self.engine.free_paged_state(req)
        req.output_ids.clear()
        req.cache_hit_tokens = 0
        req.status = RequestStatus.WAITING
        with self._lock:
            self.waiting.appendleft(req)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _check_finished(self, req: Request, token_id: int) -> bool:
        """Decide whether a request should stop generating."""
        if req.is_finished:
            return True
        if self.engine.is_stop_token(token_id):
            return True
        return False

    def _stream_token(self, req: Request, token_id: int) -> None:
        """Push a generated token into the request's streaming queue."""
        text = self.engine.decode_token(token_id)
        req.token_queue.put(
            TokenOutput(token_id=token_id, token_text=text, finished=False)
        )

    def _finish_request(self, req: Request, finished_list: list[Request]) -> None:
        """Mark a request as finished and free its resources."""
        req.status = RequestStatus.FINISHED
        req.kv_cache = None  # release GPU memory (baseline / batched modes)
        if self.mode == "paged":
            self.engine.free_paged_state(req)
        req.token_queue.put(TokenOutput(token_id=-1, token_text="", finished=True))
        finished_list.append(req)

        self.total_finished += 1
        self.total_generated_tokens += req.num_output_tokens
        logger.info(
            "Finished request %s  (output_len=%d, running=%d, waiting=%d)",
            req.request_id,
            req.num_output_tokens,
            len(self.running),
            len(self.waiting),
        )
