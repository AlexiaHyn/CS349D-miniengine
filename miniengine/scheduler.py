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
    Continuous-batching FCFS scheduler.

    Each step() prefills any newly admitted requests (one by one) and then
    runs a single batched decode forward pass over all running requests.
    Finished requests are retired at step end so their slots are available
    immediately on the next iteration.

    Public API (thread-safe):
        add_request(req)   — enqueue a new request
        start()            — launch the background scheduling loop
        stop()             — gracefully shut down

    Internal (called from the scheduler thread):
        step()             — one full scheduling iteration
    """

    def __init__(self, engine: Engine, max_running: int = 16):
        self.engine = engine
        self.max_running = max_running

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
        One scheduling iteration — two phases:

          Phase 1: admit waiting requests (up to max_running) and prefill
                   them one by one. Each prefilled request emits its first
                   token and joins the running set.
          Phase 2: run a single batched decode forward pass over every
                   running request (including those just prefilled above).

        Requests that finish (hit EOS or max tokens) are retired at the end
        of the step; their slots become available on the next iteration.

        Returns list of requests that finished in this step.
        """
        finished: list[Request] = []

        # ── Phase 1: admit + prefill newly admitted requests ────────────
        with self._lock:
            # if not self.waiting:
            #     return finished
            # req = self.waiting.popleft()
            while self.waiting and len(self.running) < self.max_running:
                req = self.waiting.popleft()
                req.status = RequestStatus.RUNNING
                self.running.append(req)

        # ── Prefill ─────────────────────────────────────────────────────
        # req.status = RequestStatus.RUNNING
        # token_id = self.engine.prefill(req)
        # req.output_ids.append(token_id)
        # self._stream_token(req, token_id)
        for req in self.running:
            if req.kv_cache is None:
                token_id = self.engine.prefill(req)
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req,token_id):
                    self._finish_request(req, finished)
        self.running = [req for req in self.running if req.status == RequestStatus.RUNNING]
        if not self.runnning:
            return finished

        # ── Decode until finished ───────────────────────────────────────
        # while not self._check_finished(req, token_id):
        #     token_id = self.engine.decode_step(req)
        #     req.output_ids.append(token_id)
        #     self._stream_token(req, token_id)
        # self._finish_request(req, finished)
        # return finished
        token_ids = self.engine.batched_decode(self.running)
        still_running = []
        for req, token_id in zip(self.running, token_ids):
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)
            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                still_running.append(req)
        self.running = still_running
        return finished

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
        req.token_queue.put(TokenOutput(token_id=token_id, token_text=text, finished=False))

    def _finish_request(self, req: Request, finished_list: list[Request]) -> None:
        """Mark a request as finished and free its resources."""
        req.status = RequestStatus.FINISHED
        req.kv_cache = None  # release GPU memory
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
