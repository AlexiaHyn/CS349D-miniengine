"""
Speculative decoding orchestrator — Milestone 4, Track 2.

Standard autoregressive decode runs one target-model forward per
generated token.  Speculative decoding (Leviathan et al., 2022,
https://arxiv.org/abs/2211.17192) instead:

  1. drafts ``K`` candidate tokens with a small, cheap draft model
     (``K`` serial forwards on a tiny model), then
  2. verifies all ``K`` candidates in ONE batched target forward over
     ``K+1`` positions, and
  3. accepts the longest prefix the target agrees with, replaces the
     first mismatch with the target's own token, and discards the rest.

Per verify the sequence advances by ``n_accepted + 1`` tokens (the ``+1``
is the target's free "bonus" token at the first rejection / end of an
all-accepted draft) for a single target forward.  The expected advance
is the **mean accept length**; target-forwards-per-token is its
reciprocal.

This module is the conc=1 / low-concurrency path.  The draft keeps its
own contiguous KV cache (see ``Engine.draft_*``); the target keeps using
the paged pool.  Only **greedy** verification is implemented here — under
greedy the speculative output is *token-identical* to plain decode,
which is the cheapest strong correctness check (a candidate is accepted
iff it equals the target's argmax at that position).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from miniengine.core import Request
from miniengine.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class SpecStats:
    """Aggregate speculative-decoding instrumentation.

    These are the numbers the milestone asks for:
      - ``mean_accept_length`` = generated tokens / target forwards.
        (Tokens advanced per target forward.  Bar is > 2.0 at conc 1.)
      - ``target_forwards_per_token`` = its reciprocal (< 1.0 with spec;
        exactly 1.0 for plain decode).
    """

    target_forwards: int = 0          # number of verify passes
    draft_forwards: int = 0           # number of draft-model decode steps
    accepted_tokens: int = 0          # candidate tokens accepted (excl. bonus)
    bonus_tokens: int = 0             # target "free" tokens (1 per verify)
    proposed_tokens: int = 0          # candidate tokens drafted + verified

    @property
    def generated_tokens(self) -> int:
        return self.accepted_tokens + self.bonus_tokens

    @property
    def mean_accept_length(self) -> float:
        if self.target_forwards == 0:
            return 0.0
        return self.generated_tokens / self.target_forwards

    @property
    def target_forwards_per_token(self) -> float:
        if self.generated_tokens == 0:
            return 0.0
        return self.target_forwards / self.generated_tokens

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted candidates that were accepted."""
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens

    def as_dict(self) -> dict:
        return {
            "target_forwards": self.target_forwards,
            "draft_forwards": self.draft_forwards,
            "accepted_tokens": self.accepted_tokens,
            "bonus_tokens": self.bonus_tokens,
            "proposed_tokens": self.proposed_tokens,
            "generated_tokens": self.generated_tokens,
            "mean_accept_length": self.mean_accept_length,
            "target_forwards_per_token": self.target_forwards_per_token,
            "acceptance_rate": self.acceptance_rate,
        }


class SpeculativeDecoder:
    """Drives greedy speculative decoding for a single request.

    Lifecycle per request:
        prefill_target → prefill_draft → loop { draft K, verify, accept }

    The engine owns both models and the paged pool; this class owns the
    accept/reject control flow and the candidate token bookkeeping.
    """

    def __init__(self, engine: Engine, stats: SpecStats | None = None):
        if engine.draft_model is None:
            raise ValueError("SpeculativeDecoder requires a draft model on the engine")
        self.engine = engine
        self.k = engine.speculative_num_draft_tokens
        # Shared/global stats (server-level) so a benchmark run can read
        # aggregate accept length; a per-request copy could be added too.
        self.stats = stats if stats is not None else SpecStats()

    @staticmethod
    def _argmax(logits: torch.Tensor) -> int:
        """Greedy pick.  ``logits`` is (vocab,)."""
        return int(logits.argmax(dim=-1).item())

    def prefill(self, request: Request) -> int:
        """Prefill both models; return the first generated token.

        The target prefill goes through the normal paged batched path so
        the radix cache / chunked prefill still apply.  The draft prefill
        builds its contiguous KV in parallel.  The first token is the
        target's argmax at the last prompt position (greedy), identical
        to non-speculative prefill.
        """
        # Target prefill (paged) — also samples the first token.
        first_tokens = self.engine.paged_batched_prefill([request])
        first = first_tokens[0]
        # Draft prefill builds its own KV over the prompt.  We discard the
        # draft's last-position logit: the canonical first token already
        # came from the target above (keeps greedy token-identity exact).
        self.engine.draft_prefill(request)
        return first

    def step(self, request: Request) -> list[int]:
        """One speculative round: draft K, verify, accept.

        Returns the list of newly generated tokens this round (length
        ``1 .. K+1``).  The caller appends them to ``request.output_ids``,
        streams them, and stops at a stop token / max length.

        Precondition: ``request.output_ids`` is non-empty (prefill already
        produced the first token) and the target paged KV + draft KV are
        in sync with the accepted sequence.
        """
        engine = self.engine
        k = self.k

        # The last committed token is the seed for the next draft phase.
        last_token = request.output_ids[-1]

        # Target paged KV currently covers [0, cache_len): the prompt plus
        # every committed token EXCEPT ``last_token`` (whose KV the verify
        # below produces).  So ``last_token`` lives at absolute position
        # ``cache_len``.  We keep the draft KV exactly as long (the commit
        # step re-syncs it at the end of every round), so the seed token's
        # draft position is also ``cache_len``.
        cache_len = engine._state(request).cache_seq_len
        assert self._draft_kv_len(request) == cache_len, (
            f"draft KV ({self._draft_kv_len(request)}) out of sync with "
            f"target ({cache_len})"
        )

        # ── 1. Draft K tokens autoregressively ─────────────────────────
        # Each draft_decode writes the draft KV for ``cur`` at ``draft_pos``
        # and returns the logits for the NEXT draft token.  After the loop
        # the draft KV covers [0, cache_len + K) — but ``draft_tokens[-1]``
        # itself has no draft KV yet (it's only a candidate).
        draft_tokens: list[int] = []
        cur = last_token
        draft_pos = cache_len
        for _ in range(k):
            logits = engine.draft_decode(request, cur, draft_pos)
            self.stats.draft_forwards += 1
            draft_pos += 1
            cur = self._argmax(logits)
            draft_tokens.append(cur)

        # ── 2. Verify with one target forward ──────────────────────────
        # candidates = [last_token, draft_1, …, draft_K]
        candidates = [last_token] + draft_tokens
        target_logits = engine.target_verify(request, candidates)  # (K+1, vocab)
        self.stats.target_forwards += 1
        self.stats.proposed_tokens += k

        target_argmax = target_logits.argmax(dim=-1).tolist()  # len K+1

        # ── 3. Accept longest matching prefix (greedy) ─────────────────
        #   target_argmax[i] is the target's token given candidates[:i+1].
        #   draft_tokens[i] is the draft's proposed token at the same slot.
        #   Accept draft_tokens[i] iff it equals target_argmax[i].
        accepted: list[int] = []
        n_accept = 0
        for i in range(k):
            t = target_argmax[i]
            if t == draft_tokens[i]:
                accepted.append(t)
                n_accept += 1
            else:
                # First mismatch: take the target's own token (the
                # "bonus") and stop.  This token is correct because its
                # KV (at position cache_len + i) was just computed.
                accepted.append(t)
                break
        else:
            # All K drafts accepted → append the target's free bonus
            # token from the last verify position.
            accepted.append(target_argmax[k])

        self.stats.accepted_tokens += n_accept
        self.stats.bonus_tokens += 1

        # ── 4. Commit KV state for both models ─────────────────────────
        # The verify wrote candidate KV at target positions [cache_len,
        # cache_len+K]: position cache_len+i holds candidates[i]'s KV
        # (candidates = [last_token, draft_1..K]).  Of those, the slots on
        # the accepted path are cache_len (last_token) .. cache_len+n_accept
        # (the last accepted draft) — that's n_accept+1 valid positions, so
        # the sequence advances to ``cache_len + n_accept + 1``.
        #
        # The bonus token (committed at position cache_len+n_accept+1) has
        # NO valid KV yet: like last_token did, its KV will be produced by
        # the NEXT round's verify, where it becomes candidates[0].  The
        # rejected drafts' KV at positions > cache_len+n_accept is garbage
        # but harmless — it's overwritten next round.
        keep_len = cache_len + n_accept + 1
        engine.advance_target_seq(request, len(accepted))  # = n_accept + 1

        # Draft KV: the draft wrote KV for last_token + draft_1..K at
        # positions [cache_len, cache_len+K).  Positions on the committed
        # path are [0, cache_len+n_accept+1) (last_token + the n_accept
        # accepted drafts).  Truncate to that so the draft KV length tracks
        # the target's exactly; the bonus token's draft KV is (re)computed
        # as the seed of the next round.
        engine.rollback_draft_kv(request, keep_len)

        return accepted

    def _draft_kv_len(self, request: Request) -> int:
        kv = request.draft_kv
        if not kv:
            return 0
        return kv[0][0].shape[2]
