"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Modes:
  - baseline / batched : per-request KV cache.
        decode_step / batched_decode.
  - paged              : pre-allocated KVMemoryPool + flash_attn varlen,
        with optional torch.compile and CUDA-graph decode.

In paged mode, per-request bookkeeping (page_table, cache_seq_len) lives
inside the engine's ``_paged_state`` dict so the public Request type
stays mode-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import CausalLM, FlashInferContext, ModelConfig, load_weights
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


# Supported attention backends for paged mode.  Each has tradeoffs:
#   flash_attn — battle-tested, fastest in our measurements, but the
#                installed kernel typically requires page_size % 256 == 0.
#   flashinfer — supports small page sizes (e.g. 16, 32), useful when
#                short sequences are common and the per-tail waste of
#                large pages outweighs the launch overhead of more pages.
ATTENTION_BACKENDS = ("flash_attn", "flashinfer")


@dataclass
class _PagedState:
    """Per-request paged-mode bookkeeping, stored inside the engine."""

    page_table: list[int] = field(default_factory=list)
    cache_seq_len: int = 0
    # Milestone-3 Part B: how many pages at the front of ``page_table``
    # were borrowed from the radix cache (so we don't double-free them
    # on completion).  The node we hold a lock on for those pages —
    # released on free.
    cache_borrowed_pages: int = 0
    cache_node: Any = None


class Engine:
    """Model wrapper supporting baseline / batched / paged decode.

    Three orthogonal opt-in flags layered on top of paged mode:
      ``mode="paged"``        — paged KV pool + flash_attn varlen.
      ``torch_compile=True``  — compile per-layer MLP + RMSNorms.
      ``cuda_graph=True``     — capture decode graph at fixed batch sizes
                                (paged only).
    """

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        mode: str = "paged",
        # ── paged + accelerator flags (additive, all opt-in) ───────────
        torch_compile: bool = False,
        cuda_graph: bool = False,
        page_size: int = 256,
        mem_fraction_static: float | None = None,
        kv_pool_gb: float | None = None,
        activation_reserve_gb: float = 4.0,
        max_position: int = 16384,
        cuda_graph_batch_sizes: list[int] | None = None,
        cuda_graph_max_pages: int = 32,
        attention_backend: str = "flashinfer",
        flashinfer_workspace_mb: int = 128,
        prefill_chunk_size: int = 0,
        disable_radix_cache: bool = False,
    ):
        if cuda_graph and mode != "paged":
            raise ValueError("cuda_graph requires mode='paged'")
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"attention_backend must be one of {ATTENTION_BACKENDS}, got {attention_backend!r}"
            )
        if prefill_chunk_size < 0:
            raise ValueError(
                f"prefill_chunk_size must be non-negative, got {prefill_chunk_size}"
            )
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.attention_backend = attention_backend
        # Milestone-3 Part A: cap on per-step prefill q-tokens.
        # 0 disables chunking (single-shot prefill, milestone-2 behavior).
        self.prefill_chunk_size = prefill_chunk_size
        # Milestone-3 Part B: radix prefix cache (wired up after pool init).
        self._disable_radix_cache = disable_radix_cache
        self.radix_cache = None  # type: ignore[assignment]

        # ── Tokenizer (still from HF — it's just a tokenizer) ──────────
        logger.info("Loading tokenizer from %s …", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # ── Model (bare-bone PyTorch, loaded from safetensors) ──────────
        logger.info("Loading model config from %s …", model_path)
        config = ModelConfig.from_pretrained(model_path)
        logger.info(
            "Config: layers=%d, hidden=%d, heads=%d, kv_heads=%d, head_dim=%d, "
            "intermediate=%d, vocab=%d, tie_embed=%s",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.vocab_size,
            config.tie_word_embeddings,
        )

        # Build on meta device — load_weights replaces parameters with
        # GPU tensors directly, so we never allocate a CPU fp32 copy.
        with torch.device("meta"):
            self.model = CausalLM(config)
        load_weights(self.model, model_path, dtype=dtype, device=device)
        self.model.eval()

        # ── Stop tokens ─────────────────────────────────────────────────
        self.stop_token_ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)
        for tok_name in ("eos_token", "pad_token"):
            tid = getattr(self.tokenizer, f"{tok_name}_id", None)
            if tid is not None:
                self.stop_token_ids.add(tid)
        for token_str in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tid)

        # ── Paged-mode setup (only when mode == "paged") ───────────────
        self.pool: KVMemoryPool | None = None
        self.page_size = page_size
        self.max_position = max_position
        self._kv_pool_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._paged_state: dict[str, _PagedState] = {}
        self._fi_workspace: torch.Tensor | None = None
        self._fi_prefill_wrapper = None
        self._fi_decode_wrapper = None
        # Per-bucket decode wrappers + pre-allocated index buffers for the
        # flashinfer + CUDA-graph path.  Empty when graphs are disabled or
        # attention_backend != "flashinfer".
        self._fi_graph_buffers: dict[int, dict] = {}
        self._fi_graph_wrappers: dict[int, Any] = {}
        self._num_qo_heads = config.num_attention_heads
        self._num_kv_heads = config.num_key_value_heads
        self._head_dim = config.head_dim

        if mode == "paged":
            kv_pool_bytes = self._compute_kv_pool_bytes(
                mem_fraction_static=mem_fraction_static,
                kv_pool_gb=kv_pool_gb,
                activation_reserve_gb=activation_reserve_gb,
            )
            self.pool = KVMemoryPool.from_budget(
                num_layers=config.num_hidden_layers,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                page_size=page_size,
                dtype=dtype,
                device=device,
                bytes_budget=kv_pool_bytes,
            )
            self._kv_pool_caches = self.pool.kv_caches
            # Pre-grow RoPE so the paged forward stays alloc-free
            # (graph-capture and torch.compile safe).
            self.model.model.rotary_emb.preallocate(
                max_position, device=device, dtype=dtype
            )

            # Milestone-3 Part B: attach the radix prefix cache.  The
            # pool consults it when ``allocate`` would otherwise fail.
            if not disable_radix_cache:
                from miniengine.radix_cache import RadixCache

                self.radix_cache = RadixCache(self.pool)
                self.pool.attach_cache(self.radix_cache)
                logger.info("RadixCache: enabled (pool eviction on demand)")
            else:
                logger.info("RadixCache: disabled by --disable-radix-cache")

            if attention_backend == "flashinfer":
                import flashinfer

                self._fi_workspace = torch.empty(
                    flashinfer_workspace_mb * 1024 * 1024,
                    dtype=torch.uint8,
                    device=device,
                )
                # Two wrappers: one for packed prefill (q_len > 1 per req),
                # one for q_len == 1 decode.  Both reuse the same workspace.
                self._fi_prefill_wrapper = (
                    flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                        self._fi_workspace, kv_layout="NHD"
                    )
                )
                self._fi_decode_wrapper = (
                    flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        self._fi_workspace, kv_layout="NHD"
                    )
                )
                logger.info(
                    "flashinfer backend ready (workspace=%d MB, page_size=%d)",
                    flashinfer_workspace_mb,
                    page_size,
                )

        if torch_compile:
            logger.info("torch.compile per-layer MLP + RMSNorms (dynamic=True)")
            for layer in self.model.model.layers:
                layer.mlp = torch.compile(layer.mlp, dynamic=True)
                layer.input_layernorm = torch.compile(
                    layer.input_layernorm, dynamic=True
                )
                layer.post_attention_layernorm = torch.compile(
                    layer.post_attention_layernorm, dynamic=True
                )
                layer.self_attn.q_norm = torch.compile(
                    layer.self_attn.q_norm, dynamic=True
                )
                layer.self_attn.k_norm = torch.compile(
                    layer.self_attn.k_norm, dynamic=True
                )

        self.graph_runner: CudaGraphRunner | None = None
        if cuda_graph:
            if cuda_graph_batch_sizes is None:
                cuda_graph_batch_sizes = [1, 2, 4, 8, 16, 32]
            cuda_graph_batch_sizes = sorted(cuda_graph_batch_sizes)

            # flashinfer's graph mode needs a wrapper per bucket built with
            # use_cuda_graph=True and pre-allocated index buffers, so plan()
            # writes scheduler state at stable addresses every step.
            if attention_backend == "flashinfer":
                import flashinfer

                def _i32(*shape: int) -> torch.Tensor:
                    return torch.zeros(*shape, dtype=torch.int32, device=device)

                for bs in cuda_graph_batch_sizes:
                    bufs = dict(
                        batch_indices=torch.arange(
                            bs, dtype=torch.int32, device=device
                        ),
                        positions=_i32(bs),
                        kv_indptr=_i32(bs + 1),
                        kv_indices=_i32(bs * cuda_graph_max_pages),
                        kv_last_page_len=_i32(bs),
                    )
                    self._fi_graph_buffers[bs] = bufs
                    self._fi_graph_wrappers[bs] = (
                        flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                            self._fi_workspace,
                            kv_layout="NHD",
                            use_cuda_graph=True,
                            paged_kv_indptr_buffer=bufs["kv_indptr"],
                            paged_kv_indices_buffer=bufs["kv_indices"],
                            paged_kv_last_page_len_buffer=bufs["kv_last_page_len"],
                        )
                    )
                logger.info(
                    "flashinfer CUDA-graph wrappers built (buckets=%s, max_pages=%d)",
                    cuda_graph_batch_sizes,
                    cuda_graph_max_pages,
                )

            self.graph_runner = CudaGraphRunner(
                self,
                batch_sizes=cuda_graph_batch_sizes,
                max_pages_per_seq=cuda_graph_max_pages,
                max_seqlen_k=max_position,
            )
            self.graph_runner.capture_all()

        logger.info(
            "Engine ready  —  mode=%s, backend=%s, compile=%s, graph=%s, "
            "vocab=%d, stop_ids=%s, params=%dM",
            mode,
            attention_backend if mode == "paged" else "n/a",
            torch_compile,
            bool(cuda_graph),
            len(self.tokenizer),
            self.stop_token_ids,
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

    # ── KV-pool sizing ──────────────────────────────────────────────────

    def _compute_kv_pool_bytes(
        self,
        mem_fraction_static: float | None,
        kv_pool_gb: float | None,
        activation_reserve_gb: float,
    ) -> int:
        """Resolve the KV-pool byte budget from the various CLI knobs.

        Precedence:
          1. ``kv_pool_gb`` (explicit override) wins.
          2. ``mem_fraction_static`` (the milestone-spec flag): the pool
             gets (total * fraction) - already-allocated weights.
          3. Fall back to ``free GPU memory - activation_reserve_gb``.
        """
        if kv_pool_gb is not None:
            return int(kv_pool_gb * 1e9)

        if not torch.cuda.is_available():
            raise RuntimeError("paged mode requires a CUDA device")
        free, total = torch.cuda.mem_get_info()
        # weights are already allocated by this point, so total - free is a
        # close proxy for "everything else we've claimed so far".
        used = total - free

        if mem_fraction_static is not None:
            static_budget = int(total * mem_fraction_static)
            kv_pool_bytes = max(0, static_budget - used)
            logger.info(
                "mem-fraction-static=%.2f → static budget=%.1f GB, "
                "weights/etc used=%.1f GB → KV pool=%.1f GB",
                mem_fraction_static,
                static_budget / 1e9,
                used / 1e9,
                kv_pool_bytes / 1e9,
            )
            return kv_pool_bytes

        kv_pool_bytes = max(int(1e9), int(free - activation_reserve_gb * 1e9))
        logger.info(
            "GPU free=%.1f GB, activation reserve=%.1f GB → KV pool=%.1f GB",
            free / 1e9,
            activation_reserve_gb,
            kv_pool_bytes / 1e9,
        )
        return kv_pool_bytes

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize_messages(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply the model's chat template and tokenize into ids."""
        kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 models support enable_thinking; silently ignore if unsupported
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id back to a string."""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    # ── Forward passes (baseline / batched) ─────────────────────────────

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """
        Run the prefill phase for one request.

        Processes the full prompt in a single forward pass, stores the
        resulting KV cache on the request, and samples the first output
        token.

        Returns:
            The first generated token id.
        """
        input_ids = torch.tensor(
            [request.input_ids], dtype=torch.long, device=self.device
        )
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)
        request.kv_cache = kv_caches

        # Sample from the last position
        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """
        Run one decode step for a request that has already been prefilled.

        Feeds the last generated token through the model together with the
        cached KV values, updates the cache, and samples the next token.

        Returns:
            The next generated token id.
        """
        input_ids = torch.tensor(
            [[request.output_ids[-1]]], dtype=torch.long, device=self.device
        )
        # Position = current KV cache length (= num tokens already processed)
        cache_len = request.kv_cache[0][0].shape[2]  # layer 0, key tensor, seq dim
        position_ids = torch.tensor([[cache_len]], device=self.device)

        logits, kv_caches = self.model(
            input_ids, position_ids, kv_caches=request.kv_cache
        )
        request.kv_cache = kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids

    # ── Batched decode ──────────────────────────────────────────────────

    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """
        Decode one token for each request in a single forward pass.

        Pads per-request KV caches to the longest in the batch, builds a
        float attention mask that ignores padding, runs the model once,
        then extracts each request's actual KV (real prefix + new token)
        and samples its next token.
        """
        if not requests:
            return []

        batch_size = len(requests)
        num_layers = len(requests[0].kv_cache)

        # Stack last generated token from each request → (batch, 1)
        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long,
            device=self.device,
        )

        # Each request's current KV length and the per-request RoPE position
        cache_lens = [req.kv_cache[0][0].shape[2] for req in requests]
        max_cache_len = max(cache_lens)
        position_ids = torch.tensor(
            [[cl] for cl in cache_lens],
            dtype=torch.long,
            device=self.device,
        )

        # Pad and stack KV caches per layer to (batch, kv_heads, max_cache_len, head_dim)
        padded_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_list, v_list = [], []
            for req in requests:
                k, v = req.kv_cache[layer_idx]
                pad_len = max_cache_len - k.shape[2]
                if pad_len > 0:
                    k = F.pad(k, (0, 0, 0, pad_len))
                    v = F.pad(v, (0, 0, 0, pad_len))
                k_list.append(k)
                v_list.append(v)
            padded_kv_caches.append(
                (torch.cat(k_list, dim=0), torch.cat(v_list, dim=0))
            )

        # Mask shape (batch, 1, 1, max_cache_len + 1): the attention forward
        # appends the new token to the cache, so kv_len = max_cache_len + 1.
        # Mask only the padding window [cl, max_cache_len) per request.
        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            max_cache_len + 1,
            device=self.device,
            dtype=self.dtype,
        )
        for i, cl in enumerate(cache_lens):
            attention_mask[i, 0, 0, cl:max_cache_len] = float("-inf")

        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=padded_kv_caches,
            attention_mask=attention_mask,
        )

        # Extract each request's real KV (actual prefix + new token at -1).
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            cl = cache_lens[i]
            per_req_kv = []
            for layer_idx in range(num_layers):
                k_full = new_kv_caches[layer_idx][0][i : i + 1]
                v_full = new_kv_caches[layer_idx][1][i : i + 1]
                k_new = torch.cat([k_full[:, :, :cl, :], k_full[:, :, -1:, :]], dim=2)
                v_new = torch.cat([v_full[:, :, :cl, :], v_full[:, :, -1:, :]], dim=2)
                per_req_kv.append((k_new, v_new))
            req.kv_cache = per_req_kv
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids

    # ── Paged-mode bookkeeping ──────────────────────────────────────────

    def _state(self, req: Request) -> _PagedState:
        """Get-or-create per-request paged state, kept in the engine."""
        s = self._paged_state.get(req.request_id)
        if s is None:
            s = _PagedState()
            self._paged_state[req.request_id] = s
        return s

    def free_paged_state(self, req: Request) -> None:
        """Release a request's pages back to the pool.

        Milestone-3 Part B: instead of returning pages straight to the
        pool, we hand the request's full token trajectory (prompt +
        generated tokens, page-aligned) to the radix cache.  The cache
        will keep the pages alive for future hits and free any
        duplicates back to the pool.  Then we drop our borrow-lock on
        whatever prefix we hit at admission time.
        """
        s = self._paged_state.pop(req.request_id, None)
        if s is None or self.pool is None:
            return

        if self.radix_cache is not None and s.page_table:
            # Full trajectory: prompt + generated tokens, page-aligned.
            all_tokens = list(req.input_ids) + list(req.output_ids)
            ps = self.page_size
            full_pages = len(all_tokens) // ps
            # Pages corresponding to the page-aligned prefix of all_tokens.
            cacheable_pages = s.page_table[:full_pages]
            # Tail page (if any) holds tokens past the last page boundary —
            # not cacheable as a whole page; return it directly.
            tail_pages = s.page_table[full_pages:]

            if full_pages > 0:
                # Insert returns redundant pages (duplicates of what was
                # already in the tree); free those back to the pool.
                _leaf, redundant = self.radix_cache.insert_and_return(
                    all_tokens[: full_pages * ps], cacheable_pages
                )
                if redundant:
                    self.pool.free(redundant)
            # Tail page never went into the cache — straight back to pool.
            if tail_pages:
                self.pool.free(tail_pages)

            # Release the borrow lock so the prefix becomes evictable.
            if s.cache_node is not None:
                self.radix_cache.dec_lock_ref(s.cache_node)
        elif s.page_table:
            # No cache — return everything to the pool the old way.
            self.pool.free(s.page_table)

    # ── Paged prefill (varlen, packed, no padding) ──────────────────────

    @torch.inference_mode()
    def paged_batched_prefill(self, requests: list[Request]) -> list[int]:
        """Pack N prompts into one varlen forward; sample first token each.

        With ``prefill_chunk_size > 0``, the prefill is split across
        multiple forward passes so the per-step q-token total never
        exceeds the budget (Milestone 3 Part A).  Otherwise the original
        single-shot path runs unchanged.

        Milestone 3 Part B: before allocating pages, ask the radix
        cache for any page-aligned prefix hit and *borrow* its pages.
        Only the uncached tail of each prompt actually flows through
        the forward — large savings on RAG / multi-turn workloads.
        """
        if not requests:
            return []
        assert self.pool is not None

        states = [self._state(r) for r in requests]
        self._setup_prefill_pages(requests, states)

        # ``cache_seq_len`` was set to matched_tokens by the helper.
        # The actual q-tokens this prefill must process are the tail.
        full_lens = [len(r.input_ids) for r in requests]
        prefill_lens = [n - s.cache_seq_len for n, s in zip(full_lens, states)]
        total_q = sum(prefill_lens)

        # Edge case: every prompt is a 100% page-aligned cache hit —
        # nothing left to prefill.  We still need a logit for the last
        # token of each prompt so we can sample.  Fall back to a tiny
        # "1-token tail" prefill by reverting the last hit page so we
        # at least re-run that one page.  Cheap (one page per request)
        # and avoids special-casing the sampler.
        if total_q == 0:
            self._revert_last_hit_page(requests, states)
            full_lens = [len(r.input_ids) for r in requests]
            prefill_lens = [n - s.cache_seq_len for n, s in zip(full_lens, states)]
            total_q = sum(prefill_lens)

        # Chunked path: any prompt that doesn't fit in one chunk runs in
        # multiple forwards.
        chunk = self.prefill_chunk_size
        if chunk > 0 and total_q > chunk:
            return self._paged_chunked_prefill(
                requests, states, full_lens, chunk
            )

        # Single-shot path — flatten only the *uncached tail* of each prompt.
        cu = _cu_seqlens(prefill_lens, self.device)
        flat_ids: list[int] = []
        flat_pos: list[int] = []
        for r, s in zip(requests, states):
            tail_start = s.cache_seq_len
            tail_end = len(r.input_ids)
            flat_ids.extend(r.input_ids[tail_start:tail_end])
            flat_pos.extend(range(tail_start, tail_end))
        input_ids = _to_long(flat_ids, self.device).unsqueeze(0)
        position_ids = _to_long(flat_pos, self.device).unsqueeze(0)
        last_idx = (cu[1:] - 1).long()

        # The backend metadata builders need q_len + k_len per request.
        # k_len is the FULL prompt length (cached + uncached) because
        # the new q-tokens attend the whole KV history.
        k_lens = full_lens
        start_offsets = [s.cache_seq_len for s in states]

        common = dict(
            kv_pool_caches=self._kv_pool_caches,
            logits_indices=last_idx,
        )
        if self.attention_backend == "flashinfer":
            backend_kwargs = self._fi_kwargs_chunked_prefill(
                states, prefill_lens, k_lens, start_offsets
            )
        else:
            backend_kwargs = self._fa_kwargs_chunked_prefill(
                states, prefill_lens, k_lens, cu, start_offsets
            )

        logits, _ = self.model(input_ids, position_ids, **common, **backend_kwargs)
        # (1, num_seqs, vocab)

        out: list[int] = []
        for i, (r, s, n) in enumerate(zip(requests, states, full_lens)):
            s.cache_seq_len = n
            out.append(
                sample_token(logits[0, i : i + 1], r.sampling_params, r.output_ids)
            )
        return out

    # ── Radix cache helpers (Milestone 3 Part B) ───────────────────────

    def _setup_prefill_pages(
        self, requests: list[Request], states: list[_PagedState]
    ) -> None:
        """Allocate per-request page tables, reusing radix-cache pages.

        Sets ``state.page_table`` to [borrowed_cache_pages ... freshly_allocated_pages]
        and ``state.cache_seq_len`` to ``matched_tokens`` so prefill
        skips the cached prefix.  Locks the cache node so it can't be
        evicted while the request holds the borrowed pages; the lock is
        released in ``free_paged_state``.
        """
        assert self.pool is not None
        for r, s in zip(requests, states):
            full_pages = self.pool.pages_needed(len(r.input_ids))
            borrowed_pages: list[int] = []
            matched = 0
            if self.radix_cache is not None:
                result = self.radix_cache.match_prefix(r.input_ids)
                if result.matched_tokens > 0 and result.last_node is not None:
                    # Lock the matched leaf so eviction can't pull these
                    # pages out from under us while the request is
                    # in-flight.
                    self.radix_cache.inc_lock_ref(result.last_node)
                    borrowed_pages = list(result.matched_pages)
                    matched = result.matched_tokens
                    s.cache_node = result.last_node
                    s.cache_borrowed_pages = len(borrowed_pages)
                    r.cache_hit_tokens = matched
            # Allocate fresh pages for the uncached tail.
            tail_pages = full_pages - len(borrowed_pages)
            fresh = self.pool.allocate(tail_pages) if tail_pages > 0 else []
            s.page_table = borrowed_pages + fresh
            s.cache_seq_len = matched

    def _revert_last_hit_page(
        self, requests: list[Request], states: list[_PagedState]
    ) -> None:
        """Reclaim the last cached page for a 100% cache-hit prompt.

        We need at least one token to flow through prefill so we can
        sample.  Drop the last borrowed page (and lock-ref it via the
        new node, which is the parent — actually we move the lock up
        the path implicitly because dec_lock_ref decrements the path
        from the locked node to root).  Then re-allocate that page so
        we'll re-prefill the last full page from scratch.
        """
        assert self.pool is not None
        for r, s in zip(requests, states):
            if s.cache_seq_len < len(r.input_ids):
                continue  # not a 100% hit, nothing to do
            if s.cache_borrowed_pages == 0:
                continue  # already nothing borrowed; shouldn't happen
            # Drop one borrowed page off the end.
            ps = self.page_size
            # Release the lock on the deepest node; the page itself
            # stays in the cache (it'll just become evictable now).
            if self.radix_cache is not None and s.cache_node is not None:
                self.radix_cache.dec_lock_ref(s.cache_node)
                # Walk up one node — the parent is the new lock target
                # for the remaining borrowed prefix.
                s.cache_node = s.cache_node.parent if s.cache_node else None
                if s.cache_node is not None and s.cache_node is not self.radix_cache.root:
                    self.radix_cache.inc_lock_ref(s.cache_node)
                else:
                    s.cache_node = None
            # Drop the last borrowed page from the page_table and
            # allocate a fresh one to write the re-prefilled K/V into.
            s.page_table.pop()
            s.cache_borrowed_pages -= 1
            s.cache_seq_len -= ps
            r.cache_hit_tokens -= ps
            s.page_table.extend(self.pool.allocate(1))

    # ── Chunked prefill (Milestone 3 Part A) ───────────────────────────

    @torch.inference_mode()
    def _paged_chunked_prefill(
        self,
        requests: list[Request],
        states: list[_PagedState],
        seq_lens: list[int],
        chunk_size: int,
    ) -> list[int]:
        """Prefill in slices of at most ``chunk_size`` q-tokens per step.

        Walk requests in order; pack as many full or partial requests as
        fit in the next chunk, run one forward (whose q-tokens attend
        the already-prefilled K/V via ``cu_seqlens_k > cu_seqlens_q``),
        and save the logit at each request's *last* q-position the step
        it actually lands.  After the loop, every request has been fully
        prefilled and we sample one token each.

        ``activation_size ∝ chunk_size``, not ``∝ total_q`` — the OOM
        avoidance guarantee.

        Each request's starting progress is ``state.cache_seq_len`` so
        any radix-cache hit is already skipped.
        """
        # Per-request progress + the logit row we'll sample from (filled
        # in on the step that completes each request).  Initial progress
        # is whatever the cache hit covered.
        prefilled = [s.cache_seq_len for s in states]
        last_logits: list[torch.Tensor | None] = [None] * len(requests)

        while any(p < n for p, n in zip(prefilled, seq_lens)):
            # Build the next chunk: walk requests, take up to chunk_size
            # q-tokens, stop at the first one that doesn't completely fit
            # (we still include a partial slice of it so we make
            # forward progress and the chunk uses the full budget).
            chunk_reqs: list[int] = []            # indices into requests
            chunk_q_lens: list[int] = []          # q-tokens this chunk per req
            chunk_start_offsets: list[int] = []   # starting offset within each req
            budget = chunk_size
            for i in range(len(requests)):
                if prefilled[i] >= seq_lens[i]:
                    continue
                remaining = seq_lens[i] - prefilled[i]
                take = min(remaining, budget)
                chunk_reqs.append(i)
                chunk_q_lens.append(take)
                chunk_start_offsets.append(prefilled[i])
                budget -= take
                if budget <= 0:
                    break

            # Build packed (input_ids, position_ids) for this chunk and
            # sub-views of states/seq_lens that the metadata builders
            # will use to express "q-tokens in this chunk attend full
            # KV [0 .. cache_seq_len + take))".
            sub_states = [states[i] for i in chunk_reqs]
            sub_q_lens = chunk_q_lens
            # cu_seqlens_q describes the chunk's q layout.
            sub_cu_q = _cu_seqlens(sub_q_lens, self.device)
            # cu_seqlens_k spans the *full* KV up to and including this
            # chunk's new tokens — that's cache_seq_len_before + take.
            sub_k_lens = [s.cache_seq_len + q for s, q in zip(sub_states, sub_q_lens)]

            flat_ids: list[int] = []
            flat_pos: list[int] = []
            for i, ri in enumerate(chunk_reqs):
                start = chunk_start_offsets[i]
                take = chunk_q_lens[i]
                flat_ids.extend(requests[ri].input_ids[start : start + take])
                flat_pos.extend(range(start, start + take))

            input_ids = _to_long(flat_ids, self.device).unsqueeze(0)
            position_ids = _to_long(flat_pos, self.device).unsqueeze(0)

            # We want the logit at the *last* position of each chunked-in
            # request — but only when that request finishes prefill on
            # this step.  For partial chunks we discard the logits.
            last_idx_in_chunk = (sub_cu_q[1:] - 1).long()
            logits_indices = last_idx_in_chunk

            common = dict(
                kv_pool_caches=self._kv_pool_caches,
                logits_indices=logits_indices,
            )
            if self.attention_backend == "flashinfer":
                backend_kwargs = self._fi_kwargs_chunked_prefill(
                    sub_states, sub_q_lens, sub_k_lens, chunk_start_offsets
                )
            else:
                backend_kwargs = self._fa_kwargs_chunked_prefill(
                    sub_states, sub_q_lens, sub_k_lens, sub_cu_q, chunk_start_offsets
                )

            logits, _ = self.model(
                input_ids, position_ids, **common, **backend_kwargs
            )
            # logits: (1, len(chunk_reqs), vocab)

            # Update progress + commit logits for requests that just finished.
            for j, ri in enumerate(chunk_reqs):
                prefilled[ri] += chunk_q_lens[j]
                sub_states[j].cache_seq_len = prefilled[ri]
                if prefilled[ri] == seq_lens[ri]:
                    # Detach so this small slice isn't tied to the
                    # chunk's full logits buffer (Inductor may free it).
                    last_logits[ri] = logits[0, j : j + 1].clone()

        # Sample one token per request, in the original order.
        out: list[int] = []
        for r, lg in zip(requests, last_logits):
            assert lg is not None, "chunked prefill must produce a final logit"
            out.append(sample_token(lg, r.sampling_params, r.output_ids))
        return out

    def _fa_kwargs_chunked_prefill(
        self,
        states: list[_PagedState],
        q_lens: list[int],
        k_lens: list[int],
        cu_q: torch.Tensor,
        start_offsets: list[int],
    ) -> dict:
        """flash_attn varlen prefill metadata for a single chunked step.

        ``slot_mapping`` writes the new K/V into the slot indices
        starting at each request's ``cache_seq_len``; ``cu_seqlens_k``
        spans the full KV (history + this chunk).
        """
        ps = self.page_size
        slot_mapping = _to_long(
            [
                s.page_table[(start + i) // ps] * ps + (start + i) % ps
                for s, q, start in zip(states, q_lens, start_offsets)
                for i in range(q)
            ],
            self.device,
        )
        cu_k = _cu_seqlens(k_lens, self.device)
        return dict(
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=self.max_position,
            max_seqlen_k=self.max_position,
            block_table=_page_table([s.page_table for s in states], self.device),
        )

    def _fi_kwargs_chunked_prefill(
        self,
        states: list[_PagedState],
        q_lens: list[int],
        k_lens: list[int],
        start_offsets: list[int],
    ) -> dict:
        """flashinfer prefill metadata for a single chunked step."""
        assert self._fi_prefill_wrapper is not None
        ps = self.page_size

        # batch_indices / positions describe where each *new* token in
        # this chunk lives in the pool (absolute token offset within the
        # request, not within the chunk).
        batch_indices = torch.tensor(
            [b for b, q in enumerate(q_lens) for _ in range(q)],
            dtype=torch.int32,
            device=self.device,
        )
        positions = torch.tensor(
            [start + i for q, start in zip(q_lens, start_offsets) for i in range(q)],
            dtype=torch.int32,
            device=self.device,
        )

        # page_counts is the number of pages needed for the FULL KV
        # length (history + this chunk) — that's how many pages the
        # attention read needs to cover.
        page_counts = [self.pool.pages_needed(n) for n in k_lens]  # type: ignore[union-attr]
        kv_indptr = _to_int32_cumsum(page_counts, self.device)
        kv_indices = torch.tensor(
            [p for s, pc in zip(states, page_counts) for p in s.page_table[:pc]],
            dtype=torch.int32,
            device=self.device,
        )
        kv_last_page_len = torch.tensor(
            [((n - 1) % ps) + 1 if n > 0 else 0 for n in k_lens],
            dtype=torch.int32,
            device=self.device,
        )
        qo_indptr = _to_int32_cumsum(q_lens, self.device)

        self._fi_prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=self._num_qo_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim_qk=self._head_dim,
            page_size=ps,
            causal=True,
            q_data_type=self.dtype,
        )

        return dict(
            fi_ctx=FlashInferContext(
                wrapper=self._fi_prefill_wrapper,
                batch_indices=batch_indices,
                positions=positions,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_last_page_len=kv_last_page_len,
            )
        )

    # ── Backend-specific prefill metadata builders ─────────────────────

    def _fa_kwargs_prefill(
        self, states: list[_PagedState], seq_lens: list[int], cu: torch.Tensor
    ) -> dict:
        """flash_attn varlen prefill metadata: slot_mapping + block_table."""
        ps = self.page_size
        slot_mapping = _to_long(
            [
                s.page_table[i // ps] * ps + i % ps
                for s, n in zip(states, seq_lens)
                for i in range(n)
            ],
            self.device,
        )
        # max_seqlen_* pinned to max_position (a constant) so torch.compile
        # doesn't retrace as actual lengths grow.  The kernel uses
        # cu_seqlens_* for true lengths; max_seqlen_* only sizes the grid.
        return dict(
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=self.max_position,
            max_seqlen_k=self.max_position,
            block_table=_page_table([s.page_table for s in states], self.device),
        )

    def _fi_kwargs_prefill(
        self, states: list[_PagedState], seq_lens: list[int]
    ) -> dict:
        """flashinfer prefill metadata: plan() the wrapper and bundle the
        per-token append metadata."""
        assert self._fi_prefill_wrapper is not None
        ps = self.page_size

        # batch_indices[i], positions[i] tell append_paged_kv_cache where
        # the i-th token in the packed flat tensor lives in the pool.
        batch_indices = torch.tensor(
            [b for b, n in enumerate(seq_lens) for _ in range(n)],
            dtype=torch.int32,
            device=self.device,
        )
        positions = torch.tensor(
            [p for n in seq_lens for p in range(n)],
            dtype=torch.int32,
            device=self.device,
        )

        # Per-request KV layout:
        #   kv_indptr  — prefix sum of pages-per-request  (B+1,)
        #   kv_indices — flat list of page indices         (sum_pages,)
        #   kv_last_page_len — fill of the last page       (B,)
        page_counts = [len(s.page_table) for s in states]
        kv_indptr = _to_int32_cumsum(page_counts, self.device)
        kv_indices = torch.tensor(
            [p for s in states for p in s.page_table],
            dtype=torch.int32,
            device=self.device,
        )
        kv_last_page_len = torch.tensor(
            [((n - 1) % ps) + 1 if n > 0 else 0 for n in seq_lens],
            dtype=torch.int32,
            device=self.device,
        )
        qo_indptr = _to_int32_cumsum(seq_lens, self.device)

        self._fi_prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=self._num_qo_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim_qk=self._head_dim,
            page_size=ps,
            causal=True,
            q_data_type=self.dtype,
        )

        return dict(
            fi_ctx=FlashInferContext(
                wrapper=self._fi_prefill_wrapper,
                batch_indices=batch_indices,
                positions=positions,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_last_page_len=kv_last_page_len,
            )
        )

    # ── Paged decode (varlen, q_len=1 per req) ──────────────────────────

    @torch.inference_mode()
    def paged_batched_decode(self, requests: list[Request]) -> list[int]:
        """One new token per request — tries CUDA graph; falls back to eager."""
        if not requests:
            return []
        assert self.pool is not None
        states = [self._state(r) for r in requests]

        # Grow page tables if cache_seq_len + 1 outgrows the current allocation.
        for s in states:
            need = self.pool.pages_needed(s.cache_seq_len + 1)
            if need > len(s.page_table):
                s.page_table.extend(self.pool.allocate(need - len(s.page_table)))

        max_pages = max(len(s.page_table) for s in states)
        if self.graph_runner is not None and self.graph_runner.can_handle(
            len(requests), max_pages
        ):
            logits = self.graph_runner.run(requests, states)
        else:
            logits = self._paged_decode_eager(requests, states)

        out: list[int] = []
        for i, (r, s) in enumerate(zip(requests, states)):
            s.cache_seq_len += 1
            out.append(
                sample_token(logits[i : i + 1], r.sampling_params, r.output_ids)
            )
        return out

    def _paged_decode_eager(
        self, requests: list[Request], states: list[_PagedState]
    ) -> torch.Tensor:
        input_ids = _to_long(
            [r.output_ids[-1] for r in requests], self.device
        ).unsqueeze(0)
        position_ids = _to_long(
            [s.cache_seq_len for s in states], self.device
        ).unsqueeze(0)

        if self.attention_backend == "flashinfer":
            backend_kwargs = self._fi_kwargs_decode(states)
        else:
            backend_kwargs = self._fa_kwargs_decode(states, len(requests))

        logits, _ = self.model(
            input_ids,
            position_ids,
            kv_pool_caches=self._kv_pool_caches,
            **backend_kwargs,
        )
        return logits[0]  # strip the packed batch dim → (B, vocab)

    # ── Backend-specific decode metadata builders ──────────────────────

    def _fa_kwargs_decode(
        self, states: list[_PagedState], batch_size: int
    ) -> dict:
        """flash_attn varlen decode metadata (q_len == 1 per request)."""
        ps = self.page_size
        cu_q = _cu_seqlens([1] * batch_size, self.device)
        cu_k = _cu_seqlens([s.cache_seq_len + 1 for s in states], self.device)
        slot_mapping = _to_long(
            [
                s.page_table[s.cache_seq_len // ps] * ps + s.cache_seq_len % ps
                for s in states
            ],
            self.device,
        )
        return dict(
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=1,
            # max_seqlen_k pinned to max_position so torch.compile doesn't
            # recompile as cache_seq_len grows.  See note in prefill.
            max_seqlen_k=self.max_position,
            block_table=_page_table([s.page_table for s in states], self.device),
        )

    def _fi_kwargs_decode(self, states: list[_PagedState]) -> dict:
        """flashinfer decode metadata: plan() the decode wrapper."""
        assert self._fi_decode_wrapper is not None
        ps = self.page_size

        # The "+1" reflects the freshly-appended decode token: that token
        # is being scattered into the pool now, so it counts toward the
        # KV that attention reads.
        new_lens = [s.cache_seq_len + 1 for s in states]
        page_counts = [self.pool.pages_needed(n) for n in new_lens]  # type: ignore[union-attr]
        assert all(
            pc <= len(s.page_table) for pc, s in zip(page_counts, states)
        ), "decode pages must already be allocated by paged_batched_decode"

        kv_indptr = _to_int32_cumsum(page_counts, self.device)
        kv_indices = torch.tensor(
            [
                p
                for s, pc in zip(states, page_counts)
                for p in s.page_table[:pc]
            ],
            dtype=torch.int32,
            device=self.device,
        )
        kv_last_page_len = torch.tensor(
            [((n - 1) % ps) + 1 for n in new_lens],
            dtype=torch.int32,
            device=self.device,
        )

        # The new K/V slot for each request: token index = cache_seq_len.
        batch_indices = torch.arange(
            len(states), dtype=torch.int32, device=self.device
        )
        positions = torch.tensor(
            [s.cache_seq_len for s in states],
            dtype=torch.int32,
            device=self.device,
        )

        self._fi_decode_wrapper.plan(
            indptr=kv_indptr,
            indices=kv_indices,
            last_page_len=kv_last_page_len,
            num_qo_heads=self._num_qo_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            page_size=ps,
            q_data_type=self.dtype,
        )
        return dict(
            fi_ctx=FlashInferContext(
                wrapper=self._fi_decode_wrapper,
                batch_indices=batch_indices,
                positions=positions,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_last_page_len=kv_last_page_len,
            )
        )


# ── helpers (paged) ────────────────────────────────────────────────────


def _to_long(xs: list[int], device: str) -> torch.Tensor:
    return torch.tensor(xs, dtype=torch.long, device=device)


def _cu_seqlens(seq_lens: list[int], device: str) -> torch.Tensor:
    """Cumulative seqlens prefixed with 0 — varlen API format."""
    s = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), s.cumsum(0).int()]
    )


def _to_int32_cumsum(counts: list[int], device: str) -> torch.Tensor:
    """``[0, c0, c0+c1, …]`` as int32 — flashinfer's indptr/qo_indptr format."""
    s = torch.tensor(counts, dtype=torch.int32, device=device)
    return torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), s.cumsum(0).int()]
    )


def _page_table(tables: list[list[int]], device: str) -> torch.Tensor:
    """Pack ragged per-request page tables into a (B, max_pages) int32 tensor."""
    max_pages = max(len(b) for b in tables)
    bt = torch.zeros(len(tables), max_pages, dtype=torch.int32, device=device)
    for i, b in enumerate(tables):
        bt[i, : len(b)] = torch.tensor(b, dtype=torch.int32, device=device)
    return bt


# ── CUDA-graph runner (paged decode only, extra credit) ────────────────


class CudaGraphRunner:
    """Capture the paged decode forward at fixed batch sizes; replay at runtime.

    Pre-allocated input tensors hold stable identities; replay rewrites
    their contents.  Padded slots write throwaway K/V into the pool's
    reserved scratch page at distinct offsets so concurrent ``index_copy_``
    writes from the kernel don't race.

    Both paged backends are supported:

      flash_attn  — captures ``flash_attn_varlen_func``; per-step metadata
                    is `slot_mapping`, `cu_seqlens_*`, `block_table`.

      flashinfer  — captures ``wrapper.run`` + ``append_paged_kv_cache``;
                    per-step metadata is `kv_indptr`, `kv_indices`,
                    `kv_last_page_len`, `positions`.  Each bucket owns a
                    wrapper built with ``use_cuda_graph=True`` so its
                    ``plan()`` writes scheduler state into the same
                    pre-allocated buffers every step.
    """

    def __init__(
        self,
        engine: "Engine",
        batch_sizes: list[int],
        max_pages_per_seq: int,
        max_seqlen_k: int,
    ):
        assert engine.pool is not None
        self.engine = engine
        self.pool = engine.pool
        self.batch_sizes = sorted(batch_sizes)
        self.max_pages = max_pages_per_seq
        self.max_seqlen_k = max_seqlen_k
        self.scratch = self.pool.SCRATCH_PAGE
        self.device = self.pool.device
        self.page_size = self.pool.page_size
        self.graphs: dict[int, dict] = {}

    def can_handle(self, B: int, max_pages: int) -> bool:
        return B <= self.batch_sizes[-1] and max_pages <= self.max_pages

    def capture_all(self) -> None:
        self._mempool = torch.cuda.graph_pool_handle()
        for bs in self.batch_sizes:
            self._capture(bs)
        logger.info("Captured CUDA graphs for batch sizes %s", self.batch_sizes)

    def _capture(self, bs: int) -> None:
        if self.engine.attention_backend == "flashinfer":
            self._capture_flashinfer(bs)
        else:
            self._capture_flash_attn(bs)

    def _capture_flash_attn(self, bs: int) -> None:
        d = self.device
        # Pre-allocated input tensors — shapes fixed at capture, contents
        # rewritten at replay.  All point at the scratch page so a stray
        # capture-time K/V write goes nowhere harmful.
        ids = torch.zeros(1, bs, dtype=torch.long, device=d)
        pos = torch.zeros(1, bs, dtype=torch.long, device=d)
        slot = torch.full(
            (bs,), self.scratch * self.page_size, dtype=torch.long, device=d
        )
        cu = torch.arange(bs + 1, dtype=torch.int32, device=d)
        pt = torch.full((bs, self.max_pages), self.scratch, dtype=torch.int32, device=d)

        kwargs = dict(
            kv_pool_caches=self.engine._kv_pool_caches,
            slot_mapping=slot,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=1,
            max_seqlen_k=self.max_seqlen_k,
            block_table=pt,
        )
        for _ in range(3):
            with torch.inference_mode():
                _ = self.engine.model(ids, pos, **kwargs)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._mempool):
            with torch.inference_mode():
                logits, _ = self.engine.model(ids, pos, **kwargs)
        self.graphs[bs] = dict(
            graph=graph,
            input_ids=ids,
            position_ids=pos,
            slot_mapping=slot,
            cu=cu,
            page_table=pt,
            logits=logits,
        )

    def _capture_flashinfer(self, bs: int) -> None:
        """flashinfer + CUDA-graph capture.

        plan() runs *outside* the captured region (its setup kernels write
        scheduler state into the workspace at addresses pinned by the
        wrapper's construction-time buffers).  The captured region is the
        model forward, whose ``wrapper.run`` and ``append_paged_kv_cache``
        calls then read from those stable addresses on every replay.
        """
        d = self.device
        e = self.engine
        bufs = e._fi_graph_buffers[bs]
        wrapper = e._fi_graph_wrappers[bs]
        page_size = self.page_size
        max_pages = self.max_pages

        ids = torch.zeros(1, bs, dtype=torch.long, device=d)
        pos = torch.zeros(1, bs, dtype=torch.long, device=d)

        # Worst-case capture-time schedule: every request owns max_pages
        # full scratch pages.  Replay will plan() with possibly-smaller
        # totals; flashinfer's use_cuda_graph mode guarantees the kernel
        # launch shape stays constant at fixed bs.
        bufs["kv_indptr"].copy_(
            torch.arange(
                0, (bs + 1) * max_pages, max_pages, dtype=torch.int32, device=d
            )
        )
        bufs["kv_indices"].fill_(self.scratch)
        bufs["kv_last_page_len"].fill_(page_size)
        # Distinct slots in the scratch page so padded K/V scatters don't
        # race with each other.
        bufs["positions"].copy_(torch.arange(bs, dtype=torch.int32, device=d))

        wrapper.plan(
            indptr=bufs["kv_indptr"],
            indices=bufs["kv_indices"],
            last_page_len=bufs["kv_last_page_len"],
            num_qo_heads=e._num_qo_heads,
            num_kv_heads=e._num_kv_heads,
            head_dim=e._head_dim,
            page_size=page_size,
            q_data_type=e.dtype,
        )

        fi_ctx = FlashInferContext(
            wrapper=wrapper,
            batch_indices=bufs["batch_indices"],
            positions=bufs["positions"],
            kv_indptr=bufs["kv_indptr"],
            kv_indices=bufs["kv_indices"],
            kv_last_page_len=bufs["kv_last_page_len"],
        )
        kwargs = dict(kv_pool_caches=e._kv_pool_caches, fi_ctx=fi_ctx)

        for _ in range(3):
            with torch.inference_mode():
                _ = e.model(ids, pos, **kwargs)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._mempool):
            with torch.inference_mode():
                logits, _ = e.model(ids, pos, **kwargs)
        self.graphs[bs] = dict(
            graph=graph,
            input_ids=ids,
            position_ids=pos,
            bufs=bufs,
            wrapper=wrapper,
            logits=logits,
        )

    def run(
        self, requests: list[Request], states: list[_PagedState]
    ) -> torch.Tensor:
        B = len(requests)
        bs = next(b for b in self.batch_sizes if b >= B)
        if self.engine.attention_backend == "flashinfer":
            return self._run_flashinfer(requests, states, B, bs)
        return self._run_flash_attn(requests, states, B, bs)

    def _run_flash_attn(
        self,
        requests: list[Request],
        states: list[_PagedState],
        B: int,
        bs: int,
    ) -> torch.Tensor:
        g = self.graphs[bs]
        d = self.device

        g["input_ids"][0, :B].copy_(
            _to_long([r.output_ids[-1] for r in requests], d)
        )
        g["position_ids"][0, :B].copy_(
            _to_long([s.cache_seq_len for s in states], d)
        )
        g["slot_mapping"][:B].copy_(
            _to_long(
                [
                    s.page_table[s.cache_seq_len // self.page_size] * self.page_size
                    + s.cache_seq_len % self.page_size
                    for s in states
                ],
                d,
            )
        )
        seq_k = torch.tensor(
            [s.cache_seq_len + 1 for s in states], dtype=torch.int32, device=d
        )
        cu_k = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=d), seq_k.cumsum(0).int()]
        )
        g["cu"][: B + 1].copy_(cu_k)

        g["page_table"][:B].fill_(self.scratch)
        for i, s in enumerate(states):
            g["page_table"][i, : len(s.page_table)] = torch.tensor(
                s.page_table, dtype=torch.int32, device=d
            )

        if bs > B:
            n_pad = bs - B
            g["input_ids"][0, B:].zero_()
            g["position_ids"][0, B:].zero_()
            g["slot_mapping"][B:].copy_(
                self.scratch * self.page_size
                + torch.arange(n_pad, dtype=torch.long, device=d)
            )
            tail = cu_k[-1].item()
            g["cu"][B + 1 : bs + 1].copy_(
                tail + torch.arange(1, n_pad + 1, dtype=torch.int32, device=d)
            )
            g["page_table"][B:].fill_(self.scratch)

        g["graph"].replay()
        return g["logits"][0, :B]

    def _run_flashinfer(
        self,
        requests: list[Request],
        states: list[_PagedState],
        B: int,
        bs: int,
    ) -> torch.Tensor:
        g = self.graphs[bs]
        d = self.device
        e = self.engine
        bufs = g["bufs"]
        wrapper = g["wrapper"]
        page_size = self.page_size

        # Per-real-request page schedule.
        page_counts = [e.pool.pages_needed(s.cache_seq_len + 1) for s in states]
        last_lens = [((s.cache_seq_len) % page_size) + 1 for s in states]
        # Padded slots: one scratch page each (cumulative +1), last_len=1.
        n_pad = bs - B
        cum = [0]
        for pc in page_counts:
            cum.append(cum[-1] + pc)
        for _ in range(n_pad):
            cum.append(cum[-1] + 1)
        n_total = cum[-1]

        indices: list[int] = []
        for s, pc in zip(states, page_counts):
            indices.extend(s.page_table[:pc])
        indices.extend([self.scratch] * n_pad)

        positions = [s.cache_seq_len for s in states]
        positions.extend(range(n_pad))  # distinct scratch-page slots

        last_full = last_lens + [1] * n_pad

        bufs["kv_indptr"].copy_(torch.tensor(cum, dtype=torch.int32, device=d))
        bufs["kv_indices"][:n_total].copy_(
            torch.tensor(indices, dtype=torch.int32, device=d)
        )
        bufs["kv_last_page_len"].copy_(
            torch.tensor(last_full, dtype=torch.int32, device=d)
        )
        bufs["positions"].copy_(torch.tensor(positions, dtype=torch.int32, device=d))

        wrapper.plan(
            indptr=bufs["kv_indptr"],
            indices=bufs["kv_indices"][:n_total],
            last_page_len=bufs["kv_last_page_len"],
            num_qo_heads=e._num_qo_heads,
            num_kv_heads=e._num_kv_heads,
            head_dim=e._head_dim,
            page_size=page_size,
            q_data_type=e.dtype,
        )

        g["input_ids"][0, :B].copy_(
            _to_long([r.output_ids[-1] for r in requests], d)
        )
        g["position_ids"][0, :B].copy_(
            _to_long([s.cache_seq_len for s in states], d)
        )
        if bs > B:
            g["input_ids"][0, B:].zero_()
            g["position_ids"][0, B:].zero_()

        g["graph"].replay()
        return g["logits"][0, :B]
