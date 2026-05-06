"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Three decode paths:
  - decode_step(req)         : one request, used by baseline scheduler.
  - batched_decode(reqs)     : milestone-1 batched mode — pads per-request
                                KV + attention mask, single forward.
  - paged_decode(reqs)       : milestone-2 paged mode — KV lives in a
                                global pool; attention reads/writes via
                                per-request page tables.

Prefill paths:
  - prefill(req)               : single-request, contiguous-cache.
  - paged_prefill_packed(reqs) : packed batched prefill for paged mode —
                                 N prompts flattened into one packed
                                 sequence, single forward pass.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool, PagedKVMeta
from miniengine.model import CausalLM, ModelConfig, PagedAttnCtx, load_weights
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


class Engine:
    """Model wrapper supporting baseline (per-request) and batched decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        mode: str = "batched",
        page_size: int = 32,
        mem_fraction_static: float = 0.85,
        torch_compile: bool = False,
        cuda_graph: bool = False,
        cuda_graph_batch_sizes: list[int] | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.page_size = page_size
        self.mem_fraction_static = mem_fraction_static
        self.use_torch_compile = torch_compile
        self.use_cuda_graph = cuda_graph
        self.cuda_graph_batch_sizes = cuda_graph_batch_sizes or []

        # ── Tokenizer (still from HF — it's just a tokenizer) ──────────
        logger.info("Loading tokenizer from %s …", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # ── Model (bare-bone PyTorch, loaded from safetensors) ──────────
        logger.info("Loading model config from %s …", model_path)
        config = ModelConfig.from_pretrained(model_path)
        self.config = config
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

        # ── Optional torch.compile on each transformer block ────────────
        # We compile per-block (stable shapes within a block) rather than
        # the whole model — full-model compile triggers recompiles on
        # variable seq_len / kv_len.
        if torch_compile:
            self._apply_torch_compile()

        # ── Paged KV pool (constructed lazily for non-paged modes) ─────
        self.kv_pool: KVMemoryPool | None = None
        if mode == "paged":
            self.kv_pool = self._build_kv_pool()

        # ── Optional CUDA-graph runner for paged decode ─────────────────
        self.cuda_graph_runner = None
        if cuda_graph:
            if mode != "paged":
                raise ValueError("--cuda-graph requires --mode paged")
            from miniengine.cuda_graph_runner import CudaGraphRunner
            self.cuda_graph_runner = CudaGraphRunner(
                engine=self,
                bucket_batch_sizes=self.cuda_graph_batch_sizes or [1, 2, 4, 8, 16, 32],
            )
            try:
                self.cuda_graph_runner.capture()
            except Exception:
                logger.exception(
                    "CUDA graph capture failed; falling back to eager paged decode"
                )
                self.cuda_graph_runner = None

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

        logger.info(
            "Engine ready  —  vocab=%d, stop_ids=%s, params=%dM",
            len(self.tokenizer),
            self.stop_token_ids,
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

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

    # ── Forward passes ──────────────────────────────────────────────────

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

    # ── Paged-mode setup ────────────────────────────────────────────────

    def _apply_torch_compile(self) -> None:
        """Compile each TransformerBlock.

        Block-level compile keeps shapes stable within a block (only
        Python-level branching is the paged-vs-eager flag), so dynamo
        doesn't need to re-trace per step. Wrapping the whole model
        commonly hits dynamic-shape recompiles on the seq dim.
        """
        logger.info("Applying torch.compile to each TransformerBlock …")
        for i, block in enumerate(self.model.model.layers):
            self.model.model.layers[i] = torch.compile(
                block, mode="reduce-overhead", dynamic=True, fullgraph=False
            )

    def _build_kv_pool(self) -> KVMemoryPool:
        """Allocate the paged KV pool sized to mem_fraction_static.

        Budget: total GPU memory * fraction - bytes already used by the
        loaded weights. The remaining budget pays for the pool.
        """
        if self.device == "cuda" and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated()
            kv_budget = int(total * self.mem_fraction_static) - allocated
        else:
            # No CUDA — use a small default so the pool can still be
            # constructed for unit tests / CPU-only smoke tests.
            kv_budget = 1 << 30  # 1 GiB

        if kv_budget <= 0:
            raise RuntimeError(
                f"No memory left for KV pool (budget={kv_budget} B). "
                "Reduce model size or raise --mem-fraction-static."
            )

        cfg = self.config
        pool = KVMemoryPool.from_budget(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            bytes_budget=kv_budget,
        )
        logger.info(
            "KV pool: %d pages × %d tokens × %d kv_heads × %d head_dim "
            "(%d layers, ~%.2f GiB)",
            pool.num_pages,
            pool.page_size,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.num_hidden_layers,
            pool.num_pages
            * pool.page_size
            * cfg.num_key_value_heads
            * cfg.head_dim
            * cfg.num_hidden_layers
            * 2
            * torch.tensor([], dtype=self.dtype).element_size()
            / (1 << 30),
        )
        return pool

    # ── Paged: prefill + decode ─────────────────────────────────────────

    def can_admit(self, request: Request) -> bool:
        """Return True if the pool has enough free pages for this prompt."""
        assert self.kv_pool is not None
        # +1 to leave room for the first generated token.
        need = self.kv_pool.pages_needed(request.num_input_tokens + 1)
        return self.kv_pool.num_free >= need

    def _allocate_pages_for(self, request: Request, total_tokens: int) -> None:
        """Ensure request's page table covers `total_tokens`, allocating new pages."""
        assert self.kv_pool is not None
        meta: PagedKVMeta = request.kv_cache  # type: ignore[assignment]
        have_pages = len(meta.page_indices)
        need_pages = self.kv_pool.pages_needed(total_tokens)
        if need_pages > have_pages:
            extra = self.kv_pool.allocate(need_pages - have_pages)
            meta.page_indices.extend(extra)

    def free_request_pages(self, request: Request) -> None:
        """Return all pages held by `request` to the pool."""
        if self.kv_pool is None or request.kv_cache is None:
            return
        meta = request.kv_cache
        if isinstance(meta, PagedKVMeta) and meta.page_indices:
            self.kv_pool.free(meta.page_indices)
            meta.page_indices = []
            meta.length = 0

    @torch.inference_mode()
    def paged_prefill_packed(self, requests: list[Request]) -> list[int]:
        """Packed batched prefill for paged mode.

        Flattens all prompts into one packed sequence and runs a single
        forward pass through the paged-attention path. Each request gets
        its first sampled token written to output_ids.
        """
        assert self.kv_pool is not None
        if not requests:
            return []

        # Allocate page tables and assign slot mappings for each prompt.
        cu_seqlens_q = [0]
        slot_mapping_list: list[int] = []
        position_ids_list: list[int] = []
        input_ids_list: list[int] = []
        lengths_after = []
        page_indices_list: list[list[int]] = []

        for req in requests:
            prompt = req.input_ids
            L = len(prompt)
            req.kv_cache = PagedKVMeta(page_indices=[], length=0)
            self._allocate_pages_for(req, L)
            meta: PagedKVMeta = req.kv_cache
            meta.length = L
            page_indices_list.append(list(meta.page_indices))

            # Slots for the L new tokens of this prompt.
            page_size = self.kv_pool.page_size
            for tok_idx in range(L):
                page_local = tok_idx // page_size
                offset = tok_idx % page_size
                slot = meta.page_indices[page_local] * page_size + offset
                slot_mapping_list.append(slot)
                position_ids_list.append(tok_idx)

            input_ids_list.extend(prompt)
            cu_seqlens_q.append(cu_seqlens_q[-1] + L)
            lengths_after.append(L)

        # Build packed tensors. The packed sequence sits on dim 1 of
        # batch=1 input — the model treats it as one long sequence; the
        # paged-attn ctx slices it back per request.
        device = self.device
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        position_ids = torch.tensor(
            [position_ids_list], dtype=torch.long, device=device
        )
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=device)
        cu_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
        lengths_after_t = torch.tensor(
            lengths_after, dtype=torch.int32, device=device
        )

        # Pad page tables to a common max length so we can stack.
        max_pages = max(len(pt) for pt in page_indices_list)
        page_table = torch.zeros(
            len(requests), max_pages, dtype=torch.int32, device=device
        )
        for i, pt in enumerate(page_indices_list):
            page_table[i, : len(pt)] = torch.tensor(pt, dtype=torch.int32)

        ctx = PagedAttnCtx(
            kv_caches=self.kv_pool.kv_caches,
            page_size=self.kv_pool.page_size,
            page_indices=page_table,
            lengths_after=lengths_after_t,
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_q_t,
            is_decode=False,
        )

        torch.compiler.cudagraph_mark_step_begin()
        logits, _ = self.model(input_ids, position_ids, paged_ctx=ctx)
        # logits: (1, total_q, vocab). Sample the *last* logit of each
        # request's slice → first generated token.
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            last_idx = cu_seqlens_q[i + 1] - 1
            tok = sample_token(
                logits[:, last_idx, :], req.sampling_params, req.output_ids
            )
            token_ids.append(tok)
        return token_ids

    @torch.inference_mode()
    def paged_decode(self, requests: list[Request]) -> list[int]:
        """Batched decode in paged mode — one new token per request."""
        assert self.kv_pool is not None
        if not requests:
            return []

        # Try CUDA graph fast path first.
        if self.cuda_graph_runner is not None:
            replayed = self.cuda_graph_runner.try_replay(requests)
            if replayed is not None:
                return replayed

        page_size = self.kv_pool.page_size
        device = self.device

        input_ids_list: list[int] = []
        position_ids_list: list[int] = []
        slot_mapping_list: list[int] = []
        lengths_after = []
        page_indices_list: list[list[int]] = []

        for req in requests:
            meta: PagedKVMeta = req.kv_cache  # type: ignore[assignment]
            old_len = meta.length
            new_len = old_len + 1
            self._allocate_pages_for(req, new_len)
            meta.length = new_len

            # Slot for the single new token.
            page_local = old_len // page_size
            offset = old_len % page_size
            slot = meta.page_indices[page_local] * page_size + offset
            slot_mapping_list.append(slot)

            input_ids_list.append(req.output_ids[-1])
            position_ids_list.append(old_len)
            lengths_after.append(new_len)
            page_indices_list.append(list(meta.page_indices))

        batch = len(requests)
        # batch=batch, seq_len=1 — one new query per request.
        input_ids = torch.tensor(
            [[t] for t in input_ids_list], dtype=torch.long, device=device
        ).view(1, batch)  # paged path treats packed dim as the seq axis
        position_ids = torch.tensor(
            [position_ids_list], dtype=torch.long, device=device
        )
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device=device)
        cu_q_t = torch.arange(batch + 1, dtype=torch.int32, device=device)
        lengths_after_t = torch.tensor(
            lengths_after, dtype=torch.int32, device=device
        )

        max_pages = max(len(pt) for pt in page_indices_list)
        page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
        for i, pt in enumerate(page_indices_list):
            page_table[i, : len(pt)] = torch.tensor(pt, dtype=torch.int32)

        ctx = PagedAttnCtx(
            kv_caches=self.kv_pool.kv_caches,
            page_size=page_size,
            page_indices=page_table,
            lengths_after=lengths_after_t,
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_q_t,
            is_decode=True,
        )

        torch.compiler.cudagraph_mark_step_begin()
        logits, _ = self.model(input_ids, position_ids, paged_ctx=ctx)
        # logits: (1, batch, vocab). Sample one per request.
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            tok = sample_token(
                logits[:, i, :], req.sampling_params, req.output_ids
            )
            token_ids.append(tok)
        return token_ids
