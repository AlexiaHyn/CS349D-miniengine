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
                                per-request page tables, dispatched to
                                flash_attn_with_kvcache.

Prefill paths:
  - prefill(req)               : single-request, contiguous-cache.
  - paged_prefill_packed(reqs) : packed batched prefill for paged mode —
                                 N prompts flattened into one packed
                                 sequence, single flash_attn_varlen_func
                                 forward pass.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch._dynamo
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import (
    CausalLM,
    ModelConfig,
    PagedMeta,
    RotaryEmbedding,
    TransformerBlock,
    load_weights,
)
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)

# ── Process-wide perf knobs (Round-1 optimizations) ────────────────────
# 1. TF32 for fp32 matmul reductions (lm_head + RMSNorm variance).
#    Accuracy impact on Qwen3-8B is < 1e-3, MMLU unchanged.
torch.set_float32_matmul_precision("high")

# 2. Inductor cache room for many specialized graphs:
#    5 sub-regions × 36 layers × multiple bucket batch sizes = a lot of
#    distinct compiled shapes. Default cache_size_limit=8 evicts and
#    re-traces. Bumping it keeps everything resident.
torch._dynamo.config.cache_size_limit = 256


class Engine:
    """Model wrapper supporting baseline / batched / paged decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        mode: str = "batched",
        page_size: int = 256,
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

        # ── Stop tokens ─────────────────────────────────────────────────
        self.stop_token_ids = self._collect_stop_token_ids()

        # ── Paged KV pool (constructed only for --mode paged) ──────────
        self.kv_pool: KVMemoryPool | None = None
        if mode == "paged":
            # flash-attn 2.x's paged-kv kernel hard-requires
            # block_size % 256 == 0; surface the constraint up front.
            if self.page_size % 256 != 0:
                raise ValueError(
                    f"--page-size must be a multiple of 256 in paged mode "
                    f"(flash-attn paged-kv kernel constraint), got {self.page_size}."
                )
            self._build_kv_pool(config)

        # ── Optional torch.compile (Part C) ─────────────────────────────
        # Done AFTER the pool is built so the RoPE warm-up below sees the
        # final cache footprint.
        if torch_compile:
            self._apply_torch_compile(config)

        # ── Optional CUDA-graph runner for paged decode (extra credit) ─
        self.cuda_graph_runner = None
        if cuda_graph:
            if mode != "paged":
                raise ValueError("--cuda-graph requires --mode paged")
            from miniengine.cuda_graph_runner import CudaGraphRunner

            self.cuda_graph_runner = CudaGraphRunner(
                engine=self,
                bucket_batch_sizes=self.cuda_graph_batch_sizes
                or [1, 2, 4, 8, 16, 32],
            )
            try:
                self.cuda_graph_runner.capture()
            except Exception:
                logger.exception(
                    "CUDA graph capture failed; falling back to eager paged decode"
                )
                self.cuda_graph_runner = None

        logger.info(
            "Engine ready  —  vocab=%d, stop_ids=%s, params=%dM",
            len(self.tokenizer),
            self.stop_token_ids,
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

    # ── Setup helpers ───────────────────────────────────────────────────

    def _collect_stop_token_ids(self) -> set[int]:
        ids: set[int] = set()
        for attr in ("eos_token_id", "pad_token_id"):
            tid = getattr(self.tokenizer, attr, None)
            if tid is not None:
                ids.add(tid)
        for token_str in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                ids.add(tid)
        return ids

    def _build_kv_pool(self, config: ModelConfig) -> None:
        """Allocate the KV pool from
        (mem_fraction_static * total_vram) - weights_bytes."""
        if not torch.cuda.is_available() or self.device == "cpu":
            raise RuntimeError("paged mode currently requires a CUDA device")

        total_vram = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory
        weights_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        bytes_budget = int(self.mem_fraction_static * total_vram) - weights_bytes
        if bytes_budget <= 0:
            raise RuntimeError(
                f"mem_fraction_static={self.mem_fraction_static} leaves no room "
                f"for the KV pool (weights={weights_bytes / 1e9:.2f} GB, "
                f"budget={self.mem_fraction_static * total_vram / 1e9:.2f} GB)"
            )

        self.kv_pool = KVMemoryPool.from_budget(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            bytes_budget=bytes_budget,
        )
        logger.info(
            "KV pool ready  —  %s  (budget=%.2f GB, weights=%.2f GB, total=%.2f GB)",
            self.kv_pool,
            bytes_budget / 1e9,
            weights_bytes / 1e9,
            total_vram / 1e9,
        )

    def _apply_torch_compile(self, config: ModelConfig) -> None:
        """Compile five sub-regions per block ("narrow island" style).

        We compile at boundaries that already break dynamo's graph
        anyway (the flash-attn call), so each compiled region stays a
        clean Inductor graph and shape changes in attention don't force
        the rest of the block to retrace.

        Pre-extends the RoPE cos/sin cache before compiling so the
        `.item()`-bearing lazy-grow branch never fires under the compiled
        path (it would graph-break).
        """
        # Prime the RoPE cache so its lazy-grow `.item()` branch never
        # fires inside compiled forwards (graph break + sync).
        max_pos = min(config.max_position_embeddings, 16384)
        for module in self.model.modules():
            if isinstance(module, RotaryEmbedding):
                module.extend_to(max_pos)
        logger.info("Pre-extended RoPE cos/sin cache to length %d", max_pos)

        opts = dict(mode="default", dynamic=True)
        n = 0
        for block in self.model.modules():
            if not isinstance(block, TransformerBlock):
                continue
            # First block uses bare input_layernorm (residual=None branch);
            # blocks 1+ go through _add_norm_pre. Compile both.
            block.input_layernorm = torch.compile(block.input_layernorm, **opts)
            block._add_norm_pre = torch.compile(block._add_norm_pre, **opts)
            block._add_norm_post = torch.compile(block._add_norm_post, **opts)
            block.mlp = torch.compile(block.mlp, **opts)
            block.self_attn._proj_qkv_with_rope = torch.compile(
                block.self_attn._proj_qkv_with_rope, **opts
            )
            n += 1

        # Also wrap the final hidden -> logits projection. lm_head is a
        # 152K-vocab matmul that runs every decode step; compiling it
        # lets Inductor pick a tuned epilogue and (with TF32 enabled
        # process-wide) use TF32 accumulation.
        self.model.project_logits = torch.compile(
            self.model.project_logits, **opts
        )

        logger.info(
            "torch.compile: wrapped %d blocks + project_logits "
            "(input_norm + qkv_with_rope + add_norm_pre + add_norm_post + mlp + lm_head)",
            n,
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

    # ── Forward passes (baseline mode) ──────────────────────────────────

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """One-request prefill (baseline mode). Stores KV on the request,
        returns the first sampled token."""
        input_ids = torch.tensor(
            [request.input_ids], dtype=torch.long, device=self.device
        )
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)
        request.kv_cache = kv_caches
        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """One-request decode step (baseline mode)."""
        input_ids = torch.tensor(
            [[request.output_ids[-1]]], dtype=torch.long, device=self.device
        )
        cache_len = request.kv_cache[0][0].shape[2]
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

    # ── Batched decode (milestone-1 mode) ───────────────────────────────

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

    # ── Paged: prefill + decode (milestone-2 mode) ──────────────────────

    def can_admit(self, request: Request) -> bool:
        """Return True if the pool has enough free pages for this prompt
        plus one headroom page for the first decode step."""
        assert self.kv_pool is not None
        need = self.kv_pool.pages_needed(request.num_input_tokens) + 1
        return self.kv_pool.num_free >= need

    def free_request_pages(self, request: Request) -> None:
        """Return a request's pages to the pool (paged mode)."""
        if self.kv_pool is None:
            return
        if request.page_indices:
            self.kv_pool.free(request.page_indices)
            request.page_indices = []
            request.seq_len = 0

    @torch.inference_mode()
    def paged_prefill_packed(self, requests: list[Request]) -> list[int]:
        """Packed batched prefill for paged mode.

        Flattens all prompts into one packed sequence and runs a single
        flash_attn_varlen_func forward pass through the paged-attention
        path. Each request's `seq_len` is set on success and the first
        sampled token is returned.

        Pre-condition: scheduler has already allocated `request.page_indices`
        sized to fit the prompt. This method does NOT touch the free list.
        """
        assert self.kv_pool is not None
        if not requests:
            return []
        device = self.device
        page_size = self.kv_pool.page_size

        seq_lens = [r.num_input_tokens for r in requests]
        cu = [0]
        for sl in seq_lens:
            cu.append(cu[-1] + sl)

        flat_ids: list[int] = []
        flat_pos: list[int] = []
        flat_slot: list[int] = []
        for r in requests:
            flat_ids.extend(r.input_ids)
            flat_pos.extend(range(r.num_input_tokens))
            for tok_idx in range(r.num_input_tokens):
                page_id = r.page_indices[tok_idx // page_size]
                flat_slot.append(page_id * page_size + (tok_idx % page_size))

        input_ids = torch.tensor(flat_ids, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = torch.tensor(
            flat_pos, dtype=torch.long, device=device
        ).unsqueeze(0)
        meta = PagedMeta(
            phase="prefill",
            page_size=page_size,
            cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor(cu, dtype=torch.int32, device=device),
            max_seqlen_q=max(seq_lens),
            max_seqlen_k=max(seq_lens),
            slot_mapping=torch.tensor(flat_slot, dtype=torch.long, device=device),
        )

        # Run only the transformer body; we'll project just the last hidden
        # state of each request through lm_head — running it over the full
        # packed sequence wastes a (sum_q, vocab) matmul we'd discard.
        hidden, _ = self.model.model(
            input_ids,
            position_ids,
            kv_caches=self.kv_pool.kv_caches,
            paged_meta=meta,
        )
        last_indices = torch.tensor(
            [cu[i + 1] - 1 for i in range(len(requests))],
            dtype=torch.long,
            device=device,
        )
        last_hidden = hidden[:, last_indices, :]
        # Use the (potentially compiled) project_logits so prefill also
        # benefits from the lm_head compile.
        logits = self.model.project_logits(last_hidden)
        # logits: (1, batch, vocab)

        token_ids: list[int] = []
        for i, r in enumerate(requests):
            tok = sample_token(logits[:, i, :], r.sampling_params, r.output_ids)
            r.seq_len = r.num_input_tokens
            token_ids.append(tok)
        return token_ids

    @torch.inference_mode()
    def paged_decode(self, requests: list[Request]) -> list[int]:
        """Decode one token per running request via flash_attn_with_kvcache.

        Pre-condition: scheduler has ensured each request has ≥1 unused
        slot in its current last page (it tops up by allocating one more
        page just before this call when the previous page filled).
        """
        assert self.kv_pool is not None
        if not requests:
            return []

        # Try CUDA graph fast path first (extra credit).
        if self.cuda_graph_runner is not None:
            replayed = self.cuda_graph_runner.try_replay(requests)
            if replayed is not None:
                return replayed

        device = self.device
        B = len(requests)

        input_ids = torch.tensor(
            [[r.output_ids[-1]] for r in requests],
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.tensor(
            [[r.seq_len] for r in requests],
            dtype=torch.long,
            device=device,
        )
        # cache_seqlens is the KV length BEFORE this step (flash_attn_with_kvcache
        # writes the new token at this position and reads up to it+1).
        cache_seqlens = torch.tensor(
            [r.seq_len for r in requests],
            dtype=torch.int32,
            device=device,
        )

        max_pages = max(len(r.page_indices) for r in requests)
        block_table = torch.zeros((B, max_pages), dtype=torch.int32, device=device)
        for i, r in enumerate(requests):
            block_table[i, : len(r.page_indices)] = torch.tensor(
                r.page_indices, dtype=torch.int32, device=device
            )

        meta = PagedMeta(
            phase="decode",
            page_size=self.kv_pool.page_size,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
        )

        logits, _ = self.model(
            input_ids,
            position_ids,
            kv_caches=self.kv_pool.kv_caches,
            paged_meta=meta,
        )
        # logits: (B, 1, vocab)
        token_ids: list[int] = []
        for i, r in enumerate(requests):
            tok = sample_token(
                logits[i : i + 1, -1, :], r.sampling_params, r.output_ids
            )
            r.seq_len += 1
            token_ids.append(tok)
        return token_ids
