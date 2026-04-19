"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Design note:
  The current API is single-request (prefill / decode_step).  A natural
  first optimisation is to add batched versions that pad sequences and run
  multiple requests through a single forward pass.

  For tensor parallelism, the bare-bone nn.Linear layers in model.py can
  be sharded directly: Q/K/V/gate/up column-wise, O/down row-wise, with
  an all-reduce after the row-parallel matmul.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.model import CausalLM, ModelConfig, load_weights
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


class Engine:
    """Bare-bone model wrapper for single-request prefill and decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.device = device
        self.dtype = dtype

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

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=request.kv_cache)
        request.kv_cache = kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids
    
    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]):
        '''
        - Stacks the last generated token from each request → `(batch, 1)`
        - Pads per-request KV caches to the max cache length in the batch
        - Builds an attention mask to ignore padding positions
        - Runs one batched model forward
        - Extracts per-request KV (actual portion + new token) and samples
        '''
        batch = len(requests)

        # stack last generated token
        last_tokens = torch.tensor(
            [[req.output_ids[-1]] for req in requests], dtype=torch.long, device=self.device
        )

        # get max cache length
        cache_lens = [req.kv_cache[0][0].shape[2] for req in requests]
        max_cache_len= max(cache_lens)

        # position_ids
        position_ids = torch.tensor(
            [[l] for l in cache_lens], dtype=torch.long, device=self.device
        )

        # pad per-request KV caches to max_cache_len
        num_layers = len(requests[0].kv_cache)
        batched_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            ks, vs = [], []
            for i, req in enumerate(requests):
                k, v = req.kv_cache[layer_idx]  # (batch_size, kv_heads, max_cache_len, head_dim)
                pad = max_cache_len - cache_lens[i]
                k = torch.nn.functional.pad(k, (0, 0, 0, pad))
                v = torch.nn.functional.pad(v, (0, 0, 0, pad))
                ks.append(k)
                vs.append(v)
            batched_kv.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))

        # attention mask
        # (batch, 1, 1, max_cache_len + 1)
        # additive float mask: 0.0 = attend, -inf = ignore padding
        float_mask = torch.full(
            (batch, 1, 1, max_cache_len + 1), float("-inf"),
            dtype=self.dtype, device=self.device
        )
        for i, L in enumerate(cache_lens):
            float_mask[i, 0, 0, :L] = 0.0          # valid cached tokens
            float_mask[i, 0, 0, max_cache_len] = 0.0  # new token position (always last)

        # run one batched forward pass
        logits, new_kv_caches = self.model(
            last_tokens, position_ids, kv_caches=batched_kv, attention_mask=float_mask
        )

        # extract per-request KV (actual portion + new token) and sample
        # (batch, kv_heads, max_cache_len+1, head_dim)
        token_ids: list[int] = []
        for i, (req, L) in enumerate(zip(requests, cache_lens)):
            req.kv_cache = [
                (
                    torch.cat([new_kv_caches[l][0][i:i+1, :, :L, :],
                               new_kv_caches[l][0][i:i+1, :, -1:, :]], dim=2), # k tensor
                    torch.cat([new_kv_caches[l][1][i:i+1, :, :L, :],
                               new_kv_caches[l][1][i:i+1, :, -1:, :]], dim=2), # v tensor
                )
                for l in range(num_layers)
            ]
            token_ids.append(
                sample_token(logits[i:i+1, -1, :], req.sampling_params, req.output_ids)
            )

        return token_ids


