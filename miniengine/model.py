"""
Bare-bone Qwen3 transformer in pure PyTorch.

No HuggingFace model classes — just nn.Module, nn.Linear, and manual
attention with KV cache.  Weight names match the HuggingFace checkpoint
so we can load safetensors directly via load_state_dict().

Architecture (Qwen3-4B as reference):
    Embedding(151936, 2560)
    36 x TransformerBlock:
        RMSNorm → Attention(GQA + QK-Norm + RoPE) → RMSNorm → SwiGLU MLP
    RMSNorm
    LM Head (tied with embedding)

Paged-attention extension (Milestone 2):
    Each Attention.forward optionally takes a `paged_ctx` PagedAttnCtx
    that points at the global KV pool (per-layer K/V slabs) plus per-
    request page tables. When supplied, attention writes new K/V into
    the pool by slot, and reads the per-request KV by gathering pages.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Config ──────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Model architecture config, loaded from HuggingFace config.json."""

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128  # explicit, NOT hidden_size // num_heads
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = True

    @classmethod
    def from_pretrained(cls, model_path: str) -> ModelConfig:
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=getattr(hf, "rope_theta", 10000.0),
            max_position_embeddings=getattr(hf, "max_position_embeddings", 4096),
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
        )


# ── Building blocks ─────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Variance in fp32 — bf16 mean-of-squares loses too much precision.
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes and caches cos/sin tables, indexed by position_ids at
    forward time.  The cache grows on-demand so we never allocate for the
    full 256K context upfront.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len: int = 0

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids: (batch, seq_len) integer positions.

        Returns:
            cos, sin each of shape (batch, 1, seq_len, head_dim) —
            broadcastable over the head dimension.
        """
        max_pos = int(position_ids.max().item()) + 1

        if self._cos is None or max_pos > self._cached_len:
            length = max(max_pos, self._cached_len * 2, 256)
            t = torch.arange(
                length, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.outer(t, self.inv_freq)  # (length, head_dim/2)
            emb = torch.cat([freqs, freqs], dim=-1)  # (length, head_dim)
            self._cos = emb.cos()
            self._sin = emb.sin()
            self._cached_len = length

        # Index into cache: (batch, seq_len, head_dim) → add head dim
        cos = self._cos[position_ids].unsqueeze(2)  # (batch, seq_len, 1, head_dim)
        sin = self._sin[position_ids].unsqueeze(2)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply RoPE to x.

    x:   (batch, num_heads, seq_len, head_dim)
    cos: (batch, seq_len, 1, head_dim)  — broadcast over heads
    sin: same shape as cos
    """
    # Cast cos/sin to x.dtype — fp32 cos/sin would silently promote q/k.
    cos = cos.transpose(1, 2).to(x.dtype)
    sin = sin.transpose(1, 2).to(x.dtype)
    return x * cos + _rotate_half(x) * sin


# ── Paged attention support ─────────────────────────────────────────────


@dataclass
class PagedAttnCtx:
    """Per-step context for paged attention.

    Fields are layer-agnostic (page_indices/lengths/slot_mapping describe
    where each token lives in the pool); the layer index selects the
    right slab from `kv_caches`.

    Attributes
    ----------
    kv_caches:
        The pool's stable per-layer (K, V) tensors.
        Shape per slab: (num_pages, page_size, num_kv_heads, head_dim).
    page_size:
        Tokens per page (matches the pool).
    page_indices:
        Padded int32 tensor of shape (batch, max_num_pages). For
        request i its first lengths[i]/page_size+1 entries are valid.
    lengths_after:
        int32 tensor (batch,) — KV length AFTER writing the new tokens
        from this forward (= old length + new tokens for that request).
    slot_mapping:
        int64 tensor (total_new_tokens,) — flat slot indices into a
        per-layer K (or V) viewed as (num_pages * page_size, kv_heads,
        head_dim). One entry per *new* token in execution order.
        Used to write fresh K/V into the pool.
    cu_seqlens_q:
        Optional int32 tensor (batch + 1,) cumulative query lengths.
        Provided by packed prefill (batch dim is collapsed). None for
        decode (single-token-per-request, packed in sequence dim).
    is_decode:
        True if every request contributes exactly one new query token.
        Decode can use a tighter attention path (no causal mask within
        the new tokens).
    """

    kv_caches: list[tuple[torch.Tensor, torch.Tensor]]
    page_size: int
    page_indices: torch.Tensor
    lengths_after: torch.Tensor
    slot_mapping: torch.Tensor
    cu_seqlens_q: torch.Tensor | None = None
    is_decode: bool = False


def _paged_attn_forward(
    q: torch.Tensor,                # (1, num_heads, total_q, head_dim) packed
    k_new: torch.Tensor,            # (1, num_kv_heads, total_q, head_dim) packed
    v_new: torch.Tensor,            # (1, num_kv_heads, total_q, head_dim) packed
    paged_ctx: PagedAttnCtx,
    layer_idx: int,
    num_kv_groups: int,
) -> torch.Tensor:
    """Paged attention: write new K/V into pool, then attend per request.

    Returns out: (1, num_heads, total_q, head_dim) packed.

    Two internal paths share the same write step but differ in how they read:

    Decode (is_decode=True):
        Fully batched, fixed-shape — no Python branching on tensor values.
        All requests are processed in a single batched SDPA call with a
        float mask derived from lengths_after. This path is CUDA-graph-
        compatible because no .tolist() or .item() is called.

    Prefill (is_decode=False):
        Per-request Python loop with .tolist() to handle variable q_lens
        and causal masking. Shapes vary per request, so this path is NOT
        CUDA-graph-compatible. Prefill is never captured in graphs.
    """
    K_pool, V_pool = paged_ctx.kv_caches[layer_idx]
    num_pages, page_size, num_kv_heads, head_dim = K_pool.shape

    # Flat views into the per-layer slab for slot-indexed scatter-write.
    K_flat = K_pool.view(num_pages * page_size, num_kv_heads, head_dim)
    V_flat = V_pool.view(num_pages * page_size, num_kv_heads, head_dim)

    total_q = k_new.shape[2]
    # (total_q, kv_heads, head_dim) for index_copy_
    k_pack = k_new.squeeze(0).transpose(0, 1).contiguous()
    v_pack = v_new.squeeze(0).transpose(0, 1).contiguous()
    K_flat.index_copy_(0, paged_ctx.slot_mapping, k_pack)
    V_flat.index_copy_(0, paged_ctx.slot_mapping, v_pack)

    batch = paged_ctx.page_indices.shape[0]
    num_heads = q.shape[1]
    # q_pack: (total_q, num_heads, head_dim)
    q_pack = q.squeeze(0).transpose(0, 1).contiguous()

    if paged_ctx.is_decode:
        # ── Fully batched, fixed-shape decode path ──────────────────────
        # For decode total_q == batch (one new token per request).
        # All tensor shapes are fixed at capture time → CUDA-graph-safe.
        page_indices = paged_ctx.page_indices   # (batch, max_pages) int32
        lengths_after = paged_ctx.lengths_after  # (batch,) int32
        max_num_pages = page_indices.shape[1]
        max_kv_len = max_num_pages * page_size

        # Gather every page of every request in one index op.
        # Padding entries in page_indices (zeros) fetch page 0 of the pool;
        # those positions are masked to -inf in attn_mask below.
        flat_idx = page_indices.view(-1).long()          # (batch * max_pages,)
        k_all = K_pool[flat_idx].reshape(batch, max_kv_len, num_kv_heads, head_dim)
        v_all = V_pool[flat_idx].reshape(batch, max_kv_len, num_kv_heads, head_dim)
        # → (batch, kv_heads, max_kv_len, head_dim)
        k_all = k_all.transpose(1, 2)
        v_all = v_all.transpose(1, 2)

        if num_kv_groups > 1:
            k_all = (
                k_all[:, :, None, :, :]
                .expand(-1, -1, num_kv_groups, -1, -1)
                .reshape(batch, num_heads, max_kv_len, head_dim)
            )
            v_all = (
                v_all[:, :, None, :, :]
                .expand(-1, -1, num_kv_groups, -1, -1)
                .reshape(batch, num_heads, max_kv_len, head_dim)
            )

        # Float attention mask: 0 for valid KV positions, -inf for padding.
        # Computed entirely with tensor ops — no .tolist() / .item().
        # positions: (1, max_kv_len);  lengths_after: (batch, 1) after unsqueeze
        positions = torch.arange(max_kv_len, device=q.device).unsqueeze(0)
        invalid = positions >= lengths_after.to(torch.int64).unsqueeze(1)  # (batch, max_kv_len)
        attn_mask = torch.zeros(
            batch, 1, 1, max_kv_len, device=q.device, dtype=q.dtype
        ).masked_fill(invalid.unsqueeze(1).unsqueeze(2), float("-inf"))

        # q_pack: (batch, num_heads, head_dim) → (batch, num_heads, 1, head_dim)
        q_batch = q_pack.unsqueeze(2)
        out_i = F.scaled_dot_product_attention(q_batch, k_all, v_all, attn_mask=attn_mask)
        # (batch, num_heads, 1, head_dim) → (batch, num_heads, head_dim)
        out_pack = out_i.squeeze(2)
    else:
        # ── Per-request loop path (prefill only) ────────────────────────
        if paged_ctx.cu_seqlens_q is not None:
            cu_q = paged_ctx.cu_seqlens_q
        else:
            cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32)

        lengths_after_list = paged_ctx.lengths_after.tolist()
        cu_q_list = cu_q.tolist()
        page_indices = paged_ctx.page_indices  # (batch, max_pages)

        out_pack = torch.zeros(total_q, num_heads, head_dim, device=q.device, dtype=q.dtype)

        for i in range(batch):
            q_start = cu_q_list[i]
            q_end = cu_q_list[i + 1]
            q_len = q_end - q_start
            if q_len == 0:
                continue

            kv_len = lengths_after_list[i]
            if kv_len == 0:
                continue

            n_pages = (kv_len + page_size - 1) // page_size
            idx = page_indices[i, :n_pages].long()
            gathered_k = K_pool.index_select(0, idx)
            gathered_v = V_pool.index_select(0, idx)
            k_seq = gathered_k.reshape(n_pages * page_size, num_kv_heads, head_dim)[:kv_len]
            v_seq = gathered_v.reshape(n_pages * page_size, num_kv_heads, head_dim)[:kv_len]

            # (1, kv_heads, kv_len, head_dim)
            k_seq = k_seq.transpose(0, 1).unsqueeze(0)
            v_seq = v_seq.transpose(0, 1).unsqueeze(0)

            if num_kv_groups > 1:
                k_seq = (
                    k_seq[:, :, None, :, :]
                    .expand(-1, -1, num_kv_groups, -1, -1)
                    .reshape(1, num_kv_heads * num_kv_groups, kv_len, head_dim)
                )
                v_seq = (
                    v_seq[:, :, None, :, :]
                    .expand(-1, -1, num_kv_groups, -1, -1)
                    .reshape(1, num_kv_heads * num_kv_groups, kv_len, head_dim)
                )

            # (1, num_heads, q_len, head_dim)
            q_i = q_pack[q_start:q_end].transpose(0, 1).unsqueeze(0)

            if q_len == 1:
                out_i = F.scaled_dot_product_attention(q_i, k_seq, v_seq, is_causal=False)
            else:
                # Causal mask: row r (absolute pos kv_len-q_len+r) attends to col c ≤ row.
                offset = kv_len - q_len
                row = torch.arange(q_len, device=q.device).unsqueeze(1) + offset
                col = torch.arange(kv_len, device=q.device).unsqueeze(0)
                mask = torch.where(
                    col <= row,
                    torch.zeros((), dtype=q.dtype, device=q.device),
                    torch.full((), float("-inf"), dtype=q.dtype, device=q.device),
                )
                out_i = F.scaled_dot_product_attention(q_i, k_seq, v_seq, attn_mask=mask)

            # (1, num_heads, q_len, hd) → (q_len, num_heads, hd)
            out_pack[q_start:q_end] = out_i.squeeze(0).transpose(0, 1)

    # (total_q, num_heads, head_dim) → (1, num_heads, total_q, head_dim)
    return out_pack.transpose(0, 1).unsqueeze(0).contiguous()


# ── Attention ───────────────────────────────────────────────────────────


class Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA), QK-Norm,
    and Rotary Position Embeddings.

    Q projects to  num_attention_heads  × head_dim
    K projects to  num_key_value_heads  × head_dim
    V projects to  num_key_value_heads  × head_dim
    O projects back to hidden_size
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        # Qwen3: RMSNorm on Q and K after projection (per-head)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_ctx: PagedAttnCtx | None = None,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            hidden:         (batch, seq_len, hidden_size). For paged mode
                            batch=1 and seq_len = total packed tokens.
            cos, sin:       from RotaryEmbedding, broadcastable
            kv_cache:       optional (cached_k, cached_v) for non-paged
                            mode; (batch, num_kv_heads, cache_len, head_dim)
            attention_mask: optional float mask (batch, 1, q_len, kv_len)
                            for batched decode with padded KV; 0 = attend,
                            -inf = ignore.
            paged_ctx:      if set, run paged attention through the pool
                            instead of using the local kv_cache argument.
            layer_idx:      this attention layer's index, used to pick the
                            right slab from paged_ctx.kv_caches.

        Returns:
            output:       (batch, seq_len, hidden_size)
            new_kv_cache: (k, v) with updated cache, or None for paged
                          mode (KV is stored back into the pool in place).
        """
        bsz, seq_len, _ = hidden.shape

        # Project Q, K, V and reshape to (batch, heads, seq_len, head_dim)
        q = (
            self.q_proj(hidden)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # ── Paged path ──────────────────────────────────────────────────
        if paged_ctx is not None:
            out = _paged_attn_forward(
                q, k, v, paged_ctx, layer_idx, self.num_kv_groups
            )
            out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
            return self.o_proj(out), None

        # ── Legacy contiguous-cache path ────────────────────────────────
        # Append to KV cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(bsz, self.num_heads, -1, self.head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(bsz, self.num_heads, -1, self.head_dim)

        # Batched decode passes an explicit float mask; otherwise fall
        # back to the is_causal kernel path.
        if attention_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        else:
            is_causal = kv_cache is None and seq_len > 1
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Merge heads → project back
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out), new_kv


# ── MLP ─────────────────────────────────────────────────────────────────


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer block ──────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Pre-norm transformer layer: LN → Attn → residual → LN → MLP → residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_ctx: PagedAttnCtx | None = None,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden, new_kv = self.self_attn(
            hidden, cos, sin, kv_cache, attention_mask, paged_ctx, layer_idx
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = residual + hidden

        return hidden, new_kv


# ── Full model ──────────────────────────────────────────────────────────


class TransformerModel(nn.Module):
    """The core transformer: embedding → N layers → final norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, theta=config.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_ctx: PagedAttnCtx | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Args:
            input_ids:      (batch, seq_len)
            position_ids:   (batch, seq_len)
            kv_caches:      list of per-layer (key, value) caches, or None
            attention_mask: optional float mask for batched-decode SDPA
            paged_ctx:      if set, use the paged-attention pool instead
                            of contiguous per-request caches.

        Returns:
            hidden:         (batch, seq_len, hidden_size)
            new_kv_caches:  list of per-layer (key, value) with appended
                            tokens, or None when paged_ctx is used
                            (KV is stored in the pool).
        """
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids)

        if paged_ctx is not None:
            for i, layer in enumerate(self.layers):
                hidden, _ = layer(
                    hidden, cos, sin, None, None, paged_ctx, i,
                )
            hidden = self.norm(hidden)
            return hidden, None

        new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden, new_kv = layer(hidden, cos, sin, kv, attention_mask)
            new_kv_caches.append(new_kv)

        hidden = self.norm(hidden)
        return hidden, new_kv_caches


class CausalLM(nn.Module):
    """
    Complete causal language model: transformer + LM head.

    The LM head may be tied with the embedding (Qwen3-4B) or separate.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = TransformerModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_ctx: PagedAttnCtx | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Returns:
            logits:        (batch, seq_len, vocab_size)
            new_kv_caches: per-layer KV caches, or None for paged mode.
        """
        hidden, new_kv_caches = self.model(
            input_ids, position_ids, kv_caches, attention_mask, paged_ctx
        )
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden)
        return logits, new_kv_caches


# ── Weight loading ──────────────────────────────────────────────────────


def load_weights(
    model: CausalLM,
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """
    Load weights from HuggingFace safetensors into the model.

    Handles both single-file and sharded checkpoints.  Weight names in the
    checkpoint match our module hierarchy exactly (by design), so we can
    use load_state_dict() directly.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    logger.info("Downloading / locating model files for %s …", model_path)
    local_path = Path(
        snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )

    # Gather all safetensor shard files
    st_files = sorted(local_path.glob("model*.safetensors"))
    if not st_files:
        # Fallback: some repos use a single "model.safetensors"
        st_files = sorted(local_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {local_path}")

    logger.info("Loading %d safetensors shard(s) …", len(st_files))

    # Load shards straight to the target device — avoids a full CPU copy.
    state_dict: dict[str, torch.Tensor] = {}
    for f in st_files:
        for key, tensor in load_file(str(f), device=device).items():
            state_dict[key] = tensor.to(dtype=dtype)

    # Drop checkpoint keys the model doesn't expect.
    model_keys = set(model.state_dict().keys())
    extra = set(state_dict.keys()) - model_keys
    for key in extra:
        del state_dict[key]
    if extra:
        logger.info("Skipped %d unexpected checkpoint keys", len(extra))

    if "lm_head.weight" in model_keys and "lm_head.weight" not in state_dict:
        logger.info("Tying lm_head.weight to embed_tokens.weight")
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    # assign=True: replace meta tensors in-place rather than copy_ into them.
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    if missing:
        logger.warning("Missing keys after load: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys after load: %s", unexpected)

    # RoPE inv_freq is a non-persistent buffer (not in checkpoint), so it's
    # still on the meta device after assign=True — materialize it now.
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.inv_freq = 1.0 / (
                module.theta
                ** (
                    torch.arange(
                        0, module.head_dim, 2, device=device, dtype=torch.float32
                    )
                    / module.head_dim
                )
            )
    logger.info(
        "Weights loaded — %d parameters on %s (%s)",
        sum(p.numel() for p in model.parameters()),
        device,
        dtype,
    )
