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
    Each Attention.forward optionally takes a `paged_meta` PagedMeta
    object that describes how the current packed batch maps onto the
    global KV pool. Attention then dispatches to flash-attn:
      - prefill: `flash_attn_varlen_func` over packed prompts, with
        new K/V scattered into the pool by `slot_mapping`.
      - decode : `flash_attn_with_kvcache` reads K/V via `block_table`
        and `cache_seqlens` and writes the new token in place.

Block layout note (Milestone 2):
    `_add_norm_pre` / `_add_norm_post` fuse the residual-add with the
    next RMSNorm. This lets `torch.compile` form a single Inductor
    region per (residual+norm) pair, fewer kernel launches per layer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
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
    def extend_to(self, length: int) -> None:
        """Pre-build the cos/sin cache up to `length` positions.

        Call this at engine startup so that `forward` never trips the
        lazy-grow branch (which calls `.item()` and graph-breaks under
        torch.compile / CUDA-graph capture).
        """
        if self._cos is not None and length <= self._cached_len:
            return
        t = torch.arange(
            length, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos()
        self._sin = emb.sin()
        self._cached_len = length

    # Back-compat alias; older code called this method `extend`.
    extend = extend_to

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids: (batch, seq_len) integer positions.

        Returns:
            cos, sin each of shape (batch, 1, seq_len, head_dim) —
            broadcastable over the head dimension.

        Note: callers must ensure the cache is pre-extended (via
        `extend_to`) to cover the maximum position they'll query. The
        only fall-back here is the very first call (cold cache); after
        that, this method is sync-free, which lets it be captured by
        CUDA graphs and torch.compile.
        """
        if self._cos is None:
            # Cold path — first call ever. Triggers a CPU sync, but
            # only happens once at startup (engine warm-up calls
            # extend_to() before any benchmark work).
            self.extend_to(
                max(int(position_ids.max().item()) + 1, 256)
            )

        # Hot path — assume cache is large enough (engine pre-extends
        # at startup). NO .item() call here so CUDA graph capture works.
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


# ── Paged attention metadata ────────────────────────────────────────────


@dataclass
class PagedMeta:
    """Per-step metadata for paged attention.

    Prefill uses cu_seqlens_* + slot_mapping (varlen flash-attn).
    Decode uses block_table + cache_seqlens (paged-kv flash-attn).

    Attributes
    ----------
    phase: "prefill" or "decode".
    page_size: tokens per page (matches the pool).

    Prefill fields:
        cu_seqlens_q / cu_seqlens_k:
            int32 (batch+1,) cumulative seqlens for the packed prompts.
        max_seqlen_q / max_seqlen_k: int — longest prompt in the batch.
        slot_mapping:
            int64 (total_prompt_tokens,) flat slot indices into a per-layer
            K (or V) view of shape (num_pages * page_size, kv_heads, head_dim).

    Decode fields:
        block_table:
            int32 (batch, max_pages) per-request page tables.
        cache_seqlens:
            int32 (batch,) current KV length per request BEFORE writing
            this step's new token; flash_attn_with_kvcache writes at
            this position and reads up to it+1.
    """

    phase: str
    page_size: int

    # Prefill
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None

    # Decode
    block_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None


# Back-compat alias: some callers (e.g. cuda_graph_runner) imported
# `PagedAttnCtx`. Keep the symbol so existing imports don't break.
PagedAttnCtx = PagedMeta


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

    def _proj_qkv_with_rope(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project + QK-norm + RoPE.

        Returns q, k as (B, H, S, D) for SDPA / SDPA-callers; v as
        (B, S, H_kv, D) for the flash-attn paths (the contiguous-cache
        path transposes v itself).
        """
        bsz, seq_len, _ = hidden.shape
        q = self.q_proj(hidden).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k, v

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_meta: PagedMeta | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """If paged_meta is set, attention runs through the KV pool and
        returns (out, None); otherwise SDPA on per-request kv_cache.

        Args:
            hidden: (batch, seq_len, hidden_size)
            cos, sin: from RotaryEmbedding
            kv_cache:
                Contiguous-cache mode: optional (cached_k, cached_v).
                Paged mode: pass the pool's per-LAYER (k_pool, v_pool)
                slab tuple (the caller in TransformerModel selects layer i).
            attention_mask:
                Float mask (batch, 1, q_len, kv_len) for batched-decode
                padded-KV path. 0 = attend, -inf = ignore.
            paged_meta:
                If set, dispatch to the flash-attn paged path and ignore
                `attention_mask`.
        """
        bsz, seq_len, _ = hidden.shape
        q, k, v = self._proj_qkv_with_rope(hidden, cos, sin)

        # ── Paged path (flash-attn) ─────────────────────────────────────
        if paged_meta is not None:
            return self._forward_paged(q, k, v, kv_cache, paged_meta)

        # ── Legacy contiguous-cache path (baseline + batched modes) ─────
        # v from _proj_qkv_with_rope is (B, S, H_kv, D); the contiguous-
        # cache path needs (B, H_kv, S, D).
        v = v.transpose(1, 2)

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

        if attention_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        else:
            is_causal = kv_cache is None and seq_len > 1
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out), new_kv

    def _forward_paged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        meta: PagedMeta,
    ) -> tuple[torch.Tensor, None]:
        """Paged attention via flash-attn (varlen prefill / paged-kv decode).

        Args:
            q, k: (B, H, S, D) from `_proj_qkv_with_rope`.
            v: (B, S, H_kv, D).
            kv_cache: this layer's (k_pool, v_pool) slabs from the pool.
                Shape: (num_pages, page_size, num_kv_heads, head_dim).
        """
        # flash-attn wants (B, S, H, D); undo the transpose RoPE produced.
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.contiguous()
        bsz, seq_len = q.shape[0], q.shape[1]
        k_pool, v_pool = kv_cache

        if meta.phase == "prefill":
            from flash_attn import flash_attn_varlen_func

            # Prompts are packed as B=1 → squeeze to (sum_q, H, D).
            q_p, k_p, v_p = q.squeeze(0), k.squeeze(0), v.squeeze(0)

            # Scatter K/V into the pool by absolute slot index.
            k_pool.view(-1, self.num_kv_heads, self.head_dim)[meta.slot_mapping] = k_p
            v_pool.view(-1, self.num_kv_heads, self.head_dim)[meta.slot_mapping] = v_p

            out = flash_attn_varlen_func(
                q_p,
                k_p,
                v_p,
                cu_seqlens_q=meta.cu_seqlens_q,
                cu_seqlens_k=meta.cu_seqlens_k,
                max_seqlen_q=meta.max_seqlen_q,
                max_seqlen_k=meta.max_seqlen_k,
                causal=True,
            )
            out = out.unsqueeze(0)
        else:
            from flash_attn import flash_attn_with_kvcache

            # Kernel writes (k, v) into the pool at the slots specified
            # by block_table + cache_seqlens AND attends. One call,
            # zero per-layer Python.
            out = flash_attn_with_kvcache(
                q,
                k_pool,
                v_pool,
                k=k,
                v=v,
                cache_seqlens=meta.cache_seqlens,
                block_table=meta.block_table,
                causal=True,
            )

        out = out.contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out), None


# ── MLP ─────────────────────────────────────────────────────────────────


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)).

    `gate_proj` and `up_proj` are fused into a single matmul (weight =
    cat([gate_w, up_w], dim=0)) — saves one cuBLAS launch and one read
    of x per layer. `load_weights` rewrites the checkpoint keys to match.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# ── Transformer block ──────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Pre-norm transformer layer: LN → Attn → residual → LN → MLP → residual.

    The residual-add is fused with the FOLLOWING RMSNorm in
    `_add_norm_pre` / `_add_norm_post` so torch.compile can pack each
    pair into a single Inductor region.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _add_norm_pre(
        self, residual: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused (residual + x) → input_layernorm.

        Returns (normed, new_residual).
        """
        residual = residual + x
        return self.input_layernorm(residual), residual

    def _add_norm_post(
        self, residual: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused (residual + x) → post_attention_layernorm."""
        residual = residual + x
        return self.post_attention_layernorm(residual), residual

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        residual: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_meta: PagedMeta | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Block forward with explicit residual passing.

        The first block in a stack receives residual=None and seeds the
        residual stream from the embedding output. Subsequent blocks
        accept (hidden, residual) and return (mlp_out, residual_after_attn).
        The caller applies the final `residual + hidden` after the last
        block (see TransformerModel.forward).
        """
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self._add_norm_pre(residual, hidden)

        hidden, new_kv = self.self_attn(
            hidden, cos, sin, kv_cache, attention_mask, paged_meta
        )
        hidden, residual = self._add_norm_post(residual, hidden)
        hidden = self.mlp(hidden)
        return hidden, residual, new_kv


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
        paged_meta: PagedMeta | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Args:
            input_ids:     (batch, seq_len)
            position_ids:  (batch, seq_len)
            kv_caches:
                Contiguous-cache mode: list of per-layer (key, value) caches,
                or None for prefill.
                Paged mode: the pool's per-layer (k_pool, v_pool) slab list.
            attention_mask: optional float mask for batched-decode SDPA
            paged_meta:    if set, use the paged-attention path.

        Returns:
            hidden:        (batch, seq_len, hidden_size)
            new_kv_caches: list of per-layer (key, value) caches with the
                            new tokens appended, or None in paged mode
                            (KV is updated in place inside the pool).
        """
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids)

        if paged_meta is not None:
            residual: torch.Tensor | None = None
            for i, layer in enumerate(self.layers):
                hidden, residual, _ = layer(
                    hidden,
                    cos,
                    sin,
                    residual,
                    kv_caches[i],  # this layer's pool slab
                    None,
                    paged_meta,
                )
            hidden = residual + hidden
            return self.norm(hidden), None

        new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        residual = None
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden, residual, new_kv = layer(
                hidden, cos, sin, residual, kv, attention_mask
            )
            new_kv_caches.append(new_kv)
        hidden = residual + hidden
        return self.norm(hidden), new_kv_caches


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

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Final hidden -> vocab logits projection.

        Pulled out so the engine can wrap it in torch.compile separately
        (vocab=152k matmul is a hot kernel; compiling it lets Inductor
        epilogue-fuse with downstream sampling preprocessing if any).
        """
        if self.config.tie_word_embeddings:
            return F.linear(hidden, self.model.embed_tokens.weight)
        return self.lm_head(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_meta: PagedMeta | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Returns:
            logits:        (batch, seq_len, vocab_size)
            new_kv_caches: per-layer KV caches, or None for paged mode.
        """
        hidden, new_kv_caches = self.model(
            input_ids, position_ids, kv_caches, attention_mask, paged_meta
        )
        logits = self.project_logits(hidden)
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

    On the way in we also fuse `mlp.gate_proj.weight` + `mlp.up_proj.weight`
    into the new `mlp.gate_up_proj.weight` (gate first, to match
    `MLP.forward`'s `chunk(2)` order).
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

    # ── Fuse mlp.gate_proj + mlp.up_proj → mlp.gate_up_proj ────────────
    # Order is gate first, up second, matching MLP.forward's chunk(2).
    fused = 0
    gate_keys = [k for k in list(state_dict.keys())
                 if k.endswith(".mlp.gate_proj.weight")]
    for gk in gate_keys:
        uk = gk.replace(".gate_proj.", ".up_proj.")
        if uk not in state_dict:
            continue
        new_key = gk.replace(".gate_proj.", ".gate_up_proj.")
        state_dict[new_key] = torch.cat([state_dict[gk], state_dict[uk]], dim=0)
        del state_dict[gk]
        del state_dict[uk]
        fused += 1
    if fused:
        logger.info("Fused gate_proj+up_proj → gate_up_proj on %d MLP blocks", fused)

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
