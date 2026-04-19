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
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes and caches cos/sin tables, indexed by position_ids at
    forward time.  The cache grows on-demand so we never allocate for the
    full 256K context upfront.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
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
            t = torch.arange(length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
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

    We transpose cos/sin to match x's layout: (batch, 1, seq_len, head_dim).
    """
    cos = cos.transpose(1, 2)  # → (batch, 1, seq_len, head_dim)
    sin = sin.transpose(1, 2)
    return x * cos + _rotate_half(x) * sin


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

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

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
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden:   (batch, seq_len, hidden_size)
            cos, sin: from RotaryEmbedding, broadcastable
            kv_cache: optional (cached_k, cached_v) each
                      (batch, num_kv_heads, cache_len, head_dim)

        Returns:
            output:       (batch, seq_len, hidden_size)
            new_kv_cache: (k, v) with updated cache
        """
        bsz, seq_len, _ = hidden.shape

        # Project Q, K, V and reshape to (batch, heads, seq_len, head_dim)
        q = self.q_proj(hidden).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

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

        # Scaled dot-product attention (uses Flash Attention when available)
        is_causal = kv_cache is None and seq_len > 1 and attention_mask is None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=is_causal)

        # Merge heads → project back
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out), new_kv


# ── MLP ─────────────────────────────────────────────────────────────────


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

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
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden, new_kv = self.self_attn(hidden, cos, sin, kv_cache, attention_mask)
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
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids:    (batch, seq_len)
            position_ids: (batch, seq_len)
            kv_caches:    list of per-layer (key, value) caches, or None

        Returns:
            hidden:         (batch, seq_len, hidden_size)
            new_kv_caches:  list of per-layer (key, value) with appended tokens
        """
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids)

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
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            logits:        (batch, seq_len, vocab_size)
            new_kv_caches: per-layer KV caches
        """
        hidden, new_kv_caches = self.model(input_ids, position_ids, kv_caches, attention_mask)
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

    # Cast model to target dtype on CPU first so load_state_dict doesn't
    # create float32 copies (important for large models on limited VRAM)
    model.to(dtype=dtype)

    # Merge all shards into one state dict
    state_dict: dict[str, torch.Tensor] = {}
    for f in st_files:
        shard = load_file(str(f), device="cpu")
        state_dict.update(shard)
        del shard

    # Cast to target dtype
    for key in state_dict:
        state_dict[key] = state_dict[key].to(dtype=dtype)

    # Drop keys the model does not expect
    # (e.g. rotary_emb.inv_freq if present in checkpoint)
    model_keys = set(model.state_dict().keys())
    extra = set(state_dict.keys()) - model_keys
    for key in extra:
        del state_dict[key]
    if extra:
        logger.info("Skipped %d unexpected checkpoint keys", len(extra))

    # Handle tied embeddings: if lm_head.weight is expected but not in checkpoint
    if "lm_head.weight" in model_keys and "lm_head.weight" not in state_dict:
        logger.info("Tying lm_head.weight to embed_tokens.weight")
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del state_dict  # free CPU memory before GPU transfer
    if missing:
        logger.warning("Missing keys after load: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys after load: %s", unexpected)

    # Move to target device
    model.to(device=device)
    logger.info("Weights loaded — %d parameters on %s (%s)", sum(p.numel() for p in model.parameters()), device, dtype)
