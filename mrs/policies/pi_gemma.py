"""The Gemma variant pi0.5 uses: adaptive RMSNorm plus gated residuals.

Two Gemma stacks run side by side over one shared attention operation — a 2B
"prefix" expert that sees images and language, and a 300M "action" expert that
sees the noisy action chunk. Each stack keeps its own weights; at every layer
their queries/keys/values are concatenated so the action tokens can attend to
the prefix.

The action expert's norms are *adaptive*: instead of a learned gain vector they
predict (scale, shift, gate) from the flow-matching timestep embedding, which is
how the timestep conditions the whole stack.

Module and parameter names deliberately match the published pi0.5 checkpoints so
their `state_dict` loads without remapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

# Imported at module load, not lazily inside `SiglipVisionTower.__init__`:
# transformers pulls in torch._dynamo, and triggering that import while a
# `torch.device("meta")` context is active raises a circular-import error.
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel


@dataclass(frozen=True)
class GemmaVariantConfig:
    """Shape of one Gemma stack."""

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0


GEMMA_VARIANTS: dict[str, GemmaVariantConfig] = {
    "gemma_300m": GemmaVariantConfig(
        width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256
    ),
    "gemma_2b": GemmaVariantConfig(
        width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256
    ),
}


def get_gemma_config(variant: str) -> GemmaVariantConfig:
    if variant not in GEMMA_VARIANTS:
        raise ValueError(f"Unknown Gemma variant {variant!r}. Known: {sorted(GEMMA_VARIANTS)}")
    return GEMMA_VARIANTS[variant]


# ---------------------------------------------------------------------------
# Norms and residuals
# ---------------------------------------------------------------------------


def gated_residual(x: Tensor, y: Tensor, gate: Tensor | None) -> Tensor:
    """`x + y`, or `x + y * gate` when the adaptive norm produced a gate."""
    return x + y if gate is None else x + y * gate


class PiGemmaRMSNorm(nn.Module):
    """RMSNorm that is either standard (learned gain) or adaptive (predicted).

    In adaptive mode a linear layer maps the conditioning vector to
    `(scale, shift, gate)`; the first two modulate the normalized activations
    and the third is handed back for the gated residual.

    Returns `(output, gate)`, where `gate` is None in the standard case.
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x: Tensor) -> Tensor:
        # Variance in float32 regardless of activation dtype, as in Gemma.
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        dtype = x.dtype
        normed = self._norm(x)

        if cond is None or self.dense is None:
            # Gemma stores the gain as (w - 1), hence the 1.0 + here.
            return (normed * (1.0 + self.weight.float())).type_as(x), None

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected conditioning dim {self.cond_dim}, got {cond.shape[-1]}")

        modulation = self.dense(cond.to(self.dense.weight.dtype))
        if x.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)

    def extra_repr(self) -> str:
        if self.dense is not None:
            return f"dim={self.dim}, eps={self.eps}, adaptive=True, cond_dim={self.cond_dim}"
        return f"dim={self.dim}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Rotary embeddings
# ---------------------------------------------------------------------------


class GemmaRotaryEmbedding(nn.Module):
    """Standard RoPE over the full head dimension."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        # (batch, seq) -> (batch, seq, head_dim)
        freqs = position_ids[..., None].float() * self.inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to `(batch, heads, seq, head_dim)` tensors."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ---------------------------------------------------------------------------
# Attention / MLP building blocks
# ---------------------------------------------------------------------------


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand `n_kv_heads` up to `n_heads` for grouped-query attention."""
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class GemmaAttentionProjections(nn.Module):
    """The q/k/v/o projections of one Gemma stack (no attention math of its own).

    The math lives in :func:`joint_attention` because pi0.5 computes attention
    once over the concatenation of both stacks' tokens.
    """

    def __init__(self, config: GemmaVariantConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.scaling = config.head_dim**-0.5

        self.q_proj = nn.Linear(config.width, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.width, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.width, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.width, bias=False)

    def project_qkv(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden_states = hidden_states.to(self.q_proj.weight.dtype)
        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v


class GemmaMLP(nn.Module):
    """Gated feed-forward block with the tanh-approximated GELU Gemma uses."""

    def __init__(self, config: GemmaVariantConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.width, config.mlp_dim, bias=False)
        self.up_proj = nn.Linear(config.width, config.mlp_dim, bias=False)
        self.down_proj = nn.Linear(config.mlp_dim, config.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.gate_proj.weight.dtype)
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class PiGemmaDecoderLayer(nn.Module):
    """One decoder layer: pre-norm attention, pre-norm MLP, both gated."""

    def __init__(self, config: GemmaVariantConfig, cond_dim: int | None = None):
        super().__init__()
        self.self_attn = GemmaAttentionProjections(config)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = PiGemmaRMSNorm(config.width, eps=config.rms_norm_eps, cond_dim=cond_dim)
        self.post_attention_layernorm = PiGemmaRMSNorm(
            config.width, eps=config.rms_norm_eps, cond_dim=cond_dim
        )


class PiGemmaStack(nn.Module):
    """A stack of decoder layers plus the final norm, without embeddings."""

    def __init__(self, config: GemmaVariantConfig, cond_dim: int | None = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [PiGemmaDecoderLayer(config, cond_dim=cond_dim) for _ in range(config.depth)]
        )
        self.norm = PiGemmaRMSNorm(config.width, eps=config.rms_norm_eps, cond_dim=cond_dim)


# ---------------------------------------------------------------------------
# The shared attention operation
# ---------------------------------------------------------------------------


def joint_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor,
    scaling: float,
) -> Tensor:
    """Softmax attention over `(batch, heads, seq, head_dim)` with an additive mask.

    `attention_mask` is `(batch, 1, q_len, kv_len)` holding 0 where attention is
    allowed and a large negative value where it is not.
    """
    n_rep = query.shape[1] // key.shape[1]
    key = repeat_kv(key, n_rep)
    value = repeat_kv(value, n_rep)

    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    scores = scores + attention_mask.to(scores.dtype)

    # Softmax in float32 even when the scores are bfloat16, as Gemma does.
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)

    out = torch.matmul(weights, value)
    return out.transpose(1, 2).contiguous()


class KVCache:
    """Per-layer key/value cache for the prefix, reused across denoising steps."""

    def __init__(self, num_layers: int):
        self.keys: list[Tensor | None] = [None] * num_layers
        self.values: list[Tensor | None] = [None] * num_layers

    def set(self, layer_idx: int, key: Tensor, value: Tensor) -> None:
        self.keys[layer_idx] = key
        self.values[layer_idx] = value

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        key, value = self.keys[layer_idx], self.values[layer_idx]
        if key is None or value is None:
            raise RuntimeError(f"KV cache for layer {layer_idx} was never filled.")
        return key, value


# ---------------------------------------------------------------------------
# Vision tower + projector
# ---------------------------------------------------------------------------


class SiglipVisionTower(nn.Module):
    """Thin wrapper over the SigLIP encoder that PaliGemma uses as its eye.

    Kept as a wrapper (rather than a reimplementation) because the encoder is a
    plain ViT with no pi0.5-specific behaviour, and this keeps the parameter
    names identical to the checkpoint.
    """

    def __init__(self, image_size: int = 224):
        super().__init__()
        config = SiglipVisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            num_channels=3,
            image_size=image_size,
            patch_size=14,
            hidden_act="gelu_pytorch_tanh",
            layer_norm_eps=1e-6,
            vision_use_head=False,
        )
        self.vision_model = SiglipVisionModel(config).vision_model

    def forward(self, pixel_values: Tensor) -> Tensor:
        target_dtype = self.vision_model.embeddings.patch_embedding.weight.dtype
        outputs = self.vision_model(pixel_values.to(target_dtype))
        return outputs.last_hidden_state


class MultiModalProjector(nn.Module):
    """Projects SigLIP patch features into the language model's width."""

    def __init__(self, vision_width: int, text_width: int):
        super().__init__()
        self.linear = nn.Linear(vision_width, text_width, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.to(self.linear.weight.dtype))


__all__ = [
    "GEMMA_VARIANTS",
    "GemmaMLP",
    "GemmaRotaryEmbedding",
    "GemmaVariantConfig",
    "KVCache",
    "MultiModalProjector",
    "PiGemmaDecoderLayer",
    "PiGemmaRMSNorm",
    "PiGemmaStack",
    "SiglipVisionTower",
    "apply_rotary_pos_emb",
    "gated_residual",
    "get_gemma_config",
    "joint_attention",
]
