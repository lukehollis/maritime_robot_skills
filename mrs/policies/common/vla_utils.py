"""Stateless helpers shared by the VLA stack: masks, embeddings, image resizing.

These reproduce the openpi conventions exactly, because the published pi0.5
weights were trained against them.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from mrs.constants import OPENPI_ATTENTION_MASK_VALUE


def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Sine-cosine embedding of a scalar per batch element (the flow timestep)."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("`time` must have shape (batch_size,)")

    # float64 on CPU/CUDA for period precision; MPS has no float64, so fall back.
    dtype = torch.float32 if device.type == "mps" else torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Build a block-wise attention matrix from a padding mask and a block mask.

    `att_masks` marks where a new attention block starts (1) or where a token
    shares the previous token's visibility (0). Cumulative sums then define
    "can attend to everything in my own block and all earlier blocks", which is
    how pi0.5 gets bidirectional attention within the prefix and causal
    attention from the action tokens onto it.
    """
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError("Both masks must be 2D (batch_size, seq_len).")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def prepare_attention_masks_4d(att_2d_masks: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """Convert a boolean mask to the additive 4D form attention kernels expect."""
    result = torch.where(att_2d_masks[:, None, :, :], 0.0, OPENPI_ATTENTION_MASK_VALUE)
    return result.to(dtype=dtype) if dtype is not None else result


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Zero-pad the last dimension up to `new_dim`, leaving it alone if already larger."""
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad(images: Tensor, height: int, width: int, mode: str = "bilinear") -> Tensor:
    """Resize preserving aspect ratio, centre-padding the remainder with black.

    Accepts `[B, C, H, W]` or `[B, H, W, C]` and returns the same layout.
    float32 inputs are assumed to be in `[0, 1]`.
    """
    channels_last = images.shape[-1] <= 4
    if images.dim() == 3:
        images = images.unsqueeze(0)
    if channels_last:
        images = images.permute(0, 3, 1, 2)

    _, _, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized = resized.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, rem_h = divmod(height - resized_height, 2)
    pad_w0, rem_w = divmod(width - resized_width, 2)
    padded = F.pad(
        resized,
        (pad_w0, pad_w0 + rem_w, pad_h0, pad_h0 + rem_h),
        mode="constant",
        value=0 if images.dtype == torch.uint8 else 0.0,
    )

    return padded.permute(0, 2, 3, 1) if channels_last else padded
