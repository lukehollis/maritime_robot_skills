"""Flow-matching primitives for action-chunk generation.

The action head is a conditional flow-matching model: at train time it regresses
the velocity field `u_t = noise - action` along the straight path
`x_t = t * noise + (1 - t) * action`; at inference it integrates that field
backwards from `t = 1` (pure noise) to `t = 0` (the action chunk).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def sample_noise(shape: tuple[int, ...], device) -> Tensor:
    """Draw the `t = 1` endpoint of the probability path."""
    return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)


def sample_time_beta(
    batch_size: int,
    device,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    scale: float = 0.999,
    offset: float = 0.001,
) -> Tensor:
    """Beta-distributed training timesteps, biased toward the low-noise end."""
    # torch's Beta sampler goes through _sample_dirichlet, which has no MPS
    # kernel, so draw on CPU and move afterwards.
    dist = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32), torch.tensor(beta, dtype=torch.float32)
    )
    time = dist.sample((batch_size,)) * scale + offset
    return time.to(dtype=torch.float32, device=device)


def euler_integrate(
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    noise: Tensor,
    num_steps: int,
) -> Tensor:
    """Forward-Euler integration of the velocity field from `t = 1` to `t = 0`.

    Args:
        denoise_fn: maps `(x_t, t)` to the velocity `v_t`, where `t` is a
            float32 tensor of shape `(batch_size,)`.
        noise: the `t = 1` sample, shape `(batch_size, chunk_size, action_dim)`.
        num_steps: number of uniform Euler steps.
    """
    batch_size = noise.shape[0]
    dt = -1.0 / num_steps
    x_t = noise

    for step in range(num_steps):
        t = 1.0 + step * dt
        t_tensor = torch.tensor(t, dtype=torch.float32, device=noise.device).expand(batch_size)
        v_t = denoise_fn(x_t, t_tensor)
        x_t = x_t + dt * v_t

    return x_t
