"""Closed-loop evaluation: drive the environment with a policy.

The bridge between the two lives here, in `env_observation_to_batch`. The env
emits `uint8` CHW images because that is what a camera gives you; the policy's
own pipeline expects floats in [0, 1] (its visual normalization mode is
IDENTITY, so nothing downstream would rescale them). Doing the conversion here
rather than inside the policy pipeline keeps the published checkpoint's
processor config untouched.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from mrs.constants import OBS_IMAGES, TASK

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Outcome of one episode."""

    success: bool
    steps: int
    total_reward: float
    seed: int | None = None
    frames: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    inference_calls: int = 0
    wall_time: float = 0.0


def env_observation_to_batch(
    observation: dict[str, np.ndarray], task: str, *, rotate_images_180: bool = False
) -> dict[str, torch.Tensor | str]:
    """Convert one env observation into the unbatched dict a policy pipeline takes.

    `rotate_images_180` reproduces the image convention openpi uses for LIBERO
    (`image[::-1, ::-1]`). This environment renders right-side-up, so the flag
    is off by default; turn it on when deploying a LIBERO-trained checkpoint.
    See `docs/findings.md` for the measured effect.
    """
    batch: dict[str, torch.Tensor | str] = {}
    for key, value in observation.items():
        if key.startswith(OBS_IMAGES):
            if value.dtype != np.uint8:
                raise TypeError(f"Expected uint8 images for {key!r}, got {value.dtype}.")
            if rotate_images_180:
                value = value[:, ::-1, ::-1]  # CHW: flip height and width
            tensor = torch.from_numpy(np.ascontiguousarray(value)).to(torch.float32) / 255.0
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(value)).to(torch.float32)
        batch[key] = tensor
    batch[TASK] = task
    return batch


def rollout_episode(
    env,
    policy,
    preprocessor,
    postprocessor,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
    record_video: bool = False,
    render_size: int | None = None,
    task: str | None = None,
    rotate_images_180: bool = False,
    progress_every: int = 0,
) -> RolloutResult:
    """Run one episode and return what happened.

    The policy owns its action chunking: `select_action` refills an internal
    queue whenever it drains, so this loop calls it once per environment step
    and the model only runs every `n_action_steps` steps.
    """
    observation, info = env.reset(seed=seed)
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    task = task or info.get(TASK) or env.config.task
    max_steps = max_steps or env.config.max_episode_steps

    result = RolloutResult(success=False, steps=0, total_reward=0.0, seed=seed)
    start = time.perf_counter()

    if record_video:
        result.frames.append(env.render(render_size))

    for step in range(max_steps):
        queue_was_empty = getattr(policy, "pending_actions", 0) == 0

        batch = preprocessor(
            env_observation_to_batch(observation, task, rotate_images_180=rotate_images_180)
        )
        action = policy.select_action(batch)
        action = postprocessor(action)

        if queue_was_empty:
            result.inference_calls += 1

        action_np = action.squeeze(0).detach().cpu().numpy()
        result.actions.append(action_np)
        result.states.append(observation["observation.state"].copy())

        observation, reward, terminated, truncated, info = env.step(action_np)
        result.total_reward += float(reward)
        result.steps = step + 1

        if record_video:
            result.frames.append(env.render(render_size))

        if progress_every and (step + 1) % progress_every == 0:
            logger.info(
                "  step %3d/%d  eef=%s  cube=%s",
                step + 1,
                max_steps,
                np.round(info["eef_position"], 3),
                np.round(info["cube_position"], 3),
            )

        if terminated or truncated:
            result.success = bool(info.get("is_success", False))
            break

    result.wall_time = time.perf_counter() - start
    return result


def evaluate(
    env,
    policy,
    preprocessor,
    postprocessor,
    *,
    episodes: int = 10,
    start_seed: int = 0,
    **rollout_kwargs,
) -> tuple[list[RolloutResult], dict]:
    """Run several episodes and summarise them."""
    results = []
    for index in range(episodes):
        seed = start_seed + index
        result = rollout_episode(env, policy, preprocessor, postprocessor, seed=seed, **rollout_kwargs)
        results.append(result)
        logger.info(
            "episode %d/%d  seed=%d  success=%s  steps=%d  inferences=%d  %.1fs",
            index + 1,
            episodes,
            seed,
            result.success,
            result.steps,
            result.inference_calls,
            result.wall_time,
        )

    successes = [r.success for r in results]
    summary = {
        "episodes": episodes,
        "successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if results else 0.0,
        "mean_steps": float(np.mean([r.steps for r in results])) if results else 0.0,
        "mean_wall_time_s": float(np.mean([r.wall_time for r in results])) if results else 0.0,
        "total_inference_calls": int(sum(r.inference_calls for r in results)),
    }
    return results, summary
