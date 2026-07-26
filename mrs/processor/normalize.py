"""(Un)normalization steps driven by dataset statistics.

Statistics are stored as a flat `"<feature>.<stat>"` safetensors mapping — the
same layout LeRobot publishes — so a checkpoint's `q01/q99/mean/std/min/max`
tensors load directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from mrs.constants import ACTION
from mrs.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from mrs.types import (
    EnvTransition,
    FeatureType,
    NormalizationMode,
    PolicyFeature,
    TransitionKey,
)


@dataclass
class _NormalizationMixin(ProcessorStep):
    """Shared stats handling and the forward/inverse transforms."""

    features: dict[str, PolicyFeature] = field(default_factory=dict)
    norm_map: dict[FeatureType, NormalizationMode] = field(default_factory=dict)
    eps: float = 1e-8
    device: str | torch.device | None = None

    _stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        # Configs come back from JSON with plain strings/lists; rebuild the enums.
        if self.features and isinstance(next(iter(self.features.values())), dict):
            self.features = {k: PolicyFeature.from_dict(v) for k, v in self.features.items()}
        if self.norm_map and all(isinstance(k, str) for k in self.norm_map):
            self.norm_map = {
                FeatureType(k): NormalizationMode(v) for k, v in self.norm_map.items()
            }

    # ---- state ----------------------------------------------------------
    def state_dict(self) -> dict[str, Tensor]:
        return {f"{key}.{stat}": t.cpu() for key, sub in self._stats.items() for stat, t in sub.items()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self._stats = {}
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)
            self._stats.setdefault(key, {})[stat_name] = tensor.to(torch.float32)
        self._reshape_visual_stats()
        self.to(device=self.device)

    def _reshape_visual_stats(self) -> None:
        """Broadcast flat per-channel image stats to `(C, 1, 1)`."""
        for key, feature in self.features.items():
            if feature.type is not FeatureType.VISUAL or key not in self._stats:
                continue
            for stat_name, tensor in self._stats[key].items():
                if tensor.ndim == 1:
                    self._stats[key][stat_name] = tensor.reshape(-1, 1, 1)

    def to(self, device=None, dtype=None) -> _NormalizationMixin:
        if device is not None:
            self.device = device
            self._stats = {
                k: {s: t.to(device) for s, t in sub.items()} for k, sub in self._stats.items()
            }
        return self

    def get_config(self) -> dict[str, Any]:
        return {
            "eps": self.eps,
            "features": {k: v.to_dict() for k, v in self.features.items()},
            "norm_map": {k.value: v.value for k, v in self.norm_map.items()},
        }

    # ---- math -----------------------------------------------------------
    def _safe_denom(self, denom: Tensor) -> Tensor:
        return torch.where(denom == 0, torch.full_like(denom, self.eps), denom)

    def _apply(self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool) -> Tensor:
        mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if mode is NormalizationMode.IDENTITY:
            return tensor

        stats = self._stats.get(key)
        if not stats:
            raise KeyError(f"No normalization stats for feature {key!r} (mode {mode.value}).")

        tensor = tensor.to(torch.float32)

        if mode is NormalizationMode.MEAN_STD:
            mean, std = self._require(stats, key, "mean", "std")
            mean, std = mean.to(tensor.device), std.to(tensor.device)
            return tensor * (std + self.eps) + mean if inverse else (tensor - mean) / (std + self.eps)

        if mode is NormalizationMode.MIN_MAX:
            lo, hi = self._require(stats, key, "min", "max")
            lo, hi = lo.to(tensor.device), hi.to(tensor.device)
            denom = self._safe_denom(hi - lo)
            return (tensor + 1.0) * denom / 2.0 + lo if inverse else 2.0 * (tensor - lo) / denom - 1.0

        if mode is NormalizationMode.QUANTILES:
            lo, hi = self._require(stats, key, "q01", "q99")
            lo, hi = lo.to(tensor.device), hi.to(tensor.device)
            denom = self._safe_denom(hi - lo)
            return (tensor + 1.0) * denom / 2.0 + lo if inverse else 2.0 * (tensor - lo) / denom - 1.0

        raise ValueError(f"Unsupported normalization mode {mode}.")

    @staticmethod
    def _require(stats: dict[str, Tensor], key: str, *names: str) -> tuple[Tensor, ...]:
        missing = [n for n in names if n not in stats]
        if missing:
            raise KeyError(f"Feature {key!r} is missing normalization stats {missing}.")
        return tuple(stats[n] for n in names)

    def _run(self, transition: EnvTransition, *, inverse: bool) -> EnvTransition:
        transition = transition.copy()

        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            observation = dict(observation)
            for key, feature in self.features.items():
                if feature.type is FeatureType.ACTION or key not in observation:
                    continue
                observation[key] = self._apply(
                    torch.as_tensor(observation[key]), key, feature.type, inverse=inverse
                )
            transition[TransitionKey.OBSERVATION] = observation

        action = transition.get(TransitionKey.ACTION)
        if action is not None and ACTION in self.features:
            transition[TransitionKey.ACTION] = self._apply(
                torch.as_tensor(action), ACTION, FeatureType.ACTION, inverse=inverse
            )

        return transition


@ProcessorStepRegistry.register("normalizer_processor")
@dataclass
class NormalizerProcessorStep(_NormalizationMixin):
    """Maps raw observations/actions into the policy's normalized space."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return self._run(transition, inverse=False)


@ProcessorStepRegistry.register("unnormalizer_processor")
@dataclass
class UnnormalizerProcessorStep(_NormalizationMixin):
    """Maps policy outputs back into physical units."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return self._run(transition, inverse=True)
