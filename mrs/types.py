"""Core feature/transition types shared by envs, processors and policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch


class FeatureType(str, Enum):
    """Semantic role of a feature, used to pick a normalization mode."""

    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"


class NormalizationMode(str, Enum):
    """How a feature is normalized before entering (or after leaving) a policy."""

    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    QUANTILES = "QUANTILES"
    IDENTITY = "IDENTITY"


class PipelineFeatureType(str, Enum):
    """Which side of the pipeline a feature description belongs to."""

    OBSERVATION = "OBSERVATION"
    ACTION = "ACTION"


@dataclass(frozen=True)
class PolicyFeature:
    """A named tensor slot: its semantic type and its per-frame shape."""

    type: FeatureType
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicyFeature:
        return cls(type=FeatureType(d["type"]), shape=tuple(d["shape"]))


class TransitionKey(str, Enum):
    """Slots of the transition dict that flows through a processor pipeline."""

    OBSERVATION = "observation"
    ACTION = "action"
    REWARD = "reward"
    DONE = "done"
    TRUNCATED = "truncated"
    INFO = "info"
    COMPLEMENTARY_DATA = "complementary_data"


# A transition is a plain dict keyed by TransitionKey. Kept as a loose alias
# (rather than a dataclass) so steps can copy/extend it cheaply.
EnvTransition = dict[TransitionKey, Any]

PolicyAction = torch.Tensor


def create_transition(
    observation: dict[str, Any] | None = None,
    action: Any = None,
    reward: float | None = None,
    done: bool | None = None,
    truncated: bool | None = None,
    info: dict[str, Any] | None = None,
    complementary_data: dict[str, Any] | None = None,
) -> EnvTransition:
    """Build a fully-populated transition dict (every key present)."""
    return {
        TransitionKey.OBSERVATION: observation if observation is not None else {},
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info if info is not None else {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


@dataclass
class EnvStepResult:
    """Gymnasium-style step return, kept explicit for readability."""

    observation: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self):
        return self.observation, self.reward, self.terminated, self.truncated, self.info
