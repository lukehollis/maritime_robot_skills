"""Processor pipeline: an ordered, serializable chain of data transforms.

Mirrors LeRobot's `processor` package. A *step* takes an `EnvTransition` and
returns a new one; a *pipeline* chains steps and knows how to save/load itself
as `<name>.json` plus one `<name>_step_<i>_<registry_name>.safetensors` per step
that carries tensor state. That on-disk layout is exactly what the published
LeRobot checkpoints use, so their preprocessor/postprocessor configs load here
unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

import torch

from mrs.types import (
    EnvTransition,
    PipelineFeatureType,
    PolicyFeature,
    TransitionKey,
    create_transition,
)

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class ProcessorStepRegistry:
    """Maps a stable string name to a step class, for (de)serialization."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def _wrap(step_cls: type):
            if name in cls._registry and cls._registry[name] is not step_cls:
                raise ValueError(f"Processor step {name!r} is already registered.")
            step_cls.registry_name = name
            cls._registry[name] = step_cls
            return step_cls

        return _wrap

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            raise KeyError(f"Unknown processor step {name!r}. Registered: {sorted(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def contains(cls, name: str) -> bool:
        return name in cls._registry


@dataclass
class ProcessorStep:
    """Base class for a single transform in a pipeline."""

    # ClassVar, not a field: `ProcessorStepRegistry.register` sets this on the
    # class, and a dataclass field would shadow it with "" on every instance.
    registry_name: ClassVar[str] = ""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        """JSON-serializable constructor arguments for this step."""
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Tensor state persisted alongside the config (e.g. norm stats)."""
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state:
            raise ValueError(f"{type(self).__name__} does not accept tensor state.")

    def reset(self) -> None:
        """Clear any per-episode state. Called on env reset."""

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> ProcessorStep:
        return self

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Describe how this step changes the feature spec. Default: unchanged."""
        return features


class DataProcessorPipeline(Generic[TIn, TOut]):
    """An ordered list of steps with save/load support."""

    def __init__(
        self,
        steps: list[ProcessorStep] | None = None,
        *,
        name: str = "processor",
        to_transition=None,
        to_output=None,
    ):
        self.steps: list[ProcessorStep] = list(steps or [])
        self.name = name
        self._to_transition = to_transition
        self._to_output = to_output

    # ---- execution ------------------------------------------------------
    def __call__(self, data: TIn) -> TOut:
        transition = self._to_transition(data) if self._to_transition else data
        for step in self.steps:
            transition = step(transition)
        return self._to_output(transition) if self._to_output else transition

    def reset(self) -> None:
        for step in self.steps:
            step.reset()

    def to(self, device=None, dtype=None) -> DataProcessorPipeline:
        for step in self.steps:
            step.to(device=device, dtype=dtype)
        return self

    def step_by_name(self, registry_name: str) -> ProcessorStep | None:
        for step in self.steps:
            if getattr(step, "registry_name", None) == registry_name:
                return step
        return None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for step in self.steps:
            features = step.transform_features(features)
        return features

    # ---- (de)serialization ---------------------------------------------
    def save_pretrained(self, save_directory: str | Path) -> None:
        from safetensors.torch import save_file

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        step_entries = []
        for i, step in enumerate(self.steps):
            registry_name = getattr(step, "registry_name", "") or type(step).__name__
            entry: dict[str, Any] = {"registry_name": registry_name, "config": step.get_config()}
            state = step.state_dict()
            if state:
                fname = f"{self.name}_step_{i}_{registry_name}.safetensors"
                save_file({k: v.contiguous().cpu() for k, v in state.items()}, save_directory / fname)
                entry["state_file"] = fname
            step_entries.append(entry)

        with open(save_directory / f"{self.name}.json", "w") as f:
            json.dump({"name": self.name, "steps": step_entries}, f, indent=1)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config_filename: str,
        to_transition=None,
        to_output=None,
        strict: bool = True,
        **hub_kwargs,
    ) -> DataProcessorPipeline:
        """Rebuild a pipeline from a published `<name>.json` + state files."""
        from safetensors.torch import load_file

        from mrs.utils.hub import resolve_file

        config_path = resolve_file(pretrained_name_or_path, config_filename, **hub_kwargs)
        with open(config_path) as f:
            spec = json.load(f)

        steps: list[ProcessorStep] = []
        for entry in spec["steps"]:
            registry_name = entry["registry_name"]
            if not ProcessorStepRegistry.contains(registry_name):
                if strict:
                    raise KeyError(
                        f"Checkpoint uses processor step {registry_name!r}, which is not implemented here."
                    )
                continue
            step_cls = ProcessorStepRegistry.get(registry_name)
            step = step_cls(**_decode_config(step_cls, entry.get("config", {})))

            state_file = entry.get("state_file")
            if state_file:
                state_path = resolve_file(pretrained_name_or_path, state_file, **hub_kwargs)
                step.load_state_dict(load_file(state_path))
            steps.append(step)

        return cls(steps, name=spec.get("name", Path(config_filename).stem),
                   to_transition=to_transition, to_output=to_output)


def _decode_config(step_cls: type, config: dict[str, Any]) -> dict[str, Any]:
    """Filter a serialized config down to the step's actual fields."""
    import dataclasses

    if not dataclasses.is_dataclass(step_cls):
        return dict(config)
    known = {f.name for f in dataclasses.fields(step_cls)}
    return {k: v for k, v in config.items() if k in known}


# ---------------------------------------------------------------------------
# Policy-facing pipelines: dict[str, Any] -> dict[str, Any] on the input side,
# Tensor -> Tensor on the output side.
# ---------------------------------------------------------------------------


def _batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """Split a flat batch dict into the observation / action / extras slots."""
    observation, complementary = {}, {}
    action = None
    for key, value in batch.items():
        if key.startswith("observation"):
            observation[key] = value
        elif key == "action":
            action = value
        else:
            complementary[key] = value
    return create_transition(observation=observation, action=action, complementary_data=complementary)


def _transition_to_batch(transition: EnvTransition) -> dict[str, Any]:
    """Flatten a transition back into the dict a policy consumes."""
    batch = dict(transition.get(TransitionKey.OBSERVATION) or {})
    batch.update(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
    action = transition.get(TransitionKey.ACTION)
    if action is not None:
        batch["action"] = action
    return batch


def _action_to_transition(action) -> EnvTransition:
    return create_transition(action=action)


def _transition_to_action(transition: EnvTransition):
    return transition.get(TransitionKey.ACTION)


class PolicyProcessorPipeline(DataProcessorPipeline[TIn, TOut]):
    """Pipeline with the batch<->transition adapters wired in."""

    @classmethod
    def make_input(cls, steps: list[ProcessorStep], name: str = "policy_preprocessor"):
        return cls(steps, name=name, to_transition=_batch_to_transition, to_output=_transition_to_batch)

    @classmethod
    def make_output(cls, steps: list[ProcessorStep], name: str = "policy_postprocessor"):
        return cls(steps, name=name, to_transition=_action_to_transition, to_output=_transition_to_action)

    @classmethod
    def load_input(cls, pretrained_name_or_path, *, config_filename="policy_preprocessor.json", **kw):
        return cls.from_pretrained(
            pretrained_name_or_path,
            config_filename=config_filename,
            to_transition=_batch_to_transition,
            to_output=_transition_to_batch,
            **kw,
        )

    @classmethod
    def load_output(cls, pretrained_name_or_path, *, config_filename="policy_postprocessor.json", **kw):
        return cls.from_pretrained(
            pretrained_name_or_path,
            config_filename=config_filename,
            to_transition=_action_to_transition,
            to_output=_transition_to_action,
            **kw,
        )
