"""Base config for pretrained policies.

Mirrors LeRobot's `PreTrainedConfig`: a dataclass carrying the input/output
feature spec plus a `type` discriminator, serialized to `config.json` next to
the weights. Subclasses register themselves so `PreTrainedConfig.from_pretrained`
can dispatch on the `"type"` field of a published checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from mrs.constants import ACTION, PRETRAINED_CONFIG_NAME
from mrs.types import FeatureType, NormalizationMode, PolicyFeature

_CONFIG_REGISTRY: dict[str, type["PreTrainedConfig"]] = {}

# Policy modules that register a config on import. Imported lazily so that
# deserializing a config does not pull in heavy optional dependencies for
# policies the caller is not using.
_LAZY_CONFIG_MODULES: dict[str, str] = {"pi05": "mrs.policies.pi05.configuration_pi05"}


def _ensure_registered(type_name: str) -> None:
    if type_name in _CONFIG_REGISTRY or type_name not in _LAZY_CONFIG_MODULES:
        return
    import importlib

    importlib.import_module(_LAZY_CONFIG_MODULES[type_name])


def register_config(name: str):
    """Class decorator registering a config subclass under its `type` string."""

    def _wrap(cls: type[PreTrainedConfig]) -> type[PreTrainedConfig]:
        cls.type = name
        _CONFIG_REGISTRY[name] = cls
        return cls

    return _wrap


@dataclass
class PreTrainedConfig:
    """Common fields every policy config carries."""

    type: ClassVar[str] = "base"

    n_obs_steps: int = 1
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str = "cpu"
    use_amp: bool = False

    # ---- feature views -------------------------------------------------
    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {k: v for k, v in self.input_features.items() if v.type is FeatureType.VISUAL}

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for v in self.input_features.values():
            if v.type is FeatureType.STATE:
                return v
        return None

    @property
    def action_feature(self) -> PolicyFeature | None:
        return self.output_features.get(ACTION)

    @property
    def normalization_mapping(self) -> dict[FeatureType, NormalizationMode]:
        raise NotImplementedError

    def validate_features(self) -> None:
        if not self.input_features:
            raise ValueError(f"{type(self).__name__} requires at least one input feature.")
        if ACTION not in self.output_features:
            raise ValueError(f"{type(self).__name__} requires an '{ACTION}' output feature.")

    # ---- (de)serialization ---------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name in ("input_features", "output_features"):
                value = {k: v.to_dict() for k, v in value.items()}
            elif isinstance(value, tuple):
                value = list(value)
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreTrainedConfig:
        """Build a config, dispatching on `type` and ignoring unknown keys.

        Unknown keys are tolerated on purpose: LeRobot checkpoints carry
        training-only fields (optimizer/scheduler settings, hub metadata) that
        are irrelevant for inference here.
        """
        d = dict(d)
        type_name = d.pop("type", cls.type)
        _ensure_registered(type_name)
        target = _CONFIG_REGISTRY.get(type_name, cls)
        if target is PreTrainedConfig:
            raise ValueError(f"Unknown policy type {type_name!r}. Registered: {sorted(_CONFIG_REGISTRY)}")

        known = {f.name for f in dataclasses.fields(target)}
        kwargs = {k: v for k, v in d.items() if k in known}
        for key in ("input_features", "output_features"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = {k: PolicyFeature.from_dict(v) for k, v in kwargs[key].items()}
        return target(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, **hub_kwargs) -> PreTrainedConfig:
        from mrs.utils.hub import resolve_file

        path = resolve_file(pretrained_name_or_path, PRETRAINED_CONFIG_NAME, **hub_kwargs)
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        with open(save_directory / PRETRAINED_CONFIG_NAME, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
