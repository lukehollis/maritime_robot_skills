"""The small, general-purpose processor steps: rename, batch, device, tokenize."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from mrs.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    TASK,
)
from mrs.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from mrs.types import EnvTransition, TransitionKey


@ProcessorStepRegistry.register("rename_observations_processor")
@dataclass
class RenameObservationsProcessorStep(ProcessorStep):
    """Rename observation keys so an env's naming matches a checkpoint's."""

    rename_map: dict[str, str] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.rename_map:
            return transition
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        transition[TransitionKey.OBSERVATION] = {
            self.rename_map.get(k, k): v for k, v in observation.items()
        }
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"rename_map": self.rename_map}


@ProcessorStepRegistry.register("to_batch_processor")
@dataclass
class AddBatchDimensionProcessorStep(ProcessorStep):
    """Turn a single unbatched sample into a batch of one.

    Tensors gain a leading axis; the `task` string becomes a one-element list,
    which is what the downstream prompt builder iterates over.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            transition[TransitionKey.OBSERVATION] = {
                k: _batch_tensor(v) for k, v in observation.items()
            }

        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = _batch_tensor(action)

        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary:
            complementary = dict(complementary)
            task = complementary.get(TASK)
            if isinstance(task, str):
                complementary[TASK] = [task]
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary

        return transition


def _batch_tensor(value):
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    return value


@ProcessorStepRegistry.register("device_processor")
@dataclass
class DeviceProcessorStep(ProcessorStep):
    """Move every tensor in the transition to a device (and optionally a dtype)."""

    device: str = "cpu"
    float_dtype: str | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            transition[TransitionKey.OBSERVATION] = {k: self._move(v) for k, v in observation.items()}

        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = self._move(action)

        return transition

    def _move(self, value):
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if not isinstance(value, torch.Tensor):
            return value
        value = value.to(self.device)
        # Only floats are cast: integer token ids and boolean masks must survive.
        if self.float_dtype is not None and value.dtype.is_floating_point:
            value = value.to(getattr(torch, self.float_dtype))
        return value

    def to(self, device=None, dtype=None) -> DeviceProcessorStep:
        if device is not None:
            self.device = str(device)
        return self

    def get_config(self) -> dict[str, Any]:
        return {"device": self.device, "float_dtype": self.float_dtype}


@ProcessorStepRegistry.register("to_cpu_processor")
@dataclass
class ToCPUProcessorStep(DeviceProcessorStep):
    device: str = "cpu"


# `google/paligemma-3b-pt-224` is gated behind the Gemma licence. These
# ungated repos carry a byte-identical copy of its tokenizer; they are used only
# when the canonical repo is unreachable, and the result is checked against the
# properties the pi0.5 prompt format depends on.
TOKENIZER_MIRRORS: dict[str, tuple[str, ...]] = {
    "google/paligemma-3b-pt-224": ("leo009/paligemma-3b-pt-224",),
}

EXPECTED_TOKENIZER_PROPERTIES: dict[str, dict[str, int]] = {
    "google/paligemma-3b-pt-224": {"vocab_size": 257152, "bos_token_id": 2, "eos_token_id": 1},
}


def _load_tokenizer(name: str):
    """Load a tokenizer, falling back to a verified mirror if the repo is gated."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:
        mirrors = TOKENIZER_MIRRORS.get(name, ())
        if not mirrors:
            raise

        for mirror in mirrors:
            try:
                tokenizer = AutoTokenizer.from_pretrained(mirror)
            except Exception:
                continue
            _verify_tokenizer(tokenizer, name, mirror)
            logging.getLogger(__name__).warning(
                "%s is not accessible: %s Using the mirror %s instead.", name, _gating_hint(exc, name), mirror
            )
            return tokenizer
        raise


def _gating_hint(exc: Exception, name: str) -> str:
    """Turn a Hub access failure into the specific step the user still owes.

    401 and 403 need different fixes and are easy to confuse: one means no
    credentials, the other means credentials that have not been granted access.
    """
    message = str(exc)
    url = f"https://huggingface.co/{name}"

    if "403" in message:
        return (
            f"you are authenticated, but this account has not been granted access. "
            f"Accept the licence at {url} (it is gated manually, so approval may not be instant)."
        )
    if "401" in message:
        return f"no valid credentials. Run `hf auth login`, then accept the licence at {url}."
    return f"{type(exc).__name__}. See {url}."


def _verify_tokenizer(tokenizer, canonical_name: str, mirror: str) -> None:
    """Fail loudly if a mirror does not match the canonical tokenizer's contract."""
    expected = EXPECTED_TOKENIZER_PROPERTIES.get(canonical_name)
    if not expected:
        return
    actual = {key: getattr(tokenizer, key, None) for key in expected}
    if actual != expected:
        raise ValueError(
            f"Tokenizer mirror {mirror!r} does not match {canonical_name!r}: "
            f"expected {expected}, got {actual}."
        )


@ProcessorStepRegistry.register("tokenizer_processor")
@dataclass
class TokenizerProcessorStep(ProcessorStep):
    """Tokenize the task prompt into `observation.language.{tokens,attention_mask}`."""

    tokenizer_name: str | None = None
    max_length: int = 200
    task_key: str = TASK
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    def __post_init__(self):
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self.tokenizer_name is None:
                raise ValueError("TokenizerProcessorStep requires `tokenizer_name`.")
            self._tokenizer = _load_tokenizer(self.tokenizer_name)
        return self._tokenizer

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        task = complementary.get(self.task_key)
        if task is None:
            raise ValueError(f"No {self.task_key!r} found in complementary data for tokenization.")

        encoded = self.tokenizer(
            task if isinstance(task, list) else [task],
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )

        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        observation[OBS_LANGUAGE_TOKENS] = encoded["input_ids"]
        observation[OBS_LANGUAGE_ATTENTION_MASK] = encoded["attention_mask"].to(torch.bool)
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "task_key": self.task_key,
            "padding_side": self.padding_side,
            "padding": self.padding,
            "truncation": self.truncation,
            "tokenizer_name": self.tokenizer_name,
        }


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Convert absolute actions to deltas relative to the current state.

    Present so that checkpoints which serialize this step (e.g. `pi05_base`)
    load. Every published pi0.5 checkpoint ships it disabled.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.enabled:
            raise NotImplementedError(
                "Relative-action training is not supported by this inference stack."
            )
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(RelativeActionsProcessorStep):
    """Inverse of :class:`RelativeActionsProcessorStep`; likewise a no-op when disabled."""
