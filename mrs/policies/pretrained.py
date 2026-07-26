"""Base class for policies that can be published to / loaded from the Hub."""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import TypeVar

import torch
from torch import Tensor, nn

from mrs.configs.policies import PreTrainedConfig
from mrs.constants import MODEL_WEIGHTS_NAME

T = TypeVar("T", bound="PreTrainedPolicy")

logger = logging.getLogger(__name__)


class PreTrainedPolicy(nn.Module, abc.ABC):
    """A policy: `select_action` for control, `forward` for training loss.

    Subclasses declare `config_class` and `name`, and are responsible for
    maintaining any temporal state (e.g. an action queue) that `reset` clears.
    """

    config_class: type[PreTrainedConfig]
    name: str

    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        self.config = config

    # ---- interface ------------------------------------------------------
    @abc.abstractmethod
    def reset(self) -> None:
        """Clear per-episode state. Call at the start of every episode."""

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action for the current observation."""

    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Return `(loss, metrics)` for a training batch."""

    # ---- persistence ----------------------------------------------------
    def save_pretrained(self, save_directory: str | Path) -> None:
        from safetensors.torch import save_file

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_directory)

        state = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(state, save_directory / MODEL_WEIGHTS_NAME)

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Instantiate from a local directory or Hub repo id."""
        from safetensors.torch import load_file

        from mrs.utils.hub import resolve_file

        hub_kwargs = {
            k: kwargs.pop(k)
            for k in ("revision", "cache_dir", "token", "local_files_only", "force_download")
            if k in kwargs
        }

        if config is None:
            config = cls.config_class.from_pretrained(pretrained_name_or_path, **hub_kwargs)

        policy = cls(config, **kwargs)

        weights_path = resolve_file(pretrained_name_or_path, MODEL_WEIGHTS_NAME, **hub_kwargs)
        state_dict = load_file(weights_path)
        policy.load_pretrained_state_dict(state_dict, strict=strict)
        return policy

    def load_pretrained_state_dict(self, state_dict: dict[str, Tensor], *, strict: bool = True) -> None:
        """Load published weights. Override to remap keys or handle tied weights."""
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        _report_load(missing, unexpected)


def _report_load(missing: list[str], unexpected: list[str]) -> None:
    if missing:
        logger.warning("Missing %d key(s) when loading weights, e.g. %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected %d key(s) when loading weights, e.g. %s", len(unexpected), unexpected[:5])
    if not missing and not unexpected:
        logger.info("All weight keys matched.")


def get_device_from_parameters(module: nn.Module) -> torch.device:
    return next(iter(module.parameters())).device
