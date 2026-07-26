"""Load a policy together with the processor pipelines it was published with."""

from __future__ import annotations

import logging
from pathlib import Path

from mrs.configs.policies import PreTrainedConfig
from mrs.constants import POSTPROCESSOR_CONFIG_NAME, PREPROCESSOR_CONFIG_NAME
from mrs.policies.pretrained import PreTrainedPolicy
from mrs.processor import PolicyProcessorPipeline
from mrs.types import FeatureType

logger = logging.getLogger(__name__)

POLICY_REGISTRY: dict[str, type[PreTrainedPolicy]] = {}


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    if name not in POLICY_REGISTRY:
        # Imported lazily so `transformers` is only needed when pi0.5 is used.
        if name == "pi05":
            from mrs.policies.pi05 import PI05Policy

            POLICY_REGISTRY["pi05"] = PI05Policy
        else:
            raise ValueError(f"Unknown policy type {name!r}.")
    return POLICY_REGISTRY[name]


def make_policy(
    pretrained_name_or_path: str | Path,
    *,
    device: str | None = None,
    config_overrides: dict | None = None,
    **hub_kwargs,
) -> tuple[PreTrainedPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Return `(policy, preprocessor, postprocessor)` ready for rollout.

    The processor pipelines come from the checkpoint itself, so normalization
    statistics and the prompt format always match the weights.
    """
    config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **hub_kwargs)
    if device is not None:
        config.device = device
    for key, value in (config_overrides or {}).items():
        setattr(config, key, value)

    policy_cls = get_policy_class(config.type)
    logger.info("Building %s on %s", policy_cls.__name__, config.device)

    policy = policy_cls.from_pretrained(pretrained_name_or_path, config=config, **hub_kwargs)
    policy.to(config.device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.load_input(
        pretrained_name_or_path, config_filename=PREPROCESSOR_CONFIG_NAME, **hub_kwargs
    )
    postprocessor = PolicyProcessorPipeline.load_output(
        pretrained_name_or_path, config_filename=POSTPROCESSOR_CONFIG_NAME, **hub_kwargs
    )

    # The published pipelines record whatever device the checkpoint was trained
    # or exported on; retarget them at ours.
    preprocessor.to(device=config.device)
    postprocessor.to(device="cpu")

    _validate_normalization(config, preprocessor)

    return policy, preprocessor, postprocessor


def _validate_normalization(config: PreTrainedConfig, preprocessor: PolicyProcessorPipeline) -> None:
    """Fail loudly when a checkpoint carries no usable normalization statistics.

    Base checkpoints published for fine-tuning ship an empty `features` map, and
    a normalizer with no features is a no-op: raw observations would flow
    straight into the model. For pi0.5 that is especially quiet, because the
    state is discretized into bins spanning [-1, 1] — unnormalized values simply
    saturate to the end bins and the prompt looks superficially well-formed.
    """
    normalizer = preprocessor.step_by_name("normalizer_processor")
    if normalizer is None:
        raise ValueError("The checkpoint's preprocessor has no normalizer step.")

    required = {
        name
        for name, feature in config.input_features.items()
        if feature.type is not FeatureType.VISUAL
    }
    required |= set(config.output_features)

    missing = sorted(name for name in required if name not in normalizer.features)
    if missing:
        raise ValueError(
            f"{pretrained_hint(config)} declares no normalization statistics for {missing}. "
            "This is what base checkpoints intended for fine-tuning look like; they cannot be "
            "run zero-shot. Supply statistics via `make_pi05_pre_post_processors(config, "
            "dataset_stats=...)`, or use a fine-tuned checkpoint."
        )


def pretrained_hint(config: PreTrainedConfig) -> str:
    return f"This {config.type} checkpoint"
