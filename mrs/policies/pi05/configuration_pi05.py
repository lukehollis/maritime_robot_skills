"""Configuration for the pi0.5 policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from mrs.configs.policies import PreTrainedConfig, register_config
from mrs.types import FeatureType, NormalizationMode, PolicyFeature

DEFAULT_IMAGE_SIZE = 224


@register_config("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    """Everything needed to rebuild a pi0.5 model and its I/O contract.

    Field names and defaults follow the published `config.json` of the LeRobot
    pi0.5 checkpoints, so those files deserialize directly into this class.
    """

    # Backbone sizes.
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"

    # Action chunking.
    chunk_size: int = 50
    n_action_steps: int = 50

    # pi0.5 pads state and action to a fixed width so one checkpoint serves
    # many embodiments; the extra dimensions are zeros.
    max_action_dim: int = 32
    max_state_dim: int = 32

    # Flow matching.
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Inputs.
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    tokenizer_max_length: int = 200
    tokenizer_name: str = "google/paligemma-3b-pt-224"

    # Training-only knobs, kept so training configs round-trip.
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    gradient_checkpointing: bool = False

    normalization_mapping_override: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.image_resolution, list):
            self.image_resolution = tuple(self.image_resolution)
        if self.image_resolution[0] != self.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects a square image resolution, got {self.image_resolution}."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})."
            )

    @property
    def normalization_mapping(self) -> dict[FeatureType, NormalizationMode]:
        """Default modes; a loaded checkpoint's processor config takes precedence."""
        default = {
            FeatureType.VISUAL: NormalizationMode.IDENTITY,
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        }
        for key, value in self.normalization_mapping_override.items():
            default[FeatureType(key)] = NormalizationMode(value)
        return default

    def validate_features(self) -> None:
        super().validate_features()
        if not self.image_features:
            raise ValueError("pi0.5 requires at least one visual input feature.")
        if self.robot_state_feature is None:
            raise ValueError("pi0.5 requires an 'observation.state' input feature.")

        state_dim = self.robot_state_feature.shape[0]
        if state_dim > self.max_state_dim:
            raise ValueError(f"State dim {state_dim} exceeds max_state_dim {self.max_state_dim}.")

        action_dim = self.output_features["action"].shape[0]
        if action_dim > self.max_action_dim:
            raise ValueError(f"Action dim {action_dim} exceeds max_action_dim {self.max_action_dim}.")

    @property
    def empty_camera_keys(self) -> list[str]:
        """Image slots the checkpoint expects but that carry no real camera.

        pi0.5 was pretrained with three camera slots. Checkpoints fine-tuned on
        two-camera setups declare the third as an `empty_camera_*` feature; it
        is fed an all -1 image with its attention mask cleared.
        """
        return [k for k in self.image_features if "empty_camera" in k]

    @property
    def real_camera_keys(self) -> list[str]:
        return [k for k in self.image_features if "empty_camera" not in k]
