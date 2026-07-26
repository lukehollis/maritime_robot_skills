"""Canonical string keys used across observations, batches and processor steps.

Mirrors LeRobot's `lerobot.utils.constants` so that checkpoints and processor
configs published for LeRobot policies load without renaming.
"""

OBS_PREFIX = "observation."
OBS_STATE = "observation.state"
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
OBS_ENV_STATE = "observation.environment_state"

OBS_LANGUAGE = "observation.language"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"

ACTION = "action"
REWARD = "next.reward"
DONE = "next.done"
TRUNCATED = "next.truncated"

TASK = "task"

# openpi uses a large finite negative constant rather than -inf for masked
# attention logits; reproduced here so numerics match the released checkpoints.
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38

PRETRAINED_CONFIG_NAME = "config.json"
PREPROCESSOR_CONFIG_NAME = "policy_preprocessor.json"
POSTPROCESSOR_CONFIG_NAME = "policy_postprocessor.json"
MODEL_WEIGHTS_NAME = "model.safetensors"
