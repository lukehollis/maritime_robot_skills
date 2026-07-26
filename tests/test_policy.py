"""Tests for the pi0.5 implementation and the processor pipelines.

Split into tests that need only CPU and a tiny random model, and tests marked
`slow` that download and run the real 3.4 B-parameter checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mrs.configs.policies import PreTrainedConfig
from mrs.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from mrs.policies.common.vla_utils import make_att_2d_masks, pad_vector, resize_with_pad
from mrs.policies.pi05 import PI05Config
from mrs.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerStep
from mrs.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from mrs.types import FeatureType, NormalizationMode, PolicyFeature, TransitionKey, create_transition

CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_published_config_deserializes_into_pi05config():
    config = PreTrainedConfig.from_dict(
        {
            "type": "pi05",
            "chunk_size": 50,
            "n_action_steps": 50,
            "image_resolution": [224, 224],
            "input_features": {
                "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.state": {"type": "STATE", "shape": [8]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [7]}},
            # Training-only fields a real checkpoint carries; must be ignored.
            "optimizer_lr": 2.5e-05,
            "scheduler_warmup_steps": 1000,
        }
    )
    assert isinstance(config, PI05Config)
    assert config.image_resolution == (224, 224)
    assert config.output_features["action"].shape == (7,)
    assert config.robot_state_feature.shape == (8,)


def test_config_rejects_non_square_images():
    with pytest.raises(ValueError, match="square"):
        PI05Config(image_resolution=(224, 256))


def test_config_rejects_executing_more_actions_than_the_chunk_holds():
    with pytest.raises(ValueError, match="n_action_steps"):
        PI05Config(chunk_size=10, n_action_steps=50)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def test_prefix_attends_bidirectionally_and_action_tokens_attend_causally():
    """pi0.5 splits attention into blocks: the prefix sees itself, the action
    block sees the prefix, and the prefix must never see the action block."""
    pad = torch.ones(1, 5, dtype=torch.bool)
    # Three prefix tokens (block 0) then two action tokens (block 1).
    att = torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.bool)

    mask = make_att_2d_masks(pad, att)[0]

    assert mask[:3, :3].all()          # prefix is bidirectional
    assert not mask[:3, 3:].any()      # prefix cannot see actions
    assert mask[3:, :3].all()          # actions can see the prefix
    assert mask[3:, 3:].all()          # actions see each other


def test_padding_masks_exclude_padded_tokens():
    pad = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    att = torch.zeros(1, 3, dtype=torch.bool)
    mask = make_att_2d_masks(pad, att)[0]
    assert not mask[:, 2].any()
    assert not mask[2, :].any()


def test_pad_vector_pads_but_never_truncates():
    assert pad_vector(torch.ones(2, 7), 32).shape == (2, 32)
    assert torch.equal(pad_vector(torch.ones(2, 7), 32)[:, 7:], torch.zeros(2, 25))
    # Already wide enough: returned unchanged, matching openpi behaviour.
    assert pad_vector(torch.ones(2, 32), 8).shape == (2, 32)


def test_resize_with_pad_preserves_aspect_ratio():
    image = torch.zeros(1, 3, 100, 200, dtype=torch.float32)
    resized = resize_with_pad(image, 224, 224)
    assert resized.shape == (1, 3, 224, 224)
    # A 2:1 image becomes 224x112 centred, so the top and bottom are padding.
    assert torch.all(resized[0, :, :50, :] == 0.0)


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


def _stats_transition(state):
    return create_transition(
        observation={"observation.state": torch.as_tensor(state)},
        complementary_data={"task": ["pick up the block"]},
    )


def test_normalizer_and_unnormalizer_are_inverses():
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (3,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
    stats = {
        "observation.state.mean": torch.tensor([1.0, 2.0, 3.0]),
        "observation.state.std": torch.tensor([0.5, 2.0, 4.0]),
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map)
    normalizer.load_state_dict(stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map)
    unnormalizer.load_state_dict(stats)

    original = torch.tensor([1.5, 0.0, 11.0])
    normalized = normalizer(_stats_transition(original))[TransitionKey.OBSERVATION]["observation.state"]
    assert torch.allclose(normalized, torch.tensor([1.0, -1.0, 2.0]), atol=1e-5)

    restored = unnormalizer(
        create_transition(observation={"observation.state": normalized})
    )[TransitionKey.OBSERVATION]["observation.state"]
    assert torch.allclose(restored, original, atol=1e-4)


def test_quantile_normalization_maps_q01_and_q99_to_minus_one_and_one():
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (2,))}
    normalizer = NormalizerProcessorStep(
        features=features, norm_map={FeatureType.STATE: NormalizationMode.QUANTILES}
    )
    normalizer.load_state_dict(
        {
            "observation.state.q01": torch.tensor([-1.0, 0.0]),
            "observation.state.q99": torch.tensor([1.0, 10.0]),
        }
    )
    out = normalizer(_stats_transition(torch.tensor([-1.0, 10.0])))
    assert torch.allclose(
        out[TransitionKey.OBSERVATION]["observation.state"], torch.tensor([-1.0, 1.0]), atol=1e-6
    )


def test_steps_report_their_registry_name_on_instances():
    """`registry_name` must not be a dataclass field, or every instance would
    shadow the class attribute the registry sets, breaking save and lookup."""
    from mrs.processor import DeviceProcessorStep, ProcessorStepRegistry

    assert NormalizerProcessorStep().registry_name == "normalizer_processor"
    assert DeviceProcessorStep().registry_name == "device_processor"
    assert ProcessorStepRegistry.get("normalizer_processor") is NormalizerProcessorStep


def test_pipeline_round_trips_through_disk(tmp_path):
    from mrs.processor import PolicyProcessorPipeline

    features = {"observation.state": PolicyFeature(FeatureType.STATE, (2,))}
    step = NormalizerProcessorStep(
        features=features, norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD}
    )
    step.load_state_dict(
        {"observation.state.mean": torch.tensor([1.0, 2.0]),
         "observation.state.std": torch.tensor([2.0, 4.0])}
    )
    PolicyProcessorPipeline.make_input([step], name="policy_preprocessor").save_pretrained(tmp_path)

    reloaded = PolicyProcessorPipeline.load_input(tmp_path)
    out = reloaded({"observation.state": torch.tensor([3.0, 6.0]), "task": "x"})
    assert torch.allclose(out["observation.state"], torch.tensor([[1.0, 1.0]]), atol=1e-5)


def test_missing_stats_raise_rather_than_silently_passing_data_through():
    normalizer = NormalizerProcessorStep(
        features={"observation.state": PolicyFeature(FeatureType.STATE, (2,))},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
    )
    with pytest.raises(KeyError):
        normalizer(_stats_transition(torch.zeros(2)))


def test_state_is_discretized_into_the_prompt():
    """pi0.5 has no state encoder: proprioception must arrive as text."""
    step = Pi05PrepareStateTokenizerStep()
    # Already normalized to [-1, 1] by the preceding normalizer.
    transition = _stats_transition(torch.tensor([[-1.0, 0.0, 1.0]]))
    prompt = step(transition)[TransitionKey.COMPLEMENTARY_DATA]["task"][0]

    assert prompt.startswith("Task: pick up the block, State: ")
    assert prompt.endswith(";\nAction: ")

    bins = [int(token) for token in prompt.split("State: ")[1].split(";")[0].split()]
    assert bins == [0, 128, 255]


def test_state_discretization_does_not_clamp_out_of_range_values():
    """MEAN_STD normalization is unbounded, so bins outside 0..255 are normal.

    The reference implementation does not clamp either, so training saw these
    same indices; clamping would emit prompt tokens the checkpoint never saw.
    """
    step = Pi05PrepareStateTokenizerStep()
    transition = _stats_transition(torch.tensor([[-9.0, 9.0]]))
    prompt = step(transition)[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    bins = [int(token) for token in prompt.split("State: ")[1].split(";")[0].split()]
    assert bins == [-1, 255]


def test_underscores_and_newlines_are_cleaned_from_the_task():
    step = Pi05PrepareStateTokenizerStep()
    transition = create_transition(
        observation={"observation.state": torch.zeros(1, 2)},
        complementary_data={"task": ["  pick_up\nthe block  "]},
    )
    prompt = step(transition)[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    assert prompt.startswith("Task: pick up the block, State: ")


# ---------------------------------------------------------------------------
# The real checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_checkpoint_loads_with_every_weight_accounted_for():
    """A silent key mismatch would leave parts of the model randomly initialised."""
    from safetensors import safe_open

    from mrs.policies.pi05 import PI05Policy
    from mrs.utils.hub import resolve_file

    config = PreTrainedConfig.from_pretrained(CHECKPOINT)
    with torch.device("meta"):
        policy = PI05Policy(config)

    own = set(policy.state_dict())
    with safe_open(resolve_file(CHECKPOINT, "model.safetensors"), framework="pt") as handle:
        checkpoint = set(handle.keys())

    mapped = {key if key.startswith("model.") else f"model.{key}" for key in checkpoint}

    assert not own - mapped, f"model parameters absent from the checkpoint: {sorted(own - mapped)}"
    # The action expert's vocabulary head is the one tensor we deliberately drop.
    assert mapped - own == {"model.paligemma_with_expert.gemma_expert.lm_head.weight"}


@pytest.mark.slow
def test_end_to_end_inference_produces_a_usable_action_chunk():
    from mrs.policies import make_policy

    policy, preprocessor, postprocessor = make_policy(CHECKPOINT, device="cpu")

    observation = {
        "observation.images.image": torch.rand(3, 256, 256),
        "observation.images.image2": torch.rand(3, 256, 256),
        "observation.state": torch.tensor([-0.04, 0.03, 0.76, 2.97, -0.22, -0.13, 0.027, -0.027]),
        "task": "pick up the red block and place it on the white plate",
    }

    batch = preprocessor(observation)
    assert batch[OBS_LANGUAGE_TOKENS].shape == (1, 200)
    assert batch[OBS_LANGUAGE_ATTENTION_MASK].dtype == torch.bool

    chunk = postprocessor(policy.predict_action_chunk(batch))

    assert chunk.shape == (1, 50, 7)
    assert torch.isfinite(chunk).all()
    # Actions live in the normalized [-1, 1] control space of the demonstrations.
    assert chunk.abs().max() < 3.0


@pytest.mark.slow
def test_action_queue_triggers_one_inference_per_n_action_steps():
    from mrs.policies import make_policy

    policy, preprocessor, _ = make_policy(
        CHECKPOINT, device="cpu", config_overrides={"n_action_steps": 4}
    )
    batch = preprocessor(
        {
            "observation.images.image": torch.rand(3, 256, 256),
            "observation.images.image2": torch.rand(3, 256, 256),
            "observation.state": torch.zeros(8),
            "task": "pick up the red block",
        }
    )

    policy.reset()
    assert policy.pending_actions == 0

    first = policy.select_action(batch)
    assert first.shape == (1, 7)
    assert policy.pending_actions == 3  # one popped, three cached

    for _ in range(3):
        policy.select_action(batch)
    assert policy.pending_actions == 0


@pytest.mark.slow
def test_a_checkpoint_without_statistics_is_rejected_rather_than_run_unnormalized():
    """Base checkpoints ship an empty feature map; a no-op normalizer would send
    raw values into a model that discretizes state over [-1, 1]."""
    from mrs.policies.factory import _validate_normalization
    from mrs.processor import NormalizerProcessorStep, PolicyProcessorPipeline

    config = PreTrainedConfig.from_pretrained(CHECKPOINT)
    empty = PolicyProcessorPipeline.make_input(
        [NormalizerProcessorStep(features={}, norm_map={"STATE": "QUANTILES"})]
    )
    with pytest.raises(ValueError, match="no normalization statistics"):
        _validate_normalization(config, empty)
