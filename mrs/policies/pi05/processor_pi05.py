"""The pi0.5 pre/post-processing pipelines.

The distinctive step is :class:`Pi05PrepareStateTokenizerStep`: pi0.5 has no
continuous state encoder, so proprioception is discretized into 256 bins and
spliced into the text prompt. That step therefore *must* run after
normalization, which is what puts the state into the [-1, 1] range the bins
cover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mrs.constants import OBS_STATE, TASK
from mrs.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    ToCPUProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from mrs.types import EnvTransition, TransitionKey
from mrs.policies.pi05.configuration_pi05 import PI05Config

NUM_STATE_BINS = 256


@ProcessorStepRegistry.register("pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerStep(ProcessorStep):
    """Fold the discretized robot state into the task prompt.

    Produces `"Task: <task>, State: <b0> <b1> ...;\\nAction: "`, where each
    `b_i` is the index of a 256-wide uniform bin over [-1, 1].
    """

    max_state_dim: int = 32
    task_key: str = TASK

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = (transition.get(TransitionKey.OBSERVATION) or {}).get(OBS_STATE)
        if state is None:
            raise ValueError(f"pi0.5 requires {OBS_STATE!r} in the observation.")

        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        tasks = complementary.get(self.task_key)
        if tasks is None:
            raise ValueError(f"pi0.5 requires {self.task_key!r} in the complementary data.")
        if isinstance(tasks, str):
            tasks = [tasks]

        state_np = torch.as_tensor(state).detach().cpu().numpy()
        # Left edges of 256 uniform bins over [-1, 1]; digitize returns 1-based
        # indices, hence the -1.
        #
        # Deliberately not clamped. MEAN_STD normalization does not bound the
        # state to [-1, 1], so a value more than one standard deviation from the
        # mean lands outside the bin range and yields an index below 0 or above
        # 255. The reference implementation behaves identically, so training saw
        # the same out-of-range indices; clamping here would put tokens in the
        # prompt that the checkpoint was never trained on.
        bin_edges = np.linspace(-1, 1, NUM_STATE_BINS + 1)[:-1]
        discretized = np.digitize(state_np, bins=bin_edges) - 1

        prompts = []
        for i, task in enumerate(tasks):
            cleaned = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized[i]))
            prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")

        complementary[self.task_key] = prompts
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def get_config(self) -> dict:
        return {}


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Build the input and output pipelines for a pi0.5 policy.

    Input:  rename -> batch -> normalize -> build prompt -> tokenize -> to device
    Output: unnormalize -> to CPU
    """
    features = {**config.input_features, **config.output_features}
    norm_map = config.normalization_mapping

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, device=config.device)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, device="cpu")

    if dataset_stats:
        flat = {
            f"{key}.{stat}": torch.as_tensor(value)
            for key, sub in dataset_stats.items()
            for stat, value in sub.items()
        }
        normalizer.load_state_dict(flat)
        unnormalizer.load_state_dict(flat)

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        normalizer,
        Pi05PrepareStateTokenizerStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [unnormalizer, ToCPUProcessorStep()]

    return (
        PolicyProcessorPipeline.make_input(input_steps),
        PolicyProcessorPipeline.make_output(output_steps),
    )
