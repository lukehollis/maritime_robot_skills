# mrs — Panda pick-and-place + pi0.5 inference

A MuJoCo pick-and-place environment on a Franka Emika Panda, and a self-contained
PyTorch implementation of **pi0.5** that loads the released checkpoints and runs
closed-loop inference against it.

The architecture mirrors LeRobot's — the same config / processor-pipeline /
policy layering, and the same on-disk checkpoint layout — but none of the code is
LeRobot's, and the package does not depend on it. Published LeRobot pi0.5
checkpoints load here unmodified, weight-for-weight.

> **Zero-shot success is 0%.** The stack is verified end to end (every weight
> accounted for, expert solves the task 12/12, ~1 s per action chunk), but the
> LIBERO-finetuned pi0.5 checkpoint does not transfer to this scene. See
> [`docs/findings.md`](docs/findings.md) for the measurements. Closing that gap
> is a fine-tuning problem; the training loss and a demonstration source are
> included, the training loop is not.

## Install

```bash
pip install -e .
```

MuJoCo Menagerie (~2 GB, for the Panda model) is cloned into `.cache/` on first
use. Override the location with `MRS_ASSET_DIR`.

The PaliGemma tokenizer lives in a repo gated behind the Gemma licence. Either
accept it and authenticate:

```bash
hf auth login          # after accepting at huggingface.co/google/paligemma-3b-pt-224
```

or do nothing — an ungated mirror is used automatically, and is checked against
the canonical tokenizer's vocabulary size and special-token ids before use.

## Run

```bash
# The privileged-state expert: proves the task and the controller (fast, no weights)
python -m mrs.scripts.eval --policy scripted --episodes 10 --video

# pi0.5 (downloads ~7.5 GB on first run)
python -m mrs.scripts.eval --policy pi05 --episodes 5 --n-action-steps 10 --video

# Look at what the cameras see
python -m mrs.scripts.visualize --out outputs/cameras.png
```

```bash
pytest -m "not slow"    # 28 tests, ~10 s
pytest                  # adds 4 tests that exercise the real checkpoint
```

## The interface

Chosen to match the robosuite/LIBERO setup the pi0.5 checkpoints were fine-tuned
on, so a released checkpoint drops in without an adapter.

| | |
|---|---|
| `observation.images.image` | `uint8 [3, 256, 256]` — fixed third-person camera |
| `observation.images.image2` | `uint8 [3, 256, 256]` — wrist camera |
| `observation.state` | `float32 [8]` — eef xyz, eef axis-angle, finger qpos, −finger qpos |
| `action` | `float32 [7]` — `dx dy dz drx dry drz gripper`, each in [−1, 1] |

One unit of translation is 5 cm, one unit of rotation is 0.5 rad, and the gripper
follows robosuite's sign convention (+1 closes). Control runs at 20 Hz, matching
the demonstrations.

Axis-angle uses robosuite's `2·acos(w)` form, which lands in [0, 2π] rather than
wrapping to [−π, π] — the checkpoint's normalization statistics assume it.

The workcell geometry is not arbitrary: it places the reachable workspace inside
the end-effector pose distribution recorded in the checkpoint's own
normalization statistics (x ∈ [−0.19, 0.05], y ∈ [−0.14, 0.22], z ∈ [0.64, 0.88]),
so observations are interpolation rather than extrapolation for the policy. The
home pose is asserted to sit inside that band in `tests/test_env.py`.

## How pi0.5 runs

Two phases per action chunk.

**Prefix prefill.** Camera images go through SigLIP and are projected to the
language model's width. The prompt — the task text *plus the robot state,
discretized into 256 bins and written out as integers* — is embedded from the
token table. The 2 B Gemma runs once over this prefix with full bidirectional
attention, and every layer's keys and values are cached.

pi0.5 has no `state_proj`: proprioception is text. That is why the normalizer
must run before the prompt is built — the bins cover [−1, 1], which is the range
normalization produces.

**Flow-matching denoise.** A `(50, 32)` Gaussian sample is integrated from `t=1`
to `t=0` in 10 Euler steps. Each step re-runs only the 300 M action expert,
attending into the cached prefix. The timestep conditions the expert through
adaptive RMSNorm — the norms predict `(scale, shift, gate)` from the timestep
embedding instead of holding a learned gain — so no timestep token is needed.

Both stacks share one attention operation per layer: their queries, keys and
values are concatenated so the action tokens can see the prefix, while a
block-structured mask keeps the prefix from seeing the actions.

## Layout

```
mrs/
  types.py, constants.py       feature types, transition keys, canonical key strings
  configs/policies.py          PreTrainedConfig base; deserializes published config.json
  processor/
    pipeline.py                ProcessorStep, registry, save/load of pipelines
    normalize.py               MEAN_STD / MIN_MAX / QUANTILES / IDENTITY
    steps.py                   rename, batch, device, tokenize (+ gated-repo fallback)
  policies/
    pretrained.py              PreTrainedPolicy: select_action, forward, from_pretrained
    pi_gemma.py                adaptive RMSNorm, gated residuals, joint attention, KV cache
    common/                    flow matching, attention masks, resize-with-pad
    pi05/                      config, model, and the state-into-prompt processor
  envs/
    scene.py                   builds the workcell with MjSpec
    controllers.py             damped-least-squares differential IK
    panda_pick_place.py        the environment
    scripted_policy.py         privileged-state expert / demonstration source
  rollout.py                   closed-loop evaluation, env↔policy bridge
  scripts/eval.py, visualize.py
```

The pieces the environment does not need (`transformers`, the checkpoint) are
imported lazily, so `mrs.envs` works on its own.

## Using a different checkpoint

```bash
python -m mrs.scripts.eval --checkpoint lerobot/pi05_base --episodes 3
```

Any pi0.5 checkpoint following the LeRobot layout works. `pi05_base` ships an
empty feature map — it is a base model meant for fine-tuning — which would make
its normalizer a silent no-op. `make_policy` checks for this at load time and
raises, rather than letting raw observations reach a model that discretizes
state over [−1, 1].
