---
name: robo-policy-deploy
description: Stage 1 of policy evaluation. Fetch a policy checkpoint (pi0.5 or another LeRobot-layout VLA) from Hugging Face or a local path, load it with its published processor pipelines, and bind it to a generated environment's observation and action contract. Use after robo-env-create has produced a validated environment and before robo-eval runs rollouts.
argument-hint: [env-slug] [checkpoint repo-id or path] [--n-action-steps N] [--device cuda|mps|cpu]
allowed-tools: Read, Write, Edit, Glob, Bash
---

> **STUB.** The contract, the checks and the failure modes below are settled;
> the automation is not written. Follow the manual procedure and report exactly
> what you ran.

Load policy `$1` and bind it to `envs/$0`.

## What already works

`mrs.policies.make_policy` does the real work and needs no wrapper:

```python
from mrs.policies import make_policy
policy, preprocessor, postprocessor = make_policy(
    "lerobot/pi05_libero_finetuned_v044", device="mps",
    config_overrides={"n_action_steps": 10},
)
```

It loads the checkpoint's *own* processor pipelines, so normalization
statistics and prompt format always match the weights. `mrs.rollout.evaluate`
then drives any environment exposing the standard contract — including one
produced by `robo-env-create`, which is the point of keeping that contract
fixed.

An end-to-end run against the hand-written scene already exists:

```bash
python -m mrs.scripts.eval --policy pi05 --episodes 5 --n-action-steps 10 --video
```

## What this skill must add

1. **Resolve the policy from a human description.** "pi 0.5" → a concrete
   checkpoint id. Maintain a small table of known-good checkpoints rather than
   guessing repo ids; a wrong id fails late and expensively.
2. **Fail fast on base checkpoints.** `make_policy` already raises when a
   checkpoint ships an empty feature map — `pi05_base` is published for
   fine-tuning and its normalizer would be a silent no-op. Surface that as a
   clear message, not a traceback.
3. **Bind to the generated env.** Confirm `spec.control.scene_image_key` and
   `wrist_image_key` match the checkpoint's expected feature names, and that
   `image_size` matches what it was trained on.
4. **Report the device and the cost.** Note parameters, chunk size, actions
   executed per inference, and measured seconds per inference before anyone
   launches a long evaluation.
5. **Write `envs/$0/policy.json`** recording checkpoint, device, overrides and
   resolved feature mapping, so `robo-eval` does not re-derive them.

## Known traps

- **`--rotate-images-180`.** openpi's LIBERO pipeline feeds images rotated;
  these environments render right-side-up. The flag exists on `mrs.scripts.eval`
  and materially changes results. Decide deliberately, record the choice.
- **Zero-shot transfer is not a given.** `docs/findings.md` records 0 % success
  for the LIBERO-finetuned pi0.5 checkpoint on the hand-written pick-place
  scene, with the stack verified end to end and a scripted expert solving it
  12/12. A generated scene is further from the training distribution, not
  closer. Report a zero honestly; it is a real measurement, not a bug to hide.
- **First run downloads ~7.5 GB.** Say so before starting.

## Hand-off

`robo-task-define` (stage 2) defines success. `robo-eval` (stage 3) runs the
rollouts. This skill only has to make the policy loadable and prove one
inference produces a finite 7-D action in the right range.
