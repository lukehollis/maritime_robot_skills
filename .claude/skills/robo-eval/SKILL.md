---
name: robo-eval
description: Stage 3 of policy evaluation. Run closed-loop rollouts of a deployed policy against a generated environment, record videos and per-episode traces, break results down by failure mode, and write a report comparing the policy against the scripted-expert ceiling. Use once robo-env-create, robo-policy-deploy and robo-task-define have all completed for an environment.
argument-hint: [env-slug] [--episodes N] [--seed N] [--video] [--compare scripted]
allowed-tools: Read, Write, Edit, Glob, Bash
---

> **STUB.** The rollout machinery exists and works; the reporting, sweep and
> failure-attribution layers are not written. Follow the manual procedure and
> report exactly what you ran.

Evaluate the policy recorded in `envs/$0/policy.json` against `envs/$0`.

## What already works

`mrs.rollout.evaluate` runs closed-loop episodes against anything exposing the
standard contract and returns per-episode results plus a summary. It handles
action chunking correctly: the policy refills its own queue, so the model runs
once per `n_action_steps` while the loop steps every control tick.

```python
from mrs.envs.scenegen import load_env
from mrs.policies import make_policy
from mrs.rollout import evaluate

env = load_env("envs/$0")
policy, pre, post = make_policy(checkpoint, device=device)
results, summary = evaluate(env, policy, pre, post, episodes=10, record_video=True)
```

`mrs/scripts/eval.py` is the reference driver, including video writing and
`summary.json`.

## What this skill must add

1. **Read the pipeline's own artifacts** — `envs/$0/spec.json`,
   `policy.json`, `task.json` — instead of taking parameters again on the
   command line.
2. **Failure attribution.** `info["failure_modes"]` already carries the
   triggered terms; aggregate them into a table. This is the deliverable that
   makes an evaluation useful.
3. **The expert comparison.** Always report the scripted expert's rate on the
   same seeds beside the policy's. A policy at 40 % against an expert at 100 %
   and one at 40 % against an expert at 45 % are different findings.
4. **Seed discipline.** Same seeds across compared policies, reported
   explicitly. Different seeds make two numbers incomparable.
5. **Cost.** Wall time per episode, inference calls, seconds per inference.
6. **Video for failures at minimum.** A failure you have not watched is a
   failure you have not diagnosed.
7. **Write `envs/$0/eval/<timestamp>/`** with `summary.json`, per-episode
   traces and videos.

## Reporting honestly

Zero-shot success on a generated scene may well be **0 %**. `docs/findings.md`
records exactly that for the LIBERO-finetuned pi0.5 checkpoint on the
hand-written pick-place scene, with every weight accounted for and a scripted
expert solving it 12/12. That is a finding about transfer, not a bug to
engineer around.

What makes a zero informative:

- the expert's rate on the same seeds, proving the task is solvable;
- the failure-mode breakdown, showing *how* it fails — never grasping is a
  different diagnosis from grasping and misplacing;
- confirmation the observation contract matched (image size, key names,
  rotation convention);
- episode count, so the reader knows 0/5 from 0/50.

Do not tune the environment until the policy succeeds. That converts an
evaluation into a demonstration, and silently: every check still passes, the
video looks good, and the number no longer measures anything. If the scene
needs changing, say what changed and re-run everything.

## Sweeps

The environment parameters worth sweeping — because they change the task rather
than the scenery — are conveyor `speed` and `duty`, `spawn_range` width,
`n_action_steps`, and part size. Each needs its own environment package or an
explicit override recorded in the report.

## Hand-off

This is the last stage. The deliverable is a report a person can act on:
success rate with its expert ceiling, failure modes ranked by frequency, cost,
and the exact commands to reproduce it.
