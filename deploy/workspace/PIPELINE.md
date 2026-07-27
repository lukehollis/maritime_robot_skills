# The evaluation pipeline, and the machine it runs on

Generated from the repo on every boot. If this disagrees with your memory of
how something works, this file is right.

## The five stages

| stage | skill | produces |
|---|---|---|
| 0 | `robo-env-create` | `envs/<slug>/spec.json` — a validated, runnable scene |
| 1 | `robo-policy-deploy` | `envs/<slug>/policy.json` — a checkpoint bound to it |
| 2 | `robo-task-define` | success/failure predicates + a scripted-expert baseline |
| 3 | `robo-eval` | `envs/<slug>/eval/<timestamp>/` — rates, failure modes, video |
| 4 | `robo-report` | `envs/<slug>/eval/<timestamp>/report.html` — the deliverable |

Stage 0 and stage 4 are built. Stages 1–3 are stubs: their contracts and traps
are written down, the automation is not. When you run one, do the manual
procedure it describes and say plainly what you ran and what is missing.

An experiment is not finished until stage 4 has produced a report and you have
told the user what it says.

## Where things live

```
/data/lab/                working directory for everything you do
  envs/<slug>/            one environment package (brief, spec, assets, evals)
  runs/                   scratch space for a single experiment's intermediates
  jobs/<id>/              mrs-job bookkeeping: cmd, log, status, pid
  .cache/                 mujoco_menagerie, fetched per robot on demand
  .claude/                the pipeline's own skills, scripts and rules (read-only)
/data/hf/                 Hugging Face cache — checkpoints land here
/data/logs/               xvfb.log, blender.log, menagerie.log
```

`/data` is **10 GB total** and the pi0.5 checkpoint is 7.5 GB of it. Keep one
checkpoint at a time; `du -sh /data/hf` before downloading a second one, and
delete the old one deliberately rather than discovering the disk full halfway
through a rollout.

For the same reason the Menagerie clone is pruned to the robots this lab uses
(the full 82 would leave no room for a checkpoint). `.cache/mujoco_menagerie/
PRUNED.md` records what was removed and how to get it back — if a task needs a
robot that is not there, that file is the answer, not a bug.

## The container

- **No GPU.** 4 vCPU, 16 GB RAM. pi0.5 runs at roughly **20–30 s per action
  chunk** here against ~1 s on an accelerator. A 10-episode rollout at
  `n_action_steps=10` is on the order of **1.5–2 hours**. Plan around that; do
  not "quickly try 50 episodes".
- **Blender is already running** on the `:99` virtual display with the
  BlenderMCP addon connected on port 9876. The addon cannot run headless, which
  is why the X server exists. If the `blender` MCP tools stop responding, read
  `/data/logs/blender.log` and tell the user — do not try to relaunch Blender.
- **Rendering is software.** Blender's viewport is Mesa llvmpipe and MuJoCo
  renders through OSMesa (`MUJOCO_GL=osmesa`). Both work; both are slow. Prefer
  a few well-chosen renders over sweeping the camera.
- **Python is `/opt/venv/bin/python3`** — the one with `mujoco`, `torch` and
  `mrs` in it. Your shell has it first on PATH, so plain `python3` is correct.
  But Maritime's own `exec` channel uses a different PATH where `python3` is
  Debian's and imports of `mujoco` or `mrs` fail with `ModuleNotFoundError`. If
  you ever see that, you are running the wrong interpreter: use the absolute
  path, and do not conclude the package is missing.

## Long jobs

`maritime chat` is a request/response call, and a rollout outlives it. So
anything slow gets dispatched and watched, never awaited inline:

```bash
mrs-job start eval-mailsort -- python -m mrs.scripts.eval --policy pi05 --episodes 10 --video
mrs-job list
mrs-job status eval-mailsort
mrs-job log eval-mailsort -n 50
```

`start` returns immediately with a job id. The job keeps running between your
turns, so a good pattern is: dispatch, tell the user it started and roughly how
long it will take, then check `mrs-job status` when they next write — or poll it
yourself a few times if you are mid-task and have nothing else to do.

Never run a rollout or a checkpoint download in the foreground. If a foreground
command has been going for more than a minute or two, you have made a mistake:
kill it and re-dispatch through `mrs-job`.

## Reporting

```bash
mrs-report envs/<slug>/eval/<timestamp>
```

Renders `report.html` — a single self-contained file with the success rate
against the scripted-expert ceiling, the failure-mode breakdown, per-episode
timelines and the scene stills. Read it back to the user in prose: the number,
what dominated the failures, and whether the expert baseline makes the number
trustworthy. A report they have to open to understand is a report you have not
finished writing.
