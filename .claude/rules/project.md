# Robot policy evaluation pipeline

Four stages turn a sentence — *"evaluate pi0.5 with a Franka Panda on a mail
sorting task with three envelope sizes, and report failure modes"* — into a
measured result.

| stage | skill | produces |
|---|---|---|
| 0 | `robo-env-create` | `envs/<slug>/spec.json` — a validated, runnable scene |
| 1 | `robo-policy-deploy` | `envs/<slug>/policy.json` — a loaded checkpoint bound to it |
| 2 | `robo-task-define` | success/failure predicates + `task.json` expert baseline |
| 3 | `robo-eval` | `envs/<slug>/eval/<timestamp>/` — rates, failure modes, video |

Stage 0 is built. Stages 1–3 are stubs: their contracts and traps are written
down, the automation is not. When you run one, do the manual procedure and say
plainly what you ran and what is missing.

## Layout

```
mrs/                          the library — hand-written reference implementation
  envs/panda_pick_place.py    the reference environment; generated ones mirror its contract
  envs/scenegen/              spec-driven scene generation (stage 0's runtime)
    spec.py                   SceneSpec: the durable description of a scene
    builder.py                SceneSpec -> MjModel
    dynamics.py               conveyors, turntables, movers, baked animation
    env.py                    SceneEnv: same obs/action contract as the reference
    success.py                declarative predicates
    robots.py                 Menagerie robot registry (panda verified)
  policies/                   pi0.5 inference stack
  rollout.py                  closed-loop evaluation

.claude/scripts/env/          stage 0 tooling
  blender_kit.py              runs INSIDE Blender: primitives, tagging, render, export
  build_env.py                scene_graph.json -> envs/<slug>/spec.json  (no bpy)
  validate_env.py             the gate: penetration, stability, reach, framing, contract
  watch_env.py                live interactive viewer, driven by the scripted expert
  examples/mail_sorting.py    a complete worked spec
  examples/parcel_sorting.py  the same cell with graspable parts

envs/<slug>/                  one environment package
  brief.md                    what was asked for
  spec.json                   THE artifact
  assets/                     exported meshes, if any
  cameras.png                 what the policy cameras see
  validation.json             the gate's report
```

## Conventions

**Metres, kilograms, seconds, radians.** Quaternions are `(w, x, y, z)`.
`export_scene_graph` refuses to run if Blender's unit scale is not 1.0.

**`spec.json` is generated, never hand-written.** Author in Blender, export,
migrate. The one exception is the `success` block, which stage 2 owns and edits
in place — which is why stage 2 runs after migration is finished.

**The observation contract is fixed** and every generated environment preserves
it, so released pi0.5 checkpoints load without an adapter:

```
observation.images.image   uint8 [3, 256, 256]
observation.images.image2  uint8 [3, 256, 256]
observation.state          float32 [8]
action                     float32 [7]  in [-1, 1]
```

Changing it is a decision about the whole pipeline, not a scene-level tweak.

**Run from the repo root.** `mrs` is imported from the working directory; it is
not pip-installed in this checkout. The scripts under `.claude/scripts/env/`
insert the repo root on `sys.path` themselves.

## Watching a scene run

```bash
mjpython .claude/scripts/env/watch_env.py envs/parcel_sorting
```

On macOS this needs `mjpython`, not `python3` — the interactive viewer must own
the main thread's UI run loop. Detection is via `mujoco.viewer._MJPYTHON`;
`sys.executable` still reports plain `python3` under mjpython, so testing it
rejects legitimate runs.

`--policy idle` holds the arm still so you can watch the scene's own dynamics;
`--speed 2` runs at twice real time.

Two things that make a working scene look broken in the viewer, both learned
the hard way:

- **Set `env.render_observations = False`.** Rendering the two policy cameras
  every control step dominates the cost and the viewer draws the scene itself.
- **The expert finishing is not the task completing.** Success requires the
  predicate to hold for `success_hold_steps` consecutive steps while parts
  settle, and that lag reaches 160 steps when a part is dropped into a bin.
  Ending the episode at `expert.done` reports correct runs as incomplete.

## Reachability of a task, not just a layout

A parallel-jaw gripper cannot pick a flat object off a surface — the Panda's
finger pads are centred on the grip site, so there is nowhere for them to go
but into the table. `envs/mail_sorting` (flat envelopes) is a valid scene that
the Panda cannot solve; `envs/parcel_sorting` is the same cell with parts thick
enough to grasp. `validate_env.py` checks that the layout is *reachable*; it
does not check that the parts are *graspable*. Prove that with the scripted
expert before evaluating a policy.

## Blender

Blender must be running with a GUI — the BlenderMCP addon refuses to serve in
background mode. The addon auto-starts its socket on port 9876 when Blender
launches.

`execute_blender_code` evaluates each call in a fresh namespace. Import
`blender_kit` as a module rather than `exec`-ing it; `sys.modules` persists
across calls and an `exec`'d helper does not.

## Reporting

Say what you measured, including when it is zero. `docs/findings.md` records
0 % zero-shot success for the LIBERO-finetuned pi0.5 checkpoint on the
hand-written scene, with the stack verified end to end and a scripted expert
solving it 12/12. A generated scene is further from the training distribution,
not closer.

Never tune an environment until the policy succeeds. That converts an
evaluation into a demonstration, and it does so silently — every check still
passes and the number stops meaning anything.
