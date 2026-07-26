---
name: robo-env-create
description: Stage 0 of policy evaluation. Turn a human task description ("evaluate pi0.5 on a Franka sorting three envelope sizes") into a validated, runnable MuJoCo environment — authored visually in Blender through blender-mcp, inspected and corrected over a render/critique loop, then migrated to MuJoCo with the robot placed from mujoco_menagerie. Use whenever a new evaluation scene, workcell, or task environment is needed, including dynamic scenes with conveyors, turntables, moving obstacles or animated fixtures.
argument-hint: [env-slug] [task description, robot, objects, dynamics]
allowed-tools: Read, Write, Edit, Glob, Bash, mcp__blender__execute_blender_code, mcp__blender__get_viewport_screenshot, mcp__blender__get_scene_info
---

Build environment `$0` from the task description in `$ARGUMENTS`.

You are producing **one artifact**: `envs/$0/spec.json`, a scene specification
that `mrs.envs.scenegen.load_env` compiles into a MuJoCo environment with the
same observation and action contract as the hand-written
`mrs.envs.panda_pick_place`. Everything else — the Blender scene, the renders,
the scene graph — is working material.

Read `.claude/rules/project.md` for the pipeline and file conventions. The four
reference documents in `reference/` beside this file are the detail:

| file | read it when |
|---|---|
| `blender-authoring.md` | building or editing the Blender scene |
| `dynamics.md` | anything in the scene moves |
| `mujoco-migration.md` | exporting, migrating, or debugging the compiled model |
| `inspection-loop.md` | running the three review passes |

## Before anything else

If `$0` is missing, ask for the environment slug. If the task description does
not say what the robot must *do*, ask — a scene without a task is not
gradeable, and you will build the wrong geometry.

Do not ask about numbers you can choose. Table height, bin spacing, belt speed
and camera placement are yours to pick; the defaults in
`mrs/envs/scenegen/spec.py` are calibrated against the pi0.5 checkpoint's
workspace statistics and are the right starting point.

Check the prerequisites once, and stop if either fails:

```bash
python3 -c "import mujoco, mrs.envs.scenegen; print('mujoco', mujoco.__version__)"
ls .cache/mujoco_menagerie/franka_emika_panda/panda.xml
```

Blender must be running with a GUI and the BlenderMCP addon connected. The
addon refuses to serve in background mode, and `mcp__blender__get_scene_info`
is the cheapest way to confirm the socket is live.

## 1. Write the brief

Before touching Blender, write `envs/$0/brief.md`: the task sentence the policy
will receive, the robot, every manipulable object with its real-world
dimensions, the fixtures, what moves, and what counts as success and failure.

This is not ceremony. Envelope sizes, bin openings and belt speed determine the
geometry, and discovering halfway through that "three sizes of envelope" means
three *bins* as well as three *parts* costs a full rebuild.

Ground the dimensions in reality. A US letter envelope is 241 × 105 mm; a
"small" mail item is nearer 90 × 64 mm. Guessed sizes produce scenes that look
plausible and grade nonsensically.

## 2. Author the scene in Blender

Load the kit once per Blender session. It is imported as a module, not `exec`'d,
because `execute_blender_code` runs every call in a fresh namespace — an
`exec`'d helper is gone by the next call, while `sys.modules` persists.

```python
import sys; sys.path.insert(0, '/absolute/path/to/.claude/scripts/env')
import importlib, blender_kit; importlib.reload(blender_kit)
from blender_kit import *
mrs_reset_scene()
```

Every subsequent call starts with the two-line import (cheap — the module is
cached) and then builds geometry with `add_table`, `add_bin`, `add_box`,
`add_cylinder`, `add_conveyor_marker`, `add_empty`.

Tag as you build, never afterwards. An object's `mrs_role` is what decides
whether it becomes a welded fixture, a free rigid body, or a hinge — and
untagged objects migrate as static decor, which is a silent, plausible-looking
wrong answer. `blender-authoring.md` has the full property table.

The arm's mount is an empty tagged `robot_mount`; it is what fixes where the
Panda lands and therefore what the whole layout must be reachable from.

## 3. Inspect and edit — three passes

Run the loop in `inspection-loop.md` exactly three times before exporting. Each
pass is: render every angle, *look at the images*, measure what the images made
you suspect, then edit.

```python
render_views('/tmp/$0/pass1', resolution=420)   # six angles
check_overlaps()                                 # bodies sharing space
check_resting(surface_z=0.685)                   # parts floating or sunk
measure()                                        # positions and extents
```

Read the rendered PNGs. Do not skip to the numbers: `check_overlaps` cannot
tell you the bins are unreachably far from the arm, that the belt runs the
wrong way, or that the scene is lit so flatly the policy camera sees a grey
wash. Only looking does.

Three passes is a floor, not a target. Keep going while any pass still changes
something.

## 4. Export and migrate

```python
export_scene_graph('/tmp/$0/scene_graph.json')
```

```bash
python3 .claude/scripts/env/build_env.py \
  --graph /tmp/$0/scene_graph.json \
  --out envs/$0 \
  --task "<the task sentence from the brief>"
```

The graph carries world transforms, local extents, every `mrs_*` property and
any baked animation. `build_env.py` never imports `bpy`, so if a migration is
wrong you can re-run it against the same graph without touching Blender.

## 5. Validate — this is the gate

```bash
python3 .claude/scripts/env/validate_env.py envs/$0 --sheet
```

Non-zero exit means the environment is not finished. Fix and re-run; do not
report an environment that fails its own validator, and do not weaken a check
to make it pass.

Then look at `envs/$0/cameras.png`. It shows what the policy's scene and wrist
cameras actually see, which is the only view that matters — a scene that is
beautiful from a review angle and unreadable from `agentview` is a failed
scene.

Each check maps to a specific failure documented in `mujoco-migration.md`.
`layout_penetration` is the one to take most seriously: overlapping bodies at
t=0 make MuJoCo emit a huge acceleration and auto-reset, so the symptom is not
a crash but a scene that mysteriously never moves.

## 6. Watch it run

The validator proves the scene is sound. It does not prove the task is
*doable* — a layout can be perfectly reachable and still hold parts the
gripper physically cannot pick.

```bash
mjpython .claude/scripts/env/watch_env.py envs/$0
```

This drives the scene with `ScriptedSorter` and loops with a new seed each
episode. macOS needs `mjpython`, not `python3`. Use `--policy idle` to watch
the scene's own dynamics with the arm parked.

If the expert cannot complete the task, the environment is not finished,
however green the validator is. The commonest cause by far is parts that are
too thin to grasp: a parallel jaw needs the finger pads to straddle the
object's mid-height, and a flat item lying on a surface leaves nowhere for them
to go. `envs/mail_sorting` is exactly this case, and `envs/parcel_sorting` is
its graspable counterpart.

## 7. Hand off

Report: the package path, body and dynamics counts, the validation table, the
contact sheet path, and the task sentence.

`spec.json` ships with an empty `success` block. That is expected — success and
failure predicates are `robo-task-define`'s job (stage 2), and it will edit this
same file. Say so explicitly rather than inventing success terms, unless the
user asked for them here.

## Worked reference

`envs/mail_sorting/` is a complete, validated example produced by exactly this
procedure: a Panda, a powered roller conveyor with a stop-and-go duty cycle and
an end stop, three envelope sizes in separate spawn lanes, three bins. Its
generator is `.claude/scripts/env/examples/mail_sorting.py`. Read it when a new
scene is structurally similar; copy its numbers only when the task is.

## Rules that are not negotiable

- **Metres, always.** `export_scene_graph` refuses to run if the Blender unit
  scale is not 1.0, because a scene authored in centimetres compiles into a
  model that looks right and behaves nothing like it.
- **Never hand-write `spec.json`.** Author in Blender, export, migrate. A spec
  edited by hand and a Blender scene that disagrees is a scene you cannot
  iterate on.
- **`implicitfast`, not Euler**, for any scene with a velocity actuator. The
  builder warns; heed it. See `dynamics.md` for why.
- **Report what failed.** A partially working environment described accurately
  is useful. One described as finished is not.
