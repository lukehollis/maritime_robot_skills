# Authoring a robot scene in Blender

Blender is the authoring surface. MuJoCo is the runtime. The bridge is a set of
`mrs_*` custom properties, because a mesh alone cannot say whether a box is a
table, an envelope, or a drawer front.

## Session setup

`execute_blender_code` evaluates every call in a fresh namespace, so helpers
defined with `exec` vanish between calls. Import the kit as a module instead —
`sys.modules` persists for the life of the Blender process:

```python
import sys; sys.path.insert(0, '/absolute/path/to/.claude/scripts/env')
import importlib, blender_kit; importlib.reload(blender_kit)
from blender_kit import *
```

Use `importlib.reload` only when the kit itself changed; the two-line import
alone is enough afterwards and costs nothing.

`mrs_reset_scene()` clears the file and forces metric units at scale 1.0. Run it
before building. Blender's default scene contains a cube, a camera and a light
that will otherwise migrate into your workcell.

## The role property

`mrs_role` decides what an object becomes in MuJoCo. It is the single most
important tag and there is no safe way to infer it from geometry.

| role | becomes | use for |
|---|---|---|
| `static` | welded body, collidable | tables, bins, walls, fixtures, rails |
| `free` | free joint, 6 DoF | anything the robot manipulates |
| `hinged` | revolute joint | doors, lids, flaps, rollers |
| `sliding` | prismatic joint | drawers, pushers, linear stages |
| `mocap` | kinematic body, no DoF | scripted movers, moving obstacles |
| `robot_mount` | the arm's base frame | exactly one empty per scene |
| `camera` | a named camera | policy or inspection viewpoints |
| `decor` | visible, no collision | table legs, background, signage |
| `ignore` | nothing | construction guides, conveyor markers |

An untagged object exports as `static`. That default is deliberately the boring
one: it will be visible and solid, and nothing will try to actuate it.

## Property reference

Everything is passed as a keyword to `add_*` or `tag()` and stored as `mrs_<key>`.

**Physics** — `mass` (kg), `density` (kg/m³, ignored if `mass` is set),
`friction` (3 floats: sliding, torsional, rolling), `condim` (3 or 4; use 4 when
torsional friction matters, e.g. a flat part that should not spin in the
gripper), `solref` (2 floats; `[0.01, 1.0]` is a slightly softer, more forgiving
contact than the `[0.02, 1.0]` default).

**Collision** — `collision`: `box` | `cylinder` | `sphere` | `capsule` | `mesh`
| `none`. Primitives are always preferable: they are exact, cheap, and never
produce the degenerate contacts a concave mesh does. Reach for `mesh` only when
the shape genuinely drives the task, and expect MuJoCo to use its convex hull.
`none` keeps the object visible but removes it from the contact solve.

**Articulation** — `joint_axis` (3 floats), `joint_range` (2 floats),
`joint_damping`, `joint_stiffness`, `joint_springref`, `joint_armature`, and
`parent_body` to nest one body inside another. A drawer is a `sliding` body
whose `parent_body` is the static cabinet.

**Actuation** — `actuator`: `position` | `velocity` | `intvelocity` | `motor`,
plus `actuator_kp`, `actuator_kv`, `actuator_ctrlrange`, `actuator_default`.

**Spawn randomisation** — `spawn_x`, `spawn_y`, `spawn_z`, `spawn_yaw`, each a
pair resampled on every `reset()`. Absent axes hold the authored pose.

**Task metadata** — `tags`, a comma-separated string. Success predicates and
the evaluation stage select on these, so name them for what they mean
(`envelope,size_large`) rather than for where they sit.

## Composite helpers

`add_table(name, size, top_z=...)` places the table so its **top surface** lands
exactly on `top_z`. Authoring by surface height rather than centre height
removes the most common off-by-a-half-thickness error in these scenes.

`add_bin(name, inner=(w, d), depth=..., location=...)` builds a floor plus four
walls. A single box with a hole is not expressible in MuJoCo primitives, and a
concave mesh bin produces far worse contacts than five boxes.

`add_conveyor_marker(...)` draws a flat proxy slab and records the parameters;
the actual roller bed is generated at migration time. Do not model rollers by
hand — see `dynamics.md`.

## Placement rules that save a rebuild

**Everything the robot touches must be inside the workspace box.** The default
is x ∈ [−0.32, 0.20], y ∈ [−0.36, 0.36], z ∈ [0.645, 1.00], relative to a base
at (−0.56, 0, 0.63). Commands outside it are silently clamped, so an
out-of-reach bin does not fail loudly — the arm just stops short. The validator
checks this; checking it while you lay out is cheaper.

**Parts rest on surfaces, not near them.** Set a free body's centre to
`surface_z + half_thickness + 0.002`. The 2 mm is deliberate: touching exactly
produces a contact at t=0, and the validator flags overlaps.

**Give every free body its own lane.** Free bodies with overlapping spawn
ranges will eventually be sampled into each other, and interpenetration at
t=0 diverges the first step. Space lane centres by more than the sum of
adjacent half-widths, and keep the jitter inside the remaining clearance.

**Leave the approach clear.** The gripper needs roughly 15 cm of vertical
clearance above a part. Bins with tall walls are stylish and unpickable.

## Verification helpers

`measure()` — position and extent of every tagged object.
`check_overlaps()` — bounding-box overlaps. Over-reports (a bin legitimately
overlaps its own walls' boxes); treat hits as things to look at.
`check_resting(surface_z)` — free bodies whose underside is not on the surface.
`summary()` — one line per object, cheap after every edit.

These answer questions renders cannot. Renders answer questions these cannot.
Use both; see `inspection-loop.md`.
