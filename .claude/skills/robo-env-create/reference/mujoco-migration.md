# Blender → MuJoCo migration

The migration is deliberately one-way and lossy. Blender keeps shading,
modifiers and non-convex detail. What crosses over is what MuJoCo can simulate:
transforms, extents, collision proxies, mass, friction, joints, actuators, and
the `mrs_*` properties that say which is which.

```
Blender scene ──export_scene_graph──> scene_graph.json ──build_env.py──> envs/<slug>/spec.json
                                                                              │
                                                              mrs.envs.scenegen.load_env
                                                                              ↓
                                                                    compiled MjModel
```

`build_env.py` never imports `bpy`. A bad migration can be re-run against the
same graph without going back to Blender, and the conversion is testable
off-line.

## What the graph carries

Per object: world position, world quaternion `(w,x,y,z)`, scale, **local
dimensions**, world axis-aligned bounds, parent, and every `mrs_*` property.

The distinction between `dimensions` and `extent` matters. `dimensions` is the
object's own extent along its local axes and is what becomes the MuJoCo geom
size. `extent` is the world axis-aligned bounding box, which differs the moment
an object is rotated and is used only for the overlap and clearance checks. A
migration that used `extent` for size would silently inflate every rotated
object.

Blender full extents become MuJoCo half-extents. Blender's cylinder primitive
extends along local +z, which matches MuJoCo's, so cylinders survive rotation
correctly.

## Two transform traps, both silent

Both of these produce a model that compiles cleanly, validates, and is wrong.
Both are fixed in the tooling and covered by `tests/test_blender_migration.py`;
they are written down because anything that touches the conversion can
reintroduce them.

**Nested bodies are parent-relative.** Blender exports *world* transforms, but
MuJoCo interprets a nested body's `pos` and `quat` relative to its parent.
Passing world values straight through applies the parent transform twice — a
drawer inside a cabinet at (0, −0.6, 0.45) lands 0.6 m further out and 0.45 m
higher. `build_env.py` converts with `_to_parent_frame`, which also undoes the
parent's rotation. Anything using `parent_body` depends on this.

**Mesh OBJs must be exported in local space.** `bpy.ops.wm.obj_export` writes
world-space vertices; MuJoCo then applies the body transform on top, placing
the mesh at exactly twice its position. `_export_obj` neutralises translation
and rotation for the duration of the export while keeping scale baked in, since
the MuJoCo mesh asset is registered at scale (1, 1, 1), and restores the
object's pose afterwards.

The check that catches both: compile the model, then compare each body's
world-space geom bounding box against the `bounds_min`/`bounds_max` the graph
recorded. Position-only comparison misses a dropped rotation and a wrong size;
the AABB comparison catches all three.

## Robot placement

The empty tagged `robot_mount` becomes `RobotSpec.mount_pos/mount_quat`. Its
`robot` property selects from `mrs/envs/scenegen/robots.py`, which holds the
Menagerie-specific naming — joint and actuator names, hand body, gripper
convention, grasp-site offset.

Only `panda` is registered, verified against
`.cache/mujoco_menagerie/franka_emika_panda/panda.xml`. Menagerie's `fr3` is
deliberately *not* registered: it ships as a bare arm with no hand body and no
gripper actuator, so it cannot run a manipulation task unmodified. Adding a
robot is a small explicit act — `ROBOT_RECIPE` in that file is the procedure.
Do not guess actuator names or gripper polarity; a wrong polarity fails
silently by releasing when it should grip.

The Panda is attached exactly as `mrs/envs/scene.py` does it: load the
unmodified Menagerie model, add the grasp site and wrist camera to its `hand`
body programmatically, attach under a prefixed frame. The grasp site sits
103.4 mm along the hand's +z — the point robosuite reports as the
end-effector, and therefore what the pi0.5 checkpoint's state statistics
describe.

`home_qpos` defaults to `(0, 0.10, 0, −2.45, 0, 2.55, 0.785)`, not Menagerie's
`home` keyframe. It places the gripper inside the end-effector pose band the
LIBERO-finetuned checkpoint was normalised over, so observations are
interpolation rather than extrapolation.

## The observation contract

Preserved exactly, so a released checkpoint drops in without an adapter:

```
observation.images.image   uint8 [3, 256, 256]   scene camera
observation.images.image2  uint8 [3, 256, 256]   wrist camera
observation.state          float32 [8]           eef xyz, axis-angle, finger, -finger
action                     float32 [7]           dx dy dz drx dry drz gripper, in [-1, 1]
```

A scene therefore **must** have one camera with `role='scene'` and one with
`role='wrist'`. If the Blender scene defines neither, `build_env.py` supplies
the reference placements rather than failing — check the result rather than
assuming it framed your task.

## Validator failures and what they mean

**`compile`** — MjSpec rejected the scene. Usually a duplicate body name, a
`parent_body` naming something that does not exist, or a body tagged `hinged`
with no joint axis.

**`layout_penetration`** — two bodies overlap by more than 2 mm at t=0, before
physics runs. This is the one to take most seriously. MuJoCo responds to the
resulting enormous acceleration by emitting `BADQACC` and auto-resetting, so
the symptom is not a crash but a scene that never moves. Almost always: a
fixture placed by centre when you meant by surface, or two free bodies whose
spawn ranges overlap.

**`settled_penetration`** — overlap greater than 6 mm *after* settling. The
looser threshold is deliberate: contact softness legitimately compresses a
resting contact by a millimetre or two, and only gross overlap indicates a
body sunk into another.

**`reachability`** — a manipulable body sits outside the commanded workspace
box. Commands outside it are clamped by the leash, so the arm silently stops
short instead of failing loudly. Move the object, or move the robot mount.

**`scene_camera_framing`** — a task body is outside the policy camera's
frustum. The policy sees only what that camera sees; a perfectly built scene
the camera cannot see is a failed scene.

**`stability`** — divergence during 200 idle steps. Read `dynamics.md`; it is
nearly always a velocity actuator with the explicit integrator, or missing
armature.

**`dynamics`** — a conveyor or belt that transported nothing.

**`observation_contract`** — image shape or dtype wrong. This one matters
disproportionately: the pi0.5 stack normalises rather than validates, so wrong
data does not raise, it just produces garbage actions.

## Re-running

The pipeline is idempotent. Re-export, re-migrate and re-validate as often as
you like; `build_env.py` overwrites `spec.json` and copies meshes fresh.

What it does **not** preserve is hand-editing of `spec.json`. If you edit the
spec directly and then re-migrate, your edits are gone — which is why
`robo-task-define` owns the `success` block and runs *after* migration is
finished.
