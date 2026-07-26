# Dynamic scenes

A static pick-and-place scene tests whether a policy can grasp. A moving scene
tests whether it can grasp *in time* — and that is usually the interesting
question. A conveyor that stops and starts, a turntable that carries a part out
of reach, an obstacle that sweeps through the workspace: each turns a
one-shot problem into a timing problem, and policies that look identical on
static benchmarks separate immediately.

Everything here lives in `mrs/envs/scenegen/dynamics.py`. Each element has two
halves: a **build-time expansion** that adds bodies and actuators before the
model compiles, and a **run-time driver** that writes to `data` once per physics
step. Drivers run every substep, not every control step, so a belt advances
smoothly between the policy's 20 Hz decisions instead of teleporting.

## The numerical rule you must not skip

A velocity servo on a conveyor roller is a stiff damper. The roller's
rotational inertia is around 2 × 10⁻⁴ kg·m² and a useful gain is kv ≈ 1–10, so
the servo's time constant is far shorter than the 2 ms timestep. With the
explicit Euler integrator it overshoots, oscillates, and diverges within about
ten steps.

MuJoCo does not raise on divergence. It emits a `BADQACC` warning and
auto-resets the state. The symptom is therefore **a scene that silently never
moves**, which reads as "my conveyor parameters are wrong" and sends you
tuning speeds for an hour.

Two things fix it and the tooling applies both:

1. **`implicitfast`** integrates actuator damping implicitly. It is the default
   in `ControlSpec` and the builder emits a warning if you select `euler`
   alongside a velocity actuator.
2. **Armature** of roughly `kv × timestep` floors the effective rotor inertia.
   Macros set this automatically; `build_env.py` applies the same rule to
   hand-tagged velocity actuators.

If a dynamic scene behaves as though frozen, check
`data.warning[mjWARN_BADQACC].number` before you change anything else.
`validate_env.py`'s `stability` check does exactly this.

## Elements

### `roller_conveyor`

A real powered roller bed: N cylinders on hinges with velocity servos, plus
side rails and an end stop. Parts are carried by genuine rolling contact, so
they rotate, nudge each other, and queue against the stop for free.

```python
add_conveyor_marker('infeed', origin=(-0.10, -0.06, 0.66), direction='+y',
                    length=0.42, width=0.20, roller_radius=0.025, speed=0.05,
                    spacing=0.07, roller_mass=0.15, kv=4.0, rail_height=0.035,
                    end_stop=True, duty_period=9.0, duty_on_fraction=0.55)
```

`speed` is the belt surface speed in m/s; the driver converts it to ω = v / r.
The sign works out so a positive speed always drives along `direction` — the
roller axis is ẑ × d, which makes the top-surface velocity ω·r·d for any
horizontal direction.

`duty` makes it stop and go: `period=9.0, on_fraction=0.55` runs the belt for
the first 55 % of every nine-second cycle. This is the cheapest way to turn a
grasping task into a timing task.

`end_stop` (on by default) is not optional in practice. Without it, parts ride
off the downstream end and land on the floor, which looks like a physics bug.
Real infeeds queue parts against a backstop at the pick position, and that
queue — parts jostling while the belt slips underneath — is usually the
behaviour worth evaluating.

Rollers should be **smaller than the shortest part dimension**. A part shorter
than the roller pitch falls into the gap. Keep `spacing < 2.6 × roller_radius`
and check the smallest part against it.

### `belt_field`

A conveyor with no moving parts: free bodies inside a region have their
horizontal velocity relaxed toward a target. Far cheaper, and it never jams —
which is exactly why it is the wrong choice when jamming is a failure mode you
want to measure. Use it for background flow, or when the belt is a delivery
mechanism rather than part of the task.

### `turntable`

A disc on a hinge with a velocity servo. Parts on it orbit, which makes the
grasp pose time-varying in orientation as well as position.

### `mover`

A mocap body following a path — `harmonic` (sine along an axis) or `waypoints`
(piecewise linear, optionally looping). Mocap bodies are unaffected by contact:
a mover pushes the world without the world pushing back. That is what you want
for a moving obstacle, a passing human hand, or a fixture on an external axis,
and emphatically not what you want for anything the robot must grasp — it will
be immovable.

### `joint_cycle`

Drives an actuated hinge or slide through its range on a schedule, with a dwell
at each end: doors, gates, lids, indexing fixtures.

### `baked`

Replays animation authored in Blender. F-curves do not survive the trip, so
`export_scene_graph` samples every animated object into a time/pose table and
the driver interpolates it. Only `mocap` bodies can be driven this way — a body
with real degrees of freedom would fight the physics rather than follow the
curve.

To use it: animate the object in Blender as normal, tag it `role='mocap'`, and
set the scene frame range to cover the motion. The migration wires up the
driver automatically.

## Choosing

| you want | use |
|---|---|
| parts delivered, contact fidelity matters | `roller_conveyor` |
| parts delivered, throughput matters | `belt_field` |
| a timing constraint | any of the above with `duty` |
| a moving obstacle | `mover` |
| motion you drew by hand | `baked` on a mocap body |
| a fixture that opens and closes | `joint_cycle` |
| orientation that changes with time | `turntable` |

## Verifying motion

`validate_env.py` runs the scene for 200 idle control steps with the arm held
still and reports how far each free body travelled. A conveyor that transports
nothing is the check failing, and the usual causes are, in order: the part
spawned beside the belt rather than on it; the roller pitch is wider than the
part; the direction is right but the part is already against the end stop.
