"""Moving scene elements.

A dynamic element has two halves, and this module owns both so they cannot
drift apart:

* **expansion** — a macro that adds ordinary bodies, joints and actuators to the
  scene before it is compiled. A roller conveyor is not a special model
  primitive; it is sixteen cylinders on hinges with velocity servos.
* **a driver** — an object that writes to `data.ctrl`, `data.mocap_pos` or
  `data.qvel` once per physics step while the episode runs.

Some kinds use only one half. `mover` is a pure driver over a mocap body;
`turntable` is mostly expansion with a driver that just holds a speed.

Numerical note, learned the hard way. A velocity servo on a conveyor roller is
a stiff damper: the roller's rotational inertia is around 2e-4 kg m^2, while a
useful gain is kv ~ 1-10. With the explicit Euler integrator the servo
overshoots and the model diverges inside ten steps. Two things fix it, and this
module applies both: the scene is compiled with `implicitfast`, which
integrates actuator damping implicitly, and every driven joint gets an armature
of roughly `kv * timestep`, which floors the effective rotor inertia.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from mrs.envs.scenegen.spec import ActuatorSpec, BodySpec, JointSpec, SceneSpec

EXPANDING_KINDS = ("roller_conveyor", "turntable")
DRIVER_KINDS = ("roller_conveyor", "turntable", "mover", "belt_field", "joint_cycle",
                "baked", "severable")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        raise ValueError(f"Direction {vector!r} has no length.")
    return v / norm


def _direction(value) -> np.ndarray:
    """Accept `'+x'`, `'-y'` or an explicit vector, always returning a unit xy vector."""
    if isinstance(value, str):
        table = {
            "+x": (1, 0, 0), "x": (1, 0, 0), "-x": (-1, 0, 0),
            "+y": (0, 1, 0), "y": (0, 1, 0), "-y": (0, -1, 0),
        }
        key = value.lower().strip()
        if key not in table:
            raise ValueError(f"Belt direction {value!r} must be one of {sorted(table)} or a vector.")
        return np.array(table[key], dtype=float)
    direction = _unit(value)
    if abs(direction[2]) > 1e-6:
        raise ValueError("Conveyor and belt directions must lie in the horizontal plane.")
    return direction


def quat_from_z_to(target) -> list[float]:
    """Quaternion `(w, x, y, z)` rotating local +z onto `target`.

    Cylinders and capsules extend along their local z, so a roller lying across
    the belt needs this while its hinge axis stays in world coordinates.
    """
    z = np.array([0.0, 0.0, 1.0])
    v = _unit(target)
    dot = float(np.clip(np.dot(z, v), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    if dot < -1.0 + 1e-9:
        return [0.0, 1.0, 0.0, 0.0]  # 180 degrees about x
    axis = _unit(np.cross(z, v))
    angle = math.acos(dot)
    s = math.sin(angle / 2.0)
    return [math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s]


def _armature_for(kv: float, timestep: float) -> float:
    """Rotor inertia that keeps a `kv` velocity servo well-conditioned."""
    return max(kv * timestep, 1e-4)


# ---------------------------------------------------------------------------
# Build-time expansion
# ---------------------------------------------------------------------------


def expand(spec: SceneSpec, info) -> tuple[list[BodySpec], list[tuple[str, str, ActuatorSpec]]]:
    """Return `(bodies, extra_actuators)` with every macro resolved.

    The returned body list is the scene's own bodies plus everything the
    dynamic elements generated, in a single flat list the builder can consume
    without knowing what a conveyor is.
    """
    bodies = list(spec.bodies)
    extra: list[tuple[str, str, ActuatorSpec]] = []

    for element in spec.dynamics:
        if not element.enabled or element.kind not in EXPANDING_KINDS:
            continue
        if element.kind == "roller_conveyor":
            generated = _expand_roller_conveyor(element.name, element.params, spec)
        else:
            generated = _expand_turntable(element.name, element.params, spec)

        names = [b.name for b in generated]
        collisions = names and set(names) & {b.name for b in bodies}
        if collisions:
            raise ValueError(
                f"Dynamic element {element.name!r} generates body names that already exist: "
                f"{sorted(collisions)}"
            )
        info.expanded_bodies[element.name] = names
        bodies.extend(generated)

    return bodies, extra


def _expand_roller_conveyor(name: str, params: dict[str, Any], spec: SceneSpec) -> list[BodySpec]:
    """A powered roller bed.

    Real conveyors are modelled here rather than faked because the interaction
    that matters — a part being carried, rotating slightly, occasionally
    jamming against a neighbour — comes out of the contact solve for free.
    Use `belt_field` instead when throughput matters more than fidelity.
    """
    origin = np.asarray(params.get("origin", (0.0, 0.0, 0.70)), dtype=float)
    direction = _direction(params.get("direction", "+x"))
    length = float(params.get("length", 0.8))
    width = float(params.get("width", 0.4))
    radius = float(params.get("roller_radius", 0.03))
    spacing = float(params.get("spacing", radius * 2.6))
    mass = float(params.get("roller_mass", 0.25))
    friction = tuple(params.get("friction", (1.5, 0.01, 0.0005)))
    material = params.get("material")
    rgba = tuple(params["rgba"]) if "rgba" in params else (0.45, 0.47, 0.52, 1.0)
    kv = float(params.get("kv", 5.0))
    side_rails = bool(params.get("side_rails", True))
    end_stop = bool(params.get("end_stop", True))

    # Rollers turn about the horizontal axis perpendicular to travel. With
    # a = z_hat x d, the top-surface velocity works out to omega * radius * d,
    # so a positive command always drives the belt along `direction`.
    axis = _unit(np.cross([0.0, 0.0, 1.0], direction))
    geom_quat = quat_from_z_to(axis)

    count = max(int(round(length / spacing)), 2)
    armature = _armature_for(kv, spec.control.sim_timestep)

    bodies: list[BodySpec] = []
    for i in range(count):
        offset = (i - (count - 1) / 2.0) * spacing
        bodies.append(
            BodySpec(
                name=f"{name}_roller_{i:02d}",
                kind="hinged",
                shape="cylinder",
                size=(radius, width / 2.0),
                pos=tuple(origin + direction * offset),
                geom_quat=tuple(geom_quat),
                material=material,
                rgba=None if material else rgba,
                mass=mass,
                friction=friction,
                condim=4,
                solref=(0.01, 1.0),
                joint=JointSpec(type="hinge", axis=tuple(axis), damping=0.01, armature=armature),
                actuator=ActuatorSpec(kind="velocity", kv=kv, ctrlrange=(-80.0, 80.0)),
                tags=["conveyor", name],
            )
        )

    if side_rails:
        # Without rails a carried part walks sideways off the bed within a few
        # seconds, which reads as a physics bug rather than the intended task.
        rail_height = float(params.get("rail_height", 0.04))
        side = _unit(np.cross([0.0, 0.0, 1.0], direction))
        span = spacing * count / 2.0
        for sign, label in ((1.0, "left"), (-1.0, "right")):
            centre = origin + side * sign * (width / 2.0 + 0.012) + np.array([0.0, 0.0, radius])
            half = [0.0, 0.0, 0.0]
            half[0] = span if abs(direction[0]) > 0.5 else 0.01
            half[1] = span if abs(direction[1]) > 0.5 else 0.01
            half[2] = rail_height / 2.0
            bodies.append(
                BodySpec(
                    name=f"{name}_rail_{label}",
                    kind="static",
                    shape="box",
                    size=tuple(half),
                    pos=tuple(centre),
                    rgba=(0.30, 0.32, 0.36, 1.0),
                    friction=(0.4, 0.005, 0.0001),
                    tags=["conveyor_rail", name],
                )
            )

    if end_stop:
        # Without a stop, parts simply ride off the downstream end and land on
        # the floor, which reads as a broken scene. A real infeed queues parts
        # against a backstop at the pick position, and that queue — parts
        # nudging each other while the belt keeps slipping underneath — is
        # usually the interesting part of the task.
        stop_height = float(params.get("end_stop_height", 0.05))
        centre = origin + direction * (spacing * count / 2.0 + 0.015) + np.array(
            [0.0, 0.0, radius + stop_height / 2.0]
        )
        half = [0.0, 0.0, 0.0]
        half[0] = 0.01 if abs(direction[0]) > 0.5 else width / 2.0 + 0.02
        half[1] = 0.01 if abs(direction[1]) > 0.5 else width / 2.0 + 0.02
        half[2] = stop_height / 2.0
        bodies.append(
            BodySpec(
                name=f"{name}_end_stop",
                kind="static",
                shape="box",
                size=tuple(half),
                pos=tuple(centre),
                rgba=(0.30, 0.32, 0.36, 1.0),
                friction=(0.3, 0.005, 0.0001),
                tags=["conveyor_stop", name],
            )
        )

    return bodies


def _expand_turntable(name: str, params: dict[str, Any], spec: SceneSpec) -> list[BodySpec]:
    centre = tuple(params.get("center", (0.0, 0.0, 0.70)))
    radius = float(params.get("radius", 0.20))
    thickness = float(params.get("thickness", 0.01))
    mass = float(params.get("mass", 1.0))
    kv = float(params.get("kv", 20.0))
    return [
        BodySpec(
            name=f"{name}_disc",
            kind="hinged",
            shape="cylinder",
            size=(radius, thickness / 2.0),
            pos=centre,
            material=params.get("material"),
            rgba=tuple(params.get("rgba", (0.55, 0.56, 0.60, 1.0))),
            mass=mass,
            friction=tuple(params.get("friction", (1.4, 0.01, 0.0005))),
            condim=4,
            joint=JointSpec(
                type="hinge",
                axis=(0.0, 0.0, 1.0),
                damping=0.05,
                armature=_armature_for(kv, spec.control.sim_timestep),
            ),
            actuator=ActuatorSpec(kind="velocity", kv=kv, ctrlrange=(-30.0, 30.0)),
            tags=["turntable", name],
        )
    ]


# ---------------------------------------------------------------------------
# Run-time drivers
# ---------------------------------------------------------------------------


class Driver:
    """Writes to `data` once per physics step.

    The base deliberately declares no `name` attribute. Subclasses are
    dataclasses that declare `name: str` as their first field, and an
    inherited class attribute would be picked up as that field's default,
    forcing every field after it to carry one too.
    """

    def reset(self, model, data) -> None:  # pragma: no cover - default no-op
        pass

    def apply(self, model, data) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def state(self) -> dict[str, Any]:
        return {}


@dataclass
class ConveyorDriver(Driver):
    """Holds the roller servos at the commanded belt speed.

    `duty` turns the belt into a stop-and-go line: `{"period": 6.0,
    "on_fraction": 0.6}` runs it for the first 60% of every six-second cycle.
    A policy that has to time a pick against a moving line is a materially
    different evaluation from one that does not, which is the point.
    """

    name: str
    actuators: list[str]
    speed: float
    radius: float
    duty: dict[str, float] | None = None
    stop_body: str | None = None
    """Body whose arrival latches the belt off — a stop sensor at the station."""
    stop_at: float | None = None
    """Distance along `direction` at which that body counts as arrived."""
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    _ids: list[int] | None = None
    _stop_body_id: int = -1
    _latched: bool = False

    def reset(self, model, data) -> None:
        import mujoco

        self._ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.actuators
        ]
        missing = [n for n, i in zip(self.actuators, self._ids) if i < 0]
        if missing:
            raise ValueError(f"Conveyor {self.name!r} references unknown actuators: {missing}")

        self._latched = False
        self._stop_body_id = -1
        if self.stop_body is not None:
            self._stop_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.stop_body)
            if self._stop_body_id < 0:
                raise ValueError(
                    f"Conveyor {self.name!r} watches unknown stop_body {self.stop_body!r}."
                )

    def _running(self, t: float) -> bool:
        if not self.duty:
            return True
        period = float(self.duty.get("period", 0.0))
        if period <= 0:
            return True
        return (t % period) < period * float(self.duty.get("on_fraction", 0.5))

    def apply(self, model, data) -> None:
        # A powered roller that keeps driving under a part already pinned
        # against the end stop does not merely idle: the tangential force
        # levers a thin part upward until it slips, which launches it. A real
        # indexing line stops the belt on a sensor at the station, and this is
        # that sensor.
        if not self._latched and self._stop_body_id >= 0 and self.stop_at is not None:
            travel = float(np.dot(data.xpos[self._stop_body_id], np.asarray(self.direction)))
            if travel >= self.stop_at:
                self._latched = True

        omega = 0.0 if self._latched else (self.speed / self.radius) * self._running(data.time)
        for actuator_id in self._ids or ():
            data.ctrl[actuator_id] = omega

    def state(self) -> dict[str, Any]:
        return {"latched": self._latched, "speed": self.speed}


@dataclass
class TurntableDriver(Driver):
    name: str
    actuator: str
    angular_speed: float
    _id: int = -1

    def reset(self, model, data) -> None:
        import mujoco

        self._id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.actuator)
        if self._id < 0:
            raise ValueError(f"Turntable {self.name!r} references unknown actuator {self.actuator!r}.")

    def apply(self, model, data) -> None:
        data.ctrl[self._id] = self.angular_speed


@dataclass
class MoverDriver(Driver):
    """Scripted motion of a mocap body.

    Mocap bodies are unaffected by contact, so a mover pushes the world around
    without the world pushing back. That is what you want for a moving
    obstacle, a passing human hand, or a fixture on an external axis — and what
    you do not want for anything the robot is meant to grasp.
    """

    name: str
    body: str
    path: str = "harmonic"
    params: dict[str, Any] = None
    _mocap_id: int = -1
    _origin: np.ndarray | None = None

    def reset(self, model, data) -> None:
        import mujoco

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body)
        if body_id < 0:
            raise ValueError(f"Mover {self.name!r} references unknown body {self.body!r}.")
        self._mocap_id = int(model.body_mocapid[body_id])
        if self._mocap_id < 0:
            raise ValueError(
                f"Mover {self.name!r} targets body {self.body!r}, which is not a mocap body. "
                f"Set its kind to 'mocap' in the scene spec."
            )
        self._origin = np.asarray(model.body_pos[body_id], dtype=float).copy()

    def apply(self, model, data) -> None:
        params = self.params or {}
        t = data.time
        if self.path == "harmonic":
            axis = _unit(params.get("axis", (0.0, 1.0, 0.0)))
            amplitude = float(params.get("amplitude", 0.2))
            period = max(float(params.get("period", 4.0)), 1e-6)
            phase = float(params.get("phase", 0.0))
            offset = axis * amplitude * math.sin(2.0 * math.pi * t / period + phase)
            data.mocap_pos[self._mocap_id] = self._origin + offset
        elif self.path == "waypoints":
            data.mocap_pos[self._mocap_id] = self._waypoint_at(t, params)
        else:
            raise ValueError(f"Mover {self.name!r} has unknown path {self.path!r}.")

    def _waypoint_at(self, t: float, params: dict[str, Any]) -> np.ndarray:
        points = np.asarray(params["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Mover {self.name!r} needs `points` shaped (N, 3).")
        speed = float(params.get("speed", 0.2))
        loop = bool(params.get("loop", True))

        legs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if loop:
            legs = np.append(legs, np.linalg.norm(points[0] - points[-1]))
        total = float(legs.sum())
        if total < 1e-9:
            return points[0]

        distance = speed * t
        distance = distance % total if loop else min(distance, total)

        for index, leg in enumerate(legs):
            if distance <= leg or leg < 1e-12:
                start = points[index]
                end = points[(index + 1) % len(points)]
                alpha = 0.0 if leg < 1e-12 else distance / leg
                return start + (end - start) * alpha
            distance -= leg
        return points[-1]


@dataclass
class BeltFieldDriver(Driver):
    """A conveyor with no moving parts.

    Any free body whose origin sits inside the region has its horizontal
    velocity relaxed toward the belt velocity. Far cheaper than a roller bed
    and it never jams — which is exactly why it is the wrong choice when jamming
    is a failure mode you intend to measure.
    """

    name: str
    region_min: tuple[float, float, float]
    region_max: tuple[float, float, float]
    velocity: tuple[float, float, float]
    gain: float = 0.25
    bodies: list[str] | None = None
    _entries: list[tuple[int, int]] = None

    def reset(self, model, data) -> None:
        import mujoco

        entries = []
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            body_id = int(model.jnt_bodyid[joint_id])
            if self.bodies is not None:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if name not in self.bodies:
                    continue
            entries.append((body_id, int(model.jnt_dofadr[joint_id])))
        self._entries = entries

    def apply(self, model, data) -> None:
        low = np.asarray(self.region_min, dtype=float)
        high = np.asarray(self.region_max, dtype=float)
        target = np.asarray(self.velocity, dtype=float)

        for body_id, dof in self._entries or ():
            position = data.xpos[body_id]
            if np.any(position < low) or np.any(position > high):
                continue
            current = data.qvel[dof : dof + 3]
            # Relax rather than assign: a hard set fights the contact solver and
            # makes parts jitter through each other.
            data.qvel[dof : dof + 2] = current[:2] + self.gain * (target[:2] - current[:2])


@dataclass
class JointCycleDriver(Driver):
    """Opens and closes an actuated joint on a schedule (doors, gates, lids)."""

    name: str
    actuator: str
    low: float
    high: float
    period: float = 8.0
    dwell: float = 0.35
    """Fraction of each half-cycle spent held at the endpoint."""
    _id: int = -1

    def reset(self, model, data) -> None:
        import mujoco

        self._id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.actuator)
        if self._id < 0:
            raise ValueError(f"Joint cycle {self.name!r} references unknown actuator {self.actuator!r}.")

    def apply(self, model, data) -> None:
        period = max(self.period, 1e-6)
        phase = (data.time % period) / period
        dwell = min(max(self.dwell, 0.0), 0.49)
        if phase < 0.5:
            alpha = _ramp(phase / 0.5, dwell)
        else:
            alpha = 1.0 - _ramp((phase - 0.5) / 0.5, dwell)
        data.ctrl[self._id] = self.low + (self.high - self.low) * alpha


def _ramp(u: float, dwell: float) -> float:
    """0 -> 1 over [dwell, 1 - dwell], held flat at either end."""
    span = 1.0 - 2.0 * dwell
    if span <= 1e-6:
        return 0.0 if u < 0.5 else 1.0
    return float(np.clip((u - dwell) / span, 0.0, 1.0))


@dataclass
class BakedDriver(Driver):
    """Replays a trajectory sampled from Blender's animation curves.

    `samples` is `[[t, v0, v1, ...], ...]`. With `target='mocap'` the values are
    an xyz position (optionally followed by a wxyz quaternion); with
    `target='actuator'` a single value is written to `data.ctrl`.
    """

    name: str
    target: str
    samples: list[list[float]]
    body: str | None = None
    actuator: str | None = None
    loop: bool = True
    _times: np.ndarray = None
    _values: np.ndarray = None
    _id: int = -1

    def reset(self, model, data) -> None:
        import mujoco

        table = np.asarray(self.samples, dtype=float)
        if table.ndim != 2 or table.shape[0] < 2:
            raise ValueError(f"Baked driver {self.name!r} needs at least two samples.")
        self._times = table[:, 0]
        self._values = table[:, 1:]

        if self.target == "mocap":
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body or "")
            if body_id < 0:
                raise ValueError(f"Baked driver {self.name!r} references unknown body {self.body!r}.")
            self._id = int(model.body_mocapid[body_id])
            if self._id < 0:
                raise ValueError(f"Baked driver {self.name!r} targets non-mocap body {self.body!r}.")
        elif self.target == "actuator":
            self._id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.actuator or "")
            if self._id < 0:
                raise ValueError(
                    f"Baked driver {self.name!r} references unknown actuator {self.actuator!r}."
                )
        else:
            raise ValueError(f"Baked driver {self.name!r} has unknown target {self.target!r}.")

    def apply(self, model, data) -> None:
        span = self._times[-1] - self._times[0]
        t = data.time
        if self.loop and span > 1e-9:
            t = self._times[0] + ((t - self._times[0]) % span)

        values = [np.interp(t, self._times, self._values[:, i]) for i in range(self._values.shape[1])]

        if self.target == "mocap":
            data.mocap_pos[self._id] = values[:3]
            if len(values) >= 7:
                quat = np.asarray(values[3:7], dtype=float)
                norm = np.linalg.norm(quat)
                if norm > 1e-9:
                    data.mocap_quat[self._id] = quat / norm
        else:
            data.ctrl[self._id] = values[0]


# ---------------------------------------------------------------------------
# Driver assembly
# ---------------------------------------------------------------------------


def make_drivers(spec: SceneSpec, info) -> list[Driver]:
    """Build the run-time drivers for a scene, in spec order."""
    drivers: list[Driver] = []

    for element in spec.dynamics:
        if not element.enabled:
            continue
        params = element.params
        kind = element.kind

        if kind == "roller_conveyor":
            roller_bodies = [
                name for name in info.expanded_bodies.get(element.name, []) if "_roller_" in name
            ]
            drivers.append(
                ConveyorDriver(
                    name=element.name,
                    actuators=[info.actuators[b] for b in roller_bodies],
                    speed=float(params.get("speed", 0.2)),
                    radius=float(params.get("roller_radius", 0.03)),
                    duty=params.get("duty"),
                    stop_body=params.get("stop_body"),
                    stop_at=(float(params["stop_at"]) if "stop_at" in params else None),
                    direction=tuple(_direction(params.get("direction", "+x"))),
                )
            )
        elif kind == "turntable":
            disc = f"{element.name}_disc"
            drivers.append(
                TurntableDriver(
                    name=element.name,
                    actuator=info.actuators[disc],
                    angular_speed=float(params.get("angular_speed", 1.0)),
                )
            )
        elif kind == "mover":
            drivers.append(
                MoverDriver(
                    name=element.name,
                    body=params["body"],
                    path=params.get("path", "harmonic"),
                    params=params,
                )
            )
        elif kind == "belt_field":
            drivers.append(
                BeltFieldDriver(
                    name=element.name,
                    region_min=tuple(params["region_min"]),
                    region_max=tuple(params["region_max"]),
                    velocity=tuple(params["velocity"]),
                    gain=float(params.get("gain", 0.25)),
                    bodies=params.get("bodies"),
                )
            )
        elif kind == "joint_cycle":
            body = params["body"]
            drivers.append(
                JointCycleDriver(
                    name=element.name,
                    actuator=info.actuators[body],
                    low=float(params.get("low", 0.0)),
                    high=float(params.get("high", 1.0)),
                    period=float(params.get("period", 8.0)),
                    dwell=float(params.get("dwell", 0.35)),
                )
            )
        elif kind == "severable":
            drivers.append(
                SeverDriver(
                    name=element.name,
                    equality=params["equality"],
                    blade=info.mounted_bodies.get(params["blade"], params["blade"]),
                    site=params["site"],
                    radius=float(params.get("radius", 0.05)),
                    min_speed=float(params.get("min_speed", 0.15)),
                    transfer=float(params.get("transfer", 0.6)),
                    freed_body=params.get("freed_body"),
                )
            )
        elif kind == "baked":
            drivers.append(
                BakedDriver(
                    name=element.name,
                    target=params.get("target", "mocap"),
                    samples=params["samples"],
                    body=params.get("body"),
                    actuator=info.actuators.get(params.get("body", ""), params.get("actuator")),
                    loop=bool(params.get("loop", True)),
                )
            )
        else:
            raise ValueError(
                f"Dynamic element {element.name!r} has unknown kind {element.kind!r}; "
                f"expected one of {DRIVER_KINDS}."
            )

    return drivers


@dataclass
class SeverDriver(Driver):
    """Releases a weld when a blade passes through the cut plane.

    MuJoCo has no fracture model, so "cutting" is modelled honestly as three
    facts that are each real:

      * the object is two bodies held rigid by a weld, so before the cut it
        behaves as one piece and resists being pushed over;
      * the blade is a non-colliding sensor geom, because a rigid blade cannot
        pass through a rigid stalk — it would simply bat it across the room;
      * on a qualifying pass the weld is deactivated through `data.eq_active`
        and the blade's own velocity is handed to the freed piece.

    That last step is the one worth being explicit about. A real cut transfers
    momentum from blade to severed piece through the contact this model does
    not have, so it is applied directly. Without it the top of the stalk simply
    balances on the stump, which looks like a failed cut.

    The trigger is `blade within `radius` of the cut site` AND `blade speed
    above `min_speed``. The speed term is what stops a slow nudge counting as a
    cut, but note it is a kinematic threshold, not a cutting-force model: this
    says nothing about whether a real blade at that speed would sever real
    bamboo.
    """

    name: str
    equality: str
    blade: str
    site: str
    radius: float = 0.05
    min_speed: float = 0.15
    cut_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Direction the edge must be travelling for a pass to count as a cut.

    Speed alone is not enough. Without this the blade severs the stalk while
    merely swinging past it into the start position, which is not a cut — it is
    the approach. Requiring the velocity component along the stroke direction
    means only a deliberate stroke registers."""
    transfer: float = 1.0
    """Fraction of the blade's velocity handed to the severed piece.

    1.0 means the freed piece leaves with the blade's own velocity, which
    is the closest this model gets to the blade driving it through."""
    freed_body: str | None = None
    _eq_id: int = -1
    _blade_id: int = -1
    _site_id: int = -1
    _freed_dof: int = -1
    _severed: bool = False
    _last_pos: Any = None

    def reset(self, model, data) -> None:
        import mujoco

        def need(objtype, name, what):
            index = mujoco.mj_name2id(model, objtype, name)
            if index < 0:
                raise ValueError(f"Sever driver {self.name!r} references unknown {what} {name!r}.")
            return index

        self._eq_id = need(mujoco.mjtObj.mjOBJ_EQUALITY, self.equality, "equality")
        self._blade_id = need(mujoco.mjtObj.mjOBJ_BODY, self.blade, "body")
        self._site_id = need(mujoco.mjtObj.mjOBJ_BODY, self.site, "body")
        self._severed = False
        self._last_pos = None
        data.eq_active[self._eq_id] = 1

        self._freed_dof = -1
        if self.freed_body:
            body_id = need(mujoco.mjtObj.mjOBJ_BODY, self.freed_body, "body")
            for joint_id in range(model.njnt):
                if (model.jnt_bodyid[joint_id] == body_id
                        and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE):
                    self._freed_dof = int(model.jnt_dofadr[joint_id])
                    break

    def _edge_distance(self, model, data, site) -> float:
        """Distance from the cut site to the blade's cutting edge.

        Measuring to the blade's centre would force the wrist directly over the
        target, which is not how a sword is swung — the hand stays off to one
        side and the edge reaches. So the edge is treated as a segment along the
        blade's longest axis and the true point-to-segment distance is used.
        """
        centre = data.xpos[self._blade_id]
        geoms = np.flatnonzero(model.geom_bodyid == self._blade_id)
        if not len(geoms):
            return float(np.linalg.norm(site - centre))

        gid = int(geoms[0])
        half = model.geom_size[gid][:3]
        axis_index = int(np.argmax(half))
        rotation = data.geom_xmat[gid].reshape(3, 3)
        axis = rotation[:, axis_index]

        offset = np.asarray(site, dtype=float) - data.geom_xpos[gid]
        t = float(np.clip(np.dot(offset, axis), -half[axis_index], half[axis_index]))
        return float(np.linalg.norm(offset - t * axis))

    def apply(self, model, data) -> None:
        blade = data.xpos[self._blade_id]

        # Finite-difference the blade rather than reading a body velocity: the
        # blade is mounted on the arm, so its motion comes from the arm's
        # joints and is not a free-joint qvel we could read directly.
        speed = 0.0
        if self._last_pos is not None:
            speed = float(np.linalg.norm(blade - self._last_pos)) / max(model.opt.timestep, 1e-9)
        velocity = (blade - self._last_pos) / max(model.opt.timestep, 1e-9) \
            if self._last_pos is not None else np.zeros(3)
        self._last_pos = np.array(blade, dtype=float)

        if self._severed:
            return

        into = float(np.dot(velocity, _unit(self.cut_direction)))
        distance = self._edge_distance(model, data, data.xpos[self._site_id])
        if distance <= self.radius and into >= self.min_speed:
            data.eq_active[self._eq_id] = 0
            self._severed = True
            if self._freed_dof >= 0:
                data.qvel[self._freed_dof : self._freed_dof + 3] += self.transfer * velocity

    def state(self) -> dict[str, Any]:
        return {"severed": self._severed}
