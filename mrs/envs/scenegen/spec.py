"""The declarative description of a generated scene.

A `SceneSpec` is the single durable artifact of environment authoring: it is
what the Blender session exports, what a human reviews, and what the MuJoCo
builder consumes. Everything downstream — the compiled model, the observation
contract, the success predicate — is derived from it, so a spec plus this
package reproduces an environment exactly.

The field names deliberately echo `mrs.envs.configs.PandaPickPlaceConfig`: a
spec that describes a single cube and a plate produces an environment with the
same observation and action contract as the hand-written one, and therefore
loads the same pi0.5 checkpoints without an adapter.

Units are SI throughout: metres, kilograms, seconds, radians. Orientations are
`(w, x, y, z)` quaternions, matching MuJoCo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

SPEC_VERSION = 1

# Shapes that map onto a MuJoCo primitive geom. `mesh` additionally requires
# `mesh_file`; `plane` is only valid on a static body.
SHAPES = ("box", "cylinder", "sphere", "capsule", "ellipsoid", "plane", "mesh")

# How a body participates in the simulation.
#   static  — welded to the world, no degrees of freedom (tables, bins, walls)
#   free    — a free joint, six DoF, the things a policy manipulates
#   hinged  — one revolute DoF relative to its parent (doors, lids, rollers)
#   sliding — one prismatic DoF relative to its parent (drawers, pushers)
#   mocap   — kinematically scripted, no DoF, unaffected by contact forces
KINDS = ("static", "free", "hinged", "sliding", "mocap")


# ---------------------------------------------------------------------------
# Leaf specs
# ---------------------------------------------------------------------------


@dataclass
class MaterialSpec:
    """Appearance only — never affects physics."""

    name: str
    rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    specular: float = 0.2
    shininess: float = 0.2
    reflectance: float = 0.0
    texture: str | None = None
    """Name of a builtin checker texture to generate, or None for a flat colour."""
    texrepeat: tuple[float, float] = (1.0, 1.0)
    checker_rgb2: tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class JointSpec:
    """The single articulated degree of freedom a body may carry.

    `armature` is not cosmetic. A small rotor inertia is what keeps a velocity
    servo on a light body (a conveyor roller) numerically stable — see
    `mrs/envs/scenegen/dynamics.py` for the sizing rule.
    """

    type: str = "hinge"  # hinge | slide
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    range: tuple[float, float] | None = None
    damping: float = 0.1
    stiffness: float = 0.0
    springref: float = 0.0
    armature: float = 0.0
    frictionloss: float = 0.0
    ref: float = 0.0


@dataclass
class ActuatorSpec:
    """A control channel driving one joint.

    `kind` is one of `position`, `velocity`, `intvelocity` or `motor`. Position
    and integrated-velocity servos take `kp`/`kv`; a velocity servo takes `kv`
    alone; a motor is raw force and takes neither.
    """

    kind: str = "position"
    kp: float = 100.0
    kv: float = 10.0
    ctrlrange: tuple[float, float] | None = None
    forcerange: tuple[float, float] | None = None
    gear: float = 1.0
    default_ctrl: float = 0.0
    """Value written to `data.ctrl` on reset."""


@dataclass
class BodySpec:
    """One rigid body in the scene.

    A body with `parent` set is nested inside that body, which is how an
    articulated assembly is expressed: a drawer is a `sliding` body whose
    parent is the static cabinet.
    """

    name: str
    kind: str = "static"
    shape: str = "box"
    size: tuple[float, ...] = (0.05, 0.05, 0.05)
    """MuJoCo geom size semantics: box is half-extents, cylinder/capsule is
    (radius, half-length), sphere is (radius,), plane is (x, y, spacing)."""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    geom_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Rotation of the geom inside its body. A cylinder's local axis is +z, so a
    roller lying along +y needs a geom rotation while its joint axis stays +y."""
    parent: str | None = None
    mount: str | None = None
    """Robot link to rigidly attach this body to, e.g. `hand`.

    A tool, not a prop: the body is added to the robot's own kinematic tree
    before attachment, so it moves with the arm and has no degrees of freedom
    of its own. `pos`/`quat` are interpreted in the link's frame."""
    material: str | None = None
    rgba: tuple[float, float, float, float] | None = None

    mass: float | None = None
    density: float | None = None
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    condim: int = 3
    solref: tuple[float, float] = (0.02, 1.0)
    solimp: tuple[float, ...] | None = None
    margin: float = 0.0

    contype: int = 1
    conaffinity: int = 1
    group: int = 0

    mesh_file: str | None = None
    """Path relative to the env package's `assets/` directory."""
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mesh_maxhullvert: int = 64

    joint: JointSpec | None = None
    actuator: ActuatorSpec | None = None

    spawn_range: dict[str, tuple[float, float]] | None = None
    """Per-axis uniform spawn ranges for `free` bodies, e.g.
    `{"x": [-0.1, 0.1], "y": [-0.2, 0.0], "yaw": [-0.4, 0.4]}`. Absent axes are
    held at `pos`. Resolved on every `reset()`."""

    tags: list[str] = field(default_factory=list)
    """Free-form labels the task and success layers select on, e.g.
    `["envelope", "size_large"]`."""


@dataclass
class EqualitySpec:
    """A constraint between two bodies, optionally released at run time.

    MuJoCo cannot fracture geometry, so a severable object is modelled as two
    bodies held rigid by a weld that a driver deactivates via `data.eq_active`
    when the cut condition is met. Before the cut the pair behaves as one rigid
    object; after it they are independent. That is a real constraint being
    released, not a visual trick.
    """

    name: str
    body1: str
    body2: str
    type: str = "weld"  # weld | connect
    active: bool = True
    solref: tuple[float, float] = (0.02, 1.0)


@dataclass
class DynamicSpec:
    """A moving element of the scene.

    Some kinds expand into bodies and actuators at build time
    (`roller_conveyor`, `turntable`); others are pure run-time drivers that
    write to `data` each control step (`mover`, `belt_field`, `baked`). See
    `dynamics.py`, which owns both halves.
    """

    name: str
    kind: str
    """roller_conveyor | belt_field | turntable | mover | baked | joint_cycle"""
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class CameraSpec:
    name: str
    pos: tuple[float, float, float] = (1.0, 0.0, 1.2)
    target: tuple[float, float, float] | None = (0.0, 0.0, 0.7)
    """When set, orientation is derived by aiming at this point."""
    quat: tuple[float, float, float, float] | None = None
    fovy: float = 45.0
    mount: str | None = None
    """Body to attach the camera to. `None` means the world; a robot link name
    (e.g. `hand`) makes it a wrist camera that moves with the arm."""
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    role: str = "scene"
    """scene | wrist | inspection. Only `scene` and `wrist` feed the policy;
    `inspection` cameras exist for human review and validation renders."""


@dataclass
class RobotSpec:
    """Which arm to mount and where.

    `key` indexes `mrs.envs.scenegen.robots.ROBOTS`, which carries the
    Menagerie-specific naming (actuator names, hand body, gripper convention)
    so the rest of this package stays robot-agnostic.
    """

    key: str = "panda"
    mount_pos: tuple[float, float, float] = (-0.56, 0.0, 0.63)
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    home_qpos: tuple[float, ...] | None = None
    """Overrides the registry default when set."""
    prefix: str = "robot_"
    pedestal: bool = True
    """Add a plain column under the mount so the arm is not floating."""


@dataclass
class ControlSpec:
    """The policy-facing contract. Defaults reproduce `PandaPickPlaceConfig`."""

    sim_timestep: float = 0.002
    control_freq: float = 20.0
    integrator: str = "implicitfast"
    """implicitfast is required for stable velocity servos; see dynamics.py."""

    position_delta_scale: float = 0.05
    rotation_delta_scale: float = 0.5

    ik_damping: float = 0.05
    ik_max_joint_step: float = 0.05
    ik_max_total_change: float = 0.6
    nullspace_gain: float = 0.05

    position_leash: float = 0.07
    rotation_leash: float = 0.5

    workspace_min: tuple[float, float, float] = (-0.32, -0.36, 0.645)
    workspace_max: tuple[float, float, float] = (0.20, 0.36, 1.00)

    max_episode_steps: int = 400
    success_hold_steps: int = 5
    reset_settle_steps: int = 5
    reset_noise: float = 0.0

    image_size: int = 256
    render_size: int = 512
    scene_image_key: str = "observation.images.image"
    wrist_image_key: str = "observation.images.image2"

    @property
    def n_substeps(self) -> int:
        steps = round(1.0 / (self.control_freq * self.sim_timestep))
        if steps < 1:
            raise ValueError("control_freq is too high for the configured sim_timestep.")
        return steps


@dataclass
class SuccessSpec:
    """A boolean expression over predicates.

    Kept deliberately small here; `robo-task-define` is the skill that grows it.
    `mode` is `all` or `any` over `terms`, each of which names a predicate in
    `mrs.envs.scenegen.success.PREDICATES` plus its keyword arguments.
    """

    mode: str = "all"
    terms: list[dict[str, Any]] = field(default_factory=list)
    failure_terms: list[dict[str, Any]] = field(default_factory=list)
    """Predicates that, when true, end the episode as a failure — a dropped
    part, a knocked-over bin. Recorded in `info["failure_modes"]`."""


@dataclass
class WorldSpec:
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    floor: bool = True
    floor_material: str = "floor_mat"
    skybox: bool = True
    headlight_diffuse: tuple[float, float, float] = (0.25, 0.25, 0.25)
    headlight_ambient: tuple[float, float, float] = (0.2, 0.2, 0.2)
    lights: list[dict[str, Any]] = field(default_factory=list)
    offscreen_size: int = 1024


@dataclass
class SceneSpec:
    """The whole environment, in one serialisable object."""

    name: str
    task: str
    spec_version: int = SPEC_VERSION
    world: WorldSpec = field(default_factory=WorldSpec)
    materials: list[MaterialSpec] = field(default_factory=list)
    bodies: list[BodySpec] = field(default_factory=list)
    equalities: list[EqualitySpec] = field(default_factory=list)
    dynamics: list[DynamicSpec] = field(default_factory=list)
    robot: RobotSpec | None = field(default_factory=RobotSpec)
    cameras: list[CameraSpec] = field(default_factory=list)
    control: ControlSpec = field(default_factory=ControlSpec)
    success: SuccessSpec = field(default_factory=SuccessSpec)
    seed: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    """Where this came from: prompt, blend file, review-loop iteration, git sha."""

    # ---- lookup helpers -------------------------------------------------
    def body(self, name: str) -> BodySpec:
        for body in self.bodies:
            if body.name == name:
                return body
        raise KeyError(f"No body named {name!r} in scene {self.name!r}.")

    def bodies_tagged(self, tag: str) -> list[BodySpec]:
        return [b for b in self.bodies if tag in b.tags]

    def camera(self, role: str) -> CameraSpec | None:
        for cam in self.cameras:
            if cam.role == role:
                return cam
        return None

    @property
    def free_bodies(self) -> list[BodySpec]:
        return [b for b in self.bodies if b.kind == "free"]

    # ---- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SceneSpec:
        version = payload.get("spec_version", SPEC_VERSION)
        if version > SPEC_VERSION:
            raise ValueError(
                f"Scene spec version {version} is newer than this package supports "
                f"({SPEC_VERSION}). Update mrs.envs.scenegen."
            )
        return _build(cls, payload)

    @classmethod
    def load(cls, path: str | Path) -> SceneSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Dataclass <-> JSON
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Make tuples lists and numpy scalars floats, so `json` accepts the tree."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _build(cls: type, payload: Any) -> Any:
    """Rebuild nested dataclasses from plain dicts, one level of typing deep.

    Only the shapes this module actually uses are handled: nested dataclasses,
    `list[Dataclass]`, and optional dataclasses. Anything typed `dict` or
    `Any` (`DynamicSpec.params`, `SceneSpec.provenance`) passes through
    untouched, which is what makes those fields extensible without a schema
    bump.
    """
    if not is_dataclass(cls) or not isinstance(payload, dict):
        return payload

    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in payload.items():
        spec_field = known.get(key)
        if spec_field is None:
            continue  # forward compatibility: ignore unknown keys
        kwargs[key] = _coerce(spec_field.type, value)
    return cls(**kwargs)


_NESTED = {
    "WorldSpec": WorldSpec,
    "MaterialSpec": MaterialSpec,
    "BodySpec": BodySpec,
    "JointSpec": JointSpec,
    "ActuatorSpec": ActuatorSpec,
    "DynamicSpec": DynamicSpec,
    "EqualitySpec": EqualitySpec,
    "CameraSpec": CameraSpec,
    "RobotSpec": RobotSpec,
    "ControlSpec": ControlSpec,
    "SuccessSpec": SuccessSpec,
}


def _coerce(annotation: Any, value: Any) -> Any:
    """Resolve a field annotation to a dataclass and rebuild, else pass through."""
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")

    for name, klass in _NESTED.items():
        if text.startswith(f"list[{name}]") and isinstance(value, list):
            return [_build(klass, item) for item in value]
        if text.startswith(name) and isinstance(value, dict):
            return _build(klass, value)
    return value
