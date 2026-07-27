"""Compile a `SceneSpec` into a `mujoco.MjModel`.

This is the generalisation of `mrs/envs/scene.py`: same assembly strategy —
build the world with `MjSpec`, add a grasp site and wrist camera to the robot's
hand programmatically, then attach the unmodified Menagerie model — but driven
by data instead of by hand-written calls.

`look_at` is imported from the hand-written scene rather than reimplemented, so
the two builders orient cameras identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mrs.envs.scene import GRIP_SITE, look_at
from mrs.envs.scenegen.robots import RobotModel, get_robot
from mrs.envs.scenegen.spec import BodySpec, CameraSpec, MaterialSpec, SceneSpec

logger = logging.getLogger(__name__)

INTEGRATORS = {
    "euler": "mjINT_EULER",
    "rk4": "mjINT_RK4",
    "implicit": "mjINT_IMPLICIT",
    "implicitfast": "mjINT_IMPLICITFAST",
}

_SHAPE_ENUM = {
    "box": "mjGEOM_BOX",
    "cylinder": "mjGEOM_CYLINDER",
    "sphere": "mjGEOM_SPHERE",
    "capsule": "mjGEOM_CAPSULE",
    "ellipsoid": "mjGEOM_ELLIPSOID",
    "plane": "mjGEOM_PLANE",
    "mesh": "mjGEOM_MESH",
}

_JOINT_ENUM = {"hinge": "mjJNT_HINGE", "slide": "mjJNT_SLIDE", "ball": "mjJNT_BALL"}


@dataclass
class BuildInfo:
    """Names the environment needs to resolve after compilation.

    Populated during the build because some names are synthesised (a free
    body's joint, a conveyor's expanded rollers) and reconstructing them by
    string convention downstream is how naming drifts.
    """

    robot: RobotModel | None = None
    robot_prefix: str = "robot_"
    free_joints: dict[str, str] = field(default_factory=dict)
    """body name -> free joint name"""
    articulated_joints: dict[str, str] = field(default_factory=dict)
    """body name -> hinge/slide joint name"""
    actuators: dict[str, str] = field(default_factory=dict)
    """body name -> actuator name"""
    mocap_bodies: list[str] = field(default_factory=list)
    mounted_bodies: dict[str, str] = field(default_factory=dict)
    """spec body name -> compiled name (tools gain the robot prefix)"""
    equalities: list[str] = field(default_factory=list)
    expanded_bodies: dict[str, list[str]] = field(default_factory=dict)
    """dynamic element name -> body names it generated"""
    expanded_actuators: dict[str, list[str]] = field(default_factory=dict)
    cameras: dict[str, str] = field(default_factory=dict)
    """CameraSpec.name -> compiled camera name (wrist cameras gain the prefix)"""
    warnings: list[str] = field(default_factory=list)


def build_model(spec: SceneSpec, *, asset_dir: str | Path | None = None):
    """Return `(model, build_info)` for a scene spec.

    `asset_dir` is where mesh files referenced by `BodySpec.mesh_file` live;
    it defaults to an `assets/` directory beside the spec, which is the layout
    `build_env.py` writes.
    """
    import mujoco

    from mrs.envs.scenegen import dynamics

    mspec = mujoco.MjSpec()
    mspec.modelname = spec.name
    mspec.option.timestep = spec.control.sim_timestep
    mspec.option.gravity = list(spec.world.gravity)

    integrator = spec.control.integrator.lower()
    if integrator not in INTEGRATORS:
        raise ValueError(f"Unknown integrator {spec.control.integrator!r}; use one of {sorted(INTEGRATORS)}.")
    mspec.option.integrator = getattr(mujoco.mjtIntegrator, INTEGRATORS[integrator])

    info = BuildInfo()

    # Velocity servos on light rotors are stiff; an explicit integrator diverges
    # within a few steps. Rather than fail at run time with a NaN, say so here.
    if integrator == "euler" and any(
        b.actuator is not None and b.actuator.kind in ("velocity", "intvelocity") for b in spec.bodies
    ):
        info.warnings.append(
            "Euler integrator with velocity actuators: expect divergence. Use implicitfast."
        )

    if asset_dir is not None:
        mspec.meshdir = str(Path(asset_dir))

    _add_visual(mspec, mujoco, spec)
    _add_materials(mspec, mujoco, spec)

    # Dynamic macros expand into ordinary bodies and actuators, so everything
    # downstream sees one uniform body list.
    bodies, extra_actuators = dynamics.expand(spec, info)

    # Tools live on the robot's own kinematic tree, so they are added during
    # attachment rather than to the world.
    mounted = [b for b in bodies if b.mount is not None]
    bodies = [b for b in bodies if b.mount is None]

    _add_meshes(mspec, bodies + mounted)
    _add_bodies(mspec, mujoco, spec, bodies, info)
    _add_actuators(mspec, mujoco, bodies, extra_actuators, info)
    _add_world_cameras(mspec, spec, info)

    if spec.robot is not None:
        _attach_robot(mspec, mujoco, spec, info, mounted)
    elif mounted:
        raise ValueError(
            f"Scene {spec.name!r} mounts {[b.name for b in mounted]} on a robot link "
            f"but declares no robot."
        )

    _add_equalities(mspec, mujoco, spec, info)

    model = mspec.compile()
    return model, info


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


def _add_visual(mspec, mujoco, spec: SceneSpec) -> None:
    world = spec.world
    mspec.visual.headlight.diffuse = list(world.headlight_diffuse)
    mspec.visual.headlight.ambient = list(world.headlight_ambient)
    mspec.visual.headlight.specular = [0.0, 0.0, 0.0]
    mspec.visual.quality.shadowsize = 4096
    # MuJoCo refuses to render above the offscreen framebuffer size, and the
    # review renders are larger than the 640x480 default.
    mspec.visual.global_.offwidth = world.offscreen_size
    mspec.visual.global_.offheight = world.offscreen_size

    lights = world.lights or [
        {"pos": [0.4, 0.0, 2.2], "dir": [-0.3, -0.4, -1.0]},
        {"pos": [0.4, 0.0, 2.2], "dir": [-0.3, 0.4, -1.0]},
    ]
    for entry in lights:
        light = mspec.worldbody.add_light()
        light.pos = list(entry.get("pos", [0.0, 0.0, 2.0]))
        light.dir = list(entry.get("dir", [0.0, 0.0, -1.0]))
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.diffuse = list(entry.get("diffuse", [0.32, 0.32, 0.32]))
        light.specular = list(entry.get("specular", [0.05, 0.05, 0.05]))
        light.castshadow = bool(entry.get("castshadow", True))


def _add_materials(mspec, mujoco, spec: SceneSpec) -> None:
    if spec.world.skybox:
        sky = mspec.add_texture()
        sky.name = "skybox"
        sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
        sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        sky.rgb1 = [0.55, 0.6, 0.68]
        sky.rgb2 = [0.2, 0.24, 0.3]
        sky.width, sky.height = 512, 3072

    materials = list(spec.materials)
    if spec.world.floor and not any(m.name == spec.world.floor_material for m in materials):
        materials.append(
            MaterialSpec(
                name=spec.world.floor_material,
                rgba=(0.24, 0.26, 0.29, 1.0),
                texture="checker",
                texrepeat=(4.0, 4.0),
                checker_rgb2=(0.19, 0.21, 0.24),
                reflectance=0.05,
            )
        )

    for entry in materials:
        if entry.texture == "checker":
            tex = mspec.add_texture()
            tex.name = f"{entry.name}_tex"
            tex.type = mujoco.mjtTexture.mjTEXTURE_2D
            tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
            tex.rgb1 = list(entry.rgba[:3])
            tex.rgb2 = list(entry.checker_rgb2)
            tex.width, tex.height = 512, 512

        mat = mspec.add_material()
        mat.name = entry.name
        mat.rgba = list(entry.rgba)
        mat.specular = entry.specular
        mat.shininess = entry.shininess
        mat.reflectance = entry.reflectance
        if entry.texture == "checker":
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{entry.name}_tex"
            mat.texrepeat = list(entry.texrepeat)
            mat.texuniform = True


def _add_meshes(mspec, bodies: list[BodySpec]) -> None:
    seen: set[str] = set()
    for body in bodies:
        if body.shape != "mesh" or not body.mesh_file:
            continue
        name = _mesh_name(body.mesh_file)
        if name in seen:
            continue
        seen.add(name)
        mesh = mspec.add_mesh()
        mesh.name = name
        mesh.file = body.mesh_file
        mesh.scale = list(body.mesh_scale)
        mesh.maxhullvert = body.mesh_maxhullvert


def _mesh_name(mesh_file: str) -> str:
    return Path(mesh_file).stem.replace(".", "_")


# ---------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------


def _add_bodies(mspec, mujoco, spec: SceneSpec, bodies: list[BodySpec], info: BuildInfo) -> None:
    if spec.world.floor:
        floor = mspec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
        floor.material = spec.world.floor_material

    handles = {None: mspec.worldbody}
    for body in _in_parent_order(bodies):
        parent = handles.get(body.parent)
        if parent is None:
            raise ValueError(f"Body {body.name!r} names parent {body.parent!r}, which is not in the scene.")
        handles[body.name] = _add_body(mujoco, parent, body, info)


def _in_parent_order(bodies: list[BodySpec]) -> list[BodySpec]:
    """Parents before children, so a nested body always finds its handle."""
    remaining = {b.name: b for b in bodies}
    if len(remaining) != len(bodies):
        duplicates = sorted({b.name for b in bodies if list(b.name for b in bodies).count(b.name) > 1})
        raise ValueError(f"Duplicate body names in scene: {duplicates}")

    ordered: list[BodySpec] = []
    placed: set[str] = set()
    while remaining:
        ready = [b for b in remaining.values() if b.parent is None or b.parent in placed]
        if not ready:
            cycle = sorted(remaining)
            raise ValueError(f"Cycle or missing parent among bodies: {cycle}")
        for body in ready:
            ordered.append(body)
            placed.add(body.name)
            del remaining[body.name]
    return ordered


def _add_body(mujoco, parent, body: BodySpec, info: BuildInfo):
    handle = parent.add_body()
    handle.name = body.name
    handle.pos = list(body.pos)
    handle.quat = list(body.quat)

    if body.kind == "mocap":
        handle.mocap = True
        info.mocap_bodies.append(body.name)
    elif body.kind == "free":
        joint_name = f"{body.name}_joint"
        handle.add_freejoint(name=joint_name)
        info.free_joints[body.name] = joint_name
    elif body.kind in ("hinged", "sliding"):
        if body.joint is None:
            raise ValueError(f"Body {body.name!r} is {body.kind} but carries no joint spec.")
        joint_name = f"{body.name}_joint"
        joint = handle.add_joint()
        joint.name = joint_name
        joint.type = getattr(
            mujoco.mjtJoint, _JOINT_ENUM["hinge" if body.kind == "hinged" else "slide"]
        )
        joint.axis = list(body.joint.axis)
        joint.damping = body.joint.damping
        joint.stiffness = body.joint.stiffness
        joint.springref = body.joint.springref
        joint.armature = body.joint.armature
        joint.frictionloss = body.joint.frictionloss
        joint.ref = body.joint.ref
        if body.joint.range is not None:
            joint.range = list(body.joint.range)
            joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        info.articulated_joints[body.name] = joint_name
    elif body.kind != "static":
        raise ValueError(f"Body {body.name!r} has unknown kind {body.kind!r}.")

    _add_geom(mujoco, handle, body)
    return handle


def _add_geom(mujoco, handle, body: BodySpec) -> None:
    if body.shape not in _SHAPE_ENUM:
        raise ValueError(f"Body {body.name!r} has unknown shape {body.shape!r}.")

    geom = handle.add_geom()
    geom.name = f"{body.name}_geom"
    geom.type = getattr(mujoco.mjtGeom, _SHAPE_ENUM[body.shape])
    geom.quat = list(body.geom_quat)

    if body.shape == "mesh":
        if not body.mesh_file:
            raise ValueError(f"Body {body.name!r} has shape 'mesh' but no mesh_file.")
        geom.meshname = _mesh_name(body.mesh_file)
    else:
        size = list(body.size) + [0.0, 0.0, 0.0]
        geom.size = size[:3]

    if body.material:
        geom.material = body.material
    if body.rgba is not None:
        geom.rgba = list(body.rgba)

    geom.friction = list(body.friction)
    geom.condim = body.condim
    geom.solref = list(body.solref)
    if body.solimp is not None:
        geom.solimp = list(body.solimp)
    geom.margin = body.margin
    geom.contype = body.contype
    geom.conaffinity = body.conaffinity
    geom.group = body.group

    # Mass and density are alternative inertia sources; setting both makes the
    # compiler prefer mass and silently ignore density, so only ever set one.
    if body.mass is not None:
        geom.mass = body.mass
    elif body.density is not None:
        geom.density = body.density


# ---------------------------------------------------------------------------
# Actuators
# ---------------------------------------------------------------------------


def _add_actuators(mspec, mujoco, bodies, extra_actuators, info: BuildInfo) -> None:
    for body in bodies:
        if body.actuator is None:
            continue
        joint_name = info.articulated_joints.get(body.name)
        if joint_name is None:
            raise ValueError(
                f"Body {body.name!r} declares an actuator but has no hinge or slide joint "
                f"to drive (kind={body.kind!r})."
            )
        name = f"{body.name}_drive"
        _make_actuator(mspec, mujoco, name, joint_name, body.actuator)
        info.actuators[body.name] = name

    for name, joint_name, actuator in extra_actuators:
        _make_actuator(mspec, mujoco, name, joint_name, actuator)


def _make_actuator(mspec, mujoco, name: str, joint_name: str, actuator) -> None:
    handle = mspec.add_actuator()
    handle.name = name
    handle.trntype = mujoco.mjtTrn.mjTRN_JOINT
    handle.target = joint_name
    handle.gear[0] = actuator.gear

    kind = actuator.kind
    if kind == "position":
        handle.set_to_position(actuator.kp, kv=actuator.kv)
    elif kind == "velocity":
        handle.set_to_velocity(actuator.kv)
    elif kind == "intvelocity":
        handle.set_to_intvelocity(actuator.kp, kv=actuator.kv)
    elif kind == "motor":
        handle.set_to_motor()
    else:
        raise ValueError(f"Actuator {name!r} has unknown kind {kind!r}.")

    if actuator.ctrlrange is not None:
        handle.ctrlrange = list(actuator.ctrlrange)
        handle.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    if actuator.forcerange is not None:
        handle.forcerange = list(actuator.forcerange)
        handle.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE


# ---------------------------------------------------------------------------
# Cameras and robot
# ---------------------------------------------------------------------------


def _safe_up(view_dir: np.ndarray, up) -> tuple[float, float, float]:
    """Pick an up vector that is not parallel to the view direction.

    A top-down inspection camera looks straight along -z, which is exactly the
    default up vector; `look_at` then takes a cross product of two parallel
    vectors and produces NaN. Fall back to +y in that case.
    """
    norm = np.linalg.norm(view_dir)
    if norm < 1e-9:
        return tuple(up)
    if abs(float(np.dot(view_dir / norm, np.asarray(up, dtype=float)))) > 0.999:
        return (0.0, 1.0, 0.0)
    return tuple(up)


def _orient(cam: CameraSpec) -> list[float]:
    if cam.quat is not None:
        return list(cam.quat)
    if cam.target is None:
        return [1.0, 0.0, 0.0, 0.0]
    view = np.asarray(cam.pos, dtype=float) - np.asarray(cam.target, dtype=float)
    return look_at(cam.pos, cam.target, up=_safe_up(view, cam.up))


def _add_world_cameras(mspec, spec: SceneSpec, info: BuildInfo) -> None:
    for cam in spec.cameras:
        if cam.mount is not None:
            continue  # attached to a robot link instead
        handle = mspec.worldbody.add_camera()
        handle.name = cam.name
        handle.pos = list(cam.pos)
        handle.fovy = cam.fovy
        handle.quat = _orient(cam)
        info.cameras[cam.name] = cam.name


def _add_equalities(mspec, mujoco, spec: SceneSpec, info: BuildInfo) -> None:
    """Body-to-body constraints, including ones a driver later releases."""
    types = {"weld": "mjEQ_WELD", "connect": "mjEQ_CONNECT"}
    for entry in spec.equalities:
        if entry.type not in types:
            raise ValueError(f"Equality {entry.name!r} has unknown type {entry.type!r}.")
        handle = mspec.add_equality()
        handle.name = entry.name
        handle.type = getattr(mujoco.mjtEq, types[entry.type])
        handle.objtype = mujoco.mjtObj.mjOBJ_BODY
        handle.name1 = entry.body1
        handle.name2 = entry.body2
        handle.active = bool(entry.active)
        handle.solref = list(entry.solref)
        info.equalities.append(entry.name)


def _attach_robot(mspec, mujoco, spec: SceneSpec, info: BuildInfo, mounted=()) -> None:
    robot_spec = spec.robot
    model = get_robot(robot_spec.key)
    info.robot = model
    info.robot_prefix = robot_spec.prefix

    robot = mujoco.MjSpec.from_file(str(model.model_path()))
    hand = robot.body(model.hand_body)

    site = hand.add_site()
    site.name = GRIP_SITE
    site.pos = [0.0, 0.0, model.grip_site_offset]
    site.size = [0.005, 0.005, 0.005]
    site.group = 4

    for cam in spec.cameras:
        if cam.mount is None:
            continue
        mount = hand if cam.mount == model.hand_body else robot.body(cam.mount)
        handle = mount.add_camera()
        handle.name = cam.name
        handle.pos = list(cam.pos)
        handle.fovy = cam.fovy
        if cam.quat is not None:
            handle.quat = list(cam.quat)
        else:
            # A mounted camera aims along a direction in its parent's frame,
            # not at a world point.
            forward = np.asarray(cam.target if cam.target is not None else (0.0, 0.0, 1.0), float)
            handle.quat = look_at(
                cam.pos, np.asarray(cam.pos) + forward, up=_safe_up(-forward, cam.up)
            )
        info.cameras[cam.name] = f"{robot_spec.prefix}{cam.name}"

    for body in mounted:
        link = hand if body.mount == model.hand_body else robot.body(body.mount)
        if link is None:
            raise ValueError(f"Body {body.name!r} mounts on unknown robot link {body.mount!r}.")
        handle = link.add_body()
        handle.name = body.name
        handle.pos = list(body.pos)
        handle.quat = list(body.quat)
        _add_geom(mujoco, handle, body)
        info.mounted_bodies[body.name] = f"{robot_spec.prefix}{body.name}"

    if robot_spec.pedestal:
        pedestal = mspec.worldbody.add_body()
        pedestal.name = "robot_pedestal"
        pedestal.pos = [robot_spec.mount_pos[0], robot_spec.mount_pos[1], robot_spec.mount_pos[2] / 2]
        column = pedestal.add_geom()
        column.name = "robot_pedestal_geom"
        column.type = mujoco.mjtGeom.mjGEOM_BOX
        column.size = [model.mount_clearance, model.mount_clearance, max(robot_spec.mount_pos[2] / 2, 0.01)]
        column.rgba = [0.26, 0.27, 0.30, 1.0]

    frame = mspec.worldbody.add_frame()
    frame.pos = list(robot_spec.mount_pos)
    frame.quat = list(robot_spec.mount_quat)
    mspec.attach(robot, prefix=robot_spec.prefix, frame=frame)
