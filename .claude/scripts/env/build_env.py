#!/usr/bin/env python3
"""Turn a Blender `scene_graph.json` into a runnable environment package.

    python .claude/scripts/env/build_env.py \
        --graph outputs/mail_sorting/scene_graph.json \
        --out envs/mail_sorting \
        --task "sort the envelopes by size into the matching bins"

Writes `envs/<slug>/spec.json` (plus `assets/` if the graph exported meshes),
which `mrs.envs.scenegen.load_env` reads. No `bpy` import happens here, so the
conversion runs — and is testable — without Blender.

The conversion is intentionally lossless in one direction only. Blender is the
authoring surface: shading, modifiers and non-convex detail stay there. What
crosses over is the part MuJoCo can simulate — transforms, extents, collision
proxies, mass, friction, joints and actuators — and the `mrs_*` properties that
say which is which.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrs.envs.scenegen.spec import (  # noqa: E402
    ActuatorSpec, BodySpec, CameraSpec, ControlSpec, DynamicSpec, EqualitySpec,
    JointSpec, MaterialSpec, RobotSpec, SceneSpec, SuccessSpec, WorldSpec,
)

ROLE_TO_KIND = {
    "static": "static",
    "decor": "static",
    "free": "free",
    "hinged": "hinged",
    "sliding": "sliding",
    "mocap": "mocap",
}

# Blender's cylinder primitive extends along its local z, matching MuJoCo's.
SHAPE_FROM_COLLISION = {
    "box": "box",
    "cylinder": "cylinder",
    "sphere": "sphere",
    "capsule": "capsule",
    "mesh": "mesh",
}


def _as_list(value, length=None, default=None):
    if value is None:
        return default
    out = [float(v) for v in (value if isinstance(value, (list, tuple)) else [value])]
    if length is not None and len(out) != length:
        raise ValueError(f"Expected {length} numbers, got {out!r}")
    return out


def _tags(props) -> list[str]:
    raw = props.get("tags")
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def _half(dimensions, shape):
    """MuJoCo geom size from a Blender object's local extent."""
    x, y, z = (max(float(v), 1e-5) for v in dimensions)
    if shape == "box":
        return (x / 2.0, y / 2.0, z / 2.0)
    if shape in ("cylinder", "capsule"):
        return (max(x, y) / 2.0, z / 2.0)
    if shape == "sphere":
        return (max(x, y, z) / 2.0,)
    return (x / 2.0, y / 2.0, z / 2.0)


def _quat_conjugate(q):
    return [q[0], -q[1], -q[2], -q[3]]


def _quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _quat_rotate(q, v):
    """Rotate vector `v` by quaternion `q` (w, x, y, z)."""
    w, x, y, z = q
    t = [2.0 * (y * v[2] - z * v[1]), 2.0 * (z * v[0] - x * v[2]), 2.0 * (x * v[1] - y * v[0])]
    return [
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    ]


def _to_parent_frame(child_pos, child_quat, parent_pos, parent_quat):
    """World transform -> transform relative to the parent body.

    Blender exports world transforms, but MuJoCo interprets a nested body's
    `pos`/`quat` relative to its parent. Passing world values straight through
    applies the parent transform a second time, which puts a drawer inside a
    cabinet at the cabinet's offset again.
    """
    delta = [child_pos[i] - parent_pos[i] for i in range(3)]
    inverse = _quat_conjugate(parent_quat)
    return _quat_rotate(inverse, delta), _quat_multiply(inverse, child_quat)


def _spawn_range(props):
    ranges = {}
    for axis in ("x", "y", "z", "yaw"):
        value = props.get(f"spawn_{axis}")
        if value is not None:
            pair = _as_list(value, 2)
            ranges[axis] = (pair[0], pair[1])
    return ranges or None


def body_from_entry(entry: dict, world: dict[str, dict] | None = None) -> BodySpec | None:
    props = entry.get("props", {})
    role = entry.get("role", "static")
    if role in ("ignore", "robot_mount", "camera"):
        return None

    # `decor` is documented as visible-but-not-collidable, and the hand-written
    # reference scene marks its table legs the same way.
    collision = str(props.get("collision", "none" if role == "decor" else "box")).lower()
    if collision == "none":
        # Visual-only: keep it in the model so renders match Blender, but take
        # it out of the contact solve entirely.
        contype = conaffinity = 0
        shape = "box"
    else:
        contype = conaffinity = 1
        shape = SHAPE_FROM_COLLISION.get(collision, "box")

    if shape == "mesh" and not entry.get("mesh_file"):
        raise ValueError(
            f"Object {entry['name']!r} is tagged collision='mesh' but the graph has no mesh_file. "
            f"Re-export with mesh_roles including its role."
        )

    kind = ROLE_TO_KIND.get(role, "static")
    joint = None
    actuator = None

    if kind in ("hinged", "sliding"):
        joint_range = props.get("joint_range")
        joint = JointSpec(
            type="hinge" if kind == "hinged" else "slide",
            axis=tuple(_as_list(props.get("joint_axis"), 3, [0.0, 0.0, 1.0])),
            range=tuple(_as_list(joint_range, 2)) if joint_range is not None else None,
            damping=float(props.get("joint_damping", 0.1)),
            stiffness=float(props.get("joint_stiffness", 0.0)),
            springref=float(props.get("joint_springref", 0.0)),
            armature=float(props.get("joint_armature", 0.0)),
        )
        kinds = props.get("actuator")
        if kinds:
            ctrlrange = props.get("actuator_ctrlrange", joint_range)
            actuator = ActuatorSpec(
                kind=str(kinds),
                kp=float(props.get("actuator_kp", 100.0)),
                kv=float(props.get("actuator_kv", 10.0)),
                ctrlrange=tuple(_as_list(ctrlrange, 2)) if ctrlrange is not None else None,
                default_ctrl=float(props.get("actuator_default", 0.0)),
            )
            # A velocity servo on a light body needs rotor inertia to stay
            # stable; mirror the rule dynamics.py applies to its own macros.
            if actuator.kind in ("velocity", "intvelocity") and joint.armature == 0.0:
                joint.armature = max(actuator.kv * 0.002, 1e-4)

    parent_name = props.get("parent_body")
    pos, quat = list(entry["pos"]), list(entry["quat"])
    if props.get("mount"):
        # `mount_pos`/`mount_quat` give the pose in the robot link's frame
        # directly, because there is no Blender object to measure it against —
        # the link only exists after the Menagerie model is attached.
        pos = _as_list(props.get("mount_pos"), 3, [0.0, 0.0, 0.0])
        quat = _as_list(props.get("mount_quat"), 4, [1.0, 0.0, 0.0, 0.0])
    if parent_name:
        parent = (world or {}).get(parent_name)
        if parent is None:
            raise ValueError(
                f"Object {entry['name']!r} names parent_body {parent_name!r}, "
                f"which is not in the scene graph."
            )
        pos, quat = _to_parent_frame(pos, quat, parent["pos"], parent["quat"])

    rgba = props.get("rgba")
    return BodySpec(
        name=entry["name"],
        kind=kind,
        shape=shape,
        size=_half(entry.get("dimensions") or entry["extent"], shape),
        pos=tuple(pos),
        quat=tuple(quat),
        parent=parent_name,
        mount=props.get("mount"),
        rgba=tuple(_as_list(rgba, 4)) if rgba else None,
        mass=float(props["mass"]) if "mass" in props else None,
        density=float(props["density"]) if "density" in props and "mass" not in props else None,
        friction=tuple(_as_list(props.get("friction"), 3, [1.0, 0.005, 0.0001])),
        condim=int(props.get("condim", 3)),
        solref=tuple(_as_list(props.get("solref"), 2, [0.02, 1.0])),
        contype=contype,
        conaffinity=conaffinity,
        mesh_file=entry.get("mesh_file"),
        joint=joint,
        actuator=actuator,
        spawn_range=_spawn_range(props),
        tags=_tags(props),
    )


def equalities_from_graph(graph: dict) -> list[EqualitySpec]:
    """`mrs_weld_to` on a body becomes a weld constraint to the named body."""
    out = []
    for entry in graph["objects"]:
        other = entry.get("props", {}).get("weld_to")
        if not other:
            continue
        out.append(EqualitySpec(name=f"{entry['name']}_weld",
                                body1=other, body2=entry["name"], type="weld"))
    return out


def dynamics_from_graph(graph: dict) -> list[DynamicSpec]:
    elements: list[DynamicSpec] = []

    for entry in graph["objects"]:
        props = entry.get("props", {})
        kind = props.get("dynamic")
        if not kind:
            continue

        if kind == "roller_conveyor":
            params = {
                "origin": tuple(entry["pos"]),
                "direction": props.get("direction", "+y"),
                "length": float(props.get("length", 0.5)),
                "width": float(props.get("width", 0.2)),
                "roller_radius": float(props.get("roller_radius", 0.03)),
                "speed": float(props.get("speed", 0.1)),
            }
            for optional in ("spacing", "roller_mass", "kv", "material", "rail_height",
                             "end_stop", "end_stop_height", "side_rails",
                             "stop_body", "stop_at"):
                if optional in props:
                    params[optional] = props[optional]
            if "duty_period" in props:
                params["duty"] = {"period": float(props["duty_period"]),
                                  "on_fraction": float(props.get("duty_on_fraction", 0.5))}
            elements.append(DynamicSpec(name=entry["name"], kind="roller_conveyor", params=params))

        elif kind == "turntable":
            elements.append(DynamicSpec(name=entry["name"], kind="turntable", params={
                "center": tuple(entry["pos"]),
                "radius": float(props.get("radius", max(entry["extent"][0], entry["extent"][1]) / 2)),
                "thickness": float(props.get("thickness", entry["extent"][2])),
                "angular_speed": float(props.get("angular_speed", 1.0)),
            }))

        elif kind == "mover":
            elements.append(DynamicSpec(name=f"{entry['name']}_mover", kind="mover", params={
                "body": entry["name"],
                "path": props.get("path", "harmonic"),
                "axis": _as_list(props.get("axis"), 3, [0.0, 1.0, 0.0]),
                "amplitude": float(props.get("amplitude", 0.2)),
                "period": float(props.get("period", 4.0)),
            }))

        elif kind == "severable":
            elements.append(DynamicSpec(name=f"{entry['name']}_sever", kind="severable", params={
                "equality": props.get("equality", f"{entry['name']}_weld"),
                "blade": props["blade"],
                "site": props.get("site", entry["name"]),
                "freed_body": props.get("freed_body", entry["name"]),
                "radius": float(props.get("radius", 0.05)),
                "min_speed": float(props.get("min_speed", 0.15)),
                "transfer": float(props.get("transfer", 0.35)),
                **({"cut_direction": _as_list(props["cut_direction"], 3)}
                   if "cut_direction" in props else {}),
            }))

        elif kind == "joint_cycle":
            elements.append(DynamicSpec(name=f"{entry['name']}_cycle", kind="joint_cycle", params={
                "body": entry["name"],
                "low": float(props.get("cycle_low", 0.0)),
                "high": float(props.get("cycle_high", 1.0)),
                "period": float(props.get("period", 8.0)),
            }))
        else:
            raise ValueError(f"Object {entry['name']!r} has unknown mrs_dynamic={kind!r}.")

    # Anything the author keyframed becomes a replayed trajectory. Only mocap
    # bodies can be driven this way: a body with real degrees of freedom would
    # fight the physics rather than follow the curve.
    mocap = {e["name"] for e in graph["objects"] if e.get("role") == "mocap"}
    for name, samples in (graph.get("animation") or {}).items():
        if name not in mocap:
            continue
        elements.append(DynamicSpec(name=f"{name}_baked", kind="baked",
                                    params={"target": "mocap", "body": name,
                                            "samples": samples, "loop": True}))
    return elements


def robot_from_graph(graph: dict, default_key="panda") -> RobotSpec | None:
    for entry in graph["objects"]:
        if entry.get("role") != "robot_mount":
            continue
        props = entry.get("props", {})
        return RobotSpec(
            key=str(props.get("robot", default_key)),
            mount_pos=tuple(entry["pos"]),
            mount_quat=tuple(entry["quat"]),
            home_qpos=tuple(_as_list(props["home_qpos"])) if "home_qpos" in props else None,
            pedestal=bool(props.get("pedestal", True)),
        )
    return None


def cameras_from_graph(graph: dict) -> list[CameraSpec]:
    cameras = []
    for entry in graph["objects"]:
        if entry.get("role") != "camera":
            continue
        props = entry.get("props", {})
        target = props.get("target")
        cameras.append(CameraSpec(
            name=entry["name"],
            pos=tuple(entry["pos"]),
            target=tuple(_as_list(target, 3)) if target else None,
            quat=None if target else tuple(entry["quat"]),
            fovy=float(props.get("fovy", 45.0)),
            mount=props.get("mount"),
            role=str(props.get("camera_role", "inspection")),
        ))
    return cameras


DEFAULT_CAMERAS = [
    CameraSpec(name="agentview", pos=(0.86, 0.0, 1.05), target=(-0.06, 0.0, 0.70),
               fovy=45.0, role="scene"),
    CameraSpec(name="wrist", pos=(0.06, 0.0, -0.055), target=(0.30, 0.0, 1.0),
               fovy=72.0, mount="hand", up=(0.0, -1.0, 0.0), role="wrist"),
    CameraSpec(name="topdown", pos=(0.0, 0.0, 1.45), target=(0.0, 0.0, 0.66),
               fovy=55.0, role="inspection"),
    CameraSpec(name="side", pos=(0.05, -1.05, 0.95), target=(0.0, 0.0, 0.70),
               fovy=45.0, role="inspection"),
]


def spec_from_graph(graph: dict, *, name: str, task: str, seed: int | None = 0) -> SceneSpec:
    world = {entry["name"]: entry for entry in graph["objects"]}
    bodies = [b for b in (body_from_entry(e, world) for e in graph["objects"]) if b is not None]

    cameras = cameras_from_graph(graph)
    have = {c.role for c in cameras}
    for fallback in DEFAULT_CAMERAS:
        # The two-image observation contract needs a scene and a wrist camera.
        # Supply the reference placements rather than failing, and say so.
        if fallback.role in ("scene", "wrist") and fallback.role not in have:
            cameras.append(fallback)
        elif fallback.role == "inspection" and not any(c.role == "inspection" for c in cameras):
            cameras.append(fallback)

    return SceneSpec(
        name=name,
        task=task,
        world=WorldSpec(),
        materials=[],
        bodies=bodies,
        equalities=equalities_from_graph(graph),
        dynamics=dynamics_from_graph(graph),
        robot=robot_from_graph(graph),
        cameras=cameras,
        control=ControlSpec(),
        success=SuccessSpec(),  # filled in by robo-task-define
        seed=seed,
        provenance={
            "source": "blender",
            "blender": graph.get("blender"),
            "kit_version": graph.get("kit_version"),
        },
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path, required=True, help="scene_graph.json from Blender")
    parser.add_argument("--out", type=Path, required=True, help="environment package directory")
    parser.add_argument("--task", required=True, help="the language instruction the policy receives")
    parser.add_argument("--name", default=None, help="scene name (default: the output directory name)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    graph = json.loads(args.graph.read_text())
    name = args.name or args.out.name

    spec = spec_from_graph(graph, name=name, task=args.task, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    spec.save(args.out / "spec.json")

    # Meshes travel beside the spec so the package is self-contained.
    source_assets = args.graph.parent / "assets"
    copied = 0
    if source_assets.is_dir():
        import shutil

        target = args.out / "assets"
        target.mkdir(exist_ok=True)
        for mesh in source_assets.iterdir():
            if mesh.is_file():
                shutil.copy2(mesh, target / mesh.name)
                copied += 1

    kinds = {}
    for body in spec.bodies:
        kinds[body.kind] = kinds.get(body.kind, 0) + 1

    print(f"wrote {args.out / 'spec.json'}")
    print(f"  bodies    : {len(spec.bodies)}  ({', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))})")
    print(f"  dynamics  : {[d.kind for d in spec.dynamics] or 'none'}")
    print(f"  robot     : {spec.robot.key if spec.robot else 'NONE — scene cannot be stepped'}")
    print(f"  cameras   : {[(c.name, c.role) for c in spec.cameras]}")
    print(f"  meshes    : {copied}")
    if not spec.success.terms:
        print("  success   : EMPTY — hand off to robo-task-define before evaluating")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
