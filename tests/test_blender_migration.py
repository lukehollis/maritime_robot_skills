"""Blender scene graph -> SceneSpec conversion.

These run without Blender: `build_env.py` never imports `bpy`, so a synthetic
scene graph exercises the whole conversion. Each test here corresponds to a
mismatch found by comparing a compiled MuJoCo model against the Blender scene
that produced it, body by body.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_env", REPO_ROOT / ".claude" / "scripts" / "env" / "build_env.py"
)
build_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_env)


def graph(*objects, **extra):
    return {"kit_version": 1, "blender": "test", "unit_scale": 1.0,
            "objects": list(objects), "animation": {}, **extra}


def obj(name, role="static", pos=(0, 0, 0), quat=(1, 0, 0, 0), dimensions=(0.1, 0.1, 0.1), **props):
    lo = [pos[i] - dimensions[i] / 2 for i in range(3)]
    hi = [pos[i] + dimensions[i] / 2 for i in range(3)]
    return {"name": name, "type": "MESH", "role": role, "pos": list(pos), "quat": list(quat),
            "scale": [1, 1, 1], "dimensions": list(dimensions), "extent": list(dimensions),
            "bounds_min": lo, "bounds_max": hi, "parent": None,
            "props": {"role": role, **props}}


def test_full_extents_become_half_extents():
    spec = build_env.spec_from_graph(
        graph(obj("crate", dimensions=(0.20, 0.10, 0.06))), name="s", task="t")
    assert spec.body("crate").size == pytest.approx((0.10, 0.05, 0.03))


def test_rotation_survives_unchanged():
    quat = (0.9659258, 0.0, 0.0, 0.2588190)  # 30 degrees about z
    spec = build_env.spec_from_graph(graph(obj("yaw", quat=quat)), name="s", task="t")
    assert spec.body("yaw").quat == pytest.approx(quat)


def test_nested_body_position_is_relative_to_its_parent():
    """MuJoCo nests a child inside its parent's frame.

    Blender exports world transforms, so passing them straight through applies
    the parent transform twice — a drawer ends up displaced by the cabinet's
    own offset.
    """
    scene = graph(
        obj("cabinet", pos=(0.0, -0.6, 0.45), dimensions=(0.3, 0.3, 0.3)),
        obj("drawer", role="sliding", pos=(0.0, -0.6, 0.55), dimensions=(0.26, 0.26, 0.10),
            parent_body="cabinet", joint_axis=[0, 1, 0], joint_range=[0.0, 0.25]),
    )
    drawer = build_env.spec_from_graph(scene, name="s", task="t").body("drawer")

    assert drawer.parent == "cabinet"
    assert drawer.pos == pytest.approx((0.0, 0.0, 0.10), abs=1e-9)


def test_nested_body_under_a_rotated_parent():
    """The relative transform must undo the parent's rotation too."""
    half = math.sqrt(0.5)  # 90 degrees about z
    scene = graph(
        obj("base", pos=(1.0, 0.0, 0.0), quat=(half, 0.0, 0.0, half)),
        obj("arm", role="hinged", pos=(1.0, 0.2, 0.0), quat=(half, 0.0, 0.0, half),
            parent_body="base", joint_axis=[0, 0, 1]),
    )
    arm = build_env.spec_from_graph(scene, name="s", task="t").body("arm")

    # Yawing the parent +90 degrees points its local +x along world +y, so a
    # child displaced along world +y sits 0.2 m along the parent's local +x.
    assert arm.pos == pytest.approx((0.2, 0.0, 0.0), abs=1e-6)
    assert arm.quat == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-6)


def test_unknown_parent_is_an_error_not_a_silent_reparent():
    scene = graph(obj("thing", parent_body="nonexistent"))
    with pytest.raises(ValueError, match="not in the scene graph"):
        build_env.spec_from_graph(scene, name="s", task="t")


def test_decor_is_visible_but_not_collidable():
    spec = build_env.spec_from_graph(
        graph(obj("sign", role="decor"), obj("wall", role="static")), name="s", task="t")

    assert spec.body("sign").contype == 0 and spec.body("sign").conaffinity == 0
    assert spec.body("wall").contype == 1


def test_mesh_collision_requires_an_exported_file():
    scene = graph(obj("blob", role="free", collision="mesh"))
    with pytest.raises(ValueError, match="mesh_file"):
        build_env.spec_from_graph(scene, name="s", task="t")


def test_velocity_actuator_gets_armature():
    """Without rotor inertia a velocity servo on a light body diverges."""
    scene = graph(obj("roller", role="hinged", joint_axis=[0, 1, 0],
                      actuator="velocity", actuator_kv=5.0))
    roller = build_env.spec_from_graph(scene, name="s", task="t").body("roller")
    assert roller.joint.armature > 0.0


def test_conveyor_marker_becomes_a_dynamic_element_not_a_body():
    scene = graph(obj("infeed", role="ignore", dynamic="roller_conveyor",
                      pos=(-0.1, 0.0, 0.66), direction="+y", length=0.4,
                      width=0.2, roller_radius=0.025, speed=0.05))
    spec = build_env.spec_from_graph(scene, name="s", task="t")

    assert [b.name for b in spec.bodies] == []
    assert len(spec.dynamics) == 1
    assert spec.dynamics[0].kind == "roller_conveyor"
    assert spec.dynamics[0].params["origin"] == (-0.1, 0.0, 0.66)


def test_robot_mount_sets_the_arm_pose_and_is_not_a_body():
    scene = graph(obj("robot_base", role="robot_mount", pos=(-0.56, 0.0, 0.63), robot="panda"))
    spec = build_env.spec_from_graph(scene, name="s", task="t")

    assert spec.robot.key == "panda"
    assert spec.robot.mount_pos == (-0.56, 0.0, 0.63)
    assert [b.name for b in spec.bodies] == []


def test_scene_and_wrist_cameras_are_always_present():
    """The two-image observation contract cannot be satisfied without them."""
    spec = build_env.spec_from_graph(graph(obj("thing")), name="s", task="t")
    roles = {camera.role for camera in spec.cameras}
    assert {"scene", "wrist"} <= roles


def test_animation_only_drives_mocap_bodies():
    samples = [[0.0, 0, 0, 0.7, 1, 0, 0, 0], [1.0, 0, 0.3, 0.7, 1, 0, 0, 0]]
    scene = graph(
        obj("ghost", role="mocap"),
        obj("solid", role="free"),
        animation={"ghost": samples, "solid": samples},
    )
    spec = build_env.spec_from_graph(scene, name="s", task="t")

    baked = [d for d in spec.dynamics if d.kind == "baked"]
    assert [d.params["body"] for d in baked] == ["ghost"]


def test_migrated_reference_env_still_compiles():
    """End-to-end guard on the checked-in package."""
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from mrs.envs.scenegen import SceneSpec, build_model

    package = REPO_ROOT / "envs" / "mail_sorting"
    if not (package / "spec.json").is_file():
        pytest.skip("reference environment not present")

    spec = SceneSpec.load(package / "spec.json")
    model, info = build_model(spec, asset_dir=package / "assets")
    assert model.nbody > 10
    assert info.warnings == []
    assert np.isfinite(model.body_pos).all()
