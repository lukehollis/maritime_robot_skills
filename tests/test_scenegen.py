"""Tests for spec-driven scene generation.

Deliberately cheap: every test here builds a minimal scene rather than the
mail-sorting reference, so the suite stays in the same second-scale budget as
the rest of `pytest -m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from mrs.envs.scenegen import (
    ActuatorSpec,
    BodySpec,
    CameraSpec,
    ControlSpec,
    DynamicSpec,
    JointSpec,
    RobotSpec,
    SceneEnv,
    SceneSpec,
    SuccessSpec,
    build_model,
)

TABLE_TOP = 0.63


def minimal_spec(**overrides) -> SceneSpec:
    spec = SceneSpec(
        name="unit_scene",
        task="pick up the block",
        bodies=[
            BodySpec(name="table", kind="static", shape="box", size=(0.45, 0.60, 0.025),
                     pos=(0.05, 0.0, TABLE_TOP - 0.025)),
            BodySpec(name="block", kind="free", shape="box", size=(0.02, 0.02, 0.02),
                     pos=(-0.05, 0.0, TABLE_TOP + 0.021), mass=0.05,
                     tags=["target"]),
        ],
        cameras=[
            CameraSpec(name="agentview", pos=(0.88, 0.0, 1.02),
                       target=(-0.14, 0.0, 0.66), role="scene"),
            CameraSpec(name="wrist", pos=(0.06, 0.0, -0.055), target=(0.30, 0.0, 1.0),
                       fovy=72.0, mount="hand", up=(0.0, -1.0, 0.0), role="wrist"),
        ],
        robot=RobotSpec(key="panda", mount_pos=(-0.56, 0.0, TABLE_TOP)),
        control=ControlSpec(max_episode_steps=20),
        seed=0,
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def test_spec_round_trips_through_json(tmp_path):
    spec = minimal_spec()
    reloaded = SceneSpec.load(spec.save(tmp_path / "spec.json"))

    assert reloaded.name == spec.name
    assert [b.name for b in reloaded.bodies] == [b.name for b in spec.bodies]
    # Nested dataclasses must come back as dataclasses, not dicts.
    assert isinstance(reloaded.control, ControlSpec)
    assert isinstance(reloaded.cameras[0], CameraSpec)
    assert reloaded.body("block").tags == ["target"]


def test_observation_matches_the_reference_contract():
    env = SceneEnv(minimal_spec())
    observation, info = env.reset(seed=0)

    size = env.config.image_size
    for key in (env.config.scene_image_key, env.config.wrist_image_key):
        assert observation[key].shape == (3, size, size)
        assert observation[key].dtype == np.uint8

    state = observation["observation.state"]
    assert state.shape == (8,) and state.dtype == np.float32
    assert info["task"] == "pick up the block"
    env.close()


def test_home_pose_sits_inside_the_workspace_box():
    """Mirrors tests/test_env.py: the checkpoint's statistics assume it."""
    env = SceneEnv(minimal_spec())
    env.reset(seed=0)
    position, _ = env.controller.site_pose()

    assert np.all(position >= np.asarray(env.config.workspace_min))
    assert np.all(position <= np.asarray(env.config.workspace_max))
    env.close()


def test_free_body_spawn_range_is_resampled_per_seed():
    spec = minimal_spec()
    spec.body("block").spawn_range = {"x": (-0.12, 0.02), "y": (-0.16, 0.06)}
    env = SceneEnv(spec)

    env.reset(seed=1)
    first = env.body_position("block").copy()
    env.reset(seed=2)
    second = env.body_position("block").copy()

    assert not np.allclose(first[:2], second[:2])
    env.close()


def test_conveyor_expands_and_transports():
    spec = minimal_spec()
    spec.bodies = [b for b in spec.bodies if b.name != "block"]
    # Rest the parcel *on* the roller crowns (0.69 + 0.025) with 2 mm to spare.
    # Starting it a few millimetres low buries it between rollers and it never
    # gets carried, which is the most common authoring error this macro sees.
    spec.bodies.append(
        BodySpec(name="parcel", kind="free", shape="box", size=(0.04, 0.03, 0.004),
                 pos=(-0.05, -0.10, 0.721), mass=0.02, condim=4, friction=(1.1, 0.02, 0.001))
    )
    spec.dynamics = [
        DynamicSpec(name="belt", kind="roller_conveyor", params={
            "origin": (-0.05, 0.0, 0.69), "direction": "+y", "length": 0.30,
            "width": 0.16, "roller_radius": 0.025, "spacing": 0.06, "speed": 0.08,
        })
    ]
    env = SceneEnv(spec)

    rollers = [n for n in env.build.expanded_bodies["belt"] if "_roller_" in n]
    assert len(rollers) >= 2
    assert all(name in env.build.actuators for name in rollers)

    env.reset(seed=0)
    start = env.body_position("parcel").copy()
    for _ in range(120):
        env.step(np.zeros(7))
    travelled = env.body_position("parcel") - start

    # A positive speed must drive along +direction, not against it.
    assert travelled[1] > 0.02, f"parcel moved {travelled} instead of along +y"
    env.close()


def test_euler_with_a_velocity_actuator_is_flagged():
    """The failure it prevents is silent: MuJoCo auto-resets on divergence."""
    spec = minimal_spec()
    spec.control.integrator = "euler"
    spec.bodies.append(
        BodySpec(name="spinner", kind="hinged", shape="cylinder", size=(0.03, 0.02),
                 pos=(0.2, 0.2, 0.7), mass=0.2,
                 joint=JointSpec(type="hinge", axis=(0.0, 0.0, 1.0)),
                 actuator=ActuatorSpec(kind="velocity", kv=5.0))
    )
    _, info = build_model(spec)
    assert any("implicitfast" in warning for warning in info.warnings)


def test_success_and_failure_predicates_evaluate():
    from mrs.envs.scenegen import success as success_module

    spec = minimal_spec()
    spec.success = SuccessSpec(
        mode="all",
        terms=[{"predicate": "near", "body": "block", "target": [-0.05, 0.0, 0.65],
                "radius": 0.05}],
        failure_terms=[{"name": "dropped", "predicate": "below_height",
                        "body": "block", "height": 0.30}],
    )
    env = SceneEnv(spec)
    env.reset(seed=0)
    context = success_module.Context(env)

    assert success_module.evaluate(context, spec.success) is True
    assert success_module.failures(context, spec.success) == []
    env.close()


def test_unknown_robot_names_the_registered_ones():
    from mrs.envs.scenegen import get_robot

    with pytest.raises(ValueError, match="panda"):
        get_robot("definitely_not_a_robot")

    # fr3 ships without a gripper; the error must say so rather than compile
    # a scene the policy interface cannot drive.
    with pytest.raises(ValueError, match="gripper"):
        get_robot("fr3")
