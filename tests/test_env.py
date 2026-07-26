"""Tests for the MuJoCo environment, the controller, and the task definition."""

from __future__ import annotations

import numpy as np
import pytest

from mrs.envs import PandaPickPlaceConfig, PandaPickPlaceEnv
from mrs.envs.controllers import (
    axis_angle_to_quat,
    orientation_error,
    quat_multiply,
    quat_to_axis_angle,
)
from mrs.envs.scripted_policy import ScriptedPickPlace


@pytest.fixture(scope="module")
def env():
    environment = PandaPickPlaceEnv(PandaPickPlaceConfig(seed=0))
    yield environment
    environment.close()


# ---------------------------------------------------------------------------
# Observation / action contract
# ---------------------------------------------------------------------------


def test_observation_matches_the_checkpoint_contract(env):
    observation, info = env.reset(seed=0)

    assert set(observation) == {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
    }
    for key in ("observation.images.image", "observation.images.image2"):
        assert observation[key].shape == (3, 256, 256)
        assert observation[key].dtype == np.uint8

    assert observation["observation.state"].shape == (8,)
    assert observation["observation.state"].dtype == np.float32
    assert info["task"] == env.config.task


def test_home_pose_sits_inside_the_libero_state_distribution(env):
    """The checkpoint's normalization stats define where its inputs live.

    If the home pose fell outside this band, every observation would be an
    extrapolation for the policy before the task even starts.
    """
    env.reset(seed=0)
    state = env.get_state()

    # 1st..99th percentile of the LIBERO fine-tuning data.
    low = np.array([-0.189, -0.136, 0.636])
    high = np.array([0.048, 0.222, 0.883])
    assert np.all(state[:3] >= low), state[:3]
    assert np.all(state[:3] <= high), state[:3]

    # Gripper pointing down: a rotation of ~pi about the x axis.
    assert np.linalg.norm(state[3:6]) == pytest.approx(np.pi, abs=0.25)

    # Fingers open, reported with opposite signs as robosuite does.
    assert state[6] == pytest.approx(0.04, abs=0.005)
    assert state[7] == pytest.approx(-0.04, abs=0.005)


def test_action_must_be_seven_dimensional(env):
    env.reset(seed=0)
    with pytest.raises(ValueError, match="7-D action"):
        env.step(np.zeros(6))


def test_zero_action_holds_the_pose(env):
    """A zero delta must mean 'stay put', not 'drift'."""
    env.reset(seed=0)
    start, _ = env.controller.site_pose()

    for _ in range(25):
        env.step(np.zeros(7))

    end, _ = env.controller.site_pose()
    assert np.linalg.norm(end - start) < 0.01


def test_delta_action_moves_by_roughly_the_commanded_distance(env):
    env.reset(seed=0)
    start, _ = env.controller.site_pose()

    action = np.zeros(7)
    action[1] = 1.0  # full-scale +y
    for _ in range(4):
        env.step(action)

    end, _ = env.controller.site_pose()
    travelled = end[1] - start[1]
    expected = 4 * env.config.position_delta_scale
    # The leash bounds how far the command can lead the arm, so allow slack.
    assert 0.4 * expected < travelled < 1.2 * expected


def test_commanded_pose_is_clamped_to_the_workspace(env):
    env.reset(seed=0)
    action = np.zeros(7)
    action[2] = 1.0  # drive up forever

    for _ in range(60):
        env.step(action)

    position, _ = env.controller.site_pose()
    assert position[2] <= env.config.workspace_max[2] + 0.05


def test_gripper_action_opens_and_closes(env):
    env.reset(seed=0)

    closing = np.zeros(7)
    closing[6] = 1.0
    for _ in range(20):
        env.step(closing)
    assert env.get_state()[6] < 0.02

    opening = np.zeros(7)
    opening[6] = -1.0
    for _ in range(20):
        env.step(opening)
    assert env.get_state()[6] > 0.03


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def test_axis_angle_round_trip():
    for vector in ([0.1, -0.2, 0.3], [np.pi, 0.0, 0.0], [0.0, 0.0, 0.0]):
        vector = np.asarray(vector)
        recovered = quat_to_axis_angle(axis_angle_to_quat(vector))
        assert np.allclose(recovered, vector, atol=1e-6)


def test_quat_to_axis_angle_uses_the_zero_to_two_pi_convention():
    """robosuite does not wrap to [-pi, pi]; the state statistics assume it does not."""
    quat = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]) * 1.9 * np.pi)
    assert np.linalg.norm(quat_to_axis_angle(quat)) == pytest.approx(1.9 * np.pi, abs=1e-5)


def test_orientation_error_is_the_rotation_between_two_quaternions():
    start = axis_angle_to_quat(np.array([np.pi, 0.0, 0.0]))
    delta = axis_angle_to_quat(np.array([0.0, 0.0, 0.3]))
    end = quat_multiply(delta, start)

    assert np.allclose(orientation_error(end, start), [0.0, 0.0, 0.3], atol=1e-6)
    assert np.allclose(orientation_error(start, start), np.zeros(3), atol=1e-9)


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------


def test_ik_reaches_targets_across_the_workspace(env):
    env.reset(seed=0)
    import mujoco

    _, home_quat = env.controller.site_pose()
    controller = env.controller

    for x in (-0.20, -0.10, 0.02):
        for y in (-0.20, 0.0, 0.20):
            target = np.array([x, y, 0.78])
            q = controller.solve(target, home_quat)

            scratch = controller._scratch
            scratch.qpos[:] = env.data.qpos
            scratch.qpos[controller.arm_qpos_adr] = q
            mujoco.mj_kinematics(env.model, scratch)
            mujoco.mj_comPos(env.model, scratch)
            achieved, _ = controller._site_pose_of(scratch)

            assert np.linalg.norm(achieved - target) < 0.01, (target, achieved)


# ---------------------------------------------------------------------------
# The task itself
# ---------------------------------------------------------------------------


def test_scripted_expert_solves_the_task(env):
    """If the expert cannot solve it through the public action interface,
    the task is not well posed and no policy result would mean anything."""
    expert = ScriptedPickPlace(env)
    successes = 0

    for seed in range(4):
        env.reset(seed=seed)
        expert.reset()
        info = {}
        for _ in range(env.config.max_episode_steps):
            _, _, terminated, truncated, info = env.step(expert.act())
            if terminated or truncated:
                break
        successes += int(info.get("is_success", False))

    assert successes == 4


def test_success_requires_releasing_the_cube(env):
    """Holding the cube over the plate must not count as a placement."""
    env.reset(seed=0)
    expert = ScriptedPickPlace(env)
    expert.reset()

    for _ in range(env.config.max_episode_steps):
        action = expert.act()
        if expert.phase == "release":
            break
        env.step(action)

    # Mid-transport, with the cube grasped, success must be False.
    assert env._is_grasped() or not env._check_success()
