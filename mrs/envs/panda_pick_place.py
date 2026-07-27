"""A MuJoCo pick-and-place task on a Franka Emika Panda.

The observation and action contract deliberately mirrors the robosuite/LIBERO
setup that the pi0.5 checkpoints were fine-tuned on, so a released checkpoint
can be dropped in without an adapter:

    observation.images.image   uint8 [3, H, W]  fixed third-person camera
    observation.images.image2  uint8 [3, H, W]  wrist camera
    observation.state          float32 [8]      eef xyz, eef axis-angle,
                                                finger qpos, -finger qpos
    action                     float32 [7]      dx dy dz drx dry drz gripper,
                                                each in [-1, 1]

Gripper convention follows robosuite: +1 closes, -1 opens.
"""

from __future__ import annotations

import numpy as np

from mrs.envs.configs import PandaPickPlaceConfig
from mrs.envs.controllers import (
    DifferentialIKController,
    axis_angle_to_quat,
    orientation_error,
    quat_multiply,
    quat_to_axis_angle,
)
from mrs.envs.scene import GRIP_SITE, ROBOT_PREFIX, build_scene

ARM_JOINTS = [f"{ROBOT_PREFIX}joint{i}" for i in range(1, 8)]
FINGER_JOINTS = [f"{ROBOT_PREFIX}finger_joint1", f"{ROBOT_PREFIX}finger_joint2"]
ARM_ACTUATORS = [f"{ROBOT_PREFIX}actuator{i}" for i in range(1, 8)]
GRIPPER_ACTUATOR = f"{ROBOT_PREFIX}actuator8"

# The Panda tendon actuator is remapped to 0..255, with 255 fully open.
GRIPPER_CTRL_OPEN = 255.0
GRIPPER_CTRL_CLOSED = 0.0


class PandaPickPlaceEnv:
    """Gymnasium-style pick-and-place environment.

    Kept as a plain class rather than a `gym.Env` subclass so that the
    observation dict, which is keyed by policy feature names, is not forced
    through a `gym.spaces` description it does not need.
    """

    def __init__(self, config: PandaPickPlaceConfig | None = None):
        import mujoco

        self._mujoco = mujoco
        self.config = config or PandaPickPlaceConfig()

        self.model = build_scene(self.config)
        self.data = mujoco.MjData(self.model)

        self._resolve_ids()

        self.controller = DifferentialIKController(
            self.model,
            self.data,
            site_id=self.site_id,
            arm_joint_ids=self.arm_joint_ids,
            arm_dof_ids=self.arm_dof_ids,
            home_qpos=np.asarray(self.config.home_qpos),
            damping=self.config.ik_damping,
            max_joint_step=self.config.ik_max_joint_step,
            nullspace_gain=self.config.nullspace_gain,
            max_total_change=self.config.ik_max_total_change,
        )

        self._renderers: dict[int, object] = {}
        self._rng = np.random.default_rng(self.config.seed)

        self._joint_command = np.asarray(self.config.home_qpos, dtype=np.float64).copy()
        self._gripper_ctrl = GRIPPER_CTRL_OPEN
        self._step_count = 0
        self._success_streak = 0
        self._target_pos = np.zeros(3)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # ---- model introspection --------------------------------------------
    def _resolve_ids(self) -> None:
        mujoco = self._mujoco

        def joint_id(name: str) -> int:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint {name!r} not found in the compiled model.")
            return jid

        def actuator_id(name: str) -> int:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Actuator {name!r} not found in the compiled model.")
            return aid

        self.arm_joint_ids = np.array([joint_id(n) for n in ARM_JOINTS])
        self.finger_joint_ids = np.array([joint_id(n) for n in FINGER_JOINTS])
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_ids]
        self.finger_qpos_adr = self.model.jnt_qposadr[self.finger_joint_ids]
        self.arm_dof_ids = self.model.jnt_dofadr[self.arm_joint_ids]

        self.arm_actuator_ids = np.array([actuator_id(n) for n in ARM_ACTUATORS])
        self.gripper_actuator_id = actuator_id(GRIPPER_ACTUATOR)

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, f"{ROBOT_PREFIX}{GRIP_SITE}"
        )
        if self.site_id < 0:
            raise ValueError("Grasp site not found; the robot was attached incorrectly.")

        self._finger_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{ROBOT_PREFIX}{name}")
            for name in ("left_finger", "right_finger")
        }
        if -1 in self._finger_body_ids:
            raise ValueError("Finger bodies not found; the Panda model layout changed.")

        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        ]

        self.scene_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.scene_camera
        )
        self.wrist_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, f"{ROBOT_PREFIX}{self.config.wrist_camera}"
        )

    # ---- episode lifecycle -----------------------------------------------
    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._mujoco.mj_resetData(self.model, self.data)

        home = np.asarray(self.config.home_qpos, dtype=np.float64).copy()
        if self.config.reset_noise > 0:
            home += self._rng.uniform(-self.config.reset_noise, self.config.reset_noise, size=home.shape)

        self.data.qpos[self.arm_qpos_adr] = home
        self.data.qpos[self.finger_qpos_adr] = 0.04  # fully open

        cube_xy = self._sample_cube_position()
        self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = [
            cube_xy[0],
            cube_xy[1],
            self.config.table_height + self.config.cube_size / 2 + 0.001,
        ]
        self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        self._joint_command = home.copy()
        self._gripper_ctrl = GRIPPER_CTRL_OPEN
        self.data.ctrl[self.arm_actuator_ids] = home
        self.data.ctrl[self.gripper_actuator_id] = self._gripper_ctrl

        self._mujoco.mj_forward(self.model, self.data)

        # Let the arm settle onto its command and the cube onto the table.
        for _ in range(self.config.n_substeps * 5):
            self._mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        self._success_streak = 0

        # Seed the commanded pose from where the arm actually settled.
        self._target_pos, self._target_quat = self.controller.site_pose()

        return self.get_observation(), {"task": self.config.task}

    def _sample_cube_position(self) -> np.ndarray:
        """Sample a cube spawn that is not already inside the target plate."""
        target = np.asarray(self.config.target_pos)
        for _ in range(100):
            xy = np.array(
                [
                    self._rng.uniform(*self.config.cube_spawn_x),
                    self._rng.uniform(*self.config.cube_spawn_y),
                ]
            )
            if np.linalg.norm(xy - target) > self.config.target_radius + self.config.cube_size:
                return xy
        # Spawn ranges and target overlap; fall back to the far corner.
        return np.array([self.config.cube_spawn_x[0], self.config.cube_spawn_y[0]])

    # ---- stepping ---------------------------------------------------------
    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if action.shape[0] != 7:
            raise ValueError(f"Expected a 7-D action, got shape {action.shape}.")

        current_pos, _ = self.controller.site_pose()

        # Integrate the commanded pose rather than re-deriving it from the
        # measurement, so a delta of zero holds the pose exactly instead of
        # letting tracking error accumulate into a drift.
        self._target_pos = self._target_pos + action[:3] * self.config.position_delta_scale
        delta_quat = axis_angle_to_quat(action[3:6] * self.config.rotation_delta_scale)
        self._target_quat = quat_multiply(delta_quat, self._target_quat)
        self._target_quat /= np.linalg.norm(self._target_quat)

        self._apply_leash(current_pos)

        target_pos, target_quat = self._target_pos, self._target_quat

        self._gripper_ctrl = float(
            GRIPPER_CTRL_OPEN + (GRIPPER_CTRL_CLOSED - GRIPPER_CTRL_OPEN) * (action[6] + 1.0) / 2.0
        )

        # Resolve the pose target to a joint target once, then let the Panda's
        # position actuators track it over the control interval.
        self._joint_command = self.controller.solve(target_pos, target_quat)
        self.data.ctrl[self.arm_actuator_ids] = self._joint_command
        self.data.ctrl[self.gripper_actuator_id] = self._gripper_ctrl

        for _ in range(self.config.n_substeps):
            self._mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        observation = self.get_observation()
        success = self._check_success()
        self._success_streak = self._success_streak + 1 if success else 0

        terminated = self._success_streak >= self.config.success_hold_steps
        truncated = self._step_count >= self.config.max_episode_steps
        reward = 1.0 if terminated else 0.0

        info = {
            "task": self.config.task,
            "is_success": terminated,
            "cube_position": self.cube_position.tolist(),
            "eef_position": current_pos.tolist(),
            "step": self._step_count,
        }
        return observation, reward, terminated, truncated, info

    def _apply_leash(self, measured_pos: np.ndarray) -> None:
        """Keep the commanded pose inside the workspace and near the real arm."""
        config = self.config

        self._target_pos = np.clip(
            self._target_pos, np.asarray(config.workspace_min), np.asarray(config.workspace_max)
        )

        offset = self._target_pos - measured_pos
        distance = float(np.linalg.norm(offset))
        if distance > config.position_leash:
            self._target_pos = measured_pos + offset * (config.position_leash / distance)

        _, measured_quat = self.controller.site_pose()
        rotation_error = orientation_error(self._target_quat, measured_quat)
        angle = float(np.linalg.norm(rotation_error))
        if angle > config.rotation_leash:
            clamped = axis_angle_to_quat(rotation_error * (config.rotation_leash / angle))
            self._target_quat = quat_multiply(clamped, measured_quat)

    # ---- observation ------------------------------------------------------
    @property
    def cube_position(self) -> np.ndarray:
        return self.data.xpos[self.cube_body_id].copy()

    def get_state(self) -> np.ndarray:
        """The 8-D proprioceptive vector the policy consumes."""
        position, quat = self.controller.site_pose()
        axis_angle = quat_to_axis_angle(quat)
        fingers = self.data.qpos[self.finger_qpos_adr]
        # robosuite reports the two finger joints with opposite signs.
        return np.concatenate(
            [position, axis_angle, [fingers[0], -fingers[1]]]
        ).astype(np.float32)

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            self.config.scene_image_key: self.render_camera(
                self.scene_camera_id, self.config.image_size
            ),
            self.config.wrist_image_key: self.render_camera(
                self.wrist_camera_id, self.config.image_size
            ),
            "observation.state": self.get_state(),
        }

    def _check_success(self) -> bool:
        """The cube is resting on the plate, near its centre, and released."""
        cube = self.cube_position
        target = np.asarray(self.config.target_pos)

        if float(np.linalg.norm(cube[:2] - target)) > self.config.success_radius:
            return False

        resting_height = self.config.table_height + 0.01 + self.config.cube_size / 2
        if abs(cube[2] - resting_height) > 0.02:
            return False

        if self._is_grasped():
            return False

        cube_dof = self.model.jnt_dofadr[
            self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        ]
        # Loose enough to tolerate the contact chatter of a box settling on the
        # plate, tight enough to exclude a cube that is still being moved.
        return bool(np.linalg.norm(self.data.qvel[cube_dof : cube_dof + 3]) < 0.05)

    def _is_grasped(self) -> bool:
        """True while either fingertip is in contact with the cube."""
        cube_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.cube_body_id).tolist())
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if not pair & cube_geoms:
                continue
            other = (pair - cube_geoms).pop() if len(pair - cube_geoms) else None
            if other is not None and self.model.geom_bodyid[other] in self._finger_body_ids:
                return True
        return False

    # ---- rendering ---------------------------------------------------------
    def _renderer(self, size: int):
        if size not in self._renderers:
            self._renderers[size] = self._mujoco.Renderer(self.model, height=size, width=size)
        return self._renderers[size]

    def render_camera(self, camera_id: int, size: int) -> np.ndarray:
        """Render one camera as a `uint8 [3, size, size]` CHW image."""
        renderer = self._renderer(size)
        renderer.update_scene(self.data, camera=camera_id)
        frame = renderer.render()
        return np.ascontiguousarray(frame.transpose(2, 0, 1))

    def render(self, size: int | None = None) -> np.ndarray:
        """A human-facing `uint8 [H, W, 3]` frame from the scene camera."""
        size = size or self.config.render_size
        renderer = self._renderer(size)
        renderer.update_scene(self.data, camera=self.scene_camera_id)
        return renderer.render()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
