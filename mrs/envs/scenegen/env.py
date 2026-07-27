"""A spec-driven MuJoCo environment.

`SceneEnv` is `PandaPickPlaceEnv` with the scene taken out of the code and put
into a `SceneSpec`. The observation and action contract is unchanged, which is
the whole point: a generated scene evaluates the same pi0.5 checkpoints, through
the same `mrs.rollout.evaluate`, as the hand-written one.

    observation.images.image   uint8 [3, H, W]  fixed third-person camera
    observation.images.image2  uint8 [3, H, W]  wrist camera
    observation.state          float32 [8]      eef xyz, eef axis-angle,
                                                finger qpos, -finger qpos
    action                     float32 [7]      dx dy dz drx dry drz gripper

What is genuinely new is that the scene may be moving. Drivers are applied
inside the physics loop, once per substep rather than once per control step, so
a conveyor advances smoothly between the policy's 20 Hz decisions instead of
teleporting 50 mm at a time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mrs.envs.controllers import (
    DifferentialIKController,
    axis_angle_to_quat,
    orientation_error,
    quat_multiply,
    quat_to_axis_angle,
)
from mrs.envs.scene import GRIP_SITE
from mrs.envs.scenegen import dynamics as dynamics_module
from mrs.envs.scenegen import success as success_module
from mrs.envs.scenegen.builder import build_model
from mrs.envs.scenegen.spec import SceneSpec

logger = logging.getLogger(__name__)


class SceneEnv:
    """Gymnasium-style environment over a `SceneSpec`."""

    def __init__(self, spec: SceneSpec, *, asset_dir: str | Path | None = None):
        import mujoco

        self._mujoco = mujoco
        self.spec = spec
        self.config = spec.control  # rollout.py reads env.config.*

        self.model, self.build = build_model(spec, asset_dir=asset_dir)
        self.data = mujoco.MjData(self.model)

        for warning in self.build.warnings:
            logger.warning("%s: %s", spec.name, warning)

        if spec.robot is None:
            raise ValueError(
                f"Scene {spec.name!r} has no robot. A scene without an arm can be rendered and "
                f"validated, but not stepped through the policy interface."
            )
        self.robot = self.build.robot
        self.prefix = self.build.robot_prefix
        self.home_qpos = np.asarray(
            spec.robot.home_qpos if spec.robot.home_qpos is not None else self.robot.home_qpos,
            dtype=np.float64,
        )

        self._resolve_ids()

        self.controller = DifferentialIKController(
            self.model,
            self.data,
            site_id=self.site_id,
            arm_joint_ids=self.arm_joint_ids,
            arm_dof_ids=self.arm_dof_ids,
            home_qpos=self.home_qpos,
            damping=self.config.ik_damping,
            max_joint_step=self.config.ik_max_joint_step,
            nullspace_gain=self.config.nullspace_gain,
            max_total_change=self.config.ik_max_total_change,
        )

        self.drivers = dynamics_module.make_drivers(spec, self.build)

        self._renderers: dict[int, object] = {}
        self.render_observations = True
        self._rng = np.random.default_rng(spec.seed)
        self._joint_command = self.home_qpos.copy()
        self._gripper_ctrl = self.robot.gripper_ctrl_open
        self._step_count = 0
        self._success_streak = 0
        self._failure_modes: list[str] = []
        self.task_state: dict = {}
        self._target_pos = np.zeros(3)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # ---- model introspection ---------------------------------------------
    def _resolve_ids(self) -> None:
        mujoco = self._mujoco

        def need(objtype, name, what):
            index = mujoco.mj_name2id(self.model, objtype, name)
            if index < 0:
                raise ValueError(f"{what} {name!r} not found in the compiled model.")
            return index

        arm_joints = [f"{self.prefix}{n}" for n in self.robot.arm_joints]
        self.arm_joint_ids = np.array(
            [need(mujoco.mjtObj.mjOBJ_JOINT, n, "Joint") for n in arm_joints]
        )
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_dof_ids = self.model.jnt_dofadr[self.arm_joint_ids]
        self.arm_actuator_ids = np.array(
            [
                need(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.prefix}{n}", "Actuator")
                for n in self.robot.arm_actuators
            ]
        )

        if len(self.home_qpos) != len(self.arm_joint_ids):
            raise ValueError(
                f"home_qpos has {len(self.home_qpos)} entries but {self.robot.key} has "
                f"{len(self.arm_joint_ids)} arm joints."
            )

        if not self.robot.gripper_actuator:
            raise ValueError(f"Robot {self.robot.key!r} has no gripper actuator registered.")
        self.gripper_actuator_id = need(
            mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.prefix}{self.robot.gripper_actuator}", "Actuator"
        )
        finger_joints = [f"{self.prefix}{n}" for n in self.robot.finger_joints]
        self.finger_joint_ids = np.array(
            [need(mujoco.mjtObj.mjOBJ_JOINT, n, "Joint") for n in finger_joints]
        )
        self.finger_qpos_adr = self.model.jnt_qposadr[self.finger_joint_ids]
        self._finger_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}{name}")
            for name in ("left_finger", "right_finger")
        }
        self._finger_body_ids.discard(-1)

        self.site_id = need(mujoco.mjtObj.mjOBJ_SITE, f"{self.prefix}{GRIP_SITE}", "Site")

        # Free bodies: remember the dof address so velocities and spawns are
        # addressable by body name rather than by index arithmetic downstream.
        self.free_qpos_adr: dict[str, int] = {}
        self.free_dof_adr: dict[str, int] = {}
        for body_name, joint_name in self.build.free_joints.items():
            joint_id = need(mujoco.mjtObj.mjOBJ_JOINT, joint_name, "Joint")
            self.free_qpos_adr[body_name] = int(self.model.jnt_qposadr[joint_id])
            self.free_dof_adr[body_name] = int(self.model.jnt_dofadr[joint_id])

        self.articulated_qpos_adr: dict[str, int] = {}
        for body_name, joint_name in self.build.articulated_joints.items():
            joint_id = need(mujoco.mjtObj.mjOBJ_JOINT, joint_name, "Joint")
            self.articulated_qpos_adr[body_name] = int(self.model.jnt_qposadr[joint_id])

        scene_cam = self.spec.camera("scene")
        wrist_cam = self.spec.camera("wrist")
        if scene_cam is None or wrist_cam is None:
            raise ValueError(
                f"Scene {self.spec.name!r} needs one camera with role 'scene' and one with role "
                f"'wrist' to satisfy the two-image observation contract."
            )
        self.scene_camera_id = need(
            mujoco.mjtObj.mjOBJ_CAMERA, self.build.cameras[scene_cam.name], "Camera"
        )
        self.wrist_camera_id = need(
            mujoco.mjtObj.mjOBJ_CAMERA, self.build.cameras[wrist_cam.name], "Camera"
        )

    # ---- episode lifecycle -------------------------------------------------
    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._mujoco.mj_resetData(self.model, self.data)

        home = self.home_qpos.copy()
        if self.config.reset_noise > 0:
            home += self._rng.uniform(-self.config.reset_noise, self.config.reset_noise, home.shape)

        self.data.qpos[self.arm_qpos_adr] = home
        if len(self.finger_qpos_adr):
            self.data.qpos[self.finger_qpos_adr] = self.robot.finger_open_qpos

        self._place_free_bodies()
        self._reset_articulated()

        self._joint_command = home.copy()
        self._gripper_ctrl = self.robot.gripper_ctrl_open
        self.data.ctrl[self.arm_actuator_ids] = home
        self.data.ctrl[self.gripper_actuator_id] = self._gripper_ctrl

        for body_name, actuator_name in self.build.actuators.items():
            body = self.spec.body(body_name) if _has_body(self.spec, body_name) else None
            default = body.actuator.default_ctrl if body and body.actuator else 0.0
            actuator_id = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            self.data.ctrl[actuator_id] = default

        self._mujoco.mj_forward(self.model, self.data)

        for driver in self.drivers:
            driver.reset(self.model, self.data)

        for _ in range(self.config.n_substeps * self.config.reset_settle_steps):
            self._physics_step()

        self._step_count = 0
        self._success_streak = 0
        self._failure_modes = []
        # Scratch space for predicates that accumulate progress across an
        # episode (which weld sites have been serviced, and so on).
        self.task_state = {}
        self._target_pos, self._target_quat = self.controller.site_pose()

        return self.get_observation(), {"task": self.spec.task}

    def _place_free_bodies(self) -> None:
        for body in self.spec.free_bodies:
            adr = self.free_qpos_adr[body.name]
            position = np.asarray(body.pos, dtype=float).copy()
            yaw = 0.0

            ranges = body.spawn_range or {}
            for index, axis in enumerate("xyz"):
                if axis in ranges:
                    position[index] = self._rng.uniform(*ranges[axis])
            if "yaw" in ranges:
                yaw = self._rng.uniform(*ranges["yaw"])

            self.data.qpos[adr : adr + 3] = position
            # Compose the spawn yaw ON TOP of the authored orientation rather
            # than replacing it. Overwriting it silently stands up anything
            # authored lying down — a banana modelled as a horizontal capsule
            # spawns upright and buried in the table, which reads as the mesh
            # being wrong rather than the reset being wrong.
            half = yaw / 2.0
            spin = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
            self.data.qpos[adr + 3 : adr + 7] = quat_multiply(
                spin, np.asarray(body.quat, dtype=float)
            )

    def _reset_articulated(self) -> None:
        for body_name, adr in self.articulated_qpos_adr.items():
            if not _has_body(self.spec, body_name):
                continue  # generated by a dynamic macro; leave at its own zero
            body = self.spec.body(body_name)
            self.data.qpos[adr] = body.joint.ref if body.joint else 0.0

    # ---- stepping ----------------------------------------------------------
    def _physics_step(self) -> None:
        for driver in self.drivers:
            driver.apply(self.model, self.data)
        self._mujoco.mj_step(self.model, self.data)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if action.shape[0] != 7:
            raise ValueError(f"Expected a 7-D action, got shape {action.shape}.")

        current_pos, _ = self.controller.site_pose()

        self._target_pos = self._target_pos + action[:3] * self.config.position_delta_scale
        delta_quat = axis_angle_to_quat(action[3:6] * self.config.rotation_delta_scale)
        self._target_quat = quat_multiply(delta_quat, self._target_quat)
        self._target_quat /= np.linalg.norm(self._target_quat)

        self._apply_leash(current_pos)

        open_ctrl = self.robot.gripper_ctrl_open
        closed_ctrl = self.robot.gripper_ctrl_closed
        self._gripper_ctrl = float(open_ctrl + (closed_ctrl - open_ctrl) * (action[6] + 1.0) / 2.0)

        self._joint_command = self.controller.solve(self._target_pos, self._target_quat)
        self.data.ctrl[self.arm_actuator_ids] = self._joint_command
        self.data.ctrl[self.gripper_actuator_id] = self._gripper_ctrl

        for _ in range(self.config.n_substeps):
            self._physics_step()

        self._step_count += 1

        observation = self.get_observation()
        context = success_module.Context(self)
        success = success_module.evaluate(context, self.spec.success)
        self._success_streak = self._success_streak + 1 if success else 0

        triggered = success_module.failures(context, self.spec.success)
        for mode in triggered:
            if mode not in self._failure_modes:
                self._failure_modes.append(mode)

        terminated = self._success_streak >= self.config.success_hold_steps
        failed = bool(triggered) and not terminated
        truncated = self._step_count >= self.config.max_episode_steps or failed
        reward = 1.0 if terminated else 0.0

        info = {
            "task": self.spec.task,
            "is_success": terminated,
            "failure_modes": list(self._failure_modes),
            "eef_position": current_pos.tolist(),
            "step": self._step_count,
            "sim_time": float(self.data.time),
        }
        return observation, reward, terminated, truncated, info

    def _apply_leash(self, measured_pos: np.ndarray) -> None:
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

    # ---- observation --------------------------------------------------------
    def get_state(self) -> np.ndarray:
        position, quat = self.controller.site_pose()
        axis_angle = quat_to_axis_angle(quat)
        fingers = self.data.qpos[self.finger_qpos_adr]
        signs = np.asarray(self.robot.finger_sign[: len(fingers)], dtype=float)
        return np.concatenate([position, axis_angle, fingers * signs]).astype(np.float32)

    def get_observation(self) -> dict[str, np.ndarray]:
        """The policy's view of the world.

        Rendering two cameras per control step dominates the cost of stepping,
        and a privileged-state expert or an interactive viewer needs neither.
        Set `render_observations = False` to skip them; the state vector is
        always present, so success predicates and the controller are unaffected.
        """
        if not self.render_observations:
            return {"observation.state": self.get_state()}
        return {
            self.config.scene_image_key: self.render_camera(self.scene_camera_id, self.config.image_size),
            self.config.wrist_image_key: self.render_camera(self.wrist_camera_id, self.config.image_size),
            "observation.state": self.get_state(),
        }

    def body_position(self, name: str) -> np.ndarray:
        bid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"No body named {name!r}.")
        return self.data.xpos[bid].copy()

    def is_grasped(self, name: str) -> bool:
        """True while either fingertip is in contact with the named body."""
        bid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"No body named {name!r}.")
        target_geoms = set(np.flatnonzero(self.model.geom_bodyid == bid).tolist())
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if not pair & target_geoms:
                continue
            other = pair - target_geoms
            if other and self.model.geom_bodyid[other.pop()] in self._finger_body_ids:
                return True
        return False

    # ---- rendering -----------------------------------------------------------
    def _renderer(self, size: int):
        if size not in self._renderers:
            self._renderers[size] = self._mujoco.Renderer(self.model, height=size, width=size)
        return self._renderers[size]

    def render_camera(self, camera_id: int, size: int) -> np.ndarray:
        renderer = self._renderer(size)
        renderer.update_scene(self.data, camera=camera_id)
        return np.ascontiguousarray(renderer.render().transpose(2, 0, 1))

    def render(self, size: int | None = None, camera: int | str | None = None) -> np.ndarray:
        size = size or self.config.render_size
        renderer = self._renderer(size)
        renderer.update_scene(self.data, camera=self.scene_camera_id if camera is None else camera)
        return renderer.render()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _has_body(spec: SceneSpec, name: str) -> bool:
    return any(body.name == name for body in spec.bodies)


def load_env(package_dir: str | Path) -> SceneEnv:
    """Load a generated environment package written by `build_env.py`."""
    root = Path(package_dir)
    spec = SceneSpec.load(root / "spec.json")
    return SceneEnv(spec, asset_dir=root / "assets")
