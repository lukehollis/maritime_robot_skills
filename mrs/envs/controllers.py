"""End-effector pose control for the Panda.

The policy emits normalized Cartesian deltas, matching the OSC_POSE action
space of the robosuite/LIBERO demonstrations pi0.5 was trained on. Those deltas
are turned into joint targets by damped least-squares differential inverse
kinematics, which the Panda's built-in position actuators then track.

Differential IK is used in place of true operational-space control because it
needs no torque-level gain tuning to stay stable, while presenting the policy
with the identical interface: one unit of action is
`position_delta_scale` metres or `rotation_delta_scale` radians of commanded
end-effector motion.
"""

from __future__ import annotations

import numpy as np


def quat_to_axis_angle(quat_wxyz: np.ndarray) -> np.ndarray:
    """Axis-angle from a `(w, x, y, z)` quaternion, using robosuite's convention.

    The angle is `2 * acos(w)`, so it lies in [0, 2*pi] rather than being
    wrapped to [-pi, pi]. The checkpoint's state statistics were computed this
    way, so reproducing it matters.
    """
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    if quat_wxyz[0] < 0:
        # Not sign-canonicalised: robosuite feeds MuJoCo's quaternion straight
        # through, and this branch only guards against a NaN from acos.
        quat_wxyz = np.clip(quat_wxyz, -1.0, 1.0)

    w = float(np.clip(quat_wxyz[0], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if den < 1e-8:
        return np.zeros(3)
    return quat_wxyz[1:] * (2.0 * np.arccos(w)) / den


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """`(w, x, y, z)` quaternion from a rotation vector."""
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis_angle / angle
    half = angle / 2.0
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two `(w, x, y, z)` quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def orientation_error(target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    """Rotation vector taking `current_quat` to `target_quat`, in world axes."""
    conj = np.array([current_quat[0], -current_quat[1], -current_quat[2], -current_quat[3]])
    delta = quat_multiply(target_quat, conj)
    if delta[0] < 0:  # take the short way round
        delta = -delta
    angle = 2.0 * np.arccos(np.clip(delta[0], -1.0, 1.0))
    sin_half = np.sqrt(max(1.0 - delta[0] * delta[0], 0.0))
    if sin_half < 1e-8:
        return np.zeros(3)
    return delta[1:] / sin_half * angle


class DifferentialIKController:
    """Turns end-effector pose targets into Panda joint-position commands.

    The IK iterates on a scratch `MjData` driven by the *candidate* joint
    configuration rather than on the live simulation state. Solving kinematically
    like this decouples the solver from actuator tracking lag; iterating against
    the measured pose instead lets the error persist across iterations and winds
    the joint command up far past the pose that was actually requested.
    """

    def __init__(
        self,
        model,
        data,
        *,
        site_id: int,
        arm_joint_ids: np.ndarray,
        arm_dof_ids: np.ndarray,
        home_qpos: np.ndarray,
        damping: float = 0.05,
        max_joint_step: float = 0.06,
        nullspace_gain: float = 0.05,
        max_total_change: float = 0.6,
        max_iterations: int = 24,
        tolerance: float = 1e-4,
    ):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.site_id = site_id
        self.arm_joint_ids = np.asarray(arm_joint_ids)
        self.arm_dof_ids = np.asarray(arm_dof_ids)
        self.arm_qpos_adr = model.jnt_qposadr[self.arm_joint_ids]
        self.home_qpos = np.asarray(home_qpos, dtype=np.float64)
        self.damping = damping
        self.max_joint_step = max_joint_step
        self.nullspace_gain = nullspace_gain
        self.max_total_change = max_total_change
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.joint_range = model.jnt_range[self.arm_joint_ids].copy()

        self._scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ---- state readout --------------------------------------------------
    def site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Measured `(position, quaternion_wxyz)` of the grasp site."""
        return self._site_pose_of(self.data)

    def _site_pose_of(self, data) -> tuple[np.ndarray, np.ndarray]:
        position = data.site_xpos[self.site_id].copy()
        quat = np.zeros(4)
        self._mujoco.mju_mat2Quat(quat, data.site_xmat[self.site_id])
        return position, quat

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_adr].copy()

    # ---- control --------------------------------------------------------
    def solve(
        self, target_pos: np.ndarray, target_quat: np.ndarray, q_init: np.ndarray | None = None
    ) -> np.ndarray:
        """Joint configuration whose grasp-site pose matches the target."""
        scratch = self._scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0.0

        q_start = (self.arm_qpos() if q_init is None else np.asarray(q_init, dtype=np.float64)).copy()
        q = q_start.copy()
        identity = np.eye(len(self.arm_dof_ids))

        lower = np.maximum(self.joint_range[:, 0], q_start - self.max_total_change)
        upper = np.minimum(self.joint_range[:, 1], q_start + self.max_total_change)

        for _ in range(self.max_iterations):
            scratch.qpos[self.arm_qpos_adr] = q
            self._mujoco.mj_kinematics(self.model, scratch)
            self._mujoco.mj_comPos(self.model, scratch)

            position, quat = self._site_pose_of(scratch)
            error = np.concatenate(
                [target_pos - position, orientation_error(target_quat, quat)]
            )
            if np.linalg.norm(error) < self.tolerance:
                break

            self._mujoco.mj_jacSite(self.model, scratch, self._jacp, self._jacr, self.site_id)
            jac = np.vstack([self._jacp, self._jacr])[:, self.arm_dof_ids]

            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e. The
            # damping trades tracking accuracy for bounded motion near
            # singularities.
            jjt = jac @ jac.T + (self.damping**2) * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, error)

            # Drive the redundant 7th degree of freedom back toward the home
            # posture, which keeps the elbow clear of the table.
            if self.nullspace_gain > 0.0:
                pinv = jac.T @ np.linalg.solve(jjt, np.eye(6))
                dq += (identity - pinv @ jac) @ (self.nullspace_gain * (self.home_qpos - q))

            norm = np.linalg.norm(dq)
            if norm > self.max_joint_step:
                dq *= self.max_joint_step / norm

            q = np.clip(q + dq, lower, upper)

        return q
