"""Menagerie robot descriptions.

Every arm in MuJoCo Menagerie names its joints, actuators and gripper
differently, and each has its own gripper control convention. Confining that
knowledge to one table keeps the scene builder, the controller and the
environment robot-agnostic.

Only entries that have been checked against the actual MJCF are registered.
Adding one is deliberately a small, explicit act — see `ROBOT_RECIPE` at the
bottom of this file — because a wrong actuator name fails at model-compile
time with a confusing message, and a wrong gripper convention fails silently
by grasping when it should release.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mrs.envs.assets import menagerie_path


@dataclass(frozen=True)
class RobotModel:
    """Everything robot-specific the rest of the package needs."""

    key: str
    menagerie_dir: str
    model_file: str

    arm_joints: tuple[str, ...]
    arm_actuators: tuple[str, ...]
    home_qpos: tuple[float, ...]

    hand_body: str
    """Body the grasp site and wrist camera are attached to."""
    grip_site_offset: float
    """Distance along the hand's +z from its frame to the point between the
    fingertips — the point robosuite reports as the end-effector."""

    finger_joints: tuple[str, ...] = ()
    gripper_actuator: str | None = None
    gripper_ctrl_open: float = 0.0
    gripper_ctrl_closed: float = 0.0
    finger_open_qpos: float = 0.0
    finger_sign: tuple[float, ...] = (1.0, -1.0)
    """Sign applied to each finger joint when assembling the state vector.
    robosuite reports the Panda's two finger joints with opposite signs and the
    pi0.5 normalization statistics assume it."""

    wrist_camera_pos: tuple[float, float, float] = (0.06, 0.0, -0.055)
    wrist_camera_forward: tuple[float, float, float] = (0.30, 0.0, 1.0)
    wrist_camera_fovy: float = 72.0
    wrist_camera_up: tuple[float, float, float] = (0.0, -1.0, 0.0)

    mount_clearance: float = 0.09
    """Half-width of the pedestal column drawn under the base."""
    notes: str = ""

    def model_path(self, **kwargs):
        return menagerie_path(**kwargs) / self.menagerie_dir / self.model_file

    def prefixed(self, prefix: str, names) -> list[str]:
        return [f"{prefix}{name}" for name in names]


PANDA = RobotModel(
    key="panda",
    menagerie_dir="franka_emika_panda",
    model_file="panda.xml",
    arm_joints=tuple(f"joint{i}" for i in range(1, 8)),
    arm_actuators=tuple(f"actuator{i}" for i in range(1, 8)),
    # Not the Menagerie `home` keyframe. This configuration puts the gripper
    # inside the end-effector pose band recorded in the LIBERO-finetuned pi0.5
    # normalization statistics, and matches mrs.envs.configs.
    home_qpos=(0.0, 0.10, 0.0, -2.45, 0.0, 2.55, 0.785),
    hand_body="hand",
    grip_site_offset=0.1034,
    finger_joints=("finger_joint1", "finger_joint2"),
    gripper_actuator="actuator8",
    # hand.xml remaps the split tendon's (0, 0.04) range onto (0, 255),
    # with 255 fully open.
    gripper_ctrl_open=255.0,
    gripper_ctrl_closed=0.0,
    finger_open_qpos=0.04,
    notes="Verified against menagerie franka_emika_panda/panda.xml.",
)


ROBOTS: dict[str, RobotModel] = {
    "panda": PANDA,
}


UNSUPPORTED = {
    "fr3": (
        "menagerie's franka_fr3/fr3.xml ships the arm only — no hand body and no "
        "gripper actuator — so it cannot run a manipulation task unmodified. "
        "Attach a gripper MJCF first, then register the combined model."
    ),
}


def get_robot(key: str) -> RobotModel:
    if key in ROBOTS:
        return ROBOTS[key]
    if key in UNSUPPORTED:
        raise ValueError(f"Robot {key!r} is not usable as shipped: {UNSUPPORTED[key]}")
    raise ValueError(
        f"Unknown robot {key!r}. Registered: {sorted(ROBOTS)}. "
        f"To add one, follow ROBOT_RECIPE in mrs/envs/scenegen/robots.py."
    )


def register(model: RobotModel) -> RobotModel:
    """Add a robot at runtime. Generated env packages may call this."""
    ROBOTS[model.key] = model
    return model


ROBOT_RECIPE = """\
Adding a Menagerie arm to ROBOTS
================================

1. Find the model:      ls .cache/mujoco_menagerie/<dir>/
2. Read its names:      python -c "
     import mujoco; s = mujoco.MjSpec.from_file('<path>.xml')
     print([j.name for j in s.joints]); print([a.name for a in s.actuators])
     print([b.name for b in s.bodies])"
3. Confirm it HAS a gripper. Several Menagerie arms are shipped bare.
4. Determine the gripper convention by driving the actuator to each end of its
   ctrlrange and reading the finger joint qpos. Do not assume +1 closes.
5. Measure grip_site_offset: the distance along the hand body's +z from its
   frame origin to the midpoint between the fingertips, with the gripper open.
6. Pick home_qpos so the end-effector sits inside the workspace box you intend
   to use, then assert it in a test the way tests/test_env.py does.
"""
