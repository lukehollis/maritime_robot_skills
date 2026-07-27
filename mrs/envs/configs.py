"""Environment configuration.

The default geometry is chosen so the reachable workspace overlaps the
end-effector pose distribution recorded in the LIBERO-finetuned pi0.5
checkpoint's normalization statistics, which are (1st..99th percentile):

    x  [-0.19, +0.05]    y  [-0.14, +0.22]    z  [+0.64, +0.88]

Those statistics also fix the observation contract: an 8-D state of
`[eef_xyz, eef_axis_angle, finger_qpos, -finger_qpos]` and a 7-D action of
`[dx, dy, dz, drx, dry, drz, gripper]` in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PandaPickPlaceConfig:
    """Everything that defines the pick-and-place task and its interface."""

    # ---- task ----------------------------------------------------------
    task: str = "pick up the red block and place it on the white plate"
    max_episode_steps: int = 400
    success_hold_steps: int = 5
    """Consecutive control steps the success predicate must hold before the
    episode terminates, so a cube merely bouncing through the target does not
    count."""

    # ---- workcell geometry (metres) ------------------------------------
    table_height: float = 0.63
    table_center: tuple[float, float] = (0.05, 0.0)
    table_size: tuple[float, float] = (0.90, 1.20)

    robot_base_pos: tuple[float, float] = (-0.56, 0.0)
    robot_base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    grip_site_offset: float = 0.1034
    """Distance from the Panda hand frame to the point between the fingertips."""

    # ---- objects --------------------------------------------------------
    cube_size: float = 0.04
    cube_mass: float = 0.05
    cube_spawn_x: tuple[float, float] = (-0.12, 0.04)
    cube_spawn_y: tuple[float, float] = (-0.18, 0.06)

    target_pos: tuple[float, float] = (-0.02, 0.24)
    target_radius: float = 0.075
    success_radius: float = 0.06
    """Planar distance from the plate centre within which the cube counts as placed."""

    # ---- control --------------------------------------------------------
    sim_timestep: float = 0.002
    control_freq: float = 20.0
    """Matches the 20 Hz LIBERO demonstrations the checkpoint was trained on."""

    position_delta_scale: float = 0.05
    """Metres of commanded end-effector translation per unit action."""
    rotation_delta_scale: float = 0.5
    """Radians of commanded end-effector rotation per unit action."""

    ik_damping: float = 0.05
    ik_max_joint_step: float = 0.05
    ik_max_total_change: float = 0.6
    """Cap on how far one IK solve may move any joint from its starting value.
    Keeps the solver on the local branch instead of folding the arm into a
    distant configuration with the same end-effector pose."""
    nullspace_gain: float = 0.05

    position_leash: float = 0.07
    """How far the commanded end-effector position may lead the measured one.
    The command is integrated rather than re-derived from the measurement each
    step, so that deltas mean exactly what they say and tracking error cannot
    random-walk the pose; the leash stops the command running away when the arm
    is blocked."""
    rotation_leash: float = 0.5

    workspace_min: tuple[float, float, float] = (-0.32, -0.36, 0.645)
    workspace_max: tuple[float, float, float] = (0.20, 0.36, 1.00)
    """Commanded end-effector positions are clamped to this box, which keeps the
    arm above the table and inside the region the cameras observe."""

    home_qpos: tuple[float, ...] = (0.0, 0.10, 0.0, -2.45, 0.0, 2.55, 0.785)
    reset_noise: float = 0.0
    """Uniform noise (radians) added to the home joint configuration on reset."""

    # ---- cameras ---------------------------------------------------------
    image_size: int = 256

    scene_camera: str = "agentview"
    scene_camera_pos: tuple[float, float, float] = (0.88, 0.0, 1.02)
    scene_camera_target: tuple[float, float, float] = (-0.14, 0.0, 0.66)
    scene_camera_fovy: float = 45.0

    wrist_camera: str = "wrist"
    wrist_camera_pos: tuple[float, float, float] = (0.06, 0.0, -0.055)
    wrist_camera_forward: tuple[float, float, float] = (0.30, 0.0, 1.0)
    wrist_camera_fovy: float = 72.0

    # ---- observation naming ----------------------------------------------
    scene_image_key: str = "observation.images.image"
    wrist_image_key: str = "observation.images.image2"
    """Defaults match the LIBERO-finetuned pi0.5 checkpoint's feature names."""

    seed: int | None = None

    render_size: int = 512
    """Resolution of the human-facing rollout video, independent of policy input."""

    metadata: dict = field(default_factory=dict)

    @property
    def n_substeps(self) -> int:
        """Physics steps per control step."""
        steps = round(1.0 / (self.control_freq * self.sim_timestep))
        if steps < 1:
            raise ValueError("control_freq is too high for the configured sim_timestep.")
        return steps
