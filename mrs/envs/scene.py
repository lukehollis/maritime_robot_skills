"""Builds the pick-and-place workcell as a MuJoCo model.

The scene is assembled with `MjSpec` rather than a static XML so the Menagerie
Panda can be used unmodified: the grasp site and the wrist camera are added
programmatically to its `hand` body before attachment.

Layout is chosen so the end-effector workspace overlaps the pose distribution
the LIBERO-finetuned pi0.5 checkpoint was trained on (see
`mrs/envs/configs.py` for the numbers): the robot is mounted at table height
facing +x, with the manipulation area centred just in front of it.
"""

from __future__ import annotations

import numpy as np

from mrs.envs.assets import panda_model_path
from mrs.envs.configs import PandaPickPlaceConfig

GRIP_SITE = "grip_site"
ROBOT_PREFIX = "robot_"


def look_at(position, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """Camera orientation quaternion `(w, x, y, z)` for a camera aimed at `target`.

    MuJoCo cameras look down their local -z, so the local z axis points from
    the target back toward the camera.
    """
    import mujoco

    z_axis = np.asarray(position, float) - np.asarray(target, float)
    z_axis /= np.linalg.norm(z_axis)

    x_axis = np.cross(np.asarray(up, float), z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    # MuJoCo stores rotations column-major-by-axis: columns are the local axes.
    rotation = np.column_stack([x_axis, y_axis, z_axis]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation)
    return quat.tolist()


def build_scene(config: PandaPickPlaceConfig):
    """Return a compiled `mujoco.MjModel` for the workcell."""
    import mujoco

    table_top = config.table_height
    spec = mujoco.MjSpec()
    spec.modelname = "panda_pick_place"
    spec.option.timestep = config.sim_timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    _add_visual_defaults(spec, mujoco)
    _add_assets(spec, mujoco)
    _add_static_geometry(spec, mujoco, config, table_top)
    _add_objects(spec, mujoco, config, table_top)
    _add_scene_cameras(spec, config)
    _attach_robot(spec, mujoco, config, table_top)

    return spec.compile()


# ---------------------------------------------------------------------------
# Scene pieces
# ---------------------------------------------------------------------------


def _add_visual_defaults(spec, mujoco) -> None:
    # Exposure is deliberately conservative: the wrist camera ends up very close
    # to the table, and a brighter rig blows that view out to flat white.
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
    spec.visual.headlight.ambient = [0.2, 0.2, 0.2]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 4096
    # The offscreen framebuffer defaults to 640x480, which is smaller than the
    # video render size; MuJoCo refuses to render above it.
    spec.visual.global_.offwidth = 1024
    spec.visual.global_.offheight = 1024

    # Two directional key lights from opposite sides, so the gripper does not
    # cast the object it is reaching for into its own shadow.
    for dir_ in ((-0.3, -0.4, -1.0), (-0.3, 0.4, -1.0)):
        light = spec.worldbody.add_light()
        light.pos = [0.4, 0.0, 2.2]
        light.dir = list(dir_)
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.diffuse = [0.32, 0.32, 0.32]
        light.specular = [0.05, 0.05, 0.05]
        light.castshadow = True


def _add_assets(spec, mujoco) -> None:
    sky = spec.add_texture()
    sky.name = "skybox"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.55, 0.6, 0.68]
    sky.rgb2 = [0.2, 0.24, 0.3]
    sky.width, sky.height = 512, 3072

    floor_tex = spec.add_texture()
    floor_tex.name = "floor_tex"
    floor_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_tex.rgb1 = [0.24, 0.26, 0.29]
    floor_tex.rgb2 = [0.19, 0.21, 0.24]
    floor_tex.width, floor_tex.height = 512, 512

    floor_mat = spec.add_material()
    floor_mat.name = "floor_mat"
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    floor_mat.texrepeat = [4, 4]
    floor_mat.texuniform = True
    floor_mat.reflectance = 0.05

    wood_tex = spec.add_texture()
    wood_tex.name = "wood_tex"
    wood_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    wood_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    wood_tex.rgb1 = [0.62, 0.50, 0.36]
    wood_tex.rgb2 = [0.57, 0.45, 0.32]
    wood_tex.width, wood_tex.height = 512, 512

    table_mat = spec.add_material()
    table_mat.name = "table_mat"
    table_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "wood_tex"
    table_mat.texrepeat = [6, 8]
    table_mat.texuniform = True
    table_mat.specular = 0.05
    table_mat.shininess = 0.05

    for name, rgba in (
        ("cube_mat", [0.78, 0.11, 0.10, 1.0]),
        ("plate_mat", [0.88, 0.89, 0.92, 1.0]),
        ("pedestal_mat", [0.26, 0.27, 0.30, 1.0]),
    ):
        mat = spec.add_material()
        mat.name = name
        mat.rgba = rgba
        mat.specular = 0.2
        mat.shininess = 0.2


def _add_static_geometry(spec, mujoco, config: PandaPickPlaceConfig, table_top: float) -> None:
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.material = "floor_mat"

    half_thickness = 0.025
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = [config.table_center[0], config.table_center[1], table_top - half_thickness]

    top = table.add_geom()
    top.name = "table_top"
    top.type = mujoco.mjtGeom.mjGEOM_BOX
    top.size = [config.table_size[0] / 2, config.table_size[1] / 2, half_thickness]
    top.material = "table_mat"
    top.friction = [1.0, 0.005, 0.0001]

    # Four legs, purely visual context for the camera.
    for sx in (-1, 1):
        for sy in (-1, 1):
            leg = table.add_geom()
            leg.name = f"table_leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}"
            leg.type = mujoco.mjtGeom.mjGEOM_BOX
            leg.size = [0.03, 0.03, (table_top - half_thickness) / 2]
            leg.pos = [
                sx * (config.table_size[0] / 2 - 0.06),
                sy * (config.table_size[1] / 2 - 0.06),
                -(table_top + half_thickness) / 2,
            ]
            leg.material = "pedestal_mat"
            leg.contype, leg.conaffinity = 0, 0

    pedestal = spec.worldbody.add_body()
    pedestal.name = "pedestal"
    pedestal.pos = [config.robot_base_pos[0], config.robot_base_pos[1], table_top / 2]
    column = pedestal.add_geom()
    column.name = "pedestal_column"
    column.type = mujoco.mjtGeom.mjGEOM_BOX
    column.size = [0.09, 0.09, table_top / 2]
    column.material = "pedestal_mat"


def _add_objects(spec, mujoco, config: PandaPickPlaceConfig, table_top: float) -> None:
    """The manipulated cube (free body) and the fixed target plate."""
    half = config.cube_size / 2

    cube = spec.worldbody.add_body()
    cube.name = "cube"
    cube.pos = [0.0, 0.0, table_top + half]
    cube.add_freejoint(name="cube_joint")

    cube_geom = cube.add_geom()
    cube_geom.name = "cube_geom"
    cube_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    cube_geom.size = [half, half, half]
    cube_geom.material = "cube_mat"
    cube_geom.mass = config.cube_mass
    # High tangential friction and a slightly soft contact keep a pinch grasp
    # from squirting the cube out from between the fingertips.
    cube_geom.friction = [1.4, 0.01, 0.0005]
    cube_geom.solref = [0.008, 1.0]
    cube_geom.condim = 4

    plate = spec.worldbody.add_body()
    plate.name = "plate"
    plate.pos = [config.target_pos[0], config.target_pos[1], table_top + 0.005]
    plate_geom = plate.add_geom()
    plate_geom.name = "plate_geom"
    plate_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    plate_geom.size = [config.target_radius, 0.005, 0.0]
    plate_geom.material = "plate_mat"
    plate_geom.friction = [1.0, 0.005, 0.0001]


def _add_scene_cameras(spec, config: PandaPickPlaceConfig) -> None:
    """The fixed third-person camera, aimed like LIBERO's `agentview`."""
    cam = spec.worldbody.add_camera()
    cam.name = config.scene_camera
    cam.pos = list(config.scene_camera_pos)
    cam.fovy = config.scene_camera_fovy
    cam.quat = look_at(config.scene_camera_pos, config.scene_camera_target)


def _attach_robot(spec, mujoco, config: PandaPickPlaceConfig, table_top: float) -> None:
    """Load the Panda, give it a grasp site and wrist camera, then attach it."""
    robot = mujoco.MjSpec.from_file(str(panda_model_path()))

    hand = robot.body("hand")

    site = hand.add_site()
    site.name = GRIP_SITE
    # Midway between the fingertips: the point robosuite reports as the
    # end-effector, and therefore what the checkpoint's state statistics describe.
    site.pos = [0.0, 0.0, config.grip_site_offset]
    site.size = [0.005, 0.005, 0.005]
    site.group = 4

    wrist = hand.add_camera()
    wrist.name = config.wrist_camera
    wrist.pos = list(config.wrist_camera_pos)
    wrist.fovy = config.wrist_camera_fovy
    # Looks forward along the gripper's approach axis (+z of the hand frame),
    # tilted to keep the fingertips in frame.
    wrist.quat = look_at(
        config.wrist_camera_pos,
        np.asarray(config.wrist_camera_pos) + np.asarray(config.wrist_camera_forward),
        up=(0.0, -1.0, 0.0),
    )

    frame = spec.worldbody.add_frame()
    frame.pos = [config.robot_base_pos[0], config.robot_base_pos[1], table_top]
    frame.quat = list(config.robot_base_quat)
    spec.attach(robot, prefix=ROBOT_PREFIX, frame=frame)
