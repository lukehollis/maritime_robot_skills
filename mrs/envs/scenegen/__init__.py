"""Spec-driven scene generation: Blender authoring -> MuJoCo environment.

The hand-written `mrs.envs.panda_pick_place` is the reference this package
generalises. A `SceneSpec` describing one cube and one plate compiles to an
environment with the identical observation and action contract, so generated
scenes and the reference scene are interchangeable from a policy's point of
view.

    from mrs.envs.scenegen import SceneSpec, SceneEnv, load_env

    env = load_env("envs/mail_sorting")
    observation, info = env.reset(seed=0)
"""

from mrs.envs.scenegen.builder import BuildInfo, build_model
from mrs.envs.scenegen.env import SceneEnv, load_env
from mrs.envs.scenegen.robots import ROBOTS, RobotModel, get_robot, register
from mrs.envs.scenegen.spec import (
    ActuatorSpec,
    BodySpec,
    CameraSpec,
    ControlSpec,
    DynamicSpec,
    EqualitySpec,
    JointSpec,
    MaterialSpec,
    RobotSpec,
    SceneSpec,
    SuccessSpec,
    WorldSpec,
)

__all__ = [
    "ActuatorSpec",
    "BodySpec",
    "BuildInfo",
    "CameraSpec",
    "ControlSpec",
    "DynamicSpec",
    "EqualitySpec",
    "JointSpec",
    "MaterialSpec",
    "ROBOTS",
    "RobotModel",
    "RobotSpec",
    "SceneEnv",
    "SceneSpec",
    "SuccessSpec",
    "WorldSpec",
    "build_model",
    "get_robot",
    "load_env",
    "register",
]
