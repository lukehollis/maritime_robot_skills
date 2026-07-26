"""The mail-sorting reference scene, written directly against the spec API.

    python3 .claude/scripts/env/examples/mail_sorting.py

`envs/mail_sorting/spec.json` was produced through the full pipeline — authored
in Blender, exported, migrated by `build_env.py`. This file is the same scene
expressed as Python, and exists for two reasons the migrated spec cannot serve:

* it is readable, so it works as a template for a structurally similar cell;
* it carries a filled-in `SuccessSpec`, which a stage-0 migration deliberately
  leaves empty (success predicates belong to `robo-task-define`).

Run it directly to build the scene, step it, and print the physical checks.
Copy its numbers only when the task is genuinely similar; copy its *structure*
freely.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrs.envs.scenegen import (  # noqa: E402
    ActuatorSpec, BodySpec, CameraSpec, ControlSpec, DynamicSpec, JointSpec,
    MaterialSpec, RobotSpec, SceneSpec, SuccessSpec, WorldSpec, SceneEnv,
)

TABLE_TOP = 0.63
BELT_X = -0.10
BELT_Z = 0.66
ROLLER_R = 0.025
BELT_SURFACE = BELT_Z + ROLLER_R
BIN_X = 0.12
BIN_YS = {"small": -0.22, "medium": 0.0, "large": 0.22}
LANES = {"small": -0.24, "medium": -0.14, "large": -0.02}
ENVELOPES = {           # tag       half-extents (x, y, z)      mass
    "small":  ((0.045, 0.032, 0.003), 0.010),
    "medium": ((0.062, 0.044, 0.004), 0.018),
    "large":  ((0.085, 0.058, 0.005), 0.030),
}


def build_spec() -> SceneSpec:
    materials = [
        MaterialSpec("table_mat", rgba=(0.62, 0.50, 0.36, 1.0), texture="checker",
                     texrepeat=(6.0, 8.0), checker_rgb2=(0.57, 0.45, 0.32),
                     specular=0.05, shininess=0.05),
        MaterialSpec("bin_mat", rgba=(0.30, 0.38, 0.52, 1.0)),
        MaterialSpec("steel_mat", rgba=(0.45, 0.47, 0.52, 1.0), specular=0.5, shininess=0.5),
        MaterialSpec("paper_small", rgba=(0.92, 0.88, 0.72, 1.0)),
        MaterialSpec("paper_medium", rgba=(0.88, 0.82, 0.90, 1.0)),
        MaterialSpec("paper_large", rgba=(0.78, 0.86, 0.92, 1.0)),
    ]

    bodies: list[BodySpec] = [
        BodySpec(name="table", kind="static", shape="box", size=(0.45, 0.60, 0.025),
                 pos=(0.05, 0.0, TABLE_TOP - 0.025), material="table_mat",
                 friction=(1.0, 0.005, 0.0001), tags=["surface"]),
    ]

    # Three open-topped bins: floor plus four walls, so an envelope can land in
    # one and stay there.
    for label, y in BIN_YS.items():
        half_x, half_y, wall_h, t = 0.080, 0.080, 0.030, 0.005
        base = f"bin_{label}"
        bodies.append(BodySpec(name=base, kind="static", shape="box",
                               size=(half_x, half_y, t), pos=(BIN_X, y, TABLE_TOP + t),
                               material="bin_mat", tags=["bin", f"bin_{label}"]))
        for sx, sy, sz_name in ((1, 0, "px"), (-1, 0, "nx"), (0, 1, "py"), (0, -1, "ny")):
            bodies.append(BodySpec(
                name=f"{base}_wall_{sz_name}", kind="static", shape="box",
                size=(t if sx else half_x, half_y if sx else t, wall_h),
                pos=(BIN_X + sx * half_x, y + sy * half_y, TABLE_TOP + t + wall_h),
                material="bin_mat", tags=["bin_wall"]))

    # The parts. They spawn at the upstream end of the belt and ride it in.
    for label, (size, mass) in ENVELOPES.items():
        bodies.append(BodySpec(
            name=f"envelope_{label}", kind="free", shape="box", size=size,
            pos=(BELT_X, LANES[label], BELT_SURFACE + size[2] + 0.002),
            material=f"paper_{label}", mass=mass,
            friction=(1.1, 0.02, 0.001), condim=4, solref=(0.01, 1.0),
            # One lane per size. Lane centres are spaced by more than the sum of
            # adjacent half-widths and the jitter stays inside the clearance, so
            # no sampled spawn can put two envelopes inside each other.
            spawn_range={"x": (BELT_X - 0.025, BELT_X + 0.025),
                         "y": (LANES[label] - 0.008, LANES[label] + 0.008),
                         "yaw": (-0.25, 0.25)},
            tags=["envelope", f"size_{label}"]))

    # A moving scene: the belt runs, stops, and runs again.
    dynamics = [
        DynamicSpec(name="infeed", kind="roller_conveyor", params={
            "origin": (BELT_X, -0.06, BELT_Z), "direction": "+y",
            "length": 0.42, "width": 0.20, "roller_radius": ROLLER_R,
            "spacing": 0.07, "roller_mass": 0.15, "kv": 4.0,
            "material": "steel_mat", "speed": 0.05, "end_stop": True,
            "duty": {"period": 9.0, "on_fraction": 0.55},
            "rail_height": 0.035,
        }),
    ]

    cameras = [
        CameraSpec(name="agentview", pos=(0.86, 0.0, 1.05), target=(-0.06, 0.0, 0.70),
                   fovy=45.0, role="scene"),
        CameraSpec(name="wrist", pos=(0.06, 0.0, -0.055), target=(0.30, 0.0, 1.0),
                   fovy=72.0, mount="hand", up=(0.0, -1.0, 0.0), role="wrist"),
        CameraSpec(name="topdown", pos=(0.0, 0.0, 1.45), target=(0.0, 0.0, 0.66),
                   fovy=55.0, role="inspection"),
        CameraSpec(name="side", pos=(0.05, -1.05, 0.95), target=(0.0, 0.0, 0.70),
                   fovy=45.0, role="inspection"),
    ]

    success = SuccessSpec(
        mode="all",
        terms=[{"predicate": "inside", "body": f"envelope_{label}",
                "container": f"bin_{label}", "pad": 0.01}
               for label in ENVELOPES]
        + [{"predicate": "each_tagged", "tag": "envelope",
            "term": {"predicate": "at_rest", "speed": 0.05}}],
        failure_terms=[
            {"name": f"dropped_{label}", "predicate": "below_height",
             "body": f"envelope_{label}", "height": TABLE_TOP - 0.10}
            for label in ENVELOPES
        ],
    )

    return SceneSpec(
        name="mail_sorting",
        task="sort the envelopes by size into the matching bins",
        world=WorldSpec(),
        materials=materials,
        bodies=bodies,
        dynamics=dynamics,
        robot=RobotSpec(key="panda", mount_pos=(-0.56, 0.0, TABLE_TOP)),
        cameras=cameras,
        control=ControlSpec(max_episode_steps=600),
        success=success,
        seed=0,
        provenance={"source": "reference example", "author": "robo-env-create"},
    )


if __name__ == "__main__":
    import mujoco

    spec = build_spec()
    env = SceneEnv(spec)
    print(f"compiled  nq={env.model.nq} nv={env.model.nv} nu={env.model.nu} "
          f"nbody={env.model.nbody} ngeom={env.model.ngeom}")
    print(f"drivers   {[d.name for d in env.drivers]}")
    print(f"conveyor rollers: {len(env.build.expanded_bodies['infeed'])}")

    obs, info = env.reset(seed=0)
    print(f"obs keys  {sorted(obs)}")
    print(f"state     {np.round(obs['observation.state'], 3)}")
    print(f"images    {obs['observation.images.image'].shape} {obs['observation.images.image'].dtype}")

    # t=0 interpenetration audit
    deep = []
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        if c.dist < -0.002:
            n = lambda g: mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            deep.append((n(c.geom1), n(c.geom2), round(float(c.dist), 4)))
    print(f"penetrations at reset: {deep if deep else 'none'}")

    start = {k: env.body_position(f"envelope_{k}").copy() for k in ENVELOPES}
    warn0 = env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number

    for _ in range(240):                     # 12 s of belt motion, arm holding still
        env.step(np.zeros(7))

    warn = env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number - warn0
    print(f"\nbadqacc during 240 steps: {warn}")
    for k in ENVELOPES:
        end = env.body_position(f"envelope_{k}")
        print(f"  envelope_{k:<7} dy={end[1]-start[k][1]:+.3f}  z={end[2]:.3f}")

    eef = env.controller.site_pose()[0]
    print(f"\neef at home: {np.round(eef, 3)}")
    lo = np.asarray(spec.control.workspace_min); hi = np.asarray(spec.control.workspace_max)
    print(f"inside workspace box: {bool(np.all(eef >= lo) and np.all(eef <= hi))}")
    env.close()
