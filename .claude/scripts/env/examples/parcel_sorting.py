"""Parcel sorting — the graspable sibling of `mail_sorting`.

    python3 .claude/scripts/env/examples/parcel_sorting.py --write envs/parcel_sorting

Same cell as `mail_sorting`: Panda, powered roller conveyor, three destination
bins. The one difference is the parts.

A parallel-jaw gripper cannot pick a flat envelope off a surface. The Panda's
fingertips sit about 8 mm below the grip site, and a 6 mm envelope lying on a
roller bed leaves nowhere for them to go but into the rollers. Real mail
sorters use suction for flats for exactly this reason. So this variant carries
padded mailers and small parcels — 30 to 50 mm thick, still in three sizes —
which a pinch grasp can actually lift.

Grasp widths are kept under 70 mm because the Panda's gripper opens to 80 mm
and needs margin, and yaw jitter is small because the scripted expert grasps
without reorienting the hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrs.envs.scenegen import (  # noqa: E402
    BodySpec, CameraSpec, ControlSpec, DynamicSpec, MaterialSpec,
    RobotSpec, SceneSpec, SuccessSpec, WorldSpec,
)

TABLE_TOP = 0.63
BELT_X, BELT_Z, ROLLER_R = -0.10, 0.66, 0.025
BELT_SURFACE = BELT_Z + ROLLER_R
BIN_X = 0.12
BIN_YS = {"small": -0.22, "medium": 0.0, "large": 0.22}
LANES = {"small": -0.23, "medium": -0.10, "large": 0.05}

# half-extents (x, y, z), mass, colour. The y half-extent is the grasp width.
PARCELS = {
    "small":  ((0.037, 0.025, 0.015), 0.020, (0.86, 0.72, 0.52, 1.0)),
    "medium": ((0.047, 0.030, 0.020), 0.035, (0.72, 0.60, 0.44, 1.0)),
    "large":  ((0.057, 0.035, 0.025), 0.055, (0.60, 0.50, 0.38, 1.0)),
}


def build_spec() -> SceneSpec:
    materials = [
        MaterialSpec("table_mat", rgba=(0.62, 0.50, 0.36, 1.0), texture="checker",
                     texrepeat=(6.0, 8.0), checker_rgb2=(0.57, 0.45, 0.32),
                     specular=0.05, shininess=0.05),
        MaterialSpec("bin_mat", rgba=(0.28, 0.36, 0.50, 1.0)),
        MaterialSpec("steel_mat", rgba=(0.45, 0.47, 0.52, 1.0), specular=0.5, shininess=0.5),
    ]
    for label, (_, _, colour) in PARCELS.items():
        materials.append(MaterialSpec(f"parcel_{label}", rgba=colour, specular=0.1))

    bodies = [
        BodySpec(name="table", kind="static", shape="box", size=(0.45, 0.60, 0.025),
                 pos=(0.05, 0.0, TABLE_TOP - 0.025), material="table_mat",
                 friction=(1.0, 0.005, 0.0001), tags=["surface"]),
    ]

    # Shallow bins: deep enough to hold a parcel, shallow enough that the
    # gripper can release above the walls without a long descent.
    for label, y in BIN_YS.items():
        half, wall_h, t = 0.085, 0.028, 0.005
        base = f"bin_{label}"
        bodies.append(BodySpec(name=base, kind="static", shape="box",
                               size=(half, half, t), pos=(BIN_X, y, TABLE_TOP + t),
                               material="bin_mat", tags=["bin", f"size_{label}"]))
        for sx, sy, tag in ((1, 0, "px"), (-1, 0, "nx"), (0, 1, "py"), (0, -1, "ny")):
            bodies.append(BodySpec(
                name=f"{base}_wall_{tag}", kind="static", shape="box",
                size=(t if sx else half, half if sx else t, wall_h),
                pos=(BIN_X + sx * half, y + sy * half, TABLE_TOP + t + wall_h),
                material="bin_mat", tags=["bin_wall"]))

    for label, (size, mass, _) in PARCELS.items():
        bodies.append(BodySpec(
            name=f"parcel_{label}", kind="free", shape="box", size=size,
            pos=(BELT_X, LANES[label], BELT_SURFACE + size[2] + 0.002),
            material=f"parcel_{label}", mass=mass,
            friction=(1.7, 0.02, 0.001), condim=4, solref=(0.008, 1.0),
            spawn_range={"x": (BELT_X - 0.02, BELT_X + 0.02),
                         "y": (LANES[label] - 0.01, LANES[label] + 0.01),
                         # Small: the expert grasps without reorienting the hand.
                         "yaw": (-0.06, 0.06)},
            tags=["parcel", f"size_{label}"]))

    dynamics = [
        DynamicSpec(name="infeed", kind="roller_conveyor", params={
            "origin": (BELT_X, -0.06, BELT_Z), "direction": "+y",
            "length": 0.44, "width": 0.20, "roller_radius": ROLLER_R,
            "spacing": 0.065, "roller_mass": 0.15, "kv": 4.0,
            "material": "steel_mat", "speed": 0.035, "end_stop": True,
            "duty": {"period": 10.0, "on_fraction": 0.45},
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
    ]

    success = SuccessSpec(
        mode="all",
        terms=[{"predicate": "inside", "body": f"parcel_{label}",
                "container": f"bin_{label}", "pad": 0.02} for label in PARCELS]
             + [{"predicate": "each_tagged", "tag": "parcel",
                 "term": {"predicate": "at_rest", "speed": 0.05}}],
        failure_terms=[{"name": f"dropped_{label}", "predicate": "below_height",
                        "body": f"parcel_{label}", "height": TABLE_TOP - 0.10}
                       for label in PARCELS],
    )

    return SceneSpec(
        name="parcel_sorting",
        task="sort the parcels by size into the matching bins",
        world=WorldSpec(), materials=materials, bodies=bodies, dynamics=dynamics,
        robot=RobotSpec(key="panda", mount_pos=(-0.56, 0.0, TABLE_TOP)),
        cameras=cameras, control=ControlSpec(max_episode_steps=1200),
        success=success, seed=0,
        provenance={"source": "parcel_sorting example",
                    "note": "graspable variant of mail_sorting"},
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", type=Path, default=None, help="write spec.json here")
    parser.add_argument("--episodes", type=int, default=1, help="scripted-expert trial episodes")
    args = parser.parse_args(argv)

    spec = build_spec()
    if args.write:
        args.write.mkdir(parents=True, exist_ok=True)
        spec.save(args.write / "spec.json")
        print(f"wrote {args.write / 'spec.json'}")

    from mrs.envs.scenegen import SceneEnv
    from mrs.envs.scenegen.scripted import ScriptedSorter, assignments_by_tag

    env = SceneEnv(spec)
    pairs = assignments_by_tag(spec, object_tag="parcel", destination_prefix="bin_")
    print(f"assignments: {pairs}")

    expert = ScriptedSorter(env, pairs)
    successes = 0
    for episode in range(args.episodes):
        env.reset(seed=episode)
        expert.reset()
        info = {}
        for _ in range(spec.control.max_episode_steps):
            _, _, terminated, truncated, info = env.step(expert.act())
            if terminated or truncated:
                break
        placed = sum(
            1 for body, dest in pairs
            if abs(env.body_position(body)[0] - env.body_position(dest)[0]) < 0.10
            and abs(env.body_position(body)[1] - env.body_position(dest)[1]) < 0.10
        )
        successes += bool(info.get("is_success"))
        print(f"episode {episode}: success={info.get('is_success')} "
              f"placed={placed}/{len(pairs)} steps={info.get('step')} "
              f"failures={info.get('failure_modes')}")

    print(f"\nexpert success: {successes}/{args.episodes}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
