#!/usr/bin/env python3
"""Prove a generated environment is physically and geometrically sound.

    python .claude/scripts/env/validate_env.py envs/mail_sorting --sheet

Runs the checks that a render cannot answer, writes `validation.json` into the
package, and optionally a contact sheet of every camera. Exit status is 1 if
any check fails, so this can gate the authoring loop.

Every check here exists because the corresponding failure actually happened
while this tooling was built:

  penetration   two bodies overlapping at t=0 make MuJoCo's solver produce a
                huge qacc on the first step. It auto-resets, so the symptom is
                not an exception — it is a scene that silently never moves.
  stability     stiff actuators diverge a few hundred steps in, long after any
                snapshot you would have looked at.
  reachability  a beautiful cell the arm physically cannot touch.
  framing       objects outside the policy camera's frustum. The policy sees
                only what the camera sees; nothing else about the scene matters.
  contract      images or state of the wrong shape or dtype silently break the
                checkpoint's normalization rather than raising.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrs.envs.scenegen import SceneSpec, SceneEnv  # noqa: E402

PENETRATION_LIMIT = 0.002       # metres, at t=0 before anything settles
SETTLED_LIMIT = 0.006           # metres, after settling: contact softness is fine


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name, passed, detail, **extra):
        self.checks.append({"check": name, "pass": bool(passed), "detail": detail, **extra})
        return passed

    @property
    def ok(self) -> bool:
        return all(c["pass"] for c in self.checks)

    def render(self) -> str:
        lines = []
        for check in self.checks:
            mark = "PASS" if check["pass"] else "FAIL"
            lines.append(f"  [{mark}] {check['check']:<22} {check['detail']}")
        return "\n".join(lines)


def _geom_name(model, index):
    import mujoco

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or f"geom{index}"


def penetrations(env, limit):
    """Contacts overlapping by more than `limit` metres."""
    found = []
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        if contact.dist < -limit:
            found.append({
                "a": _geom_name(env.model, contact.geom1),
                "b": _geom_name(env.model, contact.geom2),
                "depth_mm": round(float(-contact.dist) * 1000, 2),
            })
    return found


def check_static_pose(env, report: Report) -> None:
    """Interpenetration in the authored layout, before physics runs."""
    import mujoco

    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[env.arm_qpos_adr] = env.home_qpos
    if len(env.finger_qpos_adr):
        env.data.qpos[env.finger_qpos_adr] = env.robot.finger_open_qpos
    env._place_free_bodies()
    mujoco.mj_forward(env.model, env.data)

    hits = penetrations(env, PENETRATION_LIMIT)
    report.add(
        "layout_penetration",
        not hits,
        "no bodies overlap at t=0" if not hits
        else f"{len(hits)} overlapping pair(s), worst {max(h['depth_mm'] for h in hits):.1f} mm",
        overlaps=hits[:12],
    )


def check_stability(env, report: Report, steps: int) -> None:
    """Hold the arm still and let the scene run. Nothing should diverge."""
    import mujoco

    env.reset(seed=0)
    before = env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
    zero = np.zeros(7)
    for _ in range(steps):
        env.step(zero)
    diverged = env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number - before

    report.add("stability", diverged == 0,
               f"{steps} idle steps, {diverged} divergence warning(s)",
               bad_qacc=int(diverged))

    settled = penetrations(env, SETTLED_LIMIT)
    report.add("settled_penetration", not settled,
               "no deep overlap after settling" if not settled
               else f"{len(settled)} pair(s) overlapping >{SETTLED_LIMIT * 1000:.0f} mm",
               overlaps=settled[:12])

    finite = bool(np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qvel)))
    report.add("finite_state", finite, "qpos and qvel finite" if finite else "NaN or inf in state")


def check_reachability(env, report: Report) -> None:
    """Every manipulable body must sit inside the commanded workspace box.

    The box is what `_apply_leash` clamps end-effector commands to, so a part
    outside it is unreachable no matter what the policy does — the command is
    silently clipped short.
    """
    low = np.asarray(env.config.workspace_min)
    high = np.asarray(env.config.workspace_max)

    unreachable = []
    for body in env.spec.free_bodies:
        position = env.body_position(body.name)
        # Compare in the plane plus a generous vertical band: the part is
        # reached from above, so its own z only has to be below the box top.
        outside = (np.any(position[:2] < low[:2]) or np.any(position[:2] > high[:2])
                   or position[2] > high[2])
        if outside:
            unreachable.append({"body": body.name, "pos": [round(float(v), 3) for v in position]})

    report.add("reachability", not unreachable,
               f"{len(env.spec.free_bodies)} manipulable bodies inside the workspace box"
               if not unreachable else f"{len(unreachable)} outside: "
               f"{', '.join(u['body'] for u in unreachable)}",
               workspace_min=low.tolist(), workspace_max=high.tolist(), unreachable=unreachable)


def _in_frustum(env, camera_id, point, aspect=1.0, margin=0.0):
    """Is a world point inside this camera's view frustum?"""
    position = env.data.cam_xpos[camera_id]
    rotation = env.data.cam_xmat[camera_id].reshape(3, 3)
    # MuJoCo cameras look down their local -z.
    local = rotation.T @ (np.asarray(point) - position)
    depth = -local[2]
    if depth <= 1e-6:
        return False
    fovy = np.radians(env.model.cam_fovy[camera_id])
    half_h = np.tan(fovy / 2.0) * depth
    half_w = half_h * aspect
    return abs(local[0]) <= half_w * (1 + margin) and abs(local[1]) <= half_h * (1 + margin)


def check_framing(env, report: Report) -> None:
    """The policy's scene camera must actually see the task."""
    env.reset(seed=0)
    targets = [b.name for b in env.spec.free_bodies]
    targets += [b.name for b in env.spec.bodies if "bin" in b.tags or "target" in b.tags]

    missed = [
        name for name in dict.fromkeys(targets)
        if not _in_frustum(env, env.scene_camera_id, env.body_position(name))
    ]
    report.add("scene_camera_framing", not missed,
               f"all {len(set(targets))} task bodies in frame" if not missed
               else f"out of frame: {', '.join(missed)}",
               out_of_frame=missed)


def check_contract(env, report: Report) -> None:
    """Observation shapes and dtypes the policy stack depends on."""
    observation, _ = env.reset(seed=0)
    size = env.config.image_size
    problems = []

    for key in (env.config.scene_image_key, env.config.wrist_image_key):
        image = observation.get(key)
        if image is None:
            problems.append(f"{key} missing")
        elif image.dtype != np.uint8 or image.shape != (3, size, size):
            problems.append(f"{key} is {image.shape} {image.dtype}, expected (3, {size}, {size}) uint8")

    state = observation.get("observation.state")
    if state is None or state.dtype != np.float32 or state.ndim != 1:
        problems.append(f"observation.state is {None if state is None else (state.shape, state.dtype)}")

    report.add("observation_contract", not problems,
               "images (3,H,W) uint8 and float32 state" if not problems else "; ".join(problems))


def check_dynamics(env, report: Report, steps: int) -> None:
    """Moving elements must actually move something."""
    if not env.spec.dynamics:
        report.add("dynamics", True, "no dynamic elements declared")
        return

    env.reset(seed=0)
    start = {b.name: env.body_position(b.name).copy() for b in env.spec.free_bodies}
    zero = np.zeros(7)
    for _ in range(steps):
        env.step(zero)

    moved = {
        name: round(float(np.linalg.norm(env.body_position(name) - origin)), 4)
        for name, origin in start.items()
    }
    any_moved = any(distance > 0.01 for distance in moved.values())
    kinds = [d.kind for d in env.spec.dynamics]

    # A conveyor that transports nothing is the single most common outcome of
    # a mis-signed direction or a part spawned beside the belt rather than on it.
    transporting = [k for k in kinds if k in ("roller_conveyor", "belt_field")]
    report.add("dynamics", any_moved or not transporting,
               f"{kinds} moved parts by {moved}" if any_moved
               else f"{kinds} present but no free body moved more than 10 mm in {steps} steps",
               displacement=moved)


def contact_sheet(env, path: Path, size: int = 400) -> Path:
    """One PNG showing every camera, for the visual half of the review."""
    import imageio.v3 as iio
    import mujoco

    env.reset(seed=0)
    tiles = []
    for cam in env.spec.cameras:
        compiled = env.build.cameras.get(cam.name, cam.name)
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, compiled)
        if camera_id < 0:
            continue
        tiles.append((cam.name, env.render(size=size, camera=camera_id)))

    if not tiles:
        raise RuntimeError("no cameras to render")

    per_row = min(3, len(tiles))
    rows = []
    for start in range(0, len(tiles), per_row):
        chunk = [image for _, image in tiles[start:start + per_row]]
        while len(chunk) < per_row:
            chunk.append(np.zeros_like(chunk[0]))
        rows.append(np.hstack(chunk))

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.vstack(rows))
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", type=Path, help="environment package directory (holds spec.json)")
    parser.add_argument("--steps", type=int, default=200, help="idle control steps for the stability check")
    parser.add_argument("--sheet", action="store_true", help="also write cameras.png")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    spec_path = args.package / "spec.json"
    if not spec_path.is_file():
        print(f"no spec.json in {args.package}", file=sys.stderr)
        return 2

    report = Report()
    spec = SceneSpec.load(spec_path)

    try:
        env = SceneEnv(spec, asset_dir=args.package / "assets")
    except Exception as error:  # noqa: BLE001 - the compile error is the result
        report.add("compile", False, f"{type(error).__name__}: {error}")
        print(report.render())
        return 1

    report.add("compile", True,
               f"nq={env.model.nq} nv={env.model.nv} nu={env.model.nu} "
               f"nbody={env.model.nbody} ngeom={env.model.ngeom}")
    for warning in env.build.warnings:
        report.add("build_warning", False, warning)

    check_static_pose(env, report)
    check_contract(env, report)
    check_reachability(env, report)
    check_framing(env, report)
    check_stability(env, report, args.steps)
    check_dynamics(env, report, args.steps)

    if not spec.success.terms:
        report.add("success_defined", True,
                   "no success terms yet — expected before robo-task-define runs")

    payload = {"package": str(args.package), "scene": spec.name, "ok": report.ok,
               "checks": report.checks}
    if args.sheet:
        payload["contact_sheet"] = str(contact_sheet(env, args.package / "cameras.png"))

    (args.package / "validation.json").write_text(json.dumps(payload, indent=2))

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"{spec.name}: {'OK' if report.ok else 'PROBLEMS'}")
        print(report.render())
        if args.sheet:
            print(f"\n  contact sheet -> {payload['contact_sheet']}")
        print(f"  report        -> {args.package / 'validation.json'}")

    env.close()
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
