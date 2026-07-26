#!/usr/bin/env python3
"""Watch a generated environment run live, in an interactive MuJoCo viewer.

    mjpython .claude/scripts/env/watch_env.py envs/parcel_sorting

On macOS this MUST be launched with `mjpython`, not `python3`: the interactive
viewer needs to own the main thread's UI run loop, and CPython will not give it
one. The script says so and exits rather than failing deep inside GLFW.

By default it drives the scene with the scripted expert and restarts with a new
seed whenever an episode ends, so the cell runs continuously — useful for
eyeballing whether a scene behaves like the thing it is meant to represent
before any policy is trained against it.

Controls are the standard viewer ones: drag to orbit, scroll to zoom,
double-click a body to select it, backspace to reset the camera. Close the
window to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", type=Path, help="environment package (holds spec.json)")
    parser.add_argument("--policy", default="auto", choices=["auto", "sort", "weld", "idle"],
                        help="auto picks the expert the scene calls for — a sorter when free "
                             "bodies pair with destinations, a welder when the scene carries "
                             "tool sites; idle holds the arm still so you can watch the "
                             "scene's own dynamics")
    parser.add_argument("--object-tag", default=None,
                        help="tag identifying manipulable objects (default: auto-detect)")
    parser.add_argument("--destination-prefix", default="bin_")
    parser.add_argument("--site-tag", default="weld_point",
                        help="tag marking tool-task sites (welding, dispensing, inspection)")
    parser.add_argument("--speed", type=float, default=1.0, help="real-time multiplier")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first episode")
    parser.add_argument("--episodes", type=int, default=0, help="0 runs until the window closes")
    parser.add_argument("--pause", type=float, default=1.5, help="seconds between episodes")
    parser.add_argument("--grace-steps", type=int, default=240,
                        help="steps to keep running after the expert finishes, so the "
                             "success predicate's hold window can register. Measured lag "
                             "on parcel_sorting reaches 160 steps when the last parcel "
                             "settles slowly, so anything much under 200 reports correct "
                             "runs as incomplete")
    args = parser.parse_args(argv)

    try:
        import mujoco.viewer
    except ImportError:
        print("mujoco.viewer is unavailable; install a mujoco build with viewer support.",
              file=sys.stderr)
        return 2

    # `sys.executable` still reports plain python3 under mjpython, so testing it
    # rejects legitimate runs. mujoco.viewer sets `_MJPYTHON` only when the
    # launcher installed the main-thread run loop, which is the actual
    # precondition.
    if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
        print(
            "On macOS the interactive viewer must run under mjpython, which owns the\n"
            "main-thread UI run loop. Re-run as:\n\n"
            f"    mjpython {' '.join([str(Path(__file__)), *(argv or sys.argv[1:])])}\n",
            file=sys.stderr,
        )
        return 2

    from mrs.envs.scenegen import SceneEnv, SceneSpec
    from mrs.envs.scenegen.scripted import (
        ScriptedSorter, ScriptedWelder, assignments_by_tag, sites_by_tag,
    )

    spec = SceneSpec.load(args.package / "spec.json")
    env = SceneEnv(spec, asset_dir=args.package / "assets")
    # The viewer draws the scene itself; rendering the policy cameras every
    # step would roughly triple the cost for no visible benefit.
    env.render_observations = False

    tag = args.object_tag or _guess_object_tag(spec)
    pairs = assignments_by_tag(spec, object_tag=tag, destination_prefix=args.destination_prefix) \
        if tag else []
    sites = sites_by_tag(spec, args.site_tag)

    # A sorting scene and a tool scene need different experts, and which one a
    # scene is can be read off the scene itself: sortable pairs, or tool sites.
    mode = args.policy
    if mode == "auto":
        mode = "sort" if pairs else ("weld" if sites else "idle")

    expert = None
    if mode == "sort" and pairs:
        expert = ScriptedSorter(env, pairs)
    elif mode == "weld" and sites:
        settle = next((b.name for b in spec.free_bodies), None)
        expert = ScriptedWelder(env, sites, settle_body=settle)
    elif mode != "idle":
        print(f"no expert available for policy={mode!r} in this scene; holding idle",
              file=sys.stderr)

    print(f"scene    : {spec.name}")
    print(f"task     : {spec.task}")
    print(f"expert   : {mode} -> {type(expert).__name__ if expert else 'idle'}")
    print(f"targets  : {pairs or sites or 'none'}")
    print(f"dynamics : {[d.kind for d in spec.dynamics] or 'none'}")
    print("\nclose the viewer window to stop\n")

    dt = 1.0 / spec.control.control_freq
    episode = 0
    seed = args.seed
    settled_at: int | None = None
    env.reset(seed=seed)
    if expert:
        expert.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()

            action = expert.act() if expert is not None else np.zeros(7)

            _, _, terminated, truncated, info = env.step(action)
            viewer.sync()

            # The expert finishing is not the same as the task being complete:
            # success requires the predicate to hold for several consecutive
            # steps while the parcels settle. Cutting the episode at
            # `expert.done` reports a correct run as incomplete, so give it a
            # grace window and let `terminated` be the thing that decides.
            if expert is not None and expert.done and settled_at is None:
                settled_at = info.get("step", 0)
            expired = (settled_at is not None
                       and info.get("step", 0) - settled_at >= args.grace_steps
                       and _all_settled(env, pairs))

            if terminated or truncated or expired:
                episode += 1
                outcome = "SUCCESS" if info.get("is_success") else "incomplete"
                modes = info.get("failure_modes") or []
                # Flushed explicitly: stdout is block-buffered when redirected,
                # so without this a long run shows nothing until it exits.
                print(f"episode {episode:>3}  seed={seed}  {outcome:<11} "
                      f"steps={info.get('step')}" + (f"  failures={modes}" if modes else ""),
                      flush=True)

                if args.episodes and episode >= args.episodes:
                    break

                # Let the finished state sit on screen for a moment before the
                # scene snaps back, otherwise the reset reads as a glitch.
                deadline = time.perf_counter() + args.pause
                while viewer.is_running() and time.perf_counter() < deadline:
                    viewer.sync()
                    time.sleep(0.01)

                seed += 1
                settled_at = None
                env.reset(seed=seed)
                if expert:
                    expert.reset()
                viewer.sync()
                continue

            elapsed = time.perf_counter() - started
            remaining = dt / max(args.speed, 1e-6) - elapsed
            if remaining > 0:
                time.sleep(remaining)

    env.close()
    print(f"\nstopped after {episode} episode(s)")
    return 0


def _guess_object_tag(spec) -> str | None:
    """The tag shared by the free bodies is what the sorter selects on."""
    counts: dict[str, int] = {}
    for body in spec.bodies:
        if body.kind != "free":
            continue
        for tag in body.tags:
            if not tag.startswith("size_"):
                counts[tag] = counts.get(tag, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _all_settled(env, pairs, speed=0.05) -> bool:
    for body, _ in pairs:
        dof = env.free_dof_adr.get(body)
        if dof is not None and float(np.linalg.norm(env.data.qvel[dof:dof + 3])) > speed:
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
