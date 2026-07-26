"""Evaluate a policy on the Panda pick-and-place task.

    python -m mrs.scripts.eval --episodes 5 --video
    python -m mrs.scripts.eval --policy scripted --episodes 20
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mrs.envs import PandaPickPlaceConfig, PandaPickPlaceEnv
from mrs.rollout import evaluate
from mrs.utils.video import write_video

DEFAULT_CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"

logger = logging.getLogger("mrs.eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--policy", default="pi05", choices=["pi05", "scripted"],
                        help="pi05 runs the VLA; scripted runs the privileged-state expert.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Hub repo id or local directory holding the pi0.5 weights.")
    parser.add_argument("--device", default=None, help="torch device (default: auto-detect).")

    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the first episode.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--task", default=None, help="Override the language instruction.")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="Actions executed per inference. Lower means more frequent replanning "
                             "(the checkpoint default is a full 50-step open-loop chunk).")

    parser.add_argument("--rotate-images-180", action="store_true",
                        help="Feed the policy 180-degree-rotated camera images, reproducing the "
                             "image convention openpi uses for LIBERO.")
    parser.add_argument("--video", action="store_true", help="Record a video per episode.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    env_config = PandaPickPlaceConfig(seed=args.seed)
    if args.task:
        env_config.task = args.task
    if args.max_steps:
        env_config.max_episode_steps = args.max_steps
    env = PandaPickPlaceEnv(env_config)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.policy == "scripted":
        results, summary = _evaluate_scripted(env, args)
    else:
        results, summary = _evaluate_pi05(env, args)

    print("\n" + "=" * 62)
    print(f"  policy        : {args.policy}")
    if args.policy == "pi05":
        print(f"  checkpoint    : {args.checkpoint}")
    print(f"  task          : {env_config.task}")
    print(f"  success rate  : {summary['successes']}/{summary['episodes']} "
          f"({summary['success_rate']:.0%})")
    print(f"  mean steps    : {summary['mean_steps']:.1f}")
    print(f"  mean wall time: {summary['mean_wall_time_s']:.1f}s per episode")
    print("=" * 62)

    if args.video:
        for result in results:
            if not result.frames:
                continue
            path = write_video(
                args.out_dir / f"episode_{result.seed:03d}"
                f"_{'success' if result.success else 'fail'}.mp4",
                result.frames,
                fps=env_config.control_freq,
            )
            logger.info("wrote %s", path)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "policy": args.policy,
                "checkpoint": args.checkpoint if args.policy == "pi05" else None,
                "env": asdict(env_config),
                "summary": summary,
                "episodes": [
                    {"seed": r.seed, "success": r.success, "steps": r.steps,
                     "inference_calls": r.inference_calls, "wall_time_s": round(r.wall_time, 2)}
                    for r in results
                ],
            },
            indent=2,
            default=str,
        )
    )
    logger.info("wrote %s", summary_path)

    env.close()
    return 0


def _evaluate_pi05(env, args):
    from mrs.policies import make_policy

    device = args.device or pick_device()
    logger.info("loading %s on %s", args.checkpoint, device)

    overrides = {}
    if args.n_action_steps:
        overrides["n_action_steps"] = args.n_action_steps

    policy, preprocessor, postprocessor = make_policy(
        args.checkpoint, device=device, config_overrides=overrides
    )
    logger.info(
        "loaded %s  (%.2fB params, chunk=%d, executing %d actions per inference)",
        type(policy).__name__,
        sum(p.numel() for p in policy.parameters()) / 1e9,
        policy.config.chunk_size,
        policy.config.n_action_steps,
    )

    return evaluate(
        env,
        policy,
        preprocessor,
        postprocessor,
        episodes=args.episodes,
        start_seed=args.seed,
        record_video=args.video,
        task=args.task,
        rotate_images_180=args.rotate_images_180,
    )


def _evaluate_scripted(env, args):
    """Baseline: the privileged-state expert, through the same action interface."""
    from mrs.envs.scripted_policy import ScriptedPickPlace
    from mrs.rollout import RolloutResult

    expert = ScriptedPickPlace(env)
    results = []

    for index in range(args.episodes):
        seed = args.seed + index
        _, info = env.reset(seed=seed)
        expert.reset()

        result = RolloutResult(success=False, steps=0, total_reward=0.0, seed=seed)
        if args.video:
            result.frames.append(env.render())

        for step in range(env.config.max_episode_steps):
            action = expert.act()
            _, reward, terminated, truncated, info = env.step(action)
            result.actions.append(action)
            result.total_reward += float(reward)
            result.steps = step + 1
            if args.video:
                result.frames.append(env.render())
            if terminated or truncated:
                result.success = bool(info.get("is_success", False))
                break

        results.append(result)
        logger.info("episode %d/%d  seed=%d  success=%s  steps=%d",
                    index + 1, args.episodes, seed, result.success, result.steps)

    successes = [r.success for r in results]
    summary = {
        "episodes": args.episodes,
        "successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if results else 0.0,
        "mean_steps": float(np.mean([r.steps for r in results])) if results else 0.0,
        "mean_wall_time_s": 0.0,
        "total_inference_calls": 0,
    }
    return results, summary


if __name__ == "__main__":
    raise SystemExit(main())
