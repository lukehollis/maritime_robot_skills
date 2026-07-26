"""Render the environment's cameras, to check framing and lighting.

    python -m mrs.scripts.visualize --out outputs/cameras.png
    python -m mrs.scripts.visualize --episode --out outputs/expert.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mrs.envs import PandaPickPlaceConfig, PandaPickPlaceEnv
from mrs.utils.video import write_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs/cameras.png"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode", action="store_true",
                        help="Record a scripted-expert episode instead of a still.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    env = PandaPickPlaceEnv(PandaPickPlaceConfig(seed=args.seed))
    observation, _ = env.reset(seed=args.seed)

    if args.episode:
        from mrs.envs.scripted_policy import ScriptedPickPlace

        expert = ScriptedPickPlace(env)
        expert.reset()
        frames = [env.render()]
        for _ in range(env.config.max_episode_steps):
            _, _, terminated, truncated, _ = env.step(expert.act())
            frames.append(env.render())
            if terminated or truncated:
                break
        path = write_video(args.out, frames, fps=env.config.control_freq)
        print(f"wrote {path}  ({len(frames)} frames)")
    else:
        import imageio.v3 as iio

        panel = np.concatenate(
            [
                observation[env.config.scene_image_key].transpose(1, 2, 0),
                observation[env.config.wrist_image_key].transpose(1, 2, 0),
            ],
            axis=1,
        )
        iio.imwrite(args.out, panel)
        print(f"wrote {args.out}  (left: {env.config.scene_camera}, right: {env.config.wrist_camera})")

    state = env.get_state()
    print(f"state: eef={np.round(state[:3], 3)} axis_angle={np.round(state[3:6], 3)} "
          f"fingers={np.round(state[6:8], 3)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
