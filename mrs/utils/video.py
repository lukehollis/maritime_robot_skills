"""Write a sequence of RGB frames to a video file.

Frames are piped to the `ffmpeg` binary rather than going through an imageio
plugin, because the available plugin backends vary a lot between environments.
Falls back to an animated GIF when ffmpeg is not installed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def write_video(path: str | Path, frames: list[np.ndarray] | np.ndarray, fps: float = 20.0) -> Path:
    """Encode `frames` (a sequence of `uint8 [H, W, 3]`) and return the path written."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shaped (T, H, W, 3), got {frames.shape}.")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return _write_gif(path.with_suffix(".gif"), frames, fps)

    height, width = frames.shape[1:3]
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        # yuv420p needs even dimensions; the scale filter is a no-op otherwise.
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(path),
    ]
    try:
        subprocess.run(command, input=frames.tobytes(), check=True, capture_output=True)
        return path
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "ffmpeg failed (%s); falling back to GIF.", exc.stderr.decode()[-200:].strip()
        )
        return _write_gif(path.with_suffix(".gif"), frames, fps)


def _write_gif(path: Path, frames: np.ndarray, fps: float) -> Path:
    import imageio.v3 as iio

    iio.imwrite(path, frames, duration=1000.0 / fps, loop=0)
    return path
