"""Locate (and if necessary fetch) the MuJoCo Menagerie robot models."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
PANDA_SUBDIR = "franka_emika_panda"
PANDA_MODEL = "panda.xml"


def default_cache_dir() -> Path:
    """Where downloaded models live. Override with `MRS_ASSET_DIR`."""
    env_dir = os.environ.get("MRS_ASSET_DIR")
    if env_dir:
        return Path(env_dir)
    # Default to a cache beside the repo so a checkout stays self-contained.
    return Path(__file__).resolve().parents[2] / ".cache"


def menagerie_path(*, cache_dir: Path | None = None, download: bool = True) -> Path:
    """Return the Menagerie checkout, cloning it on first use."""
    root = (cache_dir or default_cache_dir()) / "mujoco_menagerie"
    if (root / PANDA_SUBDIR / PANDA_MODEL).is_file():
        return root

    if not download:
        raise FileNotFoundError(
            f"MuJoCo Menagerie not found at {root}. Clone it with:\n"
            f"  git clone --depth 1 {MENAGERIE_URL} {root}"
        )
    if shutil.which("git") is None:
        raise RuntimeError(f"git is required to fetch {MENAGERIE_URL}, but is not on PATH.")

    root.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning MuJoCo Menagerie into %s (one-time, ~2 GB)...", root)
    subprocess.run(
        ["git", "clone", "--depth", "1", MENAGERIE_URL, str(root)],
        check=True,
        capture_output=True,
    )
    return root


def panda_model_path(**kwargs) -> Path:
    """Path to the Franka Emika Panda MJCF."""
    return menagerie_path(**kwargs) / PANDA_SUBDIR / PANDA_MODEL
