"""Resolve a file from either a local directory or the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path


def resolve_file(
    pretrained_name_or_path: str | Path,
    filename: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
    missing_ok: bool = False,
) -> str | None:
    """Return a local path to `filename`, downloading from the Hub if needed.

    `pretrained_name_or_path` is treated as a local directory when it exists on
    disk, otherwise as a Hub repo id.
    """
    local_dir = Path(pretrained_name_or_path)
    if local_dir.is_dir():
        candidate = local_dir / filename
        if candidate.is_file():
            return str(candidate)
        if missing_ok:
            return None
        raise FileNotFoundError(f"{filename} not found in {local_dir}")

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return hf_hub_download(
            repo_id=str(pretrained_name_or_path),
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
        )
    except EntryNotFoundError:
        if missing_ok:
            return None
        raise
