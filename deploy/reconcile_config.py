"""Merge the repo's managed OpenClaw keys into the agent's live config.

A redeploy ships new managed settings (MCP servers, skill roots, model choice)
but must not discard what the running gateway wrote for itself — pairing state,
session bookkeeping, wizard flags. So this is a deep merge with the repo
winning on the keys it actually declares, rather than a copy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def deep_merge(base: dict, managed: dict) -> dict:
    """Return `base` updated by `managed`, recursing into dicts only.

    Lists are replaced wholesale: a managed list is a complete statement of
    intent (the set of skill roots, say), not something to append to.
    """
    out = dict(base)
    for key, value in managed.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--managed", required=True, type=Path)
    ap.add_argument("--target", required=True, type=Path)
    args = ap.parse_args()

    managed = json.loads(args.managed.read_text())

    if args.target.exists():
        try:
            base = json.loads(args.target.read_text())
        except json.JSONDecodeError:
            # A corrupt config is not worth preserving, and refusing to boot
            # over it would strand the agent.
            backup = args.target.with_suffix(".json.corrupt")
            args.target.rename(backup)
            print(f"unparseable config moved to {backup}")
            base = {}
    else:
        base = {}

    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(json.dumps(deep_merge(base, managed), indent=2) + "\n")
    print(f"reconciled {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
