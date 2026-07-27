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


# Root keys to delete from the live config before merging. A merge only ever
# adds, so a key we shipped by mistake would otherwise persist on /data forever.
# `_comment` was one: openclaw.json is validated strictly, and an explanatory
# key at the root made the gateway refuse to start — which panics the microVM,
# since the gateway is PID 1.
PRUNE_ROOT_KEYS = ("_comment",)


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
        except json.JSONDecodeError as exc:
            # Do not touch a config we cannot read. The earlier version renamed
            # it and merged onto {}, which produced a file with no gateway, no
            # auth and no models — the gateway rejects that at <root>, and
            # because it is PID 1 the VM panics. OpenClaw's own config format is
            # JSON5, so "does not parse as strict JSON" is a thing that can
            # legitimately happen here and must never be destructive.
            print(f"target does not parse as strict JSON ({exc}); leaving it untouched")
            return 0
    else:
        base = {}

    for key in PRUNE_ROOT_KEYS:
        if base.pop(key, None) is not None:
            print(f"pruned stale root key {key!r}")

    # Keep the pre-merge config. If the merge produces something the gateway
    # rejects it exits, and because it is PID 1 the whole microVM panics — so
    # the only way back in is `maritime exec` restoring this file.
    if base:
        args.target.with_suffix(".json.pre-mrs").write_text(json.dumps(base, indent=2) + "\n")

    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(json.dumps(deep_merge(base, managed), indent=2) + "\n")
    print(f"reconciled {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
