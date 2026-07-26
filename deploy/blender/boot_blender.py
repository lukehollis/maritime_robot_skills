"""Enable the BlenderMCP addon and open its socket, then get out of the way.

Run as `blender --factory-startup --python boot_blender.py`. Blender keeps its
own event loop alive from here; the addon's command handler runs on a timer
inside it, which is exactly why the addon refuses to serve under `blender -b`.

The work is deferred onto a timer because at --python time the window manager
has not finished building the UI, and the addon's start operator wants a real
context to attach to.
"""

import sys

import bpy

ADDON = "blender_mcp_addon"
PORT = 9876


def _enable_addon() -> bool:
    try:
        bpy.ops.preferences.addon_enable(module=ADDON)
        return True
    except Exception as exc:  # noqa: BLE001 - report and let the retry decide
        print(f"[boot] addon_enable({ADDON}) failed: {exc}", file=sys.stderr)
        return False


def _start_server() -> None:
    bpy.context.scene.blendermcp_port = PORT
    bpy.ops.blendermcp.start_server()
    print(f"[boot] BlenderMCP listening on {PORT}", flush=True)


_attempts = 0


def _boot():
    """Timer callback. Returns a delay to retry, or None to stop."""
    global _attempts
    _attempts += 1

    if not _enable_addon():
        return None if _attempts > 10 else 1.0

    try:
        _start_server()
    except Exception as exc:  # noqa: BLE001
        print(f"[boot] start_server failed: {exc}", file=sys.stderr)
        return None if _attempts > 10 else 1.0

    return None


bpy.app.timers.register(_boot, first_interval=1.0)
