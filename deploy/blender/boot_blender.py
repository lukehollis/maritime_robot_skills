"""Enable the BlenderMCP addon and open its socket, then get out of the way.

Run as `blender --python boot_blender.py`. Blender keeps its own event loop
alive from here; the addon's command handler runs on a timer inside it, which is
exactly why the addon refuses to serve under `blender -b`.

The work is deferred onto a timer because at --python time the window manager
has not finished building the UI, and the addon's start operator wants a real
context to attach to.

Getting the addon *loadable* is the fiddly part. Blender 4.2 dropped the bundled
`scripts/addons/` directory and `BLENDER_USER_SCRIPTS` did not make the file
importable either, so the reliable route is Blender's own installer: it copies
the source into whichever user addon directory this build actually scans. The
manual import is kept as a last resort — the addon is a plain module, and
`register()` is all that enabling it really does.
"""

import sys
from pathlib import Path

import bpy

ADDON = "blender_mcp_addon"
SOURCE = Path("/opt/blender-mcp/addon.py")
HOST = "127.0.0.1"
PORT = 9876


def _try_enable() -> bool:
    try:
        bpy.ops.preferences.addon_enable(module=ADDON)
        return True
    except Exception:  # noqa: BLE001 - every failure here has a fallback
        return False


def _try_install_then_enable() -> bool:
    if not SOURCE.exists():
        print(f"[boot] {SOURCE} is missing", file=sys.stderr)
        return False
    try:
        bpy.ops.preferences.addon_install(filepath=str(SOURCE), overwrite=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[boot] addon_install failed: {exc}", file=sys.stderr)
        return False
    return _try_enable()


def _try_manual_register() -> bool:
    """Load the addon as an ordinary module and register it by hand."""
    import importlib.util

    if not SOURCE.exists():
        return False
    try:
        spec = importlib.util.spec_from_file_location(ADDON, SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[ADDON] = module
        spec.loader.exec_module(module)
        module.register()
        print("[boot] addon registered manually", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[boot] manual register failed: {exc}", file=sys.stderr)
        return False


def _port_open() -> bool:
    """Is something already listening?

    Only the TCP handshake is tested, never a command: the addon answers
    commands from a timer on this very thread, so a blocking round-trip from
    here would deadlock against ourselves.
    """
    import socket

    try:
        with socket.create_connection((HOST, PORT), timeout=2):
            return True
    except OSError:
        return False


def _start_server() -> None:
    """Get a server listening on a literal address, however it gets there.

    Two things bite here. The addon defaults to binding `localhost`, and the
    Maritime microVM ships an empty /etc/hosts, so the name does not resolve and
    `start()` dies with "Name or service not known". And enabling the addon can
    bring its own server up, in which case a second one only earns EADDRINUSE.

    So: if the port already answers, we are done — that is the outcome we
    actually want, no matter which object owns it.
    """
    if _port_open():
        print(f"[boot] BlenderMCP already listening on {HOST}:{PORT}", flush=True)
        return

    module = sys.modules.get(ADDON)
    if module is None:
        import importlib

        module = importlib.import_module(ADDON)

    existing = getattr(bpy.types, "blendermcp_server", None)
    if existing is None or getattr(existing, "host", None) != HOST:
        bpy.types.blendermcp_server = module.BlenderMCPServer(host=HOST, port=PORT)

    bpy.context.scene.blendermcp_port = PORT
    bpy.ops.blendermcp.start_server()

    if not _port_open():
        raise RuntimeError("start_server returned but nothing is listening")
    print(f"[boot] BlenderMCP listening on {HOST}:{PORT}", flush=True)


_attempts = 0


def _boot():
    """Timer callback. Returns a delay to retry with, or None to stop."""
    global _attempts
    _attempts += 1

    loaded = _try_enable() or _try_install_then_enable() or _try_manual_register()
    if not loaded:
        print(f"[boot] could not load {ADDON} (attempt {_attempts})", file=sys.stderr)
        return None if _attempts >= 5 else 2.0

    try:
        _start_server()
    except Exception as exc:  # noqa: BLE001
        print(f"[boot] start_server failed: {exc}", file=sys.stderr)
        return None if _attempts >= 5 else 2.0

    return None


bpy.app.timers.register(_boot, first_interval=1.0)
