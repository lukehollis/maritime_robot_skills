#!/usr/bin/env bash
# Boot order: persistent state -> X server -> Blender+MCP -> OpenClaw gateway.
#
# Nothing here is fatal except a missing /data. If Blender fails to come up the
# gateway still starts, because an agent that can tell you why it is broken is
# worth more than a container that exits.
#
# Maritime owns some of this container already: it sets HOME=/data, seeds
# /data/.openclaw with a config pointing at its own LLM proxy and MCP servers,
# and expects the gateway on 18789. This script adds to that state, never
# replaces it.
set -uo pipefail

log() { printf '[entrypoint] %s\n' "$*" >&2; }

# maritime-init replaces PATH wholesale before launching this script, so the
# Dockerfile's ENV PATH is gone by now: `python3` resolves to Debian's (no
# mujoco, no mrs) and mrs-job/mrs-report are not on it at all. The gateway is
# this script's child, and the agent's shell commands are the gateway's
# children, so putting them back here is what makes the pipeline runnable.
export PATH=/opt/venv/bin:/opt/mrs/deploy/bin:$PATH
PYTHON=/opt/venv/bin/python3

LAB=${MRS_LAB:-/data/lab}
STATE="${HOME:-/data}/.openclaw"
LOGS=/data/logs

# --- persistent layout --------------------------------------------------------
#
# /data survives redeploys; the image does not. Everything the agent produces or
# downloads lives under /data, and $LAB is shaped to look like the project root
# the pipeline skills were written against.
mkdir -p "$LAB"/{envs,runs,jobs,outputs,.cache} "$LOGS" "${HF_HOME:-/data/hf}" || {
    log "FATAL: /data is not writable"; exit 1
}
ln -sfn /opt/mrs/.claude "$LAB/.claude"

# --- OpenClaw config ----------------------------------------------------------
#
# The repo owns the keys it declares (the Blender MCP server, the skill roots);
# Maritime and the gateway own everything else in that file. Merge, never copy.
# Keep an untouched copy of whatever Maritime handed us, before anything of ours
# runs. When the gateway rejects a config the VM panics and takes `maritime exec`
# with it, so this file is the only way to see what it was actually given.
cp "$STATE/openclaw.json" "$LOGS/openclaw.as-received.json" 2>/dev/null \
    && log "saved as-received config to $LOGS/openclaw.as-received.json"

# Written to a file, never piped into a read loop. A `cmd | while read` in the
# boot path is a way to hang forever: any process that inherits the write end of
# that pipe holds it open, `read` never sees EOF, and the gateway is never
# started — which is exactly what happened.
"$PYTHON" - >"$LOGS/config-check.log" 2>&1 <<'PY'
import json, pathlib
p = pathlib.Path("/data/.openclaw/openclaw.json")
try:
    d = json.loads(p.read_text())
    print("as-received root keys: " + ", ".join(sorted(d)))
except FileNotFoundError:
    print("as-received config: MISSING")
except json.JSONDecodeError as exc:
    # If Maritime writes JSON5, json.loads fails here and the reconcile would
    # have started from {} — dropping gateway auth and model config entirely.
    print(f"as-received config DOES NOT PARSE as strict JSON: {exc}")
PY
log "config check: $(cat "$LOGS/config-check.log" 2>/dev/null | head -1)"

# MRS_RECONCILE=0 boots the stock Maritime agent untouched: no MCP server, no
# skill roots. It is the escape hatch for exactly the situation where our own
# config changes are what is stopping the VM from coming up.
if [ "${MRS_RECONCILE:-1}" = "0" ]; then
    log "MRS_RECONCILE=0 — leaving openclaw.json alone (no MCP tools, no skills)"
else
    "$PYTHON" /opt/mrs/deploy/reconcile_config.py \
        --managed /opt/mrs/deploy/openclaw.json \
        --target "$STATE/openclaw.json" || log "WARN: config reconcile failed"

    # Diagnostics only, and strictly off the boot path: `doctor` is slow and can
    # sit there indefinitely, and nothing may block the gateway from starting.
    # The supervisor at the bottom of this script is what actually protects
    # against a bad config.
    ( cd /app && timeout 120 node openclaw.mjs doctor >"$LOGS/doctor.log" 2>&1 ) &
fi

# --- exec approvals -----------------------------------------------------------
#
# There is nobody to approve a shell command here: the only human interface is
# `maritime chat`, and an approval prompt would simply hang the pipeline. The
# blast radius is one disposable microVM whose only durable state is /data.
if [ ! -e "$STATE/exec-approvals.json" ]; then
    cat >"$STATE/exec-approvals.json" <<'JSON'
{
  "version": 1,
  "defaults": {
    "security": "full",
    "ask": "off",
    "askFallback": "full"
  }
}
JSON
    log "wrote exec-approvals.json (unattended: full, no prompts)"
fi

# --- workspace ----------------------------------------------------------------
#
# PIPELINE.md documents code, so it is regenerated from the repo every boot. The
# pointer into it is appended to AGENTS.md once, under a marker, so Maritime's
# own seeded instructions survive.
WS="$STATE/workspace"
mkdir -p "$WS"
cp /opt/mrs/deploy/workspace/PIPELINE.md "$WS/PIPELINE.md"

MARKER="<!-- mrs-lab -->"
if [ ! -e "$WS/AGENTS.md" ] || ! grep -qF "$MARKER" "$WS/AGENTS.md" 2>/dev/null; then
    cat /opt/mrs/deploy/workspace/AGENTS.append.md >>"$WS/AGENTS.md"
    log "appended pipeline instructions to AGENTS.md"
fi

# --- /etc/hosts ---------------------------------------------------------------
#
# The Maritime microVM ships an empty /etc/hosts, so `localhost` does not
# resolve. The BlenderMCP addon binds `localhost` by default and dies with
# "Name or service not known"; plenty of other tooling assumes it too.
if ! grep -q localhost /etc/hosts 2>/dev/null; then
    printf '127.0.0.1 localhost\n::1 localhost ip6-localhost\n' >>/etc/hosts \
        && log "seeded /etc/hosts" || log "WARN: could not write /etc/hosts"
fi

# --- X server -----------------------------------------------------------------
DISPLAY=${DISPLAY:-:99}
DISPLAY_NUM=${DISPLAY#:}
export DISPLAY
if ! pgrep -f "Xvfb :${DISPLAY_NUM}" >/dev/null 2>&1; then
    log "starting Xvfb on ${DISPLAY}"
    Xvfb "${DISPLAY}" -screen 0 1920x1080x24 -nolisten tcp >"$LOGS/xvfb.log" 2>&1 &
    for _ in $(seq 1 40); do
        xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1 && break
        sleep 0.25
    done
fi
if xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    log "X server up"
else
    log "WARN: X server did not come up; Blender will fail"
fi

# --- Blender + the MCP socket -------------------------------------------------
#
# No --factory-startup: the boot script installs the addon through Blender's own
# installer, which writes into the user config, and factory startup would then
# ignore what it just installed.
#
# `pgrep -x` matches the process name only. Matching the full command line would
# also match this script, which contains the word.
if ! pgrep -x blender >/dev/null 2>&1; then
    log "starting Blender"
    blender --python /opt/mrs/deploy/blender/boot_blender.py \
        >"$LOGS/blender.log" 2>&1 &
fi

PORT=${BLENDER_PORT:-9876}
for _ in $(seq 1 120); do
    (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null && break
    sleep 1
done
if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
    log "Blender MCP socket listening on ${PORT}"
else
    log "WARN: Blender MCP socket never opened — see $LOGS/blender.log"
fi

# --- MuJoCo assets ------------------------------------------------------------
#
# Menagerie is ~2 GB whole and /data has 10 GB to share with a 7.5 GB
# checkpoint, so it is fetched on demand rather than baked in. Warm the Panda in
# the background so stage 0's prerequisite check passes on the first run.
ASSETS=${MRS_ASSET_DIR:-$LAB/.cache}
if [ ! -e "${ASSETS}/mujoco_menagerie/franka_emika_panda/panda.xml" ]; then
    log "fetching mujoco_menagerie in the background"
    (
        cd "$LAB" && "$PYTHON" -c "from mrs.envs import assets; print(assets.panda_model_path())" \
            >"$LOGS/menagerie.log" 2>&1 \
            && log "menagerie ready" || log "WARN: menagerie fetch failed"
        /opt/mrs/deploy/bin/mrs-prune-assets >>"$LOGS/menagerie.log" 2>&1 || true
    ) &
else
    /opt/mrs/deploy/bin/mrs-prune-assets >>"$LOGS/menagerie.log" 2>&1 &
fi

# --- OpenClaw gateway ---------------------------------------------------------
#
# Not `exec`, deliberately. The gateway is the VM's init: if it exits, the kernel
# panics and the machine is gone, along with any chance of logging in to find out
# why. So supervise the first start instead — if it dies quickly, put Maritime's
# own config back and start again. Whatever we got wrong, the agent comes up and
# can be asked about it.
log "handing off to: $*"
cd /app || exit 1

started=$(date +%s)
"$@" &
gateway=$!
trap 'kill -TERM "$gateway" 2>/dev/null' TERM INT
wait "$gateway"
code=$?
elapsed=$(( $(date +%s) - started ))

if [ "$elapsed" -lt 90 ] && [ -e "$STATE/openclaw.json.pre-mrs" ]; then
    log "gateway exited after ${elapsed}s (code $code) — restoring the pre-merge config and retrying"
    log "the agent will come up WITHOUT the Blender MCP server or pipeline skills"
    cp "$STATE/openclaw.json.pre-mrs" "$STATE/openclaw.json"
    exec "$@"
fi

log "gateway exited after ${elapsed}s (code $code)"
exit "$code"
