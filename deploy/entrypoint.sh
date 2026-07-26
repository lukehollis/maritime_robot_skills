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
python3 /opt/mrs/deploy/reconcile_config.py \
    --managed /opt/mrs/deploy/openclaw.json \
    --target "$STATE/openclaw.json" || log "WARN: config reconcile failed"

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
DISPLAY_NUM=${DISPLAY#:}
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
if [ ! -e "${MRS_ASSET_DIR}/mujoco_menagerie/franka_emika_panda/panda.xml" ]; then
    log "fetching mujoco_menagerie in the background"
    (
        cd "$LAB" && python3 -c "from mrs.envs import assets; print(assets.panda_model_path())" \
            >"$LOGS/menagerie.log" 2>&1 \
            && log "menagerie ready" || log "WARN: menagerie fetch failed"
    ) &
fi

# --- OpenClaw gateway ---------------------------------------------------------
log "handing off to: $*"
cd /app || exit 1
exec "$@"
