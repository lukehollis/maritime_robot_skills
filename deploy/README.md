# Deploying the lab

`mrs-lab` is the four-stage evaluation pipeline running as an always-on agent on
[Maritime](https://maritime.sh), driven by `maritime chat`. Everything the
pipeline needs is inside one container: OpenClaw for the agent loop, Blender
behind a virtual display for scene authoring, MuJoCo and the pi0.5 stack for
rollouts, and Quarto for the reports.

## Layout

```
Dockerfile                 the image: OpenClaw + Xvfb/Mesa + Blender + MuJoCo + Quarto
maritime.json              agent name, framework, persona
deploy/
  entrypoint.sh            /data layout -> X server -> Blender+MCP -> gateway
  openclaw.json            the keys this repo owns: blender MCP server, skill roots
  reconcile_config.py      deep-merges those keys into the live config on boot
  blender/boot_blender.py  runs inside Blender: enable the addon, open the socket
  workspace/               PIPELINE.md and the AGENTS.md addition
  bin/mrs-job              dispatch long work, then watch it
  bin/mrs-report           render an eval run into report.html
  report/report.qmd        the report template
  skills/robo-report/      stage 4
.claude/skills/            stages 0-3
```

## Deploy

```bash
export MARITIME_TOKEN=mk_...            # or keep it in .env
maritime deploy mrs-lab --source github --repo https://github.com/<you>/maritime_robot_skills --wait
maritime logs mrs-lab --level error
maritime chat mrs-lab "what do you have installed?"
```

The agent was created with resources the platform defaults do not give you:

```bash
maritime create mrs-lab --ram 16384 --cpu 4 --disk 40 --always-on
```

`--always-on` matters. Serverless idle-sleep would suspend the container in the
middle of a two-hour rollout; `--disk` is clamped to the account's storage cap,
so confirm what you actually got rather than what you asked for:

```bash
maritime exec mrs-lab -- df -h /data /
```

## What the container is

| | |
|---|---|
| CPU | 4 vCPU, **no GPU** — pi0.5 runs ~20-30 s per action chunk |
| RAM | 16 GB (`/tmp` is a 7.9 GB tmpfs and comes out of it) |
| `/data` | 10 GB, persistent across redeploys; the pi0.5 checkpoint is 7.5 GB of it |
| `/` | provisioned from the image; the image itself is ~5 GB |
| display | Xvfb `:99` + Mesa llvmpipe, because the BlenderMCP addon refuses `blender -b` |
| MuJoCo | `MUJOCO_GL=osmesa`, software offscreen rendering |
| model | whatever Maritime's LLM proxy serves — `openai/gpt-5.4` by default |

Nothing in the image is state. `/data` holds everything the agent produces:

```
/data/lab/envs/<slug>/       environment packages and their eval runs
/data/lab/jobs/<id>/         mrs-job bookkeeping
/data/hf/                    checkpoint cache
/data/logs/                  xvfb.log, blender.log, menagerie.log
/data/.openclaw/             gateway config, workspace, sessions (Maritime seeds this)
```

## Checking it came up

```bash
maritime exec mrs-lab -- tail -20 /data/logs/blender.log
maritime exec mrs-lab -- sh -c 'echo > /dev/tcp/127.0.0.1/9876 && echo "MCP socket open"'
maritime exec mrs-lab -- python3 -c "import mujoco, mrs.envs.scenegen; print(mujoco.__version__)"
maritime exec mrs-lab -- quarto --version
```

If the Blender socket never opens, `blender.log` says why and the agent still
boots — the gateway does not depend on it. That is deliberate: an agent that can
explain the failure beats a container that exits.

## Pulling a report out

```bash
maritime exec mrs-lab -- cat /data/lab/envs/<slug>/eval/<stamp>/report.html > report.html
```

Reports are self-contained single files. Videos stay in the container; they are
too large to move through `exec` comfortably.
