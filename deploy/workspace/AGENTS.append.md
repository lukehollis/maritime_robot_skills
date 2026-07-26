
<!-- mrs-lab -->

## This machine is a robot policy evaluation lab

You take a sentence like *"evaluate pi0.5 with a Franka Panda on a mail sorting
task with three envelope sizes, and report the failure modes"* and turn it into a
measured result with a written report.

Read `PIPELINE.md` in this workspace before starting the first experiment of a
conversation. It describes the four pipeline stages, the container you are
running inside, and the constraints that make this environment different from a
workstation. Do not plan an experiment without it.

The short version:

- Work in `/data/lab`. One directory per environment under `envs/<slug>/`.
- Blender is already running with the MCP bridge connected. Use the `blender`
  MCP tools. Never try to start Blender yourself.
- Skills `robo-env-create`, `robo-policy-deploy`, `robo-task-define`,
  `robo-eval` and `robo-report` are the four stages plus the write-up. Run them
  in that order.
- Anything that takes more than a minute — checkpoint downloads, rollouts —
  goes through `mrs-job`, not a blocking shell call.
- Every experiment ends with a Quarto report. That is the deliverable, not the
  console output.

Tell the user what you are doing as you go, and when a stage fails, say what
failed and what you tried rather than quietly working around it.
