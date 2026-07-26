---
name: robo-report
description: "Stage 4 of policy evaluation. Turn a finished eval run into a self-contained Quarto report — success rate against the scripted-expert ceiling, failure-mode breakdown, per-episode outcomes, scene renders — and read the result back to the user in prose. Use after robo-eval has produced an eval directory, and whenever the user asks for a write-up, summary or comparison of experiments."
argument-hint: "[env-slug or eval directory] [--title ...]"
metadata: { "openclaw": { "emoji": "📊", "requires": { "bins": ["quarto", "mrs-report"] } } }
---

Render the evaluation in `$0` into a report, then tell the user what it says.

The report is the deliverable. A run that produced numbers nobody read is not a
finished experiment.

## Render it

```bash
mrs-report envs/<slug>/eval/<timestamp>
```

That writes `report.html` into the run directory: one self-contained file, no
sidecar assets, safe to copy anywhere. Point it at a different environment
package with `--env-dir` when the run does not sit under `envs/<slug>/eval/`.

If the user did not name a run, use the most recent:

```bash
ls -dt envs/*/eval/*/ | head -5
```

## What it reads

Everything is optional except `summary.json`; the template degrades rather than
failing, so a missing piece shows up as a thin section instead of an error.

| file | supplies |
|---|---|
| `<run>/summary.json` | the headline rates and per-episode rows |
| `<run>/expert.json` or `<run>/../scripted/summary.json` | the scripted-expert ceiling |
| `<run>/*.mp4` | rollout videos, linked (never embedded — they would dwarf the file) |
| `envs/<slug>/spec.json` | the success predicate |
| `envs/<slug>/brief.md` | the task as it was asked for |
| `envs/<slug>/validation.json` | stage 0's gate report |
| `envs/<slug>/cameras.png`, `renders/*.png` | what the scene and its cameras look like |

Per-episode failure attribution comes from a `failure_modes` list on each entry
in `summary.json`'s `episodes` array — the same strings `SceneEnv` puts in
`info["failure_modes"]`. The reference driver `mrs/scripts/eval.py` does not
record them yet. If they are absent the report says so plainly, and the honest
move is to say so to the user too rather than describing failures you did not
measure.

## Then read it back

Do not hand over a link. Say, in three or four sentences:

- the success rate, and the expert ceiling beside it;
- whether that ceiling makes the number trustworthy — **below ~90% expert
  success, the learned number means nothing and the environment is the bug**;
- what dominated the failures, named by predicate;
- the one thing you would change next.

## Comparing runs

Several runs of one environment are compared by rendering each and reading the
headline numbers side by side; there is no cross-run template. When the user
wants a sweep, write the comparison yourself in the reply and link the
individual reports. Keep the runs — the timestamped directories are the record,
and deleting them to save space loses the only copy of the evidence.

## Getting it out of the container

The report lives on `/data`, which the user cannot browse:

```bash
maritime exec mrs-lab -- cat /data/lab/envs/<slug>/eval/<stamp>/report.html > report.html
```

Offer that command when the user wants the file itself. Videos are too large to
pull through `exec` comfortably — describe them instead, or extract a
representative frame and describe that.
