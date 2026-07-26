---
name: robo-task-define
description: Stage 2 of policy evaluation. Turn a human task description into machine-checkable success and failure predicates on a generated environment, and prove the task is solvable by driving it with a scripted expert. Use after robo-env-create has produced and validated an environment, before robo-eval measures a learned policy against it.
argument-hint: [env-slug] [success description and failure modes to track]
allowed-tools: Read, Write, Edit, Glob, Bash
---

> **STUB.** The predicate vocabulary and the solvability requirement are built
> and working; the authoring automation and the scripted-expert generator are
> not. Follow the manual procedure and report exactly what you ran.

Define success and failure for `envs/$0`.

## The one rule

**An environment whose task no expert can solve is not an evaluation, it is a
bug.** Before any learned policy touches this scene, a scripted policy with
privileged state must complete it repeatedly. If the expert cannot, the number
you would have reported for the learned policy means nothing.

`mrs/envs/scenegen/scripted.py` already does this for pick-and-sort scenes.
`ScriptedSorter` takes `(object, destination)` pairs — `assignments_by_tag`
derives them from the scene's own `size_*` tags — and runs a phase machine
(approach → descend → grip → lift → transport → lower → release) emitting the
same 7-D normalized delta actions the learned policy emits. It scores 10/10 on
`envs/parcel_sorting`. `mrs/envs/scripted_policy.py` is the single-object
original.

Three settings decide whether it works at all, and each was a real failure:

- **`grasp_offset` near zero.** The Panda's finger *pads* are centred on the
  grip site, not below it. Gripping above an object's centre catches its top
  edge and it pivots out during transport.
- **`carry_gain` below `position_gain`.** A carried part is held by friction
  alone; the gain that makes the approach quick shears it loose on the lateral
  move to the bin.
- **`require_still`.** On a conveyor the hand arrives before the part stops.
  Closing on a moving target misses, and the expert then carries nothing to the
  bin while counting the object as handled.

If the expert cannot solve the scene, suspect the parts before the policy. A
parallel jaw cannot pick a flat item off a surface at all.

## Writing predicates

`spec.json` ships from stage 0 with an empty `success` block. Fill it in;
`mrs/envs/scenegen/success.py` holds the vocabulary.

```json
{
  "mode": "all",
  "terms": [
    {"predicate": "inside", "body": "envelope_large", "container": "bin_large", "pad": 0.01},
    {"predicate": "each_tagged", "tag": "envelope",
     "term": {"predicate": "at_rest", "speed": 0.05}}
  ],
  "failure_terms": [
    {"name": "dropped_large", "predicate": "below_height",
     "body": "envelope_large", "height": 0.53}
  ]
}
```

Available: `near`, `in_region`, `inside`, `resting_at_height`, `at_rest`,
`released`, `grasped`, `below_height`, `tipped`, `touching`, `all_of`, `any_of`,
`each_tagged`. Add new ones to `PREDICATES` rather than encoding task logic
elsewhere.

Three properties a good success predicate has:

- **It requires release.** A part still pinched between the fingertips has not
  been placed. Pair the placement test with `released` or `at_rest`.
- **It requires rest.** A part tumbling through the target region satisfies a
  naive position test on one frame. `success_hold_steps` (default 5) guards
  this too, but stating it is clearer.
- **It cannot be satisfied at t=0.** Check the predicate on the reset state; if
  it is already true, the task is trivially complete and the evaluation is
  meaningless.

## Failure modes

`failure_terms` are what make an evaluation diagnostic rather than a single
number. Each triggered term is recorded in `info["failure_modes"]` and ends the
episode. The taxonomy worth capturing:

| mode | predicate |
|---|---|
| dropped the part | `below_height` |
| put it in the wrong container | `inside` on a non-matching bin |
| knocked a fixture over | `tipped` |
| never grasped anything | absence of `grasped` across the episode |
| grasped and never released | `grasped` at timeout |
| ran out of time | truncation, recorded automatically |

"Failed" is not a result. "Failed by dropping the large envelope in 7 of 10
episodes, and by never grasping the small one in the other 3" is.

## What this skill must add

1. Parse a human description into terms, asking when the mapping is genuinely
   ambiguous (which bin is "the right one"?) rather than guessing.
2. Generate a task-appropriate scripted expert, or adapt `ScriptedPickPlace`.
3. Run it for N episodes and **refuse to hand off below a threshold** —
   roughly 8/10. Report the rate.
4. Verify no success term is true at reset.
5. Write the terms into `envs/$0/spec.json` and record the expert's rate in
   `envs/$0/task.json` as the ceiling `robo-eval` compares against.

## Hand-off

`robo-eval` (stage 3) measures the learned policy and compares it against the
expert's rate. Without that baseline a learned-policy number has no scale.
