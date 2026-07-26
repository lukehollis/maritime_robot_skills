# The visual inspection loop

Three passes, minimum, before export. Each pass is **render → look → measure →
edit**, in that order. The order matters: measurements only answer questions
you already thought to ask, and looking is what generates the questions.

## Why more than one angle

A single hero render hides the two errors that matter most in a robot scene:

- **A part floating above the surface it should rest on.** Invisible from any
  viewpoint looking along the gap. It becomes a part that drops and bounces
  the instant the episode starts.
- **Two objects intersecting.** Invisible whenever one hides the other. It
  becomes a divergent first physics step.

`render_views()` writes six angles — `front`, `three_q`, `top`, `left`,
`right`, `robot_eye`. Read all of them. `top` catches layout and spacing;
`front` and `left`/`right` catch height errors; `robot_eye` approximates what
the policy camera will see.

## Pass 1 — is it the right scene?

Structural. Ignore millimetres.

- Is every object the brief names present, and nothing it does not?
- Are the relative sizes credible? An envelope the size of a bin is the classic
  unit error — 90 mm typed as 0.90.
- Is the robot where a person would put it, facing the work?
- Is the layout reachable? Everything the arm touches must sit inside
  x ∈ [−0.32, 0.20], y ∈ [−0.36, 0.36] around a base at (−0.56, 0, 0.63).

```python
render_views('/tmp/<slug>/pass1', resolution=420)
print(summary())
```

Typical pass-1 edits: move the whole bin bank closer, rotate the belt, resize
parts to real dimensions.

## Pass 2 — is it physically sound?

Now millimetres.

```python
check_overlaps()                    # bodies sharing space
check_resting(surface_z=0.685)      # parts floating or sunk
measure(['envelope_small', 'bin_small'])
```

`check_overlaps` over-reports by design — a bin legitimately overlaps its own
walls, and two touching boxes share a face. Treat every hit as something to
look at, not as an error. What you are hunting is the hit you cannot explain.

Free bodies with **overlapping spawn ranges** are the subtle case: the authored
poses may be fine while the sampled ones collide. Space lane centres by more
than the sum of adjacent half-widths and keep jitter inside the clearance.

Check clearance for the gripper — roughly 15 cm above a part. Bins with tall
walls are stylish and unpickable.

## Pass 3 — can the policy see it?

The policy's whole world is two 256 × 256 images.

```python
render_views('/tmp/<slug>/pass3', views={'robot_eye': REVIEW_VIEWS['robot_eye']},
             resolution=256)
```

Render at the resolution the policy actually gets. Detail that survives at 640
px and vanishes at 256 does not exist as far as the policy is concerned.

- Are the objects that must be distinguished still distinguishable? Three
  envelope sizes that differ by 20 % are three identical light rectangles at
  256 px. Differentiate by colour as well as size.
- Does anything occlude the work — the arm parked in front of the bins, a rail
  across the pick point?
- Is the contrast adequate? A pale part on a pale table is a policy that cannot
  see its target.

Fix by moving the camera, recolouring, or respacing — in that order of
preference. Moving the camera is free; recolouring changes the task slightly;
respacing changes it a lot.

## After migration

`validate_env.py --sheet` writes `envs/<slug>/cameras.png`, rendered from the
**compiled MuJoCo model** through the actual policy cameras. Blender renders
tell you what you built; this tells you what the policy gets. They can differ —
materials do not survive migration, and MuJoCo's lighting is its own.

Look at it. It is the last chance to catch a scene that is beautiful in Blender
and unreadable at `agentview`.

## When to stop

Stop when a full pass produces no edits. Three passes is a floor, not a target:
a scene with a conveyor, articulated fixtures and six part types will take
more, and stopping at three because the instructions said three is how a broken
scene reaches evaluation.

Conversely, do not loop forever polishing appearance. The question is not "is
this beautiful" but "will this grade a policy fairly" — and past a point,
further prettiness buys nothing.
