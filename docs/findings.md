# Findings

Measurements taken while bringing this stack up, on an Apple M4 Max (68 GB, MPS).

## 1. The pi0.5 implementation matches the published checkpoint

`lerobot/pi05_libero_finetuned_v044` holds 812 tensors. The module tree here
instantiates 811 of them by exact name and shape, with zero missing and zero
unexpected. The single unused tensor is
`paligemma_with_expert.gemma_expert.lm_head.weight` — the action expert's
vocabulary head, which is never touched during control, and which this
implementation deliberately does not allocate (it would cost 0.5 GB).

Locked in by `tests/test_policy.py::test_checkpoint_loads_with_every_weight_accounted_for`.

## 2. Runtime dtypes come from the code, not from the file

The checkpoint stores the SigLIP encoder and the multimodal projector in
bfloat16, but the reference implementation casts them to float32 *at
construction*, and `load_state_dict` then casts the incoming tensors into the
parameters' dtypes. So the file's dtypes are not the runtime dtypes.

Reproducing the file's layout instead of the reference's layout is not merely a
precision difference — it leaves the vision tower internally mixed (float32
patch embedding feeding bfloat16 encoder layers), which MPS rejects outright:

```
'mps.add' op requires the same element type for all operands and results
  (tensor<1x256x1152xf32>, tensor<1152xbf16>) -> tensor<*xf32>
```

`PI05Model.apply_precision_policy` implements the reference layout: bfloat16 for
the matmul-heavy weights, float32 for the vision path, every RMSNorm, and the
action/time projections. Resulting mix: 558 float32 tensors, 253 bfloat16.

## 3. Performance

| stage | cost |
|---|---|
| checkpoint load (3.35 B params) | 11 s |
| first action chunk (50 steps, 10 denoising steps) | 3.4 s |
| subsequent chunks | 0.8 s |
| full 300-step episode, replanning every 10 steps | 28 s |

## 4. The task is solvable through the policy's action interface

The scripted expert reaches, grasps, transports and places using the same 7-D
normalized delta actions the policy emits, succeeding **12/12** across seeds.
This matters because a policy failure is only informative if the interface it is
being asked to work through is known to be sufficient.

Two controller bugs had to be fixed to get there, both worth recording:

- **Integral windup.** Iterating the IK against the *measured* pose while the
  position actuators lag means the error never clears, so the joint command
  accumulates far past the requested pose and the arm flails. Fixed by solving
  the IK on a scratch `MjData` driven by the candidate joint configuration, i.e.
  purely kinematically.
- **Orientation drift.** Deriving the pose target from the measured pose every
  step (`target = measured + delta`) lets tracking error random-walk the
  orientation. Over ~150 steps the gripper rotated roughly 90°, and the IK then
  fell into a contorted branch 4.4 cm short of the requested position, which no
  amount of extra iterations or nullspace tuning recovered. Fixed by integrating
  an explicit commanded pose, with a leash bounding how far it may lead the
  measured pose.

## 5. pi0.5 does not transfer zero-shot to this environment

This is the headline negative result. Running `lerobot/pi05_libero_finetuned_v044`
on the task: **0/3 episodes**, with the arm driving out of the workspace and
parking against the leash boundary.

To separate "the plumbing is broken" from "the policy does not recognise this
scene", the cube was teleported to nine positions with everything else held
fixed, and the mean of the first 8 commanded actions was measured (4 noise draws
each). If vision were driving control, the commanded `dy` would track the
displacement needed to reach the cube.

| camera images | corr(action_dy, needed_dy) | corr(action_dx, needed_dx) |
|---|---|---|
| as rendered | +0.04 | +0.14 |
| rotated 180° | **+0.63** | +0.07 |

Two things follow.

**The 180° rotation is the right convention for LIBERO checkpoints.** openpi's
LIBERO adapter feeds `image[::-1, ::-1]`, and that is the orientation under which
this checkpoint responds to the scene at all. Exposed as
`--rotate-images-180` / `env_observation_to_batch(..., rotate_images_180=True)`,
defaulting to off because this environment renders right-side-up and a user
training their own policy wants the natural orientation.

**Even with the correct convention, the policy is not usefully controlled by
this scene.** There is partial `y` tracking and no `x` tracking, `dz` is
positive (moving away from the table) almost everywhere, and the gripper command
sits near +0.87 (closed) regardless of whether anything is within reach.

This is the expected outcome, and it is not a defect in the port. The checkpoint
is a narrow fine-tune on 40 specific LIBERO tasks with LIBERO's meshes, textures,
lighting, camera intrinsics and object vocabulary — not a generalist. A
hand-built scene with different assets is out of distribution in every visual
respect.

**What this means for using the stack:** the inference path is verified and the
environment is verified; closing the gap is a fine-tuning problem, not a
debugging one. `PI05Policy.forward` implements the flow-matching training loss,
and `ScriptedPickPlace` produces in-domain demonstrations, so the pieces for
that are in place. Fine-tuning `lerobot/pi05_base` (or the LIBERO checkpoint) on
demonstrations recorded here is the next step; it is not implemented.
