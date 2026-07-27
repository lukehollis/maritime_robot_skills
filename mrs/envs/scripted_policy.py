"""A scripted pick-and-place expert.

Serves two purposes: it proves the task is solvable through the same action
interface the learned policy uses, and it generates demonstrations for
fine-tuning. It reads privileged state (the cube pose) straight from the
simulator, so it is not a baseline for the visual policy — only for the
controller and the task definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mrs.envs.panda_pick_place import PandaPickPlaceEnv

OPEN = -1.0
CLOSE = 1.0


@dataclass
class ScriptedPickPlace:
    """A phase machine emitting the env's 7-D normalized delta actions."""

    env: PandaPickPlaceEnv

    hover_height: float = 0.12
    """Height above the cube's centre to approach from."""
    grasp_clearance: float = 0.005
    """Vertical offset of the grasp point above the cube centre."""
    place_clearance: float = 0.045
    position_gain: float = 1.4
    """Proportional gain on the position error, in units of `position_delta_scale`."""
    xy_tolerance: float = 0.012
    z_tolerance: float = 0.012
    """Loose enough to absorb the steady-state droop of the position actuators
    when the arm is carrying the cube."""
    grip_steps: int = 8
    release_steps: int = 6
    phase_timeout: int = 60
    """Advance regardless after this many steps, so a phase cannot deadlock."""

    phase: str = field(default="approach", init=False)
    _timer: int = field(default=0, init=False)
    _phase_steps: int = field(default=0, init=False)

    def reset(self) -> None:
        self.phase = "approach"
        self._timer = 0
        self._phase_steps = 0

    def _advance(self, next_phase: str) -> None:
        self.phase = next_phase
        self._timer = 0
        self._phase_steps = 0

    # ---- helpers ---------------------------------------------------------
    def _move_towards(self, target: np.ndarray, gripper: float) -> np.ndarray:
        """Proportional Cartesian action toward `target`, with no rotation."""
        current, _ = self.env.controller.site_pose()
        error = target - current
        delta = self.position_gain * error / self.env.config.position_delta_scale

        action = np.zeros(7)
        action[:3] = np.clip(delta, -1.0, 1.0)
        action[6] = gripper
        return action

    def _reached(self, target: np.ndarray, *, xy_only: bool = False) -> bool:
        current, _ = self.env.controller.site_pose()
        if np.linalg.norm(target[:2] - current[:2]) > self.xy_tolerance:
            return False
        return xy_only or abs(target[2] - current[2]) <= self.z_tolerance

    # ---- policy ----------------------------------------------------------
    def act(self) -> np.ndarray:
        config = self.env.config
        cube = self.env.cube_position
        target_xy = np.asarray(config.target_pos)

        self._phase_steps += 1
        timed_out = self._phase_steps >= self.phase_timeout

        above_cube = np.array([cube[0], cube[1], cube[2] + self.hover_height])
        at_cube = np.array([cube[0], cube[1], cube[2] + self.grasp_clearance])
        above_target = np.array(
            [target_xy[0], target_xy[1], config.table_height + self.hover_height + 0.05]
        )
        at_target = np.array(
            [target_xy[0], target_xy[1], config.table_height + self.place_clearance + 0.02]
        )

        if self.phase == "approach":
            action = self._move_towards(above_cube, OPEN)
            if self._reached(above_cube) or timed_out:
                self._advance("descend")
            return action

        if self.phase == "descend":
            action = self._move_towards(at_cube, OPEN)
            if self._reached(at_cube) or timed_out:
                self._advance("grip")
            return action

        if self.phase == "grip":
            self._timer += 1
            if self._timer >= self.grip_steps:
                self._advance("lift")
            return self._move_towards(at_cube, CLOSE)

        if self.phase == "lift":
            lift_target = np.array([cube[0], cube[1], config.table_height + 0.22])
            action = self._move_towards(lift_target, CLOSE)
            current, _ = self.env.controller.site_pose()
            if current[2] >= lift_target[2] - self.z_tolerance or timed_out:
                self._advance("transport")
            return action

        if self.phase == "transport":
            action = self._move_towards(above_target, CLOSE)
            if self._reached(above_target, xy_only=True) or timed_out:
                self._advance("lower")
            return action

        if self.phase == "lower":
            action = self._move_towards(at_target, CLOSE)
            if self._reached(at_target) or timed_out:
                self._advance("release")
            return action

        if self.phase == "release":
            self._timer += 1
            if self._timer >= self.release_steps:
                self._advance("retreat")
            return self._move_towards(at_target, OPEN)

        # retreat
        return self._move_towards(above_target, OPEN)
