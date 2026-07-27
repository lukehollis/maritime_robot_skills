"""A scripted, privileged-state expert for pick-and-sort scenes.

`mrs.envs.scripted_policy.ScriptedPickPlace` proves one cube into one plate.
This generalises it to N objects into N destinations, which is what a sorting
task needs, and is the demonstration source and solvability proof that
`robo-task-define` (stage 2) is built around.

It reads object poses straight from the simulator, so it is not a baseline for
a visual policy — only for the controller, the reachability of the layout, and
the success predicate. It emits the same 7-D normalised delta actions the
learned policy emits, through the same interface.

Targets are tracked live rather than latched at grasp time, so the expert works
on a moving conveyor: `act()` re-reads the object's position every call and the
approach follows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

OPEN = -1.0
CLOSE = 1.0


@dataclass
class ScriptedSorter:
    """A phase machine that moves each object to its assigned destination."""

    env: object
    assignments: list[tuple[str, str]]
    """`(object_body, destination_body)` pairs, handled in order."""

    hover_height: float = 0.15
    """Height above the object to approach from."""
    grasp_offset: float = 0.004
    """Grip-site height above the object's centre at the moment of closing.

    The Panda's finger *pads* are centred on the grip site, not below it, so
    this should be near zero: lifting the site above the object's centre grips
    it near its top edge and the part pivots out under lateral acceleration."""
    lift_height: float = 0.26
    """Height above the table to carry at, clear of bin walls."""
    place_clearance: float = 0.10
    """Release height above the destination body's origin."""

    position_gain: float = 1.5
    carry_gain: float = 0.7
    """Gain used once something is in the gripper. A carried part is held only
    by friction, so the same aggressive gain that makes the approach quick
    shears it out of the fingers during the lateral move to the bin."""
    xy_tolerance: float = 0.015
    z_tolerance: float = 0.012
    grip_steps: int = 12
    release_steps: int = 8
    settle_steps: int = 6
    phase_timeout: int = 90
    """Advance regardless after this many steps, so no phase can deadlock."""

    require_still: bool = True
    """Hover above a moving object until it settles instead of grabbing at it.

    On a conveyor the part is still travelling when the hand arrives. Closing
    the gripper on a moving target misses, and the expert then carries nothing
    to the bin and counts the object as handled. Waiting is also what a real
    sorting cell does: parts are picked once they queue against the end stop."""
    still_speed: float = 0.02
    """Linear speed below which an object counts as settled (m/s)."""
    wait_timeout: int = 300
    """Give up waiting after this many control steps — longer than a belt duty
    cycle, so a stop-and-go line always gets at least one still window."""

    index: int = field(default=0, init=False)
    phase: str = field(default="approach", init=False)
    done: bool = field(default=False, init=False)
    _timer: int = field(default=0, init=False)
    _phase_steps: int = field(default=0, init=False)
    _grasp_xy: np.ndarray | None = field(default=None, init=False)

    def reset(self) -> None:
        self.index = 0
        self.phase = "approach"
        self.done = False
        self._timer = 0
        self._phase_steps = 0
        self._grasp_xy = None

    # ---- helpers ---------------------------------------------------------
    @property
    def current(self) -> tuple[str, str] | None:
        if self.index >= len(self.assignments):
            return None
        return self.assignments[self.index]

    def _advance(self, phase: str) -> None:
        self.phase = phase
        self._timer = 0
        self._phase_steps = 0

    def _next_object(self) -> None:
        self.index += 1
        self._grasp_xy = None
        if self.index >= len(self.assignments):
            self.done = True
            self._advance("finished")
        else:
            self._advance("approach")

    def _move_towards(self, target: np.ndarray, gripper: float, *, gain: float | None = None) -> np.ndarray:
        current, _ = self.env.controller.site_pose()
        error = np.asarray(target, dtype=float) - current
        gain = self.position_gain if gain is None else gain
        delta = gain * error / self.env.config.position_delta_scale

        action = np.zeros(7)
        action[:3] = np.clip(delta, -1.0, 1.0)
        action[6] = gripper
        return action

    def _reached(self, target, *, xy_only: bool = False) -> bool:
        current, _ = self.env.controller.site_pose()
        target = np.asarray(target, dtype=float)
        if np.linalg.norm(target[:2] - current[:2]) > self.xy_tolerance:
            return False
        return xy_only or abs(target[2] - current[2]) <= self.z_tolerance

    def _is_still(self, body: str) -> bool:
        dof = getattr(self.env, "free_dof_adr", {}).get(body)
        if dof is None:
            return True
        speed = float(np.linalg.norm(self.env.data.qvel[dof : dof + 3]))
        return speed < self.still_speed

    def _destination_release_point(self, name: str) -> np.ndarray:
        position = self.env.body_position(name)
        return np.array([position[0], position[1], position[2] + self.place_clearance])

    # ---- policy ----------------------------------------------------------
    def act(self) -> np.ndarray:
        if self.done or self.current is None:
            return np.zeros(7)

        body, destination = self.current
        table_z = min(self.env.config.workspace_min[2], 0.63)

        obj = self.env.body_position(body)
        above = np.array([obj[0], obj[1], obj[2] + self.hover_height])
        at = np.array([obj[0], obj[1], obj[2] + self.grasp_offset])
        carry = np.array([obj[0], obj[1], table_z + self.lift_height])
        release = self._destination_release_point(destination)
        over_release = np.array([release[0], release[1], table_z + self.lift_height])

        self._phase_steps += 1
        timed_out = self._phase_steps >= self.phase_timeout

        if self.phase == "approach":
            action = self._move_towards(above, OPEN)
            arrived = self._reached(above)
            if self.require_still and arrived and not self._is_still(body):
                # Hold station over a part that is still riding the belt. The
                # hover target tracks it, so the hand drifts along with it.
                if self._phase_steps < self.wait_timeout:
                    return action
            if arrived or timed_out:
                self._advance("descend")
            return action

        if self.phase == "descend":
            action = self._move_towards(at, OPEN)
            if self._reached(at) or timed_out:
                # Latch where the grasp happened so the lift goes straight up
                # instead of chasing an object that is now moving with the hand.
                self._grasp_xy = at[:2].copy()
                self._advance("grip")
            return action

        if self.phase == "grip":
            self._timer += 1
            if self._timer >= self.grip_steps:
                self._advance("lift")
            hold = np.array([self._grasp_xy[0], self._grasp_xy[1], at[2]])
            return self._move_towards(hold, CLOSE, gain=self.carry_gain)

        if self.phase == "lift":
            target = np.array([self._grasp_xy[0], self._grasp_xy[1], carry[2]])
            action = self._move_towards(target, CLOSE, gain=self.carry_gain)
            current, _ = self.env.controller.site_pose()
            if current[2] >= carry[2] - self.z_tolerance or timed_out:
                self._advance("transport")
            return action

        if self.phase == "transport":
            action = self._move_towards(over_release, CLOSE, gain=self.carry_gain)
            if self._reached(over_release, xy_only=True) or timed_out:
                self._advance("lower")
            return action

        if self.phase == "lower":
            action = self._move_towards(release, CLOSE, gain=self.carry_gain)
            if self._reached(release) or timed_out:
                self._advance("release")
            return action

        if self.phase == "release":
            self._timer += 1
            if self._timer >= self.release_steps:
                self._advance("retreat")
            return self._move_towards(release, OPEN)

        if self.phase == "retreat":
            self._timer += 1
            if self._timer >= self.settle_steps or timed_out:
                self._next_object()
            return self._move_towards(over_release, OPEN)

        return np.zeros(7)


def assignments_by_tag(spec, *, object_tag: str, destination_prefix: str) -> list[tuple[str, str]]:
    """Pair every body tagged `object_tag` with a destination sharing its size tag.

    `envelope_large` tagged `["parcel", "size_large"]` pairs with `bin_large`.
    Keeps the sorting rule in the scene's own vocabulary rather than in a
    hard-coded table.
    """
    pairs = []
    for body in spec.bodies_tagged(object_tag):
        size_tags = [t for t in body.tags if t.startswith("size_")]
        if not size_tags:
            continue
        destination = f"{destination_prefix}{size_tags[0][len('size_'):]}"
        if any(b.name == destination for b in spec.bodies):
            pairs.append((body.name, destination))
    return pairs


@dataclass
class ScriptedWelder:
    """A privileged-state expert for tool tasks: visit sites in order, dwell, retract.

    Spot welding is not pick-and-place, so it gets its own phase machine rather
    than a contorted `ScriptedSorter`. Nothing is grasped; the gripper closing
    is the gun trigger, and success is about where the tool tip went.

    The sites are tracked live, which matters here: they are children of a free
    panel riding a conveyor, so their world positions change until the panel
    settles against the end stop. An expert that latched the site positions at
    reset would weld five points into empty air.
    """

    env: object
    sites: list[str]

    approach_height: float = 0.10
    """Height above a site to stage at before descending."""
    tip_offset: float = 0.025
    """Standoff of the *commanded* grip-site height above the spot when the gun
    fires.

    Two constraints bracket this, and they must both be checked against the
    success predicate that grades the task:

      lower bound — the finger pads extend about 9 mm below the grip site, so
        too small a standoff drives them into the part and shoves it off the
        fixture;
      upper bound — it must be comfortably INSIDE the predicate's radius.
        Standing off 40 mm while the predicate credits 35 mm is not a tuning
        problem, it is an unsatisfiable task, and it reads exactly like a
        control failure."""
    dwell_steps: int = 20
    """Control steps to hold the trigger — the weld time.

    Must comfortably exceed the gripper's closing time. The fingers take two
    to three control steps to travel from open to shut, and those steps earn no
    credit because the gun is not yet triggered."""
    position_gain: float = 1.4
    hold_gain: float = 0.5
    """Gain while the gun is firing: enough to hold the standoff, little
    enough not to oscillate about it."""
    xy_tolerance: float = 0.012
    z_tolerance: float = 0.010
    phase_timeout: int = 100
    require_still: bool = True
    still_speed: float = 0.02
    wait_timeout: int = 320
    settle_body: str | None = None
    """Free body whose stillness gates the start of welding — the part."""

    index: int = field(default=0, init=False)
    phase: str = field(default="wait", init=False)
    done: bool = field(default=False, init=False)
    _timer: int = field(default=0, init=False)
    _phase_steps: int = field(default=0, init=False)

    def reset(self) -> None:
        self.index = 0
        self.phase = "wait"
        self.done = False
        self._timer = 0
        self._phase_steps = 0

    @property
    def current(self) -> str | None:
        return self.sites[self.index] if self.index < len(self.sites) else None

    def _advance(self, phase: str) -> None:
        self.phase = phase
        self._timer = 0
        self._phase_steps = 0

    def _move_towards(self, target, trigger: float, *, gain: float | None = None) -> np.ndarray:
        current, _ = self.env.controller.site_pose()
        error = np.asarray(target, dtype=float) - current
        gain = self.position_gain if gain is None else gain
        action = np.zeros(7)
        action[:3] = np.clip(gain * error / self.env.config.position_delta_scale, -1.0, 1.0)
        action[6] = trigger
        return action

    def _reached(self, target, *, xy_only: bool = False) -> bool:
        current, _ = self.env.controller.site_pose()
        target = np.asarray(target, dtype=float)
        if np.linalg.norm(target[:2] - current[:2]) > self.xy_tolerance:
            return False
        return xy_only or abs(target[2] - current[2]) <= self.z_tolerance

    def _part_still(self) -> bool:
        if not self.require_still or self.settle_body is None:
            return True
        dof = getattr(self.env, "free_dof_adr", {}).get(self.settle_body)
        if dof is None:
            return True
        return float(np.linalg.norm(self.env.data.qvel[dof : dof + 3])) < self.still_speed

    def act(self) -> np.ndarray:
        if self.done or self.current is None:
            return np.zeros(7)

        spot = self.env.body_position(self.current)
        stage = np.array([spot[0], spot[1], spot[2] + self.approach_height])
        touch = np.array([spot[0], spot[1], spot[2] + self.tip_offset])

        self._phase_steps += 1
        timed_out = self._phase_steps >= self.phase_timeout

        if self.phase == "wait":
            # Hold clear of the part until the line has indexed it into place.
            # Welding a moving panel puts the spots somewhere else by the time
            # the tip arrives.
            if self._part_still() or self._phase_steps >= self.wait_timeout:
                self._advance("stage")
            return self._move_towards(stage, OPEN)

        if self.phase == "stage":
            action = self._move_towards(stage, OPEN)
            if self._reached(stage, xy_only=True) or timed_out:
                self._advance("descend")
            return action

        if self.phase == "descend":
            action = self._move_towards(touch, OPEN)
            if self._reached(touch) or timed_out:
                self._advance("weld")
            return action

        if self.phase == "weld":
            self._timer += 1
            if self._timer >= self.dwell_steps:
                self._advance("retract")
            # Hold the standoff actively, at a gain low enough not to bob.
            # A zero action does NOT hold position: the environment integrates
            # the *command*, so releasing it leaves the arm wherever the
            # integrator drifted to rather than at the height that was aimed
            # for — 70 mm high, in the case that motivated this comment.
            return self._move_towards(touch, CLOSE, gain=self.hold_gain)

        if self.phase == "retract":
            action = self._move_towards(stage, OPEN)
            if self._reached(stage) or timed_out:
                self.index += 1
                if self.index >= len(self.sites):
                    self.done = True
                    self._advance("finished")
                else:
                    # Back to `wait`, not straight to `stage`: a stop-and-go
                    # line can restart mid-seam and carry the remaining spots
                    # out from under the tool.
                    self._advance("wait")
            return action

        return np.zeros(7)


def sites_by_tag(spec, tag: str) -> list[str]:
    """Site body names in spec order, which is the seam sequence."""
    return [body.name for body in spec.bodies_tagged(tag)]


@dataclass
class ScriptedCutter:
    """One continuous diagonal stroke through a row of severable targets.

    Real tameshigiri against a row of bamboo is a single sweep, not a series of
    chops: the blade descends while travelling along the row, so each stalk is
    severed a little lower than the last and the bundle is left with a diagonal
    profile. That is reproduced here geometrically — the stroke is a straight
    line from above the near end to below the far end, and each stalk's cut
    plane is placed where the blade will be when it arrives.

    The wrist stays offset back along the blade so the arm never passes over
    the stalks it is cutting.
    """

    env: object
    sites: list[str]
    blade: str = "katana_blade"

    lateral: float = 0.14
    """Offset of the wrist back along the blade from the row."""
    rise: float = 0.22
    """Height above the first cut plane to start from."""
    follow: float = 0.30
    """Depth below the last cut plane to finish at."""
    rise_scale: float = 1.0
    """Unused placeholder kept for call compatibility."""

    stage_gain: float = 1.2
    strike_gain: float = 3.0
    xy_tolerance: float = 0.015
    z_tolerance: float = 0.015
    stage_timeout: int = 240
    strike_timeout: int = 90
    clear_timeout: int = 90

    phase: str = field(default="stage", init=False)
    done: bool = field(default=False, init=False)
    _steps: int = field(default=0, init=False)
    _blade_body: str | None = field(default=None, init=False)

    def reset(self) -> None:
        self.phase = "stage"
        self.done = False
        self._steps = 0
        self._blade_body = self.env.build.mounted_bodies.get(self.blade, self.blade)

    def _advance(self, phase: str) -> None:
        self.phase = phase
        self._steps = 0

    def _blade_direction(self) -> np.ndarray:
        """Horizontal unit vector from the wrist out along the blade.

        Asked of the model rather than assumed, so the stroke stays correct
        however the sword is mounted on the flange.
        """
        grip, _ = self.env.controller.site_pose()
        offset = self.env.body_position(self._blade_body) - grip
        offset[2] = 0.0
        norm = float(np.linalg.norm(offset))
        return offset / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    def _edge_line(self):
        """`(point, direction, half_length)` of the blade's cutting edge in world."""
        import mujoco

        model, data = self.env.model, self.env.data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._blade_body)
        geoms = np.flatnonzero(model.geom_bodyid == body) if body >= 0 else []
        if not len(geoms):
            return None
        gid = int(geoms[0])
        rotation = data.geom_xmat[gid].reshape(3, 3)
        half = model.geom_size[gid][:3]
        long_axis = int(np.argmax(half))
        # Of the two faces spanning the blade's width, the lower one is the edge.
        thin_axis = int(np.argmin(half))
        width_axis = 3 - long_axis - thin_axis
        offset = rotation[:, width_axis] * half[width_axis]
        centre = data.geom_xpos[gid]
        edge = centre - offset if offset[2] > 0 else centre + offset
        # The edge normal — spine-to-edge — is the direction the blade must
        # travel to cut. Pointed downward.
        normal = rotation[:, width_axis]
        if normal[2] > 0:
            normal = -normal
        return edge, rotation[:, long_axis], float(half[long_axis]), normal

    def _drop_at(self, target) -> float:
        """Grip-site height above the cutting edge, level with `target`.

        A tilted blade has no single standoff: its edge is a sloped line, so
        the height that matters is the edge's height where it passes over the
        stalk in question. Measuring to the blade's lowest point instead aims
        the whole stroke at the tip.
        """
        line = self._edge_line()
        grip, _ = self.env.controller.site_pose()
        if line is None:
            return 0.0
        edge, direction, half, _ = line
        offset = np.asarray(target, dtype=float) - edge
        # Nearest point along the edge, matching horizontally.
        flat = direction.copy()
        flat[2] = 0.0
        norm = float(np.linalg.norm(flat))
        t = float(np.clip(np.dot(offset[:2], flat[:2]) / max(norm ** 2, 1e-9), -half, half))
        return float(grip[2] - (edge + t * direction)[2])

    def _targets(self):
        points = [self.env.body_position(name) for name in self.sites]
        along = self._blade_direction()
        line = self._edge_line()
        stroke_dir = line[3] if line is not None else np.array([0.0, 0.0, -1.0])
        # Aim from the end of the row the stroke reaches FIRST, measured along
        # the stroke itself. Ordering by distance from the wrist is wrong when
        # the blade points forward and the row runs across it — every site is
        # then equidistant and the choice is arbitrary.
        first = min(points, key=lambda p: float(np.dot(p, stroke_dir)))
        line = self._edge_line()
        # Travel along the edge normal, not straight down. A tilted blade driven
        # vertically is being shoved along its own length — it wedges through
        # sideways instead of slicing, and pushes the severed tops back onto
        # their stumps. Cutting perpendicular to the edge is both what a sword
        # does and what carries the tops clear.
        stroke = stroke_dir

        drop = self._drop_at(first)
        base = first - self.lateral * along
        base[2] = first[2] + drop

        return base - self.rise * stroke, base + self.follow * stroke

    def _move(self, target, gain) -> np.ndarray:
        current, _ = self.env.controller.site_pose()
        error = np.asarray(target, dtype=float) - current
        action = np.zeros(7)
        action[:3] = np.clip(gain * error / self.env.config.position_delta_scale, -1.0, 1.0)
        action[6] = 1.0
        return action

    def _reached(self, target) -> bool:
        current, _ = self.env.controller.site_pose()
        target = np.asarray(target, dtype=float)
        return (np.linalg.norm(target[:2] - current[:2]) <= self.xy_tolerance
                and abs(target[2] - current[2]) <= self.z_tolerance)

    def act(self) -> np.ndarray:
        if self.done:
            return np.zeros(7)

        start, end = self._targets()
        self._steps += 1

        if self.phase == "stage":
            action = self._move(start, self.stage_gain)
            if self._reached(start) or self._steps >= self.stage_timeout:
                self._advance("strike")
            return action

        if self.phase == "strike":
            action = self._move(end, self.strike_gain)
            current, _ = self.env.controller.site_pose()
            if current[2] <= end[2] + self.z_tolerance or self._steps >= self.strike_timeout:
                self._advance("clear")
            return action

        if self.phase == "clear":
            action = self._move(start, self.stage_gain)
            if self._reached(start) or self._steps >= self.clear_timeout:
                self.done = True
                self._advance("finished")
            return action

        return np.zeros(7)


