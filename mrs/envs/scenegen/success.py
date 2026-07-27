"""Declarative success and failure predicates.

Deliberately a small vocabulary. `robo-env-create` only needs enough to prove
that the environment it built is solvable and that its failure modes register;
growing this into a real task specification is `robo-task-define`'s job.

A term is a dict naming a predicate plus its arguments:

    {"predicate": "near", "body": "envelope_0", "target": "bin_small", "radius": 0.09}

Terms are combined by `SuccessSpec.mode` (`all` or `any`). Every predicate
returns a plain bool and reads only from the live `mujoco` state, so the same
expression works for termination, for per-step logging, and for the failure
taxonomy the evaluation skill reports.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class Context:
    """The view of the simulation that predicates are allowed to see."""

    def __init__(self, env):
        import mujoco

        self._mujoco = mujoco
        self.env = env
        self.model = env.model
        self.data = env.data

    # ---- lookups ---------------------------------------------------------
    def body_id(self, name: str) -> int:
        bid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"No body named {name!r} in the compiled model.")
        return bid

    def position(self, name: str) -> np.ndarray:
        return self.data.xpos[self.body_id(name)].copy()

    def quat(self, name: str) -> np.ndarray:
        return self.data.xquat[self.body_id(name)].copy()

    def linear_speed(self, name: str) -> float:
        """Magnitude of a free body's linear velocity; 0 for anything welded."""
        dof = self.env.free_dof_adr.get(name)
        if dof is None:
            return 0.0
        return float(np.linalg.norm(self.data.qvel[dof : dof + 3]))

    def half_extent(self, name: str) -> np.ndarray:
        """Half-size of the body's first geom, in its own frame."""
        bid = self.body_id(name)
        geoms = np.flatnonzero(self.model.geom_bodyid == bid)
        if not len(geoms):
            raise KeyError(f"Body {name!r} has no geoms.")
        return self.model.geom_size[geoms[0]].copy()

    def is_grasped(self, name: str) -> bool:
        return self.env.is_grasped(name)

    def eef_position(self) -> np.ndarray:
        """World position of the grasp site — the tool tip for a tool task."""
        return self.env.controller.site_pose()[0]

    def tool_triggered(self, threshold: float = 0.015) -> bool:
        """True while the gripper is commanded shut.

        For a tool task the gripper is not grasping anything; closing it is the
        trigger. Read from the finger joints rather than the last action so the
        test reflects what the hardware actually did.
        """
        fingers = self.data.qpos[self.env.finger_qpos_adr]
        return bool(np.mean(np.abs(fingers)) < threshold)

    @property
    def task_state(self) -> dict:
        """Per-episode scratch space for predicates that accumulate progress."""
        return self.env.task_state

    def touching(self, name: str, other: str) -> bool:
        a = set(np.flatnonzero(self.model.geom_bodyid == self.body_id(name)).tolist())
        b = set(np.flatnonzero(self.model.geom_bodyid == self.body_id(other)).tolist())
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if pair & a and pair & b:
                return True
        return False


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def near(ctx: Context, *, body: str, target: str | list[float], radius: float, planar: bool = True) -> bool:
    """Body centre within `radius` of a point or another body's centre."""
    position = ctx.position(body)
    point = np.asarray(ctx.position(target) if isinstance(target, str) else target, dtype=float)
    delta = position[:2] - point[:2] if planar else position - point[: len(position)]
    return float(np.linalg.norm(delta)) <= radius


def in_region(ctx: Context, *, body: str, min: list[float], max: list[float]) -> bool:
    position = ctx.position(body)
    return bool(np.all(position >= np.asarray(min)) and np.all(position <= np.asarray(max)))


def inside(
    ctx: Context,
    *,
    body: str,
    container: str,
    pad: float = 0.0,
    height: float = 0.15,
    check_height: bool = True,
) -> bool:
    """Body centre within a container body's horizontal footprint.

    Written against the container's compiled geom size rather than spec values,
    so it stays correct if the container is rescaled after the fact.

    `height` is how far above the container's top surface still counts as
    inside, and it has to be given rather than inferred: a bin is a thin floor
    slab plus four walls, so the floor's own half-extent says nothing about how
    deep the bin is. Deriving the band from it rejects a parcel resting
    normally on the floor of its own bin.
    """
    centre = ctx.position(container)
    half = ctx.half_extent(container)
    position = ctx.position(body)

    if not np.all(np.abs(position[:2] - centre[:2]) <= half[:2] + pad):
        return False
    if not check_height:
        return True
    return bool(centre[2] - half[2] - pad <= position[2] <= centre[2] + half[2] + height)


def resting_at_height(ctx: Context, *, body: str, height: float, tolerance: float = 0.02) -> bool:
    return abs(float(ctx.position(body)[2]) - height) <= tolerance


def at_rest(ctx: Context, *, body: str, speed: float = 0.05) -> bool:
    return ctx.linear_speed(body) < speed


def released(ctx: Context, *, body: str) -> bool:
    return not ctx.is_grasped(body)


def grasped(ctx: Context, *, body: str) -> bool:
    return ctx.is_grasped(body)


def below_height(ctx: Context, *, body: str, height: float) -> bool:
    """Typical dropped-on-the-floor failure test."""
    return float(ctx.position(body)[2]) < height


def tipped(ctx: Context, *, body: str, max_tilt_deg: float = 45.0) -> bool:
    """The body's local +z has fallen more than `max_tilt_deg` from world +z."""
    w, x, y, z = ctx.quat(body)
    # Third column of the rotation matrix: where the body's local z now points.
    local_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    cosine = float(np.clip(local_z[2], -1.0, 1.0))
    return np.degrees(np.arccos(cosine)) > max_tilt_deg


def touching(ctx: Context, *, body: str, other: str) -> bool:
    return ctx.touching(body, other)


def sites_serviced(
    ctx: Context,
    *,
    tag: str,
    radius: float = 0.03,
    dwell: int = 3,
    require_trigger: bool = True,
    ordered: bool = False,
    key: str = "serviced",
) -> bool:
    """Every body tagged `tag` has been visited by the tool tip and held there.

    The predicate a tool task needs — welding, dispensing, inspection — where
    nothing is picked up and success is about where the *end-effector* went
    rather than where objects ended up.

    Unlike every other predicate here this one **accumulates state**, in
    `ctx.task_state`, because "has been welded" is a fact about the episode's
    history and cannot be recovered from the current frame. The store is reset
    with the episode. It is called once per control step from `env.step`, so
    `dwell` counts control steps in contact.

    With `ordered`, a site only counts once every earlier site is done, which
    is what a seam sequence means; without it any order is accepted.
    """
    sites = [b.name for b in ctx.env.spec.bodies_tagged(tag)]
    if not sites:
        return False

    counters = ctx.task_state.setdefault(key, {name: 0 for name in sites})
    tip = ctx.eef_position()
    triggered = ctx.tool_triggered() if require_trigger else True

    for index, name in enumerate(sites):
        if counters.get(name, 0) >= dwell:
            continue
        if ordered and any(counters.get(prior, 0) < dwell for prior in sites[:index]):
            continue
        if triggered and float(np.linalg.norm(tip - ctx.position(name))) <= radius:
            counters[name] = counters.get(name, 0) + 1
            break  # one site advances per step; no credit for straddling two

    return all(counters.get(name, 0) >= dwell for name in sites)


def constraint_released(ctx: Context, *, equality: str) -> bool:
    """A named equality constraint is no longer active.

    Ground truth for a cut. Whether the severed piece then topples or balances
    on the stump is a matter of momentum and luck; whether the constraint
    holding it was released is the task.
    """
    import mujoco

    index = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality)
    if index < 0:
        raise KeyError(f"No equality named {equality!r} in the compiled model.")
    return not bool(ctx.data.eq_active[index])


def returned_to(
    ctx: Context,
    *,
    body: str,
    target: str | list[float],
    radius: float,
    exit_radius: float | None = None,
    key: str = "round_trip",
) -> bool:
    """The body has left `target`, and has now come back to it.

    A task phrased as "take it out and put it back" cannot be graded by
    position alone: the object starts where it must finish, so a plain
    containment test is already true at reset and the episode terminates on
    step one having done nothing. This records the excursion, so success means
    the round trip actually happened.

    Stateful, like `sites_serviced`, and for the same reason — "has been out"
    is a fact about the episode's history, not about the current frame. The
    store is cleared with the episode.

    `exit_radius` is how far out counts as having left; it defaults to twice
    `radius` so that jostling inside the container cannot be mistaken for an
    excursion.
    """
    exit_radius = radius * 2.0 if exit_radius is None else exit_radius
    if exit_radius <= radius:
        raise ValueError(
            f"returned_to needs exit_radius ({exit_radius}) greater than radius ({radius}); "
            f"otherwise leaving and arriving are the same test."
        )

    centre = np.asarray(ctx.position(target) if isinstance(target, str) else target, dtype=float)
    distance = float(np.linalg.norm(ctx.position(body)[:2] - centre[:2]))

    state = ctx.task_state.setdefault(key, {})
    if distance > exit_radius:
        state[body] = True
    return bool(state.get(body)) and distance <= radius


def all_of(ctx: Context, *, terms: list[dict[str, Any]]) -> bool:
    return all(evaluate_term(ctx, term) for term in terms)


def any_of(ctx: Context, *, terms: list[dict[str, Any]]) -> bool:
    return any(evaluate_term(ctx, term) for term in terms)


def each_tagged(ctx: Context, *, tag: str, term: dict[str, Any], spec=None) -> bool:
    """Apply one term to every body carrying `tag`, substituting `body`.

    Lets a three-envelope sorting task be written once rather than three times.
    """
    spec = spec if spec is not None else ctx.env.spec
    bodies = [b.name for b in spec.bodies_tagged(tag)]
    if not bodies:
        return False
    return all(evaluate_term(ctx, {**term, "body": name}) for name in bodies)


PREDICATES: dict[str, Callable[..., bool]] = {
    "near": near,
    "in_region": in_region,
    "inside": inside,
    "resting_at_height": resting_at_height,
    "at_rest": at_rest,
    "released": released,
    "grasped": grasped,
    "below_height": below_height,
    "tipped": tipped,
    "touching": touching,
    "all_of": all_of,
    "any_of": any_of,
    "each_tagged": each_tagged,
    "sites_serviced": sites_serviced,
    "constraint_released": constraint_released,
    "returned_to": returned_to,
}


def evaluate_term(ctx: Context, term: dict[str, Any]) -> bool:
    kwargs = dict(term)
    name = kwargs.pop("predicate", None)
    if name is None:
        raise ValueError(f"Success term is missing a 'predicate' key: {term!r}")
    if name not in PREDICATES:
        raise ValueError(f"Unknown predicate {name!r}. Available: {sorted(PREDICATES)}")
    # Authoring affordances, not predicate arguments: `name` labels a failure
    # mode in `info["failure_modes"]`, `comment` documents intent.
    kwargs.pop("name", None)
    kwargs.pop("comment", None)
    return bool(PREDICATES[name](ctx, **kwargs))


def evaluate(ctx: Context, spec) -> bool:
    """Evaluate a `SuccessSpec`'s success expression."""
    if not spec.terms:
        return False
    results = (evaluate_term(ctx, term) for term in spec.terms)
    return all(results) if spec.mode == "all" else any(results)


def failures(ctx: Context, spec) -> list[str]:
    """Names of every failure term currently true."""
    triggered = []
    for index, term in enumerate(spec.failure_terms):
        if evaluate_term(ctx, term):
            triggered.append(term.get("name", f"{term.get('predicate', 'term')}_{index}"))
    return triggered
