"""Differential IK solver: stack tasks and constraints into a QP, solve, integrate"""

from collections.abc import Sequence
from typing import NamedTuple

import elastiqp.jax
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from frax.core.robot import Robot
from frax.ik.constraints import (
    DEFAULT_PENALTY,
    EqualityConstraint,
    InequalityConstraint,
)
from frax.ik.tasks import Task


class SolverState(NamedTuple):
    q: Array
    qp_xyz: tuple[Array, Array, Array]  # Previous QP solution, for warm-starting


class SolverInfo(NamedTuple):
    qp_iters: Array
    qp_converged: Array
    # The objective at the step's starting q (see DiffIKSolver.objective)
    objective: Array


# converge() stops once its stepsize falls below this fraction of the full step
MIN_STEP_RATIO = 1e-3
_GAIN_ATTRS = ("gain", "gain_pos", "gain_rot")


def _max_gain(terms) -> float | None:
    """Largest gain over tasks and constraints, or None if any has no gain"""
    gains = []
    for term in terms:
        term_gains = [getattr(term, a) for a in _GAIN_ATTRS if hasattr(term, a)]
        if not term_gains:
            return None
        gains.extend(float(np.max(g)) for g in term_gains)
    return max(gains)


@jax.tree_util.register_static
class DiffIKSolver:
    """Differential IK as an elastic QP over the velocity v, which the robot
    then integrates into q:

        min  sum_tasks || weight * (J v - v_des) ||^2 + (reg + lam) * ||v||^2
        s.t. G v <= h  (inequalities and elastic equalities, L1-relaxed)
             A v == b  (hard equalities)

    with Levenberg-Marquardt damping lam = damping * sum_tasks || weight * v_des ||^2,
    the task cost at q. Gauss-Newton ignores the residuals' curvature, which grows
    with the residuals, so with a target out of reach a step can overshoot and
    settle into a cycle instead of the best compromise. The damping grows with
    the residuals and vanishes as the tasks are met (Sugihara 2011)

    q (nq,) and v (nv,) differ in size for a quaternion floating base (nq =
    nv + 1), with q_dot = E(q) v (frax's velocity_to_qdot_map). The first
    nq_floating / nv_floating entries are the floating base, if any

    Args:
        robot: frax robot: the configuration space and its integrator
        tasks (Sequence[Task]): Tracking tasks
        constraints (Sequence[InequalityConstraint | EqualityConstraint]): Inequality and
            equality constraints (may be empty)
        regularization (ArrayLike): Tikhonov regularization on v, scalar or per
            entry of v, shape (nv,)
        damping (float): Levenberg-Marquardt damping, per unit of task cost
        solver_tol (float): QP solver tolerance
        warm_start (bool): Warm-start each QP from the previous solution
    """

    def __init__(
        self,
        robot: Robot,
        tasks: Sequence[Task],
        constraints: Sequence[InequalityConstraint | EqualityConstraint] = (),
        regularization: ArrayLike = 1e-8,
        damping: float = 0.0,
        solver_tol: float = 1e-5,
        warm_start: bool = True,
    ):
        assert len(tasks) > 0
        assert damping >= 0
        self.robot = robot
        self.tasks = tuple(tasks)
        self.constraints = tuple(constraints)
        self.regularization = np.broadcast_to(
            np.asarray(regularization, dtype=float), (robot.nv,)
        ).copy()
        self.damping = damping
        self.solver_tol = solver_tol
        self.warm_start = warm_start
        # 1 / max_gain is a full Gauss-Newton step, the largest stable stepsize
        self.max_gain = _max_gain(self.tasks + self.constraints)
        for c in self.constraints:
            assert isinstance(c, InequalityConstraint | EqualityConstraint), (
                f"{type(c).__name__} is neither an InequalityConstraint nor an "
                + "EqualityConstraint"
            )
        for term in self.tasks + self.constraints:
            assert term.robot is robot, (
                f"{type(term).__name__} was built for a different robot than the solver's"
            )
        equalities = [c for c in self.constraints if isinstance(c, EqualityConstraint)]
        self.inequalities = tuple(
            c for c in self.constraints if isinstance(c, InequalityConstraint)
        )
        self.elastic_equalities = tuple(c for c in equalities if c.elastic)
        self.hard_equalities = tuple(c for c in equalities if not c.elastic)
        self.n_eq = sum(c.num_rows for c in self.hard_equalities)

        def penalties(c):
            return np.broadcast_to(np.asarray(c.penalty, dtype=float), (c.num_rows,))

        # Inequality rows, then each elastic equality's upper and lower rows
        self.penalty = np.concatenate(
            [penalties(c) for c in self.inequalities]
            + [np.tile(penalties(c), 2) for c in self.elastic_equalities]
            + [np.zeros(0)]
        )
        self.n_ineq = len(self.penalty)

    def init(self, q0: Array) -> SolverState:
        nv = self.robot.nv
        qp_xyz = (jnp.zeros(nv), jnp.zeros(self.n_eq), jnp.zeros(self.n_ineq))
        return SolverState(q=jnp.asarray(q0), qp_xyz=qp_xyz)

    def _rows(self, q: Array, ref):
        """Stacked task rows (J, v_des, w), inequality rows (G, h) (elastic
        equalities as +-pairs), hard equality rows (A, b), and the objective at q"""
        robot = self.robot
        assert q.shape == (robot.nq,), f"q has shape {q.shape}, expected ({robot.nq},)"
        Js, v_des, ws = zip(*(task.linearize(q, ref) for task in self.tasks))
        J = jnp.concatenate(Js)
        v_des = jnp.concatenate(v_des)
        w = jnp.concatenate(ws)

        Gs, hs = [jnp.zeros((0, robot.nv))], [jnp.zeros(0)]
        for con in self.inequalities:
            G, h = con.rows(q, ref)
            Gs.append(G)
            hs.append(h)
        for con in self.elastic_equalities:
            A, b = con.rows(q, ref)
            Gs.extend([A, -A])
            hs.extend([b, -b])
        G = jnp.concatenate(Gs)
        h = jnp.concatenate(hs)
        A = b = None
        if self.hard_equalities:
            As, bs = zip(*(con.rows(q, ref) for con in self.hard_equalities))
            A = jnp.concatenate(As)
            b = jnp.concatenate(bs)

        # The QP's objective at v = 0: task terms, plus the L1 penalties of the
        # rows violated at v = 0 (hard equalities at DEFAULT_PENALTY)
        objective = jnp.sum((w * v_des) ** 2) + jnp.maximum(-h, 0.0) @ self.penalty
        if b is not None:
            objective += DEFAULT_PENALTY * jnp.sum(jnp.abs(b))
        return (J, v_des, w), (G, h), (A, b), objective

    def objective(self, q: Array, ref) -> Array:
        """How well q solves the problem: the QP's objective at v = 0, i.e.
        sum_tasks || weight * v_des ||^2 plus the L1 penalties of violated rows
        (hard equalities at DEFAULT_PENALTY)"""
        return self._rows(q, ref)[3]

    def solve(
        self, q: Array, ref, warm_xyz: tuple[Array, Array, Array]
    ) -> tuple[Array, tuple[Array, Array, Array], SolverInfo]:
        """Solve the diff IK QP at q. Returns the velocity, the QP solution, and
        info (including the objective at q)"""
        (J, v_des, w), (G, h), (A, b), objective = self._rows(q, ref)
        WJ = w[:, None] * J
        lam = self.damping * jnp.sum((w * v_des) ** 2)
        P = WJ.T @ WJ + jnp.diag(self.regularization + lam)
        c = -(w * v_des) @ WJ
        result = elastiqp.jax.solve(
            P,
            c,
            G,
            h,
            penalty=self.penalty,
            A=A,
            b=b,
            eps_abs=self.solver_tol,
            warm_start=warm_xyz if self.warm_start else None,
        )
        info = SolverInfo(
            qp_iters=result.iters, qp_converged=result.converged, objective=objective
        )
        return result.x, (result.x, result.y, result.z), info

    def step(
        self, state: SolverState, ref, dt: float
    ) -> tuple[SolverState, Array, SolverInfo]:
        """One diff IK step. Returns the new state, the velocity, and info"""
        v, qp_xyz, info = self.solve(state.q, ref, state.qp_xyz)
        q = self.robot.integrate(state.q, v, dt)
        return SolverState(q=q, qp_xyz=qp_xyz), v, info

    def converge(
        self, state: SolverState, ref, max_iters: int, dt_max: float, tol: float
    ) -> tuple[SolverState, Array]:
        """Iterate on a fixed reference, accepting only steps that lower the
        objective: an accepted step doubles dt (up to dt_max), a rejected one
        halves it. Stops once max |v| <= tol, or dt < MIN_STEP_RATIO * dt_max (no
        step helps, e.g. an unreachable target)

        A redundant robot needs a nullspace task (e.g. posture) for v to vanish

        Returns the state and the number of iterations (QP solves after the first)"""
        state, ref = jax.lax.stop_gradient((state, ref))
        robot = self.robot
        v, qp_xyz, info = self.solve(state.q, ref, state.qp_xyz)
        dt_min = MIN_STEP_RATIO * dt_max

        def cond(carry):
            *_, i, done = carry
            return ~done & (i < max_iters)

        def body(carry):
            q, v, qp_xyz, objective, dt, i, _ = carry
            q_new = robot.integrate(q, v, dt)
            v_new, qp_xyz_new, info = self.solve(q_new, ref, qp_xyz)
            accept = info.objective < objective
            q, v, qp_xyz, objective = jax.tree.map(
                lambda new, old: jnp.where(accept, new, old),
                (q_new, v_new, qp_xyz_new, info.objective),
                (q, v, qp_xyz, objective),
            )
            dt = jnp.where(accept, jnp.minimum(2.0 * dt, dt_max), 0.5 * dt)
            done = (accept & (jnp.max(jnp.abs(v)) <= tol)) | (dt < dt_min)
            return q, v, qp_xyz, objective, dt, i + 1, done

        carry = (
            state.q,
            v,
            qp_xyz,
            info.objective,
            jnp.asarray(dt_max, dtype=state.q.dtype),
            0,
            jnp.max(jnp.abs(v)) <= tol,
        )
        q, _, qp_xyz, _, _, iters, _ = jax.lax.while_loop(cond, body, carry)
        return SolverState(q=q, qp_xyz=qp_xyz), iters
