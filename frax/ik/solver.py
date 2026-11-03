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


# TODO: Consider renaming the qp_xyz term
class SolverState(NamedTuple):
    q: Array
    qp_xyz: tuple[Array, Array, Array]  # Warm start data


class SolverInfo(NamedTuple):
    qp_iters: Array
    qp_converged: Array
    objective: Array


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
    """Differential IK (with an elastic QP)

        min  sum_tasks || weight * (J v - v_des) ||^2 + (reg + lam) * ||v||^2
        s.t. G v <= h  (inequalities, including elastic equalities, L1-relaxed)
             A v == b  (hard equalities)

    The Levenberg-Marquardt damping lam = damping * sum_tasks || weight * v_des ||^2
    keeps unreachable targets from cycling, and vanishes as the tasks are met

    Args:
        robot (Robot): Robot model
        tasks (Sequence[Task]): Tracking tasks
        constraints (Sequence[InequalityConstraint | EqualityConstraint]):
            Constraints. Defaults to an empty tuple (unconstrained)
        regularization (ArrayLike): Tikhonov regularization on v,
            scalar or shape (nv,). Defaults to 1e-8
        damping (float): Levenberg-Marquardt damping, per unit of task cost.
            Defaults to 0.0 (no damping)
        solver_tol (float): QP solver tolerance. Defaults to 1e-5
        warm_start (bool): Whether to warm-start each QP from the previous
            solution. Defaults to True.
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
        self.inequalities = tuple(
            c for c in self.constraints if isinstance(c, InequalityConstraint)
        )
        self.equalities = tuple(
            c for c in self.constraints if isinstance(c, EqualityConstraint)
        )
        self.n_eq = sum(c.num_rows for c in self.equalities)

        def penalties(c):
            return np.broadcast_to(np.asarray(c.penalty, dtype=float), (c.num_rows,))

        self.penalty = np.concatenate(
            [penalties(c) for c in self.inequalities] + [np.zeros(0)]
        )
        self.n_ineq = len(self.penalty)

    def init(self, q0: Array) -> SolverState:
        nv = self.robot.nv
        qp_xyz = (jnp.zeros(nv), jnp.zeros(self.n_eq), jnp.zeros(self.n_ineq))
        return SolverState(q=jnp.asarray(q0), qp_xyz=qp_xyz)

    # TODO improve the organization of the returns
    def _rows(self, q: Array, ref):
        """Stacked task (J, v_des, w), inequality (G, h), and hard equality (A, b)
        rows, and the objective at q"""
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
        G = jnp.concatenate(Gs)
        h = jnp.concatenate(hs)
        A = b = None
        if self.equalities:
            As, bs = zip(*(con.rows(q, ref) for con in self.equalities))
            A = jnp.concatenate(As)
            b = jnp.concatenate(bs)

        objective = jnp.sum((w * v_des) ** 2) + jnp.maximum(-h, 0.0) @ self.penalty
        if b is not None:
            # TODO: This seems like a bug, why is this not using the self.penalty
            objective += DEFAULT_PENALTY * jnp.sum(jnp.abs(b))
        return (J, v_des, w), (G, h), (A, b), objective

    def objective(self, q: Array, ref) -> Array:
        """The QP objective, evaluated at q and any provided reference data"""
        return self._rows(q, ref)[3]

    def solve(
        self, q: Array, ref, warm_xyz: tuple[Array, Array, Array]
    ) -> tuple[Array, tuple[Array, Array, Array], SolverInfo]:
        """Solve the differential IK QP

        Args:
            q (Array): Configuration, shape (nq,)
            ref (Any): Per-step reference data (any pytree)
            warm_xyz (tuple[Array, Array, Array]): QP warm start

        Returns:
            tuple[Array, tuple[Array, Array, Array], SolverInfo]:
                v (Array): Generalized velocity, shape (nv,)
                qp_xyz (tuple[Array, Array, Array]): QP solution data
                info (SolverInfo): Solver info, with the objective at q
        """
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
        """One diff IK step

        Args:
            state (SolverState): Current state
            ref (Any): Per-step reference data (any pytree)
            dt (float): Timestep

        Returns:
            tuple[SolverState, Array, SolverInfo]: New state, velocity, and info
        """
        v, qp_xyz, info = self.solve(state.q, ref, state.qp_xyz)
        q = self.robot.integrate(state.q, v, dt)
        return SolverState(q=q, qp_xyz=qp_xyz), v, info

    def converge(
        self, state: SolverState, ref, max_iters: int, dt_max: float, tol: float
    ) -> tuple[SolverState, Array]:
        """Iterate on a fixed reference, accepting only steps that lower the
        objective. Accepted steps double dt (up to dt_max), rejected ones halve it

        Stops once max |v| <= tol (a redundant robot needs a posture task for
        this), or dt < MIN_STEP_RATIO * dt_max

        Args:
            state (SolverState): Initial state
            ref (Any): Reference data (any pytree)
            max_iters (int): Maximum iterations
            dt_max (float): Largest stepsize
            tol (float): Convergence tolerance on max |v|

        Returns:
            tuple[SolverState, Array]: Converged state, and the number of iterations
        """
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
