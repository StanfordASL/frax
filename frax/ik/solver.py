"""Differential IK

min  sum_tasks || weight * (J v - v_des) ||^2 + (reg + lam) * ||v||^2
s.t. A v == b; G v <= h

i.e.
- Minimimize the weighted task velocity error, plus regularization
and damping terms for stability
- Subject to a set of equality and inequality constraints

Note that since we use ElastiQP as our solver, inequality constraints
are automatically (and efficiently) relaxed with an L1 penalty term.
These constraints are held exactly when the problem is feasible, and
only relaxed when the problem is infeasible*

*See the ElastiQP paper for more details, and more precise conditions
on relaxation and when this occurs

Equality constraints are still held as hard constraints with ElastiQP,
unless explicitly posed as an elastic equality. In this case, the
equality is converted to a pair of opposing elastic inequalities.
"""

from collections.abc import Sequence
from typing import Any, NamedTuple

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
    qp_warm_start: tuple[Array, Array, Array]  # Last QP sol primal + duals, (x, y, z)


class SolverInfo(NamedTuple):
    qp_iters: Array
    qp_converged: Array
    cost: Array


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
    """Differential IK with ElastiQP

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
        qp_warm_start = (jnp.zeros(nv), jnp.zeros(self.n_eq), jnp.zeros(self.n_ineq))
        q = jnp.asarray(q0, dtype=jnp.float64)  # float64 for elastiqp
        return SolverState(q=q, qp_warm_start=qp_warm_start)

    def _qp(self, q: Array, ref: Any):
        """The diff IK QP at q, and the cost of q

        Returns:
            tuple: P, c, G, h, A, b (A and b are None if there are no hard
                equalities), and the cost of q (the QP objective at v = 0)
        """
        robot = self.robot
        assert q.shape == (robot.nq,), f"q has shape {q.shape}, expected ({robot.nq},)"
        Js, v_des, ws = zip(*(task.linearize(q, ref) for task in self.tasks))
        w = jnp.concatenate(ws)
        WJ = w[:, None] * jnp.concatenate(Js)
        w_v_des = w * jnp.concatenate(v_des)
        task_error = jnp.sum(w_v_des**2)
        P = WJ.T @ WJ + jnp.diag(self.regularization + self.damping * task_error)
        c = -w_v_des @ WJ

        Gs, hs = [jnp.zeros((0, robot.nv))], [jnp.zeros(0)]
        for con in self.inequalities:
            G, h = con.rows(q, ref)
            Gs.append(G)
            hs.append(h)
        G = jnp.concatenate(Gs)
        h = jnp.concatenate(hs)
        cost = task_error + jnp.maximum(-h, 0.0) @ self.penalty

        A = b = None
        if self.equalities:
            As, bs = zip(*(con.rows(q, ref) for con in self.equalities))
            A = jnp.concatenate(As)
            b = jnp.concatenate(bs)
            # Hard equalities have no penalty, so their residual |A 0 - b| is
            # weighted like a default elastic row's
            cost += DEFAULT_PENALTY * jnp.sum(jnp.abs(b))
        return P, c, G, h, A, b, cost

    def cost(self, q: Array, ref: Any) -> Array:
        """The cost of q: the QP objective at v = 0 (if the robot doesn't move),
        for the given reference data"""
        *_, cost = self._qp(q, ref)
        return cost

    def solve(
        self, q: Array, ref: Any, qp_warm_start: tuple[Array, Array, Array]
    ) -> tuple[Array, tuple[Array, Array, Array], SolverInfo]:
        """Solve the diff IK QP

        Args:
            q (Array): Configuration, shape (nq,)
            ref (Any): Per-step reference data (any pytree)
            qp_warm_start (tuple[Array, Array, Array]): QP warm start (x, y, z)

        Returns:
            v (Array): Generalized velocity, shape (nv,)
            qp_warm_start (tuple[Array, Array, Array]): QP solution (x, y, z),
                to warm-start the next QP
            info (SolverInfo): Solver info, with the cost of q
        """
        P, c, G, h, A, b, cost = self._qp(q, ref)
        result = elastiqp.jax.solve(
            P,
            c,
            G,
            h,
            penalty=self.penalty,
            A=A,
            b=b,
            eps_abs=self.solver_tol,
            warm_start=qp_warm_start if self.warm_start else None,
        )
        info = SolverInfo(result.iters, result.converged, cost)
        return result.x, (result.x, result.y, result.z), info

    def step(
        self, state: SolverState, ref: Any, dt: float
    ) -> tuple[SolverState, Array, SolverInfo]:
        """One diff IK step

        Args:
            state (SolverState): Current state
            ref (Any): Per-step reference data (any pytree)
            dt (float): Timestep

        Returns:
            state (SolverState): New state
            v (Array): Generalized velocity, shape (nv,)
            info (SolverInfo): Solver info
        """
        v, qp_warm_start, info = self.solve(state.q, ref, state.qp_warm_start)
        q = self.robot.integrate(state.q, v, dt)
        return SolverState(q=q, qp_warm_start=qp_warm_start), v, info

    def converge(
        self, state: SolverState, ref: Any, max_iters: int, dt_max: float, tol: float
    ) -> tuple[SolverState, Array]:
        """Run differential IK until convergence

        Note: This method should generally only be called as a component of a
        global IK pipeline, or at initialization of a solver. During online
        operation, or sequential calls in an offline pipeline, generally
        only a single diff IK step() should be taken.

        Args:
            state (SolverState): Initial state
            ref (Any): Reference data (any pytree)
            max_iters (int): Maximum iterations
            dt_max (float): Largest stepsize
            tol (float): Convergence tolerance on max |v|

        Returns:
            state (SolverState): Converged state
            iters (Array): Number of iterations
        """
        state, ref = jax.lax.stop_gradient((state, ref))
        robot = self.robot
        v, qp_warm_start, info = self.solve(state.q, ref, state.qp_warm_start)
        dt_min = MIN_STEP_RATIO * dt_max

        def cond(carry):
            *_, i, done = carry
            return ~done & (i < max_iters)

        def body(carry):
            q, v, qp_warm_start, cost, dt, i, _ = carry
            q_new = robot.integrate(q, v, dt)
            v_new, qp_warm_start_new, info = self.solve(q_new, ref, qp_warm_start)
            # Accept if the step lowers the cost
            accept = info.cost < cost
            q, v, qp_warm_start, cost = jax.tree.map(
                lambda new, old: jnp.where(accept, new, old),
                (q_new, v_new, qp_warm_start_new, info.cost),
                (q, v, qp_warm_start, cost),
            )
            # If accepted, double dt, otherwise halve it
            dt = jnp.where(accept, jnp.minimum(2.0 * dt, dt_max), 0.5 * dt)
            # Stopping condition on the max |v| or on a minimum dt
            done = (accept & (jnp.max(jnp.abs(v)) <= tol)) | (dt < dt_min)
            return q, v, qp_warm_start, cost, dt, i + 1, done

        carry = (
            state.q,
            v,
            qp_warm_start,
            info.cost,
            jnp.asarray(dt_max, dtype=state.q.dtype),
            0,
            jnp.max(jnp.abs(v)) <= tol,
        )
        q, _, qp_warm_start, _, _, iters, _ = jax.lax.while_loop(cond, body, carry)
        return SolverState(q=q, qp_warm_start=qp_warm_start), iters
