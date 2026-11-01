"""Global IK by multi-start: converge the diff IK solver from many sampled
configurations, and keep the best result

For continuous motion, use the diff IK solver directly. Global IK is for when a
jump is acceptable, e.g. to pick a goal configuration or start a clip
"""

from collections.abc import Sequence
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from frax.core.robot import Robot
from frax.ik.constraints import JointLimits
from frax.ik.parallel import _resolve_devices, device_vmap
from frax.ik.solver import DiffIKSolver


class GlobalIKResult(NamedTuple):
    q: Array  # Selected configuration (nq,)
    cost: Array  # Its objective
    qs: Array  # Every converged start (num_samples, nq). Without sampling, all q
    costs: Array  # Their objectives (num_samples,)
    iters: Array  # Convergence iterations per start (num_samples,)
    sampled: Array  # Whether the random starts were used (see `cost_tol`)


def sample_configurations(
    robot: Robot,
    q: Array,
    key: Array,
    num_samples: int,
    q_min: Array,
    q_max: Array,
) -> Array:
    """num_samples configurations (num_samples, nq): q itself, then actuated
    joints uniform within [q_min, q_max] (shape (nq - nq_floating,)), with any
    floating base kept at q's"""
    joints = jax.random.uniform(
        key,
        (num_samples - 1, robot.nq - robot.nq_floating),
        minval=q_min,
        maxval=q_max,
        dtype=q.dtype,
    )
    base = jnp.broadcast_to(
        q[: robot.nq_floating], (num_samples - 1, robot.nq_floating)
    )
    return jnp.concatenate([q[None], jnp.concatenate([base, joints], axis=1)])


@jax.tree_util.register_static
class GlobalIKSolver:
    """Multi-start IK: converge a diff IK solver from q and num_samples - 1
    random configurations (see `sample_configurations`), and select a solution
    by its objective

    With cost_tol None, the lowest objective is kept. With cost_tol set, q is
    converged alone first and kept if its objective is at most cost_tol;
    otherwise the solution nearest q among those within cost_tol of the lowest

    Args:
        solver (DiffIKSolver): Diff IK solver, with the problem's tasks and constraints
        num_samples (int): Number of starts, including q
        q_min, q_max (ArrayLike | None): Sampling range for the actuated joints.
            Default to the solver's JointLimits
        max_iters (int): Max convergence iterations per start
        dt_max (float | None): Largest convergence stepsize (see
            DiffIKSolver.converge). Defaults to 1 / solver.max_gain
        tol (float): Convergence tolerance on max |v| (see DiffIKSolver.converge)
        cost_tol (float | None): Objective tolerance for acceptable solutions.
            None selects the lowest objective
        devices ("all" | "default" | Sequence[jax.Device]): Devices to split the
            starts across (see `frax.ik.parallel`). Default: JAX's default device
            alone, where the cost is linear in num_samples on CPU
    """

    def __init__(
        self,
        solver: DiffIKSolver,
        num_samples: int = 32,
        q_min: ArrayLike | None = None,
        q_max: ArrayLike | None = None,
        max_iters: int = 200,
        dt_max: float | None = None,
        tol: float = 1e-4,
        cost_tol: float | None = None,
        devices: Literal["all", "default"] | Sequence[jax.Device] = "default",
    ):
        self.solver = solver
        self.num_samples = num_samples
        if q_min is None or q_max is None:
            limits = [c for c in solver.constraints if isinstance(c, JointLimits)]
            assert limits, "Pass q_min and q_max, or give the solver JointLimits"
            q_min = limits[0].q_min if q_min is None else q_min
            q_max = limits[0].q_max if q_max is None else q_max
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        assert np.all(np.isfinite(self.q_min)) and np.all(np.isfinite(self.q_max)), (
            "Unbounded sampling range"
        )
        self.max_iters = max_iters
        if dt_max is None:
            assert solver.max_gain is not None, "A term has no gain: pass dt_max"
            dt_max = 1.0 / solver.max_gain
        self.dt_max = dt_max
        self.tol = tol
        self.cost_tol = cost_tol
        self.devices = tuple(_resolve_devices(devices))

    def solve(self, q: Array, ref, key: Array) -> GlobalIKResult:
        """Solve from the current configuration q (nq,), for a reference ref
        (as the solver's), with PRNG key for the samples"""
        solver, robot = self.solver, self.solver.robot
        joints = slice(robot.nq_floating, None)

        def converge(q0, ref):
            state, iters = solver.converge(
                solver.init(q0), ref, self.max_iters, self.dt_max, self.tol
            )
            return state.q, iters, solver.objective(state.q, ref)

        def select(qs, costs, iters, sampled):
            if self.cost_tol is None:
                best = jnp.argmin(costs)
            else:
                acceptable = costs <= jnp.min(costs) + self.cost_tol
                distance = jnp.sum((qs[:, joints] - q[joints]) ** 2, axis=1)
                best = jnp.argmin(jnp.where(acceptable, distance, jnp.inf))
            return GlobalIKResult(qs[best], costs[best], qs, costs, iters, sampled)

        def sample():
            q0s = sample_configurations(
                robot, q, key, self.num_samples, self.q_min, self.q_max
            )
            # One device is a plain vmap
            converge_all = device_vmap(converge, self.devices, in_axes=(0, None))
            qs, iters, costs = converge_all(q0s, ref)
            return select(qs, costs, iters, jnp.array(True))

        if self.cost_tol is None:
            return sample()
        q_local, iters_local, cost_local = converge(q, ref)

        def keep_local():
            tile = lambda x: jnp.broadcast_to(x, (self.num_samples,) + jnp.shape(x))
            return select(
                tile(q_local), tile(cost_local), tile(iters_local), jnp.array(False)
            )

        return jax.lax.cond(cost_local <= self.cost_tol, keep_local, sample)
