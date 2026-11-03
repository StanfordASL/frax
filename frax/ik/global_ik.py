"""Global IK with multi-start SQP

Note: for continuous motion control, you should probably stick to
differential IK. Global IK is useful for if instantaneous jumps are
acceptable: for instance, finding a configuration that achieves a
desired initial or goal pose, without worrying about continuity from
your current state.
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
    q: Array  # Best configuration
    cost: Array  # Best cost
    qs: Array  # Every converged q from multiple samples
    costs: Array  # Every cost from multiple samples
    iters: Array  # Convergence iterations per sample
    sampled: Array  # Whether samples were necessary to meet the tolerance


# TODO: It's a little weird that there is no sampling for the floating base?
# And should we do the same thing as the JointLimits constraint where it defaults
# to using the robot limits but can be adjusted?
def sample_configurations(
    robot: Robot,
    q: Array,
    key: Array,
    num_samples: int,
    q_min: Array,
    q_max: Array,
) -> Array:
    """Sample configurations for global IK

    Currently, we sample uniformly for the actuated joints, and keep
    the floating base DoFs at the nominal configuration

    Args:
        robot (Robot): Robot model
        q (Array): Nominal configuration, shape (nq,).
            This is also included as the first sample
        key (Array): PRNG key
        num_samples (int): Number of samples, including q
        q_min (Array): Lower joint bounds, shape (num_actuated,)
        q_max (Array): Upper joint bounds, shape (num_actuated,)

    Returns:
        Array: Sampled configurations, shape (num_samples, nq)
    """
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
    """Multi-start SQP global IK solver


    Args:
        solver (DiffIKSolver): Diff IK solver
        num_samples (int): Number of starts, including q
        q_min, q_max (ArrayLike | None): Sampling range for the actuated joints.
            Default to the solver's JointLimits
        max_iters (int): Max convergence iterations per start
        dt_max (float | None): Largest convergence stepsize. Defaults to
            1 / solver.max_gain
        tol (float): Convergence tolerance on max |v|
        cost_tol (float | None): Objective tolerance for acceptable solutions. If
            this term is set, then we return the closest sampled solution that is
            within cost_tol in objective. Defaults to None (select the lowest objective
            value regardless of how far the solution is in configuration space)
        devices ("all" | "default" | Sequence[jax.Device]): Devices to parallelize
            the samples across. Defaults to "default" (JAX's default device).
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
        """Solve global IK

        Args:
            q (Array): Current configuration, shape (nq,)
            ref (Any): Reference data (any pytree)
            key (Array): PRNG key for the samples

        Returns:
            GlobalIKResult: Result
        """
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
