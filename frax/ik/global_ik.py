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


def sample_configurations(
    robot: Robot,
    q: Array,
    key: Array,
    num_samples: int,
    q_min: ArrayLike | Literal["robot"] = "robot",
    q_max: ArrayLike | Literal["robot"] = "robot",
) -> Array:
    """Sample configurations for global IK

    The actuated joints are sampled uniformly. The floating base stays at q's:
    it has no joint limits to sample within, and is usually pinned by a task

    Args:
        robot (Robot): Robot model
        q (Array): Nominal configuration, shape (nq,).
            This is also included as the first sample
        key (Array): PRNG key
        num_samples (int): Number of samples, including q
        q_min (ArrayLike | "robot"): Lower joint bounds, shape (num_actuated,).
            Defaults to "robot", for robot.actuated_joint_lower_limits
        q_max (ArrayLike | "robot"): Upper joint bounds, shape (num_actuated,).
            Defaults to "robot", for robot.actuated_joint_upper_limits

    Returns:
        Array: Sampled configurations, shape (num_samples, nq)
    """
    if isinstance(q_min, str):
        assert q_min == "robot", f'q_min must be an array or "robot", got {q_min!r}'
        q_min = np.asarray(robot.actuated_joint_lower_limits, dtype=float)
        assert np.all(np.isfinite(q_min)), "Unbounded joint limits: pass q_min"
    if isinstance(q_max, str):
        assert q_max == "robot", f'q_max must be an array or "robot", got {q_max!r}'
        q_max = np.asarray(robot.actuated_joint_upper_limits, dtype=float)
        assert np.all(np.isfinite(q_max)), "Unbounded joint limits: pass q_max"
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
        q_min, q_max (ArrayLike | "robot"): Sampling range for the actuated
            joints. Default to "robot", for the robot's joint limits, narrowed
            to those of any JointLimits constraints in the solver
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
        q_min: ArrayLike | Literal["robot"] = "robot",
        q_max: ArrayLike | Literal["robot"] = "robot",
        max_iters: int = 200,
        dt_max: float | None = None,
        tol: float = 1e-4,
        cost_tol: float | None = None,
        devices: Literal["all", "default"] | Sequence[jax.Device] = "default",
    ):
        self.solver = solver
        self.num_samples = num_samples

        # Handle joint limits for determining a sampling range
        lower = np.asarray(solver.robot.actuated_joint_lower_limits, dtype=float)
        upper = np.asarray(solver.robot.actuated_joint_upper_limits, dtype=float)
        for c in solver.constraints:
            if isinstance(c, JointLimits):
                lower = np.maximum(lower, c.q_min)
                upper = np.minimum(upper, c.q_max)
        for x in (q_min, q_max):
            if isinstance(x, str):
                assert x == "robot", (
                    f'Sampling limits must be arrays or "robot", got {x!r}'
                )
        self.q_min = lower if isinstance(q_min, str) else np.asarray(q_min, dtype=float)
        self.q_max = upper if isinstance(q_max, str) else np.asarray(q_max, dtype=float)
        assert np.all(np.isfinite(self.q_min)) and np.all(np.isfinite(self.q_max)), (
            "Unbounded sampling range: pass q_min and q_max, or give the solver "
            + "JointLimits"
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
