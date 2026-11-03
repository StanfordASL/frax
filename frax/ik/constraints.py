"""IK constraints

Notes:
- We use ElastiQP for our solver, which allows us to use efficient
  elastic inequality constraints, and hard equalities (where applicable)
- Each inequality has an L1 relaxation with associated penalty w,
  typically left at a default high value like 1e6, though this can be
  tuned on a per-task basis. If this penalty is greater than the optimal
  dual multiplier for a feasible problem, the relaxation is exact
- For tuning gains, ensure that gain * dt <= 1 to avoid overshoot
- Since this solver operates on generalized velocities, all inequalities
  deal with v rather than a delta q. This is slightly different from
  other diff IK libraries like pink or mink
"""

from collections.abc import Callable
from typing import Literal, NamedTuple, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from frax.core.robot import Robot
from frax.ik.tasks import _velocity_jacobian


DEFAULT_PENALTY = 1e6  # Default elastic penalty value
_INACTIVE_DISTANCE = 1e3  # Value assigned to inactive obstacles


class InequalityConstraint:
    """Base (elastic) inequality constraint G v <= h"""

    robot: Robot
    num_rows: int
    penalty: ArrayLike

    def rows(self, q: Array, ref: Any) -> tuple[Array, Array]:
        """Rows of G v <= h for this constraint, evaluated at q

        Args:
            q (Array): Configuration, shape (nq,)
            ref (Any): A pytree (such as a dict) containing problem data
                that is updated each step

        Returns:
            G (Array): Inequality matrix rows, shape (num_rows, nv)
            h (Array): Inequality vector rows, shape (num_rows,)
        """
        raise NotImplementedError


class EqualityConstraint:
    """Base (hard) equality constraint Av = b

    Generally, you should wrap any soft task terms that may not be feasible by
    construction in an ElasticEquality
    """

    robot: Robot
    num_rows: int

    def rows(self, q: Array, ref: Any) -> tuple[Array, Array]:
        """Rows of A v = b for this constraint, evaluated at q

        Args:
            q (Array): Configuration, shape (nq,)
            ref (Any): A pytree (such as a dict) containing problem data
                that is updated each step

        Returns:
            A (Array): Equality matrix rows, shape (num_rows, nv)
            b (Array): Equality vector rows, shape (num_rows,)
        """
        raise NotImplementedError


def _resolve_limits(
    x: ArrayLike | Literal["robot"] | None, robot_x: np.ndarray, n: int, sign: float
) -> np.ndarray:
    if isinstance(x, str):
        assert x == "robot", f'A limit must be an array, "robot", or None, got {x!r}'
        x = robot_x
    if x is None:
        return np.full(n, sign * np.inf)
    x = np.asarray(x, dtype=float)
    assert x.shape == (n,), f"Joint limits must have shape ({n},), got {x.shape}"
    return x


@jax.tree_util.register_static
class JointLimits(InequalityConstraint):
    """Joint position and velocity limits on the actuated joint velocities

        max(v_min, gain * (q_min - q)) <= v <= min(v_max, gain * (q_max - q))

    Limits default to "robot" to use the robot's own actuated joint position/
    velocity limits, but you can also pass None (no limit) or an array of
    custom limits.

    Currently, limits on floating bases are not implemented.

    Args:
        robot (Robot): The frax robot model
        q_min (ArrayLike | str | None): Lower position limits, shape (num_actuated,).
            Defaults to robot.actuated_joint_lower_limits
        q_max (ArrayLike | str | None): Upper position limits, shape (num_actuated,).
            Defaults to robot.actuated_joint_upper_limits
        v_max (ArrayLike | str | None): Upper velocity limits, shape (num_actuated,).
            Defaults to robot.actuated_joint_max_velocities
        v_min (ArrayLike | str | None): Lower velocity limits, shape (num_actuated,).
            Defaults to -v_max
        gain (float): CBF gain (1/s): how fast the position limits may be approached
        penalty (float): Relaxation penalty
    """

    def __init__(
        self,
        robot: Robot,
        q_min: ArrayLike | Literal["robot"] | None = "robot",
        q_max: ArrayLike | Literal["robot"] | None = "robot",
        v_max: ArrayLike | Literal["robot"] | None = "robot",
        v_min: ArrayLike | Literal["robot"] | None = "robot",
        gain: float = 10.0,
        penalty: float = DEFAULT_PENALTY,
    ):
        assert gain > 0
        self.robot = robot
        n = robot.nv - robot.nv_floating  # Actuated joints
        self.q_min = _resolve_limits(q_min, robot.actuated_joint_lower_limits, n, -1)
        self.q_max = _resolve_limits(q_max, robot.actuated_joint_upper_limits, n, 1)
        self.v_max = _resolve_limits(v_max, robot.actuated_joint_max_velocities, n, 1)
        self.v_min = _resolve_limits(v_min, -self.v_max, n, -1)
        self.upper_idxs = np.flatnonzero(
            np.isfinite(self.q_max) | np.isfinite(self.v_max)
        )
        self.lower_idxs = np.flatnonzero(
            np.isfinite(self.q_min) | np.isfinite(self.v_min)
        )
        I = np.eye(n)
        G = np.vstack([I[self.upper_idxs], -I[self.lower_idxs]])
        self.num_rows = G.shape[0]
        assert self.num_rows > 0
        # Zero columns for the floating base
        self.G = np.hstack([np.zeros((self.num_rows, robot.nv_floating)), G])
        self.gain = gain
        self.penalty = penalty

    def rows(self, q, ref):
        pos = q[self.robot.nq_floating :]
        up, lo = self.upper_idxs, self.lower_idxs
        upper = jnp.minimum(self.v_max[up], self.gain * (self.q_max[up] - pos[up]))
        lower = jnp.maximum(self.v_min[lo], self.gain * (self.q_min[lo] - pos[lo]))
        return self.G, jnp.concatenate([upper, -lower])


class SphereObstacles(NamedTuple):
    centers: Array
    radii: Array
    active: Array


class _SphereConstraint(InequalityConstraint):
    """Helper class for CBFs operating on spherical collision models"""

    def __init__(self, robot: Robot, gain: float, penalty: float):
        assert robot.has_collision_data
        self.robot = robot
        self.radii = np.asarray(robot.flat_collision_radii)
        self.gain = gain
        self.penalty = penalty

    def _set_num_rows(self, num_candidates: int, max_rows: int | None):
        self.max_rows = (
            None if max_rows is None or max_rows >= num_candidates else max_rows
        )
        self.num_rows = num_candidates if self.max_rows is None else self.max_rows
        assert self.num_rows > 0

    def _keep(self, h):
        # If max_rows is set, keep only the closest constraints in the QP rather
        # than solving the QP with a potentially much larger set of constraints
        if self.max_rows is None:
            return slice(None)
        return jax.lax.top_k(-h, self.max_rows)[1]


@jax.tree_util.register_static
class SelfCollisionConstraint(_SphereConstraint):
    """Self-collision avoidance: h = robot.self_collision_distances(q) - margin

    Args:
        robot (Robot): Robot, with collision and self-collision data
        max_rows (int | None): If set, keep only this many closest pairs per step
        gain (float): CBF gain (1/s)
        penalty (float): Relaxation penalty
        margin (float): Additional safety margin between every collision pair
    """

    def __init__(
        self,
        robot: Robot,
        max_rows: int | None = None,
        gain: float = 10.0,
        penalty: float = DEFAULT_PENALTY,
        margin: float = 0.0,
    ):
        super().__init__(robot, gain, penalty)
        assert robot.has_sc_data
        self.margin = margin
        self._set_num_rows(len(robot.sc_pairs), max_rows)

    def rows(self, q, ref):
        joint_tfs = self.robot.joint_to_world_transforms(q)
        pos = self.robot._link_collision_positions(joint_tfs)
        dists = self.robot._self_collision_distances_from_link_data(pos, self.radii)
        h = dists - self.margin
        keep = self._keep(h)
        Jv, Jw = self.robot._joint_jacobians(joint_tfs)
        J = self.robot._self_collision_jacobian(joint_tfs, Jv, Jw, pos, keep)
        return -J, self.gain * h[keep]


@jax.tree_util.register_static
class ObstacleConstraint(_SphereConstraint):
    """Sphere obstacle avoidance: h = |p_i - c_k| - r_i - R_k - margin for each
    robot sphere i in sphere_idxs and obstacle k

    Args:
        robot (Robot): Robot, with collision data
        obstacles (SphereObstacles): Obstacles. Their count K is fixed
        sphere_idxs (ArrayLike | None): Robot collision spheres to check.
            Defaults to all
        ref_key (str | None): If set, ref[ref_key] replaces `obstacles` each step
        max_rows (int | None): If set, keep only this many closest pairs per step
        gain (float): CBF gain (1/s)
        penalty (float): Relaxation penalty
        margin (float): Safety distance from every obstacle
    """

    def __init__(
        self,
        robot: Robot,
        obstacles: SphereObstacles,
        sphere_idxs: ArrayLike | None = None,
        ref_key: str | None = None,
        max_rows: int | None = None,
        gain: float = 10.0,
        penalty: float = DEFAULT_PENALTY,
        margin: float = 0.0,
    ):
        super().__init__(robot, gain, penalty)
        self.margin = margin
        self.obstacles = SphereObstacles(
            np.asarray(obstacles.centers, dtype=float),
            np.asarray(obstacles.radii, dtype=float),
            np.asarray(obstacles.active, dtype=bool),
        )
        self.num_obstacles = self.obstacles.radii.shape[0]
        self.sphere_idxs = (
            np.arange(len(self.radii))
            if sphere_idxs is None
            else np.asarray(sphere_idxs)
        )
        self.row_spheres = np.tile(self.sphere_idxs, self.num_obstacles)
        self.ref_key = ref_key
        self._set_num_rows(len(self.row_spheres), max_rows)

    def rows(self, q, ref):
        obstacles = self.obstacles if self.ref_key is None else ref[self.ref_key]
        assert obstacles.radii.shape == (self.num_obstacles,), (
            "Obstacle count must match the constructor"
        )
        joint_tfs = self.robot.joint_to_world_transforms(q)
        pos = self.robot._link_collision_positions(joint_tfs)
        idxs = self.sphere_idxs
        delta = (pos[idxs][None] - obstacles.centers[:, None]).reshape(-1, 3)
        offsets = self.radii[idxs][None] + obstacles.radii[:, None] + self.margin
        offsets = offsets.reshape(-1)
        active = jnp.repeat(obstacles.active, len(idxs))
        dist = jnp.linalg.norm(delta, axis=-1)
        h = jnp.where(active, dist - offsets, _INACTIVE_DISTANCE)
        keep = self._keep(h)
        spheres = jnp.asarray(self.row_spheres)[keep]
        Jv, Jw = self.robot._joint_jacobians(joint_tfs)
        J = self.robot._link_collision_jacobians(joint_tfs, Jv, Jw, spheres)
        normals = delta[keep] / dist[keep, None]
        dh = jnp.einsum("ki,kij->kj", normals, J)
        dh = jnp.where(active[keep, None], dh, 0.0)
        return -dh, self.gain * h[keep]


@jax.tree_util.register_static
class FloorConstraint(_SphereConstraint):
    """Floor at z = 0: h = z_i - r_i - margin for each robot sphere i in sphere_idxs

    Args:
        robot (Robot): Robot, with collision data
        sphere_idxs (ArrayLike | None): Robot collision spheres to check.
            Defaults to all
        max_rows (int | None): If set, keep only this many lowest spheres per step
        gain (float): CBF gain (1/s)
        penalty (float): Relaxation penalty
        margin (float): Height kept above the floor
    """

    def __init__(
        self,
        robot: Robot,
        sphere_idxs: ArrayLike | None = None,
        max_rows: int | None = None,
        gain: float = 10.0,
        penalty: float = DEFAULT_PENALTY,
        margin: float = 0.0,
    ):
        super().__init__(robot, gain, penalty)
        self.margin = margin
        self.sphere_idxs = (
            np.arange(len(self.radii))
            if sphere_idxs is None
            else np.asarray(sphere_idxs)
        )
        self._set_num_rows(len(self.sphere_idxs), max_rows)

    def rows(self, q, ref):
        joint_tfs = self.robot.joint_to_world_transforms(q)
        pos = self.robot._link_collision_positions(joint_tfs)[self.sphere_idxs]
        h = pos[:, 2] - self.radii[self.sphere_idxs] - self.margin
        keep = self._keep(h)
        spheres = jnp.asarray(self.sphere_idxs)[keep]
        Jv, Jw = self.robot._joint_jacobians(joint_tfs)
        J = self.robot._link_collision_jacobians(joint_tfs, Jv, Jw, spheres)
        return -J[:, 2], self.gain * h[keep]


@jax.tree_util.register_static
class BarrierConstraint(InequalityConstraint):
    """CBF on a user-defined barrier h(q, ref) >= 0: dh/dq E(q) v >= -gain * h

    The Jacobian is by forward-mode autodiff, so this suits a handful of rows

    Args:
        robot (Robot): Robot model
        h_fn (Callable): (q, ref) -> h, shape (num_rows,)
        num_rows (int): Number of barriers
        gain (float): CBF gain (1/s)
        penalty (ArrayLike): Scalar or per-row relaxation penalty
    """

    def __init__(
        self,
        robot: Robot,
        h_fn: Callable[[Array, object], Array],
        num_rows: int,
        gain: float = 10.0,
        penalty: ArrayLike = DEFAULT_PENALTY,
    ):
        self.robot = robot
        self.h_fn = h_fn
        self.num_rows = num_rows
        self.gain = gain
        self.penalty = penalty

    def rows(self, q, ref):
        def h_fn(q):
            h = self.h_fn(q, ref)
            assert h.shape == (self.num_rows,), (
                f"h has shape {h.shape}, expected ({self.num_rows},)"
            )
            return h

        h, dh = _velocity_jacobian(self.robot, h_fn, q)
        return -dh, self.gain * h


@jax.tree_util.register_static
class ResidualEquality(EqualityConstraint):
    """Equality on a user-defined residual e(q, ref) = 0: de/dq E(q) v = -gain * e

    Args:
        robot (Robot): Robot model
        residual_fn (Callable): (q, ref) -> e, shape (num_rows,)
        num_rows (int): Residual dimension
        gain (float): Gain (1/s)
    """

    def __init__(
        self,
        robot: Robot,
        residual_fn: Callable[[Array, object], Array],
        num_rows: int,
        gain: float = 10.0,
    ):
        self.robot = robot
        self.residual_fn = residual_fn
        self.num_rows = num_rows
        self.gain = gain

    def rows(self, q, ref):
        def e_fn(q):
            e = self.residual_fn(q, ref)
            assert e.shape == (self.num_rows,), (
                f"e has shape {e.shape}, expected ({self.num_rows},)"
            )
            return e

        e, de = _velocity_jacobian(self.robot, e_fn, q)
        return de, -self.gain * e


@jax.tree_util.register_static
class ElasticEquality(InequalityConstraint):
    """An equality A v = b, relaxed with an L1 penalty, as the inequalities
    A v <= b and -A v <= -b

    Args:
        equality (EqualityConstraint): The equality to relax
        penalty (ArrayLike): Scalar or per-row relaxation penalty, shape
            (equality.num_rows,)
    """

    def __init__(
        self, equality: EqualityConstraint, penalty: ArrayLike = DEFAULT_PENALTY
    ):
        self.equality = equality
        self.robot = equality.robot
        self.num_rows = 2 * equality.num_rows
        penalty = np.broadcast_to(
            np.asarray(penalty, dtype=float), (equality.num_rows,)
        )
        self.penalty = np.tile(penalty, 2)

    @property
    def gain(self):
        return self.equality.gain

    def rows(self, q, ref):
        A, b = self.equality.rows(q, ref)
        return jnp.concatenate([A, -A]), jnp.concatenate([b, -b])
