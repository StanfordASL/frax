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

from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from frax.core.robot import Robot
from frax.ik.tasks import _velocity_jacobian, point_jacobians


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
    """Base equality constraint Av = b

    ElastiQP allows for hard equalities when applicable. Generally,
    you should use an elastic equality for any soft task terms that
    may not be feasible by construction
    """

    robot: Robot
    num_rows: int
    elastic: bool
    penalty: ArrayLike

    def rows(self, q: Array, ref: Any) -> tuple[Array, Array]:
        """Rows of A v <= b for this constraint, evaluated at q

        Args:
            q (Array): Configuration, shape (nq,)
            ref (Any): A pytree (such as a dict) containing problem data
                that is updated each step

        Returns:
            A (Array): Equality matrix rows, shape (num_rows, nv)
            b (Array): Equality vector rows, shape (num_rows,)
        """
        raise NotImplementedError


def _limits(
    x: ArrayLike | Literal["robot"] | None, robot_x: np.ndarray, n: int, sign: float
) -> np.ndarray:
    # Helper function to handle the different ways of passing a joint limit
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
        self.q_min = _limits(q_min, robot.actuated_joint_lower_limits, n, -1)
        self.q_max = _limits(q_max, robot.actuated_joint_upper_limits, n, 1)
        self.v_max = _limits(v_max, robot.actuated_joint_max_velocities, n, 1)
        self.v_min = _limits(v_min, -self.v_max, n, -1)
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
    """Spherical obstacles. Inactive ones are ignored"""

    centers: Array  # (K, 3)
    radii: Array  # (K,)
    active: Array  # (K,) bool


class _SphereConstraint(InequalityConstraint):
    """Helper class for CBFs operating on spherical collision models

    Shared machinery for CBF constraints gain * h(q) >= -dh/dq v on a frax
    robot's collision spheres, with analytic Jacobians

    With max_rows set, only the max_rows closest pairs are kept each step. Far
    pairs can't become active within a step, but too few rows let dropped pairs
    penetrate, so size it for the most simultaneous contacts expected. The
    barrier values are computed for every pair, but the Jacobians only for the
    kept rows
    """

    def __init__(self, robot: Robot, gain: float, penalty: float):
        assert robot.has_collision_data
        self.robot = robot
        # Map from each (flattened, unpadded) sphere to its parent joint
        max_spheres = np.asarray(robot.padded_collision_positions).shape[1]
        self.sphere_joints = np.asarray(robot.collision_slice_indices) // max_spheres
        self.radii = np.asarray(robot.flat_collision_radii)
        self.gain = gain
        self.penalty = penalty

    def _set_num_rows(self, num_candidates: int, max_rows: int | None):
        """Call once the number of candidate rows is known"""
        self.max_rows = (
            None if max_rows is None or max_rows >= num_candidates else max_rows
        )
        self.num_rows = num_candidates if self.max_rows is None else self.max_rows
        assert self.num_rows > 0

    def _keep(self, h):
        """The rows kept, given every candidate's barrier value h: all of them,
        or the max_rows lowest"""
        if self.max_rows is None:
            return slice(None)
        return jax.lax.top_k(-h, self.max_rows)[1]

    @staticmethod
    def _point_jacobians(joint_tfs, Jv, Jw, joints, pos):
        """Linear Jacobians (P, 3, nv) of world points pos (P, 3) fixed to joints (P,)"""
        return point_jacobians(Jv[joints], Jw[joints], pos - joint_tfs[joints, :3, 3])

    @staticmethod
    def _distance_rows(delta, offsets, J_rel):
        """Barrier values |delta| - offsets and their Jacobians n . J_rel, with
        n = delta / |delta|"""
        dist = jnp.linalg.norm(delta, axis=-1)
        n = delta / dist[..., None]
        return dist - offsets, jnp.einsum("...i,...ij->...j", n, J_rel)


@jax.tree_util.register_static
class SelfCollisionConstraint(_SphereConstraint):
    """Self-collision avoidance: h = |p_b - p_a| - r_a - r_b - tol - margin for
    each of the robot's self-collision pairs (including any spheres fixed to the
    world root), with the robot's per-pair tolerances tol. With `pairs` set, the
    sphere pairs are instead every sphere on one joint against every sphere on
    another, for each pair of joints

    Args:
        robot: frax robot, with collision data (and self-collision data,
            if pairs is None)
        pairs (Sequence[tuple[int, int, float]] | None): (joint a, joint b,
            tol) for each pair of joints to check. Defaults to the robot's pairs
        max_rows (int | None): If set, keep only this many closest pairs per step
        gain (float): CBF gain (1/s): how fast the spheres may approach each other
        penalty (float): Relaxation penalty
        margin (float): Safety distance kept between every pair, on top of tol
    """

    def __init__(
        self,
        robot: Robot,
        pairs: Sequence[tuple[int, int, float]] | None = None,
        max_rows: int | None = None,
        gain: float = 10.0,
        penalty: float = DEFAULT_PENALTY,
        margin: float = 0.0,
    ):
        super().__init__(robot, gain, penalty)
        if pairs is None:
            assert robot.has_sc_data
            sphere_pairs = np.asarray(robot.body_sc_pairs)
            tols = np.asarray(robot.body_sc_tols)
        else:
            sphere_pairs, tols = [], []
            for joint_a, joint_b, tol in pairs:
                a = np.flatnonzero(self.sphere_joints == joint_a)
                b = np.flatnonzero(self.sphere_joints == joint_b)
                assert len(a) and len(b), f"No spheres on joint {joint_a} or {joint_b}"
                sphere_pairs += [(i, j) for i in a for j in b]
                tols += [tol] * (len(a) * len(b))
            sphere_pairs, tols = np.array(sphere_pairs), np.array(tols)
        radii = self.radii
        self.margin = margin
        offsets = radii[sphere_pairs[:, 0]] + radii[sphere_pairs[:, 1]] + tols + margin
        # Spheres fixed to the world root only pair through the robot's own
        # pairs. They are indexed after the robot's spheres, as fixed points
        self.root = pairs is None and robot.has_root_collision_data
        if self.root:
            root_pairs = np.asarray(robot.root_sc_pairs)
            self.root_positions = np.asarray(robot.root_collision_positions)
            root_offsets = (
                np.asarray(robot.root_collision_radii)[root_pairs[:, 0]]
                + radii[root_pairs[:, 1]]
                + np.asarray(robot.root_sc_tols)
                + margin
            )
            root_pairs = np.stack([len(radii) + root_pairs[:, 0], root_pairs[:, 1]], 1)
            sphere_pairs = np.concatenate([sphere_pairs, root_pairs])
            offsets = np.concatenate([offsets, root_offsets])
        # Sphere indices (the robot's, then the root's) and offsets, per pair
        self.pairs_a, self.pairs_b = sphere_pairs[:, 0], sphere_pairs[:, 1]
        self.pair_offsets = offsets
        self._set_num_rows(len(sphere_pairs), max_rows)

    def _positions(self, joint_tfs):
        """World positions of the robot's collision spheres, then the root's"""
        pos = self.robot._link_collision_positions(joint_tfs)
        return jnp.concatenate([pos, self.root_positions]) if self.root else pos

    def _jacobians(self, joint_tfs, Jv, Jw, pos, spheres):
        """Linear Jacobians (P, 3, nv) of spheres (P,), indexed as in pairs_a"""
        if not self.root:
            joints = jnp.asarray(self.sphere_joints)[spheres]
            return self._point_jacobians(joint_tfs, Jv, Jw, joints, pos[spheres])
        # The root's spheres don't move: zero Jacobians
        on_robot = spheres < len(self.radii)
        joints = jnp.asarray(self.sphere_joints)[jnp.where(on_robot, spheres, 0)]
        J = self._point_jacobians(joint_tfs, Jv, Jw, joints, pos[spheres])
        return jnp.where(on_robot[:, None, None], J, 0.0)

    def barrier(self, q) -> Array:
        """Barrier values h of every pair, shape (num_pairs,)"""
        pos = self._positions(self.robot.joint_to_world_transforms(q))
        dist = jnp.linalg.norm(pos[self.pairs_b] - pos[self.pairs_a], axis=-1)
        return dist - self.pair_offsets

    def rows(self, q, ref):
        joint_tfs = self.robot.joint_to_world_transforms(q)
        pos = self._positions(joint_tfs)
        delta = pos[self.pairs_b] - pos[self.pairs_a]
        keep = self._keep(jnp.linalg.norm(delta, axis=-1) - self.pair_offsets)
        a, b = jnp.asarray(self.pairs_a)[keep], jnp.asarray(self.pairs_b)[keep]
        Jv, Jw = self.robot._joint_jacobians(joint_tfs)
        J_rel = self._jacobians(joint_tfs, Jv, Jw, pos, b) - self._jacobians(
            joint_tfs, Jv, Jw, pos, a
        )
        offsets = jnp.asarray(self.pair_offsets)[keep]
        h, dh = self._distance_rows(delta[keep], offsets, J_rel)
        return -dh, self.gain * h


@jax.tree_util.register_static
class ObstacleConstraint(_SphereConstraint):
    """Sphere obstacle avoidance: h = |p_i - c_k| - r_i - R_k - margin for each
    robot sphere i in sphere_idxs and obstacle k

    Args:
        robot: frax robot, with collision data
        obstacles (SphereObstacles): Obstacles, which also fix the obstacle count K
        sphere_idxs (ArrayLike | None): Indices of the robot's collision spheres
            to check. Defaults to all
        ref_key (str | None): If set, ref[ref_key] (a SphereObstacles with the
            same K) replaces `obstacles` on each step, for moving obstacles
        max_rows (int | None): If set, keep only this many closest pairs per step
        gain (float): CBF gain (1/s): how fast the spheres may approach the obstacles
        penalty (float): Relaxation penalty
        margin (float): Safety distance kept from every obstacle
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
        # Each row's sphere, with rows ordered (obstacle, sphere)
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
        keep = self._keep(jnp.where(active, dist - offsets, _INACTIVE_DISTANCE))
        spheres = jnp.asarray(self.row_spheres)[keep]
        Jv, Jw = self.robot._joint_jacobians(joint_tfs)
        joints = jnp.asarray(self.sphere_joints)[spheres]
        J = self._point_jacobians(joint_tfs, Jv, Jw, joints, pos[spheres])
        h, dh = self._distance_rows(delta[keep], offsets[keep], J)
        active = active[keep]
        h = jnp.where(active, h, _INACTIVE_DISTANCE)
        dh = jnp.where(active[:, None], dh, 0.0)
        return -dh, self.gain * h


@jax.tree_util.register_static
class FloorConstraint(_SphereConstraint):
    """Floor at z = 0: h = z_i - r_i - margin for each robot collision sphere i,
    and h = z_k - margin for each point k fixed to a joint frame (e.g. a foot's
    sole outline, where the spheres are too coarse)

    Args:
        robot: frax robot, with collision data
        exclude_joints (ArrayLike): Joints whose collision spheres aren't
            checked (e.g. feet whose spheres overhang the sole)
        points (Sequence[tuple[int, ArrayLike]]): (joint index, offset (3,) in
            the joint frame) for each extra point
        max_rows (int | None): If set, keep only this many lowest rows per step
        gain (float): CBF gain (1/s): how fast the robot may approach the floor
        penalty (float): Relaxation penalty
        margin (float): Height kept above the floor
    """

    def __init__(
        self,
        robot: Robot,
        exclude_joints: ArrayLike = (),
        points: Sequence[tuple[int, ArrayLike]] = (),
        max_rows: int | None = None,
        gain: float = 10.0,
        penalty: float = DEFAULT_PENALTY,
        margin: float = 0.0,
    ):
        super().__init__(robot, gain, penalty)
        self.margin = margin
        self.sphere_idxs = np.flatnonzero(~np.isin(self.sphere_joints, exclude_joints))
        self.point_joints = np.array([j for j, _ in points], dtype=int)
        self.point_offsets = np.array([o for _, o in points], dtype=float).reshape(
            -1, 3
        )
        # Each row's joint and radius: the checked spheres, then the points
        self.row_joints = np.concatenate(
            [self.sphere_joints[self.sphere_idxs], self.point_joints]
        )
        self.row_radii = np.concatenate(
            [self.radii[self.sphere_idxs], np.zeros(len(self.point_joints))]
        )
        self._set_num_rows(len(self.row_joints), max_rows)

    def _heights(self, joint_tfs) -> tuple[Array, Array]:
        """World positions (rows, 3) of the checked spheres' centers, then the
        points, and their barrier values h (rows,)"""
        spheres = self.robot._link_collision_positions(joint_tfs)[self.sphere_idxs]
        T = joint_tfs[self.point_joints]
        points = T[:, :3, 3] + jnp.einsum(
            "pij,pj->pi", T[:, :3, :3], self.point_offsets
        )
        pos = jnp.concatenate([spheres, points])
        return pos, pos[:, 2] - self.row_radii - self.margin

    def barrier(self, q) -> Array:
        """Barrier values h, spheres then points, shape (spheres + points,)"""
        return self._heights(self.robot.joint_to_world_transforms(q))[1]

    def rows(self, q, ref):
        joint_tfs = self.robot.joint_to_world_transforms(q)
        pos, h = self._heights(joint_tfs)
        keep = self._keep(h)
        Jv, Jw = self.robot._joint_jacobians(joint_tfs)
        joints = jnp.asarray(self.row_joints)[keep]
        J = self._point_jacobians(joint_tfs, Jv, Jw, joints, pos[keep])
        return -J[:, 2], self.gain * h[keep]


@jax.tree_util.register_static
class BarrierConstraint(InequalityConstraint):
    """Kinematic CBF on a user-defined barrier h(q, ref) >= 0:
    dh/dq E(q) v >= -gain * h

    The Jacobian is by forward-mode autodiff, so this suits custom barriers with a handful
    of rows. Prefer analytic Jacobians (e.g. the sphere constraints) for many

    Args:
        robot: frax robot
        h_fn (Callable): (q, ref) -> h, shape (num_rows,). The safe set is h >= 0
        num_rows (int): Number of barriers
        gain (float): CBF gain (1/s): how fast the boundary may be approached
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
    """Kinematic equality on a user-defined residual e(q, ref) = 0, held with
    error dynamics de/dq E(q) v = -gain * e

    The Jacobian is computed with autodiff (as in BarrierConstraint)

    Args:
        robot: frax robot
        residual_fn (Callable): (q, ref) -> e, shape (num_rows,)
        num_rows (int): Residual dimension
        gain (float): Gain (1/s) at which a violated equality is restored
        elastic (bool): Relax with an L1 penalty instead of holding exactly
        penalty (ArrayLike): Scalar or per-row relaxation penalty
            (if elastic)
    """

    def __init__(
        self,
        robot: Robot,
        residual_fn: Callable[[Array, object], Array],
        num_rows: int,
        gain: float = 10.0,
        elastic: bool = True,
        penalty: ArrayLike = DEFAULT_PENALTY,
    ):
        self.robot = robot
        self.residual_fn = residual_fn
        self.num_rows = num_rows
        self.gain = gain
        self.elastic = elastic
        self.penalty = penalty

    def rows(self, q, ref):
        def e_fn(q):
            e = self.residual_fn(q, ref)
            assert e.shape == (self.num_rows,), (
                f"e has shape {e.shape}, expected ({self.num_rows},)"
            )
            return e

        e, de = _velocity_jacobian(self.robot, e_fn, q)
        return de, -self.gain * e
