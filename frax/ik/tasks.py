"""Differential IK tasks

Notes:
- Tasks operate on the velocity level, as opposed to a delta-q.
- In general, a task has a least-squares structure
  || weight * (J v - v_des) ||^2
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from frax.core.manipulator import Manipulator
from frax.core.robot import Robot
from frax.utils.rotation_utils import orientation_error_quat
from frax.utils.spatial_utils import offset_transforms_and_jacobians


def _velocity_jacobian(robot: Robot, fn, q: Array) -> tuple[Array, Array]:
    """y = fn(q), shape (m,), and its Jacobian with respect to v, shape (m, nv)"""
    E = robot.velocity_to_qdot_map(q).astype(q.dtype)
    y, jvp_fn = jax.linearize(fn, q)
    return y, jax.vmap(jvp_fn, out_axes=1)(E.T)


class Task:
    """Base differential IK task"""

    robot: Robot
    num_rows: int

    def linearize(self, q: Array, ref) -> tuple[Array, Array, Array]:
        """Linearize the task about q

        Args:
            q (Array): Configuration, shape (nq,)
            ref (Any): Per-step reference data (any pytree)

        Returns:
            J (Array): Task Jacobian, shape (num_rows, nv)
            v_des (Array): Desired task velocity, shape (num_rows,)
            weight (Array): Per-row weight, shape (num_rows,)
        """
        raise NotImplementedError


def _per_row(x: ArrayLike, num_rows: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(x, dtype=float), (num_rows,)).copy()


@jax.tree_util.register_static
class ResidualTask(Task):
    """Drive a residual e(q, ref) to zero, with a forward-mode autodiff Jacobian

    Args:
        robot (Robot): Robot model
        residual_fn (Callable): (q, ref) -> e, shape (num_rows,)
        num_rows (int): Residual dimension
        gain (ArrayLike): Scalar or per-row gain (1/s)
        weight (ArrayLike): Scalar or per-row weight
    """

    def __init__(
        self,
        robot: Robot,
        residual_fn: Callable[[Array, object], Array],
        num_rows: int,
        gain: ArrayLike = 1.0,
        weight: ArrayLike = 1.0,
    ):
        self.robot = robot
        self.residual_fn = residual_fn
        self.num_rows = num_rows
        self.gain = _per_row(gain, num_rows)
        self.weight = _per_row(weight, num_rows)

    def linearize(self, q, ref):
        e, J = _velocity_jacobian(self.robot, lambda q: self.residual_fn(q, ref), q)
        return J, -self.gain * e, self.weight


@jax.tree_util.register_static
class JointTask(Task):
    """Track a joint-space task

    Note: this is currently configured to only deal with actuated
    joints. For free-floating bases, use a FrameTask

    Args:
        robot (Robot): Robot model
        idxs (ArrayLike): Indices into v to track
        ref_key (str): Reference key for the targets, shape (len(idxs),)
        gain (ArrayLike): Scalar or per-joint gain (1/s)
        weight (ArrayLike): Scalar or per-joint weight
    """

    def __init__(
        self,
        robot: Robot,
        idxs: ArrayLike,
        ref_key: str,
        gain: ArrayLike = 1.0,
        weight: ArrayLike = 1.0,
    ):
        self.robot = robot
        self.idxs = np.asarray(idxs)
        v_to_q = robot.velocity_to_configuration_index
        assert all(int(i) in v_to_q for i in self.idxs), (
            "JointTask entries must have q_dot = v (not a quaternion base's orientation)"
        )
        self.q_idxs = np.array([v_to_q[int(i)] for i in self.idxs], dtype=int)
        self.J = np.eye(robot.nv)[self.idxs]
        self.ref_key = ref_key
        self.num_rows = len(self.idxs)
        self.gain = _per_row(gain, self.num_rows)
        self.weight = _per_row(weight, self.num_rows)

    def linearize(self, q, ref):
        e = q[self.q_idxs] - ref[self.ref_key]
        return self.J, -self.gain * e, self.weight


# TODO: Come back to this pose/frame/ee task structure again in the future.
# I feel like this isn't the clearest way of writing these


class _PoseTask(Task):
    """Helper base for tasks tracking frame positions and/or orientations"""

    def __init__(
        self, robot, n, gain_pos, gain_rot, weight_pos, weight_rot, pos_key, rot_key
    ):
        assert pos_key is not None or rot_key is not None, (
            "Track position, rotation, or both"
        )
        self.robot = robot
        self.pos_key, self.rot_key = pos_key, rot_key
        self.gain_pos = _per_row(gain_pos if pos_key is not None else 0.0, n)[:, None]
        self.gain_rot = _per_row(gain_rot if rot_key is not None else 0.0, n)[:, None]
        weights = [
            np.repeat(_per_row(weight, n)[:, None], 3, 1)
            for weight, key in ((weight_pos, pos_key), (weight_rot, rot_key))
            if key is not None
        ]
        self.weight = np.concatenate(weights, axis=1).reshape(-1)
        self.num_rows = len(self.weight)

    def _pose_rows(self, tfs: Array, J: Array, ref) -> tuple[Array, Array, Array]:
        """Task rows for frame transforms tfs (n, 4, 4), with Jacobians J (n, 6, nv)"""
        pos, rot = self.pos_key is not None, self.rot_key is not None
        v_des = []
        if pos:
            v_des.append(-self.gain_pos * (tfs[:, :3, 3] - ref[self.pos_key]))
        if rot:
            rot_error = jax.vmap(orientation_error_quat)(
                tfs[:, :3, :3], ref[self.rot_key]
            )
            v_des.append(-self.gain_rot * rot_error)
        J = J[:, slice(0 if pos else 3, 6 if rot else 3)]
        return (
            J.reshape(self.num_rows, -1),
            jnp.concatenate(v_des, axis=1).reshape(-1),
            self.weight,
        )


@jax.tree_util.register_static
class FrameTask(_PoseTask):
    """Track world-frame positions and/or orientations of joint frames

    Args:
        robot (Robot): Robot model
        joints (ArrayLike): Indices of the tracked joints, shape (n,)
        gain_pos, gain_rot (ArrayLike): Scalar or per-frame gains (1/s)
        weight_pos, weight_rot (ArrayLike): Scalar or per-frame weights
        pos_key (str | None): Reference key for target positions, shape (n, 3).
            None to not track position
        rot_key (str | None): Reference key for target rotation matrices,
            shape (n, 3, 3). None to not track rotation
        offsets (ArrayLike | None): Tracked frames in their joint frames, shape
            (4, 4) or (n, 4, 4). Defaults to None (identity)
    """

    def __init__(
        self,
        robot: Robot,
        joints: ArrayLike,
        gain_pos: ArrayLike = 1.0,
        gain_rot: ArrayLike = 1.0,
        weight_pos: ArrayLike = 1.0,
        weight_rot: ArrayLike = 1.0,
        pos_key: str | None = "frame_pos",
        rot_key: str | None = "frame_rmat",
        offsets: ArrayLike | None = None,
    ):
        self.joints = np.asarray(joints)
        n = len(self.joints)
        super().__init__(
            robot, n, gain_pos, gain_rot, weight_pos, weight_rot, pos_key, rot_key
        )
        if offsets is not None:
            offsets = np.broadcast_to(
                np.asarray(offsets, dtype=float), (n, 4, 4)
            ).copy()
        self.offsets = offsets

    def linearize(self, q, ref):
        tfs = self.robot.joint_to_world_transforms(q)
        Jv, Jw = self.robot._joint_jacobians(tfs)
        tfs, Jv, Jw = tfs[self.joints], Jv[self.joints], Jw[self.joints]
        if self.offsets is not None:
            tfs, Jv, Jw = offset_transforms_and_jacobians(tfs, Jv, Jw, self.offsets)
        return self._pose_rows(tfs, jnp.concatenate([Jv, Jw], axis=1), ref)


@jax.tree_util.register_static
class EETask(_PoseTask):
    """Track the world-frame position and/or orientation of a Manipulator's end
    effector (including its `ee_offset`)

    Args:
        robot (Manipulator): Manipulator model
        gain_pos, gain_rot (float): Gains (1/s)
        weight_pos, weight_rot (float): Weights
        pos_key (str | None): Reference key for the target position, shape (3,).
            None to not track position
        rot_key (str | None): Reference key for the target rotation matrix,
            shape (3, 3). None to not track rotation
    """

    def __init__(
        self,
        robot: Manipulator,
        gain_pos: float = 1.0,
        gain_rot: float = 1.0,
        weight_pos: float = 1.0,
        weight_rot: float = 1.0,
        pos_key: str | None = "ee_pos",
        rot_key: str | None = "ee_rmat",
    ):
        super().__init__(
            robot, 1, gain_pos, gain_rot, weight_pos, weight_rot, pos_key, rot_key
        )

    def linearize(self, q, ref):
        joint_tfs = self.robot.joint_to_world_transforms(q)
        tf = self.robot._ee_transform(joint_tfs)
        J = self.robot._ee_jacobian(joint_tfs)
        keys = (key for key in (self.pos_key, self.rot_key) if key is not None)
        ref = {key: ref[key][None] for key in keys}
        return self._pose_rows(tf[None], J[None], ref)
