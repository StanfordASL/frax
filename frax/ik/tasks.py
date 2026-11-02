"""Velocity-level least-squares tasks

Each task linearizes about the current configuration into a set of rows
(J, v_des, weight), and the solver minimizes sum || weight * (J v - v_des) ||^2
over all stacked task rows

Gains are in 1/s, with error dynamics e_dot = -gain * e, so they don't depend on
the frame rate. A step of dt closes a fraction gain * dt of the error: gain * dt
= 1 is a full Gauss-Newton step, and above it the steps overshoot
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
    """Value y = fn(q), shape (m,), and its forward-mode Jacobian with respect
    to v, dy/dq E(q), shape (m, nv), with E the robot's velocity_to_qdot_map"""
    E = robot.velocity_to_qdot_map(q).astype(q.dtype)
    y, jvp_fn = jax.linearize(fn, q)
    return y, jax.vmap(jvp_fn, out_axes=1)(E.T)


class Task:
    """Base task. Subclasses define `robot` (the frax robot whose configuration
    space it acts on), `num_rows` and `linearize`"""

    robot: Robot
    num_rows: int

    def linearize(self, q: Array, ref) -> tuple[Array, Array, Array]:
        """Linearize the task about q

        Args:
            q (Array): Configuration, shape (nq,)
            ref: Reference for the current step (any pytree)

        Returns:
            tuple[Array, Array, Array]:
                J (Array): Task Jacobian, shape (num_rows, nv)
                v_des (Array): Desired task velocity, shape (num_rows,)
                weight (Array): Per-row weight, shape (num_rows,)
        """
        raise NotImplementedError


def _per_row(x: ArrayLike, num_rows: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(x, dtype=float), (num_rows,)).copy()


@jax.tree_util.register_static
class ResidualTask(Task):
    """Drive a residual e(q, ref) to zero with error dynamics e_dot = -gain * e

    The Jacobian is computed with forward-mode autodiff, and mapped from q to v
    by the robot's velocity_to_qdot_map

    Args:
        robot: frax robot
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
    """Track reference values for a subset of configuration entries

    Entries are indexed by velocity (e.g. a frax robot's `joint_name_to_index`),
    and must have a corresponding configuration entry with q_dot = v (see the
    robot's `velocity_to_configuration_index`). For a quaternion floating base,
    that excludes the orientation, which can be tracked with a ResidualTask or
    FrameTask instead

    Args:
        robot: frax robot
        idxs (ArrayLike): Indices into v to track
        ref_key (str): Key of the reference targets for the corresponding entries
            of q, shape (len(idxs),)
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


def _init_pose(task, n, gain_pos, gain_rot, weight_pos, weight_rot, pos_key, rot_key):
    """Set a pose task's gains (n, 1), keys, per-row weights and num_rows. Each
    frame has [position xyz] rows if pos_key is set, then [rotation xyz] rows if
    rot_key is. An untracked component has no rows, and zero gain"""
    assert pos_key is not None or rot_key is not None, (
        "Track position, rotation, or both"
    )
    task.pos_key, task.rot_key = pos_key, rot_key
    task.gain_pos = _per_row(gain_pos if pos_key is not None else 0.0, n)[:, None]
    task.gain_rot = _per_row(gain_rot if rot_key is not None else 0.0, n)[:, None]
    weights = [
        np.repeat(_per_row(weight, n)[:, None], 3, 1)
        for weight, key in ((weight_pos, pos_key), (weight_rot, rot_key))
        if key is not None
    ]
    task.weight = np.concatenate(weights, axis=1).reshape(-1)
    task.num_rows = len(task.weight)


def _pose_rows(task, tfs: Array, J: Array, ref) -> tuple[Array, Array, Array]:
    """Pose tracking rows for frames with transforms tfs (n, 4, 4) and
    Jacobians J (n, 6, nv) (linear, then angular), for the task's tracked
    components"""
    pos, rot = task.pos_key is not None, task.rot_key is not None
    v_des = []
    if pos:
        v_des.append(-task.gain_pos * (tfs[:, :3, 3] - ref[task.pos_key]))
    if rot:
        rot_error = jax.vmap(orientation_error_quat)(tfs[:, :3, :3], ref[task.rot_key])
        v_des.append(-task.gain_rot * rot_error)
    J = J[:, slice(0 if pos else 3, 6 if rot else 3)]
    return (
        J.reshape(task.num_rows, -1),
        jnp.concatenate(v_des, axis=1).reshape(-1),
        task.weight,
    )


@jax.tree_util.register_static
class FrameTask(Task):
    """Track world-frame positions and/or orientations of a set of frax robot
    joint frames (each link's parent joint origin, optionally offset), with
    frax's analytic Jacobians

    Set pos_key or rot_key to None to track only the other component: its rows
    are dropped, and its gain and weight are ignored

    Args:
        robot: frax robot
        joints (ArrayLike): Indices of the joints whose frames are tracked,
            shape (n,)
        gain_pos, gain_rot (ArrayLike): Scalar or per-frame gains (1/s)
        weight_pos, weight_rot (ArrayLike): Scalar or per-frame weights
        pos_key (str | None): Reference key for target positions, shape (n, 3)
        rot_key (str | None): Reference key for target rotation matrices,
            shape (n, 3, 3)
        offsets (ArrayLike | None): Transforms of the tracked frames relative
            to their joint frames, shape (4, 4) (shared) or (n, 4, 4). Defaults
            to identity
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
        rot_key: str | None = "frame_rmats",
        offsets: ArrayLike | None = None,
    ):
        self.robot = robot
        self.joints = np.asarray(joints)
        n = len(self.joints)
        _init_pose(
            self, n, gain_pos, gain_rot, weight_pos, weight_rot, pos_key, rot_key
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
        return _pose_rows(self, tfs, jnp.concatenate([Jv, Jw], axis=1), ref)


@jax.tree_util.register_static
class EETask(Task):
    """Track the world-frame position and/or orientation of a frax
    Manipulator's end effector (including its `ee_offset`), with frax's
    analytic Jacobian

    Set pos_key or rot_key to None to track only the other component (as
    FrameTask)

    Args:
        robot: frax Manipulator
        gain_pos, gain_rot (float): Gains (1/s)
        weight_pos, weight_rot (float): Weights
        pos_key (str | None): Reference key for the target position, shape (3,)
        rot_key (str | None): Reference key for the target rotation matrix,
            shape (3, 3)
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
        self.robot = robot
        _init_pose(
            self, 1, gain_pos, gain_rot, weight_pos, weight_rot, pos_key, rot_key
        )

    def linearize(self, q, ref):
        joint_tfs = self.robot.joint_to_world_transforms(q)
        tf = self.robot._ee_transform(joint_tfs)
        J = self.robot._ee_jacobian(joint_tfs)
        keys = (key for key in (self.pos_key, self.rot_key) if key is not None)
        ref = {key: ref[key][None] for key in keys}
        return _pose_rows(self, tf[None], J[None], ref)
