import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from frax.ik import (
    EETask,
    FrameTask,
    JointTask,
)
from frax.ik.tasks import _velocity_jacobian
from frax.utils.rotation_utils import Rx, Rz


def test_frame_task_jacobian(g1, g1_q_random):
    """The task Jacobian's position rows match autodiff of the tracked frame positions"""
    idxs = np.array(
        [g1.joint_name_to_index[n] for n in ("left_knee_joint", "waist_pitch_joint")]
    )
    task = FrameTask(g1, idxs)
    ref = {"frame_pos": np.zeros((2, 3)), "frame_rmats": np.tile(np.eye(3), (2, 1, 1))}
    J, _, _ = task.linearize(g1_q_random, ref)
    Jq = jax.jacfwd(lambda q: g1.joint_to_world_transforms(q)[idxs, :3, 3])(g1_q_random)
    Jv = Jq @ g1.velocity_to_qdot_map(g1_q_random)
    np.testing.assert_allclose(J.reshape(2, 6, -1)[:, :3], Jv, atol=1e-8)


def _frame_twists(tf_fn, robot, q, v):
    """World-frame linear and angular velocities (n, 6) of frames tf_fn(q)
    (n, 4, 4) along the integrated motion, d/dt tf_fn(integrate(q, v, t)) at t = 0"""
    tfs, dtfs = jax.jvp(lambda t: tf_fn(robot.integrate(q, v, t)), (0.0,), (1.0,))
    W = dtfs[:, :3, :3] @ jnp.swapaxes(tfs[:, :3, :3], 1, 2)  # skew(w)
    w = jnp.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], axis=1)
    return jnp.concatenate([dtfs[:, :3, 3], w], axis=1)


def test_frame_task_offsets(g1, g1_q_random):
    """Offset frames: targets are the offset poses, and the full Jacobian (linear
    and angular rows) matches the velocity of the offset frames"""
    idxs = np.array(
        [
            g1.joint_name_to_index[n]
            for n in ("left_wrist_yaw_joint", "right_ankle_roll_joint")
        ]
    )
    offsets = np.tile(np.eye(4), (2, 1, 1))
    offsets[0, :3, :3] = Rz(0.4) @ Rx(-0.3)
    offsets[:, :3, 3] = [[0.1, -0.02, 0.05], [0.0, 0.03, -0.04]]

    def tf_fn(q):
        return g1.joint_to_world_transforms(q)[idxs] @ offsets

    tfs = np.asarray(tf_fn(g1_q_random))
    ref = {"frame_pos": tfs[:, :3, 3], "frame_rmats": tfs[:, :3, :3]}
    task = FrameTask(g1, idxs, gain_pos=2.0, gain_rot=3.0, offsets=offsets)
    J, v_des, _ = task.linearize(jnp.asarray(g1_q_random), ref)
    np.testing.assert_allclose(v_des, 0.0, atol=1e-12)  # Reference = current pose

    v = np.random.default_rng(4).standard_normal(g1.nv)
    twists = _frame_twists(tf_fn, g1, jnp.asarray(g1_q_random), v)
    np.testing.assert_allclose(J @ v, twists.reshape(-1), atol=1e-10)


def test_frame_task_one_component(g1, g1_q_random):
    """A position- or rotation-only task is the full pose task's position or
    rotation rows, and reads only its own reference key"""
    idxs = np.array([g1.joint_name_to_index["left_knee_joint"], 5])
    pos = np.random.default_rng(6).standard_normal((2, 3))
    rmats = Rotation.random(2, random_state=7).as_matrix()
    kwargs = {"gain_pos": 2.0, "gain_rot": 3.0, "weight_pos": 4.0, "weight_rot": 5.0}
    full = FrameTask(g1, idxs, **kwargs).linearize(
        g1_q_random, {"frame_pos": pos, "frame_rmats": rmats}
    )
    for task, ref, rows in (
        (FrameTask(g1, idxs, **kwargs, rot_key=None), {"frame_pos": pos}, slice(0, 3)),
        (
            FrameTask(g1, idxs, **kwargs, pos_key=None),
            {"frame_rmats": rmats},
            slice(3, 6),
        ),
    ):
        assert task.num_rows == 6
        for x, x_full in zip(task.linearize(g1_q_random, ref), full, strict=True):
            x_full = x_full.reshape(2, 6, *x_full.shape[1:])[:, rows]
            np.testing.assert_allclose(x, x_full.reshape(x.shape))
    with pytest.raises(AssertionError, match="Track position"):
        FrameTask(g1, idxs, pos_key=None, rot_key=None)


def test_ee_task_matches_frax_ee(panda):
    """EETask tracks frax's EE (with its ee_offset): the error matches the
    ResidualTask formulation, and the Jacobian matches the EE velocity"""
    q = jnp.asarray(panda.default_configuration + 0.3)
    tf = np.asarray(panda.ee_transform(q))
    ref = {"ee_pos": tf[:3, 3] + [0.1, 0.0, -0.05], "ee_rmat": tf[:3, :3] @ Rx(0.2)}
    task = EETask(panda, gain_pos=10.0, gain_rot=5.0, weight_rot=0.5)
    J, v_des, w = task.linearize(q, ref)
    np.testing.assert_allclose(v_des[:3], -10.0 * (tf[:3, 3] - ref["ee_pos"]))
    np.testing.assert_allclose(w, [1.0] * 3 + [0.5] * 3)
    v = np.random.default_rng(5).standard_normal(panda.nv)
    twist = _frame_twists(lambda q: panda.ee_transform(q)[None], panda, q, v)
    np.testing.assert_allclose(J @ v, twist[0], atol=1e-10)


def test_velocity_jacobian_quaternion_base(g1, g1_q_random):
    """Autodiff Jacobians w.r.t. v match the derivative along the integrated motion,
    d/dt f(integrate(q, v, t)) at t = 0"""
    links = np.array([5, 12, 27])  # Root, and two links further down the tree

    def f(q):
        tfs = g1.link_to_world_transforms(q)[links]
        return jnp.concatenate([tfs[:, :3, :3].reshape(-1), tfs[:, :3, 3].reshape(-1)])

    v = np.random.default_rng(3).standard_normal(g1.nv)
    y, J = _velocity_jacobian(g1, f, jnp.asarray(g1_q_random))
    _, dy = jax.jvp(lambda t: f(g1.integrate(g1_q_random, v, t)), (0.0,), (1.0,))
    np.testing.assert_allclose(y, f(g1_q_random), atol=1e-12)
    np.testing.assert_allclose(J @ v, dy, atol=1e-10)


def test_joint_task_quaternion_base(g1, g1_q_random):
    """JointTask indexes v, and reads the matching entries of q past the quaternion"""
    knee = g1.joint_name_to_index["left_knee_joint"]
    task = JointTask(g1, [2, knee], "q")
    J, v_des, _ = task.linearize(g1_q_random, {"q": np.zeros(2)})
    np.testing.assert_allclose(J, np.eye(g1.nv)[[2, knee]])
    np.testing.assert_allclose(v_des, -g1_q_random[[2, knee + 1]])
    with pytest.raises(AssertionError, match="q_dot = v"):
        JointTask(g1, [3], "q")
