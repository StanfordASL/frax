"""Test cases for the Quadruped class, using the Unitree A2 with a quaternion floating base

See test_quaternion_floating_base.py for details on the conversion between the frax (MuJoCo)
and Pinocchio floating base conventions
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import pinocchio as pin
import pytest

from frax.robots.unitree_a2 import load_a2
from frax.assets import A2_ASSETS_DIR
from frax.utils.rotation_utils import quat_wxyz_to_rmat, wxyz_to_xyzw

urdf = A2_ASSETS_DIR / "a2.urdf"
mjcf = A2_ASSETS_DIR / "a2.xml"

NUM_ACTUATED = 12
NV = NUM_ACTUATED + 6
FEET = ("front_left", "front_right", "hind_left", "hind_right")
PIN_FOOT_FRAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

g = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])


def sample_state(nv: int = NV) -> Tuple[np.ndarray, np.ndarray]:
    """Random configuration (nq = nv + 1) and velocity (nv) for the quaternion A2"""
    pos = np.random.uniform(-1.0, 1.0, 3)
    quat = np.random.randn(4)
    quat /= np.linalg.norm(quat)
    q_act = np.random.uniform(-np.pi / 2, np.pi / 2, nv - 6)
    q = np.concatenate([pos, quat, q_act])
    qd = np.random.uniform(-1.0, 1.0, nv)
    return q, qd


def frax_to_pinocchio(q: np.ndarray, qd: np.ndarray, perm: np.ndarray):
    """Convert quaternion-based frax (MuJoCo convention) state to Pinocchio's freeflyer convention

    Note: Pinocchio orders sibling joints alphabetically (FL, FR, RL, RR), whereas frax follows
    the URDF/MJCF ordering (FL, RL, FR, RR). perm maps Pinocchio's actuated joint indices to frax's

    Returns:
        q_pin: Pinocchio configuration (XYZW quaternion), shape (nq,)
        v_pin: Pinocchio velocity (body-frame linear and angular velocity), shape (nv,)
        T: Velocity transform, v_pin = T @ qd, shape (nv, nv)
        Tdot_qd: T_dot @ qd, shape (nv,)
    """
    R = np.asarray(quat_wxyz_to_rmat(q[3:7]))
    q_pin = np.concatenate([q[:3], np.asarray(wxyz_to_xyzw(q[3:7])), q[7:][perm]])
    T = np.zeros((len(qd), len(qd)))
    T[:3, :3] = R.T
    T[3:6, 3:6] = np.eye(3)
    T[6 + np.arange(len(perm)), 6 + perm] = 1.0
    v_pin = T @ qd
    # d/dt(R^T) = -skew(omega_body) @ R^T
    Tdot_qd = np.zeros(len(qd))
    Tdot_qd[:3] = -np.cross(qd[3:6], v_pin[:3])
    return q_pin, v_pin, T, Tdot_qd


@pytest.fixture(scope="module")
def robot():
    return load_a2()


@pytest.fixture(scope="module")
def pin_model_and_data():
    model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
    return model, pin.Data(model)


@pytest.fixture(scope="module")
def mj_model_and_data():
    """MuJoCo model and data, with joint armature and damping (not modeled in the URDF) removed"""
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    model.dof_armature[:] = 0.0
    model.dof_damping[:] = 0.0
    return model, mujoco.MjData(model)


@pytest.fixture(scope="module")
def perm(robot, pin_model_and_data):
    """Actuated joint permutation: q_pin[7:] = q_frax[7:][perm]"""
    model, _ = pin_model_and_data
    return np.array([robot.joint_names.index(name) - 6 for name in model.names[2:]])


def set_mj_state(model, data, q, qd, tau=None):
    data.qpos[:] = q
    data.qvel[:] = qd
    data.qfrc_applied[:] = 0.0 if tau is None else tau
    mujoco.mj_forward(model, data)


class TestVsPinocchio:
    """Compare the A2 quadruped against Pinocchio with a freeflyer joint"""

    def test_dimensions(self, robot, pin_model_and_data):
        model, _ = pin_model_and_data
        assert robot.nq == model.nq
        assert robot.nv == model.nv
        assert robot.num_actuated_joints == NUM_ACTUATED
        assert sorted(robot.joint_names[6:]) == sorted(model.names[2:])

    def test_mass_matrix(self, robot, pin_model_and_data, perm):
        model, data = pin_model_and_data
        mass_matrix = jax.jit(robot.mass_matrix)
        for _ in range(10):
            q, qd = sample_state()
            q_pin, _, T, _ = frax_to_pinocchio(q, qd, perm)
            M_pin = pin.crba(model, data, q_pin)
            M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T
            np.testing.assert_allclose(mass_matrix(q), T.T @ M_pin @ T, atol=1e-8)

    def test_nonlinear_bias(self, robot, pin_model_and_data, perm):
        model, data = pin_model_and_data
        bias = jax.jit(robot.nonlinear_bias)
        gravity = jax.jit(robot.gravity_vector)
        cc = jax.jit(robot.centrifugal_coriolis_vector)
        for _ in range(10):
            q, qd = sample_state()
            q_pin, v_pin, T, Tdot_qd = frax_to_pinocchio(q, qd, perm)
            # tau = T^T (M_pin (T qdd + T_dot qd) + b_pin), with qdd = 0
            tau_pin = pin.rnea(model, data, q_pin, v_pin, Tdot_qd)
            expected = T.T @ tau_pin
            np.testing.assert_allclose(bias(q, qd), expected, atol=1e-8)
            np.testing.assert_allclose(gravity(q) + cc(q, qd), expected, atol=1e-8)

    def test_rnea(self, robot, pin_model_and_data, perm):
        model, data = pin_model_and_data
        rnea = jax.jit(lambda q, qd, qdd: robot.rnea(q, qd, qdd, g, None))
        for _ in range(10):
            q, qd = sample_state()
            qdd = np.random.uniform(-1.0, 1.0, robot.nv)
            q_pin, v_pin, T, Tdot_qd = frax_to_pinocchio(q, qd, perm)
            a_pin = T @ qdd + Tdot_qd
            tau_pin = pin.rnea(model, data, q_pin, v_pin, a_pin)
            np.testing.assert_allclose(rnea(q, qd, qdd), T.T @ tau_pin, atol=1e-8)

    def test_forward_dynamics(self, robot, pin_model_and_data, perm):
        model, data = pin_model_and_data
        fd = jax.jit(lambda q, qd, tau: robot.forward_dynamics(q, qd, tau, None))
        for _ in range(10):
            q, qd = sample_state()
            tau = np.random.uniform(-1.0, 1.0, robot.nv)
            q_pin, v_pin, T, Tdot_qd = frax_to_pinocchio(q, qd, perm)
            # T is orthogonal (a rotation and a permutation), so T^-T = T
            a_pin = pin.aba(model, data, q_pin, v_pin, T @ tau)
            expected = T.T @ (a_pin - Tdot_qd)
            np.testing.assert_allclose(fd(q, qd, tau), expected, rtol=1e-6, atol=1e-6)

    def test_kinematics_and_jacobians(self, robot, pin_model_and_data, perm):
        model, data = pin_model_and_data
        fk = jax.jit(robot.joint_to_world_transforms)
        joint_jacobians = jax.jit(
            lambda q: robot._joint_jacobians(robot.joint_to_world_transforms(q))
        )
        # Pinocchio joint ids for each of frax's bodies (index 5 is the base, i.e. the freeflyer)
        pin_joint_ids = [1] + [model.getJointId(name) for name in robot.joint_names[6:]]
        for _ in range(5):
            q, qd = sample_state()
            q_pin, _, T, _ = frax_to_pinocchio(q, qd, perm)
            pin.forwardKinematics(model, data, q_pin)
            pin.computeJointJacobians(model, data, q_pin)
            tfs = fk(q)
            Jvs, Jws = joint_jacobians(q)
            for my_idx, pin_idx in zip(range(5, robot.nv), pin_joint_ids):
                oMi = data.oMi[pin_idx]
                np.testing.assert_allclose(
                    tfs[my_idx, :3, :3], oMi.rotation, atol=1e-10
                )
                np.testing.assert_allclose(
                    tfs[my_idx, :3, 3], oMi.translation, atol=1e-10
                )
                J_pin = pin.getJointJacobian(
                    model, data, pin_idx, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                J_pin = J_pin @ T
                np.testing.assert_allclose(Jvs[my_idx], J_pin[:3], atol=1e-10)
                np.testing.assert_allclose(Jws[my_idx], J_pin[3:], atol=1e-10)

    def test_feet(self, robot, pin_model_and_data, perm):
        model, data = pin_model_and_data

        @jax.jit
        def feet_data(q):
            tfs = robot.joint_to_world_transforms(q)
            T_feet = [getattr(robot, f"_{f}_foot_transform")(tfs) for f in FEET]
            J_feet = [getattr(robot, f"_{f}_foot_jacobian")(tfs) for f in FEET]
            return T_feet, J_feet

        # The foot offsets coincide with the URDF's (fixed) foot links, which pinocchio keeps as frames
        frame_ids = [model.getFrameId(name) for name in PIN_FOOT_FRAMES]
        for _ in range(5):
            q, qd = sample_state()
            q_pin, _, T, _ = frax_to_pinocchio(q, qd, perm)
            pin.forwardKinematics(model, data, q_pin)
            pin.updateFramePlacements(model, data)
            pin.computeJointJacobians(model, data, q_pin)
            T_feet, J_feet = feet_data(q)
            for fid, T_foot, J_foot in zip(frame_ids, T_feet, J_feet):
                np.testing.assert_allclose(
                    T_foot, data.oMf[fid].homogeneous, atol=1e-10
                )
                J_pin = pin.getFrameJacobian(
                    model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                np.testing.assert_allclose(J_foot, J_pin @ T, atol=1e-10)


class TestVsMujoco:
    """Compare the A2 quadruped against MuJoCo's free joint directly

    Note: Inertial parameters in the MJCF are rounded relative to the URDF, so we use
    tolerances relative to the quantities' magnitudes
    """

    def test_joint_ordering(self, robot, mj_model_and_data):
        model, _ = mj_model_and_data
        mj_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(1, model.njnt)
        ]
        assert mj_names == list(robot.joint_names[6:])
        assert model.nq == robot.nq
        assert model.nv == robot.nv

    def test_mass_matrix(self, robot, mj_model_and_data):
        model, data = mj_model_and_data
        mass_matrix = jax.jit(robot.mass_matrix)
        for _ in range(10):
            q, qd = sample_state()
            set_mj_state(model, data, q, qd)
            M_mj = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, data, M_mj)
            np.testing.assert_allclose(mass_matrix(q), M_mj, atol=1e-4)

    def test_bias(self, robot, mj_model_and_data):
        model, data = mj_model_and_data
        bias = jax.jit(robot.nonlinear_bias)
        for _ in range(10):
            q, qd = sample_state()
            set_mj_state(model, data, q, qd)
            np.testing.assert_allclose(
                bias(q, qd), data.qfrc_bias, rtol=1e-5, atol=1e-3
            )

    def test_forward_dynamics(self, robot, mj_model_and_data):
        model, data = mj_model_and_data
        fd = jax.jit(lambda q, qd, tau: robot.forward_dynamics(q, qd, tau, None))
        for _ in range(10):
            q, qd = sample_state()
            tau = np.random.uniform(-1.0, 1.0, robot.nv)
            set_mj_state(model, data, q, qd, tau)
            # qacc_smooth is the acceleration without any constraint forces (contact, limits)
            qdd = np.asarray(fd(q, qd, tau))
            np.testing.assert_allclose(
                qdd, data.qacc_smooth, atol=1e-4 * np.max(np.abs(qdd))
            )

    def test_kinematics_and_jacobians(self, robot, mj_model_and_data):
        model, data = mj_model_and_data
        fk = jax.jit(robot.joint_to_world_transforms)
        joint_jacobians = jax.jit(
            lambda q: robot._joint_jacobians(robot.joint_to_world_transforms(q))
        )
        for _ in range(5):
            q, qd = sample_state()
            set_mj_state(model, data, q, qd)
            tfs = fk(q)
            Jvs, Jws = joint_jacobians(q)
            for my_idx in range(5, robot.nv):
                body_id = my_idx - 4  # MuJoCo's body 0 is the world, body 1 is the base
                np.testing.assert_allclose(
                    tfs[my_idx, :3, 3], data.xpos[body_id], atol=1e-6
                )
                np.testing.assert_allclose(
                    tfs[my_idx, :3, :3], data.xmat[body_id].reshape(3, 3), atol=1e-5
                )
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
                np.testing.assert_allclose(Jvs[my_idx], jacp, atol=1e-5)
                np.testing.assert_allclose(Jws[my_idx], jacr, atol=1e-5)


@pytest.mark.parametrize("foot", FEET)
def test_feet_jacobians_and_derivatives_autodiff(robot, foot):
    """Check the analytical foot Jacobians and derivatives against autodiff"""
    transform_func = getattr(robot, f"{foot}_foot_transform")
    J_func = getattr(robot, f"{foot}_foot_jacobian")
    J_and_Jdot_func = getattr(robot, f"{foot}_foot_jacobian_and_derivative")
    for _ in range(3):
        q, qd = sample_state()
        E = robot.configuration_velocity_map(q)
        J, Jdot = J_and_Jdot_func(q, qd)
        np.testing.assert_allclose(J, J_func(q), atol=1e-12)
        # Linear part of the Jacobian via autodiff
        Jv_ad = jax.jacobian(lambda q: transform_func(q)[:3, 3])(q) @ E
        np.testing.assert_allclose(J[:3], Jv_ad, atol=1e-10)
        # Jdot via a directional derivative of J along q_dot = E @ qd
        _, Jdot_ad = jax.jvp(J_func, (q,), (E @ qd,))
        np.testing.assert_allclose(Jdot, Jdot_ad, atol=1e-10)
