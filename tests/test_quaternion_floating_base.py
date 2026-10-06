"""Test cases for the quaternion-based floating base representation

The quaternion floating base uses MuJoCo's free joint conventions:
- q = [position (world), WXYZ quaternion (world), actuated joint positions], nq = nv + 1
- v = [linear velocity (world), angular velocity (body), actuated joint velocities]

So, we can compare against MuJoCo directly. For Pinocchio, the freeflyer joint uses
XYZW quaternions and a body-frame linear velocity, so we convert between the two:
    v_pin = T @ v, where T = blockdiag(R^T, I)
which implies M_frax = T^T M_pin T, and tau_frax = T^T tau_pin.
Note that T depends on the state, so T_dot @ v must be accounted for in the accelerations.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import pinocchio as pin
import pytest

from frax.core.manipulator import Manipulator
from frax.robots.unitree_g1 import load_g1
from frax.assets import G1_ASSETS_DIR, FRANKA_ASSETS_DIR
from frax.utils.free_floating_utils import pose_and_twist_to_virtual_joints
from frax.utils.rotation_utils import (
    quat_wxyz_to_rmat,
    rotate_vector_by_quat_wxyz,
    wxyz_to_xyzw,
)

fixed_root_urdf = G1_ASSETS_DIR / "g1_29dof_rev_1_0.urdf"
mjcf = G1_ASSETS_DIR / "g1_29dof_rev_1_0.xml"

NUM_ACTUATED = 29

g = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])


def sample_state(nv: int = 35) -> Tuple[np.ndarray, np.ndarray]:
    """Random configuration (nq = nv + 1) and velocity (nv) for the quaternion G1"""
    pos = np.random.uniform(-1.0, 1.0, 3)
    quat = np.random.randn(4)
    quat /= np.linalg.norm(quat)
    q_act = np.random.uniform(-np.pi / 2, np.pi / 2, nv - 6)
    q = np.concatenate([pos, quat, q_act])
    v = np.random.uniform(-1.0, 1.0, nv)
    return q, v


def frax_to_pinocchio(q: np.ndarray, v: np.ndarray):
    """Convert quaternion-based frax (MuJoCo convention) state to Pinocchio's freeflyer convention

    Returns:
        q_pin: Pinocchio configuration (XYZW quaternion), shape (nq,)
        v_pin: Pinocchio velocity (body-frame linear and angular velocity), shape (nv,)
        T: Velocity transform, v_pin = T @ v, shape (nv, nv)
        Tdot_v: T_dot @ v, shape (nv,)
    """
    R = np.asarray(quat_wxyz_to_rmat(q[3:7]))
    q_pin = np.concatenate([q[:3], np.asarray(wxyz_to_xyzw(q[3:7])), q[7:]])
    T = np.eye(len(v))
    T[:3, :3] = R.T
    v_pin = T @ v
    # d/dt(R^T) = -skew(omega_body) @ R^T
    Tdot_v = np.zeros(len(v))
    Tdot_v[:3] = -np.cross(v[3:6], v_pin[:3])
    return q_pin, v_pin, T, Tdot_v


@pytest.fixture(scope="module")
def robot():
    return load_g1(floating_base="quaternion")


@pytest.fixture(scope="module")
def euler_robot():
    return load_g1(floating_base="euler")


@pytest.fixture(scope="module")
def pin_model_and_data():
    model = pin.buildModelFromUrdf(fixed_root_urdf, pin.JointModelFreeFlyer())
    return model, pin.Data(model)


@pytest.fixture(scope="module")
def mj_model_and_data():
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    return model, mujoco.MjData(model)


def set_mj_state(model, data, q, v, tau=None):
    data.qpos[:] = q
    data.qvel[:] = v
    data.qfrc_applied[:] = 0.0 if tau is None else tau
    mujoco.mj_forward(model, data)


class TestVsPinocchio:
    """Compare the quaternion floating base against Pinocchio's freeflyer joint"""

    def test_dimensions(self, robot, pin_model_and_data):
        model, _ = pin_model_and_data
        assert robot.nq == model.nq
        assert robot.nv == model.nv
        assert robot.num_actuated_joints == NUM_ACTUATED

    def test_mass_matrix(self, robot, pin_model_and_data):
        model, data = pin_model_and_data
        mass_matrix = jax.jit(robot.mass_matrix)
        for _ in range(10):
            q, v = sample_state()
            q_pin, _, T, _ = frax_to_pinocchio(q, v)
            M_pin = pin.crba(model, data, q_pin)
            M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T
            np.testing.assert_allclose(mass_matrix(q), T.T @ M_pin @ T, atol=1e-8)

    def test_nonlinear_bias(self, robot, pin_model_and_data):
        model, data = pin_model_and_data
        bias = jax.jit(robot.nonlinear_bias)
        gravity = jax.jit(robot.gravity_vector)
        cc = jax.jit(robot.centrifugal_coriolis_vector)
        for _ in range(10):
            q, v = sample_state()
            q_pin, v_pin, T, Tdot_v = frax_to_pinocchio(q, v)
            # The bias in frax's coordinates includes the effect of T_dot
            # tau = T^T (M_pin (T a + T_dot v) + b_pin), with a = 0
            tau_pin = pin.rnea(model, data, q_pin, v_pin, Tdot_v)
            expected = T.T @ tau_pin
            np.testing.assert_allclose(bias(q, v), expected, atol=1e-8)
            np.testing.assert_allclose(gravity(q) + cc(q, v), expected, atol=1e-8)

    def test_rnea(self, robot, pin_model_and_data):
        model, data = pin_model_and_data
        rnea = jax.jit(lambda q, v, a: robot.rnea(q, v, a, g, None))
        for _ in range(10):
            q, v = sample_state()
            a = np.random.uniform(-1.0, 1.0, robot.nv)
            q_pin, v_pin, T, Tdot_v = frax_to_pinocchio(q, v)
            a_pin = T @ a + Tdot_v
            tau_pin = pin.rnea(model, data, q_pin, v_pin, a_pin)
            np.testing.assert_allclose(rnea(q, v, a), T.T @ tau_pin, atol=1e-8)

    def test_forward_dynamics(self, robot, pin_model_and_data):
        model, data = pin_model_and_data
        fd = jax.jit(lambda q, v, tau: robot.forward_dynamics(q, v, tau, None))
        for _ in range(10):
            q, v = sample_state()
            tau = np.random.uniform(-1.0, 1.0, robot.nv)
            q_pin, v_pin, T, Tdot_v = frax_to_pinocchio(q, v)
            # T is orthogonal, so T^-T = T
            a_pin = pin.aba(model, data, q_pin, v_pin, T @ tau)
            expected = T.T @ (a_pin - Tdot_v)
            np.testing.assert_allclose(fd(q, v, tau), expected, rtol=1e-6, atol=1e-6)

    def test_kinematics_and_jacobians(self, robot, pin_model_and_data):
        model, data = pin_model_and_data
        fk = jax.jit(robot.joint_to_world_transforms)
        joint_jacobians = jax.jit(
            lambda q: robot._joint_jacobians(robot.joint_to_world_transforms(q))
        )
        for _ in range(5):
            q, v = sample_state()
            q_pin, _, T, _ = frax_to_pinocchio(q, v)
            pin.forwardKinematics(model, data, q_pin)
            pin.computeJointJacobians(model, data, q_pin)
            tfs = fk(q)
            Jvs, Jws = joint_jacobians(q)
            # Index 5 is the base (pelvis); pinocchio joint 1 is the freeflyer
            for my_idx in range(5, robot.nv):
                pin_idx = my_idx - 4
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


def test_floating_manipulator_vs_pinocchio():
    """A free-floating manipulator is a pure kinematic chain, which uses the scanned FK path"""
    urdf = str(FRANKA_ASSETS_DIR / "panda.urdf")
    model = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
    data = pin.Data(model)
    robot = Manipulator(urdf, floating_base="quaternion")
    assert robot.is_pure_kinematic_chain

    @jax.jit
    def get_data(q, v):
        tfs = robot.joint_to_world_transforms(q)
        return tfs, robot._mass_matrix(tfs), robot._nonlinear_bias(v, tfs)

    for _ in range(5):
        q, v = sample_state(robot.nv)
        q_pin, v_pin, T, Tdot_v = frax_to_pinocchio(q, v)
        tfs, M, bias = get_data(q, v)
        M_pin = pin.crba(model, data, q_pin)
        M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T
        np.testing.assert_allclose(M, T.T @ M_pin @ T, atol=1e-10)
        tau_pin = pin.rnea(model, data, q_pin, v_pin, Tdot_v)
        np.testing.assert_allclose(bias, T.T @ tau_pin, atol=1e-10)
        pin.forwardKinematics(model, data, q_pin)
        for my_idx in range(5, robot.nv):
            np.testing.assert_allclose(
                tfs[my_idx], data.oMi[my_idx - 4].homogeneous, atol=1e-12
            )


class TestVsMujoco:
    """Compare the quaternion floating base against MuJoCo's free joint directly

    Note: the G1 MJCF's inertial parameters are rounded to ~6 significant figures relative to
    the URDF, so we use tolerances that are relative to the magnitudes of the quantities
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
            q, v = sample_state()
            set_mj_state(model, data, q, v)
            M_mj = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, data, M_mj)
            np.testing.assert_allclose(mass_matrix(q), M_mj, atol=1e-4)

    def test_bias(self, robot, mj_model_and_data):
        model, data = mj_model_and_data
        bias = jax.jit(robot.nonlinear_bias)
        for _ in range(10):
            q, v = sample_state()
            set_mj_state(model, data, q, v)
            np.testing.assert_allclose(bias(q, v), data.qfrc_bias, rtol=1e-5, atol=1e-3)

    def test_forward_dynamics(self, robot, mj_model_and_data):
        model, data = mj_model_and_data
        fd = jax.jit(lambda q, v, tau: robot.forward_dynamics(q, v, tau, None))
        for _ in range(10):
            q, v = sample_state()
            tau = np.random.uniform(-1.0, 1.0, robot.nv)
            set_mj_state(model, data, q, v, tau)
            # qacc_smooth is the acceleration without any constraint forces (contact, limits)
            a = np.asarray(fd(q, v, tau))
            np.testing.assert_allclose(
                a, data.qacc_smooth, atol=1e-4 * np.max(np.abs(a))
            )

    def test_kinematics_and_jacobians(self, robot, mj_model_and_data):
        model, data = mj_model_and_data
        fk = jax.jit(robot.joint_to_world_transforms)
        joint_jacobians = jax.jit(
            lambda q: robot._joint_jacobians(robot.joint_to_world_transforms(q))
        )
        for _ in range(5):
            q, v = sample_state()
            set_mj_state(model, data, q, v)
            tfs = fk(q)
            Jvs, Jws = joint_jacobians(q)
            for my_idx in range(5, robot.nv):
                body_id = (
                    my_idx - 4
                )  # MuJoCo's body 0 is the world, body 1 is the pelvis
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

    def test_integrate(self, robot, mj_model_and_data):
        model, _ = mj_model_and_data
        for _ in range(10):
            q, v = sample_state()
            dt = 0.1
            q_mj = q.copy()
            mujoco.mj_integratePos(model, q_mj, v, dt)
            np.testing.assert_allclose(robot.integrate(q, v, dt), q_mj, atol=1e-12)


class TestVsEuler:
    """Check that the quaternion and euler representations describe the same physics"""

    @staticmethod
    def sample_matching_states():
        q_quat, v_quat = sample_state()
        pos, quat = q_quat[:3], q_quat[3:7]
        vel = v_quat[:3]
        omega_world = np.asarray(rotate_vector_by_quat_wxyz(quat, v_quat[3:6]))
        q_ff, v_ff = pose_and_twist_to_virtual_joints(pos, quat, vel, omega_world)
        q_euler = np.concatenate([q_ff, q_quat[7:]])
        v_euler = np.concatenate([v_ff, v_quat[6:]])
        return q_euler, v_euler, q_quat, v_quat

    def test_dimensions(self, robot, euler_robot):
        assert euler_robot.nq == 35
        assert euler_robot.nv == 35
        assert robot.nq == 36
        assert robot.nv == 35

    def test_velocity_to_configuration_index(self, robot, euler_robot):
        for r in (robot, euler_robot):
            v_idxs = np.array(list(r.velocity_to_configuration_index.keys()))
            q_idxs = np.array(list(r.velocity_to_configuration_index.values()))
            # Entries with a configuration entry integrate componentwise (q_dot = v)
            q, v = r.default_configuration, np.random.uniform(-1.0, 1.0, r.nv)
            q1 = np.asarray(r.integrate(q, v, 0.1))
            np.testing.assert_allclose(q1[q_idxs], q[q_idxs] + 0.1 * v[v_idxs])
        assert euler_robot.velocity_to_configuration_index == {i: i for i in range(35)}
        assert robot.velocity_to_configuration_index == dict(
            zip([*range(3), *range(6, 35)], [*range(3), *range(7, 36)])
        )

    def test_default_configuration(self, robot, euler_robot):
        # The G1's default is standing at a nonzero height, with the same pose for both bases
        q_quat, q_euler = (
            robot.default_configuration,
            euler_robot.default_configuration,
        )
        assert q_quat[2] > 0
        np.testing.assert_array_equal(q_quat[:3], q_euler[:3])
        np.testing.assert_array_equal(q_quat[3:7], [1, 0, 0, 0])
        np.testing.assert_array_equal(q_euler[3:6], [0, 0, 0])
        np.testing.assert_array_equal(q_quat[7:], q_euler[6:])
        # If unspecified: identity base pose at the origin, zero joints
        urdf = str(FRANKA_ASSETS_DIR / "panda.urdf")
        np.testing.assert_array_equal(
            Manipulator(urdf, floating_base="quaternion").default_configuration,
            np.r_[0, 0, 0, 1, np.zeros(10)],
        )
        # Specified on construction, including the base pose
        q_joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        q_default = np.r_[0.1, 0.2, 0.3, 0, 1, 0, 0, q_joints]
        panda = Manipulator(
            urdf, floating_base="quaternion", default_configuration=q_default
        )
        np.testing.assert_array_equal(panda.default_configuration, q_default)
        # Wrong shape (missing the base pose)
        with pytest.raises(ValueError):
            Manipulator(
                urdf, floating_base="quaternion", default_configuration=q_joints
            )
        # Joints must be within the joint limits
        with pytest.raises(ValueError):
            Manipulator(urdf, default_configuration=np.zeros(7))
        # Quaternion must be unit norm
        with pytest.raises(ValueError):
            Manipulator(
                urdf,
                floating_base="quaternion",
                default_configuration=np.r_[0, 0, 0, 2, 0, 0, 0, q_joints],
            )

    def test_floating_base_limits(self, robot, euler_robot):
        for r in (robot, euler_robot):
            # Limits only cover the actuated joints, not the floating base
            for limits in (
                r.actuated_joint_lower_limits,
                r.actuated_joint_upper_limits,
                r.actuated_joint_max_velocities,
                r.actuated_joint_max_forces,
            ):
                assert limits.shape == (r.num_actuated_joints,)
                assert np.all(np.isfinite(limits))

    def test_physical_consistency(self, robot, euler_robot):
        def physical_quantities(q, v, robot_is_quat):
            r = robot if robot_is_quat else euler_robot
            tfs = r.joint_to_world_transforms(q)
            M = r._mass_matrix(tfs)
            a = r._forward_dynamics(tfs, v, jnp.zeros(r.nv), None)
            J, Jdot = r._left_hand_jacobian_and_derivative(v, tfs)
            com = r._center_of_mass(tfs)
            com_vel = r._center_of_mass_jacobian(tfs) @ v
            return (
                tfs[5:],  # All real (non-virtual) bodies
                0.5 * v @ M @ v,  # Kinetic energy
                J @ v,  # Hand velocity
                J @ a + Jdot @ v,  # Hand acceleration under passive dynamics
                com,
                com_vel,
            )

        physical_quantities = jax.jit(physical_quantities, static_argnums=2)
        for _ in range(10):
            q_e, v_e, q_q, v_q = self.sample_matching_states()
            res_e = physical_quantities(q_e, v_e, False)
            res_q = physical_quantities(q_q, v_q, True)
            for a, b in zip(res_e, res_q):
                np.testing.assert_allclose(a, b, atol=1e-8)


class TestAutodiff:
    """Check the analytical Jacobians/derivatives against autodiff, via the configuration
    velocity map. See also test_jacobians_and_derivatives.py"""

    def test_integrate_and_difference(self, robot):
        for _ in range(10):
            q, v = sample_state()
            q1 = robot.integrate(q, v, 0.5)
            np.testing.assert_allclose(robot.difference(q, q1), 0.5 * v, atol=1e-10)
            # Zero velocity should have a well-defined (non-NaN) derivative
            dq = jax.jacobian(robot.integrate, argnums=1)(q, jnp.zeros(35), 0.5)
            assert np.all(np.isfinite(dq))

    def test_velocity_to_qdot_map(self, robot):
        for _ in range(10):
            q, v = sample_state()
            E = robot.velocity_to_qdot_map(q)
            # q_dot = d/dt integrate(q, v, t) at t = 0
            q_dot = jax.jacobian(lambda t: robot.integrate(q, v, t))(0.0)
            np.testing.assert_allclose(E @ v, q_dot, atol=1e-10)
