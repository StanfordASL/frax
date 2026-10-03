"""Test cases for the Quadruped class, using the Unitree A2 with a quaternion floating base

See test_quaternion_floating_base.py for details on the conversion between the frax (MuJoCo)
and Pinocchio floating base conventions

Note: Run with unittest (python -m unittest tests/test_quadruped_dynamics.py)
"""

import unittest
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pinocchio as pin
import mujoco

from frax.robots.unitree_a2 import load_a2
from frax.assets import A2_ASSETS_DIR
from frax.utils.rotation_utils import quat_wxyz_to_rmat, wxyz_to_xyzw

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

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


class QuadrupedVsPinocchioTest(unittest.TestCase):
    """Compare the A2 quadruped against Pinocchio with a freeflyer joint"""

    @classmethod
    def setUpClass(cls):
        cls.model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
        cls.data = pin.Data(cls.model)
        cls.robot = load_a2()
        cls.mass_matrix = staticmethod(jax.jit(cls.robot.mass_matrix))
        cls.bias = staticmethod(jax.jit(cls.robot.nonlinear_bias))
        cls.gravity = staticmethod(jax.jit(cls.robot.gravity_vector))
        cls.cc = staticmethod(jax.jit(cls.robot.centrifugal_coriolis_vector))
        cls.rnea = staticmethod(jax.jit(lambda q, qd, qdd: cls.robot.rnea(q, qd, qdd, g, None)))
        cls.fd = staticmethod(jax.jit(lambda q, qd, tau: cls.robot.forward_dynamics(q, qd, tau, None)))
        cls.fk = staticmethod(jax.jit(cls.robot.joint_to_world_transforms))
        cls.joint_jacobians = staticmethod(jax.jit(
            lambda q: cls.robot._joint_jacobians(cls.robot.joint_to_world_transforms(q))
        ))

        @jax.jit
        def feet_data(q):
            tfs = cls.robot.joint_to_world_transforms(q)
            T_feet = [getattr(cls.robot, f"_{f}_foot_transform")(tfs) for f in FEET]
            J_feet = [getattr(cls.robot, f"_{f}_foot_jacobian")(tfs) for f in FEET]
            return T_feet, J_feet

        cls.feet_data = staticmethod(feet_data)
        # Pinocchio joint ids for each of frax's bodies (index 5 is the base, i.e. the freeflyer)
        cls.pin_joint_ids = [1] + [
            cls.model.getJointId(name) for name in cls.robot.joint_names[6:]
        ]
        # Actuated joint permutation: q_pin[7:] = q_frax[7:][perm]
        cls.perm = np.array(
            [cls.robot.joint_names.index(name) - 6 for name in cls.model.names[2:]]
        )
        np.random.seed(0)

    def test_dimensions(self):
        self.assertEqual(self.robot.nq, self.model.nq)
        self.assertEqual(self.robot.nv, self.model.nv)
        self.assertEqual(self.robot.num_actuated_joints, NUM_ACTUATED)
        self.assertEqual(sorted(self.robot.joint_names[6:]), sorted(self.model.names[2:]))

    def test_mass_matrix(self):
        for _ in range(10):
            q, qd = sample_state()
            q_pin, _, T, _ = frax_to_pinocchio(q, qd, self.perm)
            M_pin = pin.crba(self.model, self.data, q_pin)
            M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T
            np.testing.assert_allclose(self.mass_matrix(q), T.T @ M_pin @ T, atol=1e-8)

    def test_nonlinear_bias(self):
        for _ in range(10):
            q, qd = sample_state()
            q_pin, v_pin, T, Tdot_qd = frax_to_pinocchio(q, qd, self.perm)
            # tau = T^T (M_pin (T qdd + T_dot qd) + b_pin), with qdd = 0
            tau_pin = pin.rnea(self.model, self.data, q_pin, v_pin, Tdot_qd)
            expected = T.T @ tau_pin
            np.testing.assert_allclose(self.bias(q, qd), expected, atol=1e-8)
            np.testing.assert_allclose(
                self.gravity(q) + self.cc(q, qd), expected, atol=1e-8
            )

    def test_rnea(self):
        for _ in range(10):
            q, qd = sample_state()
            qdd = np.random.uniform(-1.0, 1.0, self.robot.nv)
            q_pin, v_pin, T, Tdot_qd = frax_to_pinocchio(q, qd, self.perm)
            a_pin = T @ qdd + Tdot_qd
            tau_pin = pin.rnea(self.model, self.data, q_pin, v_pin, a_pin)
            np.testing.assert_allclose(
                self.rnea(q, qd, qdd), T.T @ tau_pin, atol=1e-8
            )

    def test_forward_dynamics(self):
        for _ in range(10):
            q, qd = sample_state()
            tau = np.random.uniform(-1.0, 1.0, self.robot.nv)
            q_pin, v_pin, T, Tdot_qd = frax_to_pinocchio(q, qd, self.perm)
            # T is orthogonal (a rotation and a permutation), so T^-T = T
            a_pin = pin.aba(self.model, self.data, q_pin, v_pin, T @ tau)
            expected = T.T @ (a_pin - Tdot_qd)
            np.testing.assert_allclose(
                self.fd(q, qd, tau), expected, rtol=1e-6, atol=1e-6
            )

    def test_kinematics_and_jacobians(self):
        for _ in range(5):
            q, qd = sample_state()
            q_pin, _, T, _ = frax_to_pinocchio(q, qd, self.perm)
            pin.forwardKinematics(self.model, self.data, q_pin)
            pin.computeJointJacobians(self.model, self.data, q_pin)
            tfs = self.fk(q)
            Jvs, Jws = self.joint_jacobians(q)
            for my_idx, pin_idx in zip(range(5, self.robot.nv), self.pin_joint_ids):
                oMi = self.data.oMi[pin_idx]
                np.testing.assert_allclose(tfs[my_idx, :3, :3], oMi.rotation, atol=1e-10)
                np.testing.assert_allclose(tfs[my_idx, :3, 3], oMi.translation, atol=1e-10)
                J_pin = pin.getJointJacobian(
                    self.model, self.data, pin_idx, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                J_pin = J_pin @ T
                np.testing.assert_allclose(Jvs[my_idx], J_pin[:3], atol=1e-10)
                np.testing.assert_allclose(Jws[my_idx], J_pin[3:], atol=1e-10)

    def test_feet(self):
        # The foot offsets coincide with the URDF's (fixed) foot links, which pinocchio keeps as frames
        frame_ids = [self.model.getFrameId(name) for name in PIN_FOOT_FRAMES]
        for _ in range(5):
            q, qd = sample_state()
            q_pin, _, T, _ = frax_to_pinocchio(q, qd, self.perm)
            pin.forwardKinematics(self.model, self.data, q_pin)
            pin.updateFramePlacements(self.model, self.data)
            pin.computeJointJacobians(self.model, self.data, q_pin)
            T_feet, J_feet = self.feet_data(q)
            for fid, T_foot, J_foot in zip(frame_ids, T_feet, J_feet):
                np.testing.assert_allclose(
                    T_foot, self.data.oMf[fid].homogeneous, atol=1e-10
                )
                J_pin = pin.getFrameJacobian(
                    self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                np.testing.assert_allclose(J_foot, J_pin @ T, atol=1e-10)


class QuadrupedVsMujocoTest(unittest.TestCase):
    """Compare the A2 quadruped against MuJoCo's free joint directly

    Note: the MJCF includes joint armature and damping, which are not modeled in the URDF,
    so these are zeroed out for the comparison. Inertial parameters in the MJCF are also
    rounded relative to the URDF, so we use tolerances relative to the quantities' magnitudes
    """

    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(mjcf))
        cls.model.dof_armature[:] = 0.0
        cls.model.dof_damping[:] = 0.0
        cls.data = mujoco.MjData(cls.model)
        cls.robot = load_a2()
        cls.mass_matrix = staticmethod(jax.jit(cls.robot.mass_matrix))
        cls.bias = staticmethod(jax.jit(cls.robot.nonlinear_bias))
        cls.fd = staticmethod(jax.jit(lambda q, qd, tau: cls.robot.forward_dynamics(q, qd, tau, None)))
        cls.fk = staticmethod(jax.jit(cls.robot.joint_to_world_transforms))
        cls.joint_jacobians = staticmethod(jax.jit(
            lambda q: cls.robot._joint_jacobians(cls.robot.joint_to_world_transforms(q))
        ))
        np.random.seed(1)

    def _set_state(self, q, qd, tau=None):
        self.data.qpos[:] = q
        self.data.qvel[:] = qd
        self.data.qfrc_applied[:] = 0.0 if tau is None else tau
        mujoco.mj_forward(self.model, self.data)

    def test_joint_ordering(self):
        mj_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(1, self.model.njnt)
        ]
        self.assertEqual(mj_names, list(self.robot.joint_names[6:]))
        self.assertEqual(self.model.nq, self.robot.nq)
        self.assertEqual(self.model.nv, self.robot.nv)

    def test_mass_matrix(self):
        for _ in range(10):
            q, qd = sample_state()
            self._set_state(q, qd)
            M_mj = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, self.data, M_mj)
            np.testing.assert_allclose(self.mass_matrix(q), M_mj, atol=1e-4)

    def test_bias(self):
        for _ in range(10):
            q, qd = sample_state()
            self._set_state(q, qd)
            np.testing.assert_allclose(
                self.bias(q, qd), self.data.qfrc_bias, rtol=1e-5, atol=1e-3
            )

    def test_forward_dynamics(self):
        for _ in range(10):
            q, qd = sample_state()
            tau = np.random.uniform(-1.0, 1.0, self.robot.nv)
            self._set_state(q, qd, tau)
            # qacc_smooth is the acceleration without any constraint forces (contact, limits)
            qdd = np.asarray(self.fd(q, qd, tau))
            np.testing.assert_allclose(
                qdd, self.data.qacc_smooth, atol=1e-4 * np.max(np.abs(qdd))
            )

    def test_kinematics_and_jacobians(self):
        for _ in range(5):
            q, qd = sample_state()
            self._set_state(q, qd)
            tfs = self.fk(q)
            Jvs, Jws = self.joint_jacobians(q)
            for my_idx in range(5, self.robot.nv):
                body_id = my_idx - 4  # MuJoCo's body 0 is the world, body 1 is the base
                np.testing.assert_allclose(
                    tfs[my_idx, :3, 3], self.data.xpos[body_id], atol=1e-6
                )
                np.testing.assert_allclose(
                    tfs[my_idx, :3, :3],
                    self.data.xmat[body_id].reshape(3, 3),
                    atol=1e-5,
                )
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
                np.testing.assert_allclose(Jvs[my_idx], jacp, atol=1e-5)
                np.testing.assert_allclose(Jws[my_idx], jacr, atol=1e-5)


class QuadrupedAutodiffTest(unittest.TestCase):
    """Check the analytical foot Jacobians and derivatives against autodiff"""

    @classmethod
    def setUpClass(cls):
        cls.robot = load_a2()
        np.random.seed(2)

    def test_feet_jacobians_and_derivatives(self):
        for foot in FEET:
            transform_func = getattr(self.robot, f"{foot}_foot_transform")
            J_func = getattr(self.robot, f"{foot}_foot_jacobian")
            J_and_Jdot_func = getattr(self.robot, f"{foot}_foot_jacobian_and_derivative")
            for _ in range(3):
                q, qd = sample_state()
                E = self.robot.configuration_velocity_map(q)
                J, Jdot = J_and_Jdot_func(q, qd)
                np.testing.assert_allclose(J, J_func(q), atol=1e-12)
                # Linear part of the Jacobian via autodiff
                Jv_ad = jax.jacobian(lambda q: transform_func(q)[:3, 3])(q) @ E
                np.testing.assert_allclose(J[:3], Jv_ad, atol=1e-10)
                # Jdot via a directional derivative of J along q_dot = E @ qd
                _, Jdot_ad = jax.jvp(J_func, (q,), (E @ qd,))
                np.testing.assert_allclose(Jdot, Jdot_ad, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
