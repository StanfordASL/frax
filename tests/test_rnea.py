"""Test cases for the Recursive Newton Euler Algorithm (RNEA)"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pinocchio as pin

from frax.robots.franka_panda import load_panda
from frax.assets import FRANKA_ASSETS_DIR

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_static
class RNEATest(unittest.TestCase):
    """Comparing RNEA output to Pinocchio's values"""

    @classmethod
    def setUpClass(cls):
        cls.model = pin.buildModelFromUrdf(FRANKA_ASSETS_DIR / "panda.urdf")
        cls.data = pin.Data(cls.model)
        cls.robot = load_panda()
        cls.num_joints = cls.robot.nv
        np.random.seed(0)

    @jax.jit
    def my_rnea(self, q, v, a, fext):
        g_accel = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
        return self.robot.rnea(q, v, a, g_accel, fext)

    def test_zero_v_a_f(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            dq = np.zeros(self.num_joints)
            ddq = np.zeros(self.num_joints)
            fext = None
            pin.forwardKinematics(self.model, self.data, q, dq, ddq)
            pin.updateFramePlacements(self.model, self.data)
            tau = pin.rnea(self.model, self.data, q, dq, ddq)  # , fext)
            tau_mine = self.my_rnea(q, dq, ddq, fext)
            np.testing.assert_array_almost_equal(tau, tau_mine, decimal=4)

    def test_nonzero_v_zero_a_f(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            dq = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            ddq = np.zeros(self.num_joints)
            fext = None
            pin.forwardKinematics(self.model, self.data, q, dq, ddq)
            pin.updateFramePlacements(self.model, self.data)
            tau = pin.rnea(self.model, self.data, q, dq, ddq)  # , fext)
            tau_mine = self.my_rnea(q, dq, ddq, fext)
            np.testing.assert_array_almost_equal(tau, tau_mine, decimal=4)

    def test_nonzero_v_a_zero_f(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            dq = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            ddq = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            fext = None
            pin.forwardKinematics(self.model, self.data, q, dq, ddq)
            pin.updateFramePlacements(self.model, self.data)
            tau = pin.rnea(self.model, self.data, q, dq, ddq)  # , fext)
            tau_mine = self.my_rnea(q, dq, ddq, fext)
            np.testing.assert_array_almost_equal(tau, tau_mine, decimal=4)

    def test_nonzero_v_a_f(self):
        pass
        # TODO! Need to adjust reference frames for forces
        # i.e. mine are defined in the root frame
        # and pinocchio's are defined in the local frame of the joints


if __name__ == "__main__":
    unittest.main()
