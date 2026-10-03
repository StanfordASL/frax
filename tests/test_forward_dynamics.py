"""Test cases for forward dynamics"""

import unittest

import jax
import numpy as np
import pinocchio as pin

from frax.robots.franka_panda import load_panda
from frax.assets import FRANKA_ASSETS_DIR

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_static
class ForwardDynamicsTest(unittest.TestCase):
    """Comparing FD output to Pinocchio's values (from ABA)"""

    @classmethod
    def setUpClass(cls):
        cls.model = pin.buildModelFromUrdf(FRANKA_ASSETS_DIR / "panda.urdf")
        cls.data = pin.Data(cls.model)
        cls.robot = load_panda()
        cls.num_joints = cls.robot.nv
        np.random.seed(0)

    @jax.jit
    def my_fd(self, q, v, tau, fext):
        return self.robot.forward_dynamics(q, v, tau, fext)

    def test_zero_v_tau_f(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            dq = np.zeros(self.num_joints)
            tau = np.zeros(self.num_joints)
            fext = None
            a = pin.aba(self.model, self.data, q, dq, tau)  # , fext)
            a_mine = self.my_fd(q, dq, tau, fext)
            np.testing.assert_array_almost_equal(a, a_mine, decimal=4)

    def test_nonzero_v_zero_tau_f(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            dq = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            tau = np.zeros(self.num_joints)
            fext = None
            a = pin.aba(self.model, self.data, q, dq, tau)  # , fext)
            a_mine = self.my_fd(q, dq, tau, fext)
            np.testing.assert_array_almost_equal(a, a_mine, decimal=4)

    def test_nonzero_v_tau_zero_f(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            dq = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            tau = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            fext = None
            a = pin.aba(self.model, self.data, q, dq, tau)  # , fext)
            a_mine = self.my_fd(q, dq, tau, fext)
            np.testing.assert_array_almost_equal(a, a_mine, decimal=4)

    def test_nonzero_v_tau_f(self):
        pass
        # TODO! Need to adjust reference frames for forces
        # i.e. mine are defined in the root frame
        # and pinocchio's are defined in the local frame of the joints


if __name__ == "__main__":
    unittest.main()
