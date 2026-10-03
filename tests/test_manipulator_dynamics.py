"""Test cases for the Manipulator class"""

import unittest

import jax
import numpy as np
import pinocchio as pin

from frax.core.manipulator import Manipulator
from frax.assets import FRANKA_ASSETS_DIR

URDF = FRANKA_ASSETS_DIR / "panda.urdf"

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


class PinocchioDynamicsTest(unittest.TestCase):
    """Test cases to validate the manipulator dynamics against Pinocchio's values"""

    @classmethod
    def setUpClass(cls):
        cls.model = pin.buildModelFromUrdf(URDF)
        cls.data = pin.Data(cls.model)
        cls.robot = Manipulator(URDF)
        cls.num_joints = cls.robot.nv
        np.random.seed(0)

    def test_mass_matrix(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            dq = np.zeros(self.num_joints)
            pin.forwardKinematics(self.model, self.data, q, dq)
            pin.updateFramePlacements(self.model, self.data)
            M = self.robot.mass_matrix(q)
            Mpin = pin.crba(self.model, self.data, q)
            np.testing.assert_array_almost_equal(M, Mpin, decimal=4)

    def test_nonlinear_effects(self):
        for i in range(10):
            q = np.random.uniform(-np.pi / 2, np.pi / 2, self.num_joints)
            dq = np.random.rand(self.num_joints)
            pin.forwardKinematics(self.model, self.data, q, dq)
            pin.updateFramePlacements(self.model, self.data)
            bias = pin.nle(self.model, self.data, q, dq)
            G = self.robot.gravity_vector(q)
            C = self.robot.centrifugal_coriolis_vector(q, dq)
            np.testing.assert_array_almost_equal(G + C, bias, decimal=4)


if __name__ == "__main__":
    unittest.main()
