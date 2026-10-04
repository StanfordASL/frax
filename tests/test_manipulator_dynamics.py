"""Test cases for the Manipulator class, comparing the dynamics against Pinocchio"""

import numpy as np
import pinocchio as pin
import pytest

from frax.assets import FRANKA_ASSETS_DIR
from frax.core.manipulator import Manipulator

URDF = FRANKA_ASSETS_DIR / "panda.urdf"


@pytest.fixture(scope="module")
def robot():
    return Manipulator(URDF)


@pytest.fixture(scope="module")
def pin_model_and_data():
    model = pin.buildModelFromUrdf(URDF)
    return model, pin.Data(model)


def test_mass_matrix(robot, pin_model_and_data):
    model, data = pin_model_and_data
    for _ in range(10):
        q = np.random.uniform(-np.pi / 2, np.pi / 2, robot.nv)
        M_pin = pin.crba(model, data, q)
        np.testing.assert_array_almost_equal(robot.mass_matrix(q), M_pin, decimal=4)


def test_nonlinear_effects(robot, pin_model_and_data):
    model, data = pin_model_and_data
    for _ in range(10):
        q = np.random.uniform(-np.pi / 2, np.pi / 2, robot.nv)
        v = np.random.rand(robot.nv)
        bias = pin.nle(model, data, q, v)
        G = robot.gravity_vector(q)
        C = robot.centrifugal_coriolis_vector(q, v)
        np.testing.assert_array_almost_equal(G + C, bias, decimal=4)
