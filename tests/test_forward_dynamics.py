"""Test cases for forward dynamics, comparing against Pinocchio's values (from ABA)"""

import jax
import numpy as np
import pinocchio as pin
import pytest

from frax.assets import FRANKA_ASSETS_DIR
from frax.robots.franka_panda import load_panda


@pytest.fixture(scope="module")
def robot():
    return load_panda()


@pytest.fixture(scope="module")
def pin_model_and_data():
    model = pin.buildModelFromUrdf(FRANKA_ASSETS_DIR / "panda.urdf")
    return model, pin.Data(model)


@pytest.mark.parametrize(
    "nonzero_v, nonzero_tau",
    [(False, False), (True, False), (True, True)],
    ids=["zero_v_tau", "nonzero_v", "nonzero_v_tau"],
)
def test_forward_dynamics(robot, pin_model_and_data, nonzero_v, nonzero_tau):
    model, data = pin_model_and_data
    fd = jax.jit(lambda q, v, tau: robot.forward_dynamics(q, v, tau, None))
    for _ in range(10):
        q = np.random.uniform(-np.pi / 2, np.pi / 2, robot.nv)
        v = np.random.uniform(-np.pi / 2, np.pi / 2, robot.nv) * nonzero_v
        tau = np.random.uniform(-np.pi / 2, np.pi / 2, robot.nv) * nonzero_tau
        a = pin.aba(model, data, q, v, tau)
        np.testing.assert_array_almost_equal(a, fd(q, v, tau), decimal=4)


@pytest.mark.skip(
    reason="TODO: external forces are defined in the root frame in frax, "
    "but in the local joint frames in Pinocchio"
)
def test_forward_dynamics_with_external_forces():
    pass
