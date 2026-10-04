"""Test cases for the Recursive Newton Euler Algorithm (RNEA), comparing against Pinocchio"""

import jax
import jax.numpy as jnp
import numpy as np
import pinocchio as pin
import pytest

from frax.assets import FRANKA_ASSETS_DIR
from frax.robots.franka_panda import load_panda

G_ACCEL = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def robot():
    return load_panda()


@pytest.fixture(scope="module")
def pin_model_and_data():
    model = pin.buildModelFromUrdf(FRANKA_ASSETS_DIR / "panda.urdf")
    return model, pin.Data(model)


@pytest.mark.parametrize(
    "nonzero_v, nonzero_a",
    [(False, False), (True, False), (True, True)],
    ids=["zero_v_a", "nonzero_v", "nonzero_v_a"],
)
def test_rnea(robot, pin_model_and_data, nonzero_v, nonzero_a):
    model, data = pin_model_and_data
    rnea = jax.jit(lambda q, v, a: robot.rnea(q, v, a, G_ACCEL, None))
    for _ in range(10):
        q = np.random.uniform(-np.pi / 2, np.pi / 2, robot.nv)
        v = np.random.uniform(-np.pi / 2, np.pi / 2, robot.nv) * nonzero_v
        a = np.random.uniform(-np.pi / 2, np.pi / 2, robot.nv) * nonzero_a
        tau = pin.rnea(model, data, q, v, a)
        np.testing.assert_array_almost_equal(tau, rnea(q, v, a), decimal=4)


@pytest.mark.skip(
    reason="TODO: external forces are defined in the root frame in frax, "
    "but in the local joint frames in Pinocchio"
)
def test_rnea_with_external_forces():
    pass
