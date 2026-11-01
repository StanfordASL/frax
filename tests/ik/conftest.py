import numpy as np
import pytest

from frax import load_g1, load_panda

# fmt: off
G1_DEFAULT_JOINT_POSITIONS = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
    0.0, 0.0, 0.0,  # torso (3)
    0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,  # left arm (7)
    0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,  # right arm (7)
])
# fmt: on


@pytest.fixture(scope="session")
def g1():
    return load_g1()


@pytest.fixture(scope="session")
def panda():
    return load_panda()


@pytest.fixture(scope="session")
def g1_q0():
    """Standing G1 configuration: root at 0.8 m (identity orientation), default
    joint positions. q = [root pos, root WXYZ quat, joints]"""
    return np.concatenate(
        [[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0], G1_DEFAULT_JOINT_POSITIONS]
    )


@pytest.fixture
def g1_q_random(g1, g1_q0):
    """A perturbed G1 configuration, away from symmetric/degenerate poses"""
    rng = np.random.default_rng(0)
    return np.asarray(g1.integrate(g1_q0, 0.3 * rng.standard_normal(g1.nv), 1.0))
