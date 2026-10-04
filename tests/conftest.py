"""Shared pytest configuration

This configures the environment for JAX (matching the timing scripts) before any test
module imports it, and seeds numpy's global RNG before each test so that results do not
depend on which tests are run, or in what order
"""

import numpy as np
import pytest

from timing.timing_utils import configure_env

configure_env("cpu")


@pytest.fixture(autouse=True)
def seed_numpy():
    np.random.seed(0)
