"""Shared pytest configuration

This configures the environment for JAX (matching the timing scripts) before any test
module imports it, and seeds numpy's global RNG before each test so that results do not
depend on which tests are run, or in what order
"""

import numpy as np
import pytest

from timing.timing_utils import configure_env

configure_env("cpu", "float64")

import jax  # noqa: E402 (after configure_env)

# Several CPU devices, for the multi-device tests of frax.ik. This must be set
# before any JAX op (importing frax runs one), and the default device is still cpu:0
jax.config.update("jax_num_cpu_devices", 4)


@pytest.fixture(autouse=True)
def seed_numpy():
    np.random.seed(0)
