"""Test that robot methods use full-precision matmuls"""

import re

import jax
import numpy as np
import pytest

from frax.robots.unitree_g1 import load_g1


@pytest.fixture(scope="module")
def robot():
    return load_g1()


def dot_precisions(func, *args) -> set:
    """Precisions of all dot_general ops in the lowered (StableHLO) program"""
    text = jax.jit(func).lower(*args).as_text()
    n_dots = len(re.findall(r"stablehlo\.dot_general", text))
    precisions = re.findall(r"precision = \[([A-Z_, ]+)\]", text)
    assert n_dots > 0 and len(precisions) == n_dots
    return set(precisions)


@pytest.mark.parametrize("outer_precision", [None, "tensorfloat32"])
def test_highest_matmul_precision(robot, outer_precision):
    q = np.asarray(robot.default_configuration)
    v = np.random.uniform(-0.5, 0.5, robot.nv)
    funcs = {
        "forward_dynamics": lambda: dot_precisions(
            lambda q, v: robot.forward_dynamics(q, v, v, None), q, v
        ),
        "_mass_matrix": lambda: dot_precisions(
            lambda q: robot._mass_matrix(robot.joint_to_world_transforms(q)), q
        ),
        "left_hand_jacobian_and_derivative": lambda: dot_precisions(
            robot.left_hand_jacobian_and_derivative, q, v
        ),
    }
    for name, get_precisions in funcs.items():
        if outer_precision is None:
            precisions = get_precisions()
        else:
            # override a lower precision set by the user's surrounding code
            with jax.default_matmul_precision(outer_precision):
                precisions = get_precisions()
        assert precisions == {"HIGHEST, HIGHEST"}, name
