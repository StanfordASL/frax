"""Test cases for transformation matrix utils"""

import numpy as np

from frax.utils.rotation_utils import quat_wxyz_to_rmat
from frax.utils.transform_utils import (
    create_transform_numpy,
    transform_point,
    transform_points,
)

TOL = 1e-12


def random_transform():
    q = np.random.randn(4)
    R = np.asarray(quat_wxyz_to_rmat(q / np.linalg.norm(q)))
    return create_transform_numpy(R, np.random.randn(3))


def test_transform_point():
    for _ in range(10):
        T = random_transform()
        p = np.random.randn(3)
        expected = (T @ np.append(p, 1.0))[:3]
        np.testing.assert_allclose(transform_point(T, p), expected, atol=TOL)


def test_transform_points():
    for _ in range(10):
        T = random_transform()
        pts = np.random.randn(7, 3)
        expected = (np.hstack([pts, np.ones((7, 1))]) @ T.T)[:, :3]
        np.testing.assert_allclose(transform_points(T, pts), expected, atol=TOL)
        # Consistent with the single-point version
        for p, p_out in zip(pts, np.asarray(transform_points(T, pts))):
            np.testing.assert_allclose(transform_point(T, p), p_out, atol=TOL)
