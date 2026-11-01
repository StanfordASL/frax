import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from frax.ik import load_bvh, resample_poses, save_motion, smooth_poses


def _rotation_clip(T, dt):
    """Poses (T, 2, 7) moving at constant linear and angular velocity, with the
    quaternion sign flipped on alternate frames"""
    t = np.arange(T)[:, None] * dt
    pos = np.stack([t * [1.0, -2.0, 0.5], t * [0.0, 3.0, 1.0]], axis=1)
    rotvecs = np.stack([t * [2.0, 0.0, 1.0], t * [0.0, -4.0, 0.0]], 1)
    quat = Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_quat(scalar_first=True)
    quat = quat.reshape(T, 2, 4) * np.where(np.arange(T) % 2, -1.0, 1.0)[:, None, None]
    return np.concatenate([pos, quat], axis=-1)


def _angles(q1, q2):
    r1 = Rotation.from_quat(q1.reshape(-1, 4), scalar_first=True)
    r2 = Rotation.from_quat(q2.reshape(-1, 4), scalar_first=True)
    return (r1.inv() * r2).magnitude()


def test_smooth_poses_constant_velocity():
    """Constant linear and angular velocity pass through unchanged, away from the edges"""
    dt, T = 0.02, 100
    poses = _rotation_clip(T, dt)
    smoothed = smooth_poses(poses, dt, sigma_pos=0.05, sigma_rot=0.05)
    inner = slice(15, T - 15)
    np.testing.assert_allclose(smoothed[inner, :, :3], poses[inner, :, :3], atol=1e-12)
    np.testing.assert_allclose(
        _angles(smoothed[inner, :, 3:], poses[inner, :, 3:]), 0, atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.norm(smoothed[..., 3:], axis=-1), 1.0)


def test_smooth_poses_reduces_noise():
    dt, T = 0.02, 200
    poses = _rotation_clip(T, dt)
    rng = np.random.default_rng(0)
    noisy = poses.copy()
    noisy[..., :3] += 0.01 * rng.standard_normal(noisy[..., :3].shape)
    noise = Rotation.from_rotvec(0.05 * rng.standard_normal((T * 2, 3)))
    noisy[..., 3:] = (
        (noise * Rotation.from_quat(poses[..., 3:].reshape(-1, 4), scalar_first=True))
        .as_quat(scalar_first=True)
        .reshape(T, 2, 4)
    )
    smoothed = smooth_poses(noisy, dt, sigma_pos=0.04, sigma_rot=0.04)
    pos_err = lambda p: np.linalg.norm(p[..., :3] - poses[..., :3], axis=-1).mean()
    assert pos_err(smoothed) < 0.5 * pos_err(noisy)
    rot_err = lambda p: _angles(p[..., 3:], poses[..., 3:]).mean()
    assert rot_err(smoothed) < 0.5 * rot_err(noisy)
    # Zero widths leave the poses unchanged
    np.testing.assert_array_equal(smooth_poses(noisy, dt), noisy)


@pytest.mark.parametrize("fps_out", [30.0, 50.0, 12.5, 7.5])
def test_resample_poses_constant_velocity(fps_out):
    """Constant linear and angular velocity (about fixed axes) resample exactly,
    at the same rate, up and down"""
    fps_in, T = 30.0, 61
    poses = _rotation_clip(T, 1 / fps_in)
    out = resample_poses(poses, fps_in, fps_out)
    T_out = int((T - 1) * fps_out / fps_in) + 1
    expected = _rotation_clip(T_out, 1 / fps_out)
    assert out.shape == expected.shape
    np.testing.assert_allclose(out[..., :3], expected[..., :3], atol=1e-12)
    np.testing.assert_allclose(_angles(out[..., 3:], expected[..., 3:]), 0, atol=1e-12)


def test_resample_poses_integer_ratio():
    """Downsampling by an integer factor keeps every k-th pose"""
    rng = np.random.default_rng(1)
    poses = np.concatenate(
        [
            rng.standard_normal((41, 3, 3)),
            Rotation.random(41 * 3, rng=rng)
            .as_quat(scalar_first=True)
            .reshape(41, 3, 4),
        ],
        axis=-1,
    )
    out = resample_poses(poses, 120.0, 30.0)
    np.testing.assert_allclose(out[..., :3], poses[::4, :, :3], atol=1e-12)
    np.testing.assert_allclose(_angles(out[..., 3:], poses[::4, :, 3:]), 0, atol=1e-10)


# Root, and a child with its own position channels and rotation order. Frame 0:
# child at its channel position. Frame 1: root turned 90 deg about Z
BVH = """HIERARCHY
ROOT Root
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT Child
    {
        OFFSET 0.0 50.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
        End Site
        {
            OFFSET 0.0 10.0 0.0
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.1
0 0 100 0 0 0 10 0 0 0 0 30
0 0 100 90 0 0 10 0 0 0 0 30
"""


def test_load_bvh(tmp_path):
    """Global poses by FK: position channels replace a bone's offset, each bone
    with its own rotation order, then into the world frame"""
    path = tmp_path / "clip.bvh"
    path.write_text(BVH)
    y_up_to_z_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    poses, names = load_bvh(path, y_up_to_z_up)
    assert names == ["Root", "Child"] and poses.shape == (2, 2, 7)
    np.testing.assert_allclose(poses[0, 0, :3], [0.0, -1.0, 0.0], atol=1e-12)
    # The child's position channels (10 cm along X), not its offset
    np.testing.assert_allclose(poses[0, 1, :3], [0.1, -1.0, 0.0], atol=1e-12)
    # Root turned 90 deg about BVH Z: BVH +X -> BVH +Y (world +Z)
    np.testing.assert_allclose(poses[1, 1, :3], [0.0, -1.0, 0.1], atol=1e-12)
    expected = Rotation.from_matrix(y_up_to_z_up) * Rotation.from_euler(
        "Z", 90, degrees=True
    )
    rot = Rotation.from_quat(poses[1, 0, 3:], scalar_first=True)
    assert (rot.inv() * expected).magnitude() < 1e-9
    # The child's rotation (30 deg about its Z), after the root's
    rot = Rotation.from_quat(poses[1, 1, 3:], scalar_first=True)
    expected = expected * Rotation.from_euler("Z", 30, degrees=True)
    assert (rot.inv() * expected).magnitude() < 1e-9


def test_load_bvh_nan(tmp_path):
    path = tmp_path / "clip.bvh"
    path.write_text(BVH.replace("0 0 100 90", "nan nan nan nan"))
    with pytest.raises(ValueError, match="NaN"):
        load_bvh(path)


def test_save_motion(g1, tmp_path):
    q = np.tile(g1.default_configuration, (5, 1))
    save_motion(tmp_path / "clip.npz", g1, q, 30.0)
    data = np.load(tmp_path / "clip.npz")
    assert data["q"].dtype == np.float32
    np.testing.assert_allclose(data["q"], q, atol=1e-6)
    assert data["fps"] == 30.0
    assert list(data["joint_names"]) == g1.joint_names[g1.nv_floating :]
    assert len(data["joint_names"]) == g1.nq - g1.nq_floating
