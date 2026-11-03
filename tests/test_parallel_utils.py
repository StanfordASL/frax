"""device_vmap: a vmap with the batch split across devices"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frax.utils.parallel_utils import _resolve_devices, device_vmap


def f(x, shared):
    return {"sum": x["a"] + shared, "norm": jnp.linalg.norm(x["b"])}


def test_matches_vmap_with_padding():
    """10 entries over 4 devices (padded to 12), with a shared argument and
    pytree inputs and outputs"""
    devices = jax.devices()
    assert len(devices) == 4
    rng = np.random.default_rng(0)
    x = {"a": rng.normal(size=(10, 3)), "b": rng.normal(size=(10, 2, 2))}
    shared = rng.normal(size=3)
    expected = jax.vmap(f, in_axes=(0, None))(x, shared)
    for batched in (
        device_vmap(f, in_axes=(0, None)),
        jax.jit(device_vmap(f, devices, in_axes=(0, None))),
    ):
        out = batched(x, shared)
        assert out["sum"].shape == (10, 3) and out["norm"].shape == (10,)
        for key in expected:
            np.testing.assert_allclose(out[key], expected[key], rtol=1e-12)


def test_small_batch_uses_fewer_devices():
    """2 entries over 4 devices: split over 2, unpadded"""
    out = jax.jit(device_vmap(lambda x: x * 2.0))(jnp.arange(2.0))
    np.testing.assert_array_equal(out, [0.0, 2.0])
    assert len(out.sharding.device_set) == 2


def test_one_device_is_vmap():
    x = {"a": jnp.arange(6.0).reshape(3, 2), "b": jnp.ones((3, 2, 2))}
    out = device_vmap(f, jax.devices()[:1], in_axes=(0, None))(x, 1.0)
    np.testing.assert_array_equal(out["sum"], x["a"] + 1.0)


def test_devices_by_name():
    """Devices by name: "all" is every device, "default" JAX's default device
    alone, and other names are rejected"""
    assert _resolve_devices("all") == jax.devices()
    assert _resolve_devices("default") == jax.devices()[:1]
    assert _resolve_devices(jax.devices()[1:3]) == jax.devices()[1:3]
    with pytest.raises(AssertionError, match='"all"'):
        _resolve_devices("al")
