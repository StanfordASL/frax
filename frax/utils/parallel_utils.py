"""Parallelism across CPU devices with JAX

To use multiple CPU cores, set XLA_FLAGS=--xla_force_host_platform_device_count=N,
or on newer JAX, call jax.config.update("jax_num_cpu_devices", N) before JAX
first runs
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# Patch for older jax, which has shard_map only under experimental
if hasattr(jax, "shard_map"):
    _shard_map = partial(jax.shard_map, check_vma=False)
else:
    from jax.experimental.shard_map import shard_map

    _shard_map = partial(shard_map, check_rep=False)


def _resolve_devices(
    devices: Literal["all", "default"] | Sequence[jax.Device],
) -> list[jax.Device]:
    if isinstance(devices, str):
        assert devices in ("all", "default"), (
            f'devices must be "all", "default", or devices, got {devices!r}'
        )
        return list(jax.devices()) if devices == "all" else jax.devices()[:1]
    return list(devices)


def device_vmap(
    f: Callable,
    devices: Literal["all", "default"] | Sequence[jax.Device] = "all",
    in_axes: int | None | tuple[int | None, ...] = 0,
) -> Callable:
    """jax.vmap(f, in_axes), with the batch split and padded across devices

    Args:
        f (Callable): Function of one batch entry
        devices ("all" | "default" | Sequence[jax.Device]): Devices to split the
            batch across: all of jax.devices(), the default device, or these
        in_axes (int | None | tuple[int | None, ...]): As for jax.vmap, but each
            entry 0 or None

    Returns:
        Callable: Batched function, with outputs batched along axis 0.
            Note: remember to wrap the function in jax.jit
    """

    def batched(*args):
        axes = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
        assert all(a in (0, None) for a in axes), "in_axes entries must be 0 or None"
        mapped = [a for a, axis in zip(args, axes) if axis == 0]
        n = jax.tree.leaves(mapped)[0].shape[0]
        devs = _resolve_devices(devices)[:n]
        vf = jax.vmap(f, in_axes=axes)
        if len(devs) <= 1:
            return vf(*args)
        pad = -n % len(devs)

        def pad_leaf(x):
            x = jnp.asarray(x)
            return jnp.concatenate([x, jnp.broadcast_to(x[:1], (pad,) + x.shape[1:])])

        args = [
            jax.tree.map(pad_leaf, a) if axis == 0 else a for a, axis in zip(args, axes)
        ]
        out = _shard_map(
            vf,
            mesh=Mesh(np.asarray(devs), ("batch",)),
            in_specs=tuple(P("batch") if axis == 0 else P() for axis in axes),
            out_specs=P("batch"),
        )(*args)
        return jax.tree.map(lambda x: x[:n], out)

    return batched
