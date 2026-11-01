"""Parallelism across devices, e.g. CPU cores exposed as JAX devices

A JAX device runs one program at a time, so a batch on one CPU device is solved
serially (elastiqp solves a vmapped batch of QPs one by one). To use several
cores, expose them as devices before JAX first runs:

    jax.config.update("jax_num_cpu_devices", N)

(or XLA_FLAGS=--xla_force_host_platform_device_count=N), then split batches
across them with `device_vmap`"""

from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def _resolve_devices(
    devices: Literal["all", "default"] | Sequence[jax.Device],
) -> list[jax.Device]:
    """The devices named by devices: "all" (jax.devices()), "default" (JAX's
    default device, the first of jax.devices()), or the devices themselves.
    Resolved at run time, as jax.devices() starts JAX's backends"""
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
    """jax.vmap(f, in_axes), with the batch split across devices, each vmapping
    its share. The batch is padded to a multiple of the number of devices (with
    copies of its first entry), and a smaller batch uses fewer devices

    Args:
        f (Callable): Function of one batch entry
        devices ("all" | "default" | Sequence[jax.Device]): Devices to split the
            batch across: all of jax.devices(), JAX's default device, or these.
            One device is a plain vmap
        in_axes: As for jax.vmap, but each entry 0 (batched along the leading
            axis) or None (shared by the whole batch)

    Returns:
        Callable: The batched function, with outputs batched along axis 0.
            Wrap it in jax.jit (for one compile for the whole call)
    """

    def batched(*args):
        axes = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
        assert all(a in (0, None) for a in axes), "in_axes entries must be 0 or None"
        mapped = [a for a, axis in zip(args, axes) if axis == 0]
        n = jax.tree.leaves(mapped)[0].shape[0]
        # No more devices than entries: padding a small batch out to every
        # device only waits on the slowest of them (e.g. efficiency cores)
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
        out = jax.shard_map(
            vf,
            mesh=Mesh(np.asarray(devs), ("batch",)),
            in_specs=tuple(P("batch") if axis == 0 else P() for axis in axes),
            out_specs=P("batch"),
            check_vma=False,  # elastiqp's FFI call has no varying-axes rule
        )(*args)
        return jax.tree.map(lambda x: x[:n], out)

    return batched
