"""Online and offline motion retargeting"""

import time
from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from frax.ik.global_ik import GlobalIKSolver
from frax.ik.solver import DiffIKSolver, SolverInfo, SolverState
from frax.utils.parallel_utils import _resolve_devices, device_vmap


@jax.tree_util.register_static
class Retargeter:
    """Differential IK-based motion retargeter.

    Given a sequence of frames, this class solves for the robot
    configurations q, based on the input DiffIKSolver

    As opposed to just using a DiffIK solver, this adds logic for
    converging to an initial configuration on the first step, and
    processing a whole sequence in a single scan

    Args:
        solver (DiffIKSolver): Diff IK solver
        preprocess (Callable): Input frame -> the solver's reference
        initial_q (Callable): First input frame -> initial guess for q
        converge_max_iters (int): Max iterations to converge onto the first frame
        converge_dt (float | None): Largest convergence stepsize. Defaults to
            1 / solver.max_gain (full Gauss-Newton step)
        converge_tol (float): Convergence tolerance on max |v|
        global_ik_init (GlobalIKSolver | None): If set, solves the first frame by
            global IK before converging, to avoid poor local solutions.
        global_ik_seed (int): Seed for the global IK samples
    """

    def __init__(
        self,
        solver: DiffIKSolver,
        preprocess: Callable,
        initial_q: Callable[[object], Array],
        converge_max_iters: int = 300,
        converge_dt: float | None = None,
        converge_tol: float = 1e-3,
        global_ik_init: GlobalIKSolver | None = None,
        global_ik_seed: int = 0,
    ):
        self.solver = solver
        self.preprocess = preprocess
        self.initial_q = initial_q
        self.converge_max_iters = converge_max_iters
        if converge_dt is None:
            assert solver.max_gain is not None, "A term has no gain: pass converge_dt"
            converge_dt = 1.0 / solver.max_gain
        self.converge_dt = converge_dt
        self.converge_tol = converge_tol
        self.global_ik_init = global_ik_init
        self.global_ik_seed = global_ik_seed

    def init(self, frame) -> tuple[SolverState, Array]:
        """Converge onto the first frame

        Returns:
            state (SolverState): Converged state
            iters (Array): Number of iterations
        """
        q, ref = self.initial_q(frame), self.preprocess(frame)
        if self.global_ik_init is not None:
            key = jax.random.key(self.global_ik_seed)
            q = self.global_ik_init.solve(q, ref, key).q
        state, iters = self.solver.converge(
            self.solver.init(q),
            ref,
            self.converge_max_iters,
            self.converge_dt,
            self.converge_tol,
        )
        return state, iters

    def step(
        self, state: SolverState, frame, dt: float
    ) -> tuple[SolverState, SolverInfo]:
        """Take one solver step towards an input frame"""
        state, _, info = self.solver.step(state, self.preprocess(frame), dt)
        return state, info

    @jax.jit
    def retarget_clip(self, frames, dt: float) -> tuple[Array, SolverInfo, Array]:
        """Retarget a full clip of input frames

        Args:
            frames (Any): Input frames, with leading time axis T (any pytree)
            dt (float): Time between frames

        Returns:
            q (Array): Configurations, shape (T, nq)
            infos (SolverInfo): Per-step info, leading dim T - 1
            init_iters (Array): Iterations to converge onto the first frame
        """
        first = jax.tree.map(lambda x: x[0], frames)
        rest = jax.tree.map(lambda x: x[1:], frames)
        init_state, init_iters = self.init(first)

        def scan_step(state, frame):
            state, info = self.step(state, frame, dt)
            return state, (state.q, info)

        _, (q, infos) = jax.lax.scan(scan_step, init_state, rest)
        return jnp.concatenate([init_state.q[None], q]), infos, init_iters


def _pad_with_last_frame(clip, length: int):
    return jax.tree.map(
        lambda x: np.concatenate([x, np.repeat(x[-1:], length - len(x), 0)]), clip
    )


class OfflineRetargeter:
    """Retargets many clips, spread across devices. Clips are padded to a multiple
    of bucket frames, and compiled shapes are cached across calls

    Args:
        retargeter (Retargeter): Retargeter
        dt (float): Time between frames
        bucket (int): Padded clip lengths are multiples of this
        devices ("all" | "default" | Sequence[jax.Device]): Devices to spread the
            clips across. Defaults to "all"
    """

    def __init__(
        self,
        retargeter: Retargeter,
        dt: float,
        bucket: int = 1024,
        devices: Literal["all", "default"] | Sequence[jax.Device] = "all",
    ):
        self.devices = _resolve_devices(devices)
        self.bucket = bucket
        self.retarget = jax.jit(
            device_vmap(
                lambda clip: retargeter.retarget_clip(clip, dt)[0], self.devices
            )
        )
        self.compiled = {}
        self.compile_time = 0.0

    def __call__(self, clips: Sequence) -> list[np.ndarray]:
        """Retarget clips

        Args:
            clips (Sequence): Clips of input frames, each a pytree with leading
                time axis T. T may differ between clips

        Returns:
            list[np.ndarray]: Configurations, one array per clip, each shape
                (T, nq) with that clip's T
        """
        n = len(self.devices)
        lengths = [len(jax.tree.leaves(clip)[0]) for clip in clips]
        order = sorted(range(len(clips)), key=lambda i: -lengths[i])
        groups = [order[i : i + n] for i in range(0, len(order), n)]
        pending = []
        for group in groups:
            length = -(-lengths[group[0]] // self.bucket) * self.bucket
            # Top up the last group so it shares the compiled shape
            members = [clips[i] for i in group] + [clips[group[0]]] * (n - len(group))
            batch = jax.tree.map(
                lambda *x: np.stack(x),
                *[_pad_with_last_frame(clip, length) for clip in members],
            )
            leaves, structure = jax.tree.flatten(batch)
            key = (structure, tuple(x.shape for x in leaves))
            if key not in self.compiled:
                start = time.perf_counter()
                self.compiled[key] = self.retarget.lower(batch).compile()
                self.compile_time += time.perf_counter() - start
            # Asynchronous dispatch
            pending.append(self.compiled[key](batch))
        qs = [None] * len(clips)
        for group, q in zip(groups, pending):
            q = np.asarray(q)
            for k, i in enumerate(group):
                qs[i] = q[k, : lengths[i]]
        return qs


class OnlineRetargeter:
    """Stateful retargeting wrapper for online operation.

    Call compile() before any strict real-time operation to
    compile the performance-critical methods.
    """

    def __init__(self, retargeter: Retargeter):
        self.retargeter = retargeter
        self.state = None
        self._init = jax.jit(retargeter.init)
        self._step = jax.jit(retargeter.step)

    def compile(self, example_frame) -> None:
        """Compile the init and step functions ahead of time

        Args:
            example_frame (Any): Example input frame (may be jax.ShapeDtypeStructs)
        """
        state, _ = jax.eval_shape(self._init, example_frame)
        self._init.lower(example_frame).compile()
        self._step.lower(state, example_frame, 0.0).compile()

    def reset(self) -> None:
        self.state = None

    def step(self, frame, dt: float) -> Array:
        if self.state is None:
            self.state, _ = self._init(frame)
        else:
            self.state, _ = self._step(self.state, frame, float(dt))
        return self.state.q
