# Inverse kinematics and retargeting: `frax.ik`

`frax.ik` does differential IK and motion retargeting on `frax` robots. A `Retargeter`'s `step` is a pure function of `(state, input frame, dt)`, so the same retargeter runs both:

- **Online**: `OnlineRetargeter` calls the jitted step once per incoming frame
- **Offline**: `OfflineRetargeter` retargets many clips, spread across devices, each in one jitted `lax.scan` (`Retargeter.retarget_clip`)

Offline, a clip of poses can first be smoothed without lag (`smooth_poses`)

## Installation

`frax.ik` needs a few extra dependencies (the `elastiqp` QP solver, and `scipy`):

```
pip install "frax[ik]"
```

or, from source, `pip install -e ".[ik]"`

## Quick start

A solver stacks tasks (weighted least-squares targets, with errors decaying at their gains, in 1/s) and constraints into one QP over the velocity `v` per step. Here, a Panda moves its end effector 20 cm to the left:

```python
import jax

jax.config.update("jax_enable_x64", True)  # The QP solver needs double precision

import numpy as np
from frax import load_panda
from frax.ik import DiffIKSolver, EETask, JointLimits, JointTask

robot = load_panda()
solver = DiffIKSolver(
    robot,
    tasks=[
        # The end-effector pose, from the reference's "ee_pos" and "ee_rmat"
        EETask(robot, gain_pos=10.0, gain_rot=5.0),
        # A weak posture task resolves the arm's redundancy (the elbow)
        JointTask(robot, np.arange(7), "q_home", gain=1.0, weight=0.1),
    ],
    constraints=[JointLimits(robot)],
)

q_home = robot.default_configuration
home = robot.ee_transform(q_home)
# The reference: every task's targets, by key
ref = {
    "ee_pos": home[:3, 3] + np.array([0.0, 0.2, 0.0]),  # 20 cm to the left
    "ee_rmat": home[:3, :3],
    "q_home": q_home,
}
step = jax.jit(solver.step)  # A step is a pure function of (state, ref, dt)
state = solver.init(q_home)
for _ in range(200):  # 2 s at 100 Hz
    state, v, info = step(state, ref, 0.01)
```

To retarget motion, wrap a solver in a `Retargeter`, with a function from each input frame to the reference (`preprocess`) and an initial guess for the first frame (`initial_q`). Then run it with `OnlineRetargeter` or `OfflineRetargeter`, or call its `init`, `step`, and `retarget_clip` directly (see `examples/ik/lafan1/lafan1_g1.py`)

## Collision avoidance

`SelfCollisionConstraint`, `ObstacleConstraint`, and `FloorConstraint` act on the robot's spherized collision model (see [Modeling Collision Geometry](modeling_collision.md)). By default, `SelfCollisionConstraint` checks the robot's own self-collision pairs, e.g. for the Unitree G1:

```python
from frax import load_g1
from frax.ik import SelfCollisionConstraint

robot = load_g1()
self_collision = SelfCollisionConstraint(robot, max_rows=64)
```

With `max_rows`, only the closest pairs are kept each step, which keeps the QP small. Size it for the most simultaneous contacts expected

## Configuration spaces

The solver takes a `frax` robot, optimizes over the velocity `v` (shape `(nv,)`), and integrates the configuration `q` (shape `(nq,)`) with the robot's `integrate`. For a fixed base, `q_dot = v`. For a quaternion floating base (`frax`'s default for humanoids), `q = [root position, root WXYZ quaternion, joints]` (as in MuJoCo's `qpos`, `nq = nv + 1`), `v = [world-frame linear velocity, body-frame angular velocity, joint velocities]`, and the quaternion is integrated on the rotation manifold

Tasks and constraints return Jacobians with respect to `v`. Autodiff-based ones (`ResidualTask`, `BarrierConstraint`) differentiate with respect to `q` and map the result to `v` with the robot's `velocity_to_qdot_map`, using forward-mode autodiff. `JointTask` indexes `v`, so it can't act on a quaternion's orientation (use a `FrameTask` or `ResidualTask` instead). `JointLimits` only covers the actuated joints (`q[nq_floating:]`, `v[nv_floating:]`), matching `frax`'s `actuated_joint_*` limits

## Examples

Install from source with the `[examples]` dependencies (`pip install -e ".[examples]"`)

Live diff IK in a [viser](https://viser.studio) viewer at http://localhost:8080:

```
python examples/ik/panda_interactive.py  # The basics: drag the Panda's end-effector target
python examples/ik/panda_obstacle.py     # Panda tracking around an obstacle you can drag
python examples/ik/a2_interactive.py     # Drag the A2 quadruped's base and feet
```

Parallelism across CPU cores, with each core exposed to JAX as a device (`device_vmap`, in `frax.utils.parallel_utils`):

```
python examples/ik/parallel_global_ik.py  # Global IK: one problem's starts split across cores
python examples/ik/lafan1/retarget.py     # Retarget the whole LAFAN1 dataset to the G1, clips across cores
python examples/ik/lafan1/view.py         # Play back the results
```

`examples/ik/lafan1/lafan1_g1.py` solves the task from [GMR](https://github.com/YanjieZe/GMR)'s `bvh_to_robot.py` with one diff IK QP per frame (GMR uses two IK stages, iterated). It needs the [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) BVH files (CC BY-NC-ND 4.0, non-commercial use only): `python examples/ik/lafan1/download.py` puts them in `data/lafan1`

The same for [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) (142k clips, 120 fps), from its uniform SOMA skeleton (one for all actors), shown next to SEED's own G1 retargets (see `examples/ik/bones_seed/dataset.py` for the data setup):

```
python examples/ik/bones_seed/retarget.py [--num_clips N]  # Retarget SEED (or N random clips) to the G1
python examples/ik/bones_seed/retarget.py --refine         # Redo clips whose first frame is off target, by global IK
python examples/ik/bones_seed/view.py                      # Play back frax's, SEED's G1, and the human, side by side
```
