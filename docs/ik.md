# Inverse kinematics and retargeting with `frax.ik`

`frax.ik` is a differential IK library, akin to [mink](https://github.com/kevinzakka/mink) and [pink](https://github.com/pink-kinematics/pink), but **faster, compatible with automatic differentiation, and with more reliable constraint enforcement.**

`frax.ik` is great for *both* online and offline IK problems. For example, applications include:

- **Online**: Real-time (>1000 Hz) velocity control of robot manipulators, online teleoperation and retargeting of humanoid robots
- **Offline**: Batch processing and retargeting of trajectories and human motion data

`frax.ik` uses [ElastiQP](https://github.com/StanfordASL/elastiqp) as the underlying QP solver for each differential IK step. This solver is *always feasible*, in the sense that it adds an (implicit) l1 relaxation on every inequality constraint. Check out the [paper](https://arxiv.org/abs/2609.19080) for more details!

## Examples

### Humanoid Retargeting

`frax.ik` can churn through retargeting problems at extreme speeds (>100,000 FPS). For comparison, [GMR](https://github.com/YanjieZe/GMR)'s stated number is 70 FPS (on a threadripper, too).


#### LAFAN1: **Retarget the entire dataset in 4 seconds on CPU:**

To fetch the dataset (CC BY-NC-ND 4.0), first run `python examples/ik/lafan1/download.py`

```
python examples/ik/lafan1/retarget.py
```

(Timing numbers are from a i9-14900KF CPU. If you add the `--self-collision` flag, it takes around 7 seconds. On my i7 laptop, it takes around 10 seconds)

#### BONES-SEED

To fetch the data for BONES-SEED, you'll need to first agree to some terms and conditions: see the [HuggingFace page](https://huggingface.co/datasets/bones-studio/seed) for more details


This dataset is significantly larger than LAFAN1, and it is a much higher framerate. Processing the full set takes roughly 1 hour on my desktop

```
python examples/ik/bones_seed/retarget.py
```

Currently, all humanoid examples are configured to use the Unitree G1, which is built-in to `frax`. If you would like to use a new robot, you'll need a URDF and a collision model: see our other docs for more details. 

### General IK problems


The best place to get started with `frax.ik` is in `examples/ik/panda_interactive.py`. Check out this file for a quick-start guide to how to define a set of IK tasks and constraints, and run the script to open up an interactive viewer for end-effector tracking!

```
python examples/ik/panda_interactive.py
```

Another interactive example is available in `examples/ik/a2_interactive.py` (quadruped IK with adjustable base pose and feet positions)

```
python examples/ik/a2_interactive.py
```