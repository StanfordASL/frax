<div align="center">
<img src="https://github.com/user-attachments/assets/a3476e2f-38a6-4f4d-95df-a399f7881eae" alt="logo"></img>
</div>

# `frax`: Fast Robot Kinematics and Dynamics in JAX

[![Paper](http://img.shields.io/badge/arXiv-2604.04310-B31B1B.svg)](https://arxiv.org/abs/2604.04310)

`frax` is a fast kinematics and dynamics library in pure Python, using [JAX](https://github.com/jax-ml/jax) for JIT-compilation and automatic differentiation. 

With `frax`, you can design high-performance inverse-kinematics and inverse-dynamics controllers at the speed of [Pinocchio](https://github.com/stack-of-tasks/pinocchio), ease of use of Python, and differentiation and parallelization-compatibility of [MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx).

On CPU, you can expect compute times for typical controllers in the *low microseconds range (~25-100 kHz)*, and on GPU or TPU, `frax` can compute dynamics terms at upwards of *100 million computations per second*, depending on your batch size. 

> [!IMPORTANT]
> `frax` is actively under development and internal operations may change between beta versions

## Installation

### From PyPI

```
pip install frax
```

### From source

```
git clone https://github.com/danielpmorton/frax
cd frax
pip install -e .
```

> [!TIP]
> If you are running on CPU, I highly recommend using JAX version `0.4.30` for the best possible performance. See [below](#performance-tips) for additional performance tips!

If you're installing from source, I highly recommend using a `uv`-managed virtual environment. In this case, `pip` commands can be replaced with `uv pip`

For more fine-grained control over the JAX install for GPU/TPU, you can also use the `[cuda12]`, `[cuda13]`, or `[tpu]` tags.

To run the examples, please install from source with the `[examples]` tag, i.e. `pip install -e ".[examples]"`

## Examples

`frax`'s "hello world": Compute your robot's mass matrix (AKA joint-space inertia matrix)
```python
import frax
import numpy as np

jax.config.update("jax_enable_x64", True)  # Recommmended for high accuracy

robot = frax.Robot("path/to/your/robot.urdf")
q = np.zeros(robot.num_joints)
M = robot.mass_matrix(q)
print(M)
```
For any operations with `frax` and `jax` more broadly, you should wrap your code in a jitted region, for instance,
```python
@jax.jit
def jit_mass_matrix(q_):
    return robot.mass_matrix(q_)

M = jit_mass_matrix(q)
print(M)
```
See the [Performance Tips](#performance-tips) section below for more advice on making your code *fast*.

Many more kinematics and dynamics terms are available (joint/link/frame transforms and Jacobians, gravity vector, centrifugal/coriolis forces, and many other values relevant to robot control). We also provide `Manipulator` and `Humanoid` classes for useful helper functions based on your robot's form-factor, and `frax` comes pre-loaded with the Franka Panda and Unitree G1. 

More advanced and interactive demos are included in the `examples` directory, as seen below

> [!NOTE]
> Remember to install from source with the `[examples]` dependencies

Examples of using `frax` for typical robot controllers:
- Differential inverse kinematics: `examples/diff_ik_demo.py`
- Operational space control: `examples/osc_demo.py`

When these scripts are launched in `trajectory` mode, you'll see the following sinusoidal tracking demo:

https://github.com/user-attachments/assets/7b471496-2124-4063-953a-d7f25776ed5b

This can also be launched in `manual` mode, for interactive mouse control of the target.

However, these controllers are fairly simple, and Pinocchio or MuJoCo could have been used to give the same Jacobians and inertial values. The real benefit of `frax` comes from the automatic differentiation through the kinematics and dynamics for flexible controller design with minimal manual Jacobian derivations. In the below demo (`examples/oscbf_demo.py`), I've reimplemented [OSCBF](https://github.com/StanfordASL/oscbf), which uses JAX's autodiff under the hood to form the CBF constraints.

https://github.com/user-attachments/assets/5ea84ec0-6b67-40fd-a974-5850499df15d

Here, we're enforcing
- Singularity avoidance
- Joint limit avoidance
- Collision avoidance with the floor
- Collision avoidance with an external obstacle
- Self-collision avoidance

... all with easy prototyping of the constraint design, in simple functions of the robot kinematics and dynamics (such as, `robot.ee_manipulability_index(q) >= eps`).

## Usage tips

- If you would like to use `frax`'s collision methods, you must first define a spherized collision model of your robot (or, use our pre-built collision models for the Franka Panda/FR3 and the Unitree G1). Check out [this page](docs/modeling_collision.md) for more info!
- For now, if you have joints in your URDF that are not part of the primary kinematic chain/tree being controlled (for instance, gripper joints), please set these as `fixed` so that they can be ignored, and so their child links' inertias can be fused into the parent. In the future, we will allow for fixing joints programmatically.


## Performance tips

- Currently, for the best performance on CPU, I recommend JAX version `0.4.30`. This is due to a degradation in the XLA compiler performance in recent JAX versions, but the JAX/XLA team is currently working on addressing this: see https://github.com/jax-ml/jax/issues/26021
- I also recommend using double precision with `jax.config.update("jax_enable_x64", True)` or by setting the environment variable `JAX_ENABLE_X64=True`. This is especially the case if you are running on CPU, whereas on GPU, this is a more nuanced decision. Generally, running in double precision on GPU will lead to about a 2-6x slowdown, depending on your batch size, because GPUs are so well-suited for single-precision operations. This choice will be application-dependent -- you might need high precision, or maybe you can get away with being a little less precise.
- I've also found that for typical CPU workflows with `frax`, it's better to configure JAX/XLA to run on a single thread. Generally, for robot controller design, the matrices involved are not quite large enough to require multithreading (unless you are doing something more parallelizable like MPPI). In this case, restricting it to a single thread can improve performance *and* also make it easier to share CPU resources across processes (multiple ROS nodes, for instance). You can use `source scripts/setup_cpu_env.sh` to load my recommended config, or feel free to modify as needed. 
- If you are designing QP-based controllers, the QP should (almost always) be solved in double precision, otherwise you may get garbage results.
- If you only need to solve for the dynamics of **one** robot instance, and you have a GPU on your computer, it will be significantly faster to use CPU. To force CPU usage, use `jax.config.update("jax_platforms", "cpu")` or set the environment variable `JAX_PLATFORMS=cpu`.
- `frax` by default does **not** wrap every method with a `@jax.jit` decorator. When calling any JAX code, add a JIT decorator to the top-most function call to ensure the best performance. 
- If you have any code that is **outside** of a jitted region, use `numpy` operations and arrays. **Inside** a jitted region, use `jax.numpy`. 

For general advice on JAX, check out the [quickstart guide](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html) and the [sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## TODOs / upcoming features

`frax` is (by design) more minimal than other libraries -- resulting in high performance on the restricted setting of interest. But, there are a few more features I'd like to add in the future:

- Add MJCF support
- Add support for collision primitives other than just spheres
- Use a quaternion representation for free-floating angular DOFs
- Add Quadruped class
- Analytical Jacobians of forward/inverse dynamics (see: Pinocchio)

The following features are unplanned:
- Closed kinematic chains
- Joint types other than revolute, prismatic, fixed, or free-floating


## Other recommended resources

`frax` might not serve your needs exactly -- that's fine! Here are some other useful repositories to look at
- [stack-of-tasks/pinocchio](https://github.com/stack-of-tasks/pinocchio) -- Robot kinematics + dynamics (C++/Python)
- [google/mujoco/mjx](https://github.com/google-deepmind/mujoco/tree/main/mjx) -- Parallelized simulation (JAX)
- [google/brax](https://github.com/google/brax) -- Parallelized simulation (JAX)
- [chungmin99/pyroki](https://github.com/chungmin99/pyroki) -- Global IK and kinematic optimization (JAX)
- [stephane-caron/pink](https://github.com/stephane-caron/pink) -- Differential IK with Pinocchio (Python)
- [kevinzakka/mink](https://github.com/kevinzakka/mink) -- Differential IK with MuJoCo (Python)
- [StanfordASL/cbfpy](https://github.com/StanfordASL/cbfpy) -- Control barrier functions (JAX)
- [StanfordASL/oscbf](https://github.com/StanfordASL/oscbf) -- Safe manipulator control (JAX)


## Citation

```
@article{morton2026frax,
  author={Morton, Daniel and Pavone, Marco},
  title={frax: Fast Robot Kinematics and Dynamics in JAX},
  journal={arXiv preprint arXiv:2604.04310},
  year={2026},
  note={ICRA 2026 Workshop on Frontiers of Optimization for Robotics},
}
```

