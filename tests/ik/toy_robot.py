"""A configuration space without kinematics, for toy problems in the tests"""

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_static
class ToyRobot:
    """n entries with q_dot = v, and frax's configuration interface"""

    def __init__(self, n: int):
        self.nq = self.nv = n
        self.nq_floating = self.nv_floating = 0
        self.velocity_to_configuration_index = {i: i for i in range(n)}
        # No joint limits of its own (JointLimits' defaults)
        self.actuated_joint_lower_limits = np.full(n, -np.inf)
        self.actuated_joint_upper_limits = np.full(n, np.inf)
        self.actuated_joint_max_velocities = np.full(n, np.inf)

    def integrate(self, q, v, dt):
        return q + v * dt

    def velocity_to_qdot_map(self, q):
        return jnp.eye(self.nv, dtype=q.dtype)
