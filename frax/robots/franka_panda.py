import numpy as np

from frax.assets import FRANKA_ASSETS_DIR
from frax.core.manipulator import Manipulator
from frax.utils.rotation_utils import Rz
from frax.utils.collision_utils import bubblify_to_mine


collision_model_file = FRANKA_ASSETS_DIR / "bubblify/panda_spherized.yml"
root_link_name = "panda_link0"
joint_to_child_mapping = {
    "panda_joint1": "panda_link1",
    "panda_joint2": "panda_link2",
    "panda_joint3": "panda_link3",
    "panda_joint4": "panda_link4",
    "panda_joint5": "panda_link5",
    "panda_joint6": "panda_link6",
    "panda_joint7": "panda_link7",
}
joint_ordering = tuple(joint_to_child_mapping.keys())
link_ordering = tuple(joint_to_child_mapping.values())


# NOTE: this self collision data is calibrated for this specific collision model
# TODO: should these be named tuples?
panda_sc_data = (
    # [first link], [first link sphere idx], [second link], [second link sphere idx], [tol]
    ("panda_link0", 0, "panda_link7", 4, 0.0736826),
    ("panda_link1", 0, "panda_link7", 4, 0.0736826),
    ("panda_link1", 2, "panda_link7", 4, 0.0736826),
    ("panda_link2", 0, "panda_link7", 4, 0.0736826),
    ("panda_link5", 0, "panda_link7", 4, 0.0736826),
    ("panda_link5", 2, "panda_link1", 2, 0.0),
    ("panda_link5", 3, "panda_link1", 2, 0.0),
)

default_q = np.array([0.0, -np.pi / 6, 0.0, -3 * np.pi / 4, 0.0, 5 * np.pi / 9, 0.0])


def load_panda() -> Manipulator:
    """Create a Manipulator object for the Franka Panda"""

    return Manipulator(
        FRANKA_ASSETS_DIR / "panda.urdf",
        ee_offset=np.block(
            [
                [Rz(-np.pi / 4), np.reshape(np.array([0.0, 0.0, 0.212]), (-1, 1))],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        collision_data=bubblify_to_mine(
            collision_model_file,
            joint_to_child_mapping,
            root_link_name=root_link_name,
            sc_data=panda_sc_data,
            add_floating_base=False,
            verbose=False,
        ),
    )


def main():
    # Quick validation that the manipulator class works
    robot = load_panda()
    qd = 0.1 * np.ones(robot.num_joints)
    transforms = robot.joint_to_world_transforms(default_q)
    M = robot._mass_matrix(transforms)
    c = robot._centrifugal_coriolis_vector(qd, transforms)
    g = robot._gravity_vector(transforms)
    J_rh = robot._ee_jacobian(transforms)
    coll_pos, coll_rad = robot._link_collision_data(transforms)
    mu_rh = robot._ee_manipulability_index(transforms)
    ee_tf = robot._ee_transform(transforms)
    np.set_printoptions(suppress=True, precision=3, linewidth=300, threshold=1e5)
    print(f"\nMass Matrix:\n{M}")
    print(f"\nCentrifugal/Coriolis Vector:\n{c}")
    print(f"\nGravity Vector:\n{g}")
    print(f"\nAncestor mask: \n{robot.ancestor_mask}")
    print(f"\nEE Jacobian: \n{J_rh}")
    print(f"\nCollision positions: \n{coll_pos}")
    print(f"\nCollision radii: \n{coll_rad}")
    print(f"\nEE manipulability index: {mu_rh}")
    print(f"\nEE transform: \n{ee_tf}")


if __name__ == "__main__":
    main()
