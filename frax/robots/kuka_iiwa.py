import numpy as np

from frax.assets import KUKA_ASSETS_DIR
from frax.core.manipulator import Manipulator
from frax.utils.collision_utils import bubblify_to_mine


collision_model_file = KUKA_ASSETS_DIR / "bubblify/iiwa14_spherized.yml"
root_link_name = "iiwa_link_0"
joint_to_child_mapping = {
    "iiwa_joint_1": "iiwa_link_1",
    "iiwa_joint_2": "iiwa_link_2",
    "iiwa_joint_3": "iiwa_link_3",
    "iiwa_joint_4": "iiwa_link_4",
    "iiwa_joint_5": "iiwa_link_5",
    "iiwa_joint_6": "iiwa_link_6",
    "iiwa_joint_7": "iiwa_link_7",
}
joint_ordering = tuple(joint_to_child_mapping.keys())
link_ordering = tuple(joint_to_child_mapping.values())


# NOTE: this self collision data is calibrated for this specific collision model
# TODO: should these be named tuples?
kuka_sc_data = (
    # [first link], [first link sphere idx], [second link], [second link sphere idx], [tol]
    ("iiwa_link_0", 0, "iiwa_link_6", 1, 0.0),
    # ("iiwa_link_1", 0, "iiwa_link_6", 1, 0.0),
    # ("iiwa_link_1", 2, "iiwa_link_6", 1, 0.0),
    # ("iiwa_link_2", 0, "iiwa_link_6", 1, 0.0),
    # ("iiwa_link_5", 0, "iiwa_link_6", 1, 0.0),
    # ("iiwa_link_5", 2, "iiwa_link_1", 2, 0.0),
    # ("iiwa_link_5", 3, "iiwa_link_1", 2, 0.0),
    # NOT DONE!
)


def load_iiwa() -> Manipulator:
    """Create a Manipulator object for the Kuka iiwa"""

    return Manipulator(
        KUKA_ASSETS_DIR / "iiwa14.urdf",
        ee_offset=np.block(
            [
                [np.eye(3), np.reshape(np.array([0.0, 0.0, 0.0]), (-1, 1))],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        collision_data=bubblify_to_mine(
            collision_model_file,
            joint_to_child_mapping,
            root_link_name=root_link_name,
            sc_data=kuka_sc_data,
            add_floating_base=False,
            verbose=False,
        ),
    )


def main():
    robot = load_iiwa()
    # Reasonable starting joint position for the iiwa
    q = np.array([0.0, np.pi / 6, 0.0, -np.pi / 2, 0.0, np.pi / 3, 0.0])
    qd = 0.1 * np.ones(robot.num_joints)
    transforms = robot.joint_to_world_transforms(q)
    M = robot._mass_matrix(transforms)
    c = robot._centrifugal_coriolis_vector(qd, transforms)
    g = robot._gravity_vector(transforms)
    J_rh = robot._ee_jacobian(transforms)
    coll_pos, coll_rad = robot._link_collision_data(transforms)
    mu_rh = robot._ee_manipulability_index(transforms)
    np.set_printoptions(suppress=True, precision=3, linewidth=300, threshold=1e5)
    print(f"\nMass Matrix:\n{M}")
    print(f"\nCentrifugal/Coriolis Vector:\n{c}")
    print(f"\nGravity Vector:\n{g}")
    print(f"\nAncestor mask: \n{robot.ancestor_mask}")
    print(f"\nEE Jacobian: \n{J_rh}")
    print(f"\nCollision positions: \n{coll_pos}")
    print(f"\nCollision radii: \n{coll_rad}")
    print(f"\nEE manipulability index: {mu_rh}")


if __name__ == "__main__":
    main()
