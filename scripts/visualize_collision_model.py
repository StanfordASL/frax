"""Visualize the robot collision model using viser

NOTE
While frax's kinematics and dynamics were carefully hand-tuned,
this script was heavily Claude-assisted, with some manual touch-ups
here and there
"""

import argparse
import time
from pathlib import Path

import jax
import numpy as np
import viser
from viser.extras import ViserUrdf

from frax import load_g1, load_panda, Robot
from frax.assets import G1_ASSETS_DIR, FRANKA_ASSETS_DIR
from frax.utils.rotation_utils import intrinsic_euler_xyz_to_quat_wxyz


# Colors
SPHERE_SC_COLOR = (220, 60, 60)  # Red — involved in self-collision pairs
SPHERE_DEFAULT_COLOR = (240, 200, 40)  # Yellow — not in any SC pair
LINE_SC_COLOR = (220, 60, 60)  # Red — self-collision pair lines
MESH_OPACITY = 0.4  # Semi-transparent robot meshes
SPHERE_OPACITY = 0.55
ROOT_SPHERE_COLOR = (220, 60, 60)  # Red — root spheres (always SC-relevant)


def visualize_collision_model(
    urdf_path: Path,
    robot: Robot,
    q_initial: np.ndarray,
    port: int = 8080,
):
    """Visualize the collision model using viser.

    Args:
        urdf_path: Path to the URDF file
        robot: frax Robot object
        q_initial: Initial joint configuration (frax convention)
        port: Viser server port
    """

    @jax.jit
    def jit_collision_positions(q):
        positions, radii = robot.link_collision_data(q)
        return positions

    server = viser.ViserServer(port=port)

    # Update the root state of the robot in viser with quaternion convention
    if robot.includes_floating_dof:
        pos = q_initial[:3]
        quat_wxyz = intrinsic_euler_xyz_to_quat_wxyz(q_initial[3:6])
        server.scene.add_frame(
            "/robot",
            position=tuple(pos),
            wxyz=tuple(quat_wxyz),
            show_axes=False,
        )

    # Load meshes
    urdf_viz = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/robot",
        mesh_color_override=(0.8, 0.8, 0.8, MESH_OPACITY),
    )

    # --- Set initial URDF configuration ---
    # Map frax joint names to yourdfpy actuated joint names for slider sync
    urdf_joint_names = urdf_viz.get_actuated_joint_names()
    urdf_joint_limits = urdf_viz.get_actuated_joint_limits()

    # Build mapping from frax joint index to yourdfpy joint index
    frax_to_urdf_map = {}
    start_idx = 6 if robot.includes_floating_dof else 0
    for i in range(start_idx, robot.num_joints):
        frax_name = robot.joint_names[i]
        # Try direct match first
        if frax_name in urdf_joint_names:
            urdf_idx = urdf_joint_names.index(frax_name)
            frax_to_urdf_map[i] = urdf_idx
        else:
            # Try stripping common prefixes
            for prefix in ["panda_", "g1_"]:
                if frax_name.startswith(prefix):
                    stripped = frax_name[len(prefix) :]
                    if stripped in urdf_joint_names:
                        urdf_idx = urdf_joint_names.index(stripped)
                        frax_to_urdf_map[i] = urdf_idx
                        break

    # Set initial URDF pose
    urdf_cfg = np.zeros(len(urdf_joint_names))
    for frax_idx, urdf_idx in frax_to_urdf_map.items():
        urdf_cfg[urdf_idx] = q_initial[frax_idx]
    urdf_viz.update_cfg(urdf_cfg)

    # --- Determine which body sphere indices are involved in self-collision ---
    sc_body_idxs = set()
    if robot.has_sc_data:
        for pair in robot.body_sc_pairs:
            sc_body_idxs.add(int(pair[0]))
            sc_body_idxs.add(int(pair[1]))
        if robot.has_root_collision_data:
            for pair in robot.root_sc_pairs:
                sc_body_idxs.add(int(pair[1]))  # body idx in root-body pairs

    collision_positions = np.asarray(jit_collision_positions(q_initial))

    sphere_handles = []
    for idx, (pos, rad) in enumerate(
        zip(collision_positions, robot.flat_collision_radii)
    ):
        color = SPHERE_SC_COLOR if idx in sc_body_idxs else SPHERE_DEFAULT_COLOR
        handle = server.scene.add_icosphere(
            f"/collision/body_sphere_{idx}",
            radius=float(rad),
            color=color,
            position=tuple(pos),
            opacity=SPHERE_OPACITY,
        )
        sphere_handles.append(handle)

    # Root spheres (fixed in world frame, don't change with joint config)
    root_sphere_handles = []
    if robot.has_root_collision_data:
        for idx, (pos, rad) in enumerate(
            zip(robot.root_collision_positions, robot.root_collision_radii)
        ):
            handle = server.scene.add_icosphere(
                f"/collision/root_sphere_{idx}",
                radius=float(rad),
                color=ROOT_SPHERE_COLOR,
                position=tuple(pos),
                opacity=SPHERE_OPACITY,
            )
            root_sphere_handles.append(handle)

    # --- Self-collision pair lines ---
    line_handle = None
    root_line_handle = None

    def _build_sc_lines(flat_positions):
        """Build line segment arrays for self-collision pairs."""
        nonlocal line_handle, root_line_handle

        # Body-to-body SC lines
        if robot.has_sc_data and len(robot.body_sc_pairs) > 0:
            segments = []
            for pair in robot.body_sc_pairs:
                p1 = np.asarray(flat_positions[int(pair[0])])
                p2 = np.asarray(flat_positions[int(pair[1])])
                segments.append([p1, p2])

            # Shape: (N, 2, 3) — N line segments, each with 2 endpoints of 3 coords
            points_arr = np.array(segments, dtype=np.float32)

            if line_handle is not None:
                line_handle.remove()
            line_handle = server.scene.add_line_segments(
                "/collision/sc_lines",
                points=points_arr,
                colors=LINE_SC_COLOR,
                line_width=2.0,
            )

        # Root-to-body SC lines
        if robot.has_root_collision_data and len(robot.root_sc_pairs) > 0:
            root_segments = []
            for pair in robot.root_sc_pairs:
                p1 = np.asarray(robot.root_collision_positions[int(pair[0])])
                p2 = np.asarray(flat_positions[int(pair[1])])
                root_segments.append([p1, p2])

            root_points_arr = np.array(root_segments, dtype=np.float32)

            if root_line_handle is not None:
                root_line_handle.remove()
            root_line_handle = server.scene.add_line_segments(
                "/collision/root_sc_lines",
                points=root_points_arr,
                colors=LINE_SC_COLOR,
                line_width=2.0,
            )

    _build_sc_lines(collision_positions)

    # --- Joint sliders ---
    with server.gui.add_folder("Joint Controls"):
        joint_sliders = []
        initial_values = []

        for joint_name, (lower, upper) in urdf_joint_limits.items():
            lower = lower if lower is not None else -np.pi
            upper = upper if upper is not None else np.pi
            # Find initial value from our config
            init_val = 0.0
            if joint_name in urdf_joint_names:
                idx = urdf_joint_names.index(joint_name)
                init_val = float(urdf_cfg[idx])

            slider = server.gui.add_slider(
                label=joint_name,
                min=lower,
                max=upper,
                step=1e-3,
                initial_value=init_val,
            )
            joint_sliders.append(slider)
            initial_values.append(init_val)

        def update_visualization():
            """Recompute FK and update sphere positions + SC lines."""
            # Build urdf config from sliders
            cfg = np.array([s.value for s in joint_sliders])
            urdf_viz.update_cfg(cfg)

            # Build frax q from sliders
            q = q_initial.copy()
            for frax_idx, urdf_idx in frax_to_urdf_map.items():
                q[frax_idx] = cfg[urdf_idx]

            # Recompute sphere positions
            updated_positions = np.asarray(jit_collision_positions(q))

            # Update sphere handle positions
            for handle, pos in zip(sphere_handles, updated_positions):
                handle.position = tuple(pos)

            # Update SC lines
            _build_sc_lines(updated_positions)

        for slider in joint_sliders:
            slider.on_update(lambda _: update_visualization())

        # Reset button
        reset_btn = server.gui.add_button("Reset to Initial")

        @reset_btn.on_click
        def _(_):
            for slider, init_val in zip(joint_sliders, initial_values):
                slider.value = init_val

    # --- Visibility controls ---
    with server.gui.add_folder("Visibility"):
        show_meshes = server.gui.add_checkbox("Robot Meshes", initial_value=True)
        show_spheres = server.gui.add_checkbox("Collision Spheres", initial_value=True)
        show_lines = server.gui.add_checkbox("SC Pair Lines", initial_value=True)

        mesh_opacity_slider = server.gui.add_slider(
            "Mesh Opacity", min=0.0, max=1.0, step=0.05, initial_value=MESH_OPACITY
        )

        @show_meshes.on_update
        def _(_):
            for mesh in urdf_viz._meshes:
                mesh.visible = show_meshes.value

        @show_spheres.on_update
        def _(_):
            for h in sphere_handles:
                h.visible = show_spheres.value
            for h in root_sphere_handles:
                h.visible = show_spheres.value

        @show_lines.on_update
        def _(_):
            if line_handle is not None:
                line_handle.visible = show_lines.value
            if root_line_handle is not None:
                root_line_handle.visible = show_lines.value

        @mesh_opacity_slider.on_update
        def _(_):
            for mesh in urdf_viz._meshes:
                mesh.opacity = mesh_opacity_slider.value

    # --- Reference grid ---
    server.scene.add_grid(
        "/grid",
        width=2.0,
        height=2.0,
        position=(0.0, 0.0, 0.0),
        cell_color=(180, 180, 180),
        cell_thickness=1.0,
    )

    print("\nCollision model info:")
    print(f"  Body spheres:     {len(collision_positions)}")
    print(
        f"  Root spheres:     {len(robot.root_collision_positions) if robot.has_root_collision_data else 0}"
    )
    print(
        f"  Body self-collision pairs:    {len(robot.body_sc_pairs) if robot.has_sc_data else 0}"
    )
    print(
        f"  Body-root self-collision pairs:    {len(robot.root_sc_pairs) if robot.has_root_collision_data else 0}"
    )
    print(f"\nOpen http://localhost:{port} in your browser to view.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


def panda_main():
    robot = load_panda()
    q = np.array([0.0, -np.pi / 6, 0.0, -3 * np.pi / 4, 0.0, 5 * np.pi / 9, 0.0])
    urdf_path = FRANKA_ASSETS_DIR / "panda.urdf"
    visualize_collision_model(urdf_path, robot, q)


def g1_main():
    robot = load_g1()
    q = np.zeros(robot.num_joints)
    # Move the robot up a bit so it's not in the floor
    q[2] = 0.8
    urdf_path = G1_ASSETS_DIR / "g1_29dof_rev_1_0.urdf"
    visualize_collision_model(urdf_path, robot, q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize robot collision model in viser"
    )
    parser.add_argument("--robot", choices=["panda", "g1"], default="panda")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    if args.robot == "panda":
        panda_main()
    else:
        g1_main()
