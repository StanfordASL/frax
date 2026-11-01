# Modeling collision geometry for a new robot in `frax`

**`frax`'s kinematics and dynamics already supports most robots you can load with a URDF** (MJCF support is pending).

But, if you would like to use `frax`'s collision / self-collision modeling, you'll need to add a spherized collision model and define self-collision pairs. 

Currently, `frax` contains tuned collision models for the following:
- Franka Panda / FR3
- Unitree G1

There are a few tools out there to automatically spherize your robot geometry, for instance, [foam](https://github.com/CoMMALab/foam) or [ballpark](https://github.com/chungmin99/ballpark), but I prefer to hand-design these collision geometries with a tool like [bubblify](https://github.com/bheijden/bubblify). It takes a short time to design a good collision model, but once this is done well, you don't have to worry about it again.

Here is a walkthrough of the steps I took to add the Franka Panda:

### 1. Find a good URDF with "accurate" inertial information. 
This can be harder than you would expect for many robots. The best place to search is usually the official `_description` repo (for instance, [franka_description](https://github.com/frankarobotics/franka_description)), and building the URDF yourself.

Some alternative methods to this are:
- Searching GitHub for "your_robot" with extension ".urdf". If it's a common robot, someone's probably uploaded the URDF already. But be warned that the inertial properties could be wrong
- Using [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py) if your robot is supported

### 2. Create a collision model
To do this with [bubblify](https://github.com/bheijden/bubblify) (my preferred method),
- Install bubblify either from source or with pip (see the repo for details)
- Load your robot with `bubblify --urdf_path path/to/your/robot.urdf`
- Add spheres interactively to each link. Tips for best performance with `frax`: minimize the maximum amount of spheres on any one link, and try to strike a good balance between number of spheres, and how tighly you represent the geometry.
- Your model will look something like this when you're done:

<img src="https://github.com/user-attachments/assets/f097b577-1197-4c7f-9392-a4b193df72ac" alt="Spherized Panda" height="300">

### 3. Create a self-collision model
Now, given the spheres from the collision model, we'll define self-collision pairs to pay attention to. This could be an exhaustive list of every pair of spheres on the model, but this would be needlessly complex for most practical situations. 

An example of how to define these pairs is located in `frax/robots/franka_panda.py` -- in general, you'll specify the link name and sphere index for both spheres, as well as an optional tolerance/inflation factor.

Alternatively, you can define the pairs by link, with `frax.utils.collision_utils.link_pairs_to_sc_data`: every sphere on one link is checked against every sphere on the other. This checks more pairs, but the pairs stay valid when you edit the spheres. See `frax/robots/unitree_g1.py` for an example. 

### 4. Verify
Now, check that the collision model looks correct once it's loaded into `frax`. There's an example script at `scripts/visualize_collision_model.py` which will bring up the following, for the Panda:

<img src="https://github.com/user-attachments/assets/3d31c668-30bb-491d-ad3e-cd2bb63ff34c" alt="Collision model" height="300">

The spheres associated with the self-collision model are indicated in red (with lines to denote the collision pairs), and standard collision spheres in yellow. 

Note that it is perfectly fine to have a self-collision model which is simplified from the full-body collision model. For instance, with the Panda, we create self-collision pairs with the central sphere on the end-effector with a tolerance to "inflate" this central EE sphere so that the full EE is covered. This significantly reduces the number of pairs that we need to check, while only introducing slightly more conservative behavior when near self-collision.
