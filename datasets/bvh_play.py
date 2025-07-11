# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to load a URDF robot and play a motion capture (BVH) animation on it.

It uses the bvh-python library to parse the BVH file and maps the animation data
to the specified URDF joint names.

Usage:
    ./isaaclab.sh -p <path/to/your/python.sh> play_bvh.py

Make sure to update the `ROBOT_URDF_PATH` and `BVH_FILE_PATH` variables in the Cfg class.
"""

import os
import torch
import math

# Try to import required packages
try:
    import bvh
    from scipy.spatial.transform import Rotation as R
except ImportError as e:
    raise ImportError(
        "This script requires 'bvh-python' and 'scipy'. Please install them in your Isaac Lab Python environment:\n"
        "pip install bvh-python scipy"
    ) from e

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": False})


import omni.kit.app

from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.sim import SimulationContext, UsdFileCfg
from omni.isaac.lab.robots import Robot, RobotCfg
from omni.isaac.lab_assets import GROUND_PLANE_CFG

# Dataclass for configuration
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

##
# Configuration
##

@dataclass
class BvhPlaybackCfg:
    """Configuration for BVH playback."""
    # Path to the BVH file
    bvh_file_path: str = "C:/Users/FFTAI/Desktop/左右后撤.bvh"  # !! IMPORTANT: UPDATE THIS PATH

    # Scale factor for BVH motion. BVH files often use centimeters. If so, set this to 0.01
    scale: float = 1.0

    # Whether to loop the animation
    loop: bool = True

    # This is the most critical part: Mapping BVH joint names to URDF joint names.
    # You MUST inspect your BVH file to find the correct names.
    # The keys are BVH joint names, and values are the corresponding URDF joint names.
    # This is a plausible mapping based on common BVH naming conventions.
    bvh_to_urdf_map: Dict[str, str] = field(
        default_factory=lambda: {
            # Hips and Waist
            "Hips": "base",  # Special keyword for root motion
            "Spine1": "waist_yaw_joint", # Often BVH spine maps to waist yaw
            # Left Leg
            "LeftUpLeg": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"],
            "LeftLeg": "left_knee_pitch_joint",
            "LeftFoot": "left_ankle_pitch_joint",
            "LeftToeBase": "left_ankle_roll_joint", # Mapping toe to ankle roll is a common strategy
            # Right Leg
            "RightUpLeg": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint"],
            "RightLeg": "right_knee_pitch_joint",
            "RightFoot": "right_ankle_pitch_joint",
            "RightToeBase": "right_ankle_roll_joint",
            # Head
            "Neck": "head_pitch_joint",
            "Head": "head_yaw_joint",
            # Left Arm
            "LeftShoulder": "left_shoulder_yaw_joint", # Often BVH Shoulder maps to URDF Yaw
            "LeftArm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint"],
            "LeftForeArm": "left_elbow_pitch_joint",
            "LeftHand": ["left_wrist_pitch_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint"],
            # Right Arm
            "RightShoulder": "right_shoulder_yaw_joint",
            "RightArm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint"],
            "RightForeArm": "right_elbow_pitch_joint",
            "RightHand": ["right_wrist_pitch_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint"],
        }
    )
    # The order of Euler angles in the BVH file (e.g., 'xyz', 'zyx').
    # Check your BVH file, but 'zxy' or 'zyx' are common.
    euler_order: str = 'zxy'


@dataclass
class PlayBvhDemoCfg:
    """Configuration for the BVH playback demo."""
    # Path to the robot URDF file
    robot_urdf_path: str = "D:/comp_repo/robot-frames-tools/resources/urdfData/grmini/urdf/GRMini1T1_raw_floating_base.urdf"  # !! IMPORTANT: UPDATE THIS PATH

    # BVH playback settings
    bvh: BvhPlaybackCfg = field(default_factory=BvhPlaybackCfg)

    # Robot configuration
    robot: RobotCfg = field(default_factory=lambda: RobotCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=PlayBvhDemoCfg.robot_urdf_path,
            # Fix base to allow base motion control from the script
            activate_contact_sensors=False,
            rigid_props=None,
            articulation_props=None,
        ),
        init_state=RobotCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0), # Initial height
            joint_pos={
                ".*": 0.0,
            },
        ),
    ))


def parse_bvh(cfg: BvhPlaybackCfg, device: str) -> Tuple:
    """
    Parses a BVH file to extract motion data.

    Returns a tuple containing:
        - base_positions (torch.Tensor): Tensor of shape (num_frames, 3)
        - base_orientations (torch.Tensor): Tensor of shape (num_frames, 4) in (w, x, y, z) format.
        - joint_positions (List[Dict[str, float]]): List of dictionaries, one per frame.
        - frame_time (float): The time duration of a single frame.
    """
    print(f"Parsing BVH file: {cfg.bvh_file_path}")
    if not os.path.exists(cfg.bvh_file_path):
        raise FileNotFoundError(f"BVH file not found at: {cfg.bvh_file_path}")

    with open(cfg.bvh_file_path) as f:
        mocap = bvh.Bvh(f.read())

    num_frames = len(mocap.frames)
    frame_time = mocap.frame_time
    print(f"Found {num_frames} frames with a frame time of {frame_time:.4f} seconds.")

    base_positions = torch.zeros(num_frames, 3, device=device)
    base_orientations = torch.zeros(num_frames, 4, device=device)
    all_frames_joint_angles = []

    # Get reverse mapping for convenience
    urdf_to_bvh_map = {}
    for bvh_name, urdf_val in cfg.bvh_to_urdf_map.items():
        if isinstance(urdf_val, list):
            for u_name in urdf_val:
                urdf_to_bvh_map[u_name] = bvh_name
        else:
            urdf_to_bvh_map[urdf_val] = bvh_name


    for frame_idx in range(num_frames):
        # -- Root position and orientation
        # Note: BVH Y is typically up, Isaac Sim Z is up. We swap y and z.
        root_pos_xyz = [float(p) for p in mocap.frame_channels(frame_idx, mocap.root.name)[:3]]
        base_positions[frame_idx, 0] = root_pos_xyz[0] * cfg.scale
        base_positions[frame_idx, 1] = root_pos_xyz[2] * cfg.scale # BVH Z -> Sim Y
        base_positions[frame_idx, 2] = root_pos_xyz[1] * cfg.scale # BVH Y -> Sim Z

        root_rot_euler_deg = [float(r) for r in mocap.frame_channels(frame_idx, mocap.root.name)[3:6]]
        # Scipy Rotation expects degrees, order can be specified. It handles conversion to quaternion.
        # We also perform the coordinate system swap for orientation here.
        # BVH (Y-up) to Isaac (Z-up) rotation: (x, y, z) -> (x, z, -y)
        # We apply a 90-degree rotation around the X-axis to the whole character.
        rot_bvh = R.from_euler(cfg.euler_order, root_rot_euler_deg, degrees=True)
        rot_sim_coord = R.from_euler('x', 90, degrees=True)
        final_rot = rot_sim_coord * rot_bvh
        quat_xyzw = final_rot.as_quat()
        base_orientations[frame_idx] = torch.tensor([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], device=device) # (w, x, y, z)

        # -- Joint angles
        frame_joint_angles = {}
        bvh_joint_names = [j.name for j in mocap.get_joints()]

        for bvh_joint_name in bvh_joint_names:
            if bvh_joint_name in cfg.bvh_to_urdf_map:
                urdf_joint_target = cfg.bvh_to_urdf_map[bvh_joint_name]
                if urdf_joint_target == "base":
                    continue

                bvh_channel_names = mocap.joint_channels(bvh_joint_name)
                bvh_joint_values_deg = [float(v) for v in mocap.frame_channels(frame_idx, bvh_joint_name)]

                # Map BVH channels (Xrotation, Yrotation, Zrotation) to corresponding URDF joints
                if isinstance(urdf_joint_target, list):
                    # This assumes a 1-to-1 mapping of rotation axes, which might need adjustment
                    for i, channel in enumerate(bvh_channel_names):
                        if 'rotation' in channel.lower() and i < len(urdf_joint_target):
                            urdf_joint_name = urdf_joint_target[i]
                            frame_joint_angles[urdf_joint_name] = math.radians(bvh_joint_values_deg[i])
                else: # single joint mapping
                    # Assuming the first rotation channel maps to the single URDF joint
                    frame_joint_angles[urdf_joint_target] = math.radians(bvh_joint_values_deg[0])


        all_frames_joint_angles.append(frame_joint_angles)

    return base_positions, base_orientations, all_frames_joint_angles, frame_time


def main():
    """Main function to run the BVH playback demo."""
    # Create the configuration
    demo_cfg = PlayBvhDemoCfg()

    # Launch the simulation
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app

    with app_launcher.app.new_context():
        # Create a simulation context
        sim = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)

        # Spawn ground plane
        sim.scene.add(GROUND_PLANE_CFG)

        # Spawn the robot
        # We need to resolve the URDF path in the config before creating the robot
        demo_cfg.robot.spawn.usd_path = demo_cfg.robot_urdf_path
        robot = Robot(cfg=demo_cfg.robot)
        sim.scene.add(robot)

        # Wait for assets to load
        sim.reset()
        robot.reset()

        # -- Parse BVH data
        base_positions, base_orientations, all_frames_joint_angles, frame_time = parse_bvh(demo_cfg.bvh, sim.device)
        num_frames = base_positions.shape[0]

        # -- Prepare for simulation loop
        # Get the joint names from the robot model to create a mapping
        urdf_joint_names = robot.joint_names
        urdf_name_to_idx_map = {name: i for i, name in enumerate(urdf_joint_names)}

        # Create a tensor to hold the joint position targets
        joint_pos_target = robot.data.default_joint_pos.clone()
        frame_idx = 0

        # Adjust simulation DT to match BVH frame rate for smoother playback
        sim.set_dt(physics_dt=frame_time, rendering_dt=frame_time)

        # --- Simulation loop ---
        while simulation_app.is_running():
            # If simulation is playing, step the animation
            if sim.is_playing():
                # Check if animation is finished
                if frame_idx >= num_frames:
                    if demo_cfg.bvh.loop:
                        print("Animation finished. Looping.")
                        frame_idx = 0
                    else:
                        print("Animation finished.")
                        # Pause simulation and break loop
                        sim.pause()
                        continue

                # Set the root state (base position and orientation)
                # Note: set_root_state expects a tensor of shape (1, 13)
                root_state = torch.cat(
                    [base_positions[frame_idx], base_orientations[frame_idx], torch.zeros(6, device=sim.device)]
                ).unsqueeze(0)
                robot.set_root_state(root_state)

                # Set the joint position targets
                current_frame_angles = all_frames_joint_angles[frame_idx]
                for joint_name, angle in current_frame_angles.items():
                    if joint_name in urdf_name_to_idx_map:
                        idx = urdf_name_to_idx_map[joint_name]
                        joint_pos_target[0, idx] = angle
                    else:
                        # This warning helps debug the bvh_to_urdf_map
                        # print(f"Warning: Joint '{joint_name}' from BVH map not found in robot model.")
                        pass

                robot.set_joint_position_target(joint_pos_target.squeeze(0))

                # Increment frame counter
                frame_idx += 1

            # Step the simulation
            sim.step()


if __name__ == "__main__":
    main()