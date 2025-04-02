import json
import time

import numpy as np
import bvhsdk
import os
from isaacsim import SimulationApp

# os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
CONFIG = {"renderer": "RayTracedLighting", "headless": args.headless,
          "width": args.width, "height": args.height, "num_frames": args.num_frames}

simulation_app = SimulationApp(launch_config=CONFIG)


from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.replicator.isaac")

from omni.replicator.isaac.scripts.writers import YCBVideoWriter
import omni.kit.app
from isaacsim.core.api import World
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics
from isaaclab.assets import Articulation
from JointState import JointState
import math


def deg_to_rad(degree):
    return degree * math.pi / 180


world = World(stage_units_in_meters=1.0)

# Setting up import configuration:
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.distance_scale = 1.0

status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path="C:/Users/FFTAI/Desktop/GIFT_v1.0.0.6_win/win-unpacked/resources/app.asar.unpacked/resources/urdfData/GR2T3/urdf/GR2T3_raw_xyz.urdf",
    import_config=import_config,
    get_articulation_root=True,
)

# 获取 URDF 机器人

world.scene.add_default_ground_plane()
world.reset()
robot = world.scene.get_object(prim_path)

# 让 Isaac Sim 识别机器人关节
articulation = Articulation(prim_path)
articulation.initialize()

# Get stage handle
stage = omni.usd.get_context().get_stage()

# Enable physics
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
# Set gravity
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
# Set solver settings
PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
physxSceneAPI.CreateEnableCCDAttr(True)
physxSceneAPI.CreateEnableStabilizationAttr(True)
physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
physxSceneAPI.CreateSolverTypeAttr("TGS")

# Add ground plane
# omni.kit.commands.execute(
#     "AddGroundPlaneCommand",
#     stage=stage,
#     planePath="/groundPlane",
#     axis="Z",
#     size=1500.0,
#     position=Gf.Vec3f(0, 0, 0),
#     color=Gf.Vec3f(0.5),
# )

# Add lighting
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(500)

# 获取 URDF 机器人的关节列表
urdf_joint_names = articulation.dof_names  # 关节名称列表

# Start simulation
omni.timeline.get_timeline_interface().play()

anim = bvhsdk.ReadFile('C:/Users/FFTAI/Desktop/rotate/rotate.bvh')
bvh_joints = anim.getlistofjoints()
joint_state = JointState(articulation.num_dof)
for i in range(100):
    world.step(render=True)
    time.sleep(1 / 60)
for i in range(anim.frames):
    if i == 0:
        continue
    joint_positions = [0.0] * articulation.num_dof  # 初始化所有关节角度
    joint_index = 0

    for name in urdf_joint_names:
        pot = 0.0
        if name == "left_hip_pitch_joint":
            rotation = anim.getJoint("LeftUpLeg").getLocalRotation(i)
            pot = rotation[0]
        elif name == "right_hip_pitch_joint":
            rotation = anim.getJoint("RightUpLeg").getLocalRotation(i)
            pot = rotation[0]
        elif name == "waist_yaw_joint":
            rotation = anim.getJoint("Spine").getLocalRotation(i)
            pot += rotation[1]
            rotation = anim.getJoint("Spine1").getLocalRotation(i)
            pot += rotation[1]
            rotation = anim.getJoint("Spine2").getLocalRotation(i)
            pot += rotation[1]
            rotation = anim.getJoint("Spine3").getLocalRotation(i)
            pot += rotation[1]
        elif name == "left_hip_roll_joint":
            rotation = anim.getJoint("LeftUpLeg").getLocalRotation(i)
            if rotation[0] >= 0.0:
                pot = rotation[2] - rotation[0] / 10
            else:
                pot = rotation[2] + rotation[0] / 10
        elif name == "right_hip_roll_joint":
            rotation = anim.getJoint("RightUpLeg").getLocalRotation(i)
            if rotation[0] >= 0.0:
                pot = rotation[2] + rotation[0] / 10
            else:
                pot = rotation[2] - rotation[0] / 10
        elif name == "left_hip_yaw_joint":
            rotation = anim.getJoint("LeftUpLeg").getLocalRotation(i)
            pot = rotation[1] + rotation[0] / 2
        elif name == "right_hip_yaw_joint":
            rotation = anim.getJoint("RightUpLeg").getLocalRotation(i)
            pot = rotation[1] - rotation[0] / 2
        elif name == "head_yaw_joint":
            rotation = anim.getJoint("Head").getLocalRotation(i)
            pot = rotation[1]
        elif name == "left_shoulder_pitch_joint":
            rotation = anim.getJoint("LeftShoulder").getLocalRotation(i)
            pot += rotation[1]
            rotation = anim.getJoint("LeftArm").getLocalRotation(i)
            pot += rotation[0]
        elif name == "right_shoulder_pitch_joint":
            rotation = anim.getJoint("RightShoulder").getLocalRotation(i)
            pot += rotation[1]
            rotation = anim.getJoint("RightArm").getLocalRotation(i)
            pot += rotation[0]
        elif name == "left_knee_pitch_joint":
            rotation = anim.getJoint("LeftLeg").getLocalRotation(i)
            pot = rotation[0]
        elif name == "right_knee_pitch_joint":
            rotation = anim.getJoint("RightLeg").getLocalRotation(i)
            pot = rotation[0]
        elif name == "head_pitch_joint":
            rotation = anim.getJoint("Head").getLocalRotation(i)
            pot = rotation[0]
        elif name == "left_shoulder_roll_joint":
            rotation = anim.getJoint("LeftArm").getLocalRotation(i)
            pot = 90 + rotation[2]
        elif name == "right_shoulder_roll_joint":
            rotation = anim.getJoint("RightArm").getLocalRotation(i)
            pot = -90 + rotation[2]
        elif name == "left_ankle_pitch_joint":
            rotation = anim.getJoint("LeftFoot").getLocalRotation(i)
            pot = rotation[0]
        elif name == "right_ankle_pitch_joint":
            rotation = anim.getJoint("RightFoot").getLocalRotation(i)
            pot = rotation[0]
        elif name == "left_shoulder_yaw_joint":
            rotation = anim.getJoint("LeftArm").getLocalRotation(i)
            pot = rotation[1]
        elif name == "right_shoulder_yaw_joint":
            rotation = anim.getJoint("RightArm").getLocalRotation(i)
            pot = rotation[1]
        elif name == "left_ankle_roll_joint":
            rotation = anim.getJoint("LeftFoot").getLocalRotation(i)
            pot = rotation[2]
        elif name == "right_ankle_roll_joint":
            rotation = anim.getJoint("RightFoot").getLocalRotation(i)
            pot = rotation[2]
        elif name == "left_elbow_pitch_joint":
            rotation = anim.getJoint("LeftForeArm").getLocalRotation(i)
            if rotation[0] < 0.0:
                t = 0.0
            else:
                t = -rotation[0]
            pot = 27.5 + t
        elif name == "right_elbow_pitch_joint":
            rotation = anim.getJoint("RightForeArm").getLocalRotation(i)
            if rotation[0] < 0.0:
                t = 0.0
            else:
                t = -rotation[0]
            pot = 27.5 + t
        elif name == "left_wrist_yaw_joint":
            rotation = anim.getJoint("LeftForeArm").getLocalRotation(i)
            pot = rotation[1]
        elif name == "right_wrist_yaw_joint":
            rotation = anim.getJoint("RightForeArm").getLocalRotation(i)
            pot = rotation[1]
        elif name == "left_wrist_pitch_joint":
            rotation = anim.getJoint("LeftHand").getLocalRotation(i)
            pot = rotation[0]
        elif name == "right_wrist_pitch_joint":
            rotation = anim.getJoint("RightHand").getLocalRotation(i)
            pot = rotation[0]
        elif name == "left_wrist_roll_joint":
            rotation = anim.getJoint("LeftHand").getLocalRotation(i)
            pot = rotation[2]
        elif name == "right_wrist_roll_joint":
            rotation = anim.getJoint("RightHand").getLocalRotation(i)
            pot = rotation[2]
        joint_positions[joint_index] = deg_to_rad(pot)
        joint_index += 1
    articulation.set_joint_positions(joint_positions)
    world.step(render=True)
    time.sleep(1 / 60)
    # joint_velocities, joint_accelerations = joint_state.update(joint_positions)
    # 计算逆动力学
    # joint_torques = robot.compute_inverse_dynamics(
    #     joint_positions=joint_positions,
    #     joint_velocities=joint_velocities,
    #     joint_accelerations=joint_accelerations
    # )

    # 施加计算出的力矩到关节
    # robot.set_joint_efforts(joint_torques)

simulation_app.close()
