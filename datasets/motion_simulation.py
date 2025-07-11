import json
from isaacsim import SimulationApp
import time
import numpy as np
import bvhsdk
import os

# os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": False})


import omni.kit.app
from isaacsim.core.api import World
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics
from omni.isaac.core.articulations import Articulation
import omni.usd
# from JointState import JointState
import math
stage = omni.usd.get_context().get_stage()

def deg_to_rad(degree):
    return degree * math.pi / 180
import logging
import isaacsim.core.utils.prims as prim_utils

logger = logging.getLogger("IsaacSim")
logger.setLevel(logging.INFO)  # 设定日志级别

# 创建控制台日志输出
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)

# 添加处理器
logger.addHandler(console_handler)

# 是否开启物理仿真
# world = World(stage_units_in_meters=1.0, physics_dt=0.0)
world = World(stage_units_in_meters=1.0)

# Setting up import configuration:
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.density = 1800
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.distance_scale = 1.0

anim = bvhsdk.ReadFile('C:/Users/FFTAI/Desktop/左右后撤.bvh')
bvh_joints = anim.getlistofjoints()
status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path="D:/comp_repo/robot-frames-tools/resources/urdfData/grmini/urdf/GRMini1T1_raw_floating_base.urdf",
    import_config=import_config,
    get_articulation_root=True,
)
logger.info("这是 INFO 日志")
logger.info(prim_path)
# prim_path = '/GRMini1T1/'
# 获取 URDF 机器人

world.scene.add_default_ground_plane()
world.reset()
robot = world.scene.get_object('/GRMiniT1')

# 让 Isaac Sim 识别机器人关节
articulation = Articulation(prim_path)
articulation.initialize()
# damping_value = 5  # 适当调整阻尼
# stiffness_value = 500.0  # 适当调整刚度

# for dof in articulation.dof_names:
#     joint_prim_path = f"{prim_path}/{dof}"  # 获取关节的 USD 路径
#     joint_prim = stage.GetPrimAtPath(joint_prim_path)

#     if not joint_prim.IsValid():
#         print(f"Warning: Joint {dof} not found in USD stage.")
#         continue

#     # 获取 Drive API 并设置参数
#     drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")  # 适用于旋转关节
#     drive.CreateDampingAttr().Set(damping_value)
#     drive.CreateStiffnessAttr().Set(stiffness_value)

# print("Damping and stiffness values have been updated.")
import omni.usd
from pxr import UsdPhysics, UsdShade, Sdf
# 创建物理材质
material_path = "/World/GroundPhysicsMaterial"
physx_material = UsdShade.Material.Define(stage, Sdf.Path(material_path))

# 添加 MaterialAPI 使其成为物理材质
physx_material_api = UsdPhysics.MaterialAPI.Apply(physx_material.GetPrim())

# 设置摩擦力和弹性
physx_material_api.CreateStaticFrictionAttr().Set(1.2)   # 静摩擦
physx_material_api.CreateDynamicFrictionAttr().Set(1.0)  # 动摩擦
physx_material_api.CreateRestitutionAttr().Set(0.0)      # 弹性恢复

# 获取地面
ground_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
mat_binding_api = UsdShade.MaterialBindingAPI.Apply(ground_prim)
mat_binding_api.Bind(physx_material, UsdShade.Tokens.weakerThanDescendants, "physics")


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
physxSceneAPI.CreateEnableGPUDynamicsAttr(True)
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
# omni.timeline.get_timeline_interface().play()

# joint_state = JointState(articulation.num_dof)
# for i in range(500):
#     world.step(render=True)
#     time.sleep(1 / 120)
# for i in range(anim.frames):
for i in range(50000):
    # anim.root.translation[1] = anim.root.translation[1]
    # root_position = [b / 100 for b in anim.root.translation[i]]
    # if i ==0:
    #     root_position = [b / 100 for b in anim.root.translation[i]]
    # else:
    #     difference = anim.root.translation[i]-anim.root.translation[i-1]
    #     root_position = [b / 100 for b in difference]
    # root_position[0],root_position[1],root_position[2] = root_position[2],root_position[0],root_position[1]-0.94

    # a = root_position
    # prim_utils.set_prim_attribute_value("/GRMini1T1", attribute_name="xformOp:translate", value=a)
    # robot = scene["GRMini1T1"]

    # root_state = robot.data.default_root_state.clone()
    # root_state[:, :3] += root_position
    # robot.write_root_pose_to_sim(root_state[:, :7])
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
            # rotation = anim.getJoint("Spine3").getLocalRotation(i)
            # pot += rotation[1]
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
        else:
            pass
        joint_positions[joint_index] = deg_to_rad(pot)
        joint_index += 1
    articulation.set_joint_positions(joint_positions)
    # a = articulation.get_world_pose()
    world.step(render=True)
    time.sleep(1 / 20)
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
