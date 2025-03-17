import json
import numpy as np
import h5py
import time

# 导入 Isaac Sim 相关模块
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # 启动 Isaac Sim

from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.articulations import ArticulationView

# 读取 JSON 运动数据
with open("motion_data.json", "r") as f:
    motion_data = json.load(f)

frames = motion_data["frames"]
num_joints = len(frames[0]["joint_positions"])

# 创建 HDF5 文件
hdf5_file = h5py.File("dataset.hdf5", "w")
hdf5_group = hdf5_file.create_group("demos")

# 初始化 Isaac Sim
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# 加载 URDF 机器人
robot_urdf_path = "assets/urdf/my_robot.usd"  # 需要先转换 URDF 为 USD
add_reference_to_stage(usd_path=robot_urdf_path, prim_path="/World/Robot")
world.reset()

# 获取机器人实例
robot = ArticulationView(prim_paths_expr="/World/Robot", name="robot")
world.scene.add(robot)

# 运行运动动画并记录数据
demo_data = []
world.reset()
for frame in frames:
    time.sleep(0.1)  # 控制播放速度
    joint_positions = np.array(frame["joint_positions"])

    # 发送指令到 Isaac Sim
    robot.set_joint_positions(joint_positions)

    # 记录数据
    demo_data.append(joint_positions)

    # 进行仿真步进
    world.step(render=True)

# 存入 HDF5
hdf5_group.create_dataset("joint_positions", data=np.array(demo_data))
hdf5_file.close()

print("HDF5 数据集创建成功！")

# 关闭仿真
simulation_app.close()
