from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
import numpy as np
import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app




import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# 初始化世界
world = World()

# 原始创建球体的代码（假设sim_utils已正确导入）
cfg_cuboid_position = sim_utils.MeshSphereCfg(
    radius=0.01,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
)
ball_prim_path = "/World/Objects/cfg_cuboid_position1"
cfg_cuboid_position.func(ball_prim_path, cfg_cuboid_position, translation=(0.15, 0.0, 2.0))
cfg_cuboid_position.func("/World/Objects/cfg_cuboid_position2", cfg_cuboid_position, translation=(0.15, 0.0, 1.0))

# 定义运动轨迹
trajectory = np.array([
    [0.15, 0.0, 2.0],
    [0.2, 0.1, 1.9],
    [0.25, 0.2, 1.8],
    [0.3, 0.3, 1.7]
])

# 初始化控制变量
current_time = 0.0
total_duration = 4.0  # 4秒完成整个轨迹循环
ball_prim = XFormPrim(ball_prim_path)

# 定义物理回调函数
def update_ball_position(world, dt):
    global current_time
    current_time += dt
    current_time %= total_duration

    num_points = len(trajectory)
    t = current_time

    # 计算当前段索引和插值比例
    segment_index = int(t / (total_duration / (num_points-1)))
    segment_index = min(segment_index, num_points-2)

    start_pos = trajectory[segment_index]
    end_pos = trajectory[segment_index + 1]

    segment_duration = total_duration / (num_points-1)
    alpha = (t % segment_duration) / segment_duration

    # 线性插值计算当前位置
    current_pos = start_pos + alpha * (end_pos - start_pos)
    ball_prim.set_world_pose(position=current_pos.tolist())

# 注册物理回调
world.add_physics_callback("update_ball_callback", update_ball_position)

# 重置并运行世界
world.reset()
while True:
    world.step(render=True)