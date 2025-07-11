
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

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.asset.importer.heightmap import HeightMap

import numpy as np

# 世界与物理场景初始化
world = World(stage_units_in_meters=1.0)
world.reset()

# 假设 height_data 是 288×583 的 np.array（float32 格式）
height_data = np.load("heightmap.npy").astype(np.float32)
h, w = height_data.shape

# 创建 HeightMap 物理地形
hf = HeightMap(
    prim_path="/World/Heightmap",
    height_data=height_data,
    horizontal_spacing=(0.05, 0.05),  # xy 分辨率
    vertical_scale=1.0,
    physics_enabled=True,
)

world.scene.add(hf)
world.reset()
