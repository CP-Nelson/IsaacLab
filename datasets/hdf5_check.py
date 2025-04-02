import h5py

# 读取 HDF5 文件
file_path = "./datasets/dataset.hdf5"  # 替换为实际文件路径
def explore_hdf5(obj, indent=0):
    """ 递归遍历 HDF5 文件结构 """
    for key in obj:
        item = obj[key]
        print(" " * indent + f"- {key}: {type(item)}")
        if isinstance(item, h5py.Group):  # 如果是组，则继续遍历
            explore_hdf5(item, indent + 2)

# 读取并遍历 HDF5 文件
with h5py.File(file_path, "r") as f:
    explore_hdf5(f)

# import h5py

# file_path = "dataset.hdf5"

# with h5py.File(file_path, "r") as f:
#     root_velocity = f["data/demo_9/obs/actions"][:]  # 读取数据
#     print("root_velocity 数据 shape:", root_velocity.shape)
#     print("root_velocity 数据示例:", root_velocity[:5])  # 预览前5行

# 画图
# import matplotlib.pyplot as plt

# with h5py.File(file_path, "r") as f:
#     eef_pos = f["data/demo_1/obs/eef_pos"][:]  # 读取数据

# plt.figure(figsize=(8, 6))
# plt.plot(eef_pos[:, 0], eef_pos[:, 1], label="EEF XY Path", color="b")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.title("End Effector XY Movement")
# plt.legend()
# plt.show()
