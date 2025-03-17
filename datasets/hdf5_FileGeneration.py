import json
import h5py
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

def parse_urdf_joints(urdf_path):
    """解析URDF文件获取关节元数据"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = {}

    for joint in root.findall('joint'):
        j_name = joint.get('name')
        j_type = joint.get('type')

        # 解析关节限制
        limit = joint.find('limit')
        if limit is not None:
            lower = float(limit.get('lower'))
            upper = float(limit.get('upper'))
        else:
            lower, upper = -np.inf, np.inf

        # 解析旋转轴
        axis = joint.find('axis')
        axis_xyz = [float(x) for x in axis.get('xyz').split()] if axis is not None else [0,0,0]

        joints[j_name] = {
            'type': j_type,
            'limits': [lower, upper],
            'axis': axis_xyz
        }
    return joints

def generate_demo_data(base_data, joint_meta, noise_std=0.01):
    """生成带噪声的演示数据"""
    demo_length = len(base_data)
    num_joints = len(joint_meta)

    # 初始化数据容器
    joint_positions = np.zeros((demo_length, num_joints))
    joint_velocities = np.zeros((demo_length, num_joints))

    # 为每个时间步添加噪声
    for t in range(demo_length):
        for j_idx, (j_name, j_info) in enumerate(joint_meta.items()):
            # 获取原始角度值
            original_angle = base_data[t][j_name]['angle']

            # 添加高斯噪声并裁剪到合法范围
            noisy_angle = original_angle + np.random.normal(0, noise_std)
            noisy_angle = np.clip(noisy_angle, j_info['limits'][0], j_info['limits'][1])

            joint_positions[t, j_idx] = noisy_angle

            # 简单速度估算（可选）
            if t > 0:
                dt = 0.01  # 时间间隔
                joint_velocities[t, j_idx] = (noisy_angle - joint_positions[t-1, j_idx]) / dt

    return joint_positions, joint_velocities

def convert_urdf_animation_to_hdf5(urdf_path, json_path, output_hdf5, num_demos=10, noise_std=0.02):
    # 解析URDF关节信息
    joint_metadata = parse_urdf_joints(urdf_path)

    # 加载原始动画数据
    with open(json_path, 'r') as f:
        base_animation = json.load(f)  # 假设是时间步列表

    demo_length = len(base_animation)
    time_step = 0.01

    # 创建HDF5文件
    with h5py.File(output_hdf5, 'w') as hdf:
        # 元数据
        hdf.attrs['num_demos'] = num_demos
        hdf.attrs['time_step'] = time_step
        hdf.attrs['urdf_path'] = str(urdf_path)
        hdf.attrs['noise_std'] = noise_std

        # 关节元数据组
        joint_group = hdf.create_group('joint_metadata')
        for j_name, j_info in joint_metadata.items():
            j = joint_group.create_group(j_name)
            j.attrs['type'] = j_info['type']
            j.create_dataset('limits', data=j_info['limits'])
            j.create_dataset('axis', data=j_info['axis'])

        # 生成演示数据
        for demo_idx in range(num_demos):
            demo_group = hdf.create_group(f'demo_{demo_idx}')
            obs_group = demo_group.create_group('observations')

            # 生成带噪声的数据
            positions, velocities = generate_demo_data(
                base_animation,
                joint_metadata,
                noise_std=noise_std
            )

            # 保存观测数据
            obs_group.create_dataset('joint_positions', data=positions)
            obs_group.create_dataset('joint_velocities', data=velocities)

            # 创建空的action数据集（根据需求补充）
            demo_group.create_dataset('actions', data=np.zeros_like(positions))

if __name__ == "__main__":
    # 配置参数
    current_dir = Path(__file__).parent
    urdf_file = current_dir / "grmini1.urdf"
    json_file = current_dir / "grmini1_walk.json"
    output_file = current_dir / "grmini_data.hdf5"

    # 执行转换
    convert_urdf_animation_to_hdf5(
        urdf_path=urdf_file,
        json_path=json_file,
        output_hdf5=output_file,
        num_demos=10,
        noise_std=0.015  # 可调节的噪声强度
    )

    print(f"生成包含10个带噪声演示的HDF5文件：{output_file}")