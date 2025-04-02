# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MIT License
#
# Copyright (c) 2021 Stanford Vision and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
The main entry point for training policies from pre-collected data.
// 传入参数
Args:
    algo: name of the algorithm to run.
    task: name of the environment.
    name: if provided, override the experiment name defined in the config
    dataset: if provided, override the dataset path defined in the config

This file has been modified from the original version in the following ways:

"""

"""Launch Isaac Sim Simulator first."""
################### 启动 ###################
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse # 命令行传参解算
import gymnasium as gym # 建立基础robomimic训练环境，可能是isaac_gym期用于构建的基础框架
import json # 传入传出config文件，输出train和validation文件记录
import numpy as np # 引入固定随机数种，确保结果可复现
import os # 文件操作
import sys # 命令行显示输出
import time # 测量时间
import torch # robomimic神经网络框架
import traceback # 错误反馈
from collections import OrderedDict # 用于构建有序字典，便于查看和存储显示
from torch.utils.data import DataLoader # pytorch数据处理

import psutil #调用显示系统当前ram资源
import robomimic.utils.env_utils as EnvUtils # robomimic环境构建，类似dataloader
import robomimic.utils.file_utils as FileUtils # robomimic加载数据集环境参数及形状参数
import robomimic.utils.obs_utils as ObsUtils # robomimic初始化加载环境变量
import robomimic.utils.torch_utils as TorchUtils # robomimic根据config内容，设定torch设备
import robomimic.utils.train_utils as TrainUtils # robomimic训练过程主交互|重要|################
from robomimic.algo import algo_factory # robomimic算法配置，指定不同类型算法
from robomimic.config import config_factory # robomimic配置区
from robomimic.utils.log_utils import DataLogger, PrintLogger # robomimic数据记录，控制台输出

# Needed so that environment is registered
import isaaclab_tasks  # noqa: F401

############### 训练开始 ###############
def train(config, device):
############## 初始化-文件记录 ##################
    """Train a model using the algorithm."""
    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    print(f">>> Saving logs into directory: {log_dir}")
    print(f">>> Saving checkpoints into directory: {ckpt_dir}")
    print(f">>> Saving videos into directory: {video_dir}")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger
###############################################


###############################################
########        初始化环境数据参数       #######
    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists  读取训练数据路径
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

    # load basic metadata from training file  config训练数据的meta数据，导出其中的env和shape
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data, all_obs_keys=config.all_obs_keys, verbose=True
    )

    # 给定env名称
    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)
###############################################


###############################################
########        创建环境       #######
    # create environment  初始化有序环境字典envs
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        # 环境创建启动，根据config的.experiment.env和.experiment.additional_envs创建环境
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        # 遍历所有环境，初始化一遍各环境存入envs字典中
        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env
            print(envs[env.name])

    print("")

    # setup for a new training run  初始化数据记录；依据config建立训练模型，依据shape_meta确立传入和传出动作空间的维度
    data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # save the config as a json file  记录config为json文件
    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data 借助TrainUtils模块，初始化训练集trainset和验证集validset
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler() # 训练数据采样器
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retrieve statistics for normalizing observations 归一化统计训练数据，用于保存后查看
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders  初始化训练集的DataLoader
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    # 判断并进行验证集初始化流程
    if config.experiment.validate:
        # cap num workers for validation dataset at 1 缩小工作线程至1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    # main training loop
    best_valid_loss = None # 不记录最佳损失值
    last_ckpt_time = time.time() # 记录训练开始时间

    # number of learning steps per epoch (defaults to a full dataset pass) 训练及验证步数
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    # 开始训练
    for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
        # 执行一次训练并返回日志
        step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps)
        model.on_epoch_end(epoch) # epoch结束时固定更新

        # setup checkpoint path
        epoch_ckpt_name = f"model_epoch_{epoch}"

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print(f"Train Epoch {epoch}")
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            print(f"Validation Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

    # terminate logging
    data_logger.close()


def main(args):
    """Train a model on a task using a specified algorithm."""
    # load config
    if args.task is not None:
        # obtain the configuration entry point
        cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"

        print(f"Loading configuration for task: {args.task}")
        print(gym.envs.registry.keys())
        print(" ")
        cfg_entry_point_file = gym.spec(args.task).kwargs.pop(cfg_entry_point_key)
        # check if entry point exists
        if cfg_entry_point_file is None:
            raise ValueError(
                f"Could not find configuration for the environment: '{args.task}'."
                f" Please check that the gym registry has the entry point: '{cfg_entry_point_key}'."
            )

        with open(cfg_entry_point_file) as f:
            ext_cfg = json.load(f)
            config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        raise ValueError("Please provide a task name through CLI arguments.")

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    # change location of experiment directory
    config.train.output_dir = os.path.abspath(os.path.join("./logs", args.log_dir, args.task))

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--algo", type=str, default=None, help="Name of the algorithm.")
    parser.add_argument("--log_dir", type=str, default="robomimic", help="Path to log directory")

    args = parser.parse_args()

    # run training
    main(args)
    # close sim app
    simulation_app.close()
