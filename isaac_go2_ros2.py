#!/usr/bin/env python3
"""
Isaac Sim Unitree Go2 ROS2 仿真系统主入口文件

该文件是整个仿真系统的核心入口，负责初始化 Isaac Sim 应用、配置 Go2 机器人环境、
设置传感器、启动 ROS2 桥接，并运行主仿真循环。

主要功能：
1. 解析命令行参数
2. 启动 Isaac Sim 
3. 加载配置文件
4. 初始化 Go2 机器人环境
5. 创建仿真场景
6. 配置传感器系统
7. 设置键盘控制
8. 初始化 ROS2 桥接
9. 运行仿真主循环
10. 清理资源
"""

import os
import hydra
import rclpy
import torch
import time
import math
import argparse
from isaaclab.app import AppLauncher

# 命令行参数解析
# 创建参数解析器，用于处理命令行输入
parser = argparse.ArgumentParser(description="Isaac Sim Unitree Go2 仿真系统")

# 添加 AppLauncher 相关的命令行参数
# AppLauncher 提供了启动 Isaac Sim 应用的功能，包括无头模式、分辨率等参数
AppLauncher.add_app_launcher_args(parser)

# 解析命令行参数
args_cli = parser.parse_args()

# 启动 Omniverse 应用
# 创建 AppLauncher 实例并启动仿真应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下是仿真系统的主要逻辑"""

# 导入必要的模块
import torch

# 从 go2 模块导入环境配置和相机跟随功能
from go2.go2_env import Go2RSLEnvCfg, camera_follow
# 导入仿真环境模块
import env.sim_env as sim_env
# 导入传感器管理模块
import go2.go2_sensors as go2_sensors
# 导入 Omniverse 相关模块
import omni
import carb
# 导入控制器模块
import go2.go2_ctrl as go2_ctrl
# 导入 ROS2 桥接模块
import ros2.go2_ros2_bridge as go2_ros2_bridge

# 配置文件路径
# 获取当前文件所在目录，并拼接配置文件目录路径
FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")


@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    """运行仿真的主函数
    
    Args:
        cfg: Hydra 配置对象，包含仿真系统的所有配置参数
    """
    
    # ==================== Go2 环境设置 ====================
    # 创建 Go2 机器人环境配置实例
    go2_env_cfg = Go2RSLEnvCfg()
    
    # 设置机器人数量
    go2_env_cfg.scene.num_envs = cfg.num_envs
    
    # 计算仿真步长缩减因子
    # 确保控制频率与物理仿真频率匹配
    # decimation = 物理仿真频率 / 控制频率
    go2_env_cfg.decimation = math.ceil(1. / go2_env_cfg.sim.dt / cfg.freq)
    
    # 设置渲染间隔，与 decimation 保持一致
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
    
    # 初始化速度命令输入张量
    # 用于存储和传递速度控制命令
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    
    # 加载强化学习策略模型
    # 这里使用粗糙地形策略，注释掉的是平坦地形策略
    # env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)
    env, policy = go2_ctrl.get_rsl_rough_policy(go2_env_cfg)

    # ==================== 仿真环境创建 ====================
    # 根据配置选择并创建仿真环境
    if cfg.env_name == "obstacle-dense":
        sim_env.create_obstacle_dense_env()  # 密集障碍物环境
    elif cfg.env_name == "obstacle-medium":
        sim_env.create_obstacle_medium_env()  # 中等密度障碍物环境
    elif cfg.env_name == "obstacle-sparse":
        sim_env.create_obstacle_sparse_env()  # 稀疏障碍物环境
    elif cfg.env_name == "warehouse":
        sim_env.create_warehouse_env()  # 简单仓库环境
    elif cfg.env_name == "warehouse-forklifts":
        sim_env.create_warehouse_forklifts_env()  # 带叉车的仓库环境
    elif cfg.env_name == "warehouse-shelves":
        sim_env.create_warehouse_shelves_env()  # 带货架的仓库环境
    elif cfg.env_name == "full-warehouse":
        sim_env.create_full_warehouse_env()  # 完整仓库环境

    # ==================== 传感器设置 ====================
    # 创建传感器管理器实例
    sm = go2_sensors.SensorManager(cfg.num_envs)
    
    # 添加 LiDAR 传感器
    # 返回 LiDAR 标注器列表，用于获取点云数据
    lidar_annotators = sm.add_rtx_lidar()
    
    # 添加相机传感器
    # 返回相机列表，用于获取图像数据
    cameras = sm.add_camera(cfg.freq)

    # ==================== 键盘控制设置 ====================
    # 获取输入接口
    system_input = carb.input.acquire_input_interface()
    
    # 订阅键盘事件
    # 注册键盘回调函数，用于处理用户输入的控制命令
    system_input.subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), 
        go2_ctrl.sub_keyboard_event
    )
    
    # ==================== ROS2 桥接初始化 ====================
    # 初始化 ROS2
    rclpy.init()
    
    # 创建机器人数据管理器
    # 负责发布传感器数据和订阅控制命令
    dm = go2_ros2_bridge.RobotDataManager(env, lidar_annotators, cameras, cfg)

    # ==================== 运行仿真 ====================
    # 计算仿真步进时间
    # sim_step_dt = 物理步长 * 缩减因子
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    
    # 重置环境，获取初始观测
    obs, _ = env.reset()
    
    # 主仿真循环
    while simulation_app.is_running():
        # 记录循环开始时间
        start_time = time.time()
        
        # 使用 PyTorch 推理模式
        # 提高性能，禁用梯度计算
        with torch.inference_mode():
            # ==================== 控制关节 ====================
            # 使用强化学习策略生成动作
            # 输入：观测数据
            # 输出：关节位置控制命令
            actions = policy(obs)

            # ==================== 环境步进 ====================
            # 执行环境步进
            # 应用动作到机器人，更新物理状态
            # 返回：新的观测、奖励、终止标志、信息
            obs, _, _, _ = env.step(actions)

            # ==================== ROS2 数据发布 ====================
            # 发布传感器数据到 ROS2 话题
            dm.pub_ros2_data()
            
            # 处理 ROS2 回调
            # 处理订阅的控制命令等
            rclpy.spin_once(dm)

            # ==================== 相机跟随 ====================
            # 如果启用相机跟随
            if cfg.camera_follow:
                # 调用相机跟随函数，使相机跟随机器人
                camera_follow(env)

            # ==================== 频率控制 ====================
            # 计算已用时间
            elapsed_time = time.time() - start_time
            
            # 如果执行时间小于目标时间
            if elapsed_time < sim_step_dt:
                # 计算需要休眠的时间
                sleep_duration = sim_step_dt - elapsed_time
                # 休眠以保持仿真频率
                time.sleep(sleep_duration)
        
        # 计算实际循环时间
        actual_loop_time = time.time() - start_time
        
        # 计算实时因子 (Real Time Factor)
        # 表示仿真时间与实际时间的比率
        # 最大值限制为 1.0，表示实时运行
        rtf = min(1.0, sim_step_dt / elapsed_time)
        
        # 打印性能信息
        # 显示步进时间和实时因子
        print(f"\r步进时间: {actual_loop_time*1000:.2f}ms, 实时因子: {rtf:.2f}", 
              end='', flush=True)
    
    # ==================== 清理资源 ====================
    # 销毁 ROS2 节点
    dm.destroy_node()
    
    # 关闭 ROS2
    rclpy.shutdown()
    
    # 关闭仿真应用
    simulation_app.close()


if __name__ == "__main__":
    """主函数入口
    
    当直接运行此脚本时，调用 run_simulator 函数
    """
    run_simulator()
