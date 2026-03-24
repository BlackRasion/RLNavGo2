#!/usr/bin/env python3
"""
Go2 机器人控制器模块

该模块负责 Go2 机器人的控制逻辑，包括：
1. 速度命令管理
2. 键盘事件处理
3. 强化学习策略加载

主要功能：
- 初始化速度命令输入张量
- 处理键盘输入控制机器人运动
- 加载预训练的强化学习策略
- 提供平坦和粗糙地形的控制策略
"""

import os
import torch
import carb
import gymnasium as gym
from isaaclab.envs import ManagerBasedEnv
from go2.go2_ctrl_cfg import unitree_go2_flat_cfg, unitree_go2_rough_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

# 速度命令输入张量
# 存储机器人的速度控制命令 (num_envs, [线速度x, 线速度y, 角速度z]) 
base_vel_cmd_input = None


def init_base_vel_cmd(num_envs):
    """初始化速度命令输入张量
    
    Args:
        num_envs: 机器人数量
    """
    global base_vel_cmd_input
    # 创建指定大小的零张量，形状为 (num_envs, 3)
    # 3 个维度分别对应：线速度x, 线速度y, 角速度z
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)


def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    """获取速度命令输入
    
    Args:
        env: 环境实例
    
    Returns:
        torch.Tensor: 速度命令张量，已移至环境所在设备
    """
    global base_vel_cmd_input
    # 克隆张量并移至环境所在设备（通常是 GPU）
    return base_vel_cmd_input.clone().to(env.device)


def sub_keyboard_event(event) -> bool:
    """处理键盘事件，更新速度命令
    
    Args:
        event: 键盘事件对象
    
    Returns:
        bool: 事件处理状态
    """
    global base_vel_cmd_input
    
    # 线速度和角速度的默认值
    lin_vel = 1.5  # 线速度 (m/s)
    ang_vel = 1.5  # 角速度 (rad/s)
    
    if base_vel_cmd_input is not None:
        # 处理按键按下事件
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # 处理第一个机器人（环境 0）的控制
            if event.input.name == 'W':
                # 前进
                base_vel_cmd_input[0] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
            elif event.input.name == 'S':
                # 后退
                base_vel_cmd_input[0] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
            elif event.input.name == 'A':
                # 左移
                base_vel_cmd_input[0] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
            elif event.input.name == 'D':
                # 右移
                base_vel_cmd_input[0] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
            elif event.input.name == 'Z':
                # 左转
                base_vel_cmd_input[0] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
            elif event.input.name == 'C':
                # 右转
                base_vel_cmd_input[0] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
            
            # 如果有多个环境，处理第二个机器人（环境 1）的控制
            if base_vel_cmd_input.shape[0] > 1:
                if event.input.name == 'I':
                    # 前进
                    base_vel_cmd_input[1] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
                elif event.input.name == 'K':
                    # 后退
                    base_vel_cmd_input[1] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
                elif event.input.name == 'J':
                    # 左移
                    base_vel_cmd_input[1] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
                elif event.input.name == 'L':
                    # 右移
                    base_vel_cmd_input[1] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
                elif event.input.name == 'M':
                    # 左转
                    base_vel_cmd_input[1] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
                elif event.input.name == '>':
                    # 右转
                    base_vel_cmd_input[1] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
        
        # 处理按键释放事件，重置命令为零
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            base_vel_cmd_input.zero_()
    return True


def get_rsl_flat_policy(cfg):
    """获取平坦地形的强化学习策略
    
    Args:
        cfg: 环境配置对象
    
    Returns:
        tuple: (环境实例, 策略函数)
    """
    # 禁用高度扫描观测（平坦地形不需要）
    cfg.observations.policy.height_scan = None
    
    # 创建平坦地形环境
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    
    # 包装为 RSL RL 向量环境
    env = RslRlVecEnvWrapper(env)

    # 加载平坦地形控制策略
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_flat_cfg
    
    # 获取模型检查点路径
    ckpt_path = get_checkpoint_path(
        log_path=os.path.abspath("ckpts"), 
        run_dir=agent_cfg["load_run"], 
        checkpoint=agent_cfg["load_checkpoint"]
    )
    
    # 创建 PPO 运行器
    ppo_runner = OnPolicyRunner(
        env, 
        agent_cfg, 
        log_dir=None, 
        device=agent_cfg["device"]
    )
    
    # 加载预训练模型
    ppo_runner.load(ckpt_path)
    
    # 获取推理策略
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    
    return env, policy


def get_rsl_rough_policy(cfg):
    """获取粗糙地形的强化学习策略
    
    Args:
        cfg: 环境配置对象
    
    Returns:
        tuple: (环境实例, 策略函数)
    """
    # 创建粗糙地形环境
    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg)
    
    # 包装为 RSL RL 向量环境
    env = RslRlVecEnvWrapper(env)

    # 加载粗糙地形控制策略
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_rough_cfg
    
    # 获取模型检查点路径
    ckpt_path = get_checkpoint_path(
        log_path=os.path.abspath("ckpts"), 
        run_dir=agent_cfg["load_run"], 
        checkpoint=agent_cfg["load_checkpoint"]
    )
    
    # 创建 PPO 运行器
    ppo_runner = OnPolicyRunner(
        env, 
        agent_cfg, 
        log_dir=None, 
        device=agent_cfg["device"]
    )
    
    # 加载预训练模型
    ppo_runner.load(ckpt_path)
    
    # 获取推理策略
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    
    return env, policy
