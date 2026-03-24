#!/usr/bin/env python3
"""
Go2 机器人强化学习控制配置文件

该文件定义了用于训练和加载 Go2 机器人控制策略的配置参数。
包含两种地形的配置：
1. 平坦地形配置 (unitree_go2_flat_cfg)
2. 粗糙地形配置 (unitree_go2_rough_cfg)

配置参数分为以下几个部分：
- 基本训练参数
- 策略网络配置
- 算法配置 (PPO)
- 保存和日志配置
- 模型加载配置
"""

# 平坦地形配置
# 适用于平坦、光滑的地面环境
unitree_go2_flat_cfg = {
    'seed': 42,
    'device': 'cuda:0',
    'num_steps_per_env': 24,
    'max_iterations': 1500,
    'empirical_normalization': False,
    'policy': {
        'class_name': 'ActorCritic',
        'init_noise_std': 1.0,
        'actor_hidden_dims': [128, 128, 128],
        'critic_hidden_dims': [128, 128, 128],
        'activation': 'elu'
    },
    'algorithm': {
        'class_name': 'PPO',
        'value_loss_coef': 1.0,
        'use_clipped_value_loss': True,
        'clip_param': 0.2,
        'entropy_coef': 0.01,
        'num_learning_epochs': 5,
        'num_mini_batches': 4,
        'learning_rate': 0.001,
        'schedule': 'adaptive',
        'gamma': 0.99,
        'lam': 0.95,
        'desired_kl': 0.01,
        'max_grad_norm': 1.0
    },
    'save_interval': 50,
    'experiment_name': 'unitree_go2_flat',
    'run_name': '',
    'logger': 'tensorboard',
    'neptune_project': 'isaaclab',
    'wandb_project': 'isaaclab',
    'resume': False,
    'load_run': 'unitree_go2',
    'load_checkpoint': 'flat_model_6800.pt'
}

unitree_go2_rough_cfg = {
        'seed': 42, 
        'device': 'cuda', 
        'num_steps_per_env': 24, 
        'max_iterations': 15000, 
        'empirical_normalization': False, 
        'policy': {
            'class_name': 'ActorCritic', 
            'init_noise_std': 1.0, 
            'actor_hidden_dims': [512, 256, 128], 
            'critic_hidden_dims': [512, 256, 128], 
            'activation': 'elu'
            }, 
        'algorithm': {
            'class_name': 'PPO', 
            'value_loss_coef': 1.0, 
            'use_clipped_value_loss': True, 
            'clip_param': 0.2, 
            'entropy_coef': 0.01, 
            'num_learning_epochs': 5, 
            'num_mini_batches': 4, 
            'learning_rate': 0.001, 
            'schedule': 'adaptive', 
            'gamma': 0.99, 
            'lam': 0.95, 
            'desired_kl': 0.01, 
            'max_grad_norm': 1.0
        }, 
        'save_interval': 50, 
        'experiment_name': 'unitree_go2_rough', 
        'run_name': '', 
        'logger': 'tensorboard', 
        'neptune_project': 'orbit', 
        'wandb_project': 'orbit', 
        'resume': False, 
        'load_run': 'unitree_go2', 
        'load_checkpoint': 'rough_model_7850.pt'
}

"""
配置参数说明：

1. 基本训练参数：
   - seed: 随机种子，确保训练的可重复性
   - device: 训练设备，建议使用 CUDA 以获得最佳性能
   - num_steps_per_env: 每个环境的训练步数
   - max_iterations: 最大训练迭代次数
   - empirical_normalization: 是否对观测数据进行经验归一化

2. 策略网络配置：
   - class_name: 策略类名称，使用 Actor-Critic 架构
   - init_noise_std: 初始噪声标准差，影响探索
   - actor_hidden_dims: Actor 网络隐藏层维度
   - critic_hidden_dims: Critic 网络隐藏层维度
   - activation: 激活函数，ELU 在强化学习中表现较好

3. PPO 算法配置：
   - class_name: 算法类名称
   - value_loss_coef: 价值损失系数
   - use_clipped_value_loss: 是否使用裁剪价值损失
   - clip_param: 裁剪参数，控制策略更新的幅度
   - entropy_coef: 熵损失系数，鼓励探索
   - num_learning_epochs: 每次收集数据后的学习轮数
   - num_mini_batches: 小批量数量，用于并行计算
   - learning_rate: 学习率
   - schedule: 学习率调度策略
   - gamma: 折扣因子，控制未来奖励的权重
   - lam: 优势函数的 Lambda 系数，用于计算广义优势估计
   - desired_kl: 期望的 KL 散度，用于自适应学习率
   - max_grad_norm: 梯度裁剪最大范数，防止梯度爆炸

4. 保存和日志配置：
   - save_interval: 模型保存间隔
   - experiment_name: 实验名称
   - run_name: 运行名称
   - logger: 日志记录器类型
   - neptune_project: Neptune 项目名称
   - wandb_project: Weights & Biases 项目名称

5. 模型加载配置：
   - resume: 是否从检查点恢复训练
   - load_run: 加载的运行名称
   - load_checkpoint: 加载的检查点文件

两个配置的主要区别：
- 粗糙地形配置 (unitree_go2_rough_cfg)：
  1. 更大的 max_iterations (15000 vs 1500)
  2. 更深的网络架构 (512, 256, 128 vs 128, 128, 128)
  3. 不同的 Neptune 和 WandB 项目名称
  4. 不同的检查点文件

这些配置参数是基于经验调优的结果，适用于不同地形条件下的四足机器人控制。
"""
