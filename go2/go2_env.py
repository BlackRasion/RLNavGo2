"""
Go2 机器人环境配置模块

该文件定义了 Go2 机器人的仿真环境配置，包括：
1. 场景配置（地面、灯光、机器人模型）
2. 传感器配置（接触传感器、高度扫描器）
3. 动作空间配置
4. 观测空间配置
5. 环境参数设置
6. 相机跟随功能
"""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg
from isaacsim.core.utils.viewports import set_camera_view
import numpy as np
from scipy.spatial.transform import Rotation as R
import go2.go2_ctrl as go2_ctrl


@configclass
class Go2SimCfg(InteractiveSceneCfg):
    """Go2 机器人仿真场景配置
    
    定义了仿真场景的基本元素，包括地面、灯光、机器人模型和传感器
    """
    
    # 地面平面配置
    ground = AssetBaseCfg(
        prim_path="/World/ground",  # 地面在场景树中的路径
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300.0, 300.0)),  # 地面配置
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0, 0, 1e-4)  # 地面位置，稍微抬高避免与其他物体粘连
        )
    )
    
    # 灯光配置
    light = AssetBaseCfg(
        prim_path="/World/Light",  # 主光源路径
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),  # 主光源配置
    )
    
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",  # 环境光路径
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),  # 环境光配置
    )
    
    # Go2 机器人配置
    unitree_go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2")
    
    # Go2 足部接触传感器
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Go2/.*_foot",  # 匹配所有足部链接
        history_length=3,  # 接触历史长度
        track_air_time=True  # 追踪空中时间
    )

    # Go2 高度扫描器（用于地形感知）
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/base",  # 扫描器安装位置
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20)),  # 扫描器偏移位置
        attach_yaw_only=True,  # 只随偏航角旋转
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 扫描网格模式
        debug_vis=False,  # 禁用调试可视化
        mesh_prim_paths=["/World/ground"],  # 扫描的网格路径
    )


@configclass
class ActionsCfg:
    """动作空间配置
    
    定义了机器人可以执行的动作类型
    """
    # 关节位置动作配置
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="unitree_go2",  # 名称
        joint_names=[".*"]  # 匹配所有关节
    )


@configclass
class ObservationsCfg:
    """环境观测空间配置
    
    定义了机器人的观测数据类型和来源
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组配置
        
        定义了用于强化学习策略的观测数据
        """

        # 观测项（顺序保持不变）
        # 基座线速度
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,  # 计算线速度的函数
            params={"asset_cfg": SceneEntityCfg(name="unitree_go2")}  
        )
        
        # 基座角速度
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,  # 计算角速度的函数
            params={"asset_cfg": SceneEntityCfg(name="unitree_go2")} 
        )
        
        # 投影重力向量
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,  # 计算投影重力的函数
            params={"asset_cfg": SceneEntityCfg(name="unitree_go2")},  
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)  # 添加噪声
        )
        
        # 速度命令
        base_vel_cmd = ObsTerm(func=go2_ctrl.base_vel_cmd)  # 获取速度命令的函数

        # 关节相对位置
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,  # 计算关节相对位置的函数
            params={"asset_cfg": SceneEntityCfg(name="unitree_go2")}  
        )
        
        # 关节相对速度
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,  # 计算关节相对速度的函数
            params={"asset_cfg": SceneEntityCfg(name="unitree_go2")}  
        )
        
        # 上一步动作
        actions = ObsTerm(func=mdp.last_action)  # 获取上一步动作的函数
        
        # 高度扫描数据
        height_scan = ObsTerm(
            func=mdp.height_scan,  # 计算高度扫描的函数
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},  
            clip=(-1.0, 1.0)  # 数据裁剪范围
        )

        def __post_init__(self) -> None:
            """初始化后处理
            
            设置观测组的额外参数
            """
            self.enable_corruption = False  # 禁用观测数据损坏
            self.concatenate_terms = True  # 合并所有观测项为一个向量

    # 观测组实例
    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """MDP 命令配置
    
    定义了环境中的命令生成方式
    """
    pass


# 基础速度命令配置
base_vel_cmd = mdp.UniformVelocityCommandCfg(
    asset_name="unitree_go2",  # 名称
    resampling_time_range=(1.0, 5.0),  # 命令重采样时间范围
    rel_heading_envs=1.0,  # 相对朝向环境的比例
    rel_standing_envs=0.0,  # 相对站立环境的比例
    debug_vis=True,  # 启用调试可视化
    heading_control_stiffness=0.0,  # 朝向控制刚度
    ranges=mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.0),  # 线速度 x 范围
        lin_vel_y=(0.0, 0.0),  # 线速度 y 范围
        ang_vel_z=(0.0, 0.0),  # 角速度 z 范围
        heading=(0, 0)  # 朝向范围
    ),
)


@configclass
class EventCfg:
    """事件配置
    
    定义了环境中的事件处理
    """
    pass


@configclass
class RewardsCfg:
    """奖励配置
    
    定义了环境中的奖励函数
    """
    pass


@configclass
class TerminationsCfg:
    """终止条件配置
    
    定义了环境的终止条件
    """
    pass


@configclass
class CurriculumCfg:
    """课程学习配置
    
    定义了环境的课程学习参数
    """
    pass


@configclass
class Go2RSLEnvCfg(ManagerBasedRLEnvCfg):
    """Go2 机器人环境配置
    
    综合配置 Go2 机器人的强化学习环境
    """
    # 场景设置
    scene = Go2SimCfg(num_envs=2, env_spacing=2.0)  # 创建 2 个机器人，间距 2.0 米

    # 基本设置
    observations = ObservationsCfg()  # 观测空间配置
    actions = ActionsCfg()  # 动作空间配置
    
    # 占位设置
    commands = CommandsCfg()  # 命令配置
    rewards = RewardsCfg()  # 奖励配置
    terminations = TerminationsCfg()  # 终止条件配置
    events = EventCfg()  # 事件配置
    curriculum = CurriculumCfg()  # 课程学习配置

    def __post_init__(self):
        """初始化后处理
        
        设置环境的额外参数
        """
        # 查看器设置
        self.viewer.eye = [-4.0, 0.0, 5.0]  # 相机位置
        self.viewer.lookat = [0.0, 0.0, 0.0]  # 相机看向的点

        # 步进设置
        self.decimation = 8  # 仿真步长缩减因子

        # 仿真设置
        self.sim.dt = 0.005  # 物理仿真步长（5ms / 200Hz）
        self.sim.render_interval = self.decimation  # 渲染间隔
        self.sim.disable_contact_processing = True  # 禁用接触处理
        self.sim.render.antialiasing_mode = None  # 禁用抗锯齿

        # RSL 环境控制设置
        self.episode_length_s = 20.0  #  episode 长度（可忽略）
        self.is_finite_horizon = False  # 无限 horizon
        self.actions.joint_pos.scale = 0.25  # 关节位置动作缩放因子

        # 设置高度扫描器更新周期
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


def camera_follow(env):
    """相机跟随函数
    
    使相机跟随机器人移动，保持良好的观察视角
    
    Args:
        env: 环境实例
    """
    # 只在单机器人情况下启用相机跟随
    if env.unwrapped.scene.num_envs == 1:
        # 获取机器人位置
        robot_position = env.unwrapped.scene["unitree_go2"].data.root_state_w[0, :3].cpu().numpy()
        
        # 获取机器人姿态
        robot_orientation = env.unwrapped.scene["unitree_go2"].data.root_state_w[0, 3:7].cpu().numpy()
        
        # 转换四元数（注意：Isaac Sim 使用 wxyz 格式，scipy 使用 xyzw 格式）
        rotation = R.from_quat([robot_orientation[1], robot_orientation[2], 
                                robot_orientation[3], robot_orientation[0]])
        
        # 计算偏航角
        yaw = rotation.as_euler('zyx')[0]
        
        # 计算偏航旋转矩阵
        yaw_rotation = R.from_euler('z', yaw).as_matrix()
        
        # 设置相机位置：相对于机器人的位置，考虑机器人朝向
        camera_position = yaw_rotation.dot(np.asarray([-4.0, 0.0, 5.0])) + robot_position
        
        # 更新相机视图
        set_camera_view(camera_position, robot_position)
