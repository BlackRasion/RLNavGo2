#!/usr/bin/env python3
"""
Isaac Sim Unitree Go2 ROS2 桥接模块

该模块实现了 Isaac Sim 仿真环境与 ROS2 之间的数据桥接，负责：
1. 发布机器人状态数据（里程计、位姿）到 ROS2 话题
2. 发布传感器数据（LiDAR 点云、相机图像）到 ROS2 话题
3. 订阅 ROS2 控制命令（速度命令）
4. 广播坐标变换（TF）
5. 同步仿真时间与 ROS2 时间

主要功能：
- 里程计发布：发布机器人的位置、姿态和速度信息
- 位姿发布：发布机器人在世界坐标系中的位姿
- 点云发布：发布 LiDAR 采集的 3D 点云数据
- 图像发布：发布相机采集的 RGB、深度和语义分割图像
- 速度控制：接收 ROS2 速度命令并传递给控制器
- TF 广播：发布传感器相对于机器人基座的坐标变换
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import numpy as np
from cv_bridge import CvBridge
import cv2
import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
import subprocess
import time
import go2.go2_ctrl as go2_ctrl

# 启用 Isaac Sim ROS2 桥接扩展
# 该扩展提供了 Isaac Sim 与 ROS2 之间的底层通信支持
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

# 导入 ROS2 桥接工具函数
from isaacsim.ros2.bridge import read_camera_info


class RobotDataManager(Node):
    """机器人数据管理器
    
    该类继承自 ROS2 Node，负责管理机器人数据的发布和订阅。
    它是 Isaac Sim 仿真与 ROS2 生态系统之间的核心桥接组件。
    
    主要职责：
    1. 创建和管理 ROS2 发布器（Publisher）
    2. 创建和管理 ROS2 订阅器（Subscriber）
    3. 发布机器人状态数据（里程计、位姿）
    4. 发布传感器数据（点云、图像）
    5. 订阅控制命令（速度命令）
    6. 广播坐标变换（TF）
    7. 同步仿真时间
    
    Attributes:
        cfg: 配置对象，包含传感器启用状态等配置
        env: Isaac Sim 环境实例
        num_envs: 机器人数量
        lidar_annotators: LiDAR 标注器列表
        cameras: 相机列表
        broadcaster: TF 广播器
        odom_pub: 里程计发布器列表
        pose_pub: 位姿发布器列表
        lidar_pub: 点云发布器列表
        cmd_vel_sub: 速度命令订阅器列表
    """
    
    def __init__(self, env, lidar_annotators, cameras, cfg):
        """初始化机器人数据管理器
        
        Args:
            env: Isaac Sim 环境实例
            lidar_annotators: LiDAR 标注器列表
            cameras: 相机列表
            cfg: 配置对象
        """
        # 调用父类初始化，设置节点名称
        super().__init__("robot_data_manager")
        
        # 保存配置对象
        self.cfg = cfg
        
        # 创建 ROS2 时间同步图
        # 该图用于将 Isaac Sim 的仿真时间发布到 ROS2 的 /clock 话题
        self.create_ros_time_graph()
        
        # 等待仿真时间设置完成
        # 循环检查直到 use_sim_time 参数设置成功
        sim_time_set = False
        while (rclpy.ok() and sim_time_set == False):
            sim_time_set = self.use_sim_time()

        # 保存环境实例
        self.env = env
        
        # 机器人数量
        self.num_envs = env.unwrapped.scene.num_envs
        
        # 保存传感器引用
        self.lidar_annotators = lidar_annotators
        self.cameras = cameras
        
        # 点云数据缓存
        self.points = []
        
        # ==================== ROS2 Broadcast设置 ====================
        # 创建 TF 广播器，用于发布坐标变换
        self.broadcaster = TransformBroadcaster(self)
        
        # ==================== ROS2 发布器列表 ====================
        self.odom_pub = []  # 里程计发布器列表
        self.pose_pub = []  # 位姿发布器列表
        self.lidar_pub = []  # 点云发布器列表
        self.semantic_seg_img_vis_pub = []  # 语义分割可视化图像发布器列表

        # ==================== ROS2 订阅器列表 ====================
        self.cmd_vel_sub = []  # 速度命令订阅器列表
        self.color_img_sub = []  # 彩色图像订阅器列表
        self.depth_img_sub = []  # 深度图像订阅器列表
        self.semantic_seg_img_sub = []  # 语义分割图像订阅器列表

        # ==================== ROS2 定时器列表 ====================
        self.lidar_publish_timer = []  # LiDAR 发布定时器列表
        
        # ==================== 为每个机器人创建发布器和订阅器 ====================
        for i in range(self.num_envs):
            # 判断是否为单机器人环境
            if (self.num_envs == 1):
                # 单机器人环境：使用简单的话题名称
                # 创建里程计发布器
                self.odom_pub.append(
                    self.create_publisher(Odometry, "unitree_go2/odom", 10))
                
                # 创建位姿发布器
                self.pose_pub.append(
                    self.create_publisher(PoseStamped, "unitree_go2/pose", 10))
                
                # 创建点云发布器
                self.lidar_pub.append(
                    self.create_publisher(PointCloud2, "unitree_go2/lidar/point_cloud", 10)
                )
                
                # 创建语义分割可视化图像发布器
                self.semantic_seg_img_vis_pub.append(
                    self.create_publisher(Image, "unitree_go2/front_cam/semantic_segmentation_image_vis", 10)
                )
                
                # 创建速度命令订阅器
                # 使用 lambda 函数捕获环境索引（固定为 0）
                self.cmd_vel_sub.append(
                    self.create_subscription(Twist, "unitree_go2/cmd_vel", 
                    lambda msg: self.cmd_vel_callback(msg, 0), 10)
                )
                
                # 创建语义分割图像订阅器
                self.semantic_seg_img_sub.append(
                    self.create_subscription(Image, "/unitree_go2/front_cam/semantic_segmentation_image", 
                    lambda msg: self.semantic_segmentation_callback(msg, 0), 10)
                )
            else:
                # 多机器人环境：使用命名空间区分不同机器人
                # 创建里程计发布器（带机器人索引）
                self.odom_pub.append(
                    self.create_publisher(Odometry, f"unitree_go2_{i}/odom", 10))
                
                # 创建位姿发布器（带机器人索引）
                self.pose_pub.append(
                    self.create_publisher(PoseStamped, f"unitree_go2_{i}/pose", 10))
                
                # 创建点云发布器（带机器人索引）
                self.lidar_pub.append(
                    self.create_publisher(PointCloud2, f"unitree_go2_{i}/lidar/point_cloud", 10)
                )
                
                # 创建语义分割可视化图像发布器（带机器人索引）
                self.semantic_seg_img_vis_pub.append(
                    self.create_publisher(Image, f"unitree_go2_{i}/front_cam/semantic_segmentation_image_vis", 10)
                )
                
                # 创建速度命令订阅器（带机器人索引）
                # 使用 lambda 函数捕获环境索引
                self.cmd_vel_sub.append(
                    self.create_subscription(Twist, f"unitree_go2_{i}/cmd_vel", 
                    lambda msg, env_idx=i: self.cmd_vel_callback(msg, env_idx), 10)
                )
                
                # 创建语义分割图像订阅器（带机器人索引）
                self.semantic_seg_img_sub.append(
                    self.create_subscription(Image, f"/unitree_go2_{i}/front_cam/semantic_segmentation_image", 
                    lambda msg, env_idx=i: self.semantic_segmentation_callback(msg, env_idx), 10)
                )
        
        # ==================== 发布频率设置 ====================
        # 使用墙钟时间（实际时间）控制发布频率
        self.odom_pose_freq = 50.0  # 里程计和位姿发布频率（Hz）
        self.lidar_freq = 15.0  # 点云发布频率（Hz）
        
        # 记录上次发布时间
        self.odom_pose_pub_time = time.time()
        self.lidar_pub_time = time.time() 
        
        # ==================== 静态坐标变换 ====================
        # 发布传感器相对于机器人基座的静态坐标变换
        self.create_static_transform()
        
        # ==================== 相机发布器设置 ====================
        # 创建相机相关的发布器
        self.create_camera_publisher()  

    def create_ros_time_graph(self):
        """创建 ROS2 时间同步图
        
        该函数创建一个 OmniGraph，用于将 Isaac Sim 的仿真时间发布到 ROS2 的 /clock 话题。
        这样可以确保 ROS2 节点使用仿真时间而不是系统时间。
        
        图结构：
        - ReadSimTime: 读取仿真时间
        - OnPlaybackTick: 在每帧触发
        - PublishClock: 发布时间到 /clock 话题
        """
        og.Controller.edit(
            {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
            {
                # 创建节点
                og.Controller.Keys.CREATE_NODES: [
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                # 连接节点
                og.Controller.Keys.CONNECT: [
                    # 将 OnPlaybackTick 的输出连接到 PublishClock 的执行输入
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    # 将 ReadSimTime 的输出连接到 PublishClock 的时间戳输入
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ],
                # 设置节点参数
                og.Controller.Keys.SET_VALUES: [
                    # 设置时钟话题名称
                    ("PublishClock.inputs:topicName", "/clock"),
                ],
            },
        )

    def use_sim_time(self):
        """设置 ROS2 使用仿真时间
        
        该函数通过命令行调用 ros2 param set 命令，将 ROS2 节点配置为使用仿真时间。
        这样可以确保 ROS2 话题的时间戳与仿真时间同步。
        
        Returns:
            bool: 总是返回 True
        """
        # 定义命令列表
        command = ["ros2", "param", "set", "/robot_data_manager", "use_sim_time", "true"]

        # 以非阻塞方式运行命令
        subprocess.Popen(command)  
        return True

    def create_static_transform(self):
        """创建并发布静态坐标变换
        
        该函数为每个机器人的 LiDAR 和相机创建静态坐标变换。
        这些变换描述了传感器相对于机器人基座的位置和姿态。
        
        坐标变换：
        - base_link -> lidar_frame: LiDAR 相对于基座的位置
        - base_link -> front_cam: 相机相对于基座的位置
        """
        for i in range(self.num_envs):
            # ==================== LiDAR 坐标变换 ====================
            # 创建静态 TF Broadcast
            lidar_broadcaster = StaticTransformBroadcaster(self)
            base_lidar_transform = TransformStamped()
            
            # 设置时间戳
            base_lidar_transform.header.stamp = self.get_clock().now().to_msg()
            
            # 设置坐标系 ID
            if (self.num_envs == 1):
                base_lidar_transform.header.frame_id = "unitree_go2/base_link"
                base_lidar_transform.child_frame_id = "unitree_go2/lidar_frame"
            else:
                base_lidar_transform.header.frame_id = f"unitree_go2_{i}/base_link"
                base_lidar_transform.child_frame_id = f"unitree_go2_{i}/lidar_frame"

            # 设置平移（LiDAR 相对于基座的位置）
            base_lidar_transform.transform.translation.x = 0.2  # 前方 0.2 米
            base_lidar_transform.transform.translation.y = 0.0   # 左右 0 米
            base_lidar_transform.transform.translation.z = 0.2   # 上方 0.2 米
            
            # 设置旋转（无旋转）
            base_lidar_transform.transform.rotation.x = 0.0
            base_lidar_transform.transform.rotation.y = 0.0
            base_lidar_transform.transform.rotation.z = 0.0
            base_lidar_transform.transform.rotation.w = 1.0
            
            # 发布坐标变换
            lidar_broadcaster.sendTransform(base_lidar_transform)
    
            # ==================== 相机坐标变换 ====================
            # 创建静态 TF Broadcast
            camera_broadcaster = StaticTransformBroadcaster(self)
            base_cam_transform = TransformStamped()
            
            # 设置坐标系 ID
            if (self.num_envs == 1):
                base_cam_transform.header.frame_id = "unitree_go2/base_link"
                base_cam_transform.child_frame_id = "unitree_go2/front_cam"
            else:
                base_cam_transform.header.frame_id = f"unitree_go2_{i}/base_link"
                base_cam_transform.child_frame_id = f"unitree_go2_{i}/front_cam"

            # 设置平移（相机相对于基座的位置）
            base_cam_transform.transform.translation.x = 0.4  # 前方 0.4 米
            base_cam_transform.transform.translation.y = 0.0   # 左右 0 米
            base_cam_transform.transform.translation.z = 0.2   # 上方 0.2 米
            
            # 设置旋转（相机朝向）
            # 四元数表示旋转：绕 Y 轴旋转 90 度
            base_cam_transform.transform.rotation.x = -0.5
            base_cam_transform.transform.rotation.y = 0.5
            base_cam_transform.transform.rotation.z = -0.5
            base_cam_transform.transform.rotation.w = 0.5
            
            # 发布坐标变换
            camera_broadcaster.sendTransform(base_cam_transform)
    
    def create_camera_publisher(self):
        """创建相机发布器
        
        该函数根据配置创建相机相关的发布器。
        支持的相机类型：
        - RGB 彩色相机
        - 深度相机
        - 语义分割相机
        """
        # self.pub_image_graph()  # 备用的图方式发布（已注释）
        
        # 检查是否启用相机
        if (self.cfg.sensor.enable_camera):
            # 根据配置创建相应的发布器
            if (self.cfg.sensor.color_image):
                self.pub_color_image()  # RGB 图像发布器
            if (self.cfg.sensor.depth_image):
                self.pub_depth_image()  # 深度图像发布器
            if (self.cfg.sensor.semantic_segmentation):
                self.pub_semantic_image()  # 语义分割图像发布器
            # self.pub_cam_depth_cloud()  # 深度点云发布器
            
            # 发布相机信息（内参等）
            self.publish_camera_info()
    
    def publish_odom(self, base_pos, base_rot, base_lin_vel_b, base_ang_vel_b, env_idx):
        """发布里程计数据
        
        该函数发布机器人的里程计信息，包括位置、姿态、线速度和角速度。
        同时发布从 map 到 base_link 的坐标变换。
        
        Args:
            base_pos: 基座位置 [x, y, z]
            base_rot: 基座姿态四元数 [w, x, y, z]
            base_lin_vel_b: 基座线速度（基座坐标系）
            base_ang_vel_b: 基座角速度（基座坐标系）
            env_idx: 环境索引
        """
        # 创建里程计消息
        odom_msg = Odometry()
        
        # 设置时间戳
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        
        # 设置坐标系
        odom_msg.header.frame_id = "map"  # 世界坐标系
        if (self.num_envs == 1):
            odom_msg.child_frame_id = "base_link"  # 机器人基座坐标系
        else:
            odom_msg.child_frame_id = f"unitree_go2_{env_idx}/base_link"
        
        # 设置位置
        odom_msg.pose.pose.position.x = base_pos[0].item()
        odom_msg.pose.pose.position.y = base_pos[1].item()
        odom_msg.pose.pose.position.z = base_pos[2].item()
        
        # 设置姿态（四元数：w, x, y, z）
        odom_msg.pose.pose.orientation.x = base_rot[1].item()
        odom_msg.pose.pose.orientation.y = base_rot[2].item()
        odom_msg.pose.pose.orientation.z = base_rot[3].item()
        odom_msg.pose.pose.orientation.w = base_rot[0].item()
        
        # 设置线速度
        odom_msg.twist.twist.linear.x = base_lin_vel_b[0].item()
        odom_msg.twist.twist.linear.y = base_lin_vel_b[1].item()
        odom_msg.twist.twist.linear.z = base_lin_vel_b[2].item()
        
        # 设置角速度
        odom_msg.twist.twist.angular.x = base_ang_vel_b[0].item()
        odom_msg.twist.twist.angular.y = base_ang_vel_b[1].item()
        odom_msg.twist.twist.angular.z = base_ang_vel_b[2].item()
        
        # 发布里程计消息
        self.odom_pub[env_idx].publish(odom_msg)

        # ==================== 发布坐标变换 ====================
        # 创建 TF 消息
        map_base_trans = TransformStamped()
        
        # 设置时间戳
        map_base_trans.header.stamp = self.get_clock().now().to_msg()
        
        # 设置坐标系
        map_base_trans.header.frame_id = "map"
        if (self.num_envs == 1):
            map_base_trans.child_frame_id = "unitree_go2/base_link"
        else:
            map_base_trans.child_frame_id = f"unitree_go2_{env_idx}/base_link"
        
        # 设置平移
        map_base_trans.transform.translation.x = base_pos[0].item()
        map_base_trans.transform.translation.y = base_pos[1].item()
        map_base_trans.transform.translation.z = base_pos[2].item()
        
        # 设置旋转
        map_base_trans.transform.rotation.x = base_rot[1].item()
        map_base_trans.transform.rotation.y = base_rot[2].item()
        map_base_trans.transform.rotation.z = base_rot[3].item()
        map_base_trans.transform.rotation.w = base_rot[0].item()
        
        # 发布坐标变换
        self.broadcaster.sendTransform(map_base_trans)
    
    def publish_pose(self, base_pos, base_rot, env_idx):
        """发布位姿数据
        
        该函数发布机器人在世界坐标系中的位姿。
        
        Args:
            base_pos: 基座位置 [x, y, z]
            base_rot: 基座姿态四元数 [w, x, y, z]
            env_idx: 环境索引
        """
        # 创建位姿消息
        pose_msg = PoseStamped()
        
        # 设置时间戳
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        
        # 设置坐标系
        pose_msg.header.frame_id = "map"
        
        # 设置位置
        pose_msg.pose.position.x = base_pos[0].item()
        pose_msg.pose.position.y = base_pos[1].item()
        pose_msg.pose.position.z = base_pos[2].item()
        
        # 设置姿态（四元数：w, x, y, z）
        pose_msg.pose.orientation.x = base_rot[1].item()
        pose_msg.pose.orientation.y = base_rot[2].item()
        pose_msg.pose.orientation.z = base_rot[3].item()
        pose_msg.pose.orientation.w = base_rot[0].item()
        
        # 发布位姿消息
        self.pose_pub[env_idx].publish(pose_msg)

    def publish_lidar_data(self, points, env_idx):
        """发布 LiDAR 点云数据
        
        该函数将 LiDAR 采集的点云数据发布到 ROS2 话题。
        
        Args:
            points: 点云数据，形状为 (N, 3) 的 numpy 数组
            env_idx: 机器人索引
        """
        # 创建点云消息
        point_cloud = PointCloud2()
        
        # 设置坐标系
        if (self.num_envs == 1):
            point_cloud.header.frame_id = "unitree_go2/lidar_frame"
        else:
            point_cloud.header.frame_id = f"unitree_go2_{env_idx}/lidar_frame"
        
        # 设置时间戳
        point_cloud.header.stamp = self.get_clock().now().to_msg()
        
        # 定义点云字段
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # 创建点云消息
        point_cloud = point_cloud2.create_cloud(point_cloud.header, fields, points)
        
        # 发布点云消息
        self.lidar_pub[env_idx].publish(point_cloud)        

    def pub_ros2_data(self):
        """发布 ROS2 数据（主方法）
        
        该函数是发布 ROS2 数据的主要方法，根据配置的频率发布里程计、位姿和点云数据。
        使用墙钟时间控制发布频率，确保数据发布速率稳定。
        """
        # 初始化发布标志
        pub_odom_pose = False
        pub_lidar = False
        
        # 计算距离上次发布的时间
        dt_odom_pose = time.time() - self.odom_pose_pub_time
        dt_lidar = time.time() - self.lidar_pub_time
        
        # 检查是否需要发布里程计和位姿
        if (dt_odom_pose >= 1. / self.odom_pose_freq):
            pub_odom_pose = True
        
        # 检查是否需要发布点云
        if (dt_lidar >= 1. / self.lidar_freq):
            pub_lidar = True

        # 发布里程计和位姿
        if (pub_odom_pose):
            # 更新发布时间
            self.odom_pose_pub_time = time.time()
            
            # 获取机器人数据
            robot_data = self.env.unwrapped.scene["unitree_go2"].data
            
            # 为每个环境发布里程计和位姿
            for i in range(self.num_envs):
                self.publish_odom(robot_data.root_state_w[i, :3],
                                robot_data.root_state_w[i, 3:7],
                                robot_data.root_lin_vel_b[i],
                                robot_data.root_ang_vel_b[i],
                                i)
                self.publish_pose(robot_data.root_state_w[i, :3],
                                robot_data.root_state_w[i, 3:7], i)
        
        # 发布点云数据
        if (self.cfg.sensor.enable_lidar):
            if (pub_lidar):
                # 更新发布时间
                self.lidar_pub_time = time.time()
                
                # 为每个环境发布点云
                for i in range(self.num_envs):
                    # 获取点云数据并重塑为 (N, 3) 形状
                    self.publish_lidar_data(
                        self.lidar_annotators[i].get_data()["data"].reshape(-1, 3), 
                        i
                    )

    def cmd_vel_callback(self, msg, env_idx):
        """速度命令回调函数
        
        该函数处理来自 ROS2 的速度命令，并将其传递给机器人控制器。
        
        Args:
            msg: Twist 消息，包含线速度和角速度
            env_idx: 环境索引
        """
        # 更新速度命令输入张量
        # 线速度 x 分量
        go2_ctrl.base_vel_cmd_input[env_idx][0] = msg.linear.x
        # 线速度 y 分量
        go2_ctrl.base_vel_cmd_input[env_idx][1] = msg.linear.y
        # 角速度 z 分量
        go2_ctrl.base_vel_cmd_input[env_idx][2] = msg.angular.z
    
    def semantic_segmentation_callback(self, img, env_idx):
        """语义分割图像回调函数
        
        该函数处理语义分割图像，将其转换为彩色可视化图像并发布。
        
        Args:
            img: Image 消息，包含语义分割数据
            env_idx: 机器人索引
        """
        # 创建 CvBridge 对象
        bridge = CvBridge()
        
        # 将 ROS 图像消息转换为 OpenCV 图像
        semantic_image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        
        # 归一化到 0-255 范围
        semantic_image_normalized = (semantic_image / semantic_image.max() * 255).astype(np.uint8)

        # 应用预定义的彩色映射
        # 使用 JET 彩色映射将语义标签转换为彩色图像
        color_mapped_image = cv2.applyColorMap(semantic_image_normalized, cv2.COLORMAP_JET)
        
        # 转换为 ROS 图像消息
        bridge = CvBridge()
        image_msg = bridge.cv2_to_imgmsg(color_mapped_image, encoding='rgb8')
        
        # 发布彩色语义分割图像
        self.semantic_seg_img_vis_pub[env_idx].publish(image_msg)

    def pub_color_image(self):
        """发布彩色图像
        
        该函数使用 Replicator 创建彩色图像发布器。
        将相机渲染产品链接到 ROS2 话题发布器。
        """
        for i in range(self.num_envs):
            # 获取渲染产品路径
            render_product = self.cameras[i]._render_product_path
            step_size = 1
            
            # 确定话题名称和坐标系 ID
            if (self.num_envs == 1):
                topic_name = "unitree_go2/front_cam/color_image"
                frame_id = "unitree_go2/front_cam"                         
            else:
                topic_name = f"unitree_go2_{i}/front_cam/color_image"
                frame_id = f"unitree_go2_{i}/front_cam"
            
            node_namespace = ""         
            queue_size = 1

            # 获取 RGB 渲染变量
            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
            
            # 获取 ROS2 图像发布器
            writer = rep.writers.get(rv + "ROS2PublishImage")
            
            # 初始化发布器
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name,
            )
            
            # 附加到渲染产品
            writer.attach([render_product])

            # 设置仿真门控步长
            # 控制上游 ROS2 发布器的执行速率
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    def pub_depth_image(self):
        """发布深度图像
        
        该函数使用 Replicator 创建深度图像发布器。
        将相机渲染产品链接到 ROS2 话题发布器。
        """
        for i in range(self.num_envs):
            # 获取渲染产品路径
            render_product = self.cameras[i]._render_product_path
            step_size = 1
            
            # 确定话题名称和坐标系 ID
            if (self.num_envs == 1):
                topic_name = "unitree_go2/front_cam/depth_image"                
                frame_id = "unitree_go2/front_cam"          
            else:
                topic_name = f"unitree_go2_{i}/front_cam/depth_image"
                frame_id = f"unitree_go2_{i}/front_cam"
            node_namespace = ""
            
            queue_size = 1

            # 获取深度渲染变量
            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                                    sd.SensorType.DistanceToImagePlane.name
                                )
            
            # 获取 ROS2 图像发布器
            writer = rep.writers.get(rv + "ROS2PublishImage")
            # 初始化发布器
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name
            )
            
            # 附加到渲染产品
            writer.attach([render_product])
            # 设置仿真门控步长
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    def pub_semantic_image(self):
        """发布语义分割图像
        
        该函数使用 Replicator 创建语义分割图像发布器。
        将相机渲染产品链接到 ROS2 话题发布器。
        """
        for i in range(self.num_envs):
            # 获取渲染产品路径
            render_product = self.cameras[i]._render_product_path
            step_size = 1
            # 确定话题名称和坐标系 ID
            if (self.num_envs == 1):
                topic_name = "unitree_go2/front_cam/semantic_segmentation_image"
                label_topic_name = "unitree_go2/front_cam/semantic_segmentation_label"                       
                frame_id = "unitree_go2/front_cam"          
            else:
                topic_name = f"unitree_go2_{i}/front_cam/semantic_segmentation_image"
                label_topic_name = f"unitree_go2_{i}/front_cam/semantic_segmentation_label"                       
                frame_id = f"unitree_go2_{i}/front_cam"
            
            node_namespace = ""
            queue_size = 1

            # 获取语义分割渲染变量
            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                                    sd.SensorType.SemanticSegmentation.name
                                )
            
            # 获取 ROS2 图像发布器
            writer = rep.writers.get("ROS2PublishSemanticSegmentation")
            
            # 初始化发布器
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name
            )
            
            # 附加到渲染产品
            writer.attach([render_product])

            # 获取语义标签发布器
            semantic_writer = rep.writers.get(
               "SemanticSegmentationSD" + f"ROS2PublishSemanticLabels"
            )
            
            # 初始化语义标签发布器
            semantic_writer.initialize(
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=label_topic_name,
            )
            
            # 附加到渲染产品
            semantic_writer.attach([render_product])

            # 设置仿真门控步长
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    def publish_camera_info(self):
        """发布相机信息
        
        该函数发布相机的内参信息，包括内参矩阵、畸变系数等。
        这些信息对于图像处理和 3D 重建非常重要。
        """
        for i in range(self.num_envs):
            # 获取渲染产品路径
            render_product = self.cameras[i]._render_product_path
            step_size = 1
            
            # 确定话题名称
            if (self.num_envs == 1):
                topic_name = "unitree_go2/front_cam/info"
            else:
                topic_name = f"unitree_go2_{i}/front_cam/info"
            
            queue_size = 1
            node_namespace = ""
            # 坐标系 ID 匹配 TF 树发布的坐标系
            frame_id = self.cameras[i].prim_path.split("/")[-1]

            # 获取 ROS2 相机信息发布器
            writer = rep.writers.get("ROS2PublishCameraInfo")
            
            # 读取相机信息
            camera_info = read_camera_info(render_product_path=render_product)
            
            # 初始化发布器
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name,
                width=camera_info["width"],
                height=camera_info["height"],
                projectionType=camera_info["projectionType"],
                k=camera_info["k"].reshape([1, 9]),
                r=camera_info["r"].reshape([1, 9]),
                p=camera_info["p"].reshape([1, 12]),
                physicalDistortionModel=camera_info["physicalDistortionModel"],
                physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
            )
            
            # 附加到渲染产品
            writer.attach([render_product])

            # 设置仿真门控步长
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                "PostProcessDispatch" + "IsaacSimulationGate", render_product
            )

            # 设置步长
            # 控制上游 ROS2 发布器的执行速率
            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)            
