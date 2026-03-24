import omni
import numpy as np
from pxr import Gf
import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

class SensorManager:
    def __init__(self, num_envs):
        self.num_envs = num_envs

    def add_rtx_lidar(self):
        """
        添加 RTX LiDAR 传感器到每个机器人
        
        Returns:
            list: LiDAR 注释器列表，用于获取 LiDAR 数据
        """
        lidar_annotators = []
        for env_idx in range(self.num_envs):
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path="/lidar", # 传感器路径
                parent=f"/World/envs/env_{env_idx}/Go2/base", # 传感器父节点
                config="Hesai_XT32_SD10", # LiDAR 配置，使用 Hesai XT32 型号
                # config="Velodyne_VLS128", # 可选配置：Velodyne VLS128
                translation=(0.2, 0, 0.2),  # 传感器位置
                orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # 传感器朝向（四元数表示） w,i,j,k
            )

            annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
            hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
            annotator.attach(hydra_texture.path)
            lidar_annotators.append(annotator)
        return lidar_annotators

    def add_camera(self, freq):
        """
        添加相机传感器到每个机器人
        
        Args:
            freq (float): 相机采样频率
            
        Returns:
            list: 相机对象列表
        """
        cameras = []
        for env_idx in range(self.num_envs):
            camera = Camera(
                prim_path=f"/World/envs/env_{env_idx}/Go2/base/front_cam", # 相机路径
                translation=np.array([0.4, 0.0, 0.2]), # 相机位置
                frequency=freq, # 相机采样频率
                resolution=(640, 480), # 相机分辨率
                orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True), # 相机朝向
                focal_length=1.5, # 相机焦距
            )
            camera.initialize()
            camera.set_focal_length(1.5)
            cameras.append(camera)
        return cameras