

import torch
import einops
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time

def _design_scene(self):
        """
        设计仿真场景
        
        创建无人机、光照、地面、静态障碍物地形和动态障碍物
        
        返回:
            list: 无人机 prim 路径列表
        """
        # =========================================================================
        # 1. 创建无人机
        # =========================================================================
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name]  # 获取无人机模型类
        cfg = drone_model.cfg_cls(force_sensor=False)                     # 创建配置
        self.drone = drone_model(cfg=cfg)
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]   # 在高度2米处生成

        # =========================================================================
        # 2. 添加光照
        # =========================================================================
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # =========================================================================
        # 3. 创建地面平面
        # =========================================================================
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        # =========================================================================
        # 4. 创建带静态障碍物的地形
        # =========================================================================
        self.map_range = [20.0, 20.0, 4.5]  # 地图范围 [x, y, z]

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(self.map_range[0]*2, self.map_range[1]*2),  # 40x40 米
                border_width=5.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,  # 静态障碍物数量
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.1),           # 宽度范围
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],  # 高度等级
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],  # 高度概率分布
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material=None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        terrain_importer = TerrainImporter(terrain_cfg)

        # =========================================================================
        # 5. 创建动态障碍物（如果启用）
        # =========================================================================
        if (self.cfg.env_dyn.num_obstacles == 0):
            return [drone_prim]
            
        # 动态障碍物分类：
        # - 3D 障碍物：立方体，可在空中漂浮
        # - 2D 障碍物：圆柱体，只能水平移动
        # 宽度分为 N_w=4 个区间: [0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]
        # 高度分为 N_h=2 个区间: [0, 0.5], [0.5, inf]（区分3D和2D障碍物）
        N_w = 4  # 宽度区间数
        N_h = 2  # 高度区间数
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0   # 3D障碍物最大高度
        self.max_obs_2d_height = 2.0   # 2D障碍物高度
        self.dyn_obs_width_res = max_obs_width / float(N_w)  # 宽度分辨率
        dyn_obs_category_num = N_w * N_h  # 总类别数 = 8
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num

        # 动态障碍物状态变量
        self.dyn_obs_list = []  # 障碍物对象列表
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_state[:, 3] = 1.  # 四元数 w 分量初始化为1
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0  # 运动步数计数
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device)

        # 辅助函数：检查位置是否满足均匀分布条件
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True            
        
        # 计算期望的障碍物间距
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles)
        curr_obs_dist = obs_dist
        prev_pos_list = []
        cuboid_category_num = 1  # 3D障碍物类别数：减少到1类
        cylinder_category_num = 7  # 2D障碍物类别数：增加到7类
        
        # 为每个类别创建障碍物
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # 为该类别的每个障碍物生成原点位置
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                start_time = time.time()
                while (True):
                    # 随机采样位置
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2])  # 3D: 随机高度
                    else:
                        oz = self.max_obs_2d_height / 2.  # 2D: 固定高度
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8  # 超时则降低距离要求
                        start_time = curr_time
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                idx = origin_idx + category_idx * self.dyn_obs_num_of_each_category
                self.dyn_obs_origin[idx] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[idx, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{idx}", "Xform", translation=origin)

            # 根据类别生成不同形状的障碍物
            if (category_idx < cuboid_category_num):
                # 3D 立方体障碍物（只有1类，使用最小宽度）
                obs_width = width = max_obs_width / float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                # 2D 圆柱体障碍物（7类，宽度从 max_obs_width/N_w 到 max_obs_width）
                cylinder_idx = category_idx - cuboid_category_num
                radius = float(cylinder_idx + 1) * max_obs_width / float(cylinder_category_num) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius=radius,
                        height=self.max_obs_2d_height, 
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
                
            self.dyn_obs_list.append(dynamic_obstacle)
            start_idx = category_idx * self.dyn_obs_num_of_each_category
            end_idx = (category_idx + 1) * self.dyn_obs_num_of_each_category
            self.dyn_obs_size[start_idx:end_idx] = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)

        return [drone_prim]
def _post_sim_step(self, tensordict: TensorDictBase):
    """
    仿真步后处理：更新传感器和动态障碍物
    
    参数:
        tensordict: 环境数据
    """
    # 更新动态障碍物位置
    if (self.cfg.env_dyn.num_obstacles != 0):
        self.move_dynamic_obstacle()

    def move_dynamic_obstacle(self):
        """
        移动动态障碍物
        
        每个障碍物在局部范围内随机移动，模拟动态环境
        运动逻辑：
        1. 当接近当前目标时，随机采样新目标
        2. 每约2秒随机改变速度
        3. 更新位置并同步到仿真器
        """
        # =========================================================================
        # 步骤1：为需要更新的障碍物随机采样新目标
        # =========================================================================
        # 计算当前位置与目标的距离
        if self.dyn_obs_step_count != 0:
            dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1))
        else:
            dyn_obs_goal_dist = torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
            
        # 距离小于阈值则需要新目标
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        
        # 在局部范围内随机采样新目标
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[2] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
    
        # 将局部目标转换到全局坐标
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        
        # 限制在地图范围内
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        # 2D障碍物保持固定高度
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height / 2.

        # =========================================================================
        # 步骤2：每约2秒随机改变速度
        # =========================================================================
        if (self.dyn_obs_step_count % int(2.0 / self.cfg.sim.dt) == 0):
            # 随机速度大小
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            # 速度方向指向目标
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3]) / torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # =========================================================================
        # 步骤3：更新位置
        # =========================================================================
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt

        # =========================================================================
        # 步骤4：同步到仿真器进行可视化
        # =========================================================================
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            start_idx = category_idx * self.dyn_obs_num_of_each_category
            end_idx = (category_idx + 1) * self.dyn_obs_num_of_each_category
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[start_idx:end_idx]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1