#!/usr/bin/env python3
"""
动态障碍物管理器模块

该模块实现了仿真环境中的动态障碍物管理功能，包括：
1. 动态障碍物的创建和初始化
2. 障碍物运动控制（随机目标点跟踪）
3. 障碍物状态更新和同步

适配 IsaacSim 4.5 + IsaacLab 框架
基于原 dyn_env.py 中的动态障碍物逻辑重构

作者: AI Assistant
日期: 2025-03-26
"""

import torch
import numpy as np
import time
from typing import Tuple, List, Optional

from isaacsim.core.utils.prims import create_prim, get_prim_at_path
from pxr import Usd, UsdGeom, Gf
import isaaclab.sim as sim_utils


class DynamicObstacleManager:
    """动态障碍物管理器
    
    管理仿真环境中的动态障碍物，支持3D立方体和2D圆柱体两种类型。
    障碍物在预定义的局部范围内随机移动，模拟动态环境。
    
    Attributes:
        num_obstacles: 动态障碍物总数
        map_range: 地图范围 [x, y, z] (米)
        local_range: 局部移动范围 [x, y, z] (米)
        vel_range: 速度范围 [min, max] (米/秒)
        device: 计算设备 (cuda/cpu)
        dt: 仿真步长 (秒)
    """
    
    def __init__(
        self,
        num_obstacles: int,
        map_range: Tuple[float, float, float],
        local_range: Tuple[float, float, float],
        vel_range: Tuple[float, float],
        device: str,
        dt: float,
        prim_path_prefix: str = "/World/DynamicObstacles"
    ):
        """初始化动态障碍物管理器
        
        Args:
            num_obstacles: 动态障碍物数量（必须是8的倍数）
            map_range: 地图范围 [x, y, z] (米)
            local_range: 局部移动范围 [x, y, z] (米)
            vel_range: 速度范围 [min, max] (米/秒)
            device: 计算设备
            dt: 仿真步长
            prim_path_prefix: 障碍物在场景树中的路径前缀
        """
        self.num_obstacles = num_obstacles
        self.map_range = list(map_range)
        self.local_range = list(local_range)
        self.vel_range = list(vel_range)
        self.device = device
        self.dt = dt
        self.prim_path_prefix = prim_path_prefix
        
        self.N_w = 4
        self.N_h = 2
        self.max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = self.max_obs_width / float(self.N_w)
        
        self.dyn_obs_category_num = self.N_w * self.N_h
        if num_obstacles > 0:
            self.dyn_obs_num_of_each_category = max(1, int(num_obstacles / self.dyn_obs_category_num))
            self.num_obstacles = self.dyn_obs_num_of_each_category * self.dyn_obs_category_num
        else:
            self.dyn_obs_num_of_each_category = 0
            self.num_obstacles = 0
        
        self.dyn_obs_prim_paths: List[str] = []
        self.dyn_obs_step_count = 0
        
        if self.num_obstacles > 0:
            self._init_state_tensors()
    
    def _init_state_tensors(self):
        """初始化状态张量"""
        self.dyn_obs_state = torch.zeros((self.num_obstacles, 13), dtype=torch.float32, device=self.device)
        self.dyn_obs_state[:, 3] = 1.0
        self.dyn_obs_goal = torch.zeros((self.num_obstacles, 3), dtype=torch.float32, device=self.device)
        self.dyn_obs_origin = torch.zeros((self.num_obstacles, 3), dtype=torch.float32, device=self.device)
        self.dyn_obs_vel = torch.zeros((self.num_obstacles, 3), dtype=torch.float32, device=self.device)
        self.dyn_obs_size = torch.zeros((self.num_obstacles, 3), dtype=torch.float32, device=self.device)
    
    def create_obstacles(self) -> List[str]:
        """创建动态障碍物
        
        Returns:
            List[str]: 创建的障碍物 prim 路径列表
        """
        print(f"[DynamicObstacleManager] create_obstacles() called, num_obstacles={self.num_obstacles}")
        
        if self.num_obstacles == 0:
            print("[DynamicObstacleManager] num_obstacles is 0, returning empty list")
            return []
        
        from pxr import UsdGeom
        
        import omni
        stage = omni.usd.get_context().get_stage()
        
        parent_path = self.prim_path_prefix
        parent_prim = stage.GetPrimAtPath(parent_path)
        if not parent_prim.IsValid():
            print(f"[DynamicObstacleManager] 创建父 prim: {parent_path}")
            parent_prim = stage.DefinePrim(parent_path, "Xform")
        
        def check_pos_validity(prev_pos_list: List[np.ndarray], curr_pos: np.ndarray, min_dist: float) -> bool:
            for prev_pos in prev_pos_list:
                if np.linalg.norm(curr_pos - prev_pos) <= min_dist:
                    return False
            return True
        
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.num_obstacles)
        curr_obs_dist = obs_dist
        prev_pos_list: List[np.ndarray] = []
        
        cuboid_category_num = cylinder_category_num = int(self.dyn_obs_category_num / self.N_h)
        print(f"[DynamicObstacleManager] 创建立方体障碍物数量={cuboid_category_num}, 创建圆柱体障碍物数量={cylinder_category_num}")
        print(f"[DynamicObstacleManager] 每个障碍物数量={self.dyn_obs_num_of_each_category}")
        
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                start_time = time.time()
                while True:
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    
                    if category_idx < cuboid_category_num:
                        oz = np.random.uniform(low=0.0, high=self.map_range[2])
                    else:
                        oz = self.max_obs_2d_height / 2.0
                    
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    
                    if curr_time - start_time > 0.1:
                        curr_obs_dist *= 0.8
                        start_time = curr_time
                    
                    if valid:
                        prev_pos_list.append(curr_pos)
                        break
                
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                idx = origin_idx + category_idx * self.dyn_obs_num_of_each_category
                
                self.dyn_obs_origin[idx] = torch.tensor(origin, dtype=torch.float32, device=self.device)
                self.dyn_obs_state[idx, :3] = torch.tensor(origin, dtype=torch.float32, device=self.device)
                
                prim_path = f"{self.prim_path_prefix}/Obstacle_{idx}"
                
                if category_idx < cuboid_category_num:
                    obs_width = width = float(category_idx + 1) * self.max_obs_width / float(self.N_w)
                    obs_height = self.max_obs_3d_height
                    
                    cuboid_cfg = sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    )
                    print(f"[DynamicObstacleManager] 创建立方体 at {prim_path}, pos={origin}, size={[width, width, self.max_obs_3d_height]}")
                    try:
                        cuboid_cfg.func(prim_path, cuboid_cfg, translation=tuple(origin))
                        print(f"[DynamicObstacleManager] 创建立方体 at {prim_path}")
                    except Exception as e:
                        print(f"[DynamicObstacleManager] 创建立方体失败，错误: {e}")
                    self.dyn_obs_size[idx] = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float32, device=self.device)
                else:
                    radius = float(category_idx - cuboid_category_num + 1) * self.max_obs_width / float(self.N_w) / 2.0
                    obs_width = radius * 2
                    obs_height = self.max_obs_2d_height
                    
                    cylinder_cfg = sim_utils.CylinderCfg(
                        radius=radius,
                        height=self.max_obs_2d_height,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    )
                    print(f"[DynamicObstacleManager] 创建圆柱体 at {prim_path}, pos={origin}, radius={radius}, height={self.max_obs_2d_height}")
                    try:
                        cylinder_cfg.func(prim_path, cylinder_cfg, translation=tuple(origin))
                        print(f"[DynamicObstacleManager] 创建圆柱体 at {prim_path}")
                    except Exception as e:
                        print(f"[DynamicObstacleManager] 创建圆柱体失败，错误: {e}")    
                    self.dyn_obs_size[idx] = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float32, device=self.device)
                
                self.dyn_obs_prim_paths.append(prim_path)
        
        print(f"[DynamicObstacleManager] 创建 {len(self.dyn_obs_prim_paths)} 个障碍物")
        return self.dyn_obs_prim_paths
    
    def update(self):
        """更新动态障碍物状态"""
        if self.num_obstacles == 0:
            return
        
        if self.dyn_obs_step_count % 100 == 0:
            print(f"[DynamicObstacleManager] 更新中... step={self.dyn_obs_step_count}, device={self.device}")
        
        self._update_goals()
        self._update_velocities()
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.dt
        self._sync_to_sim()
        self.dyn_obs_step_count += 1
    
    def _update_goals(self):
        """更新障碍物目标位置"""
        actual_device = self.dyn_obs_origin.device
        
        if self.dyn_obs_step_count != 0:
            dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal) ** 2, dim=1))
        else:
            dyn_obs_goal_dist = torch.zeros(self.dyn_obs_state.size(0), device=actual_device)
        
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5
        num_new_goal = int(torch.sum(dyn_obs_new_goal_mask).item())
        
        if num_new_goal > 0:
            sample_x_local = -self.local_range[0] + 2.0 * self.local_range[0] * torch.rand(
                self.num_obstacles, 1, dtype=torch.float32, device=actual_device
            )
            sample_y_local = -self.local_range[1] + 2.0 * self.local_range[1] * torch.rand(
                self.num_obstacles, 1, dtype=torch.float32, device=actual_device
            )
            sample_z_local = -self.local_range[2] + 2.0 * self.local_range[2] * torch.rand(
                self.num_obstacles, 1, dtype=torch.float32, device=actual_device
            )
            sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
            
            new_goals = self.dyn_obs_origin + sample_goal_local
            
            mask_3d = dyn_obs_new_goal_mask.unsqueeze(1).expand_as(self.dyn_obs_goal)
            self.dyn_obs_goal = torch.where(mask_3d, new_goals, self.dyn_obs_goal)
            
            self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
            self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
            self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0.0, max=self.map_range[2])
            
            self.dyn_obs_goal[int(self.dyn_obs_goal.size(0) / 2):, 2] = self.max_obs_2d_height / 2.0
    
    def _update_velocities(self):
        """更新障碍物速度"""
        actual_device = self.dyn_obs_origin.device
        
        update_interval = int(2.0 / self.dt)
        if self.dyn_obs_step_count == 0 or self.dyn_obs_step_count % update_interval == 0:
            self.dyn_obs_vel_norm = self.vel_range[0] + (
                self.vel_range[1] - self.vel_range[0]
            ) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float32, device=actual_device)
            
            direction = self.dyn_obs_goal - self.dyn_obs_state[:, :3]
            direction_norm = torch.norm(direction, dim=1, keepdim=True)
            direction_norm = torch.clamp(direction_norm, min=1e-6)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * (direction / direction_norm)
    
    def _sync_to_sim(self):
        """同步障碍物状态到仿真器"""
        import omni.kit.app
        from pxr import UsdGeom, Gf
        
        stage = omni.usd.get_context().get_stage()
        
        for idx, prim_path in enumerate(self.dyn_obs_prim_paths):
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                pos = self.dyn_obs_state[idx, :3].cpu().numpy()
                
                xformable = UsdGeom.Xformable(prim)
                xform_ops = xformable.GetOrderedXformOps()
                
                translate_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                
                if translate_op is None:
                    translate_op = xformable.AddTranslateOp()
                
                translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    
    def get_obstacle_states(self) -> torch.Tensor:
        """获取所有障碍物的状态"""
        return self.dyn_obs_state.clone()
    
    def get_obstacle_positions(self) -> torch.Tensor:
        """获取所有障碍物的位置"""
        return self.dyn_obs_state[:, :3].clone()
    
    def reset(self):
        """重置动态障碍物状态"""
        if self.num_obstacles == 0:
            return
        
        self.dyn_obs_state.zero_()
        self.dyn_obs_state[:, 3] = 1.0
        self.dyn_obs_vel.zero_()
        self.dyn_obs_step_count = 0
        
        self.dyn_obs_state[:, :3] = self.dyn_obs_origin.clone()
        self.dyn_obs_goal = self.dyn_obs_origin.clone()
        
        self._sync_to_sim()
