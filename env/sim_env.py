from isaacsim.core.utils.prims import define_prim, get_prim_at_path
try:
    import isaacsim.storage.native as nucleus_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.nucleus as nucleus_utils
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter
from isaaclab.terrains import TerrainGeneratorCfg
from env.terrain_cfg import HfUniformDiscreteObstaclesTerrainCfg
from env.dyn_obstacle_manager import DynamicObstacleManager
import omni.replicator.core as rep

# =============================================================================
# 全局动态障碍物管理器实例
# 用于在主仿真循环中更新动态障碍物状态
# =============================================================================
_dynamic_obstacle_manager = None

def get_dynamic_obstacle_manager():
    """获取动态障碍物管理器实例
    
    Returns:
        DynamicObstacleManager: 动态障碍物管理器实例，如果未创建则返回None
    """
    global _dynamic_obstacle_manager
    return _dynamic_obstacle_manager

def reset_dynamic_obstacle_manager():
    """重置动态障碍物管理器实例"""
    global _dynamic_obstacle_manager
    _dynamic_obstacle_manager = None

def add_semantic_label():
    ground_plane = rep.get.prims("/World/GroundPlane")
    with ground_plane:
    # Add a semantic label
        rep.modify.semantics([("class", "floor")])

def create_obstacle_sparse_env():
    add_semantic_label()
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        num_envs=1,
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),  # 地形尺寸 50m × 50m
            color_scheme="height",  # 按高度着色
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),    # 宽度 0.5-1.0m
                obstacle_height_range=(1.0, 2.0),  # 高度 1.0-2.0m
                num_obstacles=100 ,  # 障碍物数量
                obstacles_distance=2.0,  # 障碍物最小间距
                border_width=5,  # 边界宽度
                avoid_positions=[[0, 0]]  # 避开原点
            )},
        ),
        visual_material=None,     
    )
    TerrainImporter(terrain) 

def create_obstacle_medium_env():
    add_semantic_label()
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        num_envs=1,
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(1.0, 2.0),
                num_obstacles=200 ,
                obstacles_distance=2.0,
                border_width=5,
                avoid_positions=[[0, 0]]
            )},
        ),
        visual_material=None,     
    )
    TerrainImporter(terrain) 


def create_obstacle_dense_env():
    add_semantic_label()
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        num_envs=1,
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(1.0, 2.0),
                num_obstacles=400,
                obstacles_distance=2.0,
                border_width=5,
                avoid_positions=[[0, 0]]
            )},
        ),
        visual_material=None,     
    )
    TerrainImporter(terrain) 

# =============================================================================
# 动态障碍物环境创建函数
# =============================================================================

def create_dyn_obstacle_sparse_env(device: str = "cuda", dt: float = 0.005):
    """创建稀疏动态障碍物环境
    
    同时包含静态障碍物地形和动态障碍物。
    静态障碍物：100个
    动态障碍物：8个（每类1个）
    
    Args:
        device: 计算设备 (cuda/cpu)
        dt: 仿真步长 (秒)
    """
    global _dynamic_obstacle_manager
    
    add_semantic_label()
    
    # 创建静态障碍物地形
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        num_envs=1,
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(1.0, 2.0),
                num_obstacles=100,
                obstacles_distance=2.0,
                border_width=5,
                avoid_positions=[[0, 0]]
            )},
        ),
        visual_material=None,
    )
    TerrainImporter(terrain)
    
    # 创建动态障碍物管理器
    _dynamic_obstacle_manager = DynamicObstacleManager(
        num_obstacles=8,  # 8个动态障碍物（每类1个）
        map_range=(15.0, 15.0, 4.5),
        local_range=(5.0, 5.0, 2.0),
        vel_range=(0.6, 1.5),
        device=device,
        dt=dt,
        prim_path_prefix="/World/DynamicObstacles"
    )
    
    # 创建动态障碍物
    _dynamic_obstacle_manager.create_obstacles()

def create_dyn_obstacle_medium_env(device: str = "cuda", dt: float = 0.005):
    """创建中等密度动态障碍物环境
    
    同时包含静态障碍物地形和动态障碍物。
    静态障碍物：200个
    动态障碍物：16个（每类2个）
    
    Args:
        device: 计算设备 (cuda/cpu)
        dt: 仿真步长 (秒)
    """
    global _dynamic_obstacle_manager
    
    add_semantic_label()
    
    # 创建静态障碍物地形
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        num_envs=1,
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(1.0, 2.0),
                num_obstacles=200,
                obstacles_distance=2.0,
                border_width=5,
                avoid_positions=[[0, 0]]
            )},
        ),
        visual_material=None,
    )
    TerrainImporter(terrain)
    
    # 创建动态障碍物管理器
    _dynamic_obstacle_manager = DynamicObstacleManager(
        num_obstacles=16,  # 16个动态障碍物（每类2个）
        map_range=(14.0, 14.0, 4.5),
        local_range=(5.0, 5.0, 2.0),
        vel_range=(0.6, 1.5),
        device=device,
        dt=dt,
        prim_path_prefix="/World/DynamicObstacles"
    )
    
    # 创建动态障碍物
    _dynamic_obstacle_manager.create_obstacles()

def create_dyn_obstacle_dense_env(device: str = "cuda", dt: float = 0.005):
    """创建密集动态障碍物环境
    
    同时包含静态障碍物地形和动态障碍物。
    静态障碍物：300个
    动态障碍物：32个（每类4个）
    
    Args:
        device: 计算设备 (cuda/cpu)
        dt: 仿真步长 (秒)
    """
    global _dynamic_obstacle_manager
    
    add_semantic_label()
    
    # 创建静态障碍物地形
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        num_envs=1,
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(1.0, 2.0),
                num_obstacles=300,
                obstacles_distance=2.0,
                border_width=5,
                avoid_positions=[[0, 0]]
            )},
        ),
        visual_material=None,
    )
    TerrainImporter(terrain)
    
    # 创建动态障碍物管理器
    _dynamic_obstacle_manager = DynamicObstacleManager(
        num_obstacles=32,  # 32个动态障碍物（每类4个）
        map_range=(15.0, 15.0, 4.5),
        local_range=(5.0, 5.0, 2.0),
        vel_range=(0.6, 1.5),
        device=device,
        dt=dt,
        prim_path_prefix="/World/DynamicObstacles"
    )
    
    # 创建动态障碍物
    _dynamic_obstacle_manager.create_obstacles()

def update_dynamic_obstacles():
    """更新动态障碍物状态
    
    在主仿真循环中每帧调用此函数，更新所有动态障碍物的位置和状态。
    """
    global _dynamic_obstacle_manager
    if _dynamic_obstacle_manager is not None:
        _dynamic_obstacle_manager.update()

def create_warehouse_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse") # 创建 USD 基元
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    prim.GetReferences().AddReference(asset_path)    # 引用 USD 文件

def create_warehouse_forklifts_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
    prim.GetReferences().AddReference(asset_path)

def create_warehouse_shelves_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
    prim.GetReferences().AddReference(asset_path)

def create_full_warehouse_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
    prim.GetReferences().AddReference(asset_path)

def create_hospital_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Hospital")
    prim = define_prim("/World/Hospital", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Hospital/hospital.usd"
    prim.GetReferences().AddReference(asset_path)

def create_office_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Office")
    prim = define_prim("/World/Office", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Office/office.usd"
    prim.GetReferences().AddReference(asset_path)
