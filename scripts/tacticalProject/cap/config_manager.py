"""
配置管理器
从YAML配置动态生成所有飞机位置
"""
import yaml
import os
from typing import Dict, Any
from dataclasses import dataclass, field
from .coordinate_system import CoordinateSystem, BattlefieldConfig


@dataclass
class AircraftPosition:
    """飞机位置配置"""
    x_rel: float          # 战场相对X坐标(km)
    y_rel: float          # 战场相对Y坐标(km)
    heading_rel: float    # 战场相对航向(度)
    lon: float = 0.0      # 地球经度
    lat: float = 0.0      # 地球纬度
    heading: float = 0.0  # 地球航向


@dataclass 
class CAPConfig:
    """CAP任务配置"""
    # 基准参数
    a0100_lon: float = 120.6757
    a0100_lat: float = 60.0
    a0100_heading: float = 0.0
    
    # 编队参数
    wingman_y_offset: float = 100.0
    lead_x_left: float = 75.0
    lead_x_right: float = 125.0
    wingman_x_left: float = 25.0
    wingman_x_right: float = 175.0
    
    # 敌机参数
    enemy_y: float = 400.0
    enemy_x_list: list = field(default_factory=lambda: [40, 80, 120, 160])
    
    # FAOR参数
    faor_width: float = 200.0
    faor_length: float = 300.0
    
    # 飞机参数
    altitude_ft: float = 29520.0
    speed_fps: float = 984.25
    missiles: int = 4


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        self.config = CAPConfig()
        if config_path and os.path.exists(config_path):
            self._load_yaml(config_path)
        
        self.coord_sys = CoordinateSystem(BattlefieldConfig(
            a0100_lon=self.config.a0100_lon,
            a0100_lat=self.config.a0100_lat,
            a0100_heading=self.config.a0100_heading,
            a0100_x=self.config.lead_x_left,
            a0100_y=0.0
        ))
        
        self.positions: Dict[str, AircraftPosition] = {}
        self._calculate_all_positions()
    
    def _load_yaml(self, path: str):
        """加载YAML配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        cfg = self.config
        if 'base_reference' in data:
            br = data['base_reference'].get('a0100', {})
            cfg.a0100_lon = br.get('longitude', cfg.a0100_lon)
            cfg.a0100_lat = br.get('latitude', cfg.a0100_lat)
            cfg.a0100_heading = br.get('heading', cfg.a0100_heading)
        
        if 'formation' in data:
            fm = data['formation']
            cfg.wingman_y_offset = fm.get('wingman_y_offset', cfg.wingman_y_offset)
            if 'lead_x' in fm:
                cfg.lead_x_left = fm['lead_x'].get('left', cfg.lead_x_left)
                cfg.lead_x_right = fm['lead_x'].get('right', cfg.lead_x_right)
            if 'wingman_x' in fm:
                cfg.wingman_x_left = fm['wingman_x'].get('left', cfg.wingman_x_left)
                cfg.wingman_x_right = fm['wingman_x'].get('right', cfg.wingman_x_right)
        
        if 'enemy' in data:
            cfg.enemy_y = data['enemy'].get('spawn_y', cfg.enemy_y)
            cfg.enemy_x_list = data['enemy'].get('spawn_x', cfg.enemy_x_list)
    
    def _calculate_all_positions(self):
        """计算所有飞机位置"""
        cfg = self.config
        wy = cfg.wingman_y_offset
        
        # 我方飞机（战场相对坐标）
        friendly = {
            'A0100': (cfg.lead_x_left, 0, 0),        # 左长机，朝北
            'A0200': (cfg.wingman_x_left, wy, 180),  # 左僚机，朝南
            'A0300': (cfg.lead_x_right, 0, 0),       # 右长机，朝北
            'A0400': (cfg.wingman_x_right, wy, 180), # 右僚机，朝南
        }
        
        # 敌方飞机
        enemy = {}
        for i, ex in enumerate(cfg.enemy_x_list):
            enemy[f'B0{i+1}00'] = (ex, cfg.enemy_y, 180)
        
        # 转换为地球坐标
        for aid, (x, y, h) in {**friendly, **enemy}.items():
            lon, lat = self.coord_sys.battlefield_to_geodetic(x, y)
            heading = self.coord_sys.convert_heading_to_earth(h)
            self.positions[aid] = AircraftPosition(x, y, h, lon, lat, heading)
    
    def get_position(self, agent_id: str) -> AircraftPosition:
        """获取飞机位置"""
        return self.positions.get(agent_id)
    
    def export_aircraft_configs(self) -> Dict[str, Any]:
        """导出为环境配置格式"""
        cfg = self.config
        configs = {}
        
        for aid, pos in self.positions.items():
            is_friendly = aid.startswith('A')
            configs[aid] = {
                'color': 'Red' if is_friendly else 'Blue',
                'model': 'su27sk' if is_friendly else 'f16',
                'init_state': {
                    'ic_long_gc_deg': pos.lon,
                    'ic_lat_geod_deg': pos.lat,
                    'ic_h_sl_ft': cfg.altitude_ft,
                    'ic_psi_true_deg': pos.heading,
                    'ic_u_fps': cfg.speed_fps,
                },
                'missile': cfg.missiles,
            }
        return configs
