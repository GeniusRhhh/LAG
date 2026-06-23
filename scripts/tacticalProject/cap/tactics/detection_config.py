"""
协同探测V4 - 配置加载模块
从cap_config.yaml读取所有距离阈值

V5方案阈值说明：
- 200km（雷达探测范围）和400km（预警机探测范围）为装备阈值
- NLT/MELD/MTR等为战术节点阈值，基于雷达探测信息触发
"""
import os
import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionRangeConfig:
    """协同探测距离配置 - V5优化版"""
    
    # === 装备探测范围（用于判断是否开机）===
    radar_max: float = 200.0          # 机载雷达最大探测距离
    radar_track_range: float = 160.0  # 最大跟踪距离
    radar_lock_range: float = 120.0   # 最大锁定距离
    awacs_max: float = 400.0          # 预警机探测距离
    awacs_error: float = 2.5          # 预警机位置误差
    
    # === 战术节点距离（基于雷达探测信息触发）===
    nlt: float = 180.0                # NLT节点 - 意图识别
    meld: float = 150.0               # MELD节点 - 雷达融合
    mtr: float = 120.0                # MTR节点 - 协同跟踪移交
    
    # === 协同探测配置 ===
    target_lost_timeout: float = 20.0  # 目标丢失超时(秒)
    sweep_total_angle: float = 120.0   # 推磨扫描总角度(度)
    directed_spread: float = 5.0       # 定向扫描分散角度(度)
    
    # === 动态压缩参数 ===
    compression_base: float = 180.0    # 压缩基准距离（NLT理论值）
    mar_threshold: float = 35.0        # 紧急规避阈值
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'DetectionRangeConfig':
        """从YAML配置文件加载"""
        if not os.path.exists(config_path):
            return cls()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        radar = cfg.get('radar', {})
        awacs = cfg.get('awacs', {})
        ranges = cfg.get('control_ranges', {})
        coop = cfg.get('cooperative_detection', {})
        
        return cls(
            radar_max=radar.get('max_detection_range', 200.0),
            radar_track_range=radar.get('max_track_range', 160.0),
            radar_lock_range=radar.get('max_lock_range', 120.0),
            awacs_max=awacs.get('detection_range', 400.0),
            awacs_error=awacs.get('position_error', 2.5),
            nlt=ranges.get('NLT', 180.0),
            meld=ranges.get('MELD', 150.0),
            mtr=ranges.get('MTR', 120.0),
            target_lost_timeout=coop.get('target_lost_timeout', 20.0),
            sweep_total_angle=coop.get('sweep_total_angle', 120.0),
            directed_spread=coop.get('directed_spread', 5.0),
        )
    
    @classmethod
    def default(cls) -> 'DetectionRangeConfig':
        """默认配置"""
        return cls()

