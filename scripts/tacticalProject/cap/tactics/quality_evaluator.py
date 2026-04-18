"""
跟踪质量评估算法模块
- 简单版：基于更新频率×年龄×距离
- 专业版：多因子综合评估
"""
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class TrackQuality:
    """跟踪质量评估结果"""
    target_id: str
    quality_score: float  # 0-1
    is_launchable: bool  # 是否满足发射条件
    update_count: int  # 连续更新次数
    age_seconds: float  # 最后更新距今秒数
    distance_km: float  # 距离
    factors: Dict[str, float]  # 各因子得分


class SimpleQualityEvaluator:
    """简单质量评估器 - 基于更新频率×年龄×距离"""
    
    def __init__(self, min_quality_for_launch: float = 0.6):
        self.min_quality = min_quality_for_launch
        self.track_expire_time = 30.0  # 过期时间(秒)
        self.max_detection_range = 200.0  # 最大探测距离
    
    def evaluate(self, 
                 target_id: str,
                 update_count: int,
                 last_update_time: float,
                 current_time: float,
                 distance: float) -> TrackQuality:
        """简单质量评估
        
        Quality = update_factor × age_factor × distance_factor
        """
        age = current_time - last_update_time
        
        # 更新因子：连续更新5次达到满分
        update_factor = min(1.0, update_count / 5)
        
        # 年龄因子：刚更新为1.0，30秒未更新为0
        age_factor = max(0, 1.0 - age / self.track_expire_time)
        
        # 距离因子：近距离高，远距离低
        dist_factor = max(0.3, 1.0 - 0.7 * distance / self.max_detection_range)
        
        quality = update_factor * age_factor * dist_factor
        
        return TrackQuality(
            target_id=target_id,
            quality_score=quality,
            is_launchable=quality >= self.min_quality,
            update_count=update_count,
            age_seconds=age,
            distance_km=distance,
            factors={
                'update': update_factor,
                'age': age_factor,
                'distance': dist_factor
            }
        )


class AdvancedQualityEvaluator:
    """高级质量评估器 - 多因子综合评估
    
    额外考虑：
    - 多普勒盲区
    - RCS
    - 电子干扰
    - 多源确认
    """
    
    def __init__(self, min_quality_for_launch: float = 0.6):
        self.min_quality = min_quality_for_launch
        self.track_expire_time = 30.0
        self.max_detection_range = 200.0
    
    def evaluate(self,
                 target_id: str,
                 update_count: int,
                 last_update_time: float,
                 current_time: float,
                 distance: float,
                 in_notch: bool = False,
                 awacs_confirmed: bool = False,
                 radar_confirmed: bool = False,
                 detection_prob: float = 0.85) -> TrackQuality:
        """高级多因子质量评估"""
        age = current_time - last_update_time
        
        # 基础因子（与简单版相同）
        update_factor = min(1.0, update_count / 5)
        age_factor = max(0, 1.0 - age / self.track_expire_time)
        dist_factor = max(0.3, 1.0 - 0.7 * distance / self.max_detection_range)
        
        # 高级因子
        # 多普勒盲区惩罚
        notch_factor = 0.3 if in_notch else 1.0
        
        # 多源确认加成
        source_factor = 1.0
        if awacs_confirmed and radar_confirmed:
            source_factor = 1.2
        elif awacs_confirmed or radar_confirmed:
            source_factor = 1.0
        else:
            source_factor = 0.7
        
        # 探测概率因子
        prob_factor = detection_prob
        
        # 综合评分（加权几何平均）
        base_quality = update_factor * age_factor * dist_factor
        quality = base_quality * notch_factor * source_factor * prob_factor
        quality = min(1.0, quality)  # 限制在0-1
        
        return TrackQuality(
            target_id=target_id,
            quality_score=quality,
            is_launchable=quality >= self.min_quality,
            update_count=update_count,
            age_seconds=age,
            distance_km=distance,
            factors={
                'update': update_factor,
                'age': age_factor,
                'distance': dist_factor,
                'notch': notch_factor,
                'source': source_factor,
                'detection': prob_factor
            }
        )


def create_quality_evaluator(use_advanced: bool = False, **kwargs):
    """工厂函数：创建质量评估器"""
    if use_advanced:
        return AdvancedQualityEvaluator(**kwargs)
    else:
        return SimpleQualityEvaluator(**kwargs)
