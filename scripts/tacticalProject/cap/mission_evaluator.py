"""
任务评估模块 - MissionEvaluator
评估CAP任务执行效果，统计风险区威胁和我方损失
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import logging

from .picture import Picture, RiskZone

log = logging.getLogger(__name__)


class MissionResult(Enum):
    """任务结果"""
    IN_PROGRESS = "in_progress"  # 进行中
    SUCCESS = "success"          # 成功
    PARTIAL = "partial"          # 部分成功
    FAILURE = "failure"          # 失败


@dataclass
class MissionStats:
    """任务统计数据"""
    # 威胁统计
    high_zone_threats: int = 0       # 高风险区敌机数
    medium_zone_threats: int = 0     # 中风险区敌机数
    low_zone_threats: int = 0        # 低风险区敌机数
    total_threats: int = 0           # 总威胁数
    
    # 我方统计
    friendly_alive: int = 4          # 我方存活数
    friendly_lost: int = 0           # 我方损失数
    
    # 交战统计
    missiles_fired: int = 0          # 发射导弹数
    kills: int = 0                   # 击落敌机数
    
    # 时间统计
    mission_time: float = 0.0        # 任务时间(秒)
    high_zone_breach_time: float = 0.0  # 高风险区被突破时间
    high_zone_breach_events: int = 0
    ever_high_zone_breached: bool = False


@dataclass
class MissionConfig:
    """任务配置"""
    duration: float = 1200.0         # 任务时长(秒) = 20分钟
    max_loss_ratio: float = 0.75     # 任务要求为“损失不能超过75%”，等于75%不应直接判失败
    high_risk_tolerance: int = 0     # 高风险区允许敌机数
    mid_risk_tolerance: int = 2      # 中风险区允许敌机数


class MissionEvaluator:
    """任务评估器 - 评估CAP任务执行效果"""
    
    def __init__(self, config: Optional[MissionConfig] = None):
        self.config = config or MissionConfig()
        self.stats = MissionStats()
        self._initial_friendlies = 4
        self._high_zone_breach_start: Optional[float] = None
        
    def reset(self, num_friendlies: int = 4):
        """重置评估器"""
        self.stats = MissionStats()
        self._initial_friendlies = num_friendlies
        self._high_zone_breach_start = None
        
    def update(
        self,
        picture: Picture,
        friendly_alive: int,
        mission_time: float,
        missiles_fired: int = 0,
        kills: int = 0,
        current_time: Optional[float] = None,
        track_filter_kwargs: Optional[Dict[str, object]] = None,
    ):
        """
        更新任务统计
        
        Args:
            picture: 态势图
            friendly_alive: 我方存活数
            mission_time: 任务时间(秒)
            missiles_fired: 发射导弹数
            kills: 击落敌机数
        """
        # 更新威胁统计
        filter_kwargs = dict(track_filter_kwargs or {})
        if current_time is not None and "current_time" not in filter_kwargs:
            filter_kwargs["current_time"] = current_time
        self.stats.high_zone_threats = picture.get_high_zone_threat_count(**filter_kwargs)
        self.stats.medium_zone_threats = picture.get_medium_zone_threat_count(**filter_kwargs)
        self.stats.low_zone_threats = len(picture.get_threats_in_zone(RiskZone.LOW, **filter_kwargs))
        self.stats.total_threats = len(picture.get_all_threats(**filter_kwargs))
        
        # 更新我方统计
        self.stats.friendly_alive = friendly_alive
        self.stats.friendly_lost = self._initial_friendlies - friendly_alive
        
        # 更新交战统计
        self.stats.missiles_fired = missiles_fired
        self.stats.kills = kills
        
        # 更新时间统计
        self.stats.mission_time = mission_time
        
        # 跟踪高风险区突破时间
        if self.stats.high_zone_threats > self.config.high_risk_tolerance:
            if self._high_zone_breach_start is None:
                self._high_zone_breach_start = mission_time
                self.stats.high_zone_breach_events += 1
                self.stats.ever_high_zone_breached = True
                log.warning(f"⚠️ [任务评估] 高风险区被突破! 敌机数: {self.stats.high_zone_threats}")
            self.stats.high_zone_breach_time = mission_time - self._high_zone_breach_start
        else:
            self._high_zone_breach_start = None
            self.stats.high_zone_breach_time = 0.0
    
    def evaluate(self) -> MissionResult:
        """
        评估任务结果
        
        Returns:
            MissionResult: 任务结果
        """
        # 检查失败条件
        if self._check_failure():
            return MissionResult.FAILURE
        
        # 检查任务是否完成
        if self.stats.mission_time >= self.config.duration:
            if self._check_success():
                return MissionResult.SUCCESS
            return MissionResult.PARTIAL
        
        return MissionResult.IN_PROGRESS
    
    def _check_failure(self) -> bool:
        """检查是否任务失败"""
        # 条件1: 我方损失超过阈值
        if self._initial_friendlies > 0:
            loss_ratio = self.stats.friendly_lost / self._initial_friendlies
            if loss_ratio > self.config.max_loss_ratio:
                return True
        
        # 条件2: 高风险区被突破超过60秒
        if self.stats.high_zone_breach_time > 60.0:
            return True
        
        return False
    
    def _check_success(self) -> bool:
        """检查是否任务成功"""
        if self.stats.ever_high_zone_breached:
            return False

        # 条件1: 高风险区无敌机
        if self.stats.high_zone_threats > self.config.high_risk_tolerance:
            return False
        
        # 条件2: 中风险区敌机数在容忍范围内
        if self.stats.medium_zone_threats > self.config.mid_risk_tolerance:
            return False
        
        # 条件3: 我方损失在可接受范围内
        if self._initial_friendlies > 0:
            loss_ratio = self.stats.friendly_lost / self._initial_friendlies
            if loss_ratio > self.config.max_loss_ratio:
                return False
        
        return True
    
    def get_threat_level(self) -> str:
        """
        获取当前威胁等级
        
        Returns:
            威胁等级: 'critical' / 'high' / 'medium' / 'low' / 'none'
        """
        if self.stats.high_zone_threats > 0:
            return 'critical'
        if self.stats.medium_zone_threats > 2:
            return 'high'
        if self.stats.medium_zone_threats > 0:
            return 'medium'
        if self.stats.total_threats > 0:
            return 'low'
        return 'none'
    
    def get_summary(self) -> Dict:
        """获取任务摘要"""
        result = self.evaluate()
        return {
            'result': result.value,
            'threat_level': self.get_threat_level(),
            'mission_time': self.stats.mission_time,
            'friendly_alive': self.stats.friendly_alive,
            'friendly_lost': self.stats.friendly_lost,
            'high_zone_threats': self.stats.high_zone_threats,
            'high_zone_breach_events': self.stats.high_zone_breach_events,
            'ever_high_zone_breached': self.stats.ever_high_zone_breached,
            'medium_zone_threats': self.stats.medium_zone_threats,
            'total_threats': self.stats.total_threats,
            'missiles_fired': self.stats.missiles_fired,
            'kills': self.stats.kills
        }
