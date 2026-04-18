"""
协同探测V4验证 - 验证指标收集器
收集和统计协同探测算法的性能指标
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# 导入V4类型
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tactics.detection_types import (
    DetectionPhase, DetectionMode, DetectionResultV4, TargetInfo
)


@dataclass
class LockEvent:
    """锁定事件"""
    target_id: str
    time: float
    distance: float
    phase: str
    mode: str


@dataclass
class PhaseTransition:
    """阶段转换事件"""
    from_phase: str
    to_phase: str
    time: float
    distance: float


@dataclass
class ModeTransition:
    """模式转换事件"""
    from_mode: str
    to_mode: str
    time: float
    distance: float
    reason: str = ""


@dataclass
class RecoveryEvent:
    """恢复事件(从SEARCH恢复)"""
    target_id: str
    lost_time: float
    recovered_time: float
    recovery_duration: float
    distance: float


@dataclass
class VerificationMetrics:
    """验证指标汇总"""
    scenario_id: str
    
    # 时间指标
    first_lock_time: Optional[float] = None  # 首次锁定时间(秒)
    total_simulation_time: float = 0.0
    
    # 成功率指标
    lock_success: bool = False  # 在NLT前是否成功锁定
    all_targets_locked: bool = False  # 所有目标是否都被锁定
    launch_ready: bool = False  # 是否满足发射条件
    
    # 连续性指标
    track_continuity: float = 0.0  # 跟踪连续性(0-1)
    
    # 恢复指标
    recovery_times: List[float] = field(default_factory=list)  # 丢失恢复时间列表
    avg_recovery_time: Optional[float] = None
    
    # 事件记录
    lock_events: List[LockEvent] = field(default_factory=list)
    phase_transitions: List[PhaseTransition] = field(default_factory=list)
    mode_transitions: List[ModeTransition] = field(default_factory=list)
    recovery_events: List[RecoveryEvent] = field(default_factory=list)
    
    # 统计
    time_in_sweep: float = 0.0
    time_in_directed: float = 0.0
    time_in_search: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'scenario_id': self.scenario_id,
            'first_lock_time': self.first_lock_time,
            'lock_success': self.lock_success,
            'all_targets_locked': self.all_targets_locked,
            'launch_ready': self.launch_ready,
            'track_continuity': self.track_continuity,
            'avg_recovery_time': self.avg_recovery_time,
            'num_mode_switches': len(self.mode_transitions),
            'time_in_sweep': self.time_in_sweep,
            'time_in_directed': self.time_in_directed,
            'time_in_search': self.time_in_search,
        }


class VerificationCollector:
    """验证指标收集器"""
    
    def __init__(self, scenario_id: str, nlt: float = 150.0, lr: float = 100.0):
        """
        初始化收集器
        
        Args:
            scenario_id: 场景ID
            nlt: NLT距离阈值(km)
            lr: 发射距离阈值(km)
        """
        self.scenario_id = scenario_id
        self.nlt = nlt
        self.lr = lr
        
        self.metrics = VerificationMetrics(scenario_id=scenario_id)
        
        # 内部状态
        self._first_lock_targets: Dict[str, float] = {}  # {target_id: first_lock_time}
        self._lost_start: Dict[str, float] = {}  # {target_id: lost_start_time}
        self._prev_phase: Optional[DetectionPhase] = None
        self._prev_mode: Optional[DetectionMode] = None
        self._prev_time: float = 0.0
        self._track_history: List[bool] = []  # 每步是否有跟踪
        self._passed_nlt: bool = False
        self._passed_lr: bool = False
    
    def step(self, result: DetectionResultV4, distance: float, current_time: float):
        """
        每步更新指标
        
        Args:
            result: 协同探测结果
            distance: 当前最近目标距离(km)
            current_time: 当前时间(秒)
        """
        dt = current_time - self._prev_time
        
        # 1. 记录模式时间
        if self._prev_mode == DetectionMode.SWEEP:
            self.metrics.time_in_sweep += dt
        elif self._prev_mode == DetectionMode.DIRECTED:
            self.metrics.time_in_directed += dt
        elif self._prev_mode == DetectionMode.SEARCH:
            self.metrics.time_in_search += dt
        
        # 2. 检测首次锁定
        for tid, track in result.fused_tracks.items():
            if tid not in self._first_lock_targets and track.is_tracked:
                self._first_lock_targets[tid] = current_time
                self.metrics.lock_events.append(LockEvent(
                    target_id=tid,
                    time=current_time,
                    distance=distance,
                    phase=result.phase.value,
                    mode=result.mode.value
                ))
                
                # 更新首次锁定时间(取最早的)
                if self.metrics.first_lock_time is None:
                    self.metrics.first_lock_time = current_time
        
        # 3. 检测阶段转换
        if result.phase_changed and self._prev_phase is not None:
            self.metrics.phase_transitions.append(PhaseTransition(
                from_phase=self._prev_phase.value,
                to_phase=result.phase.value,
                time=current_time,
                distance=distance
            ))
        
        # 4. 检测模式转换
        if result.mode_changed and self._prev_mode is not None:
            reason = ""
            if result.mode == DetectionMode.SEARCH:
                reason = "目标丢失"
            elif result.mode == DetectionMode.SWEEP:
                reason = "预警不可用"
            elif result.mode == DetectionMode.DIRECTED:
                if self._prev_mode == DetectionMode.SEARCH:
                    reason = "目标恢复"
                else:
                    reason = "预警可用"
            
            self.metrics.mode_transitions.append(ModeTransition(
                from_mode=self._prev_mode.value,
                to_mode=result.mode.value,
                time=current_time,
                distance=distance,
                reason=reason
            ))
        
        # 5. 检测SEARCH进入/退出(恢复时间)
        if result.mode == DetectionMode.SEARCH:
            # 进入SEARCH
            for tid in result.search_regions:
                if tid not in self._lost_start:
                    self._lost_start[tid] = current_time
        else:
            # 退出SEARCH
            for tid in list(self._lost_start.keys()):
                if tid in result.fused_tracks:
                    lost_time = self._lost_start[tid]
                    recovery_duration = current_time - lost_time
                    self.metrics.recovery_events.append(RecoveryEvent(
                        target_id=tid,
                        lost_time=lost_time,
                        recovered_time=current_time,
                        recovery_duration=recovery_duration,
                        distance=distance
                    ))
                    self.metrics.recovery_times.append(recovery_duration)
                    del self._lost_start[tid]
        
        # 6. 跟踪连续性
        has_track = len(result.fused_tracks) > 0
        self._track_history.append(has_track)
        
        # 7. 检测NLT/LR穿越
        if not self._passed_nlt and distance <= self.nlt:
            self._passed_nlt = True
            # 检查锁定成功
            self.metrics.lock_success = len(self._first_lock_targets) > 0
            self.metrics.all_targets_locked = (
                len(self._first_lock_targets) >= len(result.target_states)
            )
        
        if not self._passed_lr and distance <= self.lr:
            self._passed_lr = True
            self.metrics.launch_ready = len(result.ready_for_launch) > 0
        
        # 更新状态
        self._prev_phase = result.phase
        self._prev_mode = result.mode
        self._prev_time = current_time
        self.metrics.total_simulation_time = current_time
    
    def finalize(self) -> VerificationMetrics:
        """完成收集，计算最终指标"""
        # 跟踪连续性
        if self._track_history:
            self.metrics.track_continuity = sum(self._track_history) / len(self._track_history)
        
        # 平均恢复时间
        if self.metrics.recovery_times:
            self.metrics.avg_recovery_time = np.mean(self.metrics.recovery_times)
        
        return self.metrics


class VerificationReporter:
    """验证报告生成器"""
    
    def __init__(self):
        self.all_metrics: List[VerificationMetrics] = []
    
    def add_result(self, metrics: VerificationMetrics):
        """添加一个场景的结果"""
        self.all_metrics.append(metrics)
    
    def generate_summary(self) -> str:
        """生成汇总报告"""
        if not self.all_metrics:
            return "无验证结果"
        
        lines = []
        lines.append("=" * 70)
        lines.append("协同探测V4验证报告")
        lines.append("=" * 70)
        lines.append("")
        
        # 汇总统计
        total = len(self.all_metrics)
        lock_success_count = sum(1 for m in self.all_metrics if m.lock_success)
        launch_ready_count = sum(1 for m in self.all_metrics if m.launch_ready)
        
        first_lock_times = [m.first_lock_time for m in self.all_metrics if m.first_lock_time]
        avg_first_lock = np.mean(first_lock_times) if first_lock_times else None
        
        recovery_times = []
        for m in self.all_metrics:
            recovery_times.extend(m.recovery_times)
        avg_recovery = np.mean(recovery_times) if recovery_times else None
        
        lines.append("【汇总统计】")
        lines.append(f"  场景总数: {total}")
        lines.append(f"  锁定成功率: {lock_success_count}/{total} ({lock_success_count/total*100:.1f}%)")
        lines.append(f"  发射就绪率: {launch_ready_count}/{total} ({launch_ready_count/total*100:.1f}%)")
        lines.append(f"  平均首次锁定时间: {avg_first_lock:.1f}秒" if avg_first_lock else "  平均首次锁定时间: N/A")
        lines.append(f"  平均丢失恢复时间: {avg_recovery:.1f}秒" if avg_recovery else "  平均丢失恢复时间: N/A")
        lines.append("")
        
        # 各场景详情
        lines.append("【各场景详情】")
        lines.append("-" * 70)
        lines.append(f"{'场景ID':8} {'锁定':6} {'就绪':6} {'首锁(s)':10} {'恢复(s)':10} {'连续性':8}")
        lines.append("-" * 70)
        
        for m in self.all_metrics:
            lock_str = "Y" if m.lock_success else "N"
            ready_str = "Y" if m.launch_ready else "N"
            first_lock_str = f"{m.first_lock_time:.1f}" if m.first_lock_time else "-"
            recovery_str = f"{m.avg_recovery_time:.1f}" if m.avg_recovery_time else "-"
            continuity_str = f"{m.track_continuity*100:.1f}%"
            
            lines.append(f"{m.scenario_id:8} {lock_str:6} {ready_str:6} {first_lock_str:10} {recovery_str:10} {continuity_str:8}")
        
        lines.append("-" * 70)
        lines.append("")
        
        # 失败场景分析
        failed = [m for m in self.all_metrics if not m.lock_success]
        if failed:
            lines.append("【失败场景分析】")
            for m in failed:
                lines.append(f"  {m.scenario_id}:")
                lines.append(f"    模式切换次数: {len(m.mode_transitions)}")
                lines.append(f"    SWEEP时间: {m.time_in_sweep:.1f}秒")
                lines.append(f"    SEARCH时间: {m.time_in_search:.1f}秒")
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save_report(self, filepath: str):
        """保存报告到文件"""
        report = self.generate_summary()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"验证报告已保存: {filepath}")
