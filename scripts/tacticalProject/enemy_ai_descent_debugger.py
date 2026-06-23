"""
敌方AI异常下降诊断系统
追踪完整的决策链，从高级决策到具体命令执行
"""

import logging
import os
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# 配置调试开关
DEBUG_ENABLED = os.getenv('ENEMY_DESCENT_DEBUG', '0').strip().lower() in ('1', 'true', 'yes', 'on')
DEBUG_FILE = os.path.join(os.path.dirname(__file__), 'cap_results', 'descent_debug.log')

# 确保输出目录存在
os.makedirs(os.path.dirname(DEBUG_FILE), exist_ok=True)


@dataclass
class DecisionTrace:
    """单次决策追踪"""
    timestamp: float
    agent_id: str
    current_step: int
    
    # 物理状态
    altitude_m: float
    vertical_speed_mps: float
    airspeed_mps: float
    
    # 战术决策
    tactical_phase: str
    tactical_mode: str
    threat_level: str
    threat_distance_m: float
    
    # 动作决策
    selected_action: str
    action_reason: str
    
    # 命令输出
    altitude_cmd_idx: int
    heading_cmd_idx: int
    velocity_cmd_idx: int
    
    # 命令转换过程
    command_delta_altitude: float = 0.0
    command_delta_heading: float = 0.0
    command_delta_velocity: float = 0.0
    
    # 函数调用栈
    call_stack: str = ""
    
    # 异常标记
    is_descent: bool = False
    descent_magnitude: float = 0.0
    previous_altitude: float = 0.0


class DescentDebugger:
    """下降异常调试追踪器"""
    
    def __init__(self):
        self.enabled = DEBUG_ENABLED
        self.traces: Dict[str, list] = {}  # agent_id -> [traces]
        self.last_altitude: Dict[str, float] = {}  # 上一帧的高度
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """配置日志记录器"""
        logger = logging.getLogger('DescentDebugger')
        logger.setLevel(logging.DEBUG)
        
        # 文件处理器
        handler = logging.FileHandler(DEBUG_FILE, mode='a', encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_decision_point(
        self,
        env,
        agent_id: str,
        current_time: float,
        tactical_phase: str,
        tactical_mode: str,
        threat_level: str,
        selected_action: str,
        action_reason: str = ""
    ):
        """记录决策点"""
        if not self.enabled:
            return
        
        try:
            from envs.JSBSim.core.catalog import Catalog as c
            
            # 获取物理状态
            agent = env.agents.get(agent_id)
            if not agent:
                return
            
            try:
                altitude = float(agent.get_property_value(c.position_h_sl_m))
                vert_speed = float(agent.get_property_value(c.velocities_v_down_mps)) * -1.0  # 投向上为正
                airspeed = float(agent.get_property_value(c.velocities_vc_mps))
            except:
                altitude = vert_speed = airspeed = 0.0
            
            # 检测下降
            prev_alt = self.last_altitude.get(agent_id, altitude)
            is_descending = altitude < prev_alt
            descent_mag = prev_alt - altitude if is_descending else 0.0
            
            # 记录上一高度
            self.last_altitude[agent_id] = altitude
            
            # 如果初始化traces列表
            if agent_id not in self.traces:
                self.traces[agent_id] = []
            
            trace = DecisionTrace(
                timestamp=current_time,
                agent_id=agent_id,
                current_step=getattr(env, 'current_step', 0),
                altitude_m=altitude,
                vertical_speed_mps=vert_speed,
                airspeed_mps=airspeed,
                tactical_phase=tactical_phase,
                tactical_mode=tactical_mode,
                threat_level=threat_level,
                threat_distance_m=0.0,  # TODO: 从env计算
                selected_action=selected_action,
                action_reason=action_reason,
                altitude_cmd_idx=0,
                heading_cmd_idx=0,
                velocity_cmd_idx=0,
                is_descent=is_descending,
                descent_magnitude=descent_mag,
                previous_altitude=prev_alt
            )
            
            self.traces[agent_id].append(trace)
            
            # 如果高度过低或正在下降，立即打印警告
            if altitude < 5000.0 or is_descending:
                self.logger.warning(
                    f"[决策追踪] {agent_id} phase={tactical_phase} mode={tactical_mode} "
                    f"action={selected_action} alt={altitude:.0f}m v_spd={vert_speed:.1f}m/s "
                    f"reason={action_reason}"
                )
        
        except Exception as e:
            self.logger.error(f"决策追踪错误: {e}")
    
    def log_command_execution(
        self,
        agent_id: str,
        current_time: float,
        action_type: str,
        altitude_cmd_idx: int,
        heading_cmd_idx: int,
        velocity_cmd_idx: int,
        delta_altitude: float = 0.0,
        delta_heading: float = 0.0,
        delta_velocity: float = 0.0
    ):
        """记录命令执行"""
        if not self.enabled or agent_id not in self.traces or not self.traces[agent_id]:
            return
        
        try:
            # 更新最后一条追踪的命令信息
            last_trace = self.traces[agent_id][-1]
            last_trace.altitude_cmd_idx = altitude_cmd_idx
            last_trace.heading_cmd_idx = heading_cmd_idx
            last_trace.velocity_cmd_idx = velocity_cmd_idx
            last_trace.command_delta_altitude = delta_altitude
            last_trace.command_delta_heading = delta_heading
            last_trace.command_delta_velocity = delta_velocity
            
            # 如果这是下降命令，记录详细信息
            if delta_altitude < -50.0:  # 显著下降
                self.logger.warning(
                    f"[命令执行] {agent_id} action={action_type} "
                    f"cmd_indices=({altitude_cmd_idx},{heading_cmd_idx},{velocity_cmd_idx}) "
                    f"delta_alt={delta_altitude:.0f}m"
                )
        
        except Exception as e:
            self.logger.error(f"命令执行记录错误: {e}")
    
    def log_function_call(
        self,
        agent_id: str,
        function_name: str,
        parameters: Dict[str, Any],
        return_value: Optional[Any] = None,
        altitude_before: Optional[float] = None,
        altitude_after: Optional[float] = None
    ):
        """记录函数调用"""
        if not self.enabled:
            return
        
        try:
            # 构建函数信息字符串
            param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])
            
            msg = f"[函数] {agent_id}::{function_name}({param_str})"
            
            if altitude_before is not None and altitude_after is not None:
                alt_change = altitude_after - altitude_before
                msg += f" -> alt_change={alt_change:.0f}m"
                
                # 如果有显著下降，记录为警告
                if alt_change < -100.0:
                    self.logger.warning(msg + f" DESCENT_DETECTED")
                else:
                    self.logger.debug(msg)
            else:
                self.logger.debug(msg)
        
        except Exception as e:
            self.logger.error(f"函数调用记录错误: {e}")
    
    def log_action_decision(
        self,
        agent_id: str,
        action_weights: Dict[str, float],
        selected_action: str,
        current_altitude: float
    ):
        """记录动作决策权重"""
        if not self.enabled or current_altitude > 8000.0:
            return
        
        try:
            # 按权重排序
            sorted_actions = sorted(action_weights.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_actions[:3]
            
            top_str = ", ".join([f"{act}:{w:.2f}" for act, w in top_3])
            
            self.logger.warning(
                f"[动作决策] {agent_id} 高度{current_altitude:.0f}m "
                f"选择={selected_action} "
                f"TOP3={top_str}"
            )
        
        except Exception as e:
            self.logger.error(f"动作决策记录错误: {e}")
    
    def log_altitude_command_generated(
        self,
        agent_id: str,
        altitude_cmd_idx: int,
        altitude_change_m: float,
        reason: str = ""
    ):
        """记录高度命令生成"""
        if not self.enabled:
            return
        
        try:
            if altitude_change_m < -100.0:  # 显著下降
                self.logger.warning(
                    f"[高度命令] {agent_id} idx={altitude_cmd_idx} "
                    f"delta={altitude_change_m:.0f}m "
                    f"reason={reason}"
                )
        
        except Exception as e:
            self.logger.error(f"高度命令记录错误: {e}")
    
    def export_traces(self, output_file: Optional[str] = None) -> str:
        """导出所有追踪为JSON"""
        if output_file is None:
            output_file = os.path.join(
                os.path.dirname(DEBUG_FILE),
                f"descent_traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        try:
            # 转换为字典列表
            all_traces = {}
            for agent_id, traces in self.traces.items():
                all_traces[agent_id] = [asdict(t) for t in traces]
            
            # 写入JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_traces, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"追踪数据已导出到: {output_file}")
            return output_file
        
        except Exception as e:
            self.logger.error(f"导出追踪数据失败: {e}")
            return ""
    
    def get_descent_events(self, agent_id: str) -> list:
        """获取指定智能体的所有下降事件"""
        if agent_id not in self.traces:
            return []
        
        descents = [t for t in self.traces[agent_id] if t.is_descent]
        return descents
    
    def print_summary(self):
        """打印诊断总结"""
        if not self.enabled:
            return
        
        self.logger.info("=" * 80)
        self.logger.info("【下降异常诊断总结】")
        
        for agent_id, traces in self.traces.items():
            descents = [t for t in traces if t.is_descent]
            
            if descents:
                self.logger.info(f"{agent_id}: {len(descents)}次下降事件")
                
                # 找出最大下降
                max_descent = max(descents, key=lambda t: t.descent_magnitude)
                self.logger.info(
                    f"  最大下降: {max_descent.descent_magnitude:.0f}m "
                    f"@t={max_descent.timestamp:.1f}s alt={max_descent.altitude_m:.0f}m "
                    f"action={max_descent.selected_action}"
                )
                
                # 首次下降
                first_descent = descents[0]
                self.logger.info(
                    f"  首次下降: @t={first_descent.timestamp:.1f}s "
                    f"alt={first_descent.altitude_m:.0f}m "
                    f"phase={first_descent.tactical_phase} "
                    f"action={first_descent.selected_action}"
                )
        
        self.logger.info("=" * 80)


# 全局调试器实例
_debugger = DescentDebugger() if DEBUG_ENABLED else None


def get_debugger() -> Optional[DescentDebugger]:
    """获取全局调试器实例"""
    return _debugger


def enable_debug():
    """启用调试"""
    global _debugger
    if _debugger is None:
        _debugger = DescentDebugger()


def disable_debug():
    """禁用调试"""
    global _debugger
    _debugger = None
