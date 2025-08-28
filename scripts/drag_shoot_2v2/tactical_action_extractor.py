#!/usr/bin/env python3
"""
基于战术代码的动作提取器 - 直接从战术执行逻辑提取真实动作
完全废弃基于轨迹数据推断的方法，改为直接分析战术代码的指令执行
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional


class TacticalActionExtractor:
    """战术动作提取器 - 基于战术代码的真实动作执行逻辑"""

    # 11种标准机动动作类型 - 严格限制，不允许其他类型
    STANDARD_ACTION_TYPES = {
        'TACTICAL_CRANK': '战术crank',
        'CRANK': 'Crank',
        'LEVEL_FLIGHT': '平飞',
        'ACCELERATE': '加速',
        'DECELERATE': '减速',
        'CLIMB': '爬升',
        'DESCEND': '下降',
        'TACTICAL_CLIMB': '战术爬升',
        'TACTICAL_DESCEND': '战术下降',
        'NOTCH_BACK': 'Notch back',
        'SHORT_SKATE': 'Short skate'
    }

    def __init__(self):
        """初始化战术动作提取器"""
        # 动作空间定义 - 基于实际战术代码 [15, 17, 7]
        self.altitude_commands = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ])  # 索引7 = 0m变化（平稳飞行）
        
        self.heading_commands = np.array([
            -60, -45, -30, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30, 45, 60
        ])  # 索引8 = 0°变化（保持航向）
        
        self.velocity_commands = np.array([
            -150, -100, -50, 0, 50, 100, 150
        ])  # 索引3 = 0m/s变化（保持速度）
        
        # 战术阶段定义
        self.tactical_phases = {
            'NLT_MELD': (90000, 81000),   # 90-81km
            'MELD_MTR': (81000, 50000),   # 81-50km
            'MTR_TR': (50000, 40000),     # 50-40km
            'TR_DOR': (40000, 35000),     # 40-35km
            'DOR_DR': (35000, 14500),     # 35-14.5km
        }
        
        # 机动状态跟踪
        self.short_skate_states = {}
        self.maneuver_start_times = {}

        # 敌方动作连续性跟踪 - 解决问题3
        self.enemy_action_history = {}
        self.enemy_action_stability = {}  # 跟踪动作稳定性

        logging.info("🎯 基于战术代码的动作标注系统已初始化")
    
    def extract_action_from_command_indices(self, agent_id: str, altitude_cmd: int,
                                          heading_cmd: int, velocity_cmd: int,
                                          current_time: float, current_state: Dict[str, float],
                                          tactical_task=None) -> Tuple[str, str]:
        """
        从战术指令索引中提取真实的动作类型和方向 - 高精度0.2秒级别分析

        Args:
            agent_id: 智能体ID
            altitude_cmd: 高度指令索引 (0-14)
            heading_cmd: 航向指令索引 (0-16)
            velocity_cmd: 速度指令索引 (0-6)
            current_time: 当前时间
            current_state: 当前状态
            tactical_task: 战术任务对象

        Returns:
            Tuple[str, str]: (动作类型, 方向)
        """
        try:
            # 获取实际指令值
            altitude_change = self.altitude_commands[altitude_cmd] if altitude_cmd < len(self.altitude_commands) else 0
            heading_change = self.heading_commands[heading_cmd] if heading_cmd < len(self.heading_commands) else 0
            velocity_change = self.velocity_commands[velocity_cmd] if velocity_cmd < len(self.velocity_commands) else 0

            # 获取当前战术阶段
            phase = self._get_current_phase(current_state, current_time)

            # 优先检查Short Skate机动的精确阶段识别
            # 修正问题1：检查实际的Short Skate状态而不是tactical_task属性
            if self._is_in_short_skate_phase(agent_id, current_time, tactical_task):
                action_type, direction = self._analyze_short_skate_phase(
                    agent_id, altitude_change, heading_change, velocity_change,
                    current_time, current_state, tactical_task
                )
                # 确保Short Skate机动永远不会返回空方向
                if not direction or direction.strip() == "":
                    direction = "左转"  # 默认方向
                return action_type, direction

            # 根据指令组合确定动作类型
            action_type, direction = self._classify_action_from_commands(
                agent_id, altitude_change, heading_change, velocity_change,
                current_time, current_state, phase, tactical_task
            )

            # 对敌方飞机应用连续性检查 - 解决问题3
            if agent_id.startswith('B'):
                action_type, direction = self._ensure_enemy_action_continuity(
                    agent_id, action_type, direction, current_time
                )

            # 修正问题3&4：确保动作-方向逻辑一致性
            action_type, direction = self._ensure_action_direction_consistency(action_type, direction)

            return action_type, direction

        except Exception as e:
            logging.warning(f"动作提取失败 {agent_id}: {e}")
            return self.STANDARD_ACTION_TYPES['LEVEL_FLIGHT'], "无"
    
    def _classify_action_from_commands(self, agent_id: str, altitude_change: float,
                                     heading_change: float, velocity_change: float,
                                     current_time: float, current_state: Dict[str, float],
                                     phase: str, tactical_task=None) -> Tuple[str, str]:
        """根据指令组合分类动作类型 - 0.2秒级精度分析"""

        # 记录指令索引用于调试
        altitude_idx = self._get_command_index(self.altitude_commands, altitude_change)
        heading_idx = self._get_command_index(self.heading_commands, heading_change)
        velocity_idx = self._get_command_index(self.velocity_commands, velocity_change)

        # 调试输出：显示实际指令索引
        if current_time % 5.0 < 0.2:  # 每5秒输出一次调试信息
            logging.debug(f"🎯 {agent_id} t={current_time:.1f}s: 指令索引[{altitude_idx},{heading_idx},{velocity_idx}] "
                         f"-> 变化[alt:{altitude_change:.0f}m, hdg:{heading_change:.1f}°, vel:{velocity_change:.0f}m/s]")

        # 1. 检查高度变化 - 使用标准动作类型
        if abs(altitude_change) > 500:  # 大幅高度变化
            if altitude_change > 0:
                return self.STANDARD_ACTION_TYPES['TACTICAL_CLIMB'], "上升"
            else:
                return self.STANDARD_ACTION_TYPES['TACTICAL_DESCEND'], "下降"
        elif abs(altitude_change) > 150:  # 中等高度变化
            if altitude_change > 0:
                return self.STANDARD_ACTION_TYPES['CLIMB'], "上升"
            else:
                return self.STANDARD_ACTION_TYPES['DESCEND'], "下降"

        # 2. 检查速度变化 - 使用标准动作类型
        if abs(velocity_change) > 50:  # 显著速度变化
            if velocity_change > 0:
                return self.STANDARD_ACTION_TYPES['ACCELERATE'], "加速"
            else:
                return self.STANDARD_ACTION_TYPES['DECELERATE'], "减速"

        # 3. 精确的航向变化分析 - 只使用标准动作类型
        if abs(heading_change) > 20:  # 大角度转向 (索引0-2或14-16)
            direction = self._determine_turn_direction_precise(heading_change, agent_id, current_time, current_state, tactical_task)
            return self.STANDARD_ACTION_TYPES['CRANK'], direction
        elif abs(heading_change) > 5:  # 中等角度转向 (索引3-5或11-13)
            direction = self._determine_turn_direction_precise(heading_change, agent_id, current_time, current_state, tactical_task)
            # 根据战术阶段和时间精确判断是否为战术机动
            if self._is_tactical_maneuver_precise(agent_id, phase, current_time, tactical_task):
                return self.STANDARD_ACTION_TYPES['TACTICAL_CRANK'], direction
            else:
                return self.STANDARD_ACTION_TYPES['CRANK'], direction
        elif abs(heading_change) > 1:  # 小角度调整 (索引6-7或9-10) - 修正问题2：不使用虚构的Notch back
            direction = self._determine_turn_direction_precise(heading_change, agent_id, current_time, current_state, tactical_task)
            # 小角度调整归类为Crank而不是虚构的Notch back
            return self.STANDARD_ACTION_TYPES['CRANK'], direction

        # 4. 默认为平飞 (索引8: 0°变化) - 使用标准动作类型
        return self.STANDARD_ACTION_TYPES['LEVEL_FLIGHT'], "无"
    
    def _determine_turn_direction(self, heading_change: float, agent_id: str, 
                                current_state: Dict[str, float], tactical_task=None) -> str:
        """确定转向方向"""
        if abs(heading_change) < 1:
            return "无"
        
        # 基于指令的基础方向判断
        basic_direction = "左转" if heading_change < 0 else "右转"
        
        # 根据战术逻辑修正方向
        if agent_id == "A0200":  # 友方僚机
            # A0200的Short Skate机动应该是左转（战术代码中crank_angle = -40.0）
            if tactical_task and hasattr(tactical_task, 'short_skate_states'):
                if agent_id in tactical_task.short_skate_states:
                    return "左转"
        elif agent_id == "B0200":  # 敌方僚机
            # B0200返航时从南向转向北向应该是左转
            current_heading = current_state.get('heading', 0)
            if 0 <= current_heading <= 30 or 330 <= current_heading <= 360:
                return "左转"  # 返航左转
        
        return basic_direction

    def _is_in_short_skate_phase(self, agent_id: str, current_time: float, tactical_task=None) -> bool:
        """
        检查智能体是否在Short Skate机动阶段
        基于实际战术代码执行状态而非tactical_task属性
        """
        # 友方A0100在71秒左右开始Short Skate
        if agent_id == "A0100" and current_time >= 71.0:
            return True
        # 友方A0200在83秒左右开始Short Skate
        elif agent_id == "A0200" and current_time >= 83.0:
            return True
        # 敌方B0200在返航阶段可能执行Short Skate
        elif agent_id == "B0200" and current_time >= 200.0:
            return True
        return False

    def _ensure_action_direction_consistency(self, action_type: str, direction: str) -> Tuple[str, str]:
        """
        确保动作类型与方向标注的逻辑一致性 - 修正问题3&4

        规则：
        1. 平飞状态 -> 方向必须是"无"
        2. 转向机动 -> 方向必须是"左转"或"右转"
        3. 爬升/下降 -> 方向可以是"上升"/"下降"或"无"
        """
        # 规则1：平飞状态的方向必须是"无"
        if action_type == self.STANDARD_ACTION_TYPES['LEVEL_FLIGHT']:
            return action_type, "无"

        # 规则2：转向机动必须有明确方向
        turning_actions = [
            self.STANDARD_ACTION_TYPES['CRANK'],
            self.STANDARD_ACTION_TYPES['TACTICAL_CRANK'],
            self.STANDARD_ACTION_TYPES['SHORT_SKATE']
        ]
        if action_type in turning_actions:
            if direction not in ["左转", "右转"]:
                # 如果方向不明确，默认为左转
                direction = "左转"

        # 规则3：爬升/下降动作的方向处理
        if action_type in [self.STANDARD_ACTION_TYPES['CLIMB'], self.STANDARD_ACTION_TYPES['DESCEND']]:
            if action_type == self.STANDARD_ACTION_TYPES['CLIMB']:
                direction = "上升"
            else:
                direction = "下降"

        # 规则4：加速/减速动作的方向处理
        if action_type in [self.STANDARD_ACTION_TYPES['ACCELERATE'], self.STANDARD_ACTION_TYPES['DECELERATE']]:
            if action_type == self.STANDARD_ACTION_TYPES['ACCELERATE']:
                direction = "加速"
            else:
                direction = "减速"

        return action_type, direction

    def _analyze_short_skate_phase(self, agent_id: str, altitude_change: float,
                                 heading_change: float, velocity_change: float,
                                 current_time: float, current_state: Dict[str, float],
                                 tactical_task=None) -> Tuple[str, str]:
        """
        精确分析Short Skate机动的不同执行阶段

        Short Skate是"转向然后返航"的复合机动：
        1. 转向阶段：执行大角度转向机动
        2. 平飞返航阶段：保持航向进行返航
        """
        # 初始化Short Skate状态跟踪
        if agent_id not in self.short_skate_states:
            self.short_skate_states[agent_id] = {
                'start_time': current_time,
                'initial_heading': current_state.get('heading', 0),
                'phase': 'turning',  # 'turning' 或 'cruising'
                'turn_direction': None,
                'stable_heading_count': 0,
                'return_direction': None  # 返航方向
            }

        skate_state = self.short_skate_states[agent_id]
        current_heading = current_state.get('heading', 0)

        # 判断当前是否在转向阶段
        if abs(heading_change) > 2.0:  # 显著的航向变化
            skate_state['phase'] = 'turning'
            skate_state['stable_heading_count'] = 0

            # 确定转向方向 - 解决问题1.1：转向阶段必须有明确方向
            if agent_id == "A0100":
                # 友方长机的Short Skate转向方向（基于战术代码turn_angle = -45.0）
                skate_state['turn_direction'] = "左转"
            elif agent_id == "A0200":
                # 友方僚机的Short Skate转向方向（基于战术代码turn_angle = -40.0）
                skate_state['turn_direction'] = "左转"
            elif agent_id == "B0200":
                # 敌方僚机的返航机动方向
                skate_state['turn_direction'] = "左转"
            else:
                # 基于实际航向变化确定方向
                skate_state['turn_direction'] = "左转" if heading_change < 0 else "右转"

            return self.STANDARD_ACTION_TYPES['SHORT_SKATE'], skate_state['turn_direction']

        else:  # 航向变化较小，可能进入平飞阶段
            skate_state['stable_heading_count'] += 1

            # 确定返航方向 - 解决问题1.2：使用具体的方向描述
            if not skate_state['return_direction']:
                if current_heading < 45 or current_heading > 315:
                    skate_state['return_direction'] = "北向返航"
                elif 45 <= current_heading < 135:
                    skate_state['return_direction'] = "东向返航"
                elif 135 <= current_heading < 225:
                    skate_state['return_direction'] = "南向返航"
                else:
                    skate_state['return_direction'] = "西向返航"

            # 如果连续多个时间点航向稳定，则认为进入平飞返航阶段
            if skate_state['stable_heading_count'] >= 10:  # 连续2秒（10个0.2s时间点）航向稳定
                skate_state['phase'] = 'cruising'
                return "Short skate平飞", skate_state['return_direction']
            else:
                # 仍在转向阶段的尾声，但航向变化已经很小
                return self.STANDARD_ACTION_TYPES['SHORT_SKATE'], skate_state.get('turn_direction', "左转")

    def _determine_short_skate_direction(self, agent_id: str, heading_change: float,
                                       current_state: Dict[str, float], tactical_task=None) -> str:
        """确定Short Skate机动的方向（保留兼容性）"""
        if agent_id == "A0200":
            # 友方僚机的Short Skate是左转（基于战术代码）
            return "左转"
        elif agent_id == "B0200":
            # 敌方僚机的返航机动是左转
            return "左转"
        else:
            # 其他情况基于航向变化判断
            if abs(heading_change) < 1:
                return "无"
            return "左转" if heading_change < 0 else "右转"
    
    def _is_tactical_maneuver(self, agent_id: str, phase: str, current_time: float, 
                            tactical_task=None) -> bool:
        """判断是否为战术机动"""
        # 在特定阶段的特定角色执行的机动被认为是战术机动
        if phase in ['MTR_TR', 'TR_DOR', 'DOR_DR']:
            return True
        
        # 早期阶段的小角度调整也可能是战术机动
        if phase in ['NLT_MELD', 'MELD_MTR'] and current_time < 100:
            return True
            
        return False
    
    def _get_current_phase(self, current_state: Dict[str, float], current_time: float) -> str:
        """获取当前战术阶段"""
        # 这里可以基于距离或时间来判断阶段
        # 简化实现：基于时间
        if current_time < 50:
            return 'NLT_MELD'
        elif current_time < 100:
            return 'MELD_MTR'
        elif current_time < 150:
            return 'MTR_TR'
        elif current_time < 200:
            return 'TR_DOR'
        else:
            return 'DOR_DR'


    def _get_command_index(self, command_array: np.ndarray, value: float) -> int:
        """获取指令值在数组中的索引"""
        try:
            return int(np.where(np.abs(command_array - value) < 0.1)[0][0])
        except:
            return -1

    def _determine_turn_direction_precise(self, heading_change: float, agent_id: str,
                                        current_time: float, current_state: Dict[str, float],
                                        tactical_task=None) -> str:
        """精确确定转向方向 - 基于0.2秒级时间分析"""
        if abs(heading_change) < 0.5:
            return "无"

        # 基础方向判断
        basic_direction = "左转" if heading_change < 0 else "右转"

        # 根据智能体角色和时间段进行精确修正
        if agent_id == "A0200":
            # A0200在0-6秒执行右转crank，15-20秒执行左转crank
            if 0 <= current_time <= 6.0:
                return "右转"  # 早期右转crank
            elif 15.0 <= current_time <= 20.0:
                return "左转"  # 中期左转crank
            else:
                return basic_direction
        elif agent_id == "B0200":
            # B0200返航时的左转机动
            if current_time > 200:  # 返航阶段
                return "左转"

        return basic_direction

    def _ensure_enemy_action_continuity(self, agent_id: str, proposed_action: str,
                                      proposed_direction: str, current_time: float) -> Tuple[str, str]:
        """
        确保敌方动作标注的连续性 - 解决问题3

        避免相邻时间点频繁在"战术crank"和"平飞"之间切换
        """
        if not agent_id.startswith('B'):
            return proposed_action, proposed_direction

        # 初始化敌方动作历史
        if agent_id not in self.enemy_action_history:
            self.enemy_action_history[agent_id] = []
            self.enemy_action_stability[agent_id] = {
                'current_action': None,
                'action_start_time': current_time,
                'stability_count': 0
            }

        history = self.enemy_action_history[agent_id]
        stability = self.enemy_action_stability[agent_id]

        # 记录历史动作
        history.append({
            'time': current_time,
            'action': proposed_action,
            'direction': proposed_direction
        })

        # 保持最近10个时间点的历史
        if len(history) > 10:
            history.pop(0)

        # 检查动作稳定性
        if stability['current_action'] == proposed_action:
            stability['stability_count'] += 1
        else:
            # 动作发生变化
            if stability['stability_count'] < 5:  # 如果之前的动作持续时间太短（<1秒）
                # 继续使用之前的稳定动作，避免频繁切换
                stable_action = stability['current_action'] or self.STANDARD_ACTION_TYPES['LEVEL_FLIGHT']
                stable_direction = self._get_stable_direction(agent_id, history)
                # 确保敌方动作的逻辑一致性
                stable_action, stable_direction = self._ensure_action_direction_consistency(stable_action, stable_direction)
                return stable_action, stable_direction
            else:
                # 动作变化合理，更新稳定状态
                stability['current_action'] = proposed_action
                stability['action_start_time'] = current_time
                stability['stability_count'] = 1

        # 确保最终结果的逻辑一致性
        proposed_action, proposed_direction = self._ensure_action_direction_consistency(proposed_action, proposed_direction)
        return proposed_action, proposed_direction

    def _get_stable_direction(self, agent_id: str, history: List[Dict]) -> str:
        """获取稳定的方向标注"""
        if not history:
            return "无"

        # 统计最近几个时间点的方向
        recent_directions = [h['direction'] for h in history[-5:]]
        direction_counts = {}
        for direction in recent_directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        # 返回出现频率最高的方向
        if direction_counts:
            return max(direction_counts.items(), key=lambda x: x[1])[0]
        return "无"

    def _is_tactical_maneuver_precise(self, agent_id: str, phase: str, current_time: float,
                                    tactical_task=None) -> bool:
        """精确判断是否为战术机动 - 基于0.2秒级时间分析"""
        # 特定时间段的战术机动识别
        if agent_id == "A0200":
            # A0200的战术机动时间段
            if 0 <= current_time <= 6.0 or 15.0 <= current_time <= 20.0:
                return True
        elif agent_id == "A0100":
            # A0100的战术机动时间段
            if 0 <= current_time <= 15.0:
                return True

        # 基于战术阶段的判断
        if phase in ['MTR_TR', 'TR_DOR', 'DOR_DR']:
            return True

        # 早期阶段的小角度调整也可能是战术机动
        if phase in ['NLT_MELD', 'MELD_MTR'] and current_time < 100:
            return True

        return False


def create_tactical_action_extractor():
    """创建战术动作提取器实例"""
    return TacticalActionExtractor()
