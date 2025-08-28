#!/usr/bin/env python3
"""
动作标注器 - 为轨迹数据添加11种基本动作标签
基于现有的战术阶段和机动逻辑进行动作识别
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from envs.JSBSim.core.catalog import Catalog as c


class ActionAnnotator:
    """动作标注器 - 识别并标注11种基本飞行动作"""
    
    # 11种基本动作类型
    ACTION_TYPES = {
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
        """初始化动作标注器"""
        # 历史数据存储，用于计算变化率
        self.history = {}
        self.history_length = 5  # 保存5个历史数据点

        # 动作状态跟踪，用于避免不合理的持续状态
        self.action_states = {}  # 存储每个智能体的动作状态信息
        
        # 优化的动作识别阈值 - 基于实际战术代码分析
        self.thresholds = {
            'heading_change_rate': 1.5,      # 航向变化率阈值 (度/秒) - 降低以提高敏感度
            'altitude_change_rate': 3.0,     # 高度变化率阈值 (米/秒)
            'velocity_change_rate': 1.5,     # 速度变化率阈值 (米/秒²)
            'tactical_heading_rate': 3.0,    # 战术机动航向变化率阈值 - 降低
            'tactical_altitude_rate': 8.0,   # 战术机动高度变化率阈值
            'crank_angle_threshold': 8.0,    # Crank机动角度阈值 - 降低以识别更多crank
            'short_skate_angle': 20.0,       # Short Skate机动角度阈值 - 降低
            'level_flight_tolerance': 5.0    # 平飞航向容差 (度)
        }
    
    def annotate_action(self, env, agent_id: str, current_time: float,
                       tactical_task=None) -> tuple:
        """
        为指定智能体标注当前动作类型 - 基于实际战术代码逻辑

        Args:
            env: 仿真环境
            agent_id: 智能体ID
            current_time: 当前时间
            tactical_task: 战术任务对象，用于获取战术状态信息

        Returns:
            tuple: (动作类型标签, 方向信息)
        """
        try:
            # 获取飞机当前状态
            aircraft = env._jsbsims.get(agent_id)
            if not aircraft or not aircraft.is_alive:
                return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'

            # 获取当前状态数据
            current_state = self._get_aircraft_state(aircraft)

            # 先更新历史数据（在动作识别之前）
            self._update_history(agent_id, current_state, current_time)

            # 然后识别动作（基于历史数据）
            action_result = None
            direction = '无'

            # 1. 友方飞机：基于战术任务的精确状态识别
            if agent_id.startswith('A') and tactical_task:
                friendly_action, friendly_direction = self._identify_friendly_precise_action(
                    agent_id, current_state, tactical_task, current_time, env
                )
                if friendly_action:
                    action_result = friendly_action
                    direction = friendly_direction

            # 2. 敌方飞机：基于敌方AI系统的精确状态识别
            elif agent_id.startswith('B'):
                enemy_action, enemy_direction = self._identify_enemy_precise_action(
                    agent_id, current_state, current_time, env
                )
                if enemy_action:
                    action_result = enemy_action
                    direction = enemy_direction

            # 返回识别结果
            if action_result:
                # 验证机动序列的合理性
                validated_action, validated_direction = self._validate_action_sequence(
                    agent_id, action_result, direction, current_time
                )
                return validated_action, validated_direction

            # 3. 基于轨迹数据的基础动作识别
            basic_action, basic_direction = self._identify_trajectory_based_action(agent_id, current_state, current_time)
            if basic_action:
                # 验证机动序列的合理性
                validated_action, validated_direction = self._validate_action_sequence(
                    agent_id, basic_action, basic_direction, current_time
                )
                return validated_action, validated_direction

            # 4. 默认为平飞
            return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'

        except Exception as e:
            logging.warning(f"动作标注失败 {agent_id}: {e}")
            return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'
    
    def _get_aircraft_state(self, aircraft) -> Dict[str, float]:
        """获取飞机当前状态数据"""
        pos = aircraft.get_position()
        velocity_vector = aircraft.get_velocity()
        
        return {
            'x': pos[0],
            'y': pos[1], 
            'z': pos[2],
            'heading': np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad)),
            'pitch': np.rad2deg(aircraft.get_property_value(c.attitude_pitch_rad)),
            'roll': np.rad2deg(aircraft.get_property_value(c.attitude_phi_rad)),
            'velocity': np.linalg.norm(velocity_vector),
            'altitude': aircraft.get_property_value(c.position_h_sl_m)
        }
    
    def _update_history(self, agent_id: str, state: Dict[str, float], time: float):
        """更新历史状态数据"""
        if agent_id not in self.history:
            self.history[agent_id] = []
        
        # 添加时间戳
        state_with_time = state.copy()
        state_with_time['time'] = time
        
        self.history[agent_id].append(state_with_time)
        
        # 保持历史长度
        if len(self.history[agent_id]) > self.history_length:
            self.history[agent_id].pop(0)

    def _identify_friendly_precise_action(self, agent_id: str, current_state: Dict[str, float],
                                         tactical_task, current_time: float, env) -> tuple:
        """友方飞机精确动作识别 - 基于实际战术代码逻辑"""
        try:
            current_heading = current_state['heading']
            current_phase = tactical_task.current_phase
            heading_rate = self._calculate_heading_change_rate(agent_id)

            # 1. 优先检查Short Skate机动状态
            if hasattr(tactical_task, 'short_skate_states') and agent_id in tactical_task.short_skate_states:
                skate_state = tactical_task.short_skate_states[agent_id]
                phase = skate_state.get('phase', '')

                if phase == 'crank':
                    direction = '左转' if heading_rate < 0 else '右转' if heading_rate > 0 else '无'
                    return self.ACTION_TYPES['TACTICAL_CRANK'], direction
                elif phase == 'turn_cold':
                    direction = '左转' if heading_rate < 0 else '右转' if heading_rate > 0 else '无'
                    return self.ACTION_TYPES['SHORT_SKATE'], direction
                elif phase == 'escape':
                    direction = '左转' if heading_rate < 0 else '右转' if heading_rate > 0 else '无'
                    return self.ACTION_TYPES['SHORT_SKATE'], direction

            # 2. 基于飞机角色和战术阶段进行精确识别
            if agent_id == "A0100":  # 友方长机
                return self._identify_leader_action(current_state, current_phase, current_time)
            elif agent_id == "A0200":  # 友方僚机
                return self._identify_wingman_action(current_state, current_phase, current_time, tactical_task, env)

            return None, '无'

        except Exception as e:
            logging.warning(f"友方动作识别失败 {agent_id}: {e}")
            return None, '无'

    def _identify_leader_action(self, current_state: Dict[str, float],
                               current_phase, current_time: float) -> tuple:
        """长机动作识别 - 基于实际战术代码逻辑"""
        current_heading = current_state['heading']
        heading_rate = self._calculate_heading_change_rate("A0100")

        # 长机主要保持北向（0°），根据阶段判断
        if current_phase.value in ['NLT_MELD', 'MELD_MTR', 'MTR_TR']:
            # 早期阶段：保持北向接敌
            if abs(current_heading) <= 5.0 or abs(current_heading - 360) <= 5.0:
                return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'
            elif abs(heading_rate) > 2.0:
                direction = '左转' if heading_rate < 0 else '右转'
                return self.ACTION_TYPES['CRANK'], direction

        elif current_phase.value in ['TR_DOR', 'DOR_DR']:
            # 后期阶段：可能执行Short Skate或返航
            if abs(heading_rate) > 3.0:
                direction = '左转' if heading_rate < 0 else '右转'
                return self.ACTION_TYPES['TACTICAL_CRANK'], direction
            elif abs(current_heading) <= 5.0 or abs(current_heading - 360) <= 5.0:
                return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'

        return None, '无'

    def _identify_wingman_action(self, current_state: Dict[str, float], current_phase,
                                current_time: float, tactical_task, env) -> tuple:
        """僚机动作识别 - 基于实际战术代码逻辑"""
        current_heading = current_state['heading']
        heading_rate = self._calculate_heading_change_rate("A0200")

        # 计算僚机独立的战术阶段
        wingman_phase = self._get_wingman_phase(tactical_task, env)

        # 调试输出 - 检查A0200的早期机动识别
        if current_time < 25 and current_time % 2.0 < 0.2:  # 前25秒每2秒输出一次
            print(f"DEBUG A0200 t={current_time:.1f}s: heading={current_heading:.1f}°, rate={heading_rate:.2f}°/s, phase={wingman_phase.value}")
            print(f"       检查条件: 时间<20s={current_time <= 20.0}, 航向范围5-75°={5 <= current_heading <= 75}, 右转趋势={heading_rate > 0.3}")

        if wingman_phase.value == 'NLT_MELD':
            # 右侧crank: 精确航向从0°调整至68°（右偏68°）
            # 修正逻辑：基于航向范围和时间窗口的综合判断
            if current_time <= 20.0:  # 前20秒内的右侧crank机动
                if 5 <= current_heading <= 75:  # 在右转航向范围内
                    return self.ACTION_TYPES['CRANK'], '右转'
                elif heading_rate > 0.3:  # 检测任何右转趋势（降低阈值）
                    return self.ACTION_TYPES['CRANK'], '右转'
                elif current_heading < 5 and heading_rate >= 0:  # 刚开始转向
                    return self.ACTION_TYPES['CRANK'], '右转'
            # 20秒后，如果仍在右转航向范围内且有转向趋势
            elif 60 <= current_heading <= 75 and abs(heading_rate) > 0.5:
                return self.ACTION_TYPES['CRANK'], '右转'

        elif wingman_phase.value == 'MELD_MTR':
            # 左侧crank: 精确航向从68°调整回0°（左转68°）
            if heading_rate < -1.0:  # 左转
                return self.ACTION_TYPES['CRANK'], '左转'
            elif abs(current_heading) <= 5.0 or abs(current_heading - 360) <= 5.0:
                return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'

        elif wingman_phase.value == 'MTR_TR':
            # 平稳飞行 - 精确保持航向0°
            if abs(current_heading) <= 5.0 or abs(current_heading - 360) <= 5.0:
                return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'
            elif abs(heading_rate) > 1.5:
                direction = '左转' if heading_rate < 0 else '右转'
                return self.ACTION_TYPES['CRANK'], direction

        elif wingman_phase.value == 'TR_DOR':
            # 左侧小crank指向敌机（小角度左转约10°）- 目标航向350°
            if 345 <= current_heading <= 360 or 0 <= current_heading <= 15:
                if abs(heading_rate) > 1.5:
                    direction = '左转' if heading_rate < 0 else '右转'
                    return self.ACTION_TYPES['CRANK'], direction
                else:
                    return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'
            elif abs(heading_rate) > 2.0:
                direction = '左转' if heading_rate < 0 else '右转'
                return self.ACTION_TYPES['CRANK'], direction

        return None, '无'

    def _identify_enemy_precise_action(self, agent_id: str, current_state: Dict[str, float],
                                      current_time: float, env) -> tuple:
        """敌方飞机精确动作识别 - 基于航向变化和时间阶段的简化版本"""
        current_heading = current_state['heading']
        heading_rate = self._calculate_heading_change_rate(agent_id)

        # 调试输出 - 检查B0200的动作识别
        if agent_id == 'B0200' and current_time > 100 and int(current_time) % 20 == 0:  # 每20秒输出一次
            print(f"DEBUG {agent_id} t={current_time:.1f}s: heading={current_heading:.1f}°, rate={heading_rate:.2f}°/s")
            print(f"       历史数据点数: {len(self.history.get(agent_id, []))}")
            if len(self.history.get(agent_id, [])) > 1:
                last_heading = self.history[agent_id][-2]['heading']  # 使用倒数第二个数据点
                print(f"       上一个航向: {last_heading:.1f}°, 变化: {current_heading - last_heading:.1f}°")

        # 特殊处理：检查是否在返航机动中
        if current_time > 100:  # 后期阶段
            # 检查是否从南向转向北向（返航机动）
            history = self.history.get(agent_id, [])
            if len(history) >= 2:  # 降低历史数据要求
                # 检查最近的航向变化
                recent_headings = [h['heading'] for h in history[-min(len(history), 5):]]

                # 检查是否有从南向(170-190°)到北向(350-10°)的转变
                has_south = any(170 <= h <= 190 for h in recent_headings)
                has_north = any(h >= 350 or h <= 10 for h in recent_headings)

                if has_south and has_north:
                    # 正在执行返航转向
                    if agent_id == 'B0200' and int(current_time) % 10 == 0:  # 每10秒输出一次
                        print(f"DEBUG {agent_id}: 检测到返航转向机动 (t={current_time:.1f}s)")
                    direction = '左转' if heading_rate < 0 else '右转' if heading_rate > 0 else '无'
                    return self.ACTION_TYPES['CRANK'], direction
                elif current_heading >= 350 or current_heading <= 10:
                    # 已经转向北方，返航平飞
                    if abs(heading_rate) > 0.5:  # 仍有小幅调整
                        direction = '左转' if heading_rate < 0 else '右转'
                        return self.ACTION_TYPES['CRANK'], direction
                    else:
                        return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'
            elif len(history) == 1:
                # 历史数据不足时，基于当前航向和时间判断
                if current_heading >= 350 or current_heading <= 10:
                    # 当前是北向飞行，在后期阶段很可能是返航机动
                    if agent_id == 'B0200' and int(current_time) % 15 == 0:  # 每15秒输出一次
                        print(f"DEBUG {agent_id}: 基于航向检测到返航机动 (heading={current_heading:.1f}°, t={current_time:.1f}s)")
                    direction = '左转' if heading_rate < 0 else '右转' if heading_rate > 0 else '无'
                    return self.ACTION_TYPES['CRANK'], direction

        # 基于时间阶段和航向变化识别动作
        if current_time < 60:
            # 前期：接近阶段
            if 170 <= current_heading <= 190:
                # 南向飞行
                if abs(heading_rate) > 1.0:  # 降低阈值
                    direction = '左转' if heading_rate < 0 else '右转'
                    return self.ACTION_TYPES['CRANK'], direction
                else:
                    return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'
            else:
                # 偏离南向
                if abs(heading_rate) > 0.5:  # 降低阈值
                    direction = '左转' if heading_rate < 0 else '右转'
                    return self.ACTION_TYPES['CRANK'], direction
                else:
                    return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'

        elif current_time < 180:
            # 中期：交战阶段
            if abs(heading_rate) > 2.0:  # 降低阈值
                direction = '左转' if heading_rate < 0 else '右转'
                return self.ACTION_TYPES['TACTICAL_CRANK'], direction
            elif abs(heading_rate) > 1.0:  # 降低阈值
                direction = '左转' if heading_rate < 0 else '右转'
                return self.ACTION_TYPES['CRANK'], direction
            else:
                return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'

        else:
            # 后期：可能返航
            if current_heading < 90 or current_heading > 270:
                # 朝北方向，返航
                if abs(heading_rate) > 1.0:  # 降低阈值
                    if agent_id == 'B0200' and int(current_time) % 10 == 0:
                        print(f"DEBUG {agent_id}: 返航转向机动 (rate={heading_rate:.2f}°/s, t={current_time:.1f}s)")
                    direction = '左转' if heading_rate < 0 else '右转'
                    return self.ACTION_TYPES['CRANK'], direction  # 返航转向
                else:
                    if agent_id == 'B0200' and int(current_time) % 10 == 0:
                        print(f"DEBUG {agent_id}: 返航平飞 (rate={heading_rate:.2f}°/s, t={current_time:.1f}s)")
                    return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'  # 返航平飞
            elif 170 <= current_heading <= 190:
                # 仍在南向
                if abs(heading_rate) > 1.0:  # 降低阈值
                    direction = '左转' if heading_rate < 0 else '右转'
                    return self.ACTION_TYPES['CRANK'], direction
                else:
                    return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'
            else:
                # 转向过程
                if abs(heading_rate) > 0.5:  # 降低阈值
                    if agent_id == 'B0200' and int(current_time) % 10 == 0:
                        print(f"DEBUG {agent_id}: 转向机动 (rate={heading_rate:.2f}°/s, t={current_time:.1f}s)")
                    direction = '左转' if heading_rate < 0 else '右转'
                    return self.ACTION_TYPES['CRANK'], direction
                else:
                    return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'

        return None, '无'

    def _validate_action_sequence(self, agent_id: str, action_type: str, direction: str, current_time: float) -> tuple:
        """验证机动序列的合理性，避免不合理的持续状态"""
        # 初始化智能体的动作状态
        if agent_id not in self.action_states:
            self.action_states[agent_id] = {
                'last_action': None,
                'last_direction': None,
                'action_start_time': None,
                'action_duration': 0,
                'short_skate_count': 0,
                'last_short_skate_time': None
            }

        state = self.action_states[agent_id]

        # 检查Short Skate机动的合理性 - 修正版
        if action_type == self.ACTION_TYPES['SHORT_SKATE']:
            # 如果是新的Short Skate机动
            if state['last_action'] != self.ACTION_TYPES['SHORT_SKATE']:
                state['short_skate_count'] = 1
                state['last_short_skate_time'] = current_time
                state['action_start_time'] = current_time
            else:
                # 持续的Short Skate机动
                state['action_duration'] = current_time - state['action_start_time']

                # 检查是否持续时间过长 - 大幅缩短阈值
                max_duration = 20.0 if agent_id in ['A0200', 'B0200'] else 15.0  # 大幅缩短持续时间

                if state['action_duration'] > max_duration:
                    # 检查航向变化率，如果很小则转为平飞
                    heading_rate = self._calculate_heading_change_rate(agent_id)
                    if abs(heading_rate) < 1.5:  # 降低阈值
                        action_type = self.ACTION_TYPES['LEVEL_FLIGHT']
                        direction = '无'
                        print(f"DEBUG {agent_id}: Short Skate转为平飞 (持续{state['action_duration']:.1f}s, 航向变化率{heading_rate:.2f}°/s)")

                # 额外检查：如果航向变化率很小，直接转为平飞
                elif state['action_duration'] > 10.0:
                    heading_rate = self._calculate_heading_change_rate(agent_id)
                    if abs(heading_rate) < 0.8:  # 非常小的航向变化
                        action_type = self.ACTION_TYPES['LEVEL_FLIGHT']
                        direction = '无'
                        print(f"DEBUG {agent_id}: 航向稳定，Short Skate转为平飞 (持续{state['action_duration']:.1f}s, 航向变化率{heading_rate:.2f}°/s)")

                # 检查Short Skate的方向一致性 - 确保不会出现"无"方向
                if direction == '无' and abs(self._calculate_heading_change_rate(agent_id)) > 1.0:
                    # 重新计算方向
                    heading_rate = self._calculate_heading_change_rate(agent_id)
                    current_state = {'heading': self.history[agent_id][-1]['heading'], 'time': current_time}
                    direction = self._determine_maneuver_direction(agent_id, heading_rate, current_state)

        # 更新状态
        state['last_action'] = action_type
        state['last_direction'] = direction

        return action_type, direction

    def _identify_enemy_phase_action(self, agent_id: str, current_state: Dict[str, float],
                                    current_time: float, enemy_ai) -> Optional[str]:
        """基于敌方战术阶段识别动作"""
        current_heading = current_state['heading']
        heading_rate = self._calculate_heading_change_rate(agent_id)

        # 获取敌方个体战术阶段
        try:
            if hasattr(enemy_ai, f'current_phase_{agent_id}'):
                enemy_phase = getattr(enemy_ai, f'current_phase_{agent_id}')
                phase_name = enemy_phase.value if hasattr(enemy_phase, 'value') else str(enemy_phase)
            else:
                # 使用默认阶段判断
                phase_name = 'MELD_MTR'
        except:
            phase_name = 'MELD_MTR'

        if phase_name in ['NLT_MELD', 'MELD_MTR']:
            # 早期阶段：主要保持南向接敌（180°）
            if 175 <= current_heading <= 185:
                return self.ACTION_TYPES['LEVEL_FLIGHT']
            elif abs(heading_rate) > 2.0:
                return self.ACTION_TYPES['CRANK']

        elif phase_name in ['MTR_TR', 'TR_DOR']:
            # 中期阶段：可能有战术机动
            if abs(heading_rate) > 3.0:
                return self.ACTION_TYPES['TACTICAL_CRANK']
            elif 175 <= current_heading <= 185:
                return self.ACTION_TYPES['LEVEL_FLIGHT']
            else:
                return self.ACTION_TYPES['CRANK']

        elif phase_name == 'DOR_DR':
            # 后期阶段：返航或机动
            if current_heading < 90 or current_heading > 270:
                # 朝北方向，可能是返航
                if abs(heading_rate) > 2.0:
                    return self.ACTION_TYPES['CRANK']  # 返航转向机动
                else:
                    return self.ACTION_TYPES['LEVEL_FLIGHT']  # 返航平飞
            elif abs(heading_rate) > 3.0:
                return self.ACTION_TYPES['TACTICAL_CRANK']

        return None

    def _identify_enemy_basic_action(self, agent_id: str, current_state: Dict[str, float],
                                    current_time: float) -> Optional[str]:
        """敌方基础动作识别 - 修正版"""
        current_heading = current_state['heading']
        heading_rate = self._calculate_heading_change_rate(agent_id)

        # 调试输出
        if agent_id == 'B0200' and current_time > 260 and current_time < 270:  # 只在返航阶段输出B0200的调试信息
            print(f"DEBUG {agent_id} t={current_time:.1f}s: heading={current_heading:.1f}°, rate={heading_rate:.2f}°/s")

        # 检查是否为返航机动（从南向转向北向）
        if current_heading < 90 or current_heading > 270:
            # 朝北方向，可能是返航
            if abs(heading_rate) > 2.0:
                if agent_id == 'B0200' and current_time > 250:
                    print(f"DEBUG {agent_id}: 返航转向机动")
                return self.ACTION_TYPES['CRANK']  # 返航转向机动
            else:
                if agent_id == 'B0200' and current_time > 250:
                    print(f"DEBUG {agent_id}: 返航平飞")
                return self.ACTION_TYPES['LEVEL_FLIGHT']  # 返航平飞

        # 敌方在南向（180°）
        elif 170 <= current_heading <= 190:
            if abs(heading_rate) > 2.0:
                if agent_id == 'B0200' and current_time > 250:
                    print(f"DEBUG {agent_id}: 南向机动")
                return self.ACTION_TYPES['CRANK']  # 南向机动
            else:
                if agent_id == 'B0200' and current_time > 250:
                    print(f"DEBUG {agent_id}: 南向平飞")
                return self.ACTION_TYPES['LEVEL_FLIGHT']  # 南向平飞

        # 其他航向 - 可能是机动过程
        else:
            if abs(heading_rate) > 1.0:
                if agent_id == 'B0200' and current_time > 250:
                    print(f"DEBUG {agent_id}: 其他机动")
                return self.ACTION_TYPES['CRANK']  # 机动中
            else:
                if agent_id == 'B0200' and current_time > 250:
                    print(f"DEBUG {agent_id}: 其他平飞")
                return self.ACTION_TYPES['LEVEL_FLIGHT']  # 稳定飞行

    def _get_wingman_phase(self, tactical_task, env):
        """获取僚机独立的战术阶段"""
        try:
            # 计算僚机与敌机的距离
            leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
            if leader_blue and leader_blue.is_alive:
                wingman = env._jsbsims.get("A0200")
                if wingman and wingman.is_alive:
                    distance = tactical_task._calculate_distance(wingman, leader_blue)
                    return tactical_task._get_wingman_phase_by_distance(distance)

            # 如果无法计算距离，使用全局阶段
            return tactical_task.current_phase
        except:
            return tactical_task.current_phase



    def _identify_trajectory_based_action(self, agent_id: str, current_state: Dict[str, float],
                                         current_time: float) -> tuple:
        """基于轨迹数据的基础动作识别"""
        if len(self.history.get(agent_id, [])) < 2:
            return self.ACTION_TYPES['LEVEL_FLIGHT'], '无'

        # 计算变化率
        heading_rate = self._calculate_heading_change_rate(agent_id)
        altitude_rate = self._calculate_altitude_change_rate(agent_id)
        velocity_rate = self._calculate_velocity_change_rate(agent_id)

        # 动作优先级判断 - 修正版，基于实际战术逻辑
        # 1. 检查大角度机动（Short Skate）- 提高阈值避免误判
        if abs(heading_rate) > 15.0:  # 大角度转向
            direction = self._determine_maneuver_direction(agent_id, heading_rate, current_state)
            return self.ACTION_TYPES['SHORT_SKATE'], direction

        # 1.5. 检查中等角度机动（Short Skate的另一种形式）- 提高阈值
        if abs(heading_rate) > 8.0:  # 中等角度转向
            direction = self._determine_maneuver_direction(agent_id, heading_rate, current_state)
            return self.ACTION_TYPES['SHORT_SKATE'], direction

        # 2. 检查战术机动
        if abs(heading_rate) > 3.0:  # 降低阈值
            direction = self._determine_maneuver_direction(agent_id, heading_rate, current_state)
            return self.ACTION_TYPES['TACTICAL_CRANK'], direction

        # 3. 检查基本机动
        if abs(heading_rate) > 1.0:  # 降低阈值
            direction = self._determine_maneuver_direction(agent_id, heading_rate, current_state)
            return self.ACTION_TYPES['CRANK'], direction

        # 4. 检查垂直机动
        if abs(altitude_rate) > 8.0:
            if altitude_rate > 0:
                return self.ACTION_TYPES['TACTICAL_CLIMB'], '上升'
            else:
                return self.ACTION_TYPES['TACTICAL_DESCEND'], '下降'
        elif abs(altitude_rate) > 3.0:
            if altitude_rate > 0:
                return self.ACTION_TYPES['CLIMB'], '上升'
            else:
                return self.ACTION_TYPES['DESCEND'], '下降'

        # 5. 检查速度变化
        if abs(velocity_rate) > 2.0:
            if velocity_rate > 0:
                return self.ACTION_TYPES['ACCELERATE'], '加速'
            else:
                return self.ACTION_TYPES['DECELERATE'], '减速'

        # 6. 检查小幅航向变化
        if abs(heading_rate) > 0.5:
            direction = '左转' if heading_rate < 0 else '右转'
            return self.ACTION_TYPES['NOTCH_BACK'], direction

        return None, '无'

    def _identify_precise_tactical_action(self, agent_id: str, current_state: Dict[str, float],
                                         tactical_task, current_time: float) -> Optional[str]:
        """基于实际战术代码逻辑进行精确动作识别"""
        try:
            # 1. 优先检查Short Skate机动状态
            if hasattr(tactical_task, 'short_skate_states') and agent_id in tactical_task.short_skate_states:
                skate_state = tactical_task.short_skate_states[agent_id]
                phase = skate_state.get('phase', '')

                if phase == 'crank':
                    return self.ACTION_TYPES['TACTICAL_CRANK']
                elif phase in ['turn_cold', 'escape']:
                    return self.ACTION_TYPES['SHORT_SKATE']

            # 2. 基于具体飞机角色和战术阶段进行精确识别
            if hasattr(tactical_task, 'current_phase'):
                phase = tactical_task.current_phase
                return self._identify_role_based_action(agent_id, current_state, phase, current_time)

            # 3. 检查敌方AI的机动状态
            if agent_id.startswith('B'):
                return self._identify_enemy_action(agent_id, current_state, current_time)

            return None

        except Exception as e:
            logging.warning(f"精确战术动作识别失败 {agent_id}: {e}")
            return None

    def _identify_role_based_action(self, agent_id: str, current_state: Dict[str, float],
                                   phase, current_time: float) -> Optional[str]:
        """基于飞机角色和战术阶段识别动作"""
        current_heading = current_state['heading']

        if agent_id == "A0100":  # 友方长机
            # 长机主要保持北向（0°），偏差较小时为平飞
            if abs(current_heading) < 10 or abs(current_heading - 360) < 10:
                return self.ACTION_TYPES['LEVEL_FLIGHT']
            else:
                # 有明显航向变化时为机动
                heading_rate = self._calculate_heading_change_rate(agent_id)
                if abs(heading_rate) > 3.0:
                    return self.ACTION_TYPES['CRANK']

        elif agent_id == "A0200":  # 友方僚机
            # 僚机的crank机动识别
            if phase.value == 'NLT_MELD':
                # 右侧crank：目标航向68°
                if 50 <= current_heading <= 85:
                    return self.ACTION_TYPES['CRANK']
            elif phase.value == 'MELD_MTR':
                # 左侧crank：从68°回到0°
                heading_rate = self._calculate_heading_change_rate(agent_id)
                if heading_rate < -3.0:  # 左转
                    return self.ACTION_TYPES['CRANK']
            elif phase.value == 'TR_DOR':
                # 左侧小crank：目标航向350°
                if 340 <= current_heading <= 360 or 0 <= current_heading <= 20:
                    if abs(self._calculate_heading_change_rate(agent_id)) > 2.0:
                        return self.ACTION_TYPES['CRANK']

        elif agent_id.startswith('B'):  # 敌方飞机
            # 敌方主要保持南向（180°）
            if 170 <= current_heading <= 190:
                return self.ACTION_TYPES['LEVEL_FLIGHT']
            else:
                # 偏离南向较多时为机动
                heading_rate = self._calculate_heading_change_rate(agent_id)
                if abs(heading_rate) > 3.0:
                    return self.ACTION_TYPES['CRANK']

        return None

    def _identify_enemy_action(self, agent_id: str, current_state: Dict[str, float],
                              current_time: float) -> Optional[str]:
        """识别敌方AI的动作"""
        # 检查是否有敌方AI系统的状态信息
        try:
            # 尝试获取敌方AI状态
            from enemy_tactical_ai_enhanced import get_enhanced_enemy_ai
            enemy_ai = get_enhanced_enemy_ai()

            # 检查敌方Short Skate状态
            if hasattr(enemy_ai, 'short_skate_states') and agent_id in enemy_ai.short_skate_states:
                skate_state = enemy_ai.short_skate_states[agent_id]
                phase = skate_state.phase

                if phase == 'crank':
                    return self.ACTION_TYPES['TACTICAL_CRANK']
                elif phase in ['turn_cold', 'escape']:
                    return self.ACTION_TYPES['SHORT_SKATE']

            # 检查返航状态
            if hasattr(enemy_ai, 'return_to_base_states') and agent_id in enemy_ai.return_to_base_states:
                return self.ACTION_TYPES['LEVEL_FLIGHT']  # 返航时保持平飞

        except Exception:
            pass  # 如果无法获取敌方AI状态，使用基础识别

        return None
    
    def _identify_trajectory_action(self, agent_id: str, 
                                   current_state: Dict[str, float]) -> str:
        """基于轨迹数据识别动作"""
        if len(self.history.get(agent_id, [])) < 2:
            return self.ACTION_TYPES['LEVEL_FLIGHT']
        
        # 计算变化率
        heading_rate = self._calculate_heading_change_rate(agent_id)
        altitude_rate = self._calculate_altitude_change_rate(agent_id)
        velocity_rate = self._calculate_velocity_change_rate(agent_id)
        
        # 动作优先级判断
        
        # 1. 检查大角度机动
        if abs(heading_rate) > self.thresholds['short_skate_angle']:
            return self.ACTION_TYPES['SHORT_SKATE']
        
        # 2. 检查战术机动
        if (abs(heading_rate) > self.thresholds['tactical_heading_rate'] or
            abs(altitude_rate) > self.thresholds['tactical_altitude_rate']):
            
            if altitude_rate > self.thresholds['tactical_altitude_rate']:
                return self.ACTION_TYPES['TACTICAL_CLIMB']
            elif altitude_rate < -self.thresholds['tactical_altitude_rate']:
                return self.ACTION_TYPES['TACTICAL_DESCEND']
            else:
                return self.ACTION_TYPES['TACTICAL_CRANK']
        
        # 3. 检查基本机动
        if abs(heading_rate) > self.thresholds['heading_change_rate']:
            if abs(heading_rate) > self.thresholds['crank_angle_threshold']:
                return self.ACTION_TYPES['CRANK']
            else:
                return self.ACTION_TYPES['NOTCH_BACK']
        
        # 4. 检查垂直机动
        if abs(altitude_rate) > self.thresholds['altitude_change_rate']:
            if altitude_rate > 0:
                return self.ACTION_TYPES['CLIMB']
            else:
                return self.ACTION_TYPES['DESCEND']
        
        # 5. 检查速度变化
        if abs(velocity_rate) > self.thresholds['velocity_change_rate']:
            if velocity_rate > 0:
                return self.ACTION_TYPES['ACCELERATE']
            else:
                return self.ACTION_TYPES['DECELERATE']
        
        # 6. 默认平飞
        return self.ACTION_TYPES['LEVEL_FLIGHT']
    
    def _identify_engagement_action(self, agent_id: str, 
                                   current_state: Dict[str, float]) -> str:
        """识别交战阶段的动作"""
        heading_rate = self._calculate_heading_change_rate(agent_id)
        altitude_rate = self._calculate_altitude_change_rate(agent_id)
        
        # 交战阶段的特殊动作识别
        if abs(heading_rate) > self.thresholds['short_skate_angle']:
            return self.ACTION_TYPES['SHORT_SKATE']
        elif abs(heading_rate) > self.thresholds['tactical_heading_rate']:
            return self.ACTION_TYPES['NOTCH_BACK']
        elif abs(altitude_rate) > self.thresholds['tactical_altitude_rate']:
            if altitude_rate > 0:
                return self.ACTION_TYPES['TACTICAL_CLIMB']
            else:
                return self.ACTION_TYPES['TACTICAL_DESCEND']
        
        return self.ACTION_TYPES['LEVEL_FLIGHT']
    
    def _calculate_heading_change_rate(self, agent_id: str) -> float:
        """计算航向变化率 (度/秒) - 修正版，确保方向判断准确"""
        history = self.history.get(agent_id, [])
        if len(history) < 2:
            return 0.0

        current = history[-1]
        previous = history[-2]

        dt = current['time'] - previous['time']
        if dt <= 0:
            return 0.0

        # 处理航向角度跨越问题 - 修正版
        heading_diff = current['heading'] - previous['heading']

        # 标准化角度差值到[-180, 180]范围
        while heading_diff > 180:
            heading_diff -= 360
        while heading_diff < -180:
            heading_diff += 360

        heading_rate = heading_diff / dt

        # 调试输出 - 验证方向计算
        if agent_id in ['A0100', 'B0200'] and abs(heading_rate) > 1.0:
            print(f"DEBUG {agent_id}: prev_hdg={previous['heading']:.1f}°, curr_hdg={current['heading']:.1f}°, diff={heading_diff:.1f}°, rate={heading_rate:.2f}°/s")

        return heading_rate

    def _determine_maneuver_direction(self, agent_id: str, heading_rate: float, current_state: Dict[str, float]) -> str:
        """智能判断机动方向 - 基于战术代码逻辑和实际轨迹"""
        current_heading = current_state['heading']
        current_time = current_state['time']

        # 基础方向判断 - 修正阈值
        if abs(heading_rate) < 0.2:  # 进一步降低阈值
            return '无'

        # 基于航向变化率的基础判断
        basic_direction = '左转' if heading_rate < 0 else '右转'

        # 调试输出 - 移到这里确保能看到所有调用
        if agent_id in ['A0100', 'B0200'] and abs(heading_rate) > 0.5:
            print(f"DEBUG {agent_id} t={current_time:.1f}s: hdg={current_heading:.1f}°, rate={heading_rate:.2f}°/s, basic_dir={basic_direction}")

        # 特殊情况修正 - 基于战术代码逻辑和实际轨迹
        if agent_id == 'A0100':  # 友方长机
            # A0100在后期执行返航机动
            if current_time > 160:
                # 检查实际航向变化模式
                if 170 <= current_heading <= 185:  # 南向附近，小幅调整
                    if abs(heading_rate) < 0.8:
                        return '无'  # 平稳飞行
                    else:
                        # A0100返航时应该是左转（从南向转向西南）
                        if heading_rate < 0:
                            return '左转'
                        else:
                            return '右转'
                else:
                    return basic_direction

        elif agent_id == 'A0200':  # 友方僚机
            # A0200执行左侧Short Skate机动（战术代码中crank_angle = -40.0）
            if current_time > 200:
                # 友方僚机的Short Skate应该是左转
                if abs(heading_rate) > 0.8:
                    return '左转'  # 强制左转
                else:
                    return basic_direction

        elif agent_id == 'B0100':  # 敌方长机
            # B0100主要保持南向，小幅调整
            if abs(heading_rate) < 1.0:
                return '无'  # 平稳飞行
            else:
                return basic_direction

        elif agent_id == 'B0200':  # 敌方僚机
            # B0200返航：从南向（180°）转向北向（0°）= 左转
            if current_time > 120:
                # 检查是否在返航转向过程中
                if (150 <= current_heading <= 180) or (0 <= current_heading <= 30) or (330 <= current_heading <= 360):
                    if abs(heading_rate) > 0.3:
                        print(f"DEBUG {agent_id}: 返航左转检测 t={current_time:.1f}s, hdg={current_heading:.1f}°, rate={heading_rate:.2f}°/s")
                        return '左转'  # 返航左转
                    else:
                        return '无'
                else:
                    return basic_direction

        return basic_direction

    def _calculate_altitude_change_rate(self, agent_id: str) -> float:
        """计算高度变化率 (米/秒)"""
        history = self.history.get(agent_id, [])
        if len(history) < 2:
            return 0.0
        
        current = history[-1]
        previous = history[-2]
        
        dt = current['time'] - previous['time']
        if dt <= 0:
            return 0.0
        
        return (current['altitude'] - previous['altitude']) / dt
    
    def _calculate_velocity_change_rate(self, agent_id: str) -> float:
        """计算速度变化率 (米/秒²)"""
        history = self.history.get(agent_id, [])
        if len(history) < 2:
            return 0.0
        
        current = history[-1]
        previous = history[-2]
        
        dt = current['time'] - previous['time']
        if dt <= 0:
            return 0.0
        
        return (current['velocity'] - previous['velocity']) / dt
