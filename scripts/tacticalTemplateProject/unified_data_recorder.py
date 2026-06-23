#!/usr/bin/env python3
"""
统一数据记录模块
为拖曳射击和钳形夹击项目提供标准化的CSV数据格式
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from envs.JSBSim.core.catalog import Catalog as c
# from tactical_action_extractor import TacticalActionExtractor  # 已禁用动作标注系统


try:
    from tactical_utils import TacticalUtils
except Exception:
    TacticalUtils = None


class UnifiedDataRecorder:
    """统一数据记录器 - 标准化CSV格式"""
    
    def __init__(self, project_name: str = "tactical_simulation", trajectory_mode: str = "legacy"):
        """
        初始化统一数据记录器

        Args:
            project_name: 项目名称，用于文件命名前缀
        """
        self.project_name = project_name
        self.trajectory_mode = trajectory_mode
        self.trajectory_data = []
        self.frame_level_data = []
        self.radar_data = []
        self.missile_data = []
        self._pairwise_frame_disabled = False
        self._pairwise_frame_disable_reason = ""
        self.frame_level_columns = [
            't',
            'x1', 'y1', 'z1',
            'x2', 'y2', 'z2',
            'VEL_A0100', 'VEL_A0200', 'VEL_B0100', 'VEL_B0200',
            'd1', 'h1', 'jin1', 'fang1', 'yitu1',
            'd2', 'h2', 'jin2', 'fang2', 'yitu2',
            'd3', 'h3', 'jin3', 'fang3', 'yitu3',
            'd4', 'h4', 'jin4', 'fang4', 'yitu4',
        ]

        # 战术动作提取系统 - 已禁用
        # self.action_extractor = TacticalActionExtractor()
        
        # 标准化状态值
        self.MISSILE_STATES = {
            'BOOST': 'BOOST',
            'MIDCOURSE': 'MIDCOURSE', 
            'TERMINAL': 'TERMINAL',
            'HIT': 'HIT',
            'MISS': 'MISS',
            'TIMEOUT': 'TIMEOUT',
            'ACTIVE': 'ACTIVE',
            'LAUNCHED': 'BOOST',  # 映射到BOOST
            'DESTROYED': 'HIT'    # 映射到HIT
        }
        
        self.RADAR_STATES = {
            'SEARCH': 'SEARCH',
            'TRACK': 'TRACK', 
            'LOCK': 'LOCK',
            'LOST': 'LOST'
        }
        
        self.FLIGHT_PHASES = {
            'APPROACH': 'APPROACH',
            'ENGAGEMENT': 'ENGAGEMENT',
            'DISENGAGEMENT': 'DISENGAGEMENT', 
            'RTB': 'RTB',
            'UNKNOWN': 'UNKNOWN'
        }

    def _get_radar_manager(self, env):
        """Get the radar manager, preferring the global singleton when available."""
        radar_manager = None
        try:
            import sys
            tactical_project_path = os.path.join(os.path.dirname(__file__), '..', 'tacticalProject')
            if tactical_project_path not in sys.path:
                sys.path.insert(0, tactical_project_path)
            from simulation.radar_manager import get_unified_radar_manager
            radar_manager = get_unified_radar_manager()
        except Exception:
            radar_manager = getattr(getattr(env, 'task', None), 'radar_manager', None)
        return radar_manager

    def _safe_position(self, aircraft) -> np.ndarray:
        try:
            return np.array(aircraft.get_position(), dtype=float)
        except Exception:
            return np.array([np.nan, np.nan, np.nan], dtype=float)

    def _get_heading_deg(self, aircraft) -> float:
        for attr_name in ('attitude_heading_true_rad', 'attitude_psi_rad'):
            try:
                catalog_attr = getattr(c, attr_name, None)
                if catalog_attr is None:
                    continue
                return float(np.rad2deg(aircraft.get_property_value(catalog_attr))) % 360.0
            except Exception:
                continue
        return 0.0

    def _safe_distance_m(self, aircraft1, aircraft2) -> float:
        try:
            if TacticalUtils is not None:
                return float(TacticalUtils.calculate_distance_between(aircraft1, aircraft2))
            pos1 = self._safe_position(aircraft1)
            pos2 = self._safe_position(aircraft2)
            return float(np.linalg.norm(pos1 - pos2))
        except Exception:
            return float('nan')

    def _safe_speed_mps(self, aircraft) -> float:
        try:
            velocity = np.array(aircraft.get_velocity(), dtype=float)
            if velocity.shape[0] >= 3:
                return float(np.linalg.norm(velocity[:3]))
        except Exception:
            pass
        return float('nan')

    def _safe_aspect_angle_deg(self, aircraft1, aircraft2) -> float:
        try:
            if TacticalUtils is not None and hasattr(TacticalUtils, 'calculate_aspect_angle'):
                return float(TacticalUtils.calculate_aspect_angle(aircraft1, aircraft2))
        except Exception:
            pass

        try:
            my_pos = self._safe_position(aircraft1)
            target_pos = self._safe_position(aircraft2)
            target_heading = self._get_heading_deg(aircraft2)
            dx = my_pos[0] - target_pos[0]
            dy = my_pos[1] - target_pos[1]
            bearing_to_me = (np.rad2deg(np.arctan2(dy, dx)) + 360.0) % 360.0
            angle = bearing_to_me - target_heading
            while angle > 180.0:
                angle -= 360.0
            while angle < -180.0:
                angle += 360.0
            return abs(float(angle))
        except Exception:
            return float('nan')

    def _safe_bearing_angle_deg(self, aircraft1, aircraft2) -> float:
        try:
            if TacticalUtils is not None and hasattr(TacticalUtils, 'calculate_antenna_train_angle'):
                return float(TacticalUtils.calculate_antenna_train_angle(aircraft1, aircraft2))
        except Exception:
            pass

        try:
            my_pos = self._safe_position(aircraft1)
            target_pos = self._safe_position(aircraft2)
            my_heading = self._get_heading_deg(aircraft1)
            dx = target_pos[0] - my_pos[0]
            dy = target_pos[1] - my_pos[1]
            bearing_to_target = (np.rad2deg(np.arctan2(dy, dx)) + 360.0) % 360.0
            angle = bearing_to_target - my_heading
            while angle > 180.0:
                angle -= 360.0
            while angle < -180.0:
                angle += 360.0
            return abs(float(angle))
        except Exception:
            return float('nan')

    @staticmethod
    def _normalize_text(text: Any) -> str:
        if text is None:
            return ""
        return str(text).strip().lower()

    def _map_intent_to_three_class(self, raw_intent: Any, raw_action_type: Any) -> str:
        """Compress raw intent labels into the three classes used by the dataset."""
        intent_text = self._normalize_text(raw_intent)
        action_text = self._normalize_text(raw_action_type)
        merged = f"{intent_text} {action_text}".strip()

        attack_keywords = (
            'attack',
            'attack_maneuver',
            'aggressive_approach',
            'accelerate',
            'crank',
            '攻击',
            '进攻',
        )
        escape_keywords = (
            'defense',
            'defensive',
            'defensive_positioning',
            'retreat',
            'escape',
            'evasive',
            'notch',
            'beam',
            'dive_escape',
            'chaff',
            'spiral',
            'short_skate',
            'split',
            'return_to_base',
            'turn_left',
            'turn_right',
            'climb',
            'descend',
            'decelerate',
            '防御',
            '逃逸',
            '撤退',
            '规避',
        )
        recon_keywords = (
            'reconnaissance',
            'recon',
            'search',
            'maintain_heading',
            'maintain heading',
            'level_flight',
            'level flight',
            'neutral_flight',
            'neutral flight',
            'patrol',
            'straight',
            '侦察',
            '平飞',
            '搜索',
            '巡逻',
        )

        if any(keyword in merged for keyword in attack_keywords):
            return 'attack'
        if any(keyword in merged for keyword in escape_keywords):
            return 'escape'
        if any(keyword in merged for keyword in recon_keywords):
            return 'reconnaissance'
        return 'reconnaissance'

    def _resolve_enemy_primary_target_id(self, env, enemy_id: str, tactical_task=None) -> Optional[str]:
        candidate_ids = [
            target_id
            for target_id in ('A0100', 'A0200')
            if target_id in getattr(env, 'agents', {}) and getattr(env.agents[target_id], 'is_alive', False)
        ]
        if not candidate_ids:
            return None

        radar_manager = self._get_radar_manager(env)
        if radar_manager is not None:
            try:
                locked_target = getattr(radar_manager, 'enemy_lock_targets', {}).get(enemy_id)
                if locked_target in candidate_ids:
                    return str(locked_target)
            except Exception:
                pass

            try:
                tracked_targets = getattr(radar_manager, 'enemy_radar_targets', {}).get(enemy_id, {})
                if tracked_targets:
                    for preferred_id in candidate_ids:
                        if preferred_id in tracked_targets:
                            return preferred_id
                    valid_items = [
                        (tid, data)
                        for tid, data in tracked_targets.items()
                        if tid in candidate_ids
                    ]
                    if valid_items:
                        best_tid, _ = min(
                            valid_items,
                            key=lambda item: float(getattr(item[1], 'distance', float('inf')))
                        )
                        return str(best_tid)
            except Exception:
                pass

        unified_ai = getattr(getattr(tactical_task, 'enemy_ai', None), 'unified_ai', None)
        if unified_ai is not None and hasattr(unified_ai, '_get_preferred_target_id'):
            try:
                preferred = unified_ai._get_preferred_target_id(enemy_id)
                if preferred in candidate_ids:
                    return str(preferred)
            except Exception:
                pass

        try:
            enemy_aircraft = getattr(env, '_jsbsims', {}).get(enemy_id) or getattr(env, 'agents', {}).get(enemy_id)
            if enemy_aircraft is None:
                return candidate_ids[0]
            best_target_id = min(
                candidate_ids,
                key=lambda tid: self._safe_distance_m(enemy_aircraft, env.agents[tid])
            )
            return str(best_target_id)
        except Exception:
            return candidate_ids[0]

    def _record_pairwise_intent_frame(self, env, current_time: float, tactical_task=None):
        """Record one 2v2 frame with enemy positions, pairwise geometry, and 3-class intents."""
        if self._pairwise_frame_disabled:
            return

        enemy_ids = ('B0100', 'B0200')
        friendly_ids = ('A0100', 'A0200')
        aircraft_map = {}
        for agent_id in enemy_ids + friendly_ids:
            aircraft_map[agent_id] = (
                getattr(env, '_jsbsims', {}).get(agent_id)
                or getattr(env, 'agents', {}).get(agent_id)
            )

        for agent_id in enemy_ids + friendly_ids:
            aircraft = aircraft_map.get(agent_id)
            if aircraft is None:
                self._pairwise_frame_disabled = True
                self._pairwise_frame_disable_reason = f"missing_{agent_id}"
                return
            if not getattr(aircraft, 'is_alive', False):
                self._pairwise_frame_disabled = True
                self._pairwise_frame_disable_reason = f"dead_{agent_id}"
                return
            pos = self._safe_position(aircraft)
            if np.any(np.isnan(pos)):
                self._pairwise_frame_disabled = True
                self._pairwise_frame_disable_reason = f"nan_position_{agent_id}"
                return
            if float(pos[2]) < 0.0:
                self._pairwise_frame_disabled = True
                self._pairwise_frame_disable_reason = f"negative_altitude_{agent_id}"
                return

        row = {col: np.nan for col in self.frame_level_columns}
        row['t'] = round(float(current_time), 3)

        for idx, enemy_id in enumerate(enemy_ids, start=1):
            aircraft = aircraft_map.get(enemy_id)
            if aircraft is None:
                continue
            pos = self._safe_position(aircraft)
            row[f'x{idx}'] = float(pos[0])
            row[f'y{idx}'] = float(pos[1])
            row[f'z{idx}'] = float(pos[2])

        row['VEL_A0100'] = self._safe_speed_mps(aircraft_map.get('A0100'))
        row['VEL_A0200'] = self._safe_speed_mps(aircraft_map.get('A0200'))
        row['VEL_B0100'] = self._safe_speed_mps(aircraft_map.get('B0100'))
        row['VEL_B0200'] = self._safe_speed_mps(aircraft_map.get('B0200'))

        pair_index = {
            ('B0100', 'A0100'): 1,
            ('B0100', 'A0200'): 2,
            ('B0200', 'A0100'): 3,
            ('B0200', 'A0200'): 4,
        }

        unified_ai = getattr(getattr(tactical_task, 'enemy_ai', None), 'unified_ai', None)
        enemy_intent_map: Dict[str, Tuple[str, Optional[str]]] = {}
        for enemy_id in enemy_ids:
            raw_action_type = ''
            raw_intent = ''
            if unified_ai is not None:
                try:
                    raw_action_type = unified_ai.get_current_action_type(enemy_id)
                except Exception:
                    raw_action_type = ''
                try:
                    raw_intent = unified_ai.get_current_intent(enemy_id)
                except Exception:
                    raw_intent = ''
            mapped_intent = self._map_intent_to_three_class(raw_intent, raw_action_type)
            primary_target_id = self._resolve_enemy_primary_target_id(env, enemy_id, tactical_task)
            enemy_intent_map[enemy_id] = (mapped_intent, primary_target_id)

        for enemy_id in enemy_ids:
            enemy_aircraft = aircraft_map.get(enemy_id)
            enemy_pos = self._safe_position(enemy_aircraft) if enemy_aircraft is not None else np.array([np.nan, np.nan, np.nan], dtype=float)
            mapped_intent, primary_target_id = enemy_intent_map.get(enemy_id, ('reconnaissance', None))

            for friendly_id in friendly_ids:
                key = pair_index[(enemy_id, friendly_id)]
                friendly_aircraft = aircraft_map.get(friendly_id)
                if enemy_aircraft is not None and friendly_aircraft is not None:
                    friendly_pos = self._safe_position(friendly_aircraft)
                    row[f'd{key}'] = self._safe_distance_m(enemy_aircraft, friendly_aircraft)
                    row[f'h{key}'] = float(enemy_pos[2] - friendly_pos[2])
                    row[f'jin{key}'] = self._safe_aspect_angle_deg(enemy_aircraft, friendly_aircraft)
                    row[f'fang{key}'] = self._safe_bearing_angle_deg(enemy_aircraft, friendly_aircraft)
                row[f'yitu{key}'] = mapped_intent if friendly_id == primary_target_id else 'reconnaissance'

        self.frame_level_data.append(row)
    
    def record_aircraft_trajectory(self, env, current_time: float, tactical_task=None):
        """记录飞机轨迹数据 - 包含敌方意图和我方雷达状态"""
        if self.trajectory_mode == "intent_pairwise_frame":
            self._record_pairwise_intent_frame(env, current_time, tactical_task)
            return

        for agent_id, aircraft in env._jsbsims.items():
            if aircraft.is_alive:
                pos = aircraft.get_position()
                heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
                pitch = np.rad2deg(aircraft.get_property_value(c.attitude_pitch_rad))
                roll = np.rad2deg(aircraft.get_property_value(c.attitude_phi_rad))
                velocity_vector = aircraft.get_velocity()
                velocity = np.linalg.norm(velocity_vector)

                # 构建基础轨迹数据
                trajectory_record = {
                    'Time_s': current_time,
                    'Agent_ID': agent_id,
                    'X_m': pos[0],
                    'Y_m': pos[1],
                    'Z_m': pos[2],
                    'Velocity_m_s': velocity,
                    'Heading_deg': heading,
                    'Pitch_deg': pitch,
                    'Roll_deg': roll
                }

                # 🎯 为敌方智能体添加意图识别数据
                if agent_id.startswith('B'):  # 敌方飞机
                    # 🔧 修复：正确的属性路径是 tactical_task.enemy_ai.unified_ai
                    if tactical_task and hasattr(tactical_task, 'enemy_ai') and tactical_task.enemy_ai:
                        if hasattr(tactical_task.enemy_ai, 'unified_ai') and tactical_task.enemy_ai.unified_ai:
                            try:
                                # 获取敌方当前动作类型和意图标签
                                action_type = tactical_task.enemy_ai.unified_ai.get_current_action_type(agent_id)
                                action_intent = tactical_task.enemy_ai.unified_ai.get_current_intent(agent_id)
                                
                                trajectory_record['Action_Type'] = action_type
                                trajectory_record['Action_Intent'] = action_intent
                            except Exception as e:
                                logging.warning(f"获取敌方{agent_id}意图失败: {e}")
                                trajectory_record['Action_Type'] = 'maintain_heading'
                                trajectory_record['Action_Intent'] = 'RECONNAISSANCE'
                        else:
                            trajectory_record['Action_Type'] = 'maintain_heading'
                            trajectory_record['Action_Intent'] = 'RECONNAISSANCE'
                    else:
                        trajectory_record['Action_Type'] = 'maintain_heading'
                        trajectory_record['Action_Intent'] = 'RECONNAISSANCE'
                    
                    # 🎯 添加敌方自己的雷达状态
                    try:
                        # 使用全局雷达管理器
                        radar_manager = None
                        try:
                            import sys
                            import os
                            tactical_project_path = os.path.join(os.path.dirname(__file__), '..', 'tacticalProject')
                            if tactical_project_path not in sys.path:
                                sys.path.insert(0, tactical_project_path)
                            from simulation.radar_manager import get_unified_radar_manager
                            radar_manager = get_unified_radar_manager()
                        except ImportError:
                            if hasattr(env.task, 'radar_manager'):
                                radar_manager = env.task.radar_manager
                        
                        if radar_manager:
                            # 🔧 修复问题1：确定敌方雷达类型 - 正确读取enemy_lowlevel_type
                            enemy_radar_type = 'Unknown'
                            try:
                                # 优先从env.task获取
                                if hasattr(env, 'task') and hasattr(env.task, 'enemy_lowlevel_type'):
                                    enemy_type = env.task.enemy_lowlevel_type
                                    if enemy_type == 'SU27':
                                        enemy_radar_type = 'N001VE'
                                    elif enemy_type == 'F16':
                                        enemy_radar_type = 'APG-68'
                                    else:
                                        # 默认F-16
                                        enemy_radar_type = 'APG-68'
                                else:
                                    # 默认假设F-16
                                    enemy_radar_type = 'APG-68'
                            except Exception as e:
                                logging.debug(f"获取敌方雷达类型失败: {e}")
                                enemy_radar_type = 'APG-68'
                            
                            # 获取敌方雷达状态
                            enemy_radar_status = 'SEARCH'
                            enemy_target_id = ''
                            
                            # 获取该敌方飞机的雷达状态
                            radar_state = radar_manager.get_enemy_radar_state(agent_id)
                            if radar_state:
                                if hasattr(radar_state, 'value'):
                                    enemy_radar_status = radar_state.value
                                elif hasattr(radar_state, 'name'):
                                    enemy_radar_status = radar_state.name
                                else:
                                    radar_str = str(radar_state)
                                    if '.' in radar_str:
                                        enemy_radar_status = radar_str.split('.')[-1]
                                    else:
                                        enemy_radar_status = radar_str
                            
                            # 检查敌方雷达是否在跟踪我方目标
                            if hasattr(radar_manager, 'enemy_radar_targets'):
                                targets = radar_manager.enemy_radar_targets.get(agent_id, {})
                                if targets:
                                    # 获取第一个目标ID
                                    enemy_target_id = list(targets.keys())[0] if isinstance(targets, dict) else ''
                            
                            # 检查敌方是否锁定目标
                            if hasattr(radar_manager, 'enemy_lock_targets'):
                                locked_target = radar_manager.enemy_lock_targets.get(agent_id)
                                if locked_target:
                                    enemy_radar_status = 'LOCK'
                                    enemy_target_id = locked_target
                            
                            trajectory_record['Own_Radar_Type'] = enemy_radar_type
                            trajectory_record['Own_Radar_Status'] = enemy_radar_status
                            trajectory_record['Own_Radar_Target_ID'] = enemy_target_id
                        else:
                            trajectory_record['Own_Radar_Type'] = 'Unknown'
                            trajectory_record['Own_Radar_Status'] = 'SEARCH'
                            trajectory_record['Own_Radar_Target_ID'] = ''
                    except Exception as e:
                        logging.warning(f"获取敌方{agent_id}雷达状态失败: {e}")
                        trajectory_record['Own_Radar_Type'] = 'Unknown'
                        trajectory_record['Own_Radar_Status'] = 'SEARCH'
                        trajectory_record['Own_Radar_Target_ID'] = ''
                else:
                    # 友方飞机 - 添加我方自己的雷达状态
                    trajectory_record['Action_Type'] = ''
                    trajectory_record['Action_Intent'] = ''
                    
                    # 🎯 添加我方自己的雷达状态
                    try:
                        radar_manager = None
                        try:
                            import sys
                            import os
                            tactical_project_path = os.path.join(os.path.dirname(__file__), '..', 'tacticalProject')
                            if tactical_project_path not in sys.path:
                                sys.path.insert(0, tactical_project_path)
                            from simulation.radar_manager import get_unified_radar_manager
                            radar_manager = get_unified_radar_manager()
                        except ImportError:
                            if hasattr(env.task, 'radar_manager'):
                                radar_manager = env.task.radar_manager
                        
                        if radar_manager:
                            # 🔧 修复问题1：确定我方雷达类型 - 正确读取friend_lowlevel_type
                            friendly_radar_type = 'Unknown'
                            try:
                                # 优先从env.task获取
                                if hasattr(env, 'task') and hasattr(env.task, 'friend_lowlevel_type'):
                                    friend_type = env.task.friend_lowlevel_type
                                    if friend_type == 'SU27':
                                        friendly_radar_type = 'N001VE'
                                    elif friend_type == 'F16':
                                        friendly_radar_type = 'APG-68'
                                    else:
                                        # 默认F-16
                                        friendly_radar_type = 'APG-68'
                                else:
                                    # 默认假设F-16
                                    friendly_radar_type = 'APG-68'
                            except Exception as e:
                                logging.debug(f"获取我方雷达类型失败: {e}")
                                friendly_radar_type = 'APG-68'
                            
                            # 获取我方雷达状态
                            friendly_radar_status = 'SEARCH'
                            friendly_target_id = ''
                            
                            radar_state = radar_manager.get_friendly_radar_state(agent_id)
                            if radar_state:
                                if hasattr(radar_state, 'value'):
                                    friendly_radar_status = radar_state.value
                                elif hasattr(radar_state, 'name'):
                                    friendly_radar_status = radar_state.name
                                else:
                                    radar_str = str(radar_state)
                                    if '.' in radar_str:
                                        friendly_radar_status = radar_str.split('.')[-1]
                                    else:
                                        friendly_radar_status = radar_str
                            
                            # 检查我方雷达是否在跟踪敌方目标
                            if hasattr(radar_manager, 'friendly_radar_targets'):
                                targets = radar_manager.friendly_radar_targets.get(agent_id, {})
                                if targets:
                                    friendly_target_id = list(targets.keys())[0] if isinstance(targets, dict) else ''
                            
                            # 检查我方是否锁定目标
                            if hasattr(radar_manager, 'friendly_lock_targets'):
                                locked_target = radar_manager.friendly_lock_targets.get(agent_id)
                                if locked_target:
                                    friendly_radar_status = 'LOCK'
                                    friendly_target_id = locked_target
                            
                            trajectory_record['Own_Radar_Type'] = friendly_radar_type
                            trajectory_record['Own_Radar_Status'] = friendly_radar_status
                            trajectory_record['Own_Radar_Target_ID'] = friendly_target_id
                        else:
                            trajectory_record['Own_Radar_Type'] = 'Unknown'
                            trajectory_record['Own_Radar_Status'] = 'SEARCH'
                            trajectory_record['Own_Radar_Target_ID'] = ''
                    except Exception as e:
                        logging.warning(f"获取我方{agent_id}雷达状态失败: {e}")
                        trajectory_record['Own_Radar_Type'] = 'Unknown'
                        trajectory_record['Own_Radar_Status'] = 'SEARCH'
                        trajectory_record['Own_Radar_Target_ID'] = ''

                self.trajectory_data.append(trajectory_record)

    def record_radar_data(self, env, current_time: float):
        """记录雷达数据 - 统一格式"""
        try:
            # 尝试使用雷达管理器
            radar_manager = None
            if hasattr(env.task, 'radar_manager'):
                radar_manager = env.task.radar_manager
            else:
                # 创建临时雷达管理器
                from radar_manager import RadarManager
                radar_manager = RadarManager()
            
            # ❌ 不要在这里更新雷达状态！任务的step函数已经更新过了
            # 重复更新会导致扫描时机错误和状态不一致
            # radar_manager.update_enemy_radar_states(env, current_time)
            # radar_manager.update_friendly_radar_states(env, current_time)
            
            # 获取雷达数据（直接记录已经更新过的状态）
            radar_records = radar_manager.record_radar_data(env, current_time)
            
            # 标准化雷达数据格式
            for record in radar_records:
                standardized_record = {
                    'Time_s': record.get('Time_s', current_time),
                    'Agent_ID': record.get('Agent_ID', ''),
                    'Radar_Type': record.get('Radar_Type', 'Unknown'),
                    'Status': self.RADAR_STATES.get(record.get('Status', 'SEARCH'), 'SEARCH'),
                    'Target_ID': record.get('Target_ID', ''),
                    'Target_Distance_km': record.get('Target_Distance_km', 0.0),
                    'SNR_dB': record.get('SNR_dB', 0.0),
                    'Doppler_Shift_m_s': record.get('Doppler_Shift_m_s', 0.0),
                    'Lock_Quality': record.get('Lock_Quality', 0.0),
                    'Beam_Angle_deg': record.get('Beam_Angle_deg', 0.0),
                    'Side': record.get('Side', 'Unknown'),
                    'Being_Jammed': record.get('Being_Jammed', False),
                    'Jamming_Sources': record.get('Jamming_Sources', '')
                }
                self.radar_data.append(standardized_record)
                
        except Exception as e:
            logging.warning(f"雷达数据记录失败: {e}")
            # 回退到基本雷达状态记录
            for agent_id, aircraft in env._jsbsims.items():
                if aircraft.is_alive:
                    self.radar_data.append({
                        'Time_s': current_time,
                        'Agent_ID': agent_id,
                        'Radar_Type': 'Unknown',
                        'Status': 'SEARCH',
                        'Target_ID': '',
                        'Target_Distance_km': 0.0,
                        'SNR_dB': 0.0,
                        'Doppler_Shift_m_s': 0.0,
                        'Lock_Quality': 0.0,
                        'Beam_Angle_deg': 0.0,
                        'Side': 'Unknown'
                    })
    
    def record_missile_data(self, env, current_time: float):
        """记录导弹数据 - 统一格式，包括已结束的导弹"""
        # 记录活跃导弹
        if hasattr(env, '_tempsims'):
            for missile_id, missile_sim in env._tempsims.items():
                self._record_single_missile(missile_sim, missile_id, env, current_time)

        # 记录已结束的导弹（击中或未击中）
        if hasattr(env, '_finished_missiles'):
            for missile_id, missile_sim in env._finished_missiles.items():
                # 只记录一次结束状态
                if not hasattr(self, '_recorded_finished_missiles'):
                    self._recorded_finished_missiles = set()

                if missile_id not in self._recorded_finished_missiles:
                    self._record_single_missile(missile_sim, missile_id, env, current_time, is_finished=True)
                    self._recorded_finished_missiles.add(missile_id)

    def _record_single_missile(self, missile_sim, missile_id: str, env, current_time: float, is_finished: bool = False):
        """记录单个导弹的数据"""
        try:
            # 获取导弹位置和速度
            missile_pos = missile_sim.get_position()
            missile_velocity = np.linalg.norm(missile_sim.get_velocity())

            # 获取发射者和目标ID
            launcher_id = self._get_launcher_id(missile_sim, env, missile_id)
            target_id = self._get_target_id(missile_sim, env, missile_id)

            # 获取导弹状态
            missile_status = self._get_missile_status(missile_sim)

            # 如果是已结束的导弹，强制检查最终状态
            if is_finished:
                missile_status = self._get_final_missile_status(missile_sim)

            # 计算到目标距离
            range_to_target_km = self._calculate_range_to_target(missile_sim, env, target_id)

            self.missile_data.append({
                'Time_s': current_time,
                'Missile_ID': missile_id,
                'Launcher_ID': launcher_id,
                'Target_ID': target_id,
                'X_m': missile_pos[0],
                'Y_m': missile_pos[1],
                'Z_m': missile_pos[2],
                'Velocity_m_s': missile_velocity,
                'Status': self.MISSILE_STATES.get(missile_status, 'ACTIVE'),
                'Data_Type': 'Trajectory',
                'Range_to_Target_km': range_to_target_km
            })
        except Exception as e:
            logging.warning(f"记录导弹 {missile_id} 数据失败: {e}")

    def _get_final_missile_status(self, missile_sim) -> str:
        """获取已结束导弹的最终状态 - 更严格的检查"""
        try:
            # 优先检查is_success属性
            if hasattr(missile_sim, 'is_success') and missile_sim.is_success:
                return 'HIT'

            # 检查私有状态属性 - MissileSimulator
            if hasattr(missile_sim, '_MissileSimulator__status'):
                status = missile_sim._MissileSimulator__status
                if hasattr(missile_sim, 'HIT') and status == missile_sim.HIT:
                    return 'HIT'
                elif hasattr(missile_sim, 'MISS') and status == missile_sim.MISS:
                    return 'MISS'

            # 检查R27ER导弹的私有状态
            if hasattr(missile_sim, '_R27ERMissileSimulator__status'):
                status = missile_sim._R27ERMissileSimulator__status
                if hasattr(missile_sim, 'HIT') and status == missile_sim.HIT:
                    return 'HIT'
                elif hasattr(missile_sim, 'MISS') and status == missile_sim.MISS:
                    return 'MISS'

            # 检查数值状态 - 使用动态获取的状态常量
            hit_value = getattr(missile_sim, 'HIT', 1)  # 默认HIT=1
            miss_value = getattr(missile_sim, 'MISS', 2)  # 默认MISS=2

            for attr_name in ['__status', '_status', 'status']:
                if hasattr(missile_sim, attr_name):
                    status = getattr(missile_sim, attr_name)
                    if status == 'HIT' or status == hit_value:  # 使用实际的HIT常量值
                        return 'HIT'
                    elif status == 'MISS' or status == miss_value:  # 使用实际的MISS常量值
                        return 'MISS'

            # 如果导弹不再存活但无法确定状态，默认为MISS
            if hasattr(missile_sim, 'is_alive') and not missile_sim.is_alive:
                return 'MISS'

            return 'UNKNOWN'

        except Exception as e:
            logging.warning(f"获取最终导弹状态失败: {e}")
            return 'MISS'
    
    def _get_launcher_id(self, missile_sim, env, missile_id: str) -> str:
        """获取发射者ID"""
        # 方法1：从导弹属性获取
        if hasattr(missile_sim, 'launcher_id'):
            return missile_sim.launcher_id
        elif hasattr(missile_sim, 'parent_uid'):
            return missile_sim.parent_uid
        
        # 方法2：从parent_aircraft获取
        if hasattr(missile_sim, 'parent_aircraft') and missile_sim.parent_aircraft:
            return missile_sim.parent_aircraft.uid
        
        # 方法3：从环境记录获取
        if hasattr(env, '_missile_records') and missile_id in env._missile_records:
            return env._missile_records[missile_id].get('launcher', 'Unknown')
        
        return 'Unknown'
    
    def _get_target_id(self, missile_sim, env, missile_id: str) -> str:
        """获取目标ID"""
        # 方法1：从导弹属性获取
        if hasattr(missile_sim, 'target_id'):
            return missile_sim.target_id
        elif hasattr(missile_sim, 'target_uid'):
            return missile_sim.target_uid
        
        # 方法2：从target_aircraft获取
        if hasattr(missile_sim, 'target_aircraft') and missile_sim.target_aircraft:
            return missile_sim.target_aircraft.uid
        
        # 方法3：从环境记录获取
        if hasattr(env, '_missile_records') and missile_id in env._missile_records:
            return env._missile_records[missile_id].get('target', 'Unknown')
        
        return 'Unknown'
    
    def _get_missile_status(self, missile_sim) -> str:
        """获取导弹状态 - 改进的击中检测逻辑"""
        try:
            # 优先检查is_success属性（最可靠的方法）
            if hasattr(missile_sim, 'is_success') and missile_sim.is_success:
                return 'HIT'

            # 检查导弹是否存活
            if hasattr(missile_sim, 'is_alive') and not missile_sim.is_alive:
                # 导弹已经销毁，使用状态常量比较（最准确的方法）

                # 方法1：直接比较状态常量 - R27ER导弹
                if hasattr(missile_sim, '_R27ERMissileSimulator__status'):
                    status = missile_sim._R27ERMissileSimulator__status
                    if hasattr(missile_sim, 'HIT') and status == missile_sim.HIT:
                        return 'HIT'
                    elif hasattr(missile_sim, 'MISS') and status == missile_sim.MISS:
                        return 'MISS'

                # 方法2：直接比较状态常量 - AIM-120C导弹
                if hasattr(missile_sim, '_MissileSimulator__status'):
                    status = missile_sim._MissileSimulator__status
                    if hasattr(missile_sim, 'HIT') and status == missile_sim.HIT:
                        return 'HIT'
                    elif hasattr(missile_sim, 'MISS') and status == missile_sim.MISS:
                        return 'MISS'

                # 方法3：数值状态检查（备用方法）
                # 首先尝试获取状态常量进行比较
                hit_value = getattr(missile_sim, 'HIT', 1)  # 默认HIT=1
                miss_value = getattr(missile_sim, 'MISS', 2)  # 默认MISS=2

                for attr_name in ['__status', '_status', 'status']:
                    if hasattr(missile_sim, attr_name):
                        status = getattr(missile_sim, attr_name)
                        if status == 'HIT' or status == hit_value:  # 使用实际的HIT常量值
                            return 'HIT'
                        elif status == 'MISS' or status == miss_value:  # 使用实际的MISS常量值
                            return 'MISS'

                # 如果导弹已销毁但无法确定原因，默认为MISS
                return 'MISS'

            # 方法2：导弹仍在飞行，检查飞行阶段
            if hasattr(missile_sim, '_phase'):
                phase = missile_sim._phase
                if hasattr(missile_sim, 'BOOST_PHASE') and phase == missile_sim.BOOST_PHASE:
                    return 'BOOST'
                elif hasattr(missile_sim, 'MIDCOURSE_PHASE') and phase == missile_sim.MIDCOURSE_PHASE:
                    return 'MIDCOURSE'
                elif hasattr(missile_sim, 'TERMINAL_PHASE') and phase == missile_sim.TERMINAL_PHASE:
                    return 'TERMINAL'

            # 方法3：检查导弹类型特定的阶段
            if hasattr(missile_sim, 'phase'):
                phase_str = str(missile_sim.phase).upper()
                if 'BOOST' in phase_str:
                    return 'BOOST'
                elif 'MIDCOURSE' in phase_str or 'CRUISE' in phase_str:
                    return 'MIDCOURSE'
                elif 'TERMINAL' in phase_str or 'HOMING' in phase_str:
                    return 'TERMINAL'

            # 默认状态
            return 'ACTIVE'

        except Exception as e:
            logging.warning(f"获取导弹状态失败: {e}")
            return 'UNKNOWN'
    
    def _calculate_range_to_target(self, missile_sim, env, target_id: str) -> float:
        """计算导弹到目标的距离（km）"""
        try:
            if target_id != 'Unknown' and target_id in env._jsbsims:
                target_aircraft = env._jsbsims[target_id]
                if target_aircraft.is_alive:
                    missile_pos = missile_sim.get_position()
                    target_pos = target_aircraft.get_position()
                    distance_m = np.linalg.norm(missile_pos - target_pos)
                    return distance_m / 1000.0  # 转换为km
        except Exception as e:
            logging.warning(f"计算导弹到目标距离失败: {e}")
        
        return 0.0
    
    def record_all_data(self, env, current_time: float, tactical_task=None):
        """记录所有类型的数据"""
        self.record_aircraft_trajectory(env, current_time, tactical_task)
        self.record_radar_data(env, current_time)
        self.record_missile_data(env, current_time)
    
    def save_csv_files(self, output_dir: str, timestamp: str = None, simulation_log_content: str = None, 
                       save_only_trajectory: bool = False) -> Dict[str, str]:
        """
        保存所有CSV文件
        
        Args:
            output_dir: 输出目录
            timestamp: 时间戳
            simulation_log_content: 仿真日志内容
            save_only_trajectory: 🔧 修复问题2：如果为True，只保存trajectory文件
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        # 保存飞机轨迹数据（始终保存）
        if self.trajectory_mode == "intent_pairwise_frame" and self.frame_level_data:
            trajectory_df = pd.DataFrame(self.frame_level_data, columns=self.frame_level_columns)
            trajectory_file = os.path.join(output_dir, f"{self.project_name}_trajectory_{timestamp}.csv")
            trajectory_df.to_csv(trajectory_file, index=False, encoding='utf-8-sig')
            saved_files['trajectory'] = trajectory_file
            print(f"Saved trajectory CSV: {trajectory_file}")
        elif self.trajectory_data:
            trajectory_df = pd.DataFrame(self.trajectory_data)
            trajectory_file = os.path.join(output_dir, f"{self.project_name}_trajectory_{timestamp}.csv")
            trajectory_df.to_csv(trajectory_file, index=False, encoding='utf-8-sig')
            saved_files['trajectory'] = trajectory_file
            print(f"飞机轨迹数据已保存: {trajectory_file}")
        
        # 🔧 修复问题2：如果save_only_trajectory=True，跳过其他文件
        if save_only_trajectory:
            return saved_files
        
        # 保存雷达数据
        if self.radar_data:
            radar_df = pd.DataFrame(self.radar_data)
            radar_file = os.path.join(output_dir, f"{self.project_name}_radar_status_{timestamp}.csv")
            radar_df.to_csv(radar_file, index=False, encoding='utf-8-sig')
            saved_files['radar'] = radar_file
            print(f"雷达数据已保存: {radar_file}")
        
        # 保存导弹数据
        if self.missile_data:
            missile_df = pd.DataFrame(self.missile_data)
            missile_file = os.path.join(output_dir, f"{self.project_name}_missile_trajectory_{timestamp}.csv")
            missile_df.to_csv(missile_file, index=False, encoding='utf-8-sig')
            saved_files['missile'] = missile_file
            print(f"导弹轨迹数据已保存: {missile_file}")
            
            # 生成导弹分析数据 - 传入仿真日志内容
            analysis_file = self._generate_missile_analysis(missile_df, output_dir, timestamp, simulation_log_content)
            if analysis_file:
                saved_files['missile_analysis'] = analysis_file
        
        return saved_files

    def _parse_hit_events_from_log(self, log_content: str) -> Dict[str, Dict]:
        """从仿真日志中解析导弹击中事件"""
        hit_events = {}

        if not log_content:
            return hit_events

        try:
            import re
            # 匹配击中事件的正则表达式
            # 例如: "💥 R-27ER B1001 HIT target at t=67.0s, dist=26.7m"
            # 或者: "💥 AIM-120C7 A2002 HIT target at t=88.9s, dist=36.3m"
            hit_pattern = r'💥.*?(\w+\d+)\s+HIT\s+target\s+at\s+t=(\d+\.?\d*)s'

            matches = re.findall(hit_pattern, log_content, re.IGNORECASE)

            for missile_id, time_str in matches:
                hit_time = float(time_str)
                hit_events[missile_id] = {
                    'time': hit_time,
                    'status': 'HIT'
                }
                print(f"🎯 从日志中检测到击中事件: {missile_id} 在 {hit_time}s 击中目标")

        except Exception as e:
            logging.warning(f"解析击中事件失败: {e}")

        return hit_events

    def _generate_missile_analysis(self, missile_df: pd.DataFrame, output_dir: str, timestamp: str,
                                 simulation_log_content: str = None) -> Optional[str]:
        """生成导弹轨迹分析数据 - 基于仿真日志的简化击中检测"""
        try:
            # 解析仿真日志中的击中事件
            hit_events = self._parse_hit_events_from_log(simulation_log_content)

            missile_analysis = []

            for missile_id in missile_df['Missile_ID'].unique():
                missile_data = missile_df[missile_df['Missile_ID'] == missile_id].sort_values('Time_s')

                if len(missile_data) > 0:
                    first_record = missile_data.iloc[0]
                    launch_time = first_record['Time_s']

                    # 计算飞行统计
                    max_velocity = missile_data['Velocity_m_s'].max()
                    avg_velocity = missile_data['Velocity_m_s'].mean()

                    # 计算飞行距离
                    positions = missile_data[['X_m', 'Y_m', 'Z_m']].values
                    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                    total_distance = np.sum(distances) / 1000.0  # km

                    # 确定最终状态和时间 - 优先检查轨迹数据中的HIT状态
                    hit_records = missile_data[missile_data['Status'] == 'HIT']
                    if len(hit_records) > 0:
                        # 轨迹数据中有HIT记录 - 使用轨迹数据
                        final_status = 'HIT'
                        final_time = hit_records['Time_s'].min()  # 使用最早的HIT时间
                        print(f"✅ {missile_id}: 轨迹数据显示击中目标，时间 {final_time}s")
                    else:
                        # 轨迹数据中没有HIT记录，检查仿真日志
                        hit_event = hit_events.get(missile_id)
                        if hit_event:
                            # 仿真日志中有击中事件
                            final_status = 'HIT'
                            final_time = hit_event['time']
                            print(f"✅ {missile_id}: 仿真日志显示击中目标，时间 {final_time}s")
                        else:
                            # 导弹未击中 - 查找最早的MISS状态时间
                            miss_records = missile_data[missile_data['Status'] == 'MISS']
                            if len(miss_records) > 0:
                                final_status = 'MISS'
                                final_time = miss_records['Time_s'].min()
                                print(f"❌ {missile_id}: 未击中目标，MISS时间 {final_time}s")
                            else:
                                # 如果没有MISS记录，使用最后一条记录的时间
                                final_status = 'MISS'
                                final_time = missile_data['Time_s'].max()
                                print(f"⚠️  {missile_id}: 未击中目标，使用最后记录时间 {final_time}s")

                    # 计算飞行时长
                    flight_duration = final_time - launch_time

                    missile_analysis.append({
                        'Missile_ID': missile_id,
                        'Launcher_ID': first_record['Launcher_ID'],
                        'Target_ID': first_record['Target_ID'],
                        'Launch_Time_s': launch_time,
                        'Final_Time_s': final_time,
                        'Flight_Duration_s': flight_duration,
                        'Max_Velocity_m_s': max_velocity,
                        'Avg_Velocity_m_s': avg_velocity,
                        'Total_Distance_km': total_distance,
                        'Final_Status': final_status
                    })
            
            if missile_analysis:
                analysis_df = pd.DataFrame(missile_analysis)
                analysis_file = os.path.join(output_dir, f"{self.project_name}_missile_analysis_{timestamp}.csv")
                analysis_df.to_csv(analysis_file, index=False, encoding='utf-8-sig')
                print(f"导弹分析数据已保存: {analysis_file}")
                
                # 生成摘要报告
                self._generate_missile_summary(analysis_df, output_dir, timestamp)
                
                return analysis_file
                
        except Exception as e:
            logging.error(f"生成导弹分析失败: {e}")
        
        return None
    
    def _generate_missile_summary(self, analysis_df: pd.DataFrame, output_dir: str, timestamp: str):
        """生成导弹摘要报告"""
        try:
            summary_file = os.path.join(output_dir, f"{self.project_name}_missile_summary_{timestamp}.txt")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"{self.project_name.replace('_', ' ').title()}导弹摘要报告\n")
                f.write("=" * 40 + "\n")
                f.write(f"生成时间: {datetime.now()}\n")
                f.write(f"总导弹数量: {len(analysis_df)}\n\n")
                
                if len(analysis_df) > 0:
                    f.write("导弹统计:\n")
                    f.write(f"  平均飞行时间: {analysis_df['Flight_Duration_s'].mean():.2f}秒\n")
                    f.write(f"  平均飞行距离: {analysis_df['Total_Distance_km'].mean():.2f}km\n")
                    f.write(f"  平均速度: {analysis_df['Avg_Velocity_m_s'].mean():.2f}m/s\n")
                    f.write(f"  最大速度: {analysis_df['Max_Velocity_m_s'].max():.2f}m/s\n\n")
                    
                    f.write("各导弹详情:\n")
                    for _, row in analysis_df.iterrows():
                        f.write(f"  {row['Missile_ID']}: {row['Launcher_ID']} -> {row['Target_ID']}, "
                               f"飞行{row['Flight_Duration_s']:.1f}s, 距离{row['Total_Distance_km']:.1f}km\n")
            
            print(f"导弹摘要报告已保存: {summary_file}")
            
        except Exception as e:
            logging.error(f"生成导弹摘要失败: {e}")
    
    def clear_data(self):
        """清空所有数据"""
        self.trajectory_data.clear()
        self.frame_level_data.clear()
        self.radar_data.clear()
        self.missile_data.clear()
