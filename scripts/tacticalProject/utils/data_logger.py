"""
完整数据记录系统 - 生成7个数据表
用于记录仿真过程中的所有关键数据
"""

import csv
import os
import logging
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
try:
    from envs.JSBSim.core.catalog import Catalog as c
except Exception:
    c = None


class DataLogger:
    """数据记录器 - 生成所有数据表"""
    
    def __init__(self, output_dir: str, timestamp: str = None):
        """
        初始化数据记录器
        
        Args:
            output_dir: 输出目录
            timestamp: 时间戳（用于文件名）
        """
        self.output_dir = output_dir
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据缓存
        self.trajectory_log = []
        self.radar_log = []
        self.missile_log = []
        self.threat_log = []
        self.decision_log = []
        self.phase_log = []
        
        logging.info(f"[数据记录器] 初始化完成: {output_dir}")
    
    def record_step(self, env, current_time: float):
        """按步记录核心数据：轨迹、雷达、导弹。
        该方法被 TacticalTask 每10步调用一次。
        """
        try:
            # 1) 轨迹
            for agent_id, agent in env.agents.items():
                if not agent.is_alive:
                    continue
                pos = agent.get_position()
                vel = agent.get_velocity()
                if c is not None:
                    try:
                        heading = float(np.rad2deg(agent.get_property_value(c.attitude_psi_rad)))
                        pitch = float(np.rad2deg(agent.get_property_value(getattr(c, 'attitude_pitch_rad', 'attitude/pitch-rad'))))
                        roll = float(np.rad2deg(agent.get_property_value(getattr(c, 'attitude_phi_rad', 'attitude/phi-rad'))))
                    except Exception:
                        heading, pitch, roll = 0.0, 0.0, 0.0
                else:
                    heading, pitch, roll = 0.0, 0.0, 0.0
                self.log_trajectory(current_time, agent_id, pos, vel, heading, pitch, roll)

            # 2) 雷达
            radar = getattr(getattr(env, 'task', None), 'radar_manager', None)
            if radar is not None:
                # 友方
                for aid in radar.friendly_radar_states.keys():
                    mode = radar.friendly_radar_states.get(aid)
                    targets = radar.friendly_radar_targets.get(aid, {})
                    target_id, distance, lock_q, snr = 'None', 0.0, 0.0, 0.0
                    if targets:
                        # 选择最近目标
                        try:
                            tid, t = min(targets.items(), key=lambda x: x[1].distance)
                            target_id = tid
                            distance = float(t.distance)
                            lock_q = float(getattr(t, 'track_quality', 0.0))
                            snr = float(getattr(t, 'snr', 0.0))
                        except Exception:
                            pass
                    self.log_radar_status(current_time, aid, str(getattr(mode, 'name', mode)), target_id, distance, lock_q, snr)
                # 敌方
                for bid in radar.enemy_radar_states.keys():
                    mode = radar.enemy_radar_states.get(bid)
                    targets = radar.enemy_radar_targets.get(bid, {})
                    target_id, distance, lock_q, snr = 'None', 0.0, 0.0, 0.0
                    if targets:
                        try:
                            tid, t = min(targets.items(), key=lambda x: x[1].distance)
                            target_id = tid
                            distance = float(t.distance)
                            lock_q = float(getattr(t, 'track_quality', 0.0))
                            snr = float(getattr(t, 'snr', 0.0))
                        except Exception:
                            pass
                    self.log_radar_status(current_time, bid, str(getattr(mode, 'name', mode)), target_id, distance, lock_q, snr)

            # 3) 导弹
            missiles = getattr(env, 'missiles', {}) or {}
            for mid, m in missiles.items():
                try:
                    pos = m.get_position() if hasattr(m, 'get_position') else getattr(m, 'position', (0.0, 0.0, 0.0))
                    if hasattr(m, 'get_velocity'):
                        vel = m.get_velocity()
                        speed = float(np.linalg.norm(vel))
                    else:
                        speed = float(getattr(m, 'speed', 0.0))
                    launcher_id = getattr(getattr(m, 'parent', None), 'uid', 'UNKNOWN')
                    target_obj = getattr(m, 'target', None)
                    target_id = getattr(target_obj, 'uid', 'UNKNOWN')
                    # 距离
                    if target_obj is not None and hasattr(target_obj, 'get_position'):
                        tpos = target_obj.get_position()
                        distance_to_target = float(np.linalg.norm(np.array(tpos) - np.array(pos)))
                    else:
                        distance_to_target = 0.0
                    # 制导模式（尽力获取）
                    guidance_mode = 'UNKNOWN'
                    try:
                        if hasattr(m, '_phase') and hasattr(m, 'MIDCOURSE_PHASE') and hasattr(m, 'TERMINAL_PHASE'):
                            if m._phase == getattr(m, 'MIDCOURSE_PHASE'):
                                guidance_mode = 'MID_COURSE'
                            elif m._phase == getattr(m, 'TERMINAL_PHASE'):
                                guidance_mode = 'TERMINAL'
                            else:
                                guidance_mode = 'BOOST'
                        elif hasattr(m, 'guidance_mode'):
                            guidance_mode = str(getattr(m, 'guidance_mode'))
                    except Exception:
                        guidance_mode = 'UNKNOWN'
                    self.log_missile_status(current_time, mid, launcher_id, target_id, pos, speed, distance_to_target, guidance_mode)
                except Exception:
                    continue
        except Exception as e:
            logging.debug(f"数据记录失败: {e}")
    
    def log_trajectory(self, time: float, agent_id: str, position: tuple, velocity: tuple, 
                      heading: float, pitch: float, roll: float):
        """
        记录轨迹数据
        
        Args:
            time: 时间（秒）
            agent_id: 飞机ID
            position: 位置 (x, y, z)
            velocity: 速度 (vx, vy, vz)
            heading: 航向（度）
            pitch: 俯仰（度）
            roll: 滚转（度）
        """
        self.trajectory_log.append({
            'time': time,
            'agent_id': agent_id,
            'x': position[0],
            'y': position[1],
            'z': position[2],
            'vx': velocity[0],
            'vy': velocity[1],
            'vz': velocity[2],
            'heading': heading,
            'pitch': pitch,
            'roll': roll
        })
    
    def log_radar_status(self, time: float, agent_id: str, mode: str, target_id: str = None,
                        distance: float = 0, lock_quality: float = 0, snr: float = 0):
        """
        记录雷达状态
        
        Args:
            time: 时间（秒）
            agent_id: 飞机ID
            mode: 雷达模式（SEARCH/TRACK/LOCK）
            target_id: 目标ID
            distance: 距离（米）
            lock_quality: 锁定质量（0-1）
            snr: 信噪比（dB）
        """
        self.radar_log.append({
            'time': time,
            'agent_id': agent_id,
            'mode': mode,
            'target_id': target_id or 'None',
            'distance': distance,
            'lock_quality': lock_quality,
            'snr': snr
        })
    
    def log_missile_status(self, time: float, missile_id: str, launcher_id: str, target_id: str,
                          position: tuple, speed: float, distance_to_target: float, 
                          guidance_mode: str):
        """
        记录导弹状态
        
        Args:
            time: 时间（秒）
            missile_id: 导弹ID
            launcher_id: 发射者ID
            target_id: 目标ID
            position: 位置 (x, y, z)
            speed: 速度（m/s）
            distance_to_target: 到目标距离（米）
            guidance_mode: 制导模式（MID_COURSE/TERMINAL）
        """
        self.missile_log.append({
            'time': time,
            'missile_id': missile_id,
            'launcher_id': launcher_id,
            'target_id': target_id,
            'x': position[0],
            'y': position[1],
            'z': position[2],
            'speed': speed,
            'distance_to_target': distance_to_target,
            'guidance_mode': guidance_mode
        })
    
    def log_threat_assessment(self, time: float, agent_id: str, threat_level: float,
                             distance_threat: float, angle_threat: float, 
                             altitude_threat: float, speed_threat: float, total_threat: float):
        """
        记录威胁评估
        
        Args:
            time: 时间（秒）
            agent_id: 飞机ID
            threat_level: 威胁等级
            distance_threat: 距离威胁值
            angle_threat: 角度威胁值
            altitude_threat: 高度威胁值
            speed_threat: 速度威胁值
            total_threat: 总威胁值
        """
        self.threat_log.append({
            'time': time,
            'agent_id': agent_id,
            'threat_level': threat_level,
            'distance_threat': distance_threat,
            'angle_threat': angle_threat,
            'altitude_threat': altitude_threat,
            'speed_threat': speed_threat,
            'total_threat': total_threat
        })
    
    def log_decision(self, time: float, agent_id: str, phase: str, decision_type: str,
                    selected_tactic: str = None, selected_maneuver: str = None, reason: str = None):
        """
        记录决策日志
        
        Args:
            time: 时间（秒）
            agent_id: 飞机ID
            phase: 战术阶段
            decision_type: 决策类型（TACTIC/MANEUVER/PARAMETER）
            selected_tactic: 选定战术
            selected_maneuver: 选定机动
            reason: 决策原因
        """
        self.decision_log.append({
            'time': time,
            'agent_id': agent_id,
            'phase': phase,
            'decision_type': decision_type,
            'selected_tactic': selected_tactic or 'None',
            'selected_maneuver': selected_maneuver or 'None',
            'reason': reason or 'None'
        })
    
    def log_tactical_phase(self, time: float, agent_id: str, phase: str, distance: float,
                          my_intent: str, enemy_intent: str, situation: str):
        """
        记录战术阶段
        
        Args:
            time: 时间（秒）
            agent_id: 飞机ID
            phase: 战术阶段
            distance: 距离（米）
            my_intent: 我方意图
            enemy_intent: 敌方意图
            situation: 态势（ADVANTAGE/NEUTRAL/DISADVANTAGE）
        """
        self.phase_log.append({
            'time': time,
            'agent_id': agent_id,
            'phase': phase,
            'distance': distance,
            'my_intent': my_intent,
            'enemy_intent': enemy_intent,
            'situation': situation
        })
    
    def save_all(self):
        """保存所有数据表"""
        try:
            saved_tables = []
            
            # 1. 保存轨迹数据
            if self.trajectory_log:
                self._save_trajectory(silent=True)
                saved_tables.append(f"轨迹({len(self.trajectory_log)}条)")
            
            # 2. 保存雷达状态
            if self.radar_log:
                self._save_radar_status(silent=True)
                saved_tables.append(f"雷达({len(self.radar_log)}条)")
            
            # 3. 保存导弹状态
            if self.missile_log:
                self._save_missile_status(silent=True)
                saved_tables.append(f"导弹({len(self.missile_log)}条)")
            
            # 4. 保存威胁评估
            if self.threat_log:
                self._save_threat_assessment(silent=True)
                saved_tables.append(f"威胁({len(self.threat_log)}条)")
            
            # 5. 保存决策日志
            if self.decision_log:
                self._save_decision_log(silent=True)
                saved_tables.append(f"决策({len(self.decision_log)}条)")
            
            # 6. 保存战术阶段
            if self.phase_log:
                self._save_tactical_phase(silent=True)
                saved_tables.append(f"阶段({len(self.phase_log)}条)")
            
            # 只打印一条总结
            # logging.info(f"✅ 数据表保存完成: {', '.join(saved_tables)} -> {self.output_dir}")
            
        except Exception as e:
            logging.error(f"❌ 保存数据表失败: {e}")
    
    def _save_trajectory(self, silent=False):
        """保存轨迹数据表"""
        filename = os.path.join(self.output_dir, f"trajectory_{self.timestamp}.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['time', 'agent_id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'heading', 'pitch', 'roll']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trajectory_log)
        
        if not silent:
            logging.info(f"✅ 轨迹数据表: {filename} ({len(self.trajectory_log)}条记录)")
    
    def _save_radar_status(self, silent=False):
        """保存雷达状态表"""
        filename = os.path.join(self.output_dir, f"radar_status_{self.timestamp}.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['time', 'agent_id', 'mode', 'target_id', 'distance', 'lock_quality', 'snr']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.radar_log)
        
        if not silent:
            logging.info(f"✅ 雷达状态表: {filename} ({len(self.radar_log)}条记录)")
    
    def _save_missile_status(self, silent=False):
        """保存导弹状态表"""
        filename = os.path.join(self.output_dir, f"missile_status_{self.timestamp}.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['time', 'missile_id', 'launcher_id', 'target_id', 'x', 'y', 'z', 
                         'speed', 'distance_to_target', 'guidance_mode']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.missile_log)
        
        if not silent:
            logging.info(f"✅ 导弹状态表: {filename} ({len(self.missile_log)}条记录)")
    
    def _save_threat_assessment(self, silent=False):
        """保存威胁评估表"""
        filename = os.path.join(self.output_dir, f"threat_assessment_{self.timestamp}.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['time', 'agent_id', 'threat_level', 'distance_threat', 'angle_threat',
                         'altitude_threat', 'speed_threat', 'total_threat']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.threat_log)
        
        if not silent:
            logging.info(f"✅ 威胁评估表: {filename} ({len(self.threat_log)}条记录)")
    
    def _save_decision_log(self, silent=False):
        """保存决策日志表"""
        filename = os.path.join(self.output_dir, f"decision_log_{self.timestamp}.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['time', 'agent_id', 'phase', 'decision_type', 'selected_tactic',
                         'selected_maneuver', 'reason']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.decision_log)
        
        if not silent:
            logging.info(f"✅ 决策日志表: {filename} ({len(self.decision_log)}条记录)")
    
    def _save_tactical_phase(self, silent=False):
        """保存战术阶段表"""
        filename = os.path.join(self.output_dir, f"tactical_phase_{self.timestamp}.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['time', 'agent_id', 'phase', 'distance', 'my_intent', 'enemy_intent', 'situation']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.phase_log)
        
        if not silent:
            logging.info(f"✅ 战术阶段表: {filename} ({len(self.phase_log)}条记录)")
