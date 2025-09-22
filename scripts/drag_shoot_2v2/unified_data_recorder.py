#!/usr/bin/env python3
"""
统一数据记录模块
为拖曳射击和钳形夹击项目提供标准化的CSV数据格式
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from envs.JSBSim.core.catalog import Catalog as c
# from tactical_action_extractor import TacticalActionExtractor  # 已禁用动作标注系统


class UnifiedDataRecorder:
    """统一数据记录器 - 标准化CSV格式"""
    
    def __init__(self, project_name: str = "tactical_simulation"):
        """
        初始化统一数据记录器

        Args:
            project_name: 项目名称，用于文件命名前缀
        """
        self.project_name = project_name
        self.trajectory_data = []
        self.radar_data = []
        self.missile_data = []

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
    
    def record_aircraft_trajectory(self, env, current_time: float, tactical_task=None):
        """记录飞机轨迹数据 - 纯净轨迹数据"""
        for agent_id, aircraft in env._jsbsims.items():
            if aircraft.is_alive:
                pos = aircraft.get_position()
                heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
                pitch = np.rad2deg(aircraft.get_property_value(c.attitude_pitch_rad))
                roll = np.rad2deg(aircraft.get_property_value(c.attitude_phi_rad))
                velocity_vector = aircraft.get_velocity()
                velocity = np.linalg.norm(velocity_vector)

                # 基于战术代码的动作提取
                try:
                    # 获取战术指令索引
                    if tactical_task and hasattr(tactical_task, '_get_tactical_command_indices'):
                        altitude_cmd, heading_cmd, velocity_cmd = tactical_task._get_tactical_command_indices(env, agent_id)
                    else:
                        # 默认平稳飞行指令
                        altitude_cmd, heading_cmd, velocity_cmd = 7, 8, 3

                    # 构建当前状态字典
                    current_state = {
                        'time': current_time,
                        'heading': np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad)),
                        'altitude': aircraft.get_property_value(c.position_h_sl_m),
                        'velocity': np.linalg.norm(aircraft.get_velocity()),
                        'latitude': aircraft.get_property_value(c.position_lat_geod_deg),
                        'longitude': aircraft.get_property_value(c.position_long_gc_deg)
                    }

                    # 动作标注系统已禁用 - 不再提取动作信息
                    pass
                except Exception as e:
                    logging.warning(f"数据记录异常 {agent_id}: {e}")

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

                # 为敌方智能体添加行动注释（如果有统一敌方AI系统）
                if (agent_id.startswith('B') and tactical_task and
                    hasattr(tactical_task, 'unified_enemy_ai') and
                    tactical_task.unified_enemy_ai is not None):
                    try:
                        annotation_data = tactical_task.unified_enemy_ai.get_action_annotation_for_csv(agent_id)
                        trajectory_record.update(annotation_data)

                        # 添加具体的战术动作类型
                        action_type_data = tactical_task.unified_enemy_ai.get_action_type_for_csv(agent_id)
                        trajectory_record.update(action_type_data)

                        # 每30秒记录一次注释状态
                        if current_time % 30.0 < 0.2:
                            logging.debug(f"🏷️ {agent_id} 行动注释: {annotation_data.get('Action_Intent', 'unknown')}, 动作类型: {action_type_data.get('action_type', 'N/A')}")
                    except Exception as e:
                        logging.warning(f"行动注释记录失败 {agent_id}: {e}")
                        # 添加默认注释数据
                        trajectory_record.update({
                            'Action_Intent': 'search',
                            'action_type': ''
                        })
                else:
                    # 友方飞机添加空的action_type列
                    trajectory_record.update({
                        'action_type': ''
                    })

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
            
            # 更新雷达状态
            radar_manager.update_enemy_radar_states(env, current_time)
            radar_manager.update_friendly_radar_states(env, current_time)
            
            # 获取雷达数据
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
                    'Side': record.get('Side', 'Unknown')
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
    
    def save_csv_files(self, output_dir: str, timestamp: str = None, simulation_log_content: str = None) -> Dict[str, str]:
        """保存所有CSV文件"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        # 保存飞机轨迹数据
        if self.trajectory_data:
            trajectory_df = pd.DataFrame(self.trajectory_data)
            trajectory_file = os.path.join(output_dir, f"{self.project_name}_trajectory_{timestamp}.csv")
            trajectory_df.to_csv(trajectory_file, index=False, encoding='utf-8-sig')
            saved_files['trajectory'] = trajectory_file
            print(f"飞机轨迹数据已保存: {trajectory_file}")
        
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
        self.radar_data.clear()
        self.missile_data.clear()
