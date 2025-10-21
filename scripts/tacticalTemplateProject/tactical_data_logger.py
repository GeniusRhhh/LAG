#!/usr/bin/env python3
"""
战术态势数据记录器
用于记录2v2空战态势的详细数据
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging


class TacticalDataLogger:
    """战术态势数据记录器 - 记录双方飞机的完整态势信息"""
    
    def __init__(self):
        """初始化数据记录器"""
        self.data_records = []
        self.column_names = self._init_column_names()
        
    def _init_column_names(self) -> List[str]:
        """初始化列名"""
        columns = [
            # 时间信息
            'step',
            'sim_time_s',
            
            # 距离信息 (4个)
            'dist_f1_s1_m',
            'dist_f1_s2_m',
            'dist_f2_s1_m',
            'dist_f2_s2_m',
            
            # 我方高度 (2个)
            'alt_f1_m',
            'alt_f2_m',
            
            # 敌方高度 (2个)
            'alt_s1_m',
            'alt_s2_m',
            
            # 我方速度 (2个)
            'vel_f1_ms',
            'vel_f2_ms',
            
            # 敌方速度 (2个)
            'vel_s1_ms',
            'vel_s2_ms',
            
            # 方位角 (4个)
            'azimuth_f1_s1_deg',
            'azimuth_f1_s2_deg',
            'azimuth_f2_s1_deg',
            'azimuth_f2_s2_deg',
            
            # 进入角 (4个)
            'aspect_f1_s1_deg',
            'aspect_f1_s2_deg',
            'aspect_f2_s1_deg',
            'aspect_f2_s2_deg',
            
            # 受干扰状态 (4个)
            'jammed_f1',
            'jammed_f2',
            'jammed_s1',
            'jammed_s2',
            
            # 敌方机动逻辑 (2个)
            'action_s1',
            'action_s2',
        ]
        return columns
    
    def calculate_azimuth(self, pos_from, pos_to) -> float:
        """
        计算方位角（从pos_from到pos_to的方向）
        
        Args:
            pos_from: 起点位置 [x, y, z]
            pos_to: 目标位置 [x, y, z]
            
        Returns:
            方位角（度），0度为北，顺时针增加
        """
        dx = pos_to[0] - pos_from[0]
        dy = pos_to[1] - pos_from[1]
        azimuth_rad = np.arctan2(dy, dx)
        azimuth_deg = np.rad2deg(azimuth_rad)
        # 转换为导航坐标系（0度为北）
        azimuth_deg = (90 - azimuth_deg) % 360
        return azimuth_deg
    
    def calculate_aspect_angle(self, pos_target, vel_target, pos_shooter) -> float:
        """
        计算进入角（目标航向与射手-目标连线的夹角）
        
        Args:
            pos_target: 目标位置 [x, y, z]
            vel_target: 目标速度向量 [vx, vy, vz]
            pos_shooter: 射手位置 [x, y, z]
            
        Returns:
            进入角（度），0度为正面，180度为尾追
        """
        # 计算射手到目标的向量
        los_vector = np.array(pos_target) - np.array(pos_shooter)
        los_vector_2d = los_vector[:2]  # 只考虑水平面
        
        # 目标速度向量（水平分量）
        vel_vector_2d = np.array(vel_target[:2])
        
        # 计算夹角
        if np.linalg.norm(los_vector_2d) < 1e-6 or np.linalg.norm(vel_vector_2d) < 1e-6:
            return 0.0
        
        # 计算两向量夹角
        cos_angle = np.dot(los_vector_2d, vel_vector_2d) / (
            np.linalg.norm(los_vector_2d) * np.linalg.norm(vel_vector_2d)
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        aspect_angle = np.rad2deg(np.arccos(cos_angle))
        
        return aspect_angle
    
    def calculate_distance(self, pos1, pos2) -> float:
        """计算两点间的3D距离"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def log_step(self, env, step: int, sim_time: float, enemy_ai=None):
        """
        记录一个时间步的态势数据
        
        Args:
            env: 仿真环境
            step: 当前步数
            sim_time: 仿真时间（秒）
            enemy_ai: 敌方AI系统（用于获取action_intent）
        """
        try:
            # 获取飞机代理
            agents = env.agents
            f1 = agents.get('A0100')  # 我方1号机
            f2 = agents.get('A0200')  # 我方2号机
            s1 = agents.get('B0100')  # 敌方1号机
            s2 = agents.get('B0200')  # 敌方2号机
            
            # 初始化记录
            record = {col: None for col in self.column_names}
            record['step'] = step
            record['sim_time_s'] = sim_time
            
            # 获取位置、速度
            positions = {}
            velocities = {}
            altitudes = {}
            speeds = {}
            
            for agent_id, agent in [('f1', f1), ('f2', f2), ('s1', s1), ('s2', s2)]:
                if agent and agent.is_alive:
                    positions[agent_id] = agent.get_position()
                    velocities[agent_id] = agent.get_velocity()
                    altitudes[agent_id] = positions[agent_id][2]
                    speeds[agent_id] = np.linalg.norm(velocities[agent_id])
                else:
                    positions[agent_id] = None
                    velocities[agent_id] = None
                    altitudes[agent_id] = None
                    speeds[agent_id] = None
            
            # 记录高度和速度
            for agent_id in ['f1', 'f2', 's1', 's2']:
                record[f'alt_{agent_id}_m'] = altitudes[agent_id]
                record[f'vel_{agent_id}_ms'] = speeds[agent_id]
            
            # 计算所有配对的距离、方位角、进入角
            pairs = [
                ('f1', 's1'),
                ('f1', 's2'),
                ('f2', 's1'),
                ('f2', 's2'),
            ]
            
            for shooter, target in pairs:
                pair_name = f'{shooter}_{target}'
                
                if positions[shooter] is not None and positions[target] is not None:
                    # 距离
                    distance = self.calculate_distance(positions[shooter], positions[target])
                    record[f'dist_{pair_name}_m'] = distance
                    
                    # 方位角（射手到目标）
                    azimuth = self.calculate_azimuth(positions[shooter], positions[target])
                    record[f'azimuth_{pair_name}_deg'] = azimuth
                    
                    # 进入角（目标相对于射手的角度）
                    if velocities[target] is not None:
                        aspect = self.calculate_aspect_angle(
                            positions[target],
                            velocities[target],
                            positions[shooter]
                        )
                        record[f'aspect_{pair_name}_deg'] = aspect
            
            # 受干扰状态（这里需要根据你的环境实现来获取）
            # 暂时设置为0，你需要根据实际情况修改
            for agent_id in ['f1', 'f2', 's1', 's2']:
                record[f'jammed_{agent_id}'] = 0
            
            # 敌方机动逻辑
            if enemy_ai:
                for enemy_id, col_name in [('B0100', 'action_s1'), ('B0200', 'action_s2')]:
                    try:
                        action_info = enemy_ai.get_action_annotation(enemy_id)
                        record[col_name] = action_info
                    except:
                        record[col_name] = 'unknown'
            else:
                record['action_s1'] = 'unknown'
                record['action_s2'] = 'unknown'
            
            # 添加到记录列表
            self.data_records.append(record)
            
        except Exception as e:
            logging.error(f"态势数据记录失败 (step {step}): {e}")
    
    def save_to_csv(self, filepath: str):
        """
        保存数据到CSV文件
        
        Args:
            filepath: CSV文件路径
        """
        try:
            df = pd.DataFrame(self.data_records, columns=self.column_names)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logging.info(f"✅ 态势数据已保存到: {filepath}")
            logging.info(f"   共 {len(df)} 条记录，{len(df.columns)} 列数据")
            return df
        except Exception as e:
            logging.error(f"保存态势数据失败: {e}")
            return None
    
    def get_dataframe(self) -> pd.DataFrame:
        """获取DataFrame格式的数据"""
        return pd.DataFrame(self.data_records, columns=self.column_names)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        df = self.get_dataframe()
        if df.empty:
            return {}
        
        summary = {
            'total_steps': len(df),
            'sim_time_range': (df['sim_time_s'].min(), df['sim_time_s'].max()),
            'min_distance': df[[col for col in df.columns if col.startswith('dist_')]].min().min(),
            'max_altitude_f': df[['alt_f1_m', 'alt_f2_m']].max().max(),
            'max_altitude_s': df[['alt_s1_m', 'alt_s2_m']].max().max(),
            'avg_speed_f': df[['vel_f1_ms', 'vel_f2_ms']].mean().mean(),
            'avg_speed_s': df[['vel_s1_ms', 'vel_s2_ms']].mean().mean(),
        }
        return summary
    
    def clear(self):
        """清空记录"""
        self.data_records = []


# 使用示例
if __name__ == "__main__":
    """
    使用示例：
    
    # 在仿真环境中使用
    logger = TacticalDataLogger()
    
    # 在仿真循环中
    for step in range(total_steps):
        # ... 仿真一步 ...
        logger.log_step(env, step, sim_time, enemy_ai)
    
    # 保存数据
    logger.save_to_csv('tactical_situation.csv')
    
    # 获取摘要
    summary = logger.get_summary()
    print(summary)
    """
    print("战术态势数据记录器模块")
    print("列数:", len(TacticalDataLogger().column_names))
    print("\n列名列表:")
    for i, col in enumerate(TacticalDataLogger().column_names, 1):
        print(f"{i:2d}. {col}")
