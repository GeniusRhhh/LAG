#!/usr/bin/env python3
"""
战术态势记录器 - 完整态势数据表
记录双方距离、高度、速度、方位角、进入角、干扰状态、机动逻辑

数据表结构（31列）：
- 时间戳: 1列
- 敌方坐标: 6列 (x3, y3, z3, x4, y4, z4) - B0100和B0200的坐标
- 双方距离: 4列 (d1~d4)
- 敌我高度: 4列 (h1~h4)
- 敌我速度: 4列 (v1~v4)
- 目标方位角: 4列 (θ1~θ4)
- 目标进入角: 4列 (ψ1~ψ4)
- 受干扰状态: 2列 (a0100, a0200)
- 敌方机动逻辑: 2列 (b0100, b0200)

采样频率: 0.2秒/帧
"""

import numpy as np
import pandas as pd
import math
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path


class TacticalSituationRecorder:
    """战术态势记录器 - 记录完整战术态势数据"""
    
    def __init__(self):
        """初始化记录器"""
        self.data_records = []
        self.start_time = None
        
        # 定义列名（31列）
        self.columns = [
            # 时间戳（1列）
            'Time_s',
            
            # 敌方坐标（6列）- 单位：m
            'x3_B0100_m', 'y3_B0100_m', 'z3_B0100_m',
            'x4_B0200_m', 'y4_B0200_m', 'z4_B0200_m',
            
            # 双方距离（4列）- 单位：km
            'd1_A0100_B0100_km', 'd2_A0100_B0200_km', 
            'd3_A0200_B0100_km', 'd4_A0200_B0200_km',
            
            # 敌我高度（4列）- 单位：m
            'h1_A0100_m', 'h2_A0200_m', 'h3_B0100_m', 'h4_B0200_m',
            
            # 敌我速度（4列）- 单位：m/s
            'v1_A0100_ms', 'v2_A0200_ms', 'v3_B0100_ms', 'v4_B0200_ms',
            
            # 目标方位角（4列）- 单位：度
            'theta1_A0100_B0100_deg', 'theta2_A0100_B0200_deg',
            'theta3_A0200_B0100_deg', 'theta4_A0200_B0200_deg',
            
            # 目标进入角（4列）- 单位：度
            'psi1_A0100_B0100_deg', 'psi2_A0100_B0200_deg',
            'psi3_A0200_B0100_deg', 'psi4_A0200_B0200_deg',
            
            # 受干扰状态（2列）- 0/1
            'jammed_A0100', 'jammed_A0200',
            
            # 敌方机动逻辑（2列）- 字符串
            'action_B0100', 'action_B0200'
        ]
        
        logging.info(f"📊 战术态势记录器初始化完成，共{len(self.columns)}列")
    
    def record_frame(self, env, current_time: float, 
                    action_intent_b0100: str = "Unknown",
                    action_intent_b0200: str = "Unknown") -> Dict[str, Any]:
        """
        记录单帧数据（0.2秒采样）
        
        Args:
            env: 环境对象
            current_time: 当前时间（秒）
            action_intent_b0100: B0100的机动意图
            action_intent_b0200: B0200的机动意图
            
        Returns:
            Dict: 当前帧数据
        """
        try:
            # 初始化数据行
            frame_data = {col: 0.0 for col in self.columns}
            frame_data['Time_s'] = current_time
            
            # 检查所有飞机是否存在
            agents = {}
            for agent_id in ['A0100', 'A0200', 'B0100', 'B0200']:
                if hasattr(env, 'agents') and agent_id in env.agents and env.agents[agent_id].is_alive:
                    agents[agent_id] = env.agents[agent_id]
                else:
                    agents[agent_id] = None
            
            # === 1. 记录敌方坐标（6列）===
            if agents['B0100']:
                pos_b0100 = agents['B0100'].get_position()
                frame_data['x3_B0100_m'] = pos_b0100[0]
                frame_data['y3_B0100_m'] = pos_b0100[1]
                frame_data['z3_B0100_m'] = pos_b0100[2]
            
            if agents['B0200']:
                pos_b0200 = agents['B0200'].get_position()
                frame_data['x4_B0200_m'] = pos_b0200[0]
                frame_data['y4_B0200_m'] = pos_b0200[1]
                frame_data['z4_B0200_m'] = pos_b0200[2]
            
            # === 2. 记录双方距离（4列）===
            if agents['A0100'] and agents['B0100']:
                frame_data['d1_A0100_B0100_km'] = self._calculate_distance(
                    agents['A0100'], agents['B0100']) / 1000.0
            
            if agents['A0100'] and agents['B0200']:
                frame_data['d2_A0100_B0200_km'] = self._calculate_distance(
                    agents['A0100'], agents['B0200']) / 1000.0
            
            if agents['A0200'] and agents['B0100']:
                frame_data['d3_A0200_B0100_km'] = self._calculate_distance(
                    agents['A0200'], agents['B0100']) / 1000.0
            
            if agents['A0200'] and agents['B0200']:
                frame_data['d4_A0200_B0200_km'] = self._calculate_distance(
                    agents['A0200'], agents['B0200']) / 1000.0
            
            # === 3. 记录敌我高度（4列）===
            for agent_id, key in [
                ('A0100', 'h1_A0100_m'), ('A0200', 'h2_A0200_m'),
                ('B0100', 'h3_B0100_m'), ('B0200', 'h4_B0200_m')
            ]:
                if agents[agent_id]:
                    frame_data[key] = agents[agent_id].get_position()[2]
            
            # === 4. 记录敌我速度（4列）===
            for agent_id, key in [
                ('A0100', 'v1_A0100_ms'), ('A0200', 'v2_A0200_ms'),
                ('B0100', 'v3_B0100_ms'), ('B0200', 'v4_B0200_ms')
            ]:
                if agents[agent_id]:
                    velocity = np.linalg.norm(agents[agent_id].get_velocity())
                    frame_data[key] = velocity
            
            # === 5. 记录目标方位角（4列）===
            if agents['A0100'] and agents['B0100']:
                frame_data['theta1_A0100_B0100_deg'] = self._calculate_bearing(
                    agents['A0100'], agents['B0100'])
            
            if agents['A0100'] and agents['B0200']:
                frame_data['theta2_A0100_B0200_deg'] = self._calculate_bearing(
                    agents['A0100'], agents['B0200'])
            
            if agents['A0200'] and agents['B0100']:
                frame_data['theta3_A0200_B0100_deg'] = self._calculate_bearing(
                    agents['A0200'], agents['B0100'])
            
            if agents['A0200'] and agents['B0200']:
                frame_data['theta4_A0200_B0200_deg'] = self._calculate_bearing(
                    agents['A0200'], agents['B0200'])
            
            # === 6. 记录目标进入角（4列）===
            if agents['A0100'] and agents['B0100']:
                frame_data['psi1_A0100_B0100_deg'] = self._calculate_aspect_angle(
                    agents['A0100'], agents['B0100'])
            
            if agents['A0100'] and agents['B0200']:
                frame_data['psi2_A0100_B0200_deg'] = self._calculate_aspect_angle(
                    agents['A0100'], agents['B0200'])
            
            if agents['A0200'] and agents['B0100']:
                frame_data['psi3_A0200_B0100_deg'] = self._calculate_aspect_angle(
                    agents['A0200'], agents['B0100'])
            
            if agents['A0200'] and agents['B0200']:
                frame_data['psi4_A0200_B0200_deg'] = self._calculate_aspect_angle(
                    agents['A0200'], agents['B0200'])
            
            # === 7. 记录受干扰状态（2列）===
            # 需要从雷达管理器获取
            try:
                from radar_manager import is_being_jammed
                frame_data['jammed_A0100'] = 1 if is_being_jammed('A0100') else 0
                frame_data['jammed_A0200'] = 1 if is_being_jammed('A0200') else 0
            except Exception as e:
                logging.warning(f"⚠️ 无法获取干扰状态: {e}")
                frame_data['jammed_A0100'] = 0
                frame_data['jammed_A0200'] = 0
            
            # === 8. 记录敌方机动逻辑（2列）===
            frame_data['action_B0100'] = action_intent_b0100
            frame_data['action_B0200'] = action_intent_b0200
            
            # 保存记录
            if not isinstance(self.data_records, list):
                logging.error(f"❌ data_records类型错误: {type(self.data_records)}，重新初始化为list")
                self.data_records = []
            self.data_records.append(frame_data)
            
            return frame_data
            
        except Exception as e:
            logging.error(f"❌ 记录帧数据错误: {e}")
            return {}
    
    def _calculate_distance(self, agent1, agent2) -> float:
        """计算两个智能体之间的距离（米）"""
        try:
            pos1 = np.array(agent1.get_position())
            pos2 = np.array(agent2.get_position())
            distance = np.linalg.norm(pos1 - pos2)
            return distance
        except Exception as e:
            logging.error(f"❌ 距离计算错误: {e}")
            return 0.0
    
    def _calculate_bearing(self, observer, target) -> float:
        """
        计算方位角（度）
        从观察者指向目标的方位角（相对于观察者机头方向）
        0° = 正前方，90° = 右侧，-90° = 左侧，±180° = 正后方
        """
        try:
            observer_pos = np.array(observer.get_position())
            target_pos = np.array(target.get_position())
            
            # 计算从观察者到目标的向量（水平面投影）
            to_target = target_pos[:2] - observer_pos[:2]
            
            # 获取观察者的航向角（yaw）
            try:
                observer_heading = observer.get_rpy()[2]  # 弧度
            except:
                # 如果无法获取航向，使用速度方向
                vel = observer.get_velocity()[:2]
                if np.linalg.norm(vel) > 1.0:
                    observer_heading = math.atan2(vel[1], vel[0])
                else:
                    observer_heading = 0.0
            
            # 计算目标的绝对方位角
            target_absolute_bearing = math.atan2(to_target[1], to_target[0])
            
            # 计算相对方位角
            relative_bearing = target_absolute_bearing - observer_heading
            
            # 归一化到[-π, π]
            relative_bearing = math.atan2(math.sin(relative_bearing), 
                                         math.cos(relative_bearing))
            
            # 转换为度
            bearing_deg = math.degrees(relative_bearing)
            
            return bearing_deg
            
        except Exception as e:
            logging.error(f"❌ 方位角计算错误: {e}")
            return 0.0
    
    def _calculate_aspect_angle(self, observer, target) -> float:
        """
        计算目标进入角/视角角度（度）
        相对于目标机头方向的角度
        0° = 目标正面，90° = 目标侧面，180° = 目标尾部
        """
        try:
            observer_pos = np.array(observer.get_position())
            target_pos = np.array(target.get_position())
            
            # 从目标指向观察者的向量（水平面）
            to_observer = observer_pos[:2] - target_pos[:2]
            to_observer_norm = to_observer / (np.linalg.norm(to_observer) + 1e-6)
            
            # 获取目标的航向
            try:
                target_heading = target.get_rpy()[2]  # 弧度
                target_heading_vec = np.array([
                    math.cos(target_heading),
                    math.sin(target_heading)
                ])
            except:
                # 使用速度方向
                target_vel = target.get_velocity()[:2]
                if np.linalg.norm(target_vel) > 1.0:
                    target_heading_vec = target_vel / np.linalg.norm(target_vel)
                else:
                    return 90.0  # 默认侧面
            
            # 计算夹角
            cos_angle = np.dot(target_heading_vec, to_observer_norm)
            aspect_angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
            
            return aspect_angle
            
        except Exception as e:
            logging.error(f"❌ 进入角计算错误: {e}")
            return 0.0
    
    def save_to_csv(self, filepath: str = "tactical_situation.csv"):
        """
        保存数据到CSV文件
        
        Args:
            filepath: 保存路径
        """
        try:
            # 类型安全检查
            if not isinstance(self.data_records, list):
                logging.error(f"❌ data_records类型错误: {type(self.data_records)}，无法保存")
                return
            
            if not self.data_records:
                logging.warning("⚠️ 没有数据可保存")
                return
            
            logging.info(f"📊 准备保存战术态势数据: {len(self.data_records)}帧")
            
            # 创建DataFrame
            df = pd.DataFrame(self.data_records, columns=self.columns)
            
            # 保存到CSV
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, float_format='%.3f')
            
            logging.info(f"✅ 战术态势数据已保存: {output_path}")
            logging.info(f"📊 共记录 {len(df)} 帧数据，{len(self.columns)} 列")
            
            # 打印统计信息
            self._print_statistics(df)
            
        except Exception as e:
            logging.error(f"❌ 保存CSV错误: {e}")
            logging.error(f"   data_records类型: {type(self.data_records)}")
            if hasattr(self, 'data_records'):
                logging.error(f"   data_records长度: {len(self.data_records) if isinstance(self.data_records, (list, dict)) else 'N/A'}")
            import traceback
            logging.error(traceback.format_exc())
    
    def _print_statistics(self, df: pd.DataFrame):
        """打印数据统计信息"""
        try:
            logging.info("=" * 60)
            logging.info("📈 数据统计摘要")
            logging.info("=" * 60)
            
            # 时间范围
            logging.info(f"⏱️  时间范围: {df['Time_s'].min():.1f}s - {df['Time_s'].max():.1f}s")
            logging.info(f"📏 总时长: {df['Time_s'].max() - df['Time_s'].min():.1f}s")
            logging.info(f"🎞️  总帧数: {len(df)} 帧")
            
            # 距离统计
            logging.info("\n📏 距离统计（km）:")
            for col in ['d1_A0100_B0100_km', 'd2_A0100_B0200_km', 
                       'd3_A0200_B0100_km', 'd4_A0200_B0200_km']:
                if col in df.columns:
                    logging.info(f"  {col}: 最小={df[col].min():.2f}, "
                               f"最大={df[col].max():.2f}, "
                               f"平均={df[col].mean():.2f}")
            
            # 高度统计
            logging.info("\n🛫 高度统计（m）:")
            for col in ['h1_A0100_m', 'h2_A0200_m', 'h3_B0100_m', 'h4_B0200_m']:
                if col in df.columns:
                    logging.info(f"  {col}: 最小={df[col].min():.0f}, "
                               f"最大={df[col].max():.0f}, "
                               f"平均={df[col].mean():.0f}")
            
            # 速度统计
            logging.info("\n⚡ 速度统计（m/s）:")
            for col in ['v1_A0100_ms', 'v2_A0200_ms', 'v3_B0100_ms', 'v4_B0200_ms']:
                if col in df.columns:
                    logging.info(f"  {col}: 最小={df[col].min():.1f}, "
                               f"最大={df[col].max():.1f}, "
                               f"平均={df[col].mean():.1f}")
            
            # 干扰统计
            logging.info("\n🛡️ 干扰状态统计:")
            for col in ['jammed_A0100', 'jammed_A0200']:
                if col in df.columns:
                    jammed_count = df[col].sum()
                    jammed_pct = (jammed_count / len(df)) * 100
                    logging.info(f"  {col}: {jammed_count}帧 ({jammed_pct:.1f}%)")
            
            # 机动逻辑统计
            logging.info("\n🎯 敌方机动逻辑分布:")
            for col in ['action_B0100', 'action_B0200']:
                if col in df.columns:
                    action_counts = df[col].value_counts()
                    logging.info(f"  {col}:")
                    for action, count in action_counts.items():
                        pct = (count / len(df)) * 100
                        logging.info(f"    {action}: {count}帧 ({pct:.1f}%)")
            
            logging.info("=" * 60)
            
        except Exception as e:
            logging.error(f"❌ 统计信息打印错误: {e}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        获取DataFrame对象
        
        Returns:
            pd.DataFrame: 数据表
        """
        if not self.data_records:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.data_records, columns=self.columns)
    
    def clear(self):
        """清空记录"""
        self.data_records = []
        logging.info("🗑️ 战术态势记录已清空")


# ==================== 全局实例管理 ====================

_global_recorder = None

def get_tactical_recorder() -> TacticalSituationRecorder:
    """获取全局战术态势记录器实例"""
    global _global_recorder
    if _global_recorder is None:
        _global_recorder = TacticalSituationRecorder()
    return _global_recorder


def reset_tactical_recorder():
    """重置全局记录器"""
    global _global_recorder
    _global_recorder = None
    logging.info("🔄 全局战术态势记录器已重置")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("战术态势记录器 - 使用示例")
    print("=" * 60)
    
    # 创建记录器
    recorder = TacticalSituationRecorder()
    
    # 打印列名
    print(f"\n📊 数据表结构（共{len(recorder.columns)}列）：")
    for i, col in enumerate(recorder.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\n✅ 记录器初始化完成")
    print("\n💡 集成步骤：")
    print("  1. 在仿真循环中调用 recorder.record_frame(env, time, action_b0100, action_b0200)")
    print("  2. 仿真结束后调用 recorder.save_to_csv('output.csv')")
    print("  3. 使用 pandas 分析数据: df = recorder.get_dataframe()")
