#!/usr/bin/env python3
"""
重构拖曳射击任务脚本 - 将现有任务适配到新的架构
"""

import os
import sys
import logging
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)


def create_refactored_drag_shoot_task():
    """创建重构后的拖曳射击任务"""
    
    refactored_code = '''#!/usr/bin/env python3
"""
拖曳射击战术任务 - 重构版本，继承BaseTacticalTask
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
import math

from base_tactical_task import BaseTacticalTask, TacticalType, TacticalPhase
from envs.JSBSim.core.catalog import Catalog as c


class DragShootTacticalTask(BaseTacticalTask):
    """
    拖曳射击战术任务 - 重构版本
    
    战术流程：
    1. NTL-MELD阶段: 双机保持编队，朝向敌机
    2. MELD-MTR阶段: 双机继续接近，准备发射
    3. MTR-TR阶段: 长机发射导弹，僚机滞后
    4. TR-DOR阶段: 长机执行Short Skate，僚机继续滞后
    5. DOR-DR阶段: 僚机发射导弹并执行Short Skate
    """
    
    def __init__(self, config):
        """初始化拖曳射击战术任务"""
        super().__init__(config)
        
        # 拖曳射击特有配置
        self.drag_shoot_config = {
            'wingman_delay': {
                'TR_DOR_delay': 4000,    # TR-DOR阶段僚机延迟4km
                'DOR_DR_delay': 8000,    # DOR-DR阶段僚机延迟8km
            },
            'formation_spacing': 5000,    # 编队间距 (5km)
            'leader_priority': True,      # 长机优先发射
        }
        
        # 更新配置
        if 'wingman_delay' in config:
            self.drag_shoot_config['wingman_delay'].update(config['wingman_delay'])
        
        # 编队角色定义
        self.formation_roles = {
            'A0100': 'leader',    # 长机
            'A0200': 'wingman',   # 僚机
        }
        
        # 拖曳射击状态跟踪
        self.drag_shoot_states = {
            'leader_launched': False,
            'wingman_launched': False,
            'leader_skate_started': False,
            'wingman_skate_started': False,
        }
        
        logging.info("🎯 拖曳射击战术任务初始化完成")
    
    def get_tactical_type(self) -> TacticalType:
        """获取战术类型"""
        return TacticalType.DRAG_SHOOT
    
    def _get_tactical_command_indices(self, env, agent_id: str) -> Tuple[int, int, int]:
        """获取拖曳射击战术指令索引"""
        # 根据当前阶段和角色确定战术行为
        if agent_id.startswith('A'):  # 友方
            return self._get_friendly_command_indices(env, agent_id)
        else:  # 敌方
            return self._get_enemy_command_indices(env, agent_id)
    
    def _get_friendly_command_indices(self, env, agent_id: str) -> Tuple[int, int, int]:
        """友方拖曳射击指令"""
        role = self.formation_roles.get(agent_id, 'unknown')
        
        if self.current_phase == TacticalPhase.NLT_MELD:
            # 阶段1: 双机保持编队，朝向敌机
            return self._execute_formation_approach(env, agent_id, role)
            
        elif self.current_phase == TacticalPhase.MELD_MTR:
            # 阶段2: 继续接近，准备发射
            return self._execute_approach_to_launch(env, agent_id, role)
            
        elif self.current_phase == TacticalPhase.MTR_TR:
            # 阶段3: 长机发射，僚机滞后
            return self._execute_leader_launch_phase(env, agent_id, role)
            
        elif self.current_phase == TacticalPhase.TR_DOR:
            # 阶段4: 长机Short Skate，僚机滞后
            return self._execute_leader_skate_phase(env, agent_id, role)
            
        else:  # DOR_DR
            # 阶段5: 僚机发射并Short Skate
            return self._execute_wingman_launch_phase(env, agent_id, role)
    
    def _execute_formation_approach(self, env, agent_id: str, role: str) -> Tuple[int, int, int]:
        """执行编队接近"""
        # 指向主要敌机
        enemy_pos = self._get_primary_enemy_position(env)
        if enemy_pos is None:
            return 7, 8, 3  # 平稳飞行
        
        my_pos = env.agents[agent_id].get_position()
        target_heading = self._calculate_bearing_to_target(my_pos, enemy_pos)
        
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff)) if abs(heading_diff) > 1.0 else 8
        
        return 7, heading_cmd_id, 3  # 保持高度，指向敌机，保持速度
    
    def _execute_approach_to_launch(self, env, agent_id: str, role: str) -> Tuple[int, int, int]:
        """执行接近发射位置"""
        # 继续指向敌机
        return self._execute_formation_approach(env, agent_id, role)
    
    def _execute_leader_launch_phase(self, env, agent_id: str, role: str) -> Tuple[int, int, int]:
        """执行长机发射阶段"""
        if role == 'leader':
            # 长机：积极发射导弹
            return self._leader_aggressive_behavior(env, agent_id)
        else:
            # 僚机：保持滞后位置
            return self._wingman_lag_behavior(env, agent_id)
    
    def _execute_leader_skate_phase(self, env, agent_id: str, role: str) -> Tuple[int, int, int]:
        """执行长机Short Skate阶段"""
        if role == 'leader':
            # 长机：执行Short Skate
            return self._execute_short_skate_maneuver(env, agent_id)
        else:
            # 僚机：继续滞后
            return self._wingman_lag_behavior(env, agent_id)
    
    def _execute_wingman_launch_phase(self, env, agent_id: str, role: str) -> Tuple[int, int, int]:
        """执行僚机发射阶段"""
        if role == 'wingman':
            # 僚机：发射导弹并准备Short Skate
            return self._wingman_launch_and_skate(env, agent_id)
        else:
            # 长机：继续Short Skate
            return self._execute_short_skate_maneuver(env, agent_id)
    
    def _leader_aggressive_behavior(self, env, agent_id: str) -> Tuple[int, int, int]:
        """长机积极行为"""
        # 指向主要敌机
        enemy_pos = self._get_primary_enemy_position(env)
        if enemy_pos is None:
            return 7, 8, 3
        
        my_pos = env.agents[agent_id].get_position()
        target_heading = self._calculate_bearing_to_target(my_pos, enemy_pos)
        
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff)) if abs(heading_diff) > 1.0 else 8
        
        return 7, heading_cmd_id, 4  # 保持高度，指向敌机，轻微加速
    
    def _wingman_lag_behavior(self, env, agent_id: str) -> Tuple[int, int, int]:
        """僚机滞后行为"""
        # 保持当前态势，不积极接近
        return 7, 8, 3  # 保持高度，保持航向，保持速度
    
    def _wingman_launch_and_skate(self, env, agent_id: str) -> Tuple[int, int, int]:
        """僚机发射并Short Skate"""
        # 首先尝试发射，然后准备Short Skate
        if not self.drag_shoot_states['wingman_launched']:
            # 指向敌机准备发射
            return self._leader_aggressive_behavior(env, agent_id)
        else:
            # 已发射，执行Short Skate
            return self._execute_short_skate_maneuver(env, agent_id)
    
    def _execute_short_skate_maneuver(self, env, agent_id: str) -> Tuple[int, int, int]:
        """执行Short Skate机动"""
        current_time = env.current_step * env.time_interval
        
        # 检查是否已经开始Short Skate
        if agent_id not in self.active_maneuvers:
            # 初始化Short Skate状态
            self.active_maneuvers[agent_id] = {
                'maneuver': 'short_skate',
                'start_time': current_time,
                'phase': 'turn_cold',
                'initial_heading': np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            }
        
        return self._execute_short_skate(env, agent_id, current_time)
    
    def _execute_short_skate(self, env, agent_id: str, current_time: float) -> Tuple[int, int, int]:
        """执行Short Skate机动"""
        state = self.active_maneuvers[agent_id]
        phase_time = current_time - state['start_time']
        
        if state['phase'] == 'turn_cold':
            # 转冷阶段：快速转向180度
            if phase_time < 15.0:  # 15秒转冷
                target_heading = (state['initial_heading'] + 180) % 360
                current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
                
                heading_diff = target_heading - current_heading
                while heading_diff > 180: heading_diff -= 360
                while heading_diff < -180: heading_diff += 360
                
                if abs(heading_diff) > 5.0:
                    heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
                    return 7, heading_cmd_id, 3
                else:
                    # 转冷完成，进入加速逃离
                    state['phase'] = 'escape'
                    state['escape_start_time'] = current_time
        
        # 加速逃离阶段
        return 7, 8, 6  # 保持高度，保持航向，最大加速
    
    def _get_enemy_command_indices(self, env, agent_id: str) -> Tuple[int, int, int]:
        """敌方指令 - 简单的对抗逻辑"""
        # 敌方保持朝向友方的基本对抗态势
        friendly_pos = self._get_primary_friendly_position(env)
        if friendly_pos is None:
            return 7, 8, 3
        
        my_pos = env.agents[agent_id].get_position()
        target_heading = self._calculate_bearing_to_target(my_pos, friendly_pos)
        
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff)) if abs(heading_diff) > 2.0 else 8
        
        return 7, heading_cmd_id, 3
    
    # 复用基类的辅助方法
    def _get_primary_enemy_position(self, env) -> Optional[np.ndarray]:
        """获取主要敌机位置"""
        for agent_id, agent in env.agents.items():
            if agent_id.startswith('B') and agent.is_alive:
                return agent.get_position()
        return None
    
    def _get_primary_friendly_position(self, env) -> Optional[np.ndarray]:
        """获取主要友机位置"""
        for agent_id, agent in env.agents.items():
            if agent_id.startswith('A') and agent.is_alive:
                return agent.get_position()
        return None
    
    def _calculate_bearing_to_target(self, my_pos: np.ndarray, target_pos: np.ndarray) -> float:
        """计算指向目标的方位角"""
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        bearing = np.rad2deg(np.arctan2(dx, dy))
        return bearing % 360
    
    def _convert_heading_to_index(self, heading_diff_rad: float) -> int:
        """将航向差值转换为指令索引"""
        # 找到最接近的索引
        diff_deg = np.rad2deg(heading_diff_rad)
        heading_options = [-60, -45, -30, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30, 45, 60]
        
        closest_idx = 0
        min_diff = abs(diff_deg - heading_options[0])
        
        for i, option in enumerate(heading_options):
            if abs(diff_deg - option) < min_diff:
                min_diff = abs(diff_deg - option)
                closest_idx = i
        
        return closest_idx
'''
    
    return refactored_code


def backup_original_file():
    """备份原始文件"""
    original_file = "drag_shoot_tactical_task.py"
    backup_file = f"drag_shoot_tactical_task_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    if os.path.exists(original_file):
        import shutil
        shutil.copy2(original_file, backup_file)
        logging.info(f"✅ 原始文件已备份为: {backup_file}")
        return True
    else:
        logging.warning(f"⚠️  原始文件不存在: {original_file}")
        return False


def create_refactored_file():
    """创建重构后的文件"""
    refactored_code = create_refactored_drag_shoot_task()
    
    new_file = "drag_shoot_tactical_task_refactored.py"
    
    with open(new_file, 'w', encoding='utf-8') as f:
        f.write(refactored_code)
    
    logging.info(f"✅ 重构文件已创建: {new_file}")
    return new_file


def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("🔧 拖曳射击任务重构工具")
    print("=" * 60)
    
    try:
        # 备份原始文件
        backup_success = backup_original_file()
        
        # 创建重构文件
        new_file = create_refactored_file()
        
        print(f"✅ 重构完成!")
        print(f"新文件: {new_file}")
        
        if backup_success:
            print("💡 提示: 原始文件已备份，可以安全地用新文件替换原文件")
        
        print("\n下一步:")
        print("1. 检查新文件的内容")
        print("2. 测试新的架构")
        print("3. 如果测试通过，替换原文件")
        
    except Exception as e:
        logging.error(f"❌ 重构失败: {e}")
        raise


if __name__ == "__main__":
    main()
