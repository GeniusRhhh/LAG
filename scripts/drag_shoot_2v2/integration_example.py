#!/usr/bin/env python3
"""
统一敌方AI系统集成示例
展示如何在现有战术项目中集成新的统一敌方AI系统和增强行动注释系统

集成步骤：
1. 替换现有的敌方AI实现
2. 启用行动注释系统
3. 更新数据记录系统
4. 测试和验证

作者：集成示例
版本：v1.0
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any
from enemy_ai_adapter import create_drag_shoot_enemy_ai
from enhanced_action_annotator import create_enhanced_annotator


class UnifiedEnemyAIIntegration:
    """统一敌方AI系统集成类"""
    
    def __init__(self, project_name: str = "drag_shoot"):
        """
        初始化集成系统
        
        Args:
            project_name: 项目名称
        """
        self.project_name = project_name
        
        # 创建敌方AI适配器
        if project_name == "drag_shoot":
            from enemy_ai_adapter import create_drag_shoot_enemy_ai
            self.enemy_ai = create_drag_shoot_enemy_ai()
        elif project_name == "pincer_attack":
            from enemy_ai_adapter import create_pincer_attack_enemy_ai
            self.enemy_ai = create_pincer_attack_enemy_ai()
        elif project_name == "front_back_attack":
            from enemy_ai_adapter import create_front_back_attack_enemy_ai
            self.enemy_ai = create_front_back_attack_enemy_ai()
        elif project_name == "high_low_attack":
            from enemy_ai_adapter import create_high_low_attack_enemy_ai
            self.enemy_ai = create_high_low_attack_enemy_ai()
        elif project_name == "side_by_side":
            from enemy_ai_adapter import create_side_by_side_enemy_ai
            self.enemy_ai = create_side_by_side_enemy_ai()
        else:
            from enemy_ai_adapter import create_enemy_ai_adapter
            self.enemy_ai = create_enemy_ai_adapter(project_name)
        
        # 创建增强行动注释器
        self.annotator = create_enhanced_annotator(self.enemy_ai)
        
        # 行动注释数据缓存
        self.annotation_data = {}
        
        logging.info(f"🔧 统一敌方AI系统集成完成 - 项目: {project_name}")
    
    # ==================== 主要集成接口 ====================
    
    def get_enemy_command_indices(self, env, agent_id: str) -> Tuple[int, int, int]:
        """
        获取敌方战术指令 - 替换现有战术任务中的敌方AI调用
        
        这个函数可以直接替换现有项目中的：
        - _get_enemy_command_indices()
        - enemy_tactical_ai.get_command()
        - 等其他敌方AI调用
        
        Args:
            env: 环境对象
            agent_id: 智能体ID
            
        Returns:
            Tuple[int, int, int]: (高度指令, 航向指令, 速度指令)
        """
        try:
            # 获取当前时间
            current_time = getattr(env, 'current_step', 0) * getattr(env, 'time_interval', 0.2)
            
            # 生成行动注释
            self._update_action_annotation(agent_id, current_time)
            
            # 获取战术指令
            return self.enemy_ai.get_enemy_command(env, agent_id, current_time)
            
        except Exception as e:
            logging.error(f"集成敌方AI指令生成失败 {agent_id}: {e}")
            return 7, 8, 3  # 默认平稳飞行
    
    def _update_action_annotation(self, agent_id: str, current_time: float):
        """更新行动注释数据"""
        try:
            # 生成CSV格式的注释数据
            csv_data = self.annotator.get_csv_annotation_data(agent_id, current_time)
            self.annotation_data[agent_id] = csv_data
            
        except Exception as e:
            logging.error(f"行动注释更新失败 {agent_id}: {e}")
    
    def get_action_annotation_for_csv(self, agent_id: str) -> Dict[str, str]:
        """
        获取用于CSV记录的行动注释数据 - 简化版本，只返回Action_Intent

        可以在unified_data_recorder.py中调用此函数来添加行动注释列

        Args:
            agent_id: 智能体ID

        Returns:
            Dict[str, str]: 行动注释数据
        """
        annotation_data = self.annotation_data.get(agent_id, {})
        return {
            'Action_Intent': annotation_data.get('Action_Intent', 'search')
        }
    
    # ==================== 兼容性接口 ====================
    
    def reset_for_new_episode(self):
        """新回合重置 - 在任务开始时调用"""
        self.enemy_ai.reset_all_agents()
        self.annotation_data.clear()
        logging.info(f"统一敌方AI系统已重置 - 项目: {self.project_name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'project_name': self.project_name,
            'enemy_ai_status': self.enemy_ai.get_system_status(),
            'active_annotations': list(self.annotation_data.keys()),
            'annotation_summary': {
                agent_id: self.annotator.get_annotation_summary(agent_id)
                for agent_id in self.annotation_data.keys()
            }
        }
    
    # ==================== 调试接口 ====================
    
    def enable_debug_mode(self):
        """启用调试模式"""
        self.enemy_ai.enable_debug_logging()
        logging.info("统一敌方AI系统调试模式已启用")
    
    def force_enemy_action(self, agent_id: str, action_name: str):
        """强制敌方执行特定动作 - 用于测试"""
        self.enemy_ai.force_action(agent_id, action_name)
    
    def get_detailed_enemy_status(self, agent_id: str) -> Dict[str, Any]:
        """获取详细的敌方状态信息"""
        return self.enemy_ai.get_detailed_action_info(agent_id)


# ==================== 集成辅助函数 ====================

def integrate_into_drag_shoot_task(tactical_task):
    """
    将统一敌方AI系统集成到拖曳射击任务中
    
    使用方法：
    在drag_shoot_tactical_task.py中添加：
    
    from integration_example import integrate_into_drag_shoot_task
    
    class DragShootTacticalTask(MultipleCombatTask):
        def __init__(self, ...):
            super().__init__(...)
            self.unified_enemy_ai = integrate_into_drag_shoot_task(self)
        
        def _get_enemy_command_indices(self, env, agent_id):
            return self.unified_enemy_ai.get_enemy_command_indices(env, agent_id)
    """
    integration = UnifiedEnemyAIIntegration("drag_shoot")
    
    # 将集成系统绑定到战术任务
    tactical_task.unified_enemy_ai = integration
    
    logging.info("统一敌方AI系统已集成到拖曳射击任务")
    return integration


def integrate_into_pincer_attack_task(tactical_task):
    """将统一敌方AI系统集成到钳形夹击任务中"""
    integration = UnifiedEnemyAIIntegration("pincer_attack")
    tactical_task.unified_enemy_ai = integration
    logging.info("统一敌方AI系统已集成到钳形夹击任务")
    return integration


def integrate_into_front_back_task(tactical_task):
    """将统一敌方AI系统集成到前后攻击任务中"""
    integration = UnifiedEnemyAIIntegration("front_back_attack")
    tactical_task.unified_enemy_ai = integration
    logging.info("统一敌方AI系统已集成到前后攻击任务")
    return integration


def integrate_into_high_low_task(tactical_task):
    """将统一敌方AI系统集成到上下夹击任务中"""
    integration = UnifiedEnemyAIIntegration("high_low_attack")
    tactical_task.unified_enemy_ai = integration
    logging.info("统一敌方AI系统已集成到上下夹击任务")
    return integration


def integrate_into_side_by_side_task(tactical_task):
    """将统一敌方AI系统集成到并排射击任务中"""
    integration = UnifiedEnemyAIIntegration("side_by_side")
    tactical_task.unified_enemy_ai = integration
    logging.info("统一敌方AI系统已集成到并排射击任务")
    return integration


# ==================== 数据记录集成 ====================

def enhance_unified_data_recorder(recorder, integration: UnifiedEnemyAIIntegration):
    """
    增强unified_data_recorder.py以支持行动注释
    
    使用方法：
    在unified_data_recorder.py的record_aircraft_trajectory方法中添加：
    
    # 如果有集成的敌方AI系统，添加行动注释
    if hasattr(tactical_task, 'unified_enemy_ai'):
        annotation_data = tactical_task.unified_enemy_ai.get_action_annotation_for_csv(agent_id)
        trajectory_record.update(annotation_data)
    """
    
    # 保存原始的record_aircraft_trajectory方法
    original_record = recorder.record_aircraft_trajectory
    
    def enhanced_record_aircraft_trajectory(env, current_time, tactical_task=None):
        """增强的轨迹记录方法"""
        # 调用原始记录方法
        original_record(env, current_time, tactical_task)
        
        # 如果最后一条记录是敌方智能体，添加行动注释
        if recorder.trajectory_data:
            last_record = recorder.trajectory_data[-1]
            agent_id = last_record['Agent_ID']
            
            if agent_id.startswith('B'):  # 敌方智能体
                annotation_data = integration.get_action_annotation_for_csv(agent_id)
                last_record.update(annotation_data)
    
    # 替换记录方法
    recorder.record_aircraft_trajectory = enhanced_record_aircraft_trajectory
    
    logging.info("unified_data_recorder已增强支持行动注释")


# ==================== 使用示例和测试 ====================

def test_integration():
    """测试集成系统"""
    print("🔧 统一敌方AI系统集成测试")
    
    # 测试不同项目的集成
    projects = ["drag_shoot", "pincer_attack", "front_back_attack", "high_low_attack", "side_by_side"]
    
    for project in projects:
        print(f"\n测试项目: {project}")
        integration = UnifiedEnemyAIIntegration(project)
        
        status = integration.get_system_status()
        print(f"  - 敌方AI状态: {status['enemy_ai_status']}")
        print(f"  - 活跃注释: {len(status['active_annotations'])}")
        
        # 测试强制动作
        integration.force_enemy_action("B0100", "short_skate")
        print(f"  - 强制动作测试: 完成")
    
    print("\n✅ 统一敌方AI系统集成测试完成")


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 运行测试
    test_integration()
    
    print("\n📋 集成使用说明:")
    print("1. 在战术任务类的__init__方法中调用相应的integrate_into_xxx_task函数")
    print("2. 将现有的_get_enemy_command_indices方法替换为unified_enemy_ai.get_enemy_command_indices")
    print("3. 在unified_data_recorder中调用enhance_unified_data_recorder函数启用行动注释")
    print("4. 在任务开始时调用reset_for_new_episode方法重置系统状态")
    print("5. 可选：调用enable_debug_mode启用详细日志")
