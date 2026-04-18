#!/usr/bin/env python3
"""
前后攻击战术仿真的敌方AI适配器
基于拖曳射击项目的成功架构，只保留Action_Intent列
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入统一敌方战术AI系统
from unified_enemy_tactical_ai import UnifiedEnemyTacticalAI
from enhanced_action_annotator import EnhancedActionAnnotator


class FrontBackEnemyAIAdapter:
    """前后攻击战术仿真的敌方AI适配器"""
    
    def __init__(self, project_name: str = "front_back_attack"):
        """
        初始化前后攻击敌方AI适配器
        
        Args:
            project_name: 项目名称
        """
        self.project_name = project_name
        
        # 初始化统一敌方战术AI
        self.unified_ai = UnifiedEnemyTacticalAI()
        
        # 初始化行动注释器
        self.annotator = EnhancedActionAnnotator(self.unified_ai)
        
        # 存储注释数据
        self.annotation_data = {}
        
        logging.info(f"前后攻击敌方AI适配器初始化完成 - 项目: {project_name}")
    
    def reset_for_new_episode(self):
        """为新回合重置系统状态"""
        try:
            self.unified_ai.reset_for_new_episode()
            self.annotation_data.clear()
            logging.info("前后攻击敌方AI适配器已重置")
        except Exception as e:
            logging.error(f"前后攻击敌方AI适配器重置失败: {e}")
    
    def get_enemy_action(self, env, agent_id: str, current_time: float):
        """
        获取敌方智能体的战术指令
        
        Args:
            env: 仿真环境
            agent_id: 智能体ID
            current_time: 当前时间
            
        Returns:
            Tuple[int, int, int]: (高度指令, 航向指令, 速度指令)
        """
        try:
            # 使用统一敌方战术AI生成指令
            altitude_cmd, heading_cmd, velocity_cmd = self.unified_ai.get_enemy_command(
                env, agent_id, current_time
            )
            
            # 更新行动注释
            self._update_action_annotation(agent_id, current_time)
            
            return altitude_cmd, heading_cmd, velocity_cmd
            
        except Exception as e:
            logging.error(f"前后攻击敌方{agent_id}战术指令生成失败: {e}")
            # 返回默认平稳飞行指令
            return 7, 8, 3
    
    def _update_action_annotation(self, agent_id: str, current_time: float):
        """更新行动注释数据"""
        try:
            # 生成CSV格式的注释数据（只包含Action_Intent）
            csv_data = self.annotator.get_csv_annotation_data(agent_id, current_time)
            self.annotation_data[agent_id] = csv_data
            
        except Exception as e:
            logging.error(f"前后攻击行动注释更新失败 {agent_id}: {e}")
    
    def get_action_annotation_for_csv(self, agent_id: str) -> Dict[str, str]:
        """
        获取用于CSV记录的行动注释数据（简化版本，只返回Action_Intent）

        Args:
            agent_id: 智能体ID

        Returns:
            Dict[str, str]: 行动注释数据
        """
        try:
            # 直接调用统一敌方AI系统的注释方法
            return self.unified_ai.get_action_annotation_for_csv(agent_id)
        except Exception as e:
            logging.error(f"前后攻击敌方{agent_id}行动注释获取失败: {e}")
            return {'Action_Intent': 'search'}

    def get_action_type_for_csv(self, agent_id: str) -> Dict[str, str]:
        """
        获取CSV格式的具体战术动作类型信息

        Args:
            agent_id: 智能体ID

        Returns:
            Dict[str, str]: 包含action_type的字典
        """
        try:
            # 委托给统一敌方AI系统
            if hasattr(self, 'unified_ai') and self.unified_ai:
                return self.unified_ai.get_action_type_for_csv(agent_id)
            else:
                # 如果没有统一AI系统，返回空的action_type
                return {'action_type': ''}
        except Exception as e:
            logging.error(f"前后攻击敌方{agent_id}CSV动作类型获取失败: {e}")
            return {
                'action_type': ''
            }
    
    def force_action(self, agent_id: str, action_name: str, current_time: float = 0.0):
        """强制执行特定动作 - 用于测试"""
        try:
            from unified_enemy_tactical_ai import ActionType
            action_type = ActionType(action_name)
            self.unified_ai.force_action(agent_id, action_type, current_time)
        except ValueError:
            logging.error(f"未知动作类型: {action_name}")
    
    def get_system_status(self, agent_id: str) -> Dict[str, Any]:
        """获取系统状态信息"""
        return self.unified_ai.get_system_status(agent_id)
    
    def get_current_action_intent(self, agent_id: str) -> str:
        """获取当前动作意图"""
        return self.unified_ai.get_action_annotation(agent_id)


def create_front_back_enemy_ai_integration(tactical_task):
    """为前后攻击战术任务创建敌方AI集成"""
    try:
        # 创建适配器
        adapter = FrontBackEnemyAIAdapter("front_back_attack")
        
        # 将适配器附加到战术任务
        tactical_task.unified_enemy_ai = adapter
        
        logging.info("前后攻击统一敌方AI系统集成完成")
        return adapter
        
    except Exception as e:
        logging.error(f"前后攻击敌方AI集成失败: {e}")
        return None


if __name__ == "__main__":
    # 测试适配器
    logging.basicConfig(level=logging.INFO)
    
    print("测试前后攻击敌方AI适配器...")
    adapter = FrontBackEnemyAIAdapter()
    
    # 测试基本功能
    print(f"当前动作意图: {adapter.get_current_action_intent('B0100')}")
    print(f"系统状态: {adapter.get_system_status('B0100')}")
    
    print("前后攻击敌方AI适配器测试完成")
