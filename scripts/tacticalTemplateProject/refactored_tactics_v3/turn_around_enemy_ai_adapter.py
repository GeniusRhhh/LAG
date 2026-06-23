#!/usr/bin/env python3
"""
回转射击战术项目专用敌方AI适配器
基于统一敌方战术AI系统，为回转射击项目提供专门的敌方AI集成
"""

import logging
from typing import Optional
from enemy_ai_adapter import EnemyAIAdapter
from unified_enemy_tactical_ai import UnifiedEnemyTacticalAI


def create_turn_around_enemy_ai_integration(tactical_task) -> Optional[EnemyAIAdapter]:
    """
    为回转射击战术任务创建统一敌方AI系统集成
    
    Args:
        tactical_task: 回转射击战术任务实例
        
    Returns:
        EnemyAIAdapter: 敌方AI适配器实例，如果集成失败则返回None
    """
    try:
        # 创建统一敌方AI适配器
        enemy_ai_adapter = EnemyAIAdapter(project_name="turn_around_shooting_tactical")
        
        # 将统一敌方AI系统集成到战术任务中
        tactical_task.unified_enemy_ai = enemy_ai_adapter.unified_ai
        
        # 设置项目特定的配置
        if hasattr(tactical_task.unified_enemy_ai, 'set_project_config'):
            project_config = {
                'project_name': 'turn_around_shooting_tactical',
                'tactical_phases': ['NLT_MELD', 'MELD_MTR', 'MTR_TR', 'TR_DOR', 'DOR_DR'],
                'distance_thresholds': {
                    'NLT_MELD_min': 81000,   # 81km
                    'MELD_MTR_min': 45000,   # 45km  
                    'MTR_TR_min': 41000,     # 41km
                    'TR_DOR_min': 19600,     # 19.6km - 第一次脱离+回转交战
                    'DOR_DR_min': 14500,     # 14.5km - 第二次脱离返航
                },
                'enemy_agents': ['B0100', 'B0200'],
                'friendly_agents': ['A0100', 'A0200']
            }
            tactical_task.unified_enemy_ai.set_project_config(project_config)
        
        # 验证集成
        if hasattr(tactical_task, 'unified_enemy_ai') and tactical_task.unified_enemy_ai is not None:
            logging.info("✅ 回转射击统一敌方AI系统集成成功")
            logging.info(f"   - 支持的动作类型: {len(tactical_task.unified_enemy_ai.action_weights)} 种战术模式")
            logging.info(f"   - 支持的战术阶段: 5 个阶段")
            logging.info(f"   - 敌方智能体: B0100, B0200")
            return enemy_ai_adapter
        else:
            logging.error("❌ 回转射击统一敌方AI系统集成验证失败")
            return None
            
    except Exception as e:
        logging.error(f"❌ 回转射击统一敌方AI系统集成失败: {e}")
        import traceback
        logging.error(f"详细错误信息: {traceback.format_exc()}")
        return None


def verify_turn_around_integration(tactical_task) -> bool:
    """
    验证回转射击项目的统一敌方AI系统集成状态
    
    Args:
        tactical_task: 回转射击战术任务实例
        
    Returns:
        bool: 集成验证是否成功
    """
    try:
        # 检查基本集成
        if not hasattr(tactical_task, 'unified_enemy_ai'):
            logging.error("❌ 战术任务缺少unified_enemy_ai属性")
            return False
            
        if tactical_task.unified_enemy_ai is None:
            logging.error("❌ unified_enemy_ai为None")
            return False
            
        # 检查核心方法
        required_methods = [
            'get_enemy_command',
            'get_action_annotation_for_csv',
            'reset_for_new_episode'
        ]
        
        for method_name in required_methods:
            if not hasattr(tactical_task.unified_enemy_ai, method_name):
                logging.error(f"❌ 缺少必需方法: {method_name}")
                return False
                
        # 检查敌方指令获取方法
        if not hasattr(tactical_task, '_get_enemy_command_indices'):
            logging.error("❌ 战术任务缺少_get_enemy_command_indices方法")
            return False
            
        logging.info("✅ 回转射击统一敌方AI系统集成验证通过")
        return True
        
    except Exception as e:
        logging.error(f"❌ 回转射击集成验证失败: {e}")
        return False


if __name__ == "__main__":
    """测试回转射击敌方AI适配器"""
    print("🎯 回转射击敌方AI适配器测试")
    
    # 模拟测试
    class MockTacticalTask:
        def __init__(self):
            self.unified_enemy_ai = None
            
        def _get_enemy_command_indices(self, env, agent_id):
            return 7, 8, 3
    
    # 测试集成
    mock_task = MockTacticalTask()
    adapter = create_turn_around_enemy_ai_integration(mock_task)
    
    if adapter:
        print("✅ 适配器创建成功")
        success = verify_turn_around_integration(mock_task)
        print(f"✅ 集成验证: {'成功' if success else '失败'}")
    else:
        print("❌ 适配器创建失败")

