#!/usr/bin/env python3
"""
敌方AI适配器 - 将统一敌方战术AI系统集成到现有战术任务中
提供向后兼容的接口，使新系统能够无缝替换现有的敌方AI实现

设计目标：
1. 保持现有战术任务的接口不变
2. 提供统一的敌方AI行为
3. 支持行动注释系统
4. 便于在不同项目间切换和测试

作者：敌方AI适配器
版本：v1.0
"""

import logging
from typing import Tuple, Dict, Any, Optional
from unified_enemy_tactical_ai import UnifiedEnemyTacticalAI, ActionType, TacticalMode, EnemyTacticalPhase


class EnemyAIAdapter:
    """敌方AI适配器 - 统一接口适配器"""
    
    def __init__(self, project_name: str = "tactical_simulation"):
        """
        初始化敌方AI适配器
        
        Args:
            project_name: 项目名称，用于日志标识
        """
        self.project_name = project_name
        self.unified_ai = UnifiedEnemyTacticalAI()
        
        # 兼容性映射
        self._init_compatibility_mappings()
        
        logging.info(f"🔧 敌方AI适配器初始化完成 - 项目: {project_name}")
    
    def _init_compatibility_mappings(self):
        """初始化兼容性映射"""
        # 旧系统的威胁等级映射
        self.threat_level_mapping = {
            0: "NONE",
            1: "LOW", 
            2: "MEDIUM",
            3: "HIGH",
            4: "CRITICAL",
            5: "SEVERE"
        }
        
        # 旧系统的战术模式映射
        self.tactical_mode_mapping = {
            "aggressive": TacticalMode.AGGRESSIVE,
            "defensive": TacticalMode.DEFENSIVE,
            "neutral": TacticalMode.NEUTRAL
        }
    
    # ==================== 主要接口函数 ====================
    
    def get_enemy_command_indices(self, env, agent_id: str, current_time: float = None) -> Tuple[int, int, int]:
        """
        获取敌方战术指令索引 - 主要接口函数
        兼容现有战术任务的调用方式
        
        Args:
            env: 环境对象
            agent_id: 智能体ID
            current_time: 当前时间（可选）
            
        Returns:
            Tuple[int, int, int]: (高度指令, 航向指令, 速度指令)
        """
        try:
            # 如果没有提供时间，尝试从环境获取
            if current_time is None:
                current_time = getattr(env, 'current_step', 0) * getattr(env, 'time_interval', 0.2)
            
            # 调用统一AI系统
            return self.unified_ai.get_enemy_command(env, agent_id, current_time)
            
        except Exception as e:
            logging.error(f"敌方AI适配器指令生成失败 {agent_id}: {e}")
            return 7, 8, 3  # 默认平稳飞行
    
    def get_enemy_command(self, env, agent_id: str, current_time: float = None) -> Tuple[int, int, int]:
        """获取敌方战术指令 - 兼容接口"""
        return self.get_enemy_command_indices(env, agent_id, current_time)

    def get_tactical_command_indices(self, env, agent_id: str) -> Tuple[int, int, int]:
        """兼容旧接口名称"""
        return self.get_enemy_command_indices(env, agent_id)

    def _get_enemy_command_indices(self, env, agent_id: str) -> Tuple[int, int, int]:
        """兼容旧接口名称（带下划线）"""
        return self.get_enemy_command_indices(env, agent_id)
    
    # ==================== 状态管理接口 ====================
    
    def reset_agent(self, agent_id: str):
        """重置智能体状态"""
        self.unified_ai.reset_agent(agent_id)
    
    def reset_all_agents(self):
        """重置所有智能体状态"""
        # 重置常见的敌方智能体
        for agent_id in ["B0100", "B0200", "B0300", "B0400"]:
            self.unified_ai.reset_agent(agent_id)
    
    # ==================== 行动注释接口 ====================
    
    def get_action_annotation(self, agent_id: str) -> str:
        """获取动作注释 - 用于行动注释系统"""
        return self.unified_ai.get_action_annotation(agent_id)
    
    def get_detailed_action_info(self, agent_id: str) -> Dict[str, Any]:
        """获取详细的动作信息"""
        try:
            status = self.unified_ai.get_agent_status(agent_id)
            annotation = self.get_action_annotation(agent_id)
            
            return {
                'agent_id': agent_id,
                'action_annotation': annotation,
                'tactical_mode': status['tactical_mode'],
                'current_phase': status['current_phase'],
                'current_action': status['current_action'],
                'threat_level': status['threat_level'],
                'timestamp': status.get('action_start_time', 0.0)
            }
            
        except Exception as e:
            logging.error(f"获取详细动作信息失败 {agent_id}: {e}")
            return {
                'agent_id': agent_id,
                'action_annotation': 'UNKNOWN',
                'tactical_mode': 'neutral',
                'current_phase': 'MELD_MTR',
                'current_action': 'maintain_heading',
                'threat_level': 'NONE',
                'timestamp': 0.0
            }
    
    # ==================== 兼容性接口 ====================
    
    def update_tactical_phase(self, env, agent_id: str):
        """兼容旧系统的阶段更新接口"""
        # 新系统会在get_enemy_command中自动更新阶段
        pass
    
    def assess_threat_level(self, env, agent_id: str) -> str:
        """兼容旧系统的威胁评估接口"""
        try:
            current_time = getattr(env, 'current_step', 0) * getattr(env, 'time_interval', 0.2)
            situation = self.unified_ai._analyze_situation(env, agent_id, current_time)
            threat = self.unified_ai._assess_threat(situation, agent_id, current_time)
            return threat.threat_level.name
        except Exception as e:
            logging.error(f"威胁评估失败 {agent_id}: {e}")
            return "LOW"
    
    def get_current_tactical_mode(self, agent_id: str) -> str:
        """获取当前战术模式"""
        mode = self.unified_ai.tactical_mode.get(agent_id, TacticalMode.NEUTRAL)
        return mode.value
    
    def get_current_phase(self, agent_id: str) -> str:
        """获取当前战术阶段"""
        phase = self.unified_ai.current_phase.get(agent_id, EnemyTacticalPhase.MELD_MTR)
        return phase.value
    
    # ==================== 调试和测试接口 ====================
    
    def force_tactical_mode(self, agent_id: str, mode: str):
        """强制设置战术模式 - 用于测试"""
        if mode in self.tactical_mode_mapping:
            self.unified_ai.tactical_mode[agent_id] = self.tactical_mode_mapping[mode]
            logging.info(f"强制设置敌方{agent_id}战术模式: {mode}")
    
    def force_action(self, agent_id: str, action_name: str, current_time: float = 0.0):
        """强制执行特定动作 - 用于测试"""
        try:
            action_type = ActionType(action_name)
            self.unified_ai.force_action(agent_id, action_type, current_time)
        except ValueError:
            logging.error(f"未知动作类型: {action_name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态 - 用于监控"""
        return {
            'project_name': self.project_name,
            'active_agents': list(self.unified_ai.tactical_mode.keys()),
            'total_actions': len(ActionType),
            'total_modes': len(TacticalMode),
            'total_phases': len(EnemyTacticalPhase)
        }
    
    def enable_debug_logging(self):
        """启用调试日志"""
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("敌方AI适配器调试日志已启用")
    
    def disable_debug_logging(self):
        """禁用调试日志"""
        logging.getLogger().setLevel(logging.INFO)
        logging.info("敌方AI适配器调试日志已禁用")


# ==================== 工厂函数 ====================

def create_enemy_ai_adapter(project_name: str = "tactical_simulation") -> EnemyAIAdapter:
    """
    创建敌方AI适配器实例
    
    Args:
        project_name: 项目名称
        
    Returns:
        EnemyAIAdapter: 适配器实例
    """
    return EnemyAIAdapter(project_name)


# ==================== 项目特定适配器 ====================

def create_drag_shoot_enemy_ai() -> EnemyAIAdapter:
    """创建拖曳射击项目的敌方AI适配器"""
    return create_enemy_ai_adapter("drag_shoot")

def create_pincer_attack_enemy_ai() -> EnemyAIAdapter:
    """创建钳形夹击项目的敌方AI适配器"""
    return create_enemy_ai_adapter("pincer_attack")

def create_front_back_attack_enemy_ai() -> EnemyAIAdapter:
    """创建前后攻击项目的敌方AI适配器"""
    return create_enemy_ai_adapter("front_back_attack")

def create_high_low_attack_enemy_ai() -> EnemyAIAdapter:
    """创建上下夹击项目的敌方AI适配器"""
    return create_enemy_ai_adapter("high_low_attack")

def create_side_by_side_enemy_ai() -> EnemyAIAdapter:
    """创建并排射击项目的敌方AI适配器"""
    return create_enemy_ai_adapter("side_by_side")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 测试适配器
    print("🔧 敌方AI适配器测试")
    
    # 创建不同项目的适配器
    adapters = {
        "drag_shoot": create_drag_shoot_enemy_ai(),
        "pincer_attack": create_pincer_attack_enemy_ai(),
        "front_back": create_front_back_attack_enemy_ai(),
        "high_low": create_high_low_attack_enemy_ai(),
        "side_by_side": create_side_by_side_enemy_ai()
    }
    
    for project, adapter in adapters.items():
        status = adapter.get_system_status()
        print(f"{project}: {status['total_actions']} 动作, {status['total_modes']} 模式, {status['total_phases']} 阶段")
    
    print("✅ 敌方AI适配器测试完成")
