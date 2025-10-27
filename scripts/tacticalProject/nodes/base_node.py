"""
控制距离节点基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseNode(ABC):
    """控制距离节点基类"""
    
    def __init__(self, node_name: str):
        """
        初始化节点
        
        Args:
            node_name: 节点名称
        """
        self.node_name = node_name
        self.is_active = False
    
    @abstractmethod
    def execute(self, aircraft_state: Dict, enemy_formation: list,
                decision: Dict, tactical_context: Dict) -> Dict:
        """
        执行节点核心任务
        
        Args:
            aircraft_state: 飞机状态
            enemy_formation: 敌方编队
            decision: 决策结果
            tactical_context: 战术上下文
        
        Returns:
            机动指令字典
        """
        pass
    
    def activate(self):
        """激活节点"""
        self.is_active = True
    
    def deactivate(self):
        """停用节点"""
        self.is_active = False
    
    def get_core_task(self) -> str:
        """获取核心任务描述"""
        return "Base node core task"
