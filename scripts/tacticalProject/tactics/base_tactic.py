"""
战术基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseTactic(ABC):
    """战术基类"""
    
    def __init__(self, tactic_id: int, tactic_name: str):
        """
        初始化战术
        
        Args:
            tactic_id: 战术编号
            tactic_name: 战术名称
        """
        self.tactic_id = tactic_id
        self.tactic_name = tactic_name
        self.is_active = False
        self.roles = {}  # 角色分配
    
    @abstractmethod
    def get_formation(self, aircraft_role: str) -> Dict:
        """
        获取战术队形
        
        Args:
            aircraft_role: 飞机角色 ('lead' or 'wingman')
        
        Returns:
            队形参数字典
        """
        pass
    
    @abstractmethod
    def get_node_decision(self, node_name: str, aircraft_role: str,
                         aircraft_state: Dict, enemy_formation: List,
                         decision: Dict) -> Dict:
        """
        获取特定节点的决策
        
        Args:
            node_name: 节点名称
            aircraft_role: 飞机角色
            aircraft_state: 飞机状态
            enemy_formation: 敌方编队
            decision: 基础决策结果
        
        Returns:
            战术特定的决策调整
        """
        pass
    
    def set_roles(self, roles: Dict):
        """设置角色分配"""
        self.roles = roles
    
    def activate(self):
        """激活战术"""
        self.is_active = True
    
    def deactivate(self):
        """停用战术"""
        self.is_active = False
    
    def get_description(self) -> str:
        """获取战术描述"""
        return f"{self.tactic_name} (ID: {self.tactic_id})"
