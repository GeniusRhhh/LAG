"""
MTR Prime Node (Minimum Target Range Prime)
MTR节点的变体，用于处理特定的战术决策逻辑
"""
import logging
from .mtr_node import MTRNode

class MTRPrimeNode(MTRNode):
    """
    MTR Prime节点
    继承自MTRNode，可能包含特定的扩展逻辑
    """
    def __init__(self, config=None):
        super().__init__()
        self.node_name = "MTR_PRIME"
        
    def evaluate(self, situation_data):
        """
        评估MTR Prime节点条件
        """
        # 复用MTR节点的逻辑，或者添加特定的扩展
        return super().evaluate(situation_data)
