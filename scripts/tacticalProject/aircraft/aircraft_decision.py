"""
单机决策节点系统
"""
from typing import Dict, Optional
from core.threat_evaluator import ThreatEvaluator
from intent.enemy_intent import EnemyIntentRecognizer
from utils.constants import AIRCRAFT_ROLE


class AircraftDecisionNode:
    """单机决策节点"""
    
    def __init__(self, aircraft_id: str, role: str):
        """
        初始化单机决策节点
        
        Args:
            aircraft_id: 飞机ID ('A0100' or 'A0200')
            role: 飞机角色 ('lead' or 'wingman')
        """
        self.aircraft_id = aircraft_id
        self.role = role
        self.threat_evaluator = ThreatEvaluator()
        self.intent_recognizer = EnemyIntentRecognizer()
        
        # 决策历史
        self.decision_history = []
        
        # 当前状态
        self.current_node = None
        self.current_threat = None
        self.current_enemy_intent = None
        self.assigned_target = None
    
    def make_decision(self, my_state: Dict, enemy_formation: list, 
                     current_node: str, our_intent) -> Dict:
        """
        在控制距离节点做出决策
        
        Args:
            my_state: 我机状态
            enemy_formation: 敌方编队状态列表
            current_node: 当前控制距离节点
            our_intent: 我方意图系统
        
        Returns:
            决策结果字典
                - 'action': 'continue' / 'evasion' / 'retreat'
                - 'threat': 威胁值字典
                - 'enemy_intent': 敌方意图
                - 'node': 当前节点
        """
        self.current_node = current_node
        
        # 1. 计算威胁值（使用统一威胁评估器，简化处理）
        # 注意：这里需要实际的飞机对象和环境对象，暂时使用简化版本
        try:
            # 简化的威胁评估，返回默认值
            threat_result = {
                'max_threat': 0.5,
                'threats': [{'total': 0.5}] * len(enemy_formation)
            }
        except Exception:
            threat_result = {
                'max_threat': 0.5,
                'threats': [{'total': 0.5}] * len(enemy_formation)
            }
        self.current_threat = threat_result['max_threat']
        
        # 2. 识别敌方意图（针对本机，选择威胁最大的敌机）
        if enemy_formation:
            # 找到威胁最大的敌机
            max_threat_idx = 0
            max_threat_value = 0.0
            for i, threat in enumerate(threat_result['threats']):
                if threat['total'] > max_threat_value:
                    max_threat_value = threat['total']
                    max_threat_idx = i
            
            enemy_state = enemy_formation[max_threat_idx]
            self.current_enemy_intent = self.intent_recognizer.recognize(
                my_state, enemy_state
            )
        else:
            self.current_enemy_intent = 'neutral'
        
        # 3. 判断威胁程度（简化处理）
        if self.current_threat > 0.8:
            threat_degree = 'high'
        elif self.current_threat > 0.5:
            threat_degree = 'medium'
        else:
            threat_degree = 'low'
        
        # 4. 根据我方意图判断是否继续攻击
        should_continue = our_intent.should_continue_attack(
            self.assigned_target if self.assigned_target else 'enemy1',
            self.current_enemy_intent,
            threat_degree,
            current_node
        )
        
        # 5. 确定最终行动
        if not should_continue:
            action = 'retreat'
        else:
            action = threat_degree
        
        # 6. 记录决策历史
        decision = {
            'action': action,
            'threat': self.current_threat,
            'enemy_intent': self.current_enemy_intent,
            'node': current_node,
            'timestamp': None  # 可以添加时间戳
        }
        self.decision_history.append(decision)
        
        return decision
    
    def set_assigned_target(self, target_id: str):
        """设置分配的目标"""
        self.assigned_target = target_id
    
    def get_decision_history(self) -> list:
        """获取决策历史"""
        return self.decision_history
    
    def reset(self):
        """重置决策节点"""
        self.decision_history = []
        self.current_node = None
        self.current_threat = None
        self.current_enemy_intent = None
