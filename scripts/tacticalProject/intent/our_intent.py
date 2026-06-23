"""
我方意图系统
"""
from utils.constants import OUR_INTENT


class OurIntentSystem:
    """我方意图系统"""
    
    def __init__(self, intent_type: str = OUR_INTENT['CONSERVATIVE_CLEAR']):
        """
        初始化我方意图
        
        Args:
            intent_type: 意图类型
                - 'aggressive_clear': 激进肃清
                - 'conservative_clear': 保守肃清
                - 'defensive': 防御意图
        """
        self.intent_type = intent_type
        self.enemy_disengage_count = {}  # 记录敌机连续逃逸的节点数
    
    def set_intent(self, intent_type: str):
        """设置意图类型"""
        self.intent_type = intent_type
    
    def should_continue_attack(self, enemy_id: str, enemy_intent: str,
                               threat_degree: str, current_node: str) -> bool:
        """
        根据我方意图判断是否继续攻击
        
        Args:
            enemy_id: 敌机ID
            enemy_intent: 敌方意图
            threat_degree: 威胁程度 ('retreat'/'evasion'/'continue')
            current_node: 当前节点
        
        Returns:
            True表示继续攻击
        """
        if self.intent_type == OUR_INTENT['AGGRESSIVE_CLEAR']:
            # 激进肃清：永不撤退，只要敌机未被击落就持续进攻
            return True
        
        elif self.intent_type == OUR_INTENT['CONSERVATIVE_CLEAR']:
            # 保守肃清：威胁程度大时撤退/规避
            return threat_degree == 'continue'
        
        elif self.intent_type == OUR_INTENT['DEFENSIVE']:
            # 防御意图：逼迫敌机放弃进攻
            return self._defensive_logic(enemy_id, enemy_intent, current_node)
        
        return True
    
    def _defensive_logic(self, enemy_id: str, enemy_intent: str, current_node: str) -> bool:
        """
        防御意图的决策逻辑
        
        Args:
            enemy_id: 敌机ID
            enemy_intent: 敌方意图
            current_node: 当前节点
        
        Returns:
            True表示继续攻击
        """
        # 如果敌机意图为进攻或中立，继续攻击
        if enemy_intent in ['attack', 'neutral']:
            # 重置逃逸计数
            self.enemy_disengage_count[enemy_id] = 0
            return True
        
        # 如果敌机意图为逃逸，需要连续两个节点确认
        if enemy_intent == 'disengage':
            if enemy_id not in self.enemy_disengage_count:
                self.enemy_disengage_count[enemy_id] = 0
            
            self.enemy_disengage_count[enemy_id] += 1
            
            # 连续两个节点都是逃逸，确认敌机放弃进攻
            if self.enemy_disengage_count[enemy_id] >= 2:
                return False  # 停止攻击该敌机
        
        return True
    
    def reset_disengage_count(self, enemy_id: str):
        """重置敌机逃逸计数"""
        if enemy_id in self.enemy_disengage_count:
            self.enemy_disengage_count[enemy_id] = 0
