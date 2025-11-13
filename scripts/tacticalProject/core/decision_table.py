"""
决策表模块
根据我方意图、敌我态势、敌机意图查询可用战术和机动
"""
import logging


class DecisionTable:
    """决策表 - 完整实现文档中的决策表"""
    
    def __init__(self):
        self._build_decision_tables()
    
    def _build_decision_tables(self):
        """构建决策表"""
        
        # NLT节点决策表
        self.nlt_table = {
            # (我方意图, 敌我态势, 敌机意图类型) -> (可用战术列表, 可用机动列表)
            ('AGGRESSIVE_CLEAR', 'ADVANTAGE', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('AGGRESSIVE_CLEAR', 'ADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('AGGRESSIVE_CLEAR', 'ADVANTAGE', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            
            # 激进肃清 - 敌我均势（补全缺失的NEUTRAL态势）
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('CONSERVATIVE_CLEAR', 'DISADVANTAGE', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('CONSERVATIVE_CLEAR', 'DISADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('CONSERVATIVE_CLEAR', 'DISADVANTAGE', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            
            # 保守肃清 - 敌我均势（补全缺失的NEUTRAL态势）
            ('CONSERVATIVE_CLEAR', 'NEUTRAL', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('CONSERVATIVE_CLEAR', 'NEUTRAL', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('CONSERVATIVE_CLEAR', 'NEUTRAL', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            
            ('DEFENSIVE', 'ADVANTAGE', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('DEFENSIVE', 'ADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('DEFENSIVE', 'ADVANTAGE', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('DEFENSIVE', 'DISADVANTAGE', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('DEFENSIVE', 'DISADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('DEFENSIVE', 'DISADVANTAGE', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            
            # 防御意图 - 敌我均势（补全缺失的NEUTRAL态势）
            ('DEFENSIVE', 'NEUTRAL', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('DEFENSIVE', 'NEUTRAL', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
            ('DEFENSIVE', 'NEUTRAL', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']
            ),
        }
        
        # MELD节点决策表（可调整战术）
        self.meld_table = self.nlt_table.copy()  # MELD与NLT类似，但可以包含战术爬升/下降
        
        # MTR节点决策表（维持战术或脱离）
        self.mtr_table = {}  # 简化：维持当前战术
        
        # DOR节点决策表（规避机动）
        # (我方意图, 敌我态势, 敌机意图类型) -> (可用战术列表, 可用机动列表)
        self.dor_table = {
            # 激进肃清 - 我方占优
            ('AGGRESSIVE_CLEAR', 'ADVANTAGE', 'ATTACK_TYPE'): (
                ['TACTICAL_EVASION'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'ADVANTAGE', 'NEUTRAL_TYPE'): (
                ['TACTICAL_EVASION'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'ADVANTAGE', 'ESCAPE_TYPE'): (
                ['TACTICAL_EVASION'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            # 激进肃清 - 敌方占优
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'ATTACK_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'NEUTRAL_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'ESCAPE_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            
            # 保守肃清 - 我方占优
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'ATTACK_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'NEUTRAL_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'ESCAPE_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            # 保守肃清 - 敌方占优
            ('CONSERVATIVE_CLEAR', 'DISADVANTAGE', 'ATTACK_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'DISADVANTAGE', 'NEUTRAL_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'DISADVANTAGE', 'ESCAPE_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            
            # 激进肃清/保守肃清 - 敌我均势（补全NEUTRAL态势）
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'ATTACK_TYPE'): (
                ['TACTICAL_EVASION'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'NEUTRAL_TYPE'): (
                ['TACTICAL_EVASION'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'ESCAPE_TYPE'): (
                ['TACTICAL_EVASION'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'NEUTRAL', 'ATTACK_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'NEUTRAL', 'NEUTRAL_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'NEUTRAL', 'ESCAPE_TYPE'): (
                ['TACTICAL_EVASION', 'TACTICAL_TURN'],
                ['TACTICAL_CRANK', 'CRANK', 'NOTCH_BACK', 'SHORT_SKATE']
            ),
            
            # 防御意图
            ('DEFENSIVE', 'ADVANTAGE', 'ATTACK_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('DEFENSIVE', 'ADVANTAGE', 'NEUTRAL_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('DEFENSIVE', 'ADVANTAGE', 'ESCAPE_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('DEFENSIVE', 'DISADVANTAGE', 'ATTACK_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('DEFENSIVE', 'DISADVANTAGE', 'NEUTRAL_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('DEFENSIVE', 'DISADVANTAGE', 'ESCAPE_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
            
            # 防御意图 - 敌我均势（补全NEUTRAL态势）
            ('DEFENSIVE', 'NEUTRAL', 'ATTACK_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('DEFENSIVE', 'NEUTRAL', 'NEUTRAL_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('DEFENSIVE', 'NEUTRAL', 'ESCAPE_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
        }
        
        # DR节点决策表（重新进攻或返航）
        # (我方意图, 敌我态势, 敌机意图类型) -> (可用战术列表, 可用机动列表)
        self.dr_table = {
            # 激进肃清 - 我方占优
            ('AGGRESSIVE_CLEAR', 'ADVANTAGE', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'ADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'ADVANTAGE', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            # 激进肃清 - 敌方占优
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            
            # 激进肃清 - 敌我均势（补全NEUTRAL态势）
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            
            # 保守肃清 - 我方占优
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            # 保守肃清 - 敌方占优（终止任务）
            ('CONSERVATIVE_CLEAR', 'DISADVANTAGE', 'ATTACK_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'DISADVANTAGE', 'NEUTRAL_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'DISADVANTAGE', 'ESCAPE_TYPE'): (
                ['TACTICAL_TURN'],
                ['NOTCH_BACK', 'SHORT_SKATE']
            ),
            
            # 保守肃清 - 敌我均势（补全NEUTRAL态势）
            ('CONSERVATIVE_CLEAR', 'NEUTRAL', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'NEUTRAL', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'NEUTRAL', 'ESCAPE_TYPE'): (
                ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            
            # 防御意图：不进入此阶段（在TR节点已脱离）
        }
    
    def query_candidates(self, control_distance, my_intent, enemy_intent, situation):
        """
        查询候选战术集合（用于完整智能战术选择系统）

        Args:
            control_distance: 控制距离 ('NLT', 'MELD', 'DOR', 'DR')
            my_intent: 我方意图 ('AGGRESSIVE_CLEAR', 'CONSERVATIVE_CLEAR', 'DEFENSIVE')
            enemy_intent: 敌方意图 ('ATTACK', 'NEUTRAL', 'RETREAT')
            situation: 战场态势 ('ADVANTAGE', 'NEUTRAL', 'DISADVANTAGE')

        Returns:
            list: 候选战术列表
        """
        # 将敌方意图映射到决策表格式
        intent_map = {
            'ATTACK': 'ATTACK_TYPE',
            'NEUTRAL': 'NEUTRAL_TYPE',
            'RETREAT': 'ESCAPE_TYPE'
        }
        enemy_intent_type = intent_map.get(enemy_intent, 'ATTACK_TYPE')

        # 根据控制距离选择决策表
        if control_distance == 'NLT':
            table = self.nlt_table
        elif control_distance in ['MELD', 'DOR', 'DR']:
            table = self.meld_table
        else:
            table = self.nlt_table

        # 查询决策表
        key = (my_intent, situation, enemy_intent_type)
        result = table.get(key)
        if result:
            tactics, _ = result
            logging.info(f"[决策表] {control_distance}查询: {key} -> 战术{tactics}")
            return list(tactics)
        else:
            # 默认返回
            logging.warning(f"[决策表] {control_distance}查询未找到: {key}, 使用默认")
            return ['SIDE_BY_SIDE', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK']

    def query_nlt(self, my_intent, threat_level, enemy_intent_type):
        """查询NLT节点决策表"""
        key = (my_intent, threat_level, enemy_intent_type)
        result = self.nlt_table.get(key)
        if result:
            return result
        else:
            # 默认返回
            logging.warning(f"NLT决策表未找到匹配项: {key}, 使用默认值")
            return (['SIDE_BY_SIDE'], ['LEVEL_FLIGHT'])
    
    def query_meld(self, my_intent, threat_level, enemy_intent_type):
        """查询MELD节点决策表"""
        key = (my_intent, threat_level, enemy_intent_type)
        result = self.meld_table.get(key)
        if result:
            return result
        else:
            logging.warning(f"MELD决策表未找到匹配项: {key}, 使用默认值")
            return (['SIDE_BY_SIDE'], ['LEVEL_FLIGHT'])
    
    def should_retreat_at_mtr(self, my_intent, threat_level, enemy_intent_type):
        """MTR节点：判断是否应该撤退"""
        # 防御意图且敌方脱离时撤退
        if my_intent == 'DEFENSIVE' and enemy_intent_type == 'ESCAPE_TYPE':
            return True
        return False
    
    def should_retreat_at_tr(self, my_intent, threat_level, enemy_intent_type):
        """TR节点：判断是否应该撤退"""
        # 防御意图时撤退
        if my_intent == 'DEFENSIVE':
            if threat_level == 'DISADVANTAGE' or enemy_intent_type == 'ESCAPE_TYPE':
                return True
        return False
    
    def query_dor(self, my_intent, threat_level, enemy_intent_type):
        """查询DOR节点决策表"""
        key = (my_intent, threat_level, enemy_intent_type)
        result = self.dor_table.get(key)
        if result:
            return result
        else:
            logging.warning(f"DOR决策表未找到匹配项: {key}, 使用默认值")
            return (['TACTICAL_EVASION'], ['NOTCH_BACK', 'SHORT_SKATE'])
    
    def query_dr(self, my_intent, threat_level, enemy_intent_type):
        """查询DR节点决策表"""
        key = (my_intent, threat_level, enemy_intent_type)
        result = self.dr_table.get(key)
        if result:
            return result
        else:
            logging.warning(f"DR决策表未找到匹配项: {key}, 使用默认值")
            return (['TACTICAL_TURN'], ['NOTCH_BACK', 'SHORT_SKATE'])
    
    def should_re_engage_at_dr(self, my_intent, threat_level, enemy_intent_type):
        """DR节点：判断是否应该重新进攻"""
        # 激进肃清：总是重新进攻
        if my_intent == 'AGGRESSIVE_CLEAR':
            return True
        
        # 保守肃清：我方占优时重新进攻
        if my_intent == 'CONSERVATIVE_CLEAR':
            if threat_level == 'ADVANTAGE' and enemy_intent_type != 'ESCAPE_TYPE':
                return True
        
        # 防御意图：不重新进攻
        return False
