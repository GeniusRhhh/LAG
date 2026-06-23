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

        # === 战术编号映射（对应用户定义 1~5） ===
        # 1拖曳射击 2钳形攻势 3上下夹击 4前后攻击 5并排射击
        TACTICS_1_5 = ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE']
        TACTICS_1_3 = ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK']
        TACTICS_1_3_NO_PINCER = ['DRAG_SHOOT', 'HIGH_LOW_ATTACK']  # 对应“仅1,3”
        TACTICS_4_5 = ['FRONT_BACK', 'SIDE_BY_SIDE']

        def _fill_all_situations(table: dict, my_intent: str, enemy_intent_type: str, tactics: list, maneuvers: list):
            for sit in ('ADVANTAGE', 'NEUTRAL', 'DISADVANTAGE'):
                table[(my_intent, sit, enemy_intent_type)] = (list(tactics), list(maneuvers))
        
        # NLT节点决策表
        self.nlt_table = {}
        # 机动列表暂不作为主决策依据（你要求先沿用模板动作序列），保留兼容字段
        nlt_maneuvers = ['TACTICAL_CRANK', 'CRANK', 'LEVEL_FLIGHT', 'ACCELERATE', 'DECELERATE', 'CLIMB', 'TACTICAL_CLIMB']

        # 激进肃清：全态势全意图预执行；攻击/中立→1~5，脱离→4~5
        _fill_all_situations(self.nlt_table, 'AGGRESSIVE_CLEAR', 'ATTACK_TYPE', TACTICS_1_5, nlt_maneuvers)
        _fill_all_situations(self.nlt_table, 'AGGRESSIVE_CLEAR', 'NEUTRAL_TYPE', TACTICS_1_5, nlt_maneuvers)
        _fill_all_situations(self.nlt_table, 'AGGRESSIVE_CLEAR', 'ESCAPE_TYPE', TACTICS_4_5, nlt_maneuvers)

        # 保守肃清：我方占优且敌攻→仅1~3；敌方占优且敌攻→仅1,3；其他同激进；脱离→4~5
        for sit in ('ADVANTAGE', 'NEUTRAL', 'DISADVANTAGE'):
            # 默认同激进
            self.nlt_table[('CONSERVATIVE_CLEAR', sit, 'NEUTRAL_TYPE')] = (list(TACTICS_1_5), list(nlt_maneuvers))
            self.nlt_table[('CONSERVATIVE_CLEAR', sit, 'ESCAPE_TYPE')] = (list(TACTICS_4_5), list(nlt_maneuvers))
        self.nlt_table[('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'ATTACK_TYPE')] = (list(TACTICS_1_3), list(nlt_maneuvers))
        self.nlt_table[('CONSERVATIVE_CLEAR', 'DISADVANTAGE', 'ATTACK_TYPE')] = (list(TACTICS_1_3_NO_PINCER), list(nlt_maneuvers))
        self.nlt_table[('CONSERVATIVE_CLEAR', 'NEUTRAL', 'ATTACK_TYPE')] = (list(TACTICS_1_3), list(nlt_maneuvers))

        # 防御意图：攻击/中立→1~3；脱离→4~5
        _fill_all_situations(self.nlt_table, 'DEFENSIVE', 'ATTACK_TYPE', TACTICS_1_3, nlt_maneuvers)
        _fill_all_situations(self.nlt_table, 'DEFENSIVE', 'NEUTRAL_TYPE', TACTICS_1_3, nlt_maneuvers)
        _fill_all_situations(self.nlt_table, 'DEFENSIVE', 'ESCAPE_TYPE', TACTICS_4_5, nlt_maneuvers)
        
        # MELD节点决策表（可调整战术）
        self.meld_table = {}
        meld_maneuvers = list(nlt_maneuvers) + ['TACTICAL_DESCENT']
        # 激进/保守：攻击/中立→1~3；脱离→4~5（与你给的“动作集先不考虑，沿用模板序列”一致）
        for mi in ('AGGRESSIVE_CLEAR', 'CONSERVATIVE_CLEAR'):
            _fill_all_situations(self.meld_table, mi, 'ATTACK_TYPE', TACTICS_1_3, meld_maneuvers)
            _fill_all_situations(self.meld_table, mi, 'NEUTRAL_TYPE', TACTICS_1_3, meld_maneuvers)
            _fill_all_situations(self.meld_table, mi, 'ESCAPE_TYPE', TACTICS_4_5, meld_maneuvers)
        # 防御：同上（此阶段仍允许调整战术，但后续TR/DOR会更偏脱离）
        _fill_all_situations(self.meld_table, 'DEFENSIVE', 'ATTACK_TYPE', TACTICS_1_3, meld_maneuvers)
        _fill_all_situations(self.meld_table, 'DEFENSIVE', 'NEUTRAL_TYPE', TACTICS_1_3, meld_maneuvers)
        _fill_all_situations(self.meld_table, 'DEFENSIVE', 'ESCAPE_TYPE', TACTICS_4_5, meld_maneuvers)
        
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
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'ADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'ADVANTAGE', 'ESCAPE_TYPE'): (
                ['FRONT_BACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            # 激进肃清 - 敌方占优
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'DISADVANTAGE', 'ESCAPE_TYPE'): (
                ['FRONT_BACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            
            # 激进肃清 - 敌我均势（补全NEUTRAL态势）
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('AGGRESSIVE_CLEAR', 'NEUTRAL', 'ESCAPE_TYPE'): (
                ['FRONT_BACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            
            # 保守肃清 - 我方占优
            # 保守肃清 - 我方占优：攻击/中立仅 1~3；脱离 4~5
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'ATTACK_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'NEUTRAL_TYPE'): (
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'ADVANTAGE', 'ESCAPE_TYPE'): (
                ['FRONT_BACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            # 保守肃清 - 敌方占优（终止任务 7）
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
                ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            ('CONSERVATIVE_CLEAR', 'NEUTRAL', 'ESCAPE_TYPE'): (
                ['FRONT_BACK', 'SIDE_BY_SIDE'],
                ['TACTICAL_CRANK', 'CRANK', 'SHORT_SKATE']
            ),
            
            # 防御意图：不进入此阶段（在TR节点已脱离）
        }
    
    def query_candidates(self, control_distance, my_intent, enemy_intent, situation):
        """
        查询候选战术集合（用于完整智能战术选择系统）

        Args:
            control_distance: 控制距离 ('NLT', 'MELD', 'DOR', 'DR')
            my_intent: 我方意图 (已禁用，不再过滤)
            enemy_intent: 敌方意图 ('ATTACK', 'NEUTRAL', 'RETREAT')
            situation: 战场态势 ('ADVANTAGE', 'NEUTRAL', 'DISADVANTAGE')

        Returns:
            list: 候选战术列表（全部5种战术模板均可选）
        """
        # 🔥 [已禁用our_intent] 不再依赖我方意图过滤，全部战术模板均为候选
        ALL_TACTICS = ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE']
        return list(ALL_TACTICS)

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
        # ✅ 按用户规则：防御意图在MTR阶段遇到脱离/敌方占优，终止任务7（撤离）
        if my_intent == 'DEFENSIVE':
            if enemy_intent_type == 'ESCAPE_TYPE' or threat_level == 'DISADVANTAGE':
                return True
        return False
    
    def should_retreat_at_tr(self, my_intent, threat_level, enemy_intent_type):
        """TR节点：判断是否应该撤退"""
        # ✅ 按用户规则：
        # 防御意图：我方占优且敌方攻击/中立可维持；其余（敌方占优任意意图/脱离）终止任务7撤离
        if my_intent == 'DEFENSIVE':
            if threat_level != 'ADVANTAGE' or enemy_intent_type == 'ESCAPE_TYPE':
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
