"""
战术选择算法 - 实现项目说明.md Algorithm 6-1 和 Algorithm 6-1.1

Algorithm 6-1: Tactical Template Selection
- 紧急中断检查（导弹来袭、RWR告警）
- 决策表查询
- 战术连续性检查
- ComputeTacticalFitness评分选择

Algorithm 6-1.1: ComputeTacticalFitness
- 威胁等级评估（权重0.3）
- 队形切换代价（权重0.25）
- 任务完成概率（权重0.30）
- 战术匹配度（权重0.15）
"""
import logging
import numpy as np


class TacticalSelectorAlgorithm:
    """战术选择算法（完整实现Algorithm 6-1）"""
    
    def __init__(self, threat_evaluator, decision_table):
        """
        初始化战术选择算法
        
        Args:
            threat_evaluator: 威胁评估器
            decision_table: 决策表系统
        """
        self.threat_evaluator = threat_evaluator
        self.decision_table = decision_table
        
        # MAR距离（40km）
        self.MAR_DISTANCE = 40000
        
        # 当前战术（用于连续性检查）
        self.current_tactic = None
        
        logging.info("✅ 战术选择算法初始化完成")
    
    def select_tactic(self, control_distance, my_intent, enemy_intent, situation,
                     my_aircraft, enemy_aircraft, env, current_tactic=None):
        """
        战术模板选择算法（Algorithm 6-1）
        
        Args:
            control_distance: 控制距离 ('NLT', 'MELD', 'DOR', 'DR', etc.)
            my_intent: 我方意图
            enemy_intent: 敌方意图
            situation: 战场态势
            my_aircraft: 我方飞机列表
            enemy_aircraft: 敌方飞机列表
            env: 环境对象
            current_tactic: 当前战术（用于连续性检查）
            
        Returns:
            str: 选定的战术模板名称
        """
        try:
            # 更新当前战术
            if current_tactic:
                self.current_tactic = current_tactic
            
            # ===== 1. 紧急中断检查（生存优先） =====
            
            # 检查导弹来袭
            for my_ac in my_aircraft:
                if my_ac and my_ac.is_alive:
                    if self.threat_evaluator.detect_incoming_missiles(my_ac, env):
                        logging.warning(f"🚨 检测到导弹来袭！立即执行战术规避")
                        return 'TACTICAL_EVASION'  # T6
            
            # 检查RWR告警等级（传入env参数以检测导弹）
            for my_ac in my_aircraft:
                if my_ac and my_ac.is_alive:
                    rwr_level = self.threat_evaluator.calculate_rwr_level(my_ac, enemy_aircraft, env)
                    
                    # 计算距离
                    if enemy_aircraft:
                        enemy_pos = enemy_aircraft[0].get_position() if enemy_aircraft[0].is_alive else enemy_aircraft[1].get_position()
                        my_pos = my_ac.get_position()
                        distance = np.linalg.norm(np.array(enemy_pos) - np.array(my_pos))
                    else:
                        distance = 100000
                    
                    # RWR≥4 或 距离≤MAR
                    if rwr_level >= 4 or distance <= self.MAR_DISTANCE:
                        if my_intent == 'AGGRESSIVE_CLEAR':
                            # 激进策略：继续当前战术，但在机动层调整参数
                            logging.info(f"⚠️ RWR告警{rwr_level}级或距离≤MAR，激进策略继续当前战术")
                            # continue（不返回，继续后续决策）
                        else:
                            # 保守/防御策略：执行战术回转
                            logging.warning(f"🚨 RWR告警{rwr_level}级或距离≤MAR！执行战术回转")
                            return 'TACTICAL_TURN'  # T7
            
            # ===== 2. 从决策表查询候选战术集合 =====
            candidates = self.decision_table.query_candidates(
                control_distance, my_intent, enemy_intent, situation
            )
            
            if not candidates:
                logging.warning("决策表查询无结果，使用默认战术")
                return 'SIDE_BY_SIDE'
            
            # ===== 3. 战术连续性检查（增强版 - 允许条件切换） =====
            # 检查是否应该强制切换战术
            should_force_switch = False
            
            # 条件1：敌机数量变化（有敌机被击落）
            alive_enemies = sum(1 for e in enemy_aircraft if e and e.is_alive)
            if hasattr(self, 'last_enemy_count') and alive_enemies < self.last_enemy_count:
                should_force_switch = True
                logging.info(f"🔄 敌机数量变化 ({self.last_enemy_count}→{alive_enemies})，允许战术切换")
            self.last_enemy_count = alive_enemies
            
            # 条件2：态势显著变化
            if hasattr(self, 'last_situation') and self.last_situation != situation:
                if (self.last_situation == 'DISADVANTAGE' and situation == 'ADVANTAGE') or \
                   (self.last_situation == 'ADVANTAGE' and situation == 'DISADVANTAGE'):
                    should_force_switch = True
                    logging.info(f"🔄 态势显著变化 ({self.last_situation}→{situation})，允许战术切换")
            self.last_situation = situation
            
            # 条件3：在特定节点（DOR、DR）允许切换
            if control_distance in ['DOR', 'DR']:
                should_force_switch = True
                logging.debug(f"🔄 控制距离{control_distance}，允许战术评估")
            
            # 如果当前战术在候选集中，且没有强制切换条件，保持当前战术
            if self.current_tactic and self.current_tactic in candidates and not should_force_switch:
                logging.info(f"✅ 战术连续性检查：保持当前战术 {self.current_tactic}")
                return self.current_tactic
            
            # ===== 4. 基于态势的战术评分与选择 =====
            best_score = -np.inf
            best_tactic = None
            
            for tactic in candidates:
                score = self._compute_tactical_fitness(
                    tactic, situation, my_intent, enemy_intent,
                    my_aircraft, enemy_aircraft, env
                )
                
                logging.info(f"   战术 {tactic}: 适应度 = {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_tactic = tactic
            
            if best_tactic:
                logging.info(f"🎯 选定战术: {best_tactic} (适应度: {best_score:.3f})")
                self.current_tactic = best_tactic
                return best_tactic
            else:
                logging.warning("未找到最佳战术，使用默认战术")
                return 'SIDE_BY_SIDE'
            
        except Exception as e:
            logging.error(f"战术选择错误: {e}")
            return 'SIDE_BY_SIDE'
    
    def _compute_tactical_fitness(self, tactic, situation, my_intent, enemy_intent,
                                  my_aircraft, enemy_aircraft, env):
        """
        战术适应性评估函数（Algorithm 6-1.1）
        
        Args:
            tactic: 候选战术模板
            situation: 战场态势
            my_intent: 我方意图
            enemy_intent: 敌方意图
            my_aircraft: 我方飞机列表
            enemy_aircraft: 敌方飞机列表
            env: 环境对象
            
        Returns:
            float: 适应度分数 [0, 1]
        """
        try:
            # 1. 威胁等级评估（权重0.35 - 提高威胁评估权重）
            threat_fitness = self._evaluate_threat_fitness(tactic, my_aircraft, enemy_aircraft, env)

            # 2. 队形切换代价（权重0.20 - 降低队形切换权重）
            formation_fitness = self._evaluate_formation_fitness(tactic)

            # 3. 任务完成概率（权重0.30 - 保持不变）
            success_prob = self._evaluate_success_probability(tactic, situation, my_intent, enemy_intent)

            # 4. 战术匹配度（权重0.15 - 保持不变）
            tactical_match = self._evaluate_tactical_match(tactic, my_intent, enemy_intent, situation)

            # 加权求和（优化后的权重：0.35 + 0.20 + 0.30 + 0.15 = 1.0）
            fitness_score = (
                0.35 * threat_fitness +
                0.20 * formation_fitness +
                0.30 * success_prob +
                0.15 * tactical_match
            )

            # 记录详细评分信息
            logging.debug(f"[战术评分] {tactic}: 威胁={threat_fitness:.2f}, 队形={formation_fitness:.2f}, "
                         f"成功率={success_prob:.2f}, 匹配度={tactical_match:.2f}, 总分={fitness_score:.2f}")
            
            return np.clip(fitness_score, 0.0, 1.0)

        except Exception as e:
            logging.error(f"适应度计算错误: {e}")
            return 0.5

    def _evaluate_threat_fitness(self, tactic, my_aircraft, enemy_aircraft, env):
        """
        威胁等级评估（权重0.3）

        根据项目说明.md Algorithm 6-1.1:
        - 防御性战术（T6, T7）：威胁越高，适应性越强
        - 攻击性战术：威胁越低，适应性越强
        """
        try:
            # 计算平均威胁等级
            total_threat = 0.0
            count = 0

            for my_ac in my_aircraft:
                if my_ac and my_ac.is_alive:
                    for enemy_ac in enemy_aircraft:
                        if enemy_ac and enemy_ac.is_alive:
                            threat = self.threat_evaluator.calculate_total_threat(my_ac, enemy_ac, env)
                            total_threat += threat
                            count += 1

            if count > 0:
                avg_threat = total_threat / count
            else:
                avg_threat = 0.5

            # 归一化威胁等级（假设威胁等级已经在[0,1]）
            normalized_threat = avg_threat

            # 根据战术类型计算适应度
            if tactic in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                # 防御性战术：威胁越高，适应性越强
                threat_fitness = normalized_threat  # 威胁高 → 分数高
            else:
                # 攻击性战术：威胁越低，适应性越强
                threat_fitness = 1.0 - normalized_threat  # 威胁低 → 分数高

            return threat_fitness

        except Exception as e:
            logging.error(f"威胁适应度评估错误: {e}")
            return 0.5

    def _evaluate_formation_fitness(self, tactic):
        """
        队形切换代价评估（权重0.25）

        计算从当前队形到目标队形的转换成本
        """
        try:
            # 简化版本：不同战术的队形切换代价
            # 如果当前战术与目标战术相同，代价为0
            if self.current_tactic == tactic:
                return 1.0  # 无需切换，适应度最高

            # 定义队形切换代价矩阵（简化版本）
            # 相似战术之间切换代价较低
            formation_cost = {
                ('PINCER_ATTACK', 'HIGH_LOW_ATTACK'): 0.3,  # 钳形→上下，代价中等
                ('PINCER_ATTACK', 'SEQUENTIAL_ATTACK'): 0.5,  # 钳形→前后，代价较高
                ('PINCER_ATTACK', 'SIDE_BY_SIDE'): 0.2,  # 钳形→并排，代价较低
                ('HIGH_LOW_ATTACK', 'PINCER_ATTACK'): 0.3,
                ('HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK'): 0.4,
                ('HIGH_LOW_ATTACK', 'SIDE_BY_SIDE'): 0.5,
                ('SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'): 0.2,
                ('SIDE_BY_SIDE', 'SEQUENTIAL_ATTACK'): 0.2,
            }

            # 查询切换代价
            key = (self.current_tactic, tactic) if self.current_tactic else None
            if key and key in formation_cost:
                cost = formation_cost[key]
            else:
                cost = 0.4  # 默认代价

            # 适应度 = 1 - 代价
            formation_fitness = 1.0 - cost

            return formation_fitness

        except Exception as e:
            logging.error(f"队形适应度评估错误: {e}")
            return 0.5

    def _evaluate_success_probability(self, tactic, situation, my_intent, enemy_intent):
        """
        任务完成概率评估（权重0.30）

        基于敌我态势预测战术成功率
        """
        try:
            # 基础成功率（根据态势）
            if situation == 'ADVANTAGE':
                base_prob = 0.8
            elif situation == 'NEUTRAL':
                base_prob = 0.6
            else:  # DISADVANTAGE
                base_prob = 0.4

            # 根据战术类型调整
            if tactic in ['PINCER_ATTACK', 'HIGH_LOW_ATTACK']:
                # 协同战术在优势态势下成功率更高
                if situation == 'ADVANTAGE':
                    tactic_modifier = 0.1
                else:
                    tactic_modifier = -0.1
            elif tactic in ['SEQUENTIAL_ATTACK', 'DRAG_SHOOT']:
                # 时序战术在均势下成功率较高
                if situation == 'NEUTRAL':
                    tactic_modifier = 0.1
                else:
                    tactic_modifier = 0.0
            elif tactic == 'SIDE_BY_SIDE':
                # 并排战术在劣势下成功率相对较高（简单直接）
                if situation == 'DISADVANTAGE':
                    tactic_modifier = 0.1
                else:
                    tactic_modifier = 0.0
            elif tactic in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                # 防御战术在劣势下成功率较高
                if situation == 'DISADVANTAGE':
                    tactic_modifier = 0.2
                else:
                    tactic_modifier = 0.0
            else:
                tactic_modifier = 0.0

            # 根据敌方意图调整
            if enemy_intent == 'ATTACK' and tactic in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                intent_modifier = 0.1  # 敌方攻击时，防御战术成功率提高
            elif enemy_intent == 'RETREAT' and tactic in ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE']:
                intent_modifier = 0.1  # 敌方撤退时，追击战术成功率提高
            else:
                intent_modifier = 0.0

            # 综合成功概率
            success_prob = base_prob + tactic_modifier + intent_modifier

            return np.clip(success_prob, 0.0, 1.0)

        except Exception as e:
            logging.error(f"成功概率评估错误: {e}")
            return 0.5

    def _evaluate_tactical_match(self, tactic, my_intent, enemy_intent, situation):
        """
        战术匹配度评估（权重0.15）

        衡量战术与当前意图和态势的契合程度
        """
        try:
            match_score = 0.5  # 基础匹配度

            # 根据我方意图调整
            if my_intent == 'AGGRESSIVE_CLEAR':
                # 激进肃清：偏好攻击性战术
                if tactic in ['PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'DRAG_SHOOT']:
                    match_score += 0.3
                elif tactic in ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE']:
                    match_score += 0.1
                else:  # 防御战术
                    match_score -= 0.2
            elif my_intent == 'CONSERVATIVE_CLEAR':
                # 保守肃清：偏好协同战术
                if tactic in ['PINCER_ATTACK', 'HIGH_LOW_ATTACK']:
                    match_score += 0.2
                elif tactic in ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE']:
                    match_score += 0.1
                else:
                    match_score += 0.0
            elif my_intent == 'DEFENSIVE':
                # 防御意图：偏好防御战术
                if tactic in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                    match_score += 0.3
                elif tactic in ['PINCER_ATTACK', 'HIGH_LOW_ATTACK']:
                    match_score += 0.1
                else:
                    match_score += 0.0

            # 根据态势调整
            if situation == 'ADVANTAGE' and tactic in ['PINCER_ATTACK', 'HIGH_LOW_ATTACK']:
                match_score += 0.1
            elif situation == 'DISADVANTAGE' and tactic in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                match_score += 0.1

            return np.clip(match_score, 0.0, 1.0)

        except Exception as e:
            logging.error(f"战术匹配度评估错误: {e}")
            return 0.5

