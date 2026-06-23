"""
战术决策模块
根据当前态势在各阶段做出决策：机动选择、目标分配、规避策略等
从tactical_task.py中提取，提高代码可维护性
"""
import logging
import numpy as np
from tactical_types import TacticalPhase
from core.target_assignment import get_target_with_fallback
from tactical_utils import TacticalUtils


class TacticalDecisionMaker:
    """战术决策制定器"""
    
    def __init__(self, state_manager, threat_evaluator, situation_evaluator):
        """
        Args:
            state_manager: TacticalStateManager实例
            threat_evaluator: 威胁评估器
            situation_evaluator: 态势评估器
        """
        self.state_manager = state_manager
        self.threat_evaluator = threat_evaluator
        self.situation_evaluator = situation_evaluator
    
    def make_lr_decision(self, env, agent_id: str) -> str:
        """
        LR阶段决策：Crank或直飞
        
        Args:
            env: 环境
            agent_id: 飞机ID
            
        Returns:
            'crank' 或 'straight'
        """
        
        # 获取目标
        target_id = get_target_with_fallback(agent_id, env)
        if target_id is None:
            return 'straight'
        
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft is None or not target_aircraft.is_alive:
            return 'straight'
        
        # 计算距离和aspect angle
        distance = TacticalUtils.calculate_distance_between(env.agents[agent_id], target_aircraft)
        aspect = TacticalUtils.calculate_aspect_angle(env.agents[agent_id], target_aircraft)
        
        # 决策逻辑：
        # 1. 如果目标正对我（aspect < 30°），执行Crank规避
        # 2. 如果距离<70km且aspect>120°（目标背对），直飞追击
        # 3. 其他情况根据威胁评估
        
        if aspect < 30:
            logging.info(f"🎯 [LR决策] {agent_id}: 目标正对(aspect={aspect:.0f}°) → Crank规避")
            return 'crank'
        
        if distance < 70000 and aspect > 120:
            logging.info(f"🎯 [LR决策] {agent_id}: 追击态势(dist={distance/1000:.1f}km, aspect={aspect:.0f}°) → 直飞")
            return 'straight'
        
        # 使用威胁评估
        threat_level = self.threat_evaluator.evaluate_threat(env, agent_id, target_id)
        
        if threat_level > 0.6:
            logging.info(f"🎯 [LR决策] {agent_id}: 高威胁(threat={threat_level:.2f}) → Crank规避")
            return 'crank'
        else:
            logging.info(f"🎯 [LR决策] {agent_id}: 低威胁(threat={threat_level:.2f}) → 直飞")
            return 'straight'
    
    def make_dor_decision(self, env, agent_id: str) -> str:
        """
        DOR阶段决策：选择规避机动
        
        Args:
            env: 环境
            agent_id: 飞机ID
            
        Returns:
            'SHORT_SKATE', 'NOTCH_BACK', 或 'BEAM'
        """
        
        # 获取目标
        target_id = get_target_with_fallback(agent_id, env)
        if target_id is None:
            return 'SHORT_SKATE'
        
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft is None or not target_aircraft.is_alive:
            return 'SHORT_SKATE'
        
        # 计算态势参数
        distance = TacticalUtils.calculate_distance_between(env.agents[agent_id], target_aircraft)
        aspect = TacticalUtils.calculate_aspect_angle(env.agents[agent_id], target_aircraft)
        ata = TacticalUtils.calculate_antenna_train_angle(env.agents[agent_id], target_aircraft)
        
        # 决策逻辑：
        # 1. 如果距离<40km，高威胁 → NOTCH_BACK快速规避
        # 2. 如果ATA>120°（目标在后方），且aspect<60° → NOTCH_BACK转回
        # 3. 如果40km<距离<70km，且aspect 60-120° → BEAM机动（三九机动）
        # 4. 其他情况 → SHORT_SKATE标准返航
        
        if distance < 40000:
            logging.info(f"🎯 [DOR决策] {agent_id}: 近距高威胁(dist={distance/1000:.1f}km) → NOTCH_BACK")
            return 'NOTCH_BACK'
        
        if ata > 120 and aspect < 60:
            logging.info(f"🎯 [DOR决策] {agent_id}: 目标后方+正对(ata={ata:.0f}°, aspect={aspect:.0f}°) → NOTCH_BACK")
            return 'NOTCH_BACK'
        
        if 40000 <= distance <= 70000 and 60 <= aspect <= 120:
            logging.info(f"🎯 [DOR决策] {agent_id}: 中距侧对态势 → BEAM机动")
            return 'BEAM'
        
        logging.info(f"🎯 [DOR决策] {agent_id}: 标准态势 → SHORT_SKATE")
        return 'SHORT_SKATE'
    
    def make_dr_decision(self, env, agent_id: str) -> str:
        """
        DR阶段决策：继续交战或返航
        
        Args:
            env: 环境
            agent_id: 飞机ID
            
        Returns:
            'REENGAGE' 或 'RTB' (Return To Base)
        """
        
        # 检查剩余导弹
        missiles_remaining = self.state_manager.missiles_remaining.get(agent_id, 0)
        if missiles_remaining == 0:
            logging.info(f"🎯 [DR决策] {agent_id}: 导弹耗尽 → RTB返航")
            return 'RTB'
        
        # 检查飞机状态
        aircraft = env.agents.get(agent_id)
        if aircraft is None or not aircraft.is_alive:
            return 'RTB'
        
        # 获取目标
        target_id = get_target_with_fallback(agent_id, env)
        if target_id is None:
            logging.info(f"🎯 [DR决策] {agent_id}: 无目标 → RTB返航")
            return 'RTB'
        
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft is None or not target_aircraft.is_alive:
            logging.info(f"🎯 [DR决策] {agent_id}: 目标已摧毁 → RTB返航")
            return 'RTB'
        
        # 计算距离和态势
        distance = TacticalUtils.calculate_distance_between(aircraft, target_aircraft)
        aspect = TacticalUtils.calculate_aspect_angle(aircraft, target_aircraft)
        
        # 决策逻辑：
        # 1. 距离>90km → 返航（超出有效打击范围）
        # 2. aspect<45°且距离>70km → 目标逃离，返航
        # 3. 距离45-75km，aspect>90° → 有利态势，重新交战
        # 4. 威胁评估<0.4 → 重新交战
        # 5. 其他 → 返航
        
        if distance > 90000:
            logging.info(f"🎯 [DR决策] {agent_id}: 距离过远({distance/1000:.1f}km) → RTB返航")
            return 'RTB'
        
        if aspect < 45 and distance > 70000:
            logging.info(f"🎯 [DR决策] {agent_id}: 目标逃离(aspect={aspect:.0f}°, dist={distance/1000:.1f}km) → RTB返航")
            return 'RTB'
        
        if 45000 <= distance <= 75000 and aspect > 90:
            logging.info(f"🎯 [DR决策] {agent_id}: 有利态势(dist={distance/1000:.1f}km, aspect={aspect:.0f}°) → REENGAGE")
            return 'REENGAGE'
        
        # 威胁评估
        threat_level = self.threat_evaluator.evaluate_threat(env, agent_id, target_id)
        if threat_level < 0.4:
            logging.info(f"🎯 [DR决策] {agent_id}: 低威胁({threat_level:.2f}) → REENGAGE")
            return 'REENGAGE'
        
        logging.info(f"🎯 [DR决策] {agent_id}: 默认 → RTB返航")
        return 'RTB'
    
    def decide_target_assignment(self, env, team_agents: list) -> dict:
        """
        目标分配决策
        
        Args:
            env: 环境
            team_agents: 己方飞机列表
            
        Returns:
            目标分配字典 {agent_id: target_id}
        """
        assignments = {}
        
        # 获取敌方飞机列表
        enemy_team = 'B' if team_agents[0].startswith('A') else 'A'
        enemy_agents = [aid for aid in env.agents.keys() if aid.startswith(enemy_team) and env.agents[aid].is_alive]
        
        if len(enemy_agents) == 0:
            return assignments
        
        # 简单策略：长机打长机，僚机打僚机
        for agent_id in team_agents:
            if not env.agents[agent_id].is_alive:
                continue
            
            # 长机 vs 长机
            if agent_id.endswith('100'):
                target_id = f"{enemy_team}0100" if f"{enemy_team}0100" in enemy_agents else enemy_agents[0]
            # 僚机 vs 僚机
            else:
                target_id = f"{enemy_team}0200" if f"{enemy_team}0200" in enemy_agents else enemy_agents[-1]
            
            assignments[agent_id] = target_id
        
        return assignments
    
    def should_perform_evasion(self, env, agent_id: str) -> bool:
        """
        判断是否应该执行规避机动
        
        Args:
            env: 环境
            agent_id: 飞机ID
            
        Returns:
            True表示应该规避
        """
        
        # 检查导弹威胁
        # （这里需要环境提供导弹威胁信息，暂时简化）
        
        # 检查敌机威胁
        target_id = get_target_with_fallback(agent_id, env)
        if target_id is None:
            return False
        
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft is None or not target_aircraft.is_alive:
            return False
        
        # 计算威胁度
        threat_level = self.threat_evaluator.evaluate_threat(env, agent_id, target_id)
        
        # 威胁度>0.7触发规避
        if threat_level > 0.7:
            logging.info(f"⚠️ [{agent_id}] 高威胁检测(threat={threat_level:.2f}) → 触发规避")
            return True
        
        return False
