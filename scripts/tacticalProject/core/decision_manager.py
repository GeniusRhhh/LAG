"""
顶层决策管理器
"""
from typing import Dict, Optional, Tuple
from .control_range_manager import ControlRangeManager
from .threat_calculator import ThreatCalculator
from .tactic_selector import TacticSelector
from aircraft.aircraft_decision import AircraftDecisionNode
from intent.our_intent import OurIntentSystem
from intent.enemy_intent import EnemyIntentRecognizer
from situation.situation_evaluator import SituationEvaluator
from utils.constants import OUR_INTENT


class TacticalDecisionManager:
    """顶层决策管理器"""
    
    def __init__(self, our_intent_type: str = OUR_INTENT['CONSERVATIVE_CLEAR']):
        """
        初始化顶层决策管理器
        
        Args:
            our_intent_type: 我方意图类型
        """
        # 核心组件
        self.control_range_manager = ControlRangeManager()
        self.threat_calculator = ThreatCalculator()
        self.tactic_selector = TacticSelector()
        self.our_intent = OurIntentSystem(our_intent_type)
        self.enemy_intent_recognizer = EnemyIntentRecognizer()
        self.situation_evaluator = SituationEvaluator()
        
        # 单机决策节点
        self.lead_decision = AircraftDecisionNode('A0100', 'lead')
        self.wingman_decision = AircraftDecisionNode('A0200', 'wingman')
        
        # 当前状态
        self.current_phase = 'patrol'  # 'patrol', 'intercept', 'engagement'
        self.current_tactic = None
        self.tactical_roles = {}
        self.is_second_attack = False
        
        # 目标分配（简化版：固定分配）
        self.target_assignment = {
            'lead': 'enemy1',
            'wingman': 'enemy2'
        }
        self.lead_decision.set_assigned_target('enemy1')
        self.wingman_decision.set_assigned_target('enemy2')
        
        # 时间管理
        self.dr_start_time = None
        self.second_attack_start_time = None
    
    def update(self, blue_formation: Dict, red_formation: Dict, current_time: float) -> Dict:
        """
        每帧更新决策
        
        Args:
            blue_formation: 我方编队状态
                {
                    'lead': {'position': [...], 'velocity': [...], 'heading': ...},
                    'wingman': {'position': [...], 'velocity': [...], 'heading': ...}
                }
            red_formation: 敌方编队状态
                {
                    'enemy1': {'position': [...], 'velocity': [...], 'heading': ...},
                    'enemy2': {'position': [...], 'velocity': [...], 'heading': ...}
                }
            current_time: 当前时间 (秒)
        
        Returns:
            决策结果字典
                {
                    'lead': {'action': ..., 'maneuver': ..., 'node': ...},
                    'wingman': {'action': ..., 'maneuver': ..., 'node': ...},
                    'tactic': ...,
                    'phase': ...
                }
        """
        decisions = {
            'lead': None,
            'wingman': None,
            'tactic': self.current_tactic,
            'phase': self.current_phase
        }
        
        # 1. 检查控制距离触发（长机）
        lead_node = self._check_control_range_trigger(
            'lead', 
            blue_formation['lead']['position'],
            red_formation[self.target_assignment['lead']]['position']
        )
        
        # 2. 检查控制距离触发（僚机）
        wingman_node = self._check_control_range_trigger(
            'wingman',
            blue_formation['wingman']['position'],
            red_formation[self.target_assignment['wingman']]['position']
        )
        
        # 3. 长机决策
        if lead_node:
            decisions['lead'] = self._execute_node_decision(
                'lead', lead_node, blue_formation['lead'], 
                list(red_formation.values()), current_time
            )
        
        # 4. 僚机决策
        if wingman_node:
            decisions['wingman'] = self._execute_node_decision(
                'wingman', wingman_node, blue_formation['wingman'],
                list(red_formation.values()), current_time
            )
        
        # 5. 如果没有触发节点，提供默认决策
        if decisions['lead'] is None:
            decisions['lead'] = {
                'node': 'patrol',
                'action': 'continue',
                'tactic': self.current_tactic,
                'target': 0,
                'threat': 0.5
            }
        
        if decisions['wingman'] is None:
            decisions['wingman'] = {
                'node': 'patrol',
                'action': 'continue',
                'tactic': self.current_tactic,
                'target': 1,
                'threat': 0.5
            }
        
        # 6. 协同决策
        if decisions['lead'] and decisions['wingman']:
            self._coordinate_decisions(decisions['lead'], decisions['wingman'])
        
        return decisions
    
    def _check_control_range_trigger(self, aircraft_id: str, my_pos, target_pos) -> Optional[str]:
        """检查控制距离触发"""
        return self.control_range_manager.check_range_trigger(
            aircraft_id, my_pos, target_pos
        )
    
    def _execute_node_decision(self, aircraft_id: str, node_name: str,
                               my_state: Dict, enemy_formation: list,
                               current_time: float) -> Dict:
        """
        执行节点决策
        
        Args:
            aircraft_id: 飞机ID
            node_name: 节点名称
            my_state: 我机状态
            enemy_formation: 敌方编队
            current_time: 当前时间
        
        Returns:
            决策结果
        """
        # 选择对应的决策节点
        if aircraft_id == 'lead':
            decision_node = self.lead_decision
        else:
            decision_node = self.wingman_decision
        
        # 基础节点名称（去掉1/2后缀）
        base_node = node_name.rstrip("12'")
        
        # 执行节点特定的决策
        if base_node == 'MELD':
            return self._execute_meld_decision(aircraft_id, my_state, enemy_formation)
        elif base_node == 'DOR':
            return self._execute_dor_decision(aircraft_id, my_state, enemy_formation, current_time)
        elif base_node == 'DR':
            return self._execute_dr_decision(aircraft_id, my_state, enemy_formation, current_time)
        else:
            # 其他节点：标准决策流程
            decision = decision_node.make_decision(
                my_state, enemy_formation, node_name, self.our_intent
            )
            decision['node'] = node_name
            return decision
    
    def _execute_meld_decision(self, aircraft_id: str, my_state: Dict, 
                               enemy_formation: list) -> Dict:
        """
        MELD节点决策：战术选择 + 目标分配
        """
        # 计算我方双机的威胁值
        blue_threats = self._calculate_formation_threats(my_state, enemy_formation)
        
        # 计算敌方对我方的威胁值（简化：使用相同的威胁值）
        red_threats = {
            'enemy1_to_lead': blue_threats['lead'],
            'enemy2_to_lead': blue_threats['lead'],
        }
        
        # 选择战术
        tactic, roles = self.tactic_selector.select_tactic(blue_threats, red_threats)
        self.current_tactic = tactic
        self.tactical_roles = roles
        
        # 进入交战阶段
        self.current_phase = 'engagement'
        
        return {
            'action': 'continue',
            'node': 'MELD',
            'tactic': tactic,
            'roles': roles,
            'threat': blue_threats[aircraft_id]
        }
    
    def _execute_dor_decision(self, aircraft_id: str, my_state: Dict,
                             enemy_formation: list, current_time: float) -> Dict:
        """
        DOR节点决策：规避 + 预决策下一战术
        """
        # 标准决策
        if aircraft_id == 'lead':
            decision_node = self.lead_decision
        else:
            decision_node = self.wingman_decision
        
        decision = decision_node.make_decision(
            my_state, enemy_formation, 'DOR', self.our_intent
        )
        
        # 预决策下一战术
        blue_threats = self._calculate_formation_threats(my_state, enemy_formation)
        red_threats = {
            'enemy1_to_lead': blue_threats['lead'],
            'enemy2_to_lead': blue_threats['lead'],
        }
        next_tactic, next_roles = self.tactic_selector.select_tactic(blue_threats, red_threats)
        
        decision['next_tactic'] = next_tactic
        decision['next_roles'] = next_roles
        decision['node'] = 'DOR'
        
        return decision
    
    def _execute_dr_decision(self, aircraft_id: str, my_state: Dict,
                            enemy_formation: list, current_time: float) -> Dict:
        """
        DR节点决策：Beam转侧对 + 决策下一阶段战术
        """
        # 记录DR开始时间
        if self.dr_start_time is None:
            self.dr_start_time = current_time
        
        # 标准决策
        if aircraft_id == 'lead':
            decision_node = self.lead_decision
        else:
            decision_node = self.wingman_decision
        
        decision = decision_node.make_decision(
            my_state, enemy_formation, 'DR', self.our_intent
        )
        
        # 在20秒内决策下一阶段战术
        if current_time - self.dr_start_time < 20.0:
            # 计算威胁值
            blue_threats = self._calculate_formation_threats(my_state, enemy_formation)
            red_threats = {
                'enemy1_to_lead': blue_threats['lead'],
                'enemy2_to_lead': blue_threats['lead'],
            }
            
            # 选择下一阶段战术
            next_tactic, next_roles = self.tactic_selector.select_tactic(blue_threats, red_threats)
            decision['next_tactic'] = next_tactic
            decision['next_roles'] = next_roles
            decision['should_reengage'] = decision['action'] == 'continue'
        else:
            # 超过20秒，判断是否重新转热
            decision['should_reengage'] = decision['action'] == 'continue'
        
        decision['node'] = 'DR'
        
        # 如果决定重新进攻，进入第二阶段
        if decision.get('should_reengage', False):
            self.enter_second_phase()
            self.second_attack_start_time = current_time
        
        return decision
    
    def _calculate_formation_threats(self, my_state: Dict, enemy_formation: list) -> Dict:
        """计算编队威胁值（简化版）"""
        # 这里简化处理，实际应该分别计算长机和僚机的威胁值
        threat_result = self.threat_calculator.calculate_formation_threat(
            my_state, enemy_formation
        )
        
        return {
            'lead': threat_result['max_threat'],
            'wingman': threat_result['max_threat']
        }
    
    def _coordinate_decisions(self, lead_decision: Dict, wingman_decision: Dict):
        """协同决策"""
        # 场景1：长机撤退，僚机继续
        if lead_decision['action'] == 'retreat' and wingman_decision['action'] == 'continue':
            # 僚机需要调整战术（例如从双机战术变为单机战术）
            wingman_decision['adjust_for_solo'] = True
        
        # 场景2：双机都撤退
        if lead_decision['action'] == 'retreat' and wingman_decision['action'] == 'retreat':
            # 协调撤退路线
            lead_decision['retreat_direction'] = 'left'
            wingman_decision['retreat_direction'] = 'right'
    
    def enter_second_phase(self):
        """进入第二轮进攻"""
        self.is_second_attack = True
        self.control_range_manager.enter_second_phase()
        self.dr_start_time = None
    
    def reset(self):
        """重置决策管理器"""
        self.control_range_manager.reset()
        self.lead_decision.reset()
        self.wingman_decision.reset()
        self.current_phase = 'patrol'
        self.current_tactic = None
        self.tactical_roles = {}
        self.is_second_attack = False
        self.dr_start_time = None
        self.second_attack_start_time = None
