
"""
增强行动注释系统 - 专门用于统一敌方战术AI系统
提供详细的战术行动注释，支持意图识别和行为分析

功能特点：
1. 基于实际AI决策的精确注释
2. 支持复合动作的阶段化注释
3. 提供战术意图分析
4. 兼容CSV数据记录系统

作者：增强行动注释系统
版本：v1.0
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from enemy_ai_adapter import EnemyAIAdapter


class ActionIntent(Enum):
    """行动意图枚举"""
    SEARCH = "search"                    # 搜索
    TRACKING = "tracking"                # 跟踪
    LOCK_ON = "lock_on"                 # 锁定
    MISSILE_LAUNCH = "missile_launch"    # 导弹发射
    EVASIVE_MANEUVER = "evasive_maneuver"  # 规避机动
    ATTACK_MANEUVER = "attack_maneuver"    # 攻击机动
    DEFENSIVE_MANEUVER = "defensive_maneuver"  # 防御机动
    RETURN_TO_BASE = "return_to_base"      # 返航
    DISENGAGEMENT = "disengagement"        # 脱离接触
    COORDINATION = "coordination"          # 协调配合


class TacticalPosture(Enum):
    """战术姿态枚举"""
    OFFENSIVE = "offensive"      # 攻击姿态
    DEFENSIVE = "defensive"      # 防御姿态
    NEUTRAL = "neutral"         # 中性姿态
    SUPPORTIVE = "supportive"   # 支援姿态
    EVASIVE = "evasive"        # 规避姿态


@dataclass
class ActionAnnotation:
    """行动注释数据结构"""
    agent_id: str
    timestamp: float
    action_type: str
    action_intent: ActionIntent
    tactical_posture: TacticalPosture
    phase_info: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    confidence: float = 1.0


class EnhancedActionAnnotator:
    """增强行动注释器"""
    
    def __init__(self, enemy_ai_adapter: EnemyAIAdapter):
        """
        初始化增强行动注释器
        
        Args:
            enemy_ai_adapter: 敌方AI适配器实例
        """
        self.enemy_ai = enemy_ai_adapter
        self.annotation_history = {}  # 注释历史记录
        self.action_sequences = {}    # 动作序列跟踪
        
        # 初始化注释映射
        self._init_annotation_mappings()
        
        logging.info("🏷️ 增强行动注释系统初始化完成")
    
    def _init_annotation_mappings(self):
        """初始化注释映射规则"""
        # 动作类型到意图的映射
        self.action_intent_mapping = {
            # 基础动作
            "SEARCH_NEUTRAL": ActionIntent.SEARCH,
            "TRACKING_NEUTRAL": ActionIntent.TRACKING,
            "TRACKING_ATTACK": ActionIntent.LOCK_ON,
            
            # 机动动作
            "MANEUVER_LEFT_DEFENSE": ActionIntent.EVASIVE_MANEUVER,
            "MANEUVER_RIGHT_DEFENSE": ActionIntent.EVASIVE_MANEUVER,
            "MANEUVER_LEFT_ATTACK": ActionIntent.ATTACK_MANEUVER,
            "MANEUVER_RIGHT_ATTACK": ActionIntent.ATTACK_MANEUVER,
            
            # 战术机动
            "CRANK_LEFT_ATTACK": ActionIntent.ATTACK_MANEUVER,
            "CRANK_RIGHT_ATTACK": ActionIntent.ATTACK_MANEUVER,
            "CRANK_LEFT_DEFENSE": ActionIntent.DEFENSIVE_MANEUVER,
            "CRANK_RIGHT_DEFENSE": ActionIntent.DEFENSIVE_MANEUVER,
            
            "NOTCH_MANEUVER_DEFENSE": ActionIntent.EVASIVE_MANEUVER,
            "BEAM_MANEUVER_DEFENSE": ActionIntent.EVASIVE_MANEUVER,
            "BEAM_MANEUVER_NEUTRAL": ActionIntent.DEFENSIVE_MANEUVER,
            
            # 复合动作
            "SHORT_SKATE_CRANK_ATTACK": ActionIntent.ATTACK_MANEUVER,
            "SHORT_SKATE_TURN_COLD_ATTACK": ActionIntent.DISENGAGEMENT,
            "SHORT_SKATE_ESCAPE_ATTACK": ActionIntent.RETURN_TO_BASE,
            
            "AGGRESSIVE_APPROACH_ATTACK": ActionIntent.ATTACK_MANEUVER,
            "DEFENSIVE_SPLIT_DEFENSE": ActionIntent.DEFENSIVE_MANEUVER,
            "RETURN_TO_BASE_DEFENSE": ActionIntent.RETURN_TO_BASE,
            "RETURN_TO_BASE_NEUTRAL": ActionIntent.RETURN_TO_BASE,
            
            # 高度和速度调整
            "CLIMB_ATTACK": ActionIntent.ATTACK_MANEUVER,
            "DESCEND_DEFENSE": ActionIntent.EVASIVE_MANEUVER
        }
        
        # 战术模式到姿态的映射
        self.mode_posture_mapping = {
            "aggressive": TacticalPosture.OFFENSIVE,
            "defensive": TacticalPosture.DEFENSIVE,
            "neutral": TacticalPosture.NEUTRAL
        }
    
    def annotate_action(self, agent_id: str, current_time: float) -> ActionAnnotation:
        """
        为智能体的当前行动生成注释
        
        Args:
            agent_id: 智能体ID
            current_time: 当前时间
            
        Returns:
            ActionAnnotation: 行动注释
        """
        try:
            # 获取详细动作信息
            action_info = self.enemy_ai.get_detailed_action_info(agent_id)
            
            # 生成基础注释 - 智能映射而不是硬编码映射
            action_intent = self._determine_action_intent(action_info)
            
            # 确定战术姿态
            tactical_mode = action_info['tactical_mode']
            tactical_posture = self.mode_posture_mapping.get(
                tactical_mode,
                TacticalPosture.NEUTRAL
            )
            
            # 生成阶段信息
            phase_info = self._generate_phase_info(action_info)
            
            # 生成参数信息
            parameters = self._generate_parameters(action_info, agent_id)
            
            # 计算置信度
            confidence = self._calculate_confidence(action_info, agent_id)
            
            # 创建注释
            annotation = ActionAnnotation(
                agent_id=agent_id,
                timestamp=current_time,
                action_type=action_info['current_action'],
                action_intent=action_intent,
                tactical_posture=tactical_posture,
                phase_info=phase_info,
                parameters=parameters,
                confidence=confidence
            )
            
            # 记录注释历史
            self._record_annotation(annotation)
            
            return annotation
            
        except Exception as e:
            logging.error(f"行动注释生成失败 {agent_id}: {e}")
            return ActionAnnotation(
                agent_id=agent_id,
                timestamp=current_time,
                action_type="unknown",
                action_intent=ActionIntent.SEARCH,
                tactical_posture=TacticalPosture.NEUTRAL,
                confidence=0.0
            )

    def _determine_action_intent(self, action_info: Dict[str, Any]) -> ActionIntent:
        """
        根据战术模式、威胁等级和动作类型智能确定行动意图

        Args:
            action_info: 包含战术信息的字典

        Returns:
            ActionIntent: 行动意图
        """
        try:
            tactical_mode = action_info['tactical_mode']
            threat_level = action_info['threat_level']
            current_action = action_info['current_action']
            current_phase = action_info['current_phase']

            # 将枚举转换为字符串进行处理
            mode_str = str(tactical_mode).lower() if hasattr(tactical_mode, 'name') else str(tactical_mode).lower()
            action_str = str(current_action).lower() if hasattr(current_action, 'name') else str(current_action).lower()
            phase_str = str(current_phase).lower() if hasattr(current_phase, 'name') else str(current_phase).lower()

            # 基于威胁等级的基础判断
            if threat_level in ['CRITICAL', 'SEVERE']:
                # 高威胁情况下的行动意图
                if 'aggressive' in mode_str:
                    if any(action in action_str for action in ['crank', 'turn', 'climb']):
                        return ActionIntent.ATTACK_MANEUVER
                    elif 'maintain_heading' in action_str:
                        return ActionIntent.LOCK_ON
                    else:
                        return ActionIntent.ATTACK_MANEUVER
                elif 'defensive' in mode_str:
                    if any(action in action_str for action in ['spiral', 'dive', 'notch', 'beam']):
                        return ActionIntent.EVASIVE_MANEUVER
                    elif any(action in action_str for action in ['crank', 'turn']):
                        return ActionIntent.DEFENSIVE_MANEUVER
                    else:
                        return ActionIntent.EVASIVE_MANEUVER
                else:  # neutral
                    return ActionIntent.COORDINATION

            elif threat_level in ['MEDIUM', 'HIGH']:
                # 中等威胁情况下的行动意图
                if 'aggressive' in mode_str:
                    if 'maintain_heading' in action_str:
                        return ActionIntent.LOCK_ON
                    else:
                        return ActionIntent.ATTACK_MANEUVER
                elif 'defensive' in mode_str:
                    return ActionIntent.DEFENSIVE_MANEUVER
                else:  # neutral
                    if any(action in action_str for action in ['turn', 'crank']):
                        return ActionIntent.COORDINATION
                    else:
                        return ActionIntent.TRACKING

            elif threat_level == 'LOW':
                # 低威胁情况下的行动意图
                if 'aggressive' in mode_str:
                    return ActionIntent.LOCK_ON
                elif 'defensive' in mode_str:
                    return ActionIntent.DEFENSIVE_MANEUVER
                else:  # neutral
                    return ActionIntent.COORDINATION

            else:  # NONE或其他
                # 无威胁情况下的行动意图
                return ActionIntent.SEARCH

        except Exception as e:
            logging.error(f"行动意图确定失败: {e}")
            return ActionIntent.SEARCH

    def _generate_phase_info(self, action_info: Dict[str, Any]) -> Optional[str]:
        """生成阶段信息"""
        try:
            current_phase = action_info['current_phase']
            current_action = action_info['current_action']
            
            # Short Skate特殊处理
            if "SHORT_SKATE" in action_info['action_annotation']:
                if "CRANK" in action_info['action_annotation']:
                    return "SHORT_SKATE_PHASE_1_CRANK"
                elif "TURN_COLD" in action_info['action_annotation']:
                    return "SHORT_SKATE_PHASE_2_TURN_COLD"
                elif "ESCAPE" in action_info['action_annotation']:
                    return "SHORT_SKATE_PHASE_3_ESCAPE"
                else:
                    return "SHORT_SKATE_UNKNOWN_PHASE"
            
            # 返航特殊处理
            if current_action == "return_to_base":
                return "RETURN_TO_BASE_ACTIVE"
            
            # 基于战术阶段的信息
            phase_mapping = {
                "NLT_MELD": "LONG_RANGE_ENGAGEMENT",
                "MELD_MTR": "MEDIUM_RANGE_COMBAT", 
                "MTR_TR": "MISSILE_TARGET_RANGE",
                "TR_DOR": "TARGET_TO_DYNAMIC_RANGE",
                "DOR_DR": "DYNAMIC_TO_DEFENSE_RANGE"
            }
            
            return phase_mapping.get(current_phase, current_phase)
            
        except Exception as e:
            logging.error(f"阶段信息生成失败: {e}")
            return None
    
    def _generate_parameters(self, action_info: Dict[str, Any], agent_id: str) -> Optional[Dict[str, Any]]:
        """生成参数信息"""
        try:
            parameters = {
                'tactical_mode': action_info['tactical_mode'],
                'current_phase': action_info['current_phase'],
                'threat_level': action_info['threat_level'],
                'agent_role': 'leader' if agent_id == 'B0100' else 'wingman'
            }
            
            # 添加动作特定参数
            current_action = action_info['current_action']

            # 将ActionType枚举转换为字符串进行检查
            action_str = str(current_action).lower() if hasattr(current_action, 'name') else str(current_action).lower()

            if 'crank' in action_str:
                parameters['maneuver_type'] = 'crank'
                parameters['expected_duration'] = '10-25s'
            elif 'short_skate' in action_str:
                parameters['maneuver_type'] = 'short_skate'
                parameters['expected_duration'] = '35-50s'
                parameters['phases'] = 3
            elif 'return_to_base' in action_str:
                parameters['maneuver_type'] = 'return_to_base'
                parameters['target_heading'] = '0°'
                parameters['expected_duration'] = 'until_end'
            
            return parameters
            
        except Exception as e:
            logging.error(f"参数信息生成失败: {e}")
            return None
    
    def _calculate_confidence(self, action_info: Dict[str, Any], agent_id: str) -> float:
        """计算注释置信度"""
        try:
            confidence = 1.0
            
            # 基于威胁等级调整置信度
            threat_level = action_info['threat_level']
            if threat_level in ['HIGH', 'CRITICAL', 'SEVERE']:
                confidence *= 0.95  # 高威胁下行为更可预测
            elif threat_level == 'MEDIUM':
                confidence *= 0.85
            elif threat_level == 'LOW':
                confidence *= 0.75
            else:
                confidence *= 0.6   # 无威胁时行为较随机
            
            # 基于动作类型调整置信度
            current_action = action_info['current_action']

            # 将ActionType枚举转换为字符串进行比较
            action_str = str(current_action).lower() if hasattr(current_action, 'name') else str(current_action).lower()

            if any(action in action_str for action in ['short_skate', 'return_to_base']):
                confidence *= 0.9   # 复合动作置信度较高
            elif any(action in action_str for action in ['maintain_heading']):
                confidence *= 0.7   # 基础动作置信度较低
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logging.error(f"置信度计算失败: {e}")
            return 0.5
    
    def _record_annotation(self, annotation: ActionAnnotation):
        """记录注释历史"""
        agent_id = annotation.agent_id
        
        if agent_id not in self.annotation_history:
            self.annotation_history[agent_id] = []
        
        self.annotation_history[agent_id].append(annotation)
        
        # 保持历史记录在合理范围内
        if len(self.annotation_history[agent_id]) > 100:
            self.annotation_history[agent_id] = self.annotation_history[agent_id][-50:]
    
    def get_csv_annotation_data(self, agent_id: str, current_time: float) -> Dict[str, str]:
        """
        获取CSV格式的注释数据 - 简化版本，只返回Action_Intent

        Args:
            agent_id: 智能体ID
            current_time: 当前时间

        Returns:
            Dict[str, str]: CSV列数据
        """
        try:
            annotation = self.annotate_action(agent_id, current_time)

            return {
                'Action_Intent': annotation.action_intent.value
            }

        except Exception as e:
            logging.error(f"CSV注释数据生成失败 {agent_id}: {e}")
            return {
                'Action_Intent': 'search'
            }
    
    def get_annotation_summary(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体的注释摘要"""
        if agent_id not in self.annotation_history:
            return {'total_annotations': 0}
        
        history = self.annotation_history[agent_id]
        
        # 统计各种意图的出现次数
        intent_counts = {}
        posture_counts = {}
        
        for annotation in history:
            intent = annotation.action_intent.value
            posture = annotation.tactical_posture.value
            
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            posture_counts[posture] = posture_counts.get(posture, 0) + 1
        
        return {
            'total_annotations': len(history),
            'intent_distribution': intent_counts,
            'posture_distribution': posture_counts,
            'average_confidence': np.mean([a.confidence for a in history]) if history else 0.0
        }


# ==================== 工厂函数 ====================

def create_enhanced_annotator(enemy_ai_adapter: EnemyAIAdapter) -> EnhancedActionAnnotator:
    """创建增强行动注释器实例"""
    return EnhancedActionAnnotator(enemy_ai_adapter)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    from enemy_ai_adapter import create_drag_shoot_enemy_ai
    
    print("🏷️ 增强行动注释系统测试")
    
    # 创建敌方AI适配器和注释器
    enemy_ai = create_drag_shoot_enemy_ai()
    annotator = create_enhanced_annotator(enemy_ai)
    
    print(f"支持的行动意图: {len(ActionIntent)} 种")
    print(f"支持的战术姿态: {len(TacticalPosture)} 种")
    print(f"注释映射规则: {len(annotator.action_intent_mapping)} 条")
    
    print("✅ 增强行动注释系统测试完成")
