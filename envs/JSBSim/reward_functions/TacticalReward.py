import numpy as np
from ..core.catalog import Catalog as c
from .reward_function_base import BaseRewardFunction
from ..tasks.TacticalTemplate import TacticalTemplate

class TacticalReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.reward_items = {
            "contact_guidance": 0.3,      # 接近发射条件
            "radar_lock": 0.5,            # 目标搜索成功
            "target_identified": 0.4,     # 目标识别
            "threat_assessed": 0.3,       # 威胁判断
            "target_allocated": 0.6,      # 目标分配
            "decision_made": 0.5,         # 战术决策
            "missile_launch": 2.0,        # 发射导弹
            "mid_guidance_success": 0.7,  # 中距制导
            "terminal_guidance": 0.8,     # 末制导
            "hit_success": 5.0,           # 命中目标
            "evasion_success": 0.8,       # 规避敌方导弹
            "cooperative_success": 0.7,   # 协同作战
            "long_range_engagement": 1.0,  # 新增超视距阶段
            "distance_penalty": -0.05    # 距离惩罚
        }
        self.phase_rewards = {phase: 0 for phase in TacticalTemplate.PHASES}

    def get_reward(self, task, env, agent_id):
        reward = 0
        state = task.get_state_dict(env, agent_id)
        current_phase = task.tactical_templates[agent_id].current_phase
        phase_rewards = {
            "contact_guidance": 0.8 if state["enemy_distance"] < 100000 else 0,  # MODIFIED: 提高奖励
            "target_search": 1.0 if state["radar_lock"] else 0,  # MODIFIED: 提高奖励
            "target_identification": 0.8 if state["radar_lock"] else 0,  # MODIFIED: 提高奖励
            "threat_assessment": 0.6 if state["has_warning"] else 0,
            "target_allocation": 1.2 if state["targets_assigned"] else 0,  # MODIFIED: 提高奖励
            "tactical_decision": 1.0 if state["attack_decided"] else 0,  # MODIFIED: 提高奖励
            "missile_launch": 3.0 if state["missile_launched"] else 0,  # MODIFIED: 提高奖励
            "mid_guidance_defense": 1.2 if state["missile_active"] else 0,
            "terminal_guidance": 1.5 if state["missile_active"] else 0,  # MODIFIED: 提高奖励
            "effect_assessment": 6.0 if state["missile_hit"] else 0,  # MODIFIED: 提高奖励
            "long_range_engagement": 3.0 if 55000 < state["enemy_distance"] <= 80000 and state["radar_lock"] else 0
            # MODIFIED: 放宽距离，增强奖励
        }
        reward += phase_rewards.get(current_phase, 0)
        if state["is_leader"] and abs(state["partner_angle"] - 45) < 15:
            reward += 1.0  # MODIFIED: 提高协同奖励
        if state["has_warning"] and state["enemy_distance"] > 20000:
            reward += 0.8  # MODIFIED: 提高规避奖励
        reward += -0.005 * (
            state["enemy_distance"] / 1000 if state["enemy_distance"] > 60000 else 0)  # MODIFIED: 减弱距离惩罚
        self.phase_rewards[current_phase] += reward
        return self._process(reward, agent_id)