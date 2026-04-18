from .reward_function_base import BaseRewardFunction
import numpy as np
class TemplateReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        state = task.get_state_dict(env, agent_id)
        template_id = task._last_action.get(agent_id, [0, 0])[0]
        reward = 0.0
        if template_id == 3 and state["has_warning"]:  # Notch规避导弹
            reward += 1.0
        elif template_id == 9 and abs(state["partner_angle"] - 45) < 15:  # Pincer夹角正确
            reward += 0.8
            if state["enemy_distance"] < state.get("prev_enemy_distance", np.inf):
                reward += 0.5  # 协同成功额外奖励
        elif template_id == 7 and state["enemy_distance"] > 40000:  # 超视距机动
            reward += 0.8
        elif template_id in [4, 5, 6] and state["missile_hit"]:  # 攻击命中
            reward += 2.0
        return self._process(reward, agent_id)
