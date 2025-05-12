from .reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        reward = 0  # 初始化奖励值
        if env.agents[agent_id].is_shotdown:  # 检查是否被导弹击中
            reward -= 200
        elif env.agents[agent_id].is_crash:  # 检查是否意外坠毁
            reward -= 200
        for missile in env.agents[agent_id].launch_missiles:  # 检查发射的导弹是否成功击中目标
            if missile.is_success:
                reward += 200
        return self._process(reward, agent_id)  # 对奖励进行后处理，并返回结果
