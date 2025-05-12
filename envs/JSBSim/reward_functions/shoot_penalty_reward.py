from .reward_function_base import BaseRewardFunction


class ShootPenaltyReward(BaseRewardFunction):
    """
    ShootPenaltyReward
    when launching a missile, give -10 reward for penalty, 
    to avoid launching all missiles at once 
    """
    def __init__(self, config):
        super().__init__(config)

    def reset(self, task, env):
        # 初始化每个智能体的导弹数量
        self.pre_remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        # 调用父类的 reset 方法
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        根据导弹发射行为计算奖励。

        Args:
            task: 当前任务实例
            env: 环境实例
            agent_id: 智能体的唯一标识符

        Returns:
            (float): 奖励值
        """
        reward = 0  # 初始化奖励为 0

        # 检查导弹数量是否减少
        if task.remaining_missiles[agent_id] == self.pre_remaining_missiles[agent_id] - 1:
            # 如果发射了一枚导弹，扣除 10 分
            reward -= 10
        # 更新记录的导弹数量
        self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]
        # 对奖励进行后处理并返回
        return self._process(reward, agent_id)
