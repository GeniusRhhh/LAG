import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv
from algorithms.ppo.ppo_actor import PPOActor


class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cuda'))
        self.use_prior = True


def evaluate_one_match(ego_model_path, enm_model_path, scenario="1v1/NoWeapon/HierarchySelfplay") -> float:
    """
    让 ego_model 与 enm_model 对战 1 次，返回: ego_model 的收益
    """
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SingleCombatEnv(scenario)
    env.seed(0)

    # 加载两个模型
    ego_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
    enm_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
    ego_policy.load_state_dict(torch.load(ego_model_path, map_location=device))
    enm_policy.load_state_dict(torch.load(enm_model_path, map_location=device))

    ego_policy.eval()
    enm_policy.eval()

    obs = env.reset()
    # RNN初始化
    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    enm_rnn_states = np.zeros_like(ego_rnn_states)
    masks = np.ones((1, 1), dtype=np.float32)

    sum_ego = 0.0
    while True:
        ego_obs = obs[0:1]
        enm_obs = obs[1:2]

        with torch.no_grad():
            ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
            enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)

        actions = np.concatenate([ego_actions.cpu().numpy(), enm_actions.cpu().numpy()], axis=0)
        obs, rewards, dones, infos = env.step(actions)

        sum_ego += float(rewards[0])  # ego是 0号位
        if dones.all():
            break

    env.close()
    return sum_ego  # 返回单场收益


def main():
    # 模型列表
    models = [
        "../results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/20241218_2141/actor_650.pt",
        "../results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/20241218_2141/actor_630.pt",
        "../results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/20241218_2141/actor_661.pt",
        "../results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/20241218_2141/actor_1040.pt",
        "../results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/20241218_2141/actor_35.pt",
        "../results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/20241218_2141/actor_84.pt",
        "../results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/20241218_2141/actor_0.pt"
    ]

    # 分数统计
    scores = {m: 0.0 for m in models}
    matches = {m: 0 for m in models}
    win_count = {m: 0 for m in models}  # 每个模型的胜场次数

    print("\n=== Match Results ===")
    # 两两对战: (i vs j) + (j vs i)
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue
            ego_model = models[i]
            enm_model = models[j]

            # 返回 ego 的收益
            rew_ego = evaluate_one_match(ego_model, enm_model)
            print(f"Match: [{ego_model}] (Ego) vs [{enm_model}] (Enemy) -> Ego Reward: {rew_ego:.2f}")

            # 更新统计数据
            scores[ego_model] += rew_ego
            matches[ego_model] += 1

            # 记录胜负情况
            if rew_ego > 0:
                win_count[ego_model] += 1

    # 计算平均
    print("\n=== Average Scores ===")
    for m in models:
        if matches[m] > 0:
            scores[m] = float(scores[m] / matches[m])  # 转换为标量
        else:
            scores[m] = -9999
        print(f"{m}: Average Score: {scores[m]:.2f}, Wins: {win_count[m]}/{matches[m]}")

    # 找出最佳
    best_model = max(scores, key=lambda x: scores[x])

    print("\n=== Final Scores ===")
    for m in models:
        print(f"{m}: {scores[m]:.2f}")
    print(f"\nBest model is [{best_model}] with average score {scores[best_model]:.2f}")


if __name__ == "__main__":
    main()
