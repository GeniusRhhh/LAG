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

def evaluate_one_match(ego_model_path, enm_model_path, scenario="1v1/NoWeapon/HierarchySelfplay", episodes=3) -> float:
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建环境
    env = SingleCombatEnv(scenario)
    env.seed(0)

    # 加载两个模型
    ego_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
    enm_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
    ego_policy.load_state_dict(torch.load(ego_model_path, map_location=device))
    enm_policy.load_state_dict(torch.load(enm_model_path, map_location=device))

    ego_policy.eval()
    enm_policy.eval()

    total_reward = 0.0

    for ep in range(episodes):
        obs = env.reset()
        step_rewards = 0.0

        # 初始化 RNN 状态
        ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        enm_rnn_states = np.zeros_like(ego_rnn_states)
        masks = np.ones((1, 1), dtype=np.float32)

        while True:
            enm_obs = obs[1:2, :]
            ego_obs = obs[0:1, :]

            with torch.no_grad():
                ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
                enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
            actions = np.concatenate((ego_actions.cpu().numpy(), enm_actions.cpu().numpy()), axis=0)

            obs, rewards, dones, infos = env.step(actions)
            step_rewards += rewards[0]
            if dones.all():
                break

        total_reward += step_rewards

    env.close()
    avg_reward = total_reward / episodes
    return avg_reward

def test_deterministic():
    ego_model = "../results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/20241218_2141/actor_650.pt"
    enm_model = "../results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/20241218_2141/actor_630.pt"
    scenario = "1v1/NoWeapon/HierarchySelfplay"

    runs = 5
    results = []
    for _ in range(runs):
        avg_reward = evaluate_one_match(ego_model, enm_model, scenario=scenario, episodes=1)
        results.append(float(avg_reward))

    print(f"Results over {runs} runs: {results}")
    if len(set(results)) == 1:
        print("The environment is deterministic.")
    else:
        print("The environment is not deterministic!")

if __name__ == "__main__":
    test_deterministic()
