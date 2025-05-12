import numpy as np
import torch
import logging
from datetime import datetime
from envs.JSBSim.envs import SingleControlEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.sac.sac_actor import ActorNet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
def main():
    # 创建 Heading Task 环境
    env_name = "SingleControl"
    scenario = "1/heading"
    env = SingleControlEnv(scenario)
    env.seed(42)

    # 加载 SAC 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DummyArgs:
        hidden_size = '128 128'  # 与 train_heading.sh 一致
        act_hidden_size = '128 128'  # 与 train_heading.sh 一致
        activation_id = 0  # 与 train_heading.sh 一致（Tanh）
        use_feature_normalization = True  # 修改为 True，与训练时一致
        use_recurrent_policy = False  # 与 train_heading.sh 一致
        recurrent_hidden_size = 128
        recurrent_hidden_layers = 1
        init_alpha = 1.0  # 与 train_heading.sh 一致
        target_entropy = -1.0  # 与 train_heading.sh 一致
        gamma = 0.99
        tau = 0.005
        gain = 0.01
        cuda = True
        actor_lr = 1e-4  # 与 train_heading.sh 一致
        critic_lr = 1e-4
        alpha_lr = 1e-4
        max_grad_norm = 2  # 与 train_heading.sh 一致

    args = DummyArgs()

    obs_space = env.observation_space
    act_space = env.action_space
    actor = ActorNet(args, obs_space, act_space, device=device)

    ego_run_dir = "../scripts/results/SingleControl/1/heading/sac/v1/run48"
    checkpoint_path = f"{ego_run_dir}/actor_160.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        actor.load_state_dict(checkpoint)
        logging.info(f"Loaded actor model from {checkpoint_path}")
    except FileNotFoundError:
        logging.error(f"Checkpoint file {checkpoint_path} not found.")
        return
    except RuntimeError as e:
        logging.error(f"Failed to load checkpoint: {e}")
        return
    actor.eval()

    # 重置并强制设置初始高度
    obs = env.reset()
    for agent_id, sim in env._jsbsims.items():
        sim.set_property_value(c.position_h_sl_ft, 20000.0)
        logging.info(f"Set {agent_id} altitude to: {sim.get_property_value(c.position_h_sl_ft)} ft")
    obs = env.reset()  # 再次重置以应用

    # 打印初始配置和高度
    logging.info(f"Scenario config: {env.init_states}")
    logging.info(f"Initial observation: {obs}")
    for agent_id, sim in env._jsbsims.items():
        actual_altitude = sim.get_property_value(c.position_h_sl_ft)
        logging.info(f"Actual {agent_id} altitude: {actual_altitude} ft")
    logging.info(f"Observation altitude (idx=3): {obs[0, 3] * 5000 if obs.ndim > 1 else obs[3] * 5000} ft")

    # 初始化 RNN 状态和 masks
    rnn_states = None
    if args.use_recurrent_policy:
        rnn_states = torch.zeros(
            (1, args.recurrent_hidden_layers, args.recurrent_hidden_size),
            dtype=torch.float32, device=device
        )
    masks = torch.ones((1, 1), dtype=torch.float32, device=device)

    # 推理并渲染
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    acmi_path = f"{env_name}_{scenario.replace('/', '_')}_{current_time}.acmi"
    env.render(mode='txt', filepath=acmi_path)

    done = False
    episode_reward = 0.0
    step = 0

    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            actions, _, new_rnn_states = actor(obs_tensor, rnn_states=rnn_states, masks=masks, deterministic=True)
        actions_np = actions.cpu().numpy().squeeze(0)

        next_obs, reward, done, info = env.step(actions_np)
        episode_reward += reward

        # 打印每步高度
        for agent_id, sim in env._jsbsims.items():
            actual_altitude = sim.get_property_value(c.position_h_sl_ft)
            logging.info(f"Step {step} - Actual {agent_id} altitude: {actual_altitude} ft")
        # logging.info(f"Step {step} - Observation altitude (idx=3): {next_obs[0, 3] * 5000 if next_obs.ndim > 1 else next_obs[3] * 5000} ft")

        env.render(mode='txt', filepath=acmi_path)
        obs = next_obs
        rnn_states = new_rnn_states  # 更新 RNN 状态
        step += 1

    logging.info(f"Episode finished, total reward={episode_reward}")
    logging.info(f"ACMI file saved to: {acmi_path}")
if __name__ == "__main__":
    main()