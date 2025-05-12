
import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
from datetime import datetime

# ====== 新增导入，用于分析和绘图 ======
import os
import csv
import matplotlib
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)


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


def _t2n(x):
    return x.detach().cpu().numpy()


# ====== 新增函数，用于计算AO、TA、距离 ======
def compute_AO_TA_distance(env):
    """
    根据当前env状态计算AO、TA和距离。
    假设env为SingleCombatEnv场景，提取红方(ego)和蓝方(enm)的状态计算相对信息。
    """
    agent_ids = list(env.agents.keys())
    if len(agent_ids) < 2:
        return np.nan, np.nan, np.nan

    ego_id, enm_id = agent_ids[0], agent_ids[1]

    # 从env中提取状态变量(与singlecombat_task中一致)
    state_var = [
        c.position_long_gc_deg,
        c.position_lat_geod_deg,
        c.position_h_sl_m,
        c.attitude_roll_rad,
        c.attitude_pitch_rad,
        c.attitude_heading_true_rad,
        c.velocities_v_north_mps,
        c.velocities_v_east_mps,
        c.velocities_v_down_mps,
        c.velocities_u_mps,
        c.velocities_v_mps,
        c.velocities_w_mps,
        c.velocities_vc_mps
    ]

    ego_obs_list = np.array(env.agents[ego_id].get_property_values(state_var))
    enm_obs_list = np.array(env.agents[enm_id].get_property_values(state_var))

    from envs.JSBSim.utils.utils import LLA2NEU, get2d_AO_TA_R

    ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
    enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)

    ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
    enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])

    AO1, TA1, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
    AO = np.degrees(AO1)
    TA = np.degrees(TA1)
    return AO, TA, R


num_agents = 2
render = True
ego_policy_index = 1040
enm_policy_index = 215
episode_rewards = 0
ego_run_dir = "../scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/20241212_0800"
enm_run_dir = "../scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/20241212_0800"
experiment_name = ego_run_dir.split('/')[-4]

# env_name = "1v1/ShootMissile/HierarchySelfplay"
env_name="1v1/NoWeapon/Selfplay"
env = SingleCombatEnv(env_name)
env.seed(0)
args = Args()

# 确保使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型并移动到设备（GPU 或 CPU）
ego_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=device)

ego_policy.eval()
enm_policy.eval()

ego_policy.load_state_dict(torch.load(f"{ego_run_dir}/actor_{ego_policy_index}.pt", map_location=device))
enm_policy.load_state_dict(torch.load(f"{enm_run_dir}/actor_{enm_policy_index}.pt", map_location=device))
print("Start render")
obs = env.reset()
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
if render:
    filepath = f'{env_name.replace("/", "_")}_{current_time}_{ego_policy_index}_{enm_policy_index}.txt.acmi'
    env.render(mode='txt', filepath=filepath)

ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1))
enm_obs = obs[num_agents // 2:, :]
ego_obs = obs[:num_agents // 2, :]
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)

# ====== 新增：创建CSV文件记录数据 ======
csv_filename = f"analysis_data_{current_time}_{ego_policy_index}_{enm_policy_index}_{episode_rewards}.csv"
acmi_filename = f"{env_name.replace('/', '_')}_{current_time}_{ego_policy_index}_{enm_policy_index}_{episode_rewards}.txt.acmi"
with open(csv_filename, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 表头中文描述
    writer.writerow(["步数", "AO(角度偏差)", "TA(目标角度)", "距离(m)", "奖励", "动作_ego", "动作_enm"])

step_count = 0

while True:
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)
    actions = np.concatenate((ego_actions, enm_actions), axis=0)
    # Obser reward and next obs
    obs, rewards, dones, infos = env.step(actions)
    print(f"Actions: {actions}, Rewards: {rewards}")
    rewards_ego = rewards[:num_agents // 2, ...]
    episode_rewards += rewards_ego
    if render:
        env.render(mode='txt', filepath=filepath)

    # ====== 新增：每步计算AO、TA、distance并写入CSV ======
    AO, TA, dist = compute_AO_TA_distance(env)
    with open(csv_filename, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 将动作、奖励和AO/TA/dist记录
        # 动作是二维的，这里只打印ego和enm的动作数组
        writer.writerow([step_count, AO, TA, dist, rewards_ego.item(), ego_actions.tolist(), enm_actions.tolist()])

    step_count += 1

    if dones.all():
        print(infos)
        break
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    print(f"step:{env.current_step}, bloods:{bloods}")
    enm_obs = obs[num_agents // 2:, ...]
    ego_obs = obs[:num_agents // 2, ...]

print("render episode reward of agent:", episode_rewards)
final_reward = episode_rewards.item()  # 使用.item()从1x1数组中提取数值

final_reward_formatted = f"{final_reward:.4f}".replace('.', '')
updated_csv_filename = csv_filename.replace('_0.csv', f'_{final_reward_formatted}.csv')
updated_acmi_filename = acmi_filename.replace('_0.txt.acmi', f'_{final_reward_formatted}.txt.acmi')

# ====== 新增：对CSV数据进行分析和绘图 ======
# 离线分析AO、TA、distance数据并画图（中文图例）
import pandas as pd

df = pd.read_csv(csv_filename, encoding='utf-8')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10,6))
plt.plot(df['步数'].values, df['AO(角度偏差)'].values, label='AO(角度偏差)')
plt.plot(df['步数'].values, df['TA(目标角度)'].values, label='TA(目标角度)')
plt.plot(df['步数'].values, df['距离(m)'].values, label='距离')

plt.xlabel("时间步数")
plt.ylabel("数值")
plt.title("无武器场景下AO、TA和距离随时间变化图")
plt.legend()
plt.tight_layout()
png_filename = f"metrics_over_time_{current_time}_{ego_policy_index}_{enm_policy_index}_{final_reward_formatted}.png"
plt.savefig(png_filename, dpi=300)
print(f"已生成图表: {png_filename}")

print("分析完成！请查看生成的CSV和PNG文件，以及acmi文件进行综合评估。")

