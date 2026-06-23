import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from envs.JSBSim.envs import SingleCombatEnv
from algorithms.ppo.ppo_actor import PPOActor
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

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

def run_combat(env, ego_policy, enm_policy, ego_model_id, enm_model_id, seed=0):
    """
    运行单次对战实验（PPO vs PPO），返回对战结果（成功、失败、平局）

    参数:
        env: 空战环境
        ego_policy: 红方模型
        enm_policy: 蓝方模型（基线模型）
        ego_model_id: 红方模型编号
        enm_model_id: 蓝方模型编号
        seed: 随机种子

    返回:
        result: 单次对战的结果（包含胜负结果和其他信息）
    """
    logger.info(f"开始对战: 红方模型 {ego_model_id} vs 蓝方模型 {enm_model_id}")
    num_agents = 2
    env.seed(seed)
    obs = env.reset()

    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    episode_rewards_ego = 0
    episode_rewards_enm = 0
    recent_rewards_ego = []
    recent_rewards_enm = []

    while True:
        enm_obs = obs[num_agents // 2:, :]
        ego_obs = obs[:num_agents // 2, :]

        # 获取动作
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        ego_actions = _t2n(ego_actions)
        ego_rnn_states = _t2n(ego_rnn_states)

        enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
        enm_actions = _t2n(enm_actions)
        enm_rnn_states = _t2n(enm_rnn_states)

        actions = np.concatenate((ego_actions, enm_actions), axis=0)
        obs, rewards, dones, infos = env.step(actions)

        # 累积奖励
        rewards_ego = rewards[:num_agents // 2, ...]
        rewards_enm = rewards[num_agents // 2:, ...]
        episode_rewards_ego += rewards_ego
        episode_rewards_enm += rewards_enm

        # 记录最近 5 步的奖励
        if len(recent_rewards_ego) >= 5:
            recent_rewards_ego.pop(0)
            recent_rewards_enm.pop(0)
        recent_rewards_ego.append(float(rewards_ego.sum()))
        recent_rewards_enm.append(float(rewards_enm.sum()))

        if dones.all():
            # 计算总奖励和最近奖励差
            ego_total_reward = episode_rewards_ego.sum()
            enm_total_reward = episode_rewards_enm.sum()
            reward_diff = ego_total_reward - enm_total_reward
            ego_recent_avg = np.mean(recent_rewards_ego) if recent_rewards_ego else 0
            enm_recent_avg = np.mean(recent_rewards_enm) if recent_rewards_enm else 0
            recent_diff = ego_recent_avg - enm_recent_avg

            # 胜负判断逻辑（基于奖励差和最近奖励差）
            reward_threshold = 10.0
            recent_threshold = 2.0
            outcome = "平局"
            if reward_diff > reward_threshold or recent_diff > recent_threshold:
                outcome = '成功'
            elif reward_diff < -reward_threshold or recent_diff < -recent_threshold:
                outcome = '失败'
            else:
                outcome = '平局'

            # 记录结果
            result = {
                '胜负结果': outcome,
                'ego_reward': ego_total_reward,
                'enm_reward': enm_total_reward,
                'reward_diff': reward_diff,
                'recent_diff': recent_diff
            }

            # 调试输出
            logger.debug(f"对战结果: 红方模型 {ego_model_id} vs 蓝方模型 {enm_model_id}, "
                         f"总奖励 (我方={ego_total_reward:.2f}, 敌方={enm_total_reward:.2f}), "
                         f"奖励差={reward_diff:.2f}, 最近奖励差={recent_diff:.2f}, "
                         f"胜负结果={outcome}, infos={infos}")

            return [result]

def process_combat_data(df):
    """
    处理对战数据，按回合范围分组并计算胜率、败率和平局率，以及具体数量
    """
    df['基线模型轮次'] = df['基线模型轮次'].astype(int)

    # 定义回合范围（0-714 号，分段）
    ranges = [(i, i + 100) for i in range(0, 715, 100)]  # 每 100 轮一个范围

    results = []
    for start, end in ranges:
        range_df = df[(df['基线模型轮次'] >= start) & (df['基线模型轮次'] < end)]
        if len(range_df) > 0:
            total_games = len(range_df)
            wins = sum(range_df['胜负结果'] == '成功')
            losses = sum(range_df['胜负结果'] == '失败')
            draws = sum(range_df['胜负结果'] == '平局')

            win_rate = (wins / total_games) * 100
            lose_rate = (losses / total_games) * 100
            draw_rate = (draws / total_games) * 100

            results.append({
                '回合范围': f"{start}-{end}",
                '对战场次': total_games,
                '胜场数': wins,
                '负场数': losses,
                '平局数': draws,
                '红方胜率': win_rate,
                '红方败率': lose_rate,
                '平局率': draw_rate
            })

    summary_df = pd.DataFrame(results)
    summary_df['红方胜率'] = summary_df['红方胜率'].apply(lambda x: f"{x:.1f}%")
    summary_df['红方败率'] = summary_df['红方败率'].apply(lambda x: f"{x:.1f}%")
    summary_df['平局率'] = summary_df['平局率'].apply(lambda x: f"{x:.1f}%")
    return summary_df

def create_visualization(summary_df, title, output_file):
    """
    创建可视化图表（展示胜率、败率和平局率）
    """
    summary_df['红方胜率_num'] = summary_df['红方胜率'].str.rstrip('%').astype(float)
    summary_df['红方败率_num'] = summary_df['红方败率'].str.rstrip('%').astype(float)
    summary_df['平局率_num'] = summary_df['平局率'].str.rstrip('%').astype(float)

    plt.figure(figsize=(12, 8))
    x = range(len(summary_df))
    width = 0.25

    plt.bar([i - width for i in x], summary_df['红方胜率_num'], width=width, label='红方胜率', color='red')
    plt.bar(x, summary_df['红方败率_num'], width=width, label='红方败率', color='blue')
    plt.bar([i + width for i in x], summary_df['平局率_num'], width=width, label='平局率', color='gray')

    plt.xlabel('基线模型轮次范围')
    plt.ylabel('百分比')
    plt.title(title)
    plt.xticks(x, [f"{r} 轮次" for r in summary_df['回合范围']])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

def save_results_to_excel(summary_df, file_path):
    """
    将结果保存到 Excel 文件
    """
    try:
        summary_df.to_excel(file_path, sheet_name='胜率统计', index=False)
        logger.info(f"结果已保存至 {file_path}")
    except Exception as e:
        logger.error(f"保存 Excel 文件时出错: {e}")
        summary_df.to_csv(file_path.replace('.xlsx', '.csv'), index=False, encoding='utf-8-sig')
        logger.info(f"已保存为 CSV: {file_path.replace('.xlsx', '.csv')}")

def main():
    # 环境和模型路径
    env_name = "1v1/NoWeapon/HierarchySelfplay"  # 使用 HierarchySelfplay 环境
    fsp_ppo_dir = "../results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/wandb/run-20241218_214259-wjf8f6ii/files"  # FSP-PPO 模型路径
    psro_ppo_dir = "../results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/0311PSRO"  # PSRO-PPO 模型路径
    baseline_dir = "../results/SingleCombat/1v1/NoWeapon/vsBaseline/ppo/v1/some_run_id"  # 基线模型路径（需要你提供）

    # FSP-PPO 和 PSRO-PPO 的最新模型
    fsp_ppo_latest_iter = 1040  # FSP-PPO 最新模型
    psro_ppo_latest_iter = 1323  # PSRO-PPO 最新模型
    baseline_max_iter = 714  # 基线模型 0-714 号

    # 初始化环境
    env = SingleCombatEnv(env_name)
    logger.info(f"动作空间: {env.action_space}")
    args = Args()

    # 确保使用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载 FSP-PPO 最新模型
    fsp_model_path = f"{fsp_ppo_dir}/actor_{fsp_ppo_latest_iter}.pt"
    if not os.path.exists(fsp_model_path):
        logger.error(f"FSP-PPO 最新模型 {fsp_model_path} 不存在！")
        return
    fsp_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
    fsp_policy.load_state_dict(torch.load(fsp_model_path, map_location=device, weights_only=True))
    fsp_policy.eval()
    logger.info(f"已加载 FSP-PPO 最新模型: {fsp_model_path}")

    # 加载 PSRO-PPO 最新模型
    psro_model_path = f"{psro_ppo_dir}/actor_{psro_ppo_latest_iter}.pt"
    if not os.path.exists(psro_model_path):
        logger.error(f"PSRO-PPO 最新模型 {psro_model_path} 不存在！")
        return
    psro_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
    psro_policy.load_state_dict(torch.load(psro_model_path, map_location=device, weights_only=True))
    psro_policy.eval()
    logger.info(f"已加载 PSRO-PPO 最新模型: {psro_model_path}")

    # 1. FSP-PPO 最新模型对打基线模型 0-714 号
    logger.info("\n开始 FSP-PPO 最新模型对基线模型 0-714 号的对抗...")
    fsp_vs_baseline_results = []

    for iter in tqdm(range(baseline_max_iter + 1), desc="FSP-PPO vs Baseline"):
        baseline_model_path = f"{baseline_dir}/actor_{iter}.pt"
        if not os.path.exists(baseline_model_path):
            logger.warning(f"基线模型 {iter} 号不存在，跳过...")
            continue

        # 加载基线模型
        baseline_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
        try:
            baseline_policy.load_state_dict(torch.load(baseline_model_path, map_location=device, weights_only=True))
        except RuntimeError as e:
            logger.error(f"加载基线模型 {baseline_model_path} 失败: {e}")
            continue
        baseline_policy.eval()

        # 运行对战（只打 1 次）
        results = run_combat(env, fsp_policy, baseline_policy, ego_model_id=f'FSP-PPO_{fsp_ppo_latest_iter}', enm_model_id=f'Baseline_{iter}', seed=iter)
        for result in results:
            result['基线模型轮次'] = iter
            result['红方模型'] = f'FSP-PPO_{fsp_ppo_latest_iter}'
            result['蓝方模型'] = f'Baseline_{iter}'
        fsp_vs_baseline_results.extend(results)

    # 转换为 DataFrame 并处理
    fsp_vs_baseline_df = pd.DataFrame(fsp_vs_baseline_results)
    fsp_vs_baseline_summary = process_combat_data(fsp_vs_baseline_df)
    logger.info("\nFSP-PPO 最新模型对基线模型 0-714 号的胜率统计:")
    logger.info(fsp_vs_baseline_summary.to_string(index=False))
    create_visualization(fsp_vs_baseline_summary, "FSP-PPO 最新模型对基线模型 0-714 号的胜率统计", "fsp_vs_baseline_statistics.png")
    save_results_to_excel(fsp_vs_baseline_summary, "fsp_vs_baseline_results.xlsx")

    # 2. PSRO-PPO 最新模型对打基线模型 0-714 号
    logger.info("\n开始 PSRO-PPO 最新模型对基线模型 0-714 号的对抗...")
    psro_vs_baseline_results = []

    for iter in tqdm(range(baseline_max_iter + 1), desc="PSRO-PPO vs Baseline"):
        baseline_model_path = f"{baseline_dir}/actor_{iter}.pt"
        if not os.path.exists(baseline_model_path):
            logger.warning(f"基线模型 {iter} 号不存在，跳过...")
            continue

        # 加载基线模型
        baseline_policy = PPOActor(args, env.observation_space, env.action_space, device=device)
        try:
            baseline_policy.load_state_dict(torch.load(baseline_model_path, map_location=device, weights_only=True))
        except RuntimeError as e:
            logger.error(f"加载基线模型 {baseline_model_path} 失败: {e}")
            continue
        baseline_policy.eval()

        # 运行对战（只打 1 次）
        results = run_combat(env, psro_policy, baseline_policy, ego_model_id=f'PSRO-PPO_{psro_ppo_latest_iter}', enm_model_id=f'Baseline_{iter}', seed=iter)
        for result in results:
            result['基线模型轮次'] = iter
            result['红方模型'] = f'PSRO-PPO_{psro_ppo_latest_iter}'
            result['蓝方模型'] = f'Baseline_{iter}'
        psro_vs_baseline_results.extend(results)

    # 转换为 DataFrame 并处理
    psro_vs_baseline_df = pd.DataFrame(psro_vs_baseline_results)
    psro_vs_baseline_summary = process_combat_data(psro_vs_baseline_df)
    logger.info("\nPSRO-PPO 最新模型对基线模型 0-714 号的胜率统计:")
    logger.info(psro_vs_baseline_summary.to_string(index=False))
    create_visualization(psro_vs_baseline_summary, "PSRO-PPO 最新模型对基线模型 0-714 号的胜率统计", "psro_vs_baseline_statistics.png")
    save_results_to_excel(psro_vs_baseline_summary, "psro_vs_baseline_results.xlsx")

if __name__ == "__main__":
    main()