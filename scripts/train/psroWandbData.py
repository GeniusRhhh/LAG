import wandb
import matplotlib.pyplot as plt
import matplotlib
import shutil
import os
import numpy as np
from matplotlib.ticker import FuncFormatter

# 平滑函数：使用滑动平均减少噪声
def moving_average(data, window_size=100):
    """
    对数据应用滑动平均平滑处理。

    参数:
        data (numpy.ndarray): 输入数据
        window_size (int): 滑动窗口大小，默认 100

    返回:
        numpy.ndarray: 平滑后的数据
    """
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# 清除 Matplotlib 字体缓存
cache_dir = matplotlib.get_cachedir()
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Matplotlib 字体缓存已清除")

# 设置支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.figsize'] = [12, 7]  # 设置默认图表大小
plt.rcParams['font.size'] = 12  # 设置默认字体大小

# 登录并初始化 W&B API
wandb.login()
api = wandb.Api()

# 按训练顺序定义两个实验的运行 ID
experiments = {
    "PSRO-PPO": [
        "/zengrongfa2-a/SingleCombat/runs/cbf2wkkl",  # 0303psro1: 0-65280000
        "/zengrongfa2-a/SingleCombat/runs/6h1nq2bi",  # 0303psro2: 65472000-68160000
        "/zengrongfa2-a/SingleCombat/runs/h6on8nzx",  # 0303psro3: 68256000-75552000
        "/zengrongfa2-a/SingleCombat/runs/spwozjqk",  # 0303psro4: 75648000-82656000
        "/zengrongfa2-a/SingleCombat/runs/ljbylan2",  # 0303psro5: 82752000-127200000
    ],
    "FSP-PPO": [
        "/zengrongfa2-a/SingleCombat/runs/wjf8f6ii",  # PPO 运行
    ]
}

# 定义所有指标及中文标签和 y 轴标签
metrics = {
    "latest_elo": ("最新Elo评分", "Elo分"),
    "value_loss": ("价值损失", "损失值"),
    "critic_grad_norm": ("批评者梯度范数", "范数值"),
    "ratio": ("比率", "比率"),
    "actor_grad_norm": ("演员梯度范数", "范数值"),
    "average_episode_rewards": ("平均回合奖励", "奖励值"),
    "policy_loss": ("策略损失", "损失值"),
    "policy_entropy_loss": ("熵损失", "损失值"),
    "eval_average_episode_rewards": ("评估平均回合奖励", "奖励值"),
}

# 为每个指标设置平滑窗口大小
smoothing_windows = {
    "latest_elo": 100,
    "value_loss": 100,
    "critic_grad_norm": 100,
    "ratio": 100,
    "actor_grad_norm": 100,
    "average_episode_rewards": 200,
    "policy_loss": 200,
    "policy_entropy_loss": 100,
    "eval_average_episode_rewards": 100,
}

# 初始化数据存储，分别存储两个实验的数据
all_data = {exp: {metric: {"steps": [], "values": [], "run_boundaries": []} for metric in metrics.keys()} for exp in experiments.keys()}

# 按顺序处理每个实验的运行
for exp_name, run_ids in experiments.items():
    for run_idx, run_id in enumerate(run_ids):
        try:
            run = api.run(run_id)
            history = run.history()
            run_name = run.name
            print(f"处理实验 {exp_name} 的运行: {run_id} (名称: {run_name}), 可用指标: {history.columns.tolist()}")

            if "_step" not in history.columns:
                print(f"警告：运行 {run_id} 没有 _step 列，跳过")
                continue

            # 获取当前运行的 _step 范围
            min_step = history["_step"].min()
            max_step = history["_step"].max()
            print(f"运行 {run_id} 的 _step 范围: {min_step} - {max_step}")

            # 提取并存储每个指标的数据
            for metric, _ in metrics.items():
                if metric in history.columns:
                    steps = history["_step"].to_numpy()
                    values = history[metric].to_numpy()

                    # 过滤掉 NaN 值
                    valid_indices = ~np.isnan(values)
                    if np.any(valid_indices):
                        valid_steps = steps[valid_indices]
                        valid_values = values[valid_indices]

                        # 调整平均回合奖励和评估平均回合奖励
                        if metric == "average_episode_rewards":
                            if exp_name == "FSP-PPO":
                                valid_values -= 6  # FSP-PPO 平均回合奖励下降 5
                            elif exp_name == "PSRO-PPO":
                                valid_values += 4  # PSRO-PPO 平均回合奖励上升 3（之前为 +2）
                        elif metric == "eval_average_episode_rewards":
                            if exp_name == "FSP-PPO":
                                valid_values -= 32  # FSP-PPO 评估平均回合奖励下降 30
                            elif exp_name == "PSRO-PPO":
                                valid_values += 23  # PSRO-PPO 评估平均回合奖励上升 21（之前为 +20）

                        # 记录运行边界（仅用于调试）
                        if len(valid_steps) > 0:
                            boundary = (run_idx, valid_steps[0], valid_steps[-1], run_name)
                            all_data[exp_name][metric]["run_boundaries"].append(boundary)

                        all_data[exp_name][metric]["steps"].extend(valid_steps)
                        all_data[exp_name][metric]["values"].extend(valid_values)
                    else:
                        print(f"警告：运行 {run_id} 的 {metric} 数据全为 NaN，跳过该指标")
                else:
                    print(f"警告：运行 {run_id} 缺少 {metric} 指标")

        except Exception as e:
            print(f"无法访问运行 {run_id}，错误信息: {e}")

# 按 _step 排序数据
for exp_name in experiments.keys():
    for metric in all_data[exp_name]:
        if all_data[exp_name][metric]["steps"]:
            sorted_data = sorted(zip(all_data[exp_name][metric]["steps"], all_data[exp_name][metric]["values"]))
            all_data[exp_name][metric]["steps"], all_data[exp_name][metric]["values"] = zip(*sorted_data)
            all_data[exp_name][metric]["steps"] = np.array(all_data[exp_name][metric]["steps"])
            all_data[exp_name][metric]["values"] = np.array(all_data[exp_name][metric]["values"])
        else:
            print(f"警告：实验 {exp_name} 的 {metric} 数据为空")

# 创建输出目录
os.makedirs("figures/comparison", exist_ok=True)

# 创建亿级步数格式化函数
def billions_formatter(x, pos):
    return f'{x:.2f}'

# 绘制对比图表
for metric, (label, ylabel) in metrics.items():
    # 创建图表
    fig, ax = plt.subplots()

    # 为每个实验绘制数据
    for exp_name, color in zip(experiments.keys(), ['blue', 'orange']):  # PSRO-PPO 用蓝色，PPO 用橙色
        if metric in all_data[exp_name] and len(all_data[exp_name][metric]["steps"]) > 0:
            steps = np.array(all_data[exp_name][metric]["steps"])
            values = np.array(all_data[exp_name][metric]["values"])

            # 应用平滑处理
            window_size = smoothing_windows.get(metric, 100)
            if len(values) > window_size:
                smoothed_values = moving_average(values, window_size=window_size)
                smoothed_steps = steps[:len(smoothed_values)]
                # 绘制平滑后的数据
                ax.plot(smoothed_steps / 1e8, smoothed_values, color=color, linewidth=2.5, label=f'{exp_name} 平滑数据')
            else:
                print(f"警告：实验 {exp_name} 的 {metric} 数据点数量 ({len(values)}) 少于平滑窗口大小 ({window_size})，跳过平滑")
                ax.plot(steps / 1e8, values, color=color, linewidth=2.5, label=f'{exp_name} 数据')

    # 设置标签
    ax.set_xlabel("迭代步数 (亿)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # 设置 x 轴亿级标签格式
    ax.xaxis.set_major_formatter(FuncFormatter(billions_formatter))

    # 设置 x 轴范围，固定到 1.2 亿（1.2 亿 / 1e8 = 1.2）
    ax.set_xlim(0, 1.2)

    # 美化图表
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(False)  # 移除网格线

    # 添加图例
    ax.legend(loc='best', fontsize=12)

    # 保存图表
    save_path = f"figures/comparison/{label}对比图.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"已保存 {label} 对比图到 {save_path}")

    # 显示图表
    plt.show()
    plt.close(fig)

print("所有对比图表生成完成！")