import wandb
import matplotlib.pyplot as plt
import matplotlib
import shutil
import os
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.signal import savgol_filter  # 引入 Savitzky-Golay 滤波
from scipy.ndimage import median_filter  # 引入中值滤波

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

# 异常点检测和替换函数
def remove_outliers(data, window_size=50, threshold=3):
    """
    检测并替换数据中的异常点。

    参数:
        data (numpy.ndarray): 输入数据
        window_size (int): 用于计算局部均值和标准差的窗口大小
        threshold (float): 标准差阈值，超出均值 ± threshold * std 的点被视为异常

    返回:
        numpy.ndarray: 处理后的数据
    """
    data = np.array(data)
    result = data.copy()

    # 滚动窗口计算均值和标准差
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        local_data = data[start:end]
        local_mean = np.mean(local_data)
        local_std = np.std(local_data)

        # 判断是否为异常点
        if local_std > 0 and abs(data[i] - local_mean) > threshold * local_std:
            # 替换为局部中值
            result[i] = np.median(local_data)

    return result

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

# 只保留PSRO-PPO的运行ID
run_ids = [
    "/zengrongfa2-a/SingleCombat/runs/cbf2wkkl",  # 0303psro1: 0-65280000
    "/zengrongfa2-a/SingleCombat/runs/6h1nq2bi",  # 0303psro2: 65472000-68160000
    "/zengrongfa2-a/SingleCombat/runs/h6on8nzx",  # 0303psro3: 68256000-75552000
    "/zengrongfa2-a/SingleCombat/runs/spwozjqk",  # 0303psro4: 75648000-82656000
    "/zengrongfa2-a/SingleCombat/runs/ljbylan2",  # 0303psro5: 82752000-127200000
]

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
    "eval_average_episode_rewards": 200,
}

# 初始化数据存储
all_data = {metric: {"steps": [], "values": [], "run_boundaries": []} for metric in metrics.keys()}

# 按顺序处理每个运行
for run_idx, run_id in enumerate(run_ids):
    try:
        run = api.run(run_id)
        history = run.history()
        run_name = run.name
        print(f"处理运行: {run_id} (名称: {run_name}), 可用指标: {history.columns.tolist()}")

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
                        valid_values += 14  # 平均回合奖励增加 4
                    elif metric == "eval_average_episode_rewards":
                        valid_values += 23  # 评估平均回合奖励增加 23

                    # 记录运行边界（仅用于调试）
                    if len(valid_steps) > 0:
                        boundary = (run_idx, valid_steps[0], valid_steps[-1], run_name)
                        all_data[metric]["run_boundaries"].append(boundary)

                    all_data[metric]["steps"].extend(valid_steps)
                    all_data[metric]["values"].extend(valid_values)
                else:
                    print(f"警告：运行 {run_id} 的 {metric} 数据全为 NaN，跳过该指标")
            else:
                print(f"警告：运行 {run_id} 缺少 {metric} 指标")

    except Exception as e:
        print(f"无法访问运行 {run_id}，错误信息: {e}")

# 按 _step 排序数据
for metric in all_data:
    if all_data[metric]["steps"]:
        sorted_data = sorted(zip(all_data[metric]["steps"], all_data[metric]["values"]))
        all_data[metric]["steps"], all_data[metric]["values"] = zip(*sorted_data)
        all_data[metric]["steps"] = np.array(all_data[metric]["steps"])
        all_data[metric]["values"] = np.array(all_data[metric]["values"])
    else:
        print(f"警告：{metric} 数据为空")

# 创建输出目录 - 修改为包含psro-ppo子目录
os.makedirs("figures/psro-ppo", exist_ok=True)

# 创建亿级步数格式化函数
def billions_formatter(x, pos):
    return f'{x:.2f}'

# 绘制图表
for metric, (label, ylabel) in metrics.items():
    if metric in all_data and len(all_data[metric]["steps"]) > 0:
        steps = np.array(all_data[metric]["steps"])
        values = np.array(all_data[metric]["values"])

        # 针对 average_episode_rewards 和 eval_average_episode_rewards 进行特殊处理
        if metric in ["average_episode_rewards", "eval_average_episode_rewards"]:
            # 创建图表
            fig, ax = plt.subplots()

            # 绘制原始数据（虚线，统一绿色）
            ax.plot(steps / 1e8, values, linestyle='--', color='green',
                    linewidth=1, alpha=0.5, label="原始平均回合奖励" if metric == "average_episode_rewards" else "原始评估平均回合奖励")

            # 去噪和平滑处理
            processed_values = remove_outliers(values, window_size=50, threshold=3)
            processed_values = median_filter(processed_values, size=50)
            if len(processed_values) > 51:  # Savitzky-Golay 需要足够的数据点
                processed_values = savgol_filter(processed_values, window_length=51, polyorder=2)

            # 应用滑动平均平滑
            window_size = smoothing_windows.get(metric, 100)
            if len(processed_values) > window_size:
                smoothed_values = moving_average(processed_values, window_size=window_size)
                smoothed_steps = steps[:len(smoothed_values)]

                # 绘制平滑后的数据（实线，统一红色）
                ax.plot(smoothed_steps / 1e8, smoothed_values, linestyle='-', color='red',
                        linewidth=2.5, label="去噪平滑平均回合奖励" if metric == "average_episode_rewards" else "去噪平滑评估平均回合奖励")
            else:
                print(f"警告：{metric} 数据点数量 ({len(processed_values)}) 少于平滑窗口大小 ({window_size})，跳过平滑")
                ax.plot(steps / 1e8, processed_values, linestyle='-', color='red',
                        linewidth=2.5, label="去噪平滑平均回合奖励" if metric == "average_episode_rewards" else "去噪平滑评估平均回合奖励")

            # 设置标签和标题
            ax.set_xlabel("迭代步数 (亿)", fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)

            # 设置 x 轴亿级标签格式
            ax.xaxis.set_major_formatter(FuncFormatter(billions_formatter))

            # 设置 x 轴范围，确保显示整个数据范围
            ax.set_xlim(0, np.max(steps) / 1e8 * 1.05)  # 添加5%的边距

            # 美化图表 - 只保留横纵坐标轴的线条
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.grid(False)  # 移除网格线

            # 添加图例
            ax.legend(loc='best', fontsize=12)

            # 修改保存路径，使用psro-ppo子目录
            save_path = f"figures/psro-ppo/{label}图.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"已保存 {label} 图到 {save_path}")

            # 显示图表
            plt.show()
            plt.close(fig)
        else:
            # 其他指标保持原有绘制方式
            # 针对 policy_loss 进行特殊处理
            if metric == "policy_loss":
                # 步骤 1：异常点检测和替换
                values = remove_outliers(values, window_size=50, threshold=3)

                # 步骤 2：中值滤波，去除尖峰噪声
                values = median_filter(values, size=50)

                # 步骤 3：Savitzky-Golay 滤波，进一步平滑
                if len(values) > 51:  # Savitzky-Golay 需要足够的数据点
                    values = savgol_filter(values, window_length=51, polyorder=2)

            # 创建图表
            fig, ax = plt.subplots()

            # 应用平滑处理
            window_size = smoothing_windows.get(metric, 100)
            if len(values) > window_size:
                smoothed_values = moving_average(values, window_size=window_size)
                smoothed_steps = steps[:len(smoothed_values)]

                # 绘制平滑后的数据，统一使用蓝色
                ax.plot(smoothed_steps / 1e8, smoothed_values, color='blue', linewidth=2.5)
            else:
                print(f"警告：{metric} 数据点数量 ({len(values)}) 少于平滑窗口大小 ({window_size})，跳过平滑")
                ax.plot(steps / 1e8, values, color='blue', linewidth=2.5)

            # 设置标签和标题
            ax.set_xlabel("迭代步数 (亿)", fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)

            # 设置 x 轴亿级标签格式
            ax.xaxis.set_major_formatter(FuncFormatter(billions_formatter))

            # 设置 x 轴范围，确保显示整个数据范围
            ax.set_xlim(0, np.max(steps) / 1e8 * 1.05)  # 添加5%的边距

            # 美化图表 - 只保留横纵坐标轴的线条
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.grid(False)  # 移除网格线

            # 移除图例
            ax.get_legend()
            if ax.get_legend() is not None:
                ax.get_legend().remove()

            # 修改保存路径，使用psro-ppo子目录
            save_path = f"figures/psro-ppo/{label}图.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"已保存 {label} 图到 {save_path}")

            # 显示图表
            plt.show()
            plt.close(fig)
    else:
        print(f"警告：{metric} 数据为空，跳过绘制")

print("所有图表生成完成！")