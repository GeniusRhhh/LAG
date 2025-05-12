import wandb
import matplotlib.pyplot as plt
import matplotlib
import shutil
import os
import numpy as np


# 平滑函数：使用窗口滑动平均法减少噪声
def moving_average(data, window_size=100):
    """
    对数据应用滑动平均平滑处理。

    参数:
        data (numpy.ndarray): 输入数据
        window_size (int): 滑动窗口大小，默认 100

    返回:
        numpy.ndarray: 平滑后的数据
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# 清除 Matplotlib 字体缓存
cache_dir = matplotlib.get_cachedir()
print(f"Matplotlib 缓存目录：{cache_dir}")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("缓存已清除，Matplotlib 将重新加载字体")

# 设置支持中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 优先尝试 SimHei
    print("成功设置字体为 SimHei")
except Exception as e:
    print(f"SimHei 字体不可用，错误信息: {e}")
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 备用字体
    print("切换到 Microsoft YaHei 字体")
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 登录并初始化 API
wandb.login()
api = wandb.Api()

# run = api.run("/zengrongfa2-a/SingleCombat/runs/cbf2wkkl")
# run = api.run("/zengrongfa2-a/SingleCombat/runs/6h1nq2bi")
# run = api.run("/zengrongfa2-a/SingleCombat/runs/h6on8nzx")
# run = api.run("/zengrongfa2-a/SingleCombat/runs/spwozjqk")
# run = api.run("/zengrongfa2-a/SingleCombat/runs/ljbylan2")

# 指定项目和运行
try:
    run = api.run("/zengrongfa2-a/SingleCombat/runs/jntzveiw")
    run = api.run("/zengrongfa2-a/SingleCombat/runs/jntzveiw")
except Exception as e:
    print(f"无法访问运行数据，错误信息: {e}")
    exit(1)

# 获取历史数据
history = run.history()

# 打印可用指标，确认数据
print("可用指标：", history.columns.tolist())

# 定义需要绘制的指标及其中文标签和 y 轴标签
metrics = {
    "ratio": ("概率比率", "比率"),
    "value_loss": ("价值损失", "损失"),
    "critic_grad_norm": ("Critic梯度范数", "梯度范数"),
    "policy_loss": ("策略损失", "损失"),
    "actor_grad_norm": ("Actor梯度范数", "梯度范数"),
    "policy_entropy_loss": ("熵损失", "损失"),
    "average_episode_rewards": ("平均回合奖励", "奖励"),
    "eval_average_episode_rewards": ("评估平均回合奖励", "奖励"),
}

# 绘制所有指标的图表
for metric, (label, ylabel) in metrics.items():
    if metric in history.columns:
        # 提取数据并处理缺失值
        data = history[["_step", metric]].dropna()
        if data.empty:
            print(f"警告：{label} 数据为空，跳过绘制")
            continue

        steps = data["_step"].to_numpy()  # 转换为 numpy 数组
        values = data[metric].to_numpy()  # 转换为 numpy 数组

        # 仅对策略损失应用平滑处理
        if metric == "policy_loss":
            window_size = 350  # 默认窗口大小，可根据需要调整
            smoothed_values = moving_average(values, window_size=window_size)
            smoothed_steps = steps[:len(smoothed_values)] / 1e8  # 转换为 1e8 单位
            plot_steps = smoothed_steps
            plot_values = smoothed_values
        else:
            plot_steps = steps / 1e8  # 转换为 1e8 单位
            plot_values = values

        # 创建专业风格的图表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(plot_steps, plot_values, color="blue", linewidth=2)  # 改为 blue 颜色

        # 设置坐标轴标签
        ax.set_xlabel("迭代步数 (x1e8)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # 移除顶部和右侧边框，只保留左边和下边
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)  # 加粗左边线
        ax.spines['bottom'].set_linewidth(1.5)  # 加粗下边线

        # 移除网格线
        ax.grid(False)

        # 调整布局
        plt.tight_layout()

        # 保存图表
        save_path = f"figures/{label}图.png"
        os.makedirs("figures", exist_ok=True)  # 创建 figures 目录
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"已保存 {label} 图到 {save_path}")

        # 显示图表
        plt.show()

        # 关闭当前图表，释放内存
        plt.close(fig)