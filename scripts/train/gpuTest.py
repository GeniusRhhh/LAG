import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载ELO分数数据
data = torch.load('../results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/run-20241120_125820-61h0tjgu/files/policy_pool_630.pt', map_location='cuda')

# 检查数据键
print(data.keys())

# 收集ELO分数
values = [data[str(i)] for i in range(len(data))]
# 打印所有键和对应的ELO分数
for key, value in data.items():
    print(f'Epoch {key}: ELO Score {value}')
# # 绘制图像
# plt.figure(figsize=(10, 5))
# plt.plot(values, marker='o', linestyle='-')
# plt.title('ELO分数随时间变化图')
# plt.xlabel('训练周期 (序号)')
# plt.ylabel('ELO分数')
# plt.grid(True)
# plt.show()
