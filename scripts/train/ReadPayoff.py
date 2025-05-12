import numpy as np

# 读取 .npy 文件
payoff_data = np.load('../../scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/v1/03031629/payoff_matrix.npy')

# 查看数据内容
print(payoff_data)

# 查看数据形状
print("数据形状:", payoff_data.shape)

# 查看数据类型
print("数据类型:", payoff_data.dtype)