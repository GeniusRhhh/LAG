import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 修改为正确的编码
csv_filename= '../../renders/20241222/analysis_data_20241226_164122_650_84.csv'

try:
    df = pd.read_csv(csv_filename, encoding='utf-8-sig')
except UnicodeDecodeError:
    print("尝试其他编码读取...")
    df = pd.read_csv(csv_filename, encoding='gbk', errors='replace')

time = df['时间(秒)'].to_numpy()  # 转换为 NumPy 数组
steps = df.shape[0]

plt.figure(figsize=(14, 10))

# 确保在所有绘图中转换为 NumPy 数组
plt.subplot(3, 3, 1)
plt.plot(time, df['AO(度)'].to_numpy(), label='AO(度)')
plt.xlabel('时间(秒)')
plt.ylabel('AO(度)')
plt.title('AO随时间变化')
plt.legend()

plt.subplot(3, 3, 2)
plt.plot(time, df['TA(度)'].to_numpy(), label='TA(度)', color='r')
plt.xlabel('时间(秒)')
plt.ylabel('TA(度)')
plt.title('TA随时间变化')
plt.legend()

plt.subplot(3, 3, 3)
plt.plot(time, df['距离(m)'].to_numpy(), label='距离(m)', color='g')
plt.xlabel('时间(秒)')
plt.ylabel('距离(m)')
plt.title('距离随时间变化')
plt.legend()

plt.subplot(3, 3, 4)
plt.plot(time, df['奖励'].to_numpy(), label='奖励', color='purple')
plt.xlabel('时间(秒)')
plt.ylabel('奖励')
plt.title('奖励随时间变化')
plt.legend()

if 'advantage_score' in df.columns:
    plt.subplot(3, 3, 5)
    plt.plot(time, df['advantage_score'].cumsum().to_numpy(), label='累计优势分数', color='orange')
    plt.xlabel('时间(秒)')
    plt.ylabel('累计优势分数')
    plt.title('累计优势随时间变化(越高越有利)')
    plt.legend()

if 'ego_alt(km)' in df.columns and 'enm_alt(km)' in df.columns:
    plt.subplot(3, 3, 6)
    plt.plot(time, df['ego_alt(km)'].to_numpy(), label='我方高度(km)')
    plt.plot(time, df['enm_alt(km)'].to_numpy(), label='敌方高度(km)', color='red')
    plt.xlabel('时间(秒)')
    plt.ylabel('高度(km)')
    plt.title('高度对比随时间变化')
    plt.legend()

if 'ego_speed_mach' in df.columns and 'enm_speed_mach' in df.columns:
    plt.subplot(3, 3, 7)
    plt.plot(time, df['ego_speed_mach'].to_numpy(), label='我方速度(Mach)')
    plt.plot(time, df['enm_speed_mach'].to_numpy(), label='敌方速度(Mach)', color='red')
    plt.xlabel('时间(秒)')
    plt.ylabel('速度(Mach)')
    plt.title('速度对比随时间变化')
    plt.legend()

if 'energy_diff' in df.columns:
    plt.subplot(3, 3, 8)
    plt.plot(time, df['energy_diff'].to_numpy(), label='能量优势(E_ego - E_enm)', color='brown')
    plt.xlabel('时间(秒)')
    plt.ylabel('能量优势')
    plt.title('能量优势随时间变化')
    plt.legend()

if 'ego_x' in df.columns and 'ego_y' in df.columns and 'enm_x' in df.columns and 'enm_y' in df.columns:
    ego_x = df['ego_x'].to_numpy()
    ego_y = df['ego_y'].to_numpy()
    enm_x = df['enm_x'].to_numpy()
    enm_y = df['enm_y'].to_numpy()
    initial_ego_x = ego_x[0]
    initial_ego_y = ego_y[0]
    ego_x_rel = ego_x - initial_ego_x
    ego_y_rel = ego_y - initial_ego_y
    enm_x_rel = enm_x - initial_ego_x
    enm_y_rel = enm_y - initial_ego_y

    plt.subplot(3, 3, 9)
    plt.plot(ego_x_rel, ego_y_rel, label='我方轨迹')
    plt.plot(enm_x_rel, enm_y_rel, label='敌方轨迹', color='red')
    plt.scatter(0, 0, color='blue', marker='o', label='我方初始位置')
    plt.scatter(enm_x_rel[0], enm_y_rel[0], color='red', marker='o', label='敌方初始位置')
    plt.xlabel('X方向位移(m)')
    plt.ylabel('Y方向位移(m)')
    plt.title('X-Y平面轨迹图(相对我方初始位置)')
    plt.legend()

plt.tight_layout()
plt.savefig("analysis_multiple_metrics.png", dpi=300)
plt.show()
