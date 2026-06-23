import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 请确保更改此路径以指向您的数据文件
file_path = '../../renders/20241217_latest_215_125.4102.csv'

# 从 CSV 文件中读取数据，这里假设编码问题已解决
df = pd.read_csv(file_path,encoding='gbk')

# 设置绘图
plt.figure(figsize=(12, 10))

# 绘制 AO(角度偏差)
plt.subplot(2, 2, 1)
plt.plot(df['步数'].to_numpy(), df['AO(角度偏差)'].to_numpy(), marker='o')  # 使用 to_numpy() 转换
plt.title('AO(角度偏差)随步数变化')
plt.xlabel('步数')
plt.ylabel('AO(角度偏差)')

# 绘制 TA(目标角度)
plt.subplot(2, 2, 2)
plt.plot(df['步数'].to_numpy(), df['TA(目标角度)'].to_numpy(), color='red', marker='o')  # 使用 to_numpy() 转换
plt.title('TA(目标角度)随步数变化')
plt.xlabel('步数')
plt.ylabel('TA(目标角度)')

# 绘制距离变化
plt.subplot(2, 2, 3)
plt.plot(df['步数'].to_numpy(), df['距离(m)'].to_numpy(), color='green', marker='o')  # 使用 to_numpy() 转换
plt.title('距离随步数变化')
plt.xlabel('步数')
plt.ylabel('距离(m)')

# 绘制奖励变化
plt.subplot(2, 2, 4)
plt.plot(df['步数'].to_numpy(), df['奖励'].to_numpy(), color='purple', marker='o')  # 使用 to_numpy() 转换
plt.title('奖励随步数变化')
plt.xlabel('步数')
plt.ylabel('奖励')

plt.tight_layout()
plt.show()
