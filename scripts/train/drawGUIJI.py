import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

# 读取ACMI文件
def read_acmi_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

# 解析ACMI数据，提取每个对象的轨迹
def parse_acmi_data(lines):
    trajectories = {}  # 用字典存储每个对象的轨迹数据

    for line in lines:
        line = line.strip()
        if line.startswith('#') or line.startswith('FileType') or line.startswith('FileVersion') or line.startswith('0,ReferenceTime'):
            continue  # 跳过非数据行

        # 匹配对象ID和T字段
        match = re.match(r'([A-Za-z0-9]+),T=([^,]+),Name=([^,]+),Color=([^,]+)', line)
        if match:
            obj_id = match.group(1)  # 对象ID，如A0100
            t_data = match.group(2)  # T字段数据
            name = match.group(3)   # 名称，如F16
            color = match.group(4)  # 颜色，如Red

            # 初始化该对象的轨迹数据
            if obj_id not in trajectories:
                trajectories[obj_id] = {'x': [], 'y': [], 'z': [], 'name': name, 'color': color.lower()}

            # 解析T字段：x|y|z|vx|vy|heading
            coords = t_data.split('|')
            if len(coords) >= 3:  # 确保有x, y, z
                try:
                    x = float(coords[0])
                    y = float(coords[1])
                    z = float(coords[2])
                    trajectories[obj_id]['x'].append(x)
                    trajectories[obj_id]['y'].append(y)
                    trajectories[obj_id]['z'].append(z)
                except ValueError:
                    continue  # 跳过解析失败的行

    return trajectories

# 绘制3D轨迹图，起点标注，优化空间
def plot_3d_trajectory(trajectories):
    # 创建图形
    fig = plt.figure(figsize=(9, 7))  # 正方形画布，减少多余高度
    ax = fig.add_subplot(111, projection='3d')

    # 计算所有轨迹的边界
    all_x, all_y, all_z = [], [], []
    for obj_id, data in trajectories.items():
        all_x.extend(data['x'])
        all_y.extend(data['y'])
        all_z.extend(data['z'])

    # 为每个对象绘制轨迹和起点
    for obj_id, data in trajectories.items():
        # 绘制轨迹线
        ax.plot(data['x'], data['y'], data['z'],
                color=data['color'],
                linewidth=2)

        # 标注起点（第一个点）
        if data['x']:  # 确保有数据
            ax.scatter(data['x'][0], data['y'][0], data['z'][0],
                       color=data['color'],
                       s=100,
                       marker='o')

    # 设置轴范围，紧贴数据并留一点缓冲
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    z_range = max(all_z) - min(all_z)
    ax.set_xlim(min(all_x) - 0.05 * x_range, max(all_x) + 0.05 * x_range)
    ax.set_ylim(min(all_y) - 0.05 * y_range, max(all_y) + 0.05 * y_range)
    ax.set_zlim(min(all_z) - 0.05 * z_range, max(all_z) + 0.05 * z_range)

    # 设置轴标签，调整位置和字体
    ax.set_xlabel('X (m)', fontsize=10, labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=10, labelpad=10)
    ax.set_zlabel('Z (m)', fontsize=10, labelpad=15)  # 特别增加Z轴标签间距

    # 调整刻度字体大小
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='z', labelsize=8)

    # 优化布局，留出足够空间给Z轴
    plt.tight_layout(pad=1.0)  # 增加pad值，确保Z轴完全显示

    # 显示图形
    plt.show()

# 主函数
def main():
    # 使用你的文件路径
    file_path = "../../renders/1v1_0319Hierar_100_550_-627.22_-217.95.txt.acmi"
    lines = read_acmi_file(file_path)

    # 解析数据
    trajectories = parse_acmi_data(lines)

    # 检查是否成功提取数据
    if not trajectories:
        print("未能从文件中提取到有效轨迹数据，请检查文件格式。")
        return

    # 绘制3D轨迹图
    plot_3d_trajectory(trajectories)

if __name__ == "__main__":
    main()