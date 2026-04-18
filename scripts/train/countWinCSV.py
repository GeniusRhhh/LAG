import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 假设您已经将数据导出为CSV格式，或者可以将Excel数据读取为DataFrame
# 这里模拟一下数据读取过程
df = pd.read_excel("ppo_vs_pursue_test.xlsx", sheet_name="Sheet1")

# 下面是处理数据的函数
def process_combat_data(df):
    """
    处理对战数据，按回合范围分组并计算胜率统计

    参数:
        df: 包含所有对战数据的DataFrame，应该有以下列:
            - 模型轮次
            - 对战场次
            - 我方总奖励
            - 敌方总奖励
            - 奖励差
            - 胜负结果

    返回:
        summary_df: 按回合范围汇总的统计数据
    """
    # 确保数据类型正确
    df['模型轮次'] = df['模型轮次'].astype(int)

    # 定义回合范围
    ranges = [
        (0, 200),
        (200, 400),
        (400, 600),
        (600, 714)  # 调整为您的最大轮次
    ]

    # 创建结果列表
    results = []

    for start, end in ranges:
        # 筛选该范围内的数据
        range_df = df[(df['模型轮次'] >= start) & (df['模型轮次'] < end)]

        if len(range_df) > 0:
            # 计算胜率、败率和平局率
            total_games = len(range_df)
            wins = len(range_df[range_df['胜负结果'] == '胜利'])
            losses = len(range_df[range_df['胜负结果'] == '失败'])
            draws = len(range_df[range_df['胜负结果'] == '平局'])

            win_rate = (wins / total_games) * 100
            lose_rate = (losses / total_games) * 100
            draw_rate = (draws / total_games) * 100

            # 计算平均奖励差
            avg_reward_diff = range_df['奖励差'].mean()

            # 添加到结果列表
            results.append({
                '回合范围': f"{start}-{end}",
                '对战场次': total_games,
                '红方胜率': win_rate,
                '红方败率': lose_rate,
                '平局率': draw_rate,
                '平均奖励差': avg_reward_diff
            })

    # 创建汇总DataFrame
    summary_df = pd.DataFrame(results)

    # 格式化百分比
    summary_df['红方胜率'] = summary_df['红方胜率'].apply(lambda x: f"{x:.1f}%")
    summary_df['红方败率'] = summary_df['红方败率'].apply(lambda x: f"{x:.1f}%")
    summary_df['平局率'] = summary_df['平局率'].apply(lambda x: f"{x:.1f}%")

    return summary_df


# 生成可视化图表
def create_visualization(summary_df):
    """
    基于汇总数据创建可视化图表

    参数:
        summary_df: 按回合范围汇总的统计数据
    """
    # 将百分比字符串转换回数字
    summary_df['红方胜率_num'] = summary_df['红方胜率'].str.rstrip('%').astype(float)
    summary_df['红方败率_num'] = summary_df['红方败率'].str.rstrip('%').astype(float)
    summary_df['平局率_num'] = summary_df['平局率'].str.rstrip('%').astype(float)

    # 创建柱状图
    plt.figure(figsize=(12, 8))

    x = range(len(summary_df))
    width = 0.25

    plt.bar([i - width for i in x], summary_df['红方胜率_num'], width=width, label='红方胜率', color='red')
    plt.bar(x, summary_df['红方败率_num'], width=width, label='红方败率', color='blue')
    plt.bar([i + width for i in x], summary_df['平局率_num'], width=width, label='平局率', color='gray')

    plt.xlabel('训练回合范围')
    plt.ylabel('百分比')
    plt.title('PPO 算法与 Pursue 基线模型的胜率统计')
    plt.xticks(x, summary_df['回合范围'].apply(lambda x: x + ' 回合'))
    plt.legend()

    plt.tight_layout()
    plt.savefig('ppo_vs_pursue_statistics.png')
    plt.show()


# 在Excel中创建数据透视表的说明
def create_pivot_table_instructions():
    """
    返回在Excel中创建数据透视表的步骤说明
    """
    instructions = """
    在Excel中创建数据透视表的步骤:

    1. 选择原始数据区域
    2. 点击"插入" > "数据透视表"
    3. 在数据透视表字段列表中:
       - 将"模型轮次"拖到"筛选器"区域
       - 将"胜负结果"拖到"行"区域
       - 将"胜负结果"拖到"值"区域 (默认为计数)
    4. 右键点击数据透视表中的值 > "值字段设置" > "显示值为" > "行总计的百分比"
    5. 创建自定义分组:
       - 右键点击任意模型轮次值
       - 选择"分组"
       - 设置起始值为0，结束值为714，间隔为200

    这样就可以按回合范围查看胜率、败率和平局率的变化趋势。
    """
    return instructions

# 示例使用
summary = process_combat_data(df)
print(summary)
create_visualization(summary)
print(create_pivot_table_instructions())