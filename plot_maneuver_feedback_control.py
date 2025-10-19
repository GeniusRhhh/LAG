import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 定义颜色方案
color_target = '#ffcccc'
color_actual = '#ccffcc'
color_error = '#fff4e1'
color_controller = '#e3f2fd'
color_executor = '#f3e5f5'
color_dynamics = '#ffccff'

# 辅助函数：绘制圆角矩形框
def draw_box(ax, x, y, width, height, text, color, fontsize=11):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', multialignment='center')

# 辅助函数：绘制箭头
def draw_arrow(ax, x1, y1, x2, y2, label='', color='black', style='->'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, 
                           connectionstyle="arc3", color=color, linewidth=2.5,
                           mutation_scale=20)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ===== 第1层：目标设定 =====
draw_box(ax, 0.5, 8.5, 2.5, 1, '战术目标状态\n$h_{target}$, $ψ_{target}$, $V_{target}$', 
         color_target, fontsize=12)

# ===== 第2层：实际状态反馈 =====
draw_box(ax, 11, 8.5, 2.5, 1, '飞机实际状态\n$h_{actual}$, $ψ_{actual}$, $V_{actual}$', 
         color_actual, fontsize=12)

# ===== 第3层：偏差计算 =====
draw_box(ax, 5, 7, 4, 1.2, '偏差计算模块', color_error, fontsize=12)

# 三个偏差输出
draw_box(ax, 1, 5.2, 2.5, 0.8, '高度偏差\n$Δh_{error} = h_{target} - h_{actual}$', 
         color_error, fontsize=10)
draw_box(ax, 5.5, 5.2, 2.5, 0.8, '航向偏差\n$Δψ_{error} = ψ_{target} - ψ_{actual}$', 
         color_error, fontsize=10)
draw_box(ax, 10, 5.2, 2.5, 0.8, '速度偏差\n$ΔV_{error} = V_{target} - V_{actual}$', 
         color_error, fontsize=10)

# ===== 第4层：比例控制器 =====
draw_box(ax, 5, 3.2, 4, 1.2, '比例控制器\n(动态参数调整)', color_controller, fontsize=12)

# 三个调整输出
draw_box(ax, 1, 1.4, 2.5, 0.9, "高度调整\n$Δh' = Δh + k_h·Δh_{error}$\n$k_h = 0.1$", 
         color_controller, fontsize=9)
draw_box(ax, 5.5, 1.4, 2.5, 0.9, "航向调整\n$Δψ' = Δψ + k_ψ·Δψ_{error}$\n$k_ψ = 0.2$", 
         color_controller, fontsize=9)
draw_box(ax, 10, 1.4, 2.5, 0.9, "速度调整\n$ΔV' = ΔV + k_V·ΔV_{error}$\n$k_V = 0.15$", 
         color_controller, fontsize=9)

# ===== 第5层：控制指令生成 =====
draw_box(ax, 4, 0.1, 6, 0.8, '生成底层控制指令: $(δ_a, δ_e, δ_r, T)$', 
         color_executor, fontsize=11)

# ===== 第6层：动力学执行（右侧） =====
draw_box(ax, 11, 3, 2.5, 1.5, '六自由度\n动力学模型\n(JSBSim)', color_dynamics, fontsize=11)

draw_box(ax, 11, 1, 2.5, 1.2, '飞机状态更新\n$(x,y,z,V,θ,ψ,φ)$', color_dynamics, fontsize=10)

# ===== 绘制箭头连接 =====
# 目标 -> 偏差计算
draw_arrow(ax, 2.75, 8.5, 5.5, 8.1, '目标')

# 实际状态 -> 偏差计算
draw_arrow(ax, 11.5, 8.5, 8.5, 8.1, '反馈')

# 偏差计算 -> 三个偏差
draw_arrow(ax, 7, 7, 2.25, 6.0, '', 'blue')
draw_arrow(ax, 7, 7, 6.75, 6.0, '', 'blue')
draw_arrow(ax, 7, 7, 11.25, 6.0, '', 'blue')

# 三个偏差 -> 控制器
draw_arrow(ax, 2.25, 5.2, 5.5, 4.4, '', 'green')
draw_arrow(ax, 6.75, 5.2, 7, 4.4, '', 'green')
draw_arrow(ax, 11.25, 5.2, 8.5, 4.4, '', 'green')

# 控制器 -> 三个调整
draw_arrow(ax, 7, 3.2, 2.25, 2.3, '', 'purple')
draw_arrow(ax, 7, 3.2, 6.75, 2.3, '', 'purple')
draw_arrow(ax, 7, 3.2, 11.25, 2.3, '', 'purple')

# 三个调整 -> 指令生成
draw_arrow(ax, 2.25, 1.4, 5, 0.8, '', 'orange')
draw_arrow(ax, 6.75, 1.4, 7, 0.8, '', 'orange')
draw_arrow(ax, 11.25, 1.4, 9, 0.8, '', 'orange')

# 指令生成 -> 动力学模型
draw_arrow(ax, 10, 0.5, 11, 3.75, '控制指令', 'red')

# 动力学模型 -> 状态更新
draw_arrow(ax, 12.25, 3, 12.25, 2.2, '计算', 'red')

# 状态更新 -> 反馈到实际状态（闭环）
arrow_feedback = FancyArrowPatch((12.25, 8.5), (12.25, 2.2), 
                                 arrowstyle='->', connectionstyle="arc3,rad=0.5",
                                 color='darkgreen', linewidth=3, mutation_scale=25,
                                 linestyle='--')
ax.add_patch(arrow_feedback)
ax.text(13.2, 5.5, '状态\n反馈\n回路', ha='center', fontsize=11, fontweight='bold',
        color='darkgreen', rotation=90)

# ===== 添加标题和说明 =====
ax.text(7, 9.5, '机动参数动态调整闭环控制系统', ha='center', fontsize=16, fontweight='bold')

# 添加图例说明
legend_elements = [
    mpatches.Patch(facecolor=color_target, edgecolor='black', label='目标设定层'),
    mpatches.Patch(facecolor=color_actual, edgecolor='black', label='状态感知层'),
    mpatches.Patch(facecolor=color_error, edgecolor='black', label='偏差计算层'),
    mpatches.Patch(facecolor=color_controller, edgecolor='black', label='控制调整层'),
    mpatches.Patch(facecolor=color_executor, edgecolor='black', label='指令生成层'),
    mpatches.Patch(facecolor=color_dynamics, edgecolor='black', label='物理执行层'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, ncol=3)

# 添加说明文本
textstr = '闭环控制原理：\n' \
          '1. 持续监测飞机实际状态与目标状态的偏差\n' \
          '2. 根据偏差通过比例控制器动态调整机动参数\n' \
          '3. 更新后的参数生成新的控制指令\n' \
          '4. 控制指令驱动六自由度模型更新飞机状态\n' \
          '5. 形成闭环反馈，确保飞机逐步逼近目标状态'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=2)
ax.text(0.3, 2.8, textstr, fontsize=9.5, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('maneuver_feedback_control.png', dpi=300, bbox_inches='tight')
print("图12已保存为: maneuver_feedback_control.png")
plt.show()


