"""
文档插图生成脚本（修复版）
- 移除图片中的"图X-X"标题前缀
- 修复符号乱码问题（使用LaTeX格式）
- 优化方框大小和字体
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("开始生成文档插图（修复版）...")
print("=" * 60)

# ==================== 图3-1: 雷达探测概率曲线 ====================
print("\n[1/3] 生成图3-1: 雷达探测概率曲线...")

def P_base_N001VE(R_km):
    if R_km <= 20: return 0.98
    elif R_km <= 40: return 0.95
    elif R_km <= 60: return 0.85
    elif R_km <= 75: return 0.70
    elif R_km <= 85: return 0.50
    elif R_km <= 90: return 0.25
    else: return 0.0

def P_base_APG68(R_km):
    if R_km <= 25: return 0.98
    elif R_km <= 50: return 0.96
    elif R_km <= 70: return 0.90
    elif R_km <= 85: return 0.80
    elif R_km <= 95: return 0.60
    elif R_km <= 100: return 0.35
    elif R_km <= 105: return 0.15
    else: return 0.0

distances = np.linspace(0, 120, 1200)
P_N001 = np.array([P_base_N001VE(d) for d in distances])
P_APG = np.array([P_base_APG68(d) for d in distances])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(distances, P_N001, 'r-', linewidth=3, label='N001VE (Su-27/Su-30)', alpha=0.9)
ax.plot(distances, P_APG, 'b-', linewidth=3, label='AN/APG-68(V)9 (F-16C)', alpha=0.9)
ax.axvline(x=90, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
ax.axvline(x=105, color='blue', linestyle='--', linewidth=1.5, alpha=0.6)
ax.text(90, 0.05, '90km', ha='center', fontsize=10, color='red', fontweight='bold')
ax.text(105, 0.05, '105km', ha='center', fontsize=10, color='blue', fontweight='bold')
ax.fill_between(distances, 0, P_N001, where=(distances<=90), alpha=0.1, color='red')
ax.fill_between(distances, 0, P_APG, where=(distances<=105), alpha=0.1, color='blue')
ax.text(30, 0.93, '高探测区', fontsize=12, bbox=dict(boxstyle='round', 
        facecolor='lightgreen', alpha=0.7), fontweight='bold')
ax.text(80, 0.65, '中等探测区', fontsize=12, bbox=dict(boxstyle='round', 
        facecolor='yellow', alpha=0.7), fontweight='bold')
ax.text(97, 0.25, '低探测区', fontsize=12, bbox=dict(boxstyle='round', 
        facecolor='salmon', alpha=0.7), fontweight='bold')

# 修复：使用LaTeX格式显示下标
ax.set_xlabel('距离 (km)', fontsize=13, fontweight='bold')
ax.set_ylabel(r'基础探测概率 $P_{\mathrm{base}}$', fontsize=13, fontweight='bold')
# 修复：移除"图3-1"前缀，修复RCS上标
ax.set_title(r'雷达基础探测概率与距离关系（RCS = 5.0 $\mathrm{m}^2$）', 
             fontsize=14, fontweight='bold', pad=15)

ax.set_xlim(0, 120)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')

delta_R = ((105-90)/90)*100
ax.annotate(f'APG-68探测距离\n提升 {delta_R:.1f}%', 
            xy=(105, 0.15), xytext=(108, 0.45),
            fontsize=10, color='blue', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('图3-1_雷达探测概率曲线.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 已保存: 图3-1_雷达探测概率曲线.png")

# ==================== 图3-2: 多普勒盲区示意图 ====================
print("\n[2/3] 生成图3-2: 多普勒盲区示意图...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 左图：几何关系
ax1.set_xlim(-2, 12)
ax1.set_ylim(-2, 12)
ax1.set_aspect('equal')
radar_pos = np.array([0, 0])
ax1.plot(*radar_pos, 'rs', markersize=18, label='雷达', zorder=5)
ax1.text(-0.8, -1, '雷达', fontsize=13, fontweight='bold', color='red')
theta_beam = np.linspace(-25, 25, 50)
r_beam = 11
beam_x = r_beam * np.cos(np.deg2rad(theta_beam))
beam_y = r_beam * np.sin(np.deg2rad(theta_beam))
ax1.fill_between(beam_x, beam_y, alpha=0.15, color='blue')
ax1.plot([0, r_beam*np.cos(np.deg2rad(25))], [0, r_beam*np.sin(np.deg2rad(25))], 
         'b--', linewidth=1, alpha=0.6)
ax1.plot([0, r_beam*np.cos(np.deg2rad(-25))], [0, r_beam*np.sin(np.deg2rad(-25))], 
         'b--', linewidth=1, alpha=0.6)
target_a = np.array([9, 2])
v_a = np.array([-2.5, -0.6])
ax1.plot(*target_a, 'go', markersize=14, label='被探测目标', zorder=5)
ax1.arrow(*target_a, *v_a, head_width=0.4, head_length=0.35, 
          fc='green', ec='green', linewidth=2.5, zorder=4)
ax1.text(target_a[0]-2.5, target_a[1]+1.5, '目标A\n（径向速度显著）', 
         fontsize=11, color='green', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax1.plot([radar_pos[0], target_a[0]], [radar_pos[1], target_a[1]], 
         'g:', linewidth=2, alpha=0.7, zorder=1)
target_b = np.array([7, 7])
v_b = np.array([2.5, 0])
ax1.plot(*target_b, 'ro', markersize=14, label='盲区目标 (Beam)', zorder=5)
ax1.arrow(*target_b, *v_b, head_width=0.4, head_length=0.35, 
          fc='red', ec='red', linewidth=2.5, zorder=4)
ax1.text(target_b[0]+1, target_b[1]+1.2, '目标B (Beam机动)\n垂直雷达视线', 
         fontsize=11, color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
ax1.plot([radar_pos[0], target_b[0]], [radar_pos[1], target_b[1]], 
         'r:', linewidth=2, alpha=0.7, zorder=1)
arc = Arc(target_b, 2.5, 2.5, angle=0, theta1=0, theta2=45, 
          color='red', linewidth=2.5, zorder=3)
ax1.add_patch(arc)
ax1.text(target_b[0]+1.5, target_b[1]-1, r'$\approx$90°', fontsize=12, 
         color='red', fontweight='bold')

# 修复：使用LaTeX格式
ax1.text(1, 10, r'$v_r \approx 0$  $\rightarrow$  盲区', fontsize=12, 
         color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
ax1.text(1, 9, r'$v_r > v_{\mathrm{notch}}$  $\rightarrow$  探测', fontsize=12, 
         color='green', fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

ax1.set_xlabel('水平距离 (km)', fontsize=12, fontweight='bold')
ax1.set_ylabel('横向距离 (km)', fontsize=12, fontweight='bold')
ax1.set_title('(a) 多普勒盲区几何原理', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax1.grid(True, alpha=0.2, linestyle=':')

# 右图：径向速度与探测概率
v_r = np.linspace(-300, 300, 600)
def P_detect(v_r, v_notch, P_notch, P_max):
    abs_v = np.abs(v_r)
    P = np.where(abs_v < v_notch, P_notch,
                 np.where(abs_v < 2*v_notch, 
                         P_notch + (abs_v-v_notch)/v_notch*(P_max-P_notch)*0.7,
                         np.minimum(P_max, P_notch + (abs_v-v_notch)/v_notch*(P_max-P_notch))))
    return P

P_N001 = P_detect(v_r, 50, 0.08, 0.95)
P_APG = P_detect(v_r, 30, 0.15, 0.98)

# 修复：使用LaTeX格式
ax2.plot(v_r, P_N001, 'r-', linewidth=3, 
         label=r'N001VE ($v_{\mathrm{notch}}$=50 m/s)', alpha=0.9)
ax2.plot(v_r, P_APG, 'b-', linewidth=3, 
         label=r'APG-68 ($v_{\mathrm{notch}}$=30 m/s)', alpha=0.9)
ax2.axvspan(-50, 50, alpha=0.2, color='red', label='N001VE盲区')
ax2.axvspan(-30, 30, alpha=0.2, color='orange', label='APG-68盲区')
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)

ax2.annotate(r'多普勒盲区' + '\n' + r'$|v_r| < v_{\mathrm{notch}}$', 
             xy=(0, 0.1), xytext=(-180, 0.35),
             fontsize=11, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', lw=2, color='red'),
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

ax2.set_xlabel(r'径向速度 $v_r$ (m/s)', fontsize=12, fontweight='bold')
ax2.set_ylabel(r'探测概率 $P_d$', fontsize=12, fontweight='bold')
ax2.set_title('(b) 径向速度对探测概率的影响', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.set_xlim(-300, 300)
ax2.set_ylim(0, 1.05)

# 修复：移除"图3-2"前缀
# 不使用suptitle，让子图标题就够了

plt.tight_layout()
plt.savefig('图3-2_多普勒盲区示意图.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 已保存: 图3-2_多普勒盲区示意图.png")

# ==================== 图4-1: 机动参数闭环控制 ====================
print("\n[3/3] 生成图4-1: 机动参数闭环控制结构...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# 修复：增大字体，缩小方框
def draw_box(x, y, w, h, text, color, fontsize=13):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                          edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', multialignment='center')

def draw_arrow(x1, y1, x2, y2, label='', lw=2.5):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                           color='black', linewidth=lw, mutation_scale=25, zorder=2)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x, mid_y+0.25, label, ha='center', fontsize=11, 
                fontweight='bold', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.9, edgecolor='gray'))

# 使用LaTeX格式的下标
draw_box(0.5, 6, 2, 1, r'目标状态' + '\n' + r'$h_{\mathrm{target}}$' + '\n' + 
         r'$\psi_{\mathrm{target}}$' + '\n' + r'$V_{\mathrm{target}}$', 
         '#FFE5E5', 12)

draw_box(2, 4, 1.3, 0.8, r'$-$', '#FFF8E1', 20)

draw_box(4.2, 3.8, 2.2, 1.2, r'偏差计算' + '\n' + 
         r'$\Delta h_{\mathrm{error}}$' + '\n' + 
         r'$\Delta \psi_{\mathrm{error}}$' + '\n' + 
         r'$\Delta V_{\mathrm{error}}$', 
         '#FFF8E1', 11)

draw_box(7.2, 3.8, 2.3, 1.2, r'比例控制器' + '\n' + 
         r'$k_h = 0.1$' + '\n' + 
         r'$k_\psi = 0.2$' + '\n' + 
         r'$k_V = 0.15$', 
         '#E3F2FD', 11)

draw_box(10.2, 3.8, 2, 1.2, r'更新参数' + '\n' + 
         r"$\Delta h'$" + '\n' + 
         r"$\Delta \psi'$" + '\n' + 
         r"$\Delta V'$", 
         '#E3F2FD', 11)

draw_box(10.2, 1.8, 2.2, 0.9, r'控制指令' + '\n' + 
         r'$(\delta_a, \delta_e, \delta_r, T)$', 
         '#F3E5F5', 11)

draw_box(7, 1.8, 2.5, 0.9, r'六自由度模型' + '\n' + '(JSBSim)', 
         '#E0F2F1', 12)

draw_box(3.5, 1.8, 2.5, 0.9, r'实际状态' + '\n' + 
         r'$h_{\mathrm{actual}}$' + '\n' + 
         r'$\psi_{\mathrm{actual}}$' + '\n' + 
         r'$V_{\mathrm{actual}}$', 
         '#E5F5E5', 11)

# 绘制箭头
draw_arrow(2.5, 6.5, 2.65, 4.8, '设定')
draw_arrow(3.3, 4.4, 4.2, 4.4, '')
draw_arrow(6.4, 4.4, 7.2, 4.4, '')
draw_arrow(9.5, 4.4, 10.2, 4.4, '')
draw_arrow(11.2, 3.8, 11.2, 2.7, '生成')
draw_arrow(10.2, 2.3, 9.5, 2.3, '')
draw_arrow(7, 2.3, 6, 2.3, '')

# 反馈回路
feedback_path = FancyArrowPatch((3.5, 2.3), (2.65, 4), 
                               arrowstyle='->', connectionstyle="arc3,rad=-.5",
                               color='#2E7D32', linewidth=4.5, 
                               linestyle='--', mutation_scale=25, zorder=1)
ax.add_patch(feedback_path)
ax.text(1.2, 3, '反馈', fontsize=13, fontweight='bold', color='#2E7D32', 
        rotation=90, bbox=dict(boxstyle='round', facecolor='lightgreen', 
        alpha=0.8, edgecolor='#2E7D32'))

# 修复：移除"图4-1"前缀
ax.text(7, 7.3, '机动参数动态调整闭环控制结构', ha='center', 
        fontsize=16, fontweight='bold')

# 说明文本
textstr = '闭环控制流程：\n' \
          '① 设定目标状态\n' \
          '② 与实际状态求偏差\n' \
          '③ 比例控制器调整参数\n' \
          '④ 生成控制指令执行\n' \
          '⑤ 状态更新后反馈'
props = dict(boxstyle='round', facecolor='#FFFDE7', alpha=0.95, 
             edgecolor='#F57F17', linewidth=2)
ax.text(0.3, 2.8, textstr, fontsize=11, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('图4-1_机动参数闭环控制.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 已保存: 图4-1_机动参数闭环控制.png")

print("\n" + "=" * 60)
print("✓ 所有图片生成完成！")
print("=" * 60)
print("\n生成的图片列表：")
print("  1. 图3-1_雷达探测概率曲线.png")
print("  2. 图3-2_多普勒盲区示意图.png")
print("  3. 图4-1_机动参数闭环控制.png")
print("\n修复内容：")
print("  ✓ 移除图片中的'图X-X'标题前缀")
print("  ✓ 使用LaTeX格式修复符号乱码（下标、上标）")
print("  ✓ 增大方框内字体（11-13号），优化方框大小")
print("=" * 60)


