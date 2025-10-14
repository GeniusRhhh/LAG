"""
文档插图生成脚本
运行此脚本将生成文档所需的所有图片
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("开始生成文档插图...")
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
ax.set_xlabel('距离 (km)', fontsize=13, fontweight='bold')
ax.set_ylabel('基础探测概率 $P_{base}$', fontsize=13, fontweight='bold')
ax.set_title('图3-1  雷达基础探测概率与距离关系（RCS = 5.0 m²）', 
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
ax1.text(target_b[0]+1.5, target_b[1]-1, '≈90°', fontsize=12, 
         color='red', fontweight='bold')
ax1.text(1, 10, '$v_r$ ≈ 0  →  盲区', fontsize=12, color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
ax1.text(1, 9, '$v_r$ > $v_{notch}$  →  探测', fontsize=12, color='green', 
         fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
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
ax2.plot(v_r, P_N001, 'r-', linewidth=3, label='N001VE ($v_{notch}$=50 m/s)', alpha=0.9)
ax2.plot(v_r, P_APG, 'b-', linewidth=3, label='APG-68 ($v_{notch}$=30 m/s)', alpha=0.9)
ax2.axvspan(-50, 50, alpha=0.2, color='red', label='N001VE盲区')
ax2.axvspan(-30, 30, alpha=0.2, color='orange', label='APG-68盲区')
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
ax2.annotate('多普勒盲区\n$|v_r| < v_{notch}$', xy=(0, 0.1), xytext=(-180, 0.35),
             fontsize=11, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', lw=2, color='red'),
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
ax2.set_xlabel('径向速度 $v_r$ (m/s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('探测概率 $P_d$', fontsize=12, fontweight='bold')
ax2.set_title('(b) 径向速度对探测概率的影响', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.set_xlim(-300, 300)
ax2.set_ylim(0, 1.05)
plt.suptitle('图3-2  多普勒盲区效应示意图', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('图3-2_多普勒盲区示意图.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 已保存: 图3-2_多普勒盲区示意图.png")

# ==================== 图4-1: 机动参数闭环控制 ====================
print("\n[3/3] 生成图4-1: 机动参数闭环控制结构...")

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

def draw_box(x, y, w, h, text, color, fontsize=11):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                          edgecolor='black', facecolor=color, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', multialignment='center')

def draw_arrow(x1, y1, x2, y2, label='', lw=2.5):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                           color='black', linewidth=lw, mutation_scale=25, zorder=2)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x, mid_y+0.25, label, ha='center', fontsize=10, 
                fontweight='bold', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.9, edgecolor='gray'))

draw_box(0.5, 5, 2, 1.2, '目标状态\n$h_{target}$\n$ψ_{target}$\n$V_{target}$', '#FFE5E5', 10)
draw_box(2, 3, 1.5, 1, '−', '#FFF8E1', 16)
draw_box(4.5, 3, 2.2, 1.3, '偏差计算\n$Δh_{error}$\n$Δψ_{error}$\n$ΔV_{error}$', '#FFF8E1', 10)
draw_box(7.5, 3, 2.5, 1.3, '比例控制器\n$k_h = 0.1$\n$k_ψ = 0.2$\n$k_V = 0.15$', '#E3F2FD', 10)
draw_box(10.5, 3, 2, 1.3, '更新参数\n$Δh\'$\n$Δψ\'$\n$ΔV\'$', '#E3F2FD', 10)
draw_box(10.5, 0.8, 2, 1, '控制指令\n$(δ_a,δ_e,δ_r,T)$', '#F3E5F5', 10)
draw_box(7, 0.8, 2.5, 1, '六自由度模型\n(JSBSim)', '#E0F2F1', 11)
draw_box(3.5, 0.8, 2.5, 1, '实际状态\n$h_{actual}$\n$ψ_{actual}$\n$V_{actual}$', '#E5F5E5', 10)

draw_arrow(2.5, 5.6, 2.75, 4, '设定')
draw_arrow(3.5, 3.5, 4.5, 3.65, '')
draw_arrow(6.7, 3.65, 7.5, 3.65, '')
draw_arrow(10, 3.65, 10.5, 3.65, '')
draw_arrow(11.5, 3, 11.5, 1.8, '生成')
draw_arrow(10.5, 1.3, 9.5, 1.3, '')
draw_arrow(7, 1.3, 6, 1.3, '')

feedback_path = FancyArrowPatch((3.5, 1.3), (2.75, 3), 
                               arrowstyle='->', connectionstyle="arc3,rad=-.5",
                               color='#2E7D32', linewidth=4, 
                               linestyle='--', mutation_scale=25, zorder=1)
ax.add_patch(feedback_path)
ax.text(1, 2, '反馈', fontsize=12, fontweight='bold', color='#2E7D32', 
        rotation=90, bbox=dict(boxstyle='round', facecolor='lightgreen', 
        alpha=0.8, edgecolor='#2E7D32'))

ax.text(7, 6.5, '图4-1  机动参数动态调整闭环控制结构', ha='center', 
        fontsize=15, fontweight='bold')

textstr = '闭环控制流程：\n① 设定目标状态\n② 与实际状态求偏差\n③ 比例控制器调整参数\n④ 生成控制指令执行\n⑤ 状态更新后反馈'
props = dict(boxstyle='round', facecolor='#FFFDE7', alpha=0.95, 
             edgecolor='#F57F17', linewidth=2)
ax.text(0.3, 1.8, textstr, fontsize=10, verticalalignment='top', bbox=props)

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
print("\n请将图片插入到文档相应位置（见下方说明）")
print("=" * 60)


