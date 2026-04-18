"""
协同探测与目标分配诊断脚本
用于分析：
1. 为什么B0300/B0400探测延迟30秒
2. 为什么所有飞机都攻击B0100/B0200
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 在cooperative_detection.py中添加诊断日志
# 在cooperative_engagement.py中添加诊断日志

print("""
=== 诊断方案 ===

问题1: B0300/B0400探测延迟30秒
根因分析：
- 协同探测使用DIRECTED模式时，_assign_directed_scan()根据目标方位分配扇区
- 如果B0100/B0200在左侧，B0300/B0400在右侧，但所有雷达都被分配到左侧扇区
- 导致右侧目标长时间不在任何雷达波束内

需要检查：
1. all_target_bearings是否包含所有4个目标的方位
2. 扇区分配是否覆盖了所有目标方位
3. 雷达波束是否真的扫到了B0300/B0400的方位

问题2: 所有飞机攻击B0100/B0200
根因分析：
- cooperative_engagement.py的_select_best_shooter()使用"空间对应"算法
- 当前逻辑：shooter_on_left != target_on_left时增加50km惩罚
- 但实际应该是：左编队打左目标，右编队打右目标（同侧对抗）
- 代码注释说"同侧对抗"，但实现是"异侧惩罚"，逻辑相反！

修复方案：
1. 修改_select_best_shooter()的空间对应逻辑
2. 确保_assign_directed_scan()覆盖所有目标方位
3. 添加详细日志打印目标分配过程

=== 修复步骤 ===

步骤1: 修复cooperative_engagement.py的空间对应逻辑
- 当前：shooter_on_left != target_on_left → 惩罚50km
- 修复：shooter_on_left == target_on_left → 奖励（减少距离）
       shooter_on_left != target_on_left → 惩罚（增加距离）

步骤2: 增强_assign_directed_scan()的目标覆盖
- 确保all_target_bearings包含所有探测到的目标
- 扇区分配时确保覆盖整个方位跨度
- 添加日志打印每个雷达的扇区分配

步骤3: 添加诊断日志
- 在_assign_directed_scan()中打印目标方位和扇区分配
- 在_select_best_shooter()中打印评分细节
- 在雷达扫描时打印波束位置和目标方位

""")

# 生成修复补丁
print("\n=== 生成修复补丁 ===\n")

patch_engagement = """
# 修复cooperative_engagement.py第218-227行

# 原代码（错误）：
# shooter_on_left = pos[0] < center_x
# spatial_penalty = 0
# if shooter_on_left != target_on_left:
#     spatial_penalty = 50.0  # 异侧 = 不理想
# else:
#     spatial_penalty = 0     # 同侧 = 理想

# 修复后（正确）：
shooter_on_left = pos[0] < center_x

# 🔥 修复：空间对应逻辑
# 目标在左侧(x<center) → 左编队优先(A0100/A0200)
# 目标在右侧(x>center) → 右编队优先(A0300/A0400)
if shooter_on_left == target_on_left:
    # 同侧对抗 = 理想配对，给予奖励
    spatial_penalty = -20.0  # 负值表示奖励
else:
    # 异侧对抗 = 不理想，增加惩罚
    spatial_penalty = 80.0   # 增大惩罚确保优先同侧
"""

patch_detection = """
# 修复cooperative_detection.py的_assign_directed_scan()

# 在第120行附近添加详细日志：
if self._scan_debug_counter % 10 == 0:  # 每10次打印一次
    print(f"[DEBUG 扇区分配] 目标数:{len(all_target_bearings)} 方位:{[f'{b:.1f}' for b in all_target_bearings]}")
    print(f"[DEBUG 扇区分配] 跨度:{bearing_span:.1f}° 中心:{center_bearing:.1f}° 模式:{'集中' if bearing_span < CONCENTRATED_THRESHOLD else '分散'}")
    for i, aid in enumerate(agents):
        asgn = self._assignments.get(aid)
        if asgn:
            print(f"[DEBUG 扇区分配] {aid}: 中心{asgn.azimuth_center:.1f}° 范围±{asgn.azimuth_range:.1f}°")
"""

print(patch_engagement)
print(patch_detection)

print("\n=== 执行修复 ===")
print("请运行以下命令应用修复：")
print("1. 修改 scripts/tacticalProject/cap/tactics/cooperative_engagement.py")
print("2. 修改 scripts/tacticalProject/cap/tactics/cooperative_detection.py")
print("3. 重新运行仿真验证")
