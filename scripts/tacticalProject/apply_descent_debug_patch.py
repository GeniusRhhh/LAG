#!/usr/bin/env python3
"""
为敌方AI决策过程添加详细的调试日志补丁
自动在关键函数处插入日志调用
"""

import os
import re

# 目标文件列表
files_to_patch = [
    ('enemy_ai_refactor_helpers.py', '_select_action'),
    ('enemy_ai_refactor_helpers.py', '_execute_action'),
    ('enemy_ai_maneuver_helpers.py', '_execute_dive_escape'),
    ('enemy_ai_maneuver_helpers.py', '_execute_spiral_dive'),
]

scope_root = r'd:\Pycharm\LAG\scripts\tacticalProject'


def add_import_at_top(file_path):
    """在文件顶部添加导入"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已添加
    if 'from enemy_ai_descent_trace import' in content:
        print(f"  ℹ️  {file_path} 已包含导入")
        return
    
    # 在第一个导入后添加新导入
    import_pattern = r'(from __future__ import.*?\n)'
    if re.search(import_pattern, content):
        content = re.sub(
            import_pattern,
            r'\1\ntry:\n    from enemy_ai_descent_trace import *\nexcept:\n    pass\n',
            content,
            count=1
        )
    else:
        # 没有 __future__ 导入，就在最开始添加
        content = "try:\n    from enemy_ai_descent_trace import *\nexcept:\n    pass\n\n" + content
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ 已在 {os.path.basename(file_path)} 顶部添加导入")


def patch_select_action(file_path):
    """为_select_action函数添加日志"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到_select_action函数
    func_start = None
    for i, line in enumerate(lines):
        if 'def _select_action(self' in line:
            func_start = i
            break
    
    if func_start is None:
        print(f"  ❌ 在 {file_path} 中找不到 _select_action 函数")
        return
    
    # 在函数开始处添加日志
    log_snippet = '''
        # 🔍 调试：记录动作选择过程
        try:
            dbg_agent_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m) if hasattr(env, 'agents') and agent_id in env.agents else None
            log_action_selection_trace(
                agent_id=agent_id,
                current_time=0,  # TODO: current_time参数
                current_altitude=dbg_agent_alt or 0,
                tactical_phase=str(self.current_phase.get(agent_id, 'UNKNOWN')),
                tactical_mode=str(tactical_mode.value if hasattr(tactical_mode, 'value') else tactical_mode),
                available_actions=action_weights if 'action_weights' in locals() else {}
            )
        except Exception as e:
            pass  # 调试不影响主逻辑
'''
    
    # （这个补丁比较复杂，因为_select_action是委托给helper的）
    # 我们在helper函数中添加日志会更有效
    print(f"  ℹ️  {os.path.basename(file_path)}: _select_action已通过代理调用，日志在helper中添加")


def patch_execute_action(file_path):
    """为_execute_action函数添加详细日志"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 在关键的动作类型检查处添加日志
    
    # DESCEND action patch
    descend_old = '''        elif action_type == ActionType.DESCEND:
            # 🛡️ 完全禁用下降动作，改为水平飞行
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            logging.warning(f"🛡️ {agent_id} 下降动作已禁用（高度{current_altitude:.0f}m），改为水平飞行")
            return self._execute_maintain_heading(env, agent_id)'''
    
    descend_new = '''        elif action_type == ActionType.DESCEND:
            # 🛡️ 完全禁用下降动作，改为水平飞行
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            logging.warning(f"🛡️ {agent_id} 下降动作已禁用（高度{current_altitude:.0f}m），改为水平飞行")
            try:
                log_altitude_command_generated(
                    agent_id, current_time, current_altitude,
                    "DESCEND (BUT DISABLED)",
                    reason="Action禁用:改为水平飞行"
                )
            except:
                pass
            return self._execute_maintain_heading(env, agent_id)'''
    
    if descend_old in content:
        content = content.replace(descend_old, descend_new)
        print(f"  ✓ 已为 DESCEND 动作添加日志")
    
    # DIVE_ESCAPE patch
    dive_old = '''        elif action_type == ActionType.DIVE_ESCAPE:
            return self._execute_dive_escape(env, agent_id)'''
    
    dive_new = '''        elif action_type == ActionType.DIVE_ESCAPE:
            try:
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m) if env and agent_id in env.agents else 0
                log_altitude_command_generated(
                    agent_id, current_time, current_altitude,
                    "DIVE_ESCAPE",
                    reason="执行俯冲脱离"
                )
            except:
                pass
            return self._execute_dive_escape(env, agent_id)'''
    
    if dive_old in content:
        content = content.replace(dive_old, dive_new)
        print(f"  ✓ 已为 DIVE_ESCAPE 动作添加日志")
    
    # SPIRAL_DIVE patch
    spiral_old = '''        elif action_type == ActionType.SPIRAL_DIVE:
            return self._execute_spiral_dive(env, agent_id)'''
    
    spiral_new = '''        elif action_type == ActionType.SPIRAL_DIVE:
            try:
                current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m) if env and agent_id in env.agents else 0
                log_altitude_command_generated(
                    agent_id, current_time, current_altitude,
                    "SPIRAL_DIVE",
                    reason="执行螺旋俯冲"
                )
            except:
                pass
            return self._execute_spiral_dive(env, agent_id)'''
    
    if spiral_old in content:
        content = content.replace(spiral_old, spiral_new)
        print(f"  ✓ 已为 SPIRAL_DIVE 动作添加日志")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    """主函数"""
    
    print("=" * 80)
    print("【敌方AI调试日志补丁生成器】")
    print("=" * 80)
    print()
    
    os.chdir(scope_root)
    
    # 1. 在各文件顶部添加导入
    print("[1] 添加导入...")
    files_to_process = [
        'enemy_ai_refactor_helpers.py',
        'enemy_ai_maneuver_helpers.py',
    ]
    
    for file in files_to_process:
        file_path = os.path.join(scope_root, file)
        if os.path.exists(file_path):
            add_import_at_top(file_path)
        else:
            print(f"  ❌ 文件未找到: {file}")
    
    print()
    
    # 2. 为_execute_action添加日志
    print("[2] 为_execute_action添加日志...")
    refactor_path = os.path.join(scope_root, 'enemy_ai_refactor_helpers.py')
    if os.path.exists(refactor_path):
        patch_execute_action(refactor_path)
    
    print()
    
    # 3. 生成运行说明
    print("[3] 生成使用说明...")
    
    usage_file = os.path.join(scope_root, 'cap_results', 'DEBUG_QUICKSTART.txt')
    os.makedirs(os.path.dirname(usage_file), exist_ok=True)
    
    usage_content = """
【敌方AI下降异常调试 - 快速开始】

✅ 补丁已应用！现在可以运行启用调试的仿真。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【启用调试】

在PowerShell中运行：

  Set-Location d:\\Pycharm\\LAG\\scripts\\tacticalProject
  .\\enable_descent_debug.ps1
  
  # 或者手动设置环境变量
  $env:ENEMY_DESCENT_DEBUG = '1'
  $env:CAP_ROOTCAUSE_TRACE = '1'

【运行仿真】

  python .\run_cap_simulation_native.py

【查看结果】

实时：
  Get-Content .\cap_results\\descent_debug.log -Wait -Tail 100

完整分析后：
  Select-String "B0200" .\cap_results\\descent_debug.log

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【日志文件】

1. descent_debug.log
   → 完整的决策追踪链
   → 包含：战术阶段、威胁评估、动作选择、命令生成
   → 每次下降都有记录

2. descent_diagnosis.log  
   → 系统初始化日志
   → 环境配置信息

3. DEBUGGING_GUIDE.txt
   → 详细的调试指南
   → 日志搜索技巧

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【关键搜索】

找出B0200的所有下降事件：
  Select-String "\\[决策追踪\\].*B0200" .\\cap_results\\descent_debug.log

找出最大下降：
  Select-String "is_descent=True" .\\cap_results\\descent_debug.log | 
    Sort-Object { [float]($_ -replace ".*descent_magnitude=([0-9.]+).*", '$1') } -Descending | 
    Select-Object -First 5

显示动作选择序列：
  Select-String "selected_action=" .\\cap_results\\descent_debug.log | 
    head -50

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【排查步骤】

1️⃣  运行仿真并收集日志
2️⃣  搜索 "is_descent=True" 找出所有下降
3️⃣  对于每次下降，检查：
    - selected_action: 选择了什么动作？
    - tactical_phase: 处于什么战术阶段？
    - threat_level: 威胁等级是多少？
4️⃣  追踪该动作的执行过程
5️⃣  确定是否是预期的异常或代码错误

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    with open(usage_file, 'w', encoding='utf-8') as f:
        f.write(usage_content)
    
    print(f"  ✓ 快速开始指南: {usage_file}")
    
    print()
    print("=" * 80)
    print("✅ 补丁应用完成！")
    print("=" * 80)
    print()
    print("后续步骤：")
    print("  1. PowerShell: .\\enable_descent_debug.ps1")
    print("  2. 运行仿真: python .\\run_cap_simulation_native.py")
    print("  3. 查看日志: tail -f .\\cap_results\\descent_debug.log")


if __name__ == '__main__':
    main()
"""
