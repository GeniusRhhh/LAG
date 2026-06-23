#!/usr/bin/env python
"""
在敌方AI关键函数中自动插入调试日志
"""

import os

# 1. 修改unified_enemy_tactical_ai_impl.py中的get_enemy_command()

unified_impl_path = r'd:\Pycharm\LAG\scripts\tacticalProject\unified_enemy_tactical_ai_impl.py'

# 读取文件
with open(unified_impl_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 在try块之后立即添加调试导入
debug_import = '''try:
            from enemy_ai_descent_debugger import get_debugger
            debugger = get_debugger()
        except:
            debugger = None
        '''

# 检查是否已添加
if 'from enemy_ai_descent_debugger import get_debugger' not in content:
    # 找到 "root_trace = os.getenv" 这一行，在它之前插入调试导入
    content = content.replace(
        "        try:\n            root_trace = os.getenv",
        f"        try:\n            {debug_import}\n\n            root_trace = os.getenv"
    )
    
    # 保存修改后的文件
    with open(unified_impl_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 已在 {unified_impl_path} 中添加调试导入")
else:
    print("ℹ 调试导入已存在")

# 2. 修改enemy_ai_refactor_helpers.py中的_execute_action()

refactor_helpers_path = r'd:\Pycharm\LAG\scripts\tacticalProject\enemy_ai_refactor_helpers.py'

# 读取文件
with open(refactor_helpers_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 在_execute_action函数的关键位置添加调试

# 在DESCEND处理之前添加日志
if 'elif action_type == ActionType.DESCEND:' in content:
    # 找到下降处理，添加调试
    old_descend = '''        elif action_type == ActionType.DESCEND:
            # 🛡️ 完全禁用下降动作，改为水平飞行
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            logging.warning(f"🛡️ {agent_id} 下降动作已禁用（高度{current_altitude:.0f}m），改为水平飞行")
            return self._execute_maintain_heading(env, agent_id)'''
    
    new_descend = '''        elif action_type == ActionType.DESCEND:
            # 🛡️ 完全禁用下降动作，改为水平飞行
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            logging.warning(f"🛡️ {agent_id} 下降动作已禁用（高度{current_altitude:.0f}m），改为水平飞行")
            # 🔍 调试：记录进入DESCEND的原因
            try:
                from enemy_ai_descent_debugger import get_debugger
                dbg = get_debugger()
                if dbg:
                    dbg.log_altitude_command_generated(
                        agent_id, 0, -500.0,
                        reason="DESCEND_ACTION_ATTEMPTED_BUT_DISABLED"
                    )
            except:
                pass
            return self._execute_maintain_heading(env, agent_id)'''
    
    content = content.replace(old_descend, new_descend)
    print("✓ 已在DESCEND处理添加调试日志")

# 在DIVE_ESCAPE处理之前添加日志
if 'elif action_type == ActionType.DIVE_ESCAPE:' in content:
    old_dive = '''        elif action_type == ActionType.DIVE_ESCAPE:
            return self._execute_dive_escape(env, agent_id)'''
    
    new_dive = '''        elif action_type == ActionType.DIVE_ESCAPE:
            # 🔍 调试：记录DIVE_ESCAPE
            try:
                from enemy_ai_descent_debugger import get_debugger
                dbg = get_debugger()
                if dbg:
                    dbg.log_altitude_command_generated(
                        agent_id, 0, -300.0,
                        reason="DIVE_ESCAPE_ACTION"
                    )
            except:
                pass
            return self._execute_dive_escape(env, agent_id)'''
    
    content = content.replace(old_dive, new_dive)
    print("✓ 已在DIVE_ESCAPE处理添加调试日志")

# 保存修改
with open(refactor_helpers_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✓ 已在 {refactor_helpers_path} 中添加调试日志")

print("\n✅ 所有调试日志已添加！")
print("\n使用方法：")
print("  1. 设置环境变量启用调试: $env:ENEMY_DESCENT_DEBUG='1'")
print("  2. 运行仿真")
print("  3. 查看日志: scripts/tacticalProject/cap_results/descent_debug.log")
