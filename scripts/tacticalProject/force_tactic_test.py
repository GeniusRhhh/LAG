#!/usr/bin/env python3
"""
手动测试不同战术的运行脚本
通过修改仿真参数强制选择不同战术
"""
import os
import sys

# 修改当前的tactical_task.py来强制选择不同战术
def force_tactic_selection(tactic_name):
    """强制选择特定战术"""
    file_path = "tactical_task.py"
    
    # 读取当前文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份当前文件
    with open(file_path + '.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 修改战术选择逻辑
    old_logic = '''            result = self.complete_tactical_system.select_tactic(
                control_distance=current_phase.name.split('_')[0],  # NLT, MELD, MTR
                my_aircraft_list=my_aircraft_list,
                enemy_aircraft_list=enemy_aircraft_list,
                env=env
            )
            
            # result是一个元组: (tactic_name, roles)
            self.selected_tactic, self.tactic_roles = result'''
    
    new_logic = f'''            # 强制选择特定战术进行测试
            logging.info(f"🧪 测试模式：强制选择战术 {tactic_name}")
            self.selected_tactic = '{tactic_name}'
            
            # 设置默认角色分配
            if '{tactic_name}' == 'PINCER_ATTACK':
                self.tactic_roles = {{'A0100': 'left', 'A0200': 'right'}}
            elif '{tactic_name}' == 'HIGH_LOW_ATTACK':
                self.tactic_roles = {{'A0100': 'high', 'A0200': 'low'}}
            elif '{tactic_name}' == 'SEQUENTIAL_ATTACK':
                self.tactic_roles = {{'A0100': 'first', 'A0200': 'second'}}
            elif '{tactic_name}' == 'SIDE_BY_SIDE':
                self.tactic_roles = {{'A0100': 'left', 'A0200': 'right'}}
            else:  # DRAG_SHOOT
                self.tactic_roles = {{'A0100': 'lead', 'A0200': 'wingman'}}'''
    
    # 替换内容
    new_content = content.replace(old_logic, new_logic)
    
    # 写入修改后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ 已强制设置战术为: {tactic_name}")
    print("⚠️  请运行仿真测试，完成后请手动恢复文件")

def restore_original():
    """恢复原始文件"""
    file_path = "tactical_task.py"
    backup_path = file_path + '.backup'
    
    if os.path.exists(backup_path):
        with open(backup_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        os.remove(backup_path)
        print("✅ 已恢复原始战术选择逻辑")
    else:
        print("❌ 未找到备份文件")

if __name__ == "__main__":
    tactics = ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE']
    
    print("🎯 战术测试工具")
    print("可测试的战术:")
    for i, tactic in enumerate(tactics, 1):
        print(f"  {i}. {tactic}")
    print("  6. 恢复原始逻辑")
    
    choice = input("\n请选择要测试的战术 (1-6): ").strip()
    
    if choice == '6':
        restore_original()
    elif choice in ['1', '2', '3', '4', '5']:
        idx = int(choice) - 1
        selected_tactic = tactics[idx]
        force_tactic_selection(selected_tactic)
        
        print(f"\n现在可以运行以下命令测试 {selected_tactic} 战术:")
        print("C:\\Users\\ZRF\\.conda\\envs\\lag_gpu\\python.exe scripts\\tacticalProject\\run_tactical_simulation.py")
        print("\n测试完成后，请再次运行此脚本选择'6'来恢复原始逻辑")
    else:
        print("❌ 无效选择")