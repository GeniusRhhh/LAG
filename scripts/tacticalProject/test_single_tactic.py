#!/usr/bin/env python3
"""
单战术测试脚本
快速切换并测试单个战术模板的执行效果
"""
import os
import sys
import subprocess
import time

def set_tactic_and_run(tactic_name, duration=60):
    """设置战术并运行仿真"""
    
    tactics_map = {
        'DRAG_SHOOT': '1',
        'PINCER_ATTACK': '2', 
        'HIGH_LOW_ATTACK': '3',
        'SEQUENTIAL_ATTACK': '4',
        'SIDE_BY_SIDE': '5'
    }
    
    if tactic_name not in tactics_map:
        print(f"❌ 无效的战术名称: {tactic_name}")
        return False
    
    python_exe = r"C:\Users\ZRF\.conda\envs\lag_gpu\python.exe"
    
    print(f"🎯 设置战术为: {tactic_name}")
    print("=" * 50)
    
    try:
        # Step 1: 设置战术
        choice = tactics_map[tactic_name]
        set_cmd = f'echo "{choice}" | {python_exe} force_tactic_test.py'
        result = subprocess.run(set_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 战术设置成功")
        else:
            print(f"❌ 战术设置失败: {result.stderr}")
            return False
        
        # Step 2: 运行仿真
        print(f"🚀 开始运行 {tactic_name} 仿真测试...")
        print(f"⏱️  将运行 {duration} 秒，请观察输出中的战术行为")
        print("-" * 50)
        
        # 启动仿真
        sim_process = subprocess.Popen(
            [python_exe, "run_simulation.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时显示输出
        start_time = time.time()
        tactic_behaviors = []
        
        try:
            while True:
                output = sim_process.stdout.readline()
                if output == '' and sim_process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(line)
                    
                    # 收集战术相关信息
                    if any(keyword in line for keyword in ['战术', 'tactic', '编队', 'formation', '机动', 'maneuver']):
                        tactic_behaviors.append(line)
                
                # 检查是否超时
                if time.time() - start_time > duration:
                    print(f"\n⏰ {duration}秒测试完成，正在停止仿真...")
                    sim_process.terminate()
                    break
                    
        except KeyboardInterrupt:
            print("\n⚠️  用户中断测试")
            sim_process.terminate()
        
        # 等待进程结束
        try:
            sim_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sim_process.kill()
        
        # Step 3: 总结战术行为
        print("\n" + "=" * 60)
        print(f"📊 {tactic_name} 战术行为总结:")
        print("=" * 60)
        
        if tactic_behaviors:
            print("🎯 观察到的战术行为:")
            for i, behavior in enumerate(tactic_behaviors[:10], 1):  # 显示前10条
                print(f"  {i}. {behavior}")
            if len(tactic_behaviors) > 10:
                print(f"  ... 另外 {len(tactic_behaviors) - 10} 条行为记录")
        else:
            print("⚠️  未捕获到明显的战术行为信息")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        return False
    
    finally:
        # Step 4: 恢复原始设置
        print("\n🔧 恢复原始战术选择逻辑...")
        try:
            restore_cmd = f'echo "6" | {python_exe} force_tactic_test.py'
            subprocess.run(restore_cmd, shell=True, capture_output=True, text=True, timeout=10)
            print("✅ 已恢复原始设置")
        except:
            print("⚠️  恢复设置时出现问题，请手动运行 force_tactic_test.py 选择 6")

def main():
    """主函数"""
    tactics = [
        ('DRAG_SHOOT', '拖拽射击战术'),
        ('PINCER_ATTACK', '钳形攻击战术'),
        ('HIGH_LOW_ATTACK', '高低攻击战术'), 
        ('SEQUENTIAL_ATTACK', '顺序攻击战术'),
        ('SIDE_BY_SIDE', '并肩作战战术')
    ]
    
    print("🎯 LAG 单战术测试工具")
    print("=" * 60)
    
    # 显示可选战术
    for i, (tactic_name, desc) in enumerate(tactics, 1):
        print(f"  {i}. {desc} ({tactic_name})")
    print("  0. 退出")
    
    while True:
        try:
            choice = input("\n请选择要测试的战术 (0-5): ").strip()
            
            if choice == '0':
                print("退出测试")
                break
                
            if choice in ['1', '2', '3', '4', '5']:
                tactic_name, desc = tactics[int(choice) - 1]
                
                # 询问测试时长
                duration_input = input(f"测试时长(秒，默认60): ").strip()
                duration = int(duration_input) if duration_input.isdigit() else 60
                
                print(f"\n🎯 开始测试: {desc}")
                success = set_tactic_and_run(tactic_name, duration)
                
                if success:
                    print(f"✅ {desc} 测试完成")
                else:
                    print(f"❌ {desc} 测试失败")
                    
                input("\n按Enter继续...")
            else:
                print("❌ 无效选择，请输入 0-5")
                
        except KeyboardInterrupt:
            print("\n\n👋 测试已取消")
            break
        except ValueError:
            print("❌ 请输入数字")

if __name__ == "__main__":
    # 检查当前目录
    if not os.path.exists("force_tactic_test.py"):
        print("❌ 请在 scripts/tacticalProject 目录下运行此脚本")
        sys.exit(1)
    
    main()