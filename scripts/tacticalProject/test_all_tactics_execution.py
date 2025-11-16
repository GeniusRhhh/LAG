#!/usr/bin/env python3
"""
测试所有战术模式的执行效果
自动化测试工具，逐个验证每种战术的具体行为
"""
import subprocess
import sys
import time
import os
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class TacticalTestRunner:
    """战术测试运行器"""
    
    def __init__(self):
        self.tactics = [
            ('DRAG_SHOOT', '拖拽射击战术'),
            ('PINCER_ATTACK', '钳形攻击战术'), 
            ('HIGH_LOW_ATTACK', '高低攻击战术'),
            ('SEQUENTIAL_ATTACK', '顺序攻击战术'),
            ('SIDE_BY_SIDE', '并肩作战战术')
        ]
        
        self.python_exe = r"C:\Users\ZRF\.conda\envs\lag_gpu\python.exe"
        self.force_test_script = "force_tactic_test.py"
        self.simulation_script = "run_tactical_simulation.py"
        self.test_results = {}
        
    def run_single_tactic_test(self, tactic_name, tactic_desc, test_duration=30):
        """
        运行单个战术的测试
        
        Args:
            tactic_name: 战术名称
            tactic_desc: 战术描述
            test_duration: 测试持续时间（秒）
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 开始测试: {tactic_desc} ({tactic_name})")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: 设置战术
            tactic_index = ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE'].index(tactic_name) + 1
            
            logger.info(f"📋 设置战术为: {tactic_name}")
            set_cmd = f'echo "{tactic_index}" | {self.python_exe} {self.force_test_script}'
            result = subprocess.run(set_cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.error(f"❌ 设置战术失败: {result.stderr}")
                return False
            
            logger.info(f"✅ 战术设置成功")
            
            # Step 2: 运行仿真测试
            logger.info(f"🚀 运行仿真测试 (持续{test_duration}秒)")
            sim_cmd = f"{self.python_exe} {self.simulation_script}"
            
            # 启动仿真进程
            process = subprocess.Popen(sim_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # 等待指定时间
            time.sleep(test_duration)
            
            # 终止进程
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            
            # Step 3: 分析输出
            self._analyze_test_output(tactic_name, stdout, stderr)
            
            logger.info(f"✅ {tactic_desc} 测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试 {tactic_name} 时出错: {e}")
            return False
        
        finally:
            # 恢复原始设置
            try:
                restore_cmd = f'echo "6" | {self.python_exe} {self.force_test_script}'
                subprocess.run(restore_cmd, shell=True, capture_output=True, text=True, timeout=10)
            except:
                pass
    
    def _analyze_test_output(self, tactic_name, stdout, stderr):
        """分析测试输出"""
        
        # 关键行为指标
        behaviors = {
            'tactic_selection': [],
            'phase_transitions': [],
            'maneuver_commands': [],
            'missile_launches': [],
            'formation_changes': [],
            'errors': []
        }
        
        lines = stdout.split('\n') + stderr.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 战术选择
            if '战术选择' in line or 'tactic' in line.lower():
                behaviors['tactic_selection'].append(line)
            
            # 阶段转换
            if '阶段' in line or 'phase' in line.lower():
                behaviors['phase_transitions'].append(line)
                
            # 机动指令
            if any(keyword in line for keyword in ['航向', '高度', '速度', 'heading', 'altitude', 'velocity']):
                behaviors['maneuver_commands'].append(line)
                
            # 导弹发射
            if '导弹' in line or 'missile' in line.lower():
                behaviors['missile_launches'].append(line)
                
            # 编队变化
            if '编队' in line or 'formation' in line.lower():
                behaviors['formation_changes'].append(line)
                
            # 错误信息
            if any(keyword in line.lower() for keyword in ['error', 'exception', '错误', '异常']):
                behaviors['errors'].append(line)
        
        # 保存分析结果
        self.test_results[tactic_name] = {
            'behaviors': behaviors,
            'total_lines': len(lines),
            'success': len(behaviors['errors']) == 0
        }
        
        # 输出关键行为摘要
        logger.info(f"📊 {tactic_name} 行为摘要:")
        logger.info(f"   战术选择: {len(behaviors['tactic_selection'])} 次")
        logger.info(f"   阶段转换: {len(behaviors['phase_transitions'])} 次")
        logger.info(f"   机动指令: {len(behaviors['maneuver_commands'])} 次")
        logger.info(f"   导弹发射: {len(behaviors['missile_launches'])} 次") 
        logger.info(f"   编队变化: {len(behaviors['formation_changes'])} 次")
        logger.info(f"   错误数量: {len(behaviors['errors'])} 个")
        
        # 显示前几个关键行为
        if behaviors['tactic_selection']:
            logger.info(f"   📋 战术选择示例: {behaviors['tactic_selection'][0]}")
        if behaviors['maneuver_commands']:
            logger.info(f"   ✈️  机动示例: {behaviors['maneuver_commands'][0]}")
        if behaviors['errors']:
            logger.warning(f"   ⚠️  错误示例: {behaviors['errors'][0]}")
    
    def run_all_tests(self, test_duration=30):
        """运行所有战术测试"""
        logger.info(f"\n🎯 开始完整战术测试")
        logger.info(f"📋 将测试 {len(self.tactics)} 种战术，每个持续 {test_duration} 秒")
        logger.info(f"🕒 预计总时间: {len(self.tactics) * (test_duration + 10)} 秒")
        
        successful_tests = 0
        
        for tactic_name, tactic_desc in self.tactics:
            if self.run_single_tactic_test(tactic_name, tactic_desc, test_duration):
                successful_tests += 1
            time.sleep(5)  # 测试间隔
        
        # 生成最终报告
        self._generate_final_report(successful_tests)
    
    def _generate_final_report(self, successful_tests):
        """生成最终测试报告"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 战术测试完整报告")
        logger.info(f"{'='*80}")
        
        logger.info(f"✅ 成功测试: {successful_tests}/{len(self.tactics)} 种战术")
        
        for tactic_name, tactic_desc in self.tactics:
            if tactic_name in self.test_results:
                result = self.test_results[tactic_name]
                status = "✅ 通过" if result['success'] else "❌ 失败"
                logger.info(f"   {tactic_desc:15} | {status:8} | 行为数据: {sum(len(behaviors) for behaviors in result['behaviors'].values())} 条")
        
        # 保存详细报告到文件
        self._save_detailed_report()
        
        logger.info(f"\n📄 详细报告已保存到: tactical_test_report.txt")
        logger.info(f"🎯 测试完成！")
    
    def _save_detailed_report(self):
        """保存详细报告到文件"""
        with open("tactical_test_report.txt", "w", encoding="utf-8") as f:
            f.write("LAG 战术系统测试报告\n")
            f.write("="*50 + "\n\n")
            
            for tactic_name, tactic_desc in self.tactics:
                f.write(f"战术: {tactic_desc} ({tactic_name})\n")
                f.write("-" * 30 + "\n")
                
                if tactic_name in self.test_results:
                    result = self.test_results[tactic_name]
                    
                    for behavior_type, behaviors in result['behaviors'].items():
                        f.write(f"\n{behavior_type} ({len(behaviors)} 条):\n")
                        for i, behavior in enumerate(behaviors[:5]):  # 只显示前5条
                            f.write(f"  {i+1}. {behavior}\n")
                        if len(behaviors) > 5:
                            f.write(f"  ... 另外 {len(behaviors)-5} 条\n")
                
                f.write("\n" + "="*50 + "\n")


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        test_duration = 15  # 快速测试
    else:
        test_duration = 30  # 标准测试
    
    runner = TacticalTestRunner()
    
    print("🎯 LAG 战术系统完整测试")
    print(f"📋 将测试所有 {len(runner.tactics)} 种战术模式")
    print(f"⏱️  每种战术测试 {test_duration} 秒")
    
    choice = input("\n是否开始测试? (y/N): ").strip().lower()
    if choice in ['y', 'yes']:
        runner.run_all_tests(test_duration)
    else:
        print("测试取消")


if __name__ == "__main__":
    main()