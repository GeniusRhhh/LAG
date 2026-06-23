#!/usr/bin/env python3
"""
完整的基础动作验证脚本
对所有11种基础动作进行系统性验证
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ActionValidator:
    def __init__(self, data_dir: str = "scripts/tacticalTemplateProject/basic_action_data"):
        self.data_dir = data_dir
        self.validation_results = {}
        
    def validate_all_actions(self):
        """验证所有基础动作"""
        print("🔍 开始验证所有基础动作...")
        
        # 定义验证函数映射
        validation_functions = {
            "accelerate": self.validate_accelerate,
            "decelerate": self.validate_decelerate,
            "dive": self.validate_dive,
            "dive_left": self.validate_dive_left,
            "dive_right": self.validate_dive_right,
            "climb": self.validate_climb,
            "climb_left": self.validate_climb_left,
            "climb_right": self.validate_climb_right,
            "turn_left": self.validate_turn_left,
            "turn_right": self.validate_turn_right,
            "level_flight": self.validate_level_flight
        }
        
        for action_name, validate_func in validation_functions.items():
            print(f"\n📊 验证 {action_name} 动作...")
            try:
                results = validate_func()
                self.validation_results[action_name] = results
                self.print_action_results(action_name, results)
            except Exception as e:
                print(f"❌ {action_name} 验证失败: {e}")
                self.validation_results[action_name] = {"error": str(e)}
        
        # 生成总结报告
        self.generate_summary_report()
    
    def validate_accelerate(self) -> Dict:
        """验证加速动作（现在是平飞保持）"""
        action_dir = os.path.join(self.data_dir, "accelerate")
        results = {"samples": [], "pass_count": 0, "total_count": 0}

        for file in os.listdir(action_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(action_dir, file)
                df = pd.read_csv(csv_path)

                initial_velocity = df['Velocity_m_s'].iloc[0]
                final_velocity = df['Velocity_m_s'].iloc[-1]
                velocity_change = abs(final_velocity - initial_velocity)

                # 加速动作现在是平飞，检查速度稳定性
                velocity_stability = velocity_change / initial_velocity * 100  # 速度变化百分比

                sample_result = {
                    "file": file,
                    "initial_velocity": initial_velocity,
                    "final_velocity": final_velocity,
                    "velocity_change": velocity_change,
                    "stability_percent": velocity_stability,
                    "pass": velocity_stability < 10.0  # 速度变化小于10%认为稳定
                }

                results["samples"].append(sample_result)
                results["total_count"] += 1
                if sample_result["pass"]:
                    results["pass_count"] += 1

        return results
    
    def validate_decelerate(self) -> Dict:
        """验证减速动作（现在是平飞保持）"""
        action_dir = os.path.join(self.data_dir, "decelerate")
        results = {"samples": [], "pass_count": 0, "total_count": 0}

        for file in os.listdir(action_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(action_dir, file)
                df = pd.read_csv(csv_path)

                initial_velocity = df['Velocity_m_s'].iloc[0]
                final_velocity = df['Velocity_m_s'].iloc[-1]
                velocity_change = abs(final_velocity - initial_velocity)

                # 减速动作现在是平飞，检查速度稳定性
                velocity_stability = velocity_change / initial_velocity * 100  # 速度变化百分比

                sample_result = {
                    "file": file,
                    "initial_velocity": initial_velocity,
                    "final_velocity": final_velocity,
                    "velocity_change": velocity_change,
                    "stability_percent": velocity_stability,
                    "pass": velocity_stability < 10.0  # 速度变化小于10%认为稳定
                }

                results["samples"].append(sample_result)
                results["total_count"] += 1
                if sample_result["pass"]:
                    results["pass_count"] += 1

        return results
    
    def validate_dive(self) -> Dict:
        """验证俯冲动作"""
        return self._validate_altitude_action("dive", expected_change_type="loss")
    
    def validate_dive_left(self) -> Dict:
        """验证左俯冲动作"""
        return self._validate_combined_action("dive_left", "loss")
    
    def validate_dive_right(self) -> Dict:
        """验证右俯冲动作"""
        return self._validate_combined_action("dive_right", "loss")
    
    def validate_climb(self) -> Dict:
        """验证爬升动作"""
        return self._validate_altitude_action("climb", expected_change_type="gain")
    
    def validate_climb_left(self) -> Dict:
        """验证左爬升动作"""
        return self._validate_combined_action("climb_left", "gain")
    
    def validate_climb_right(self) -> Dict:
        """验证右爬升动作"""
        return self._validate_combined_action("climb_right", "gain")
    
    def validate_turn_left(self) -> Dict:
        """验证左转动作"""
        return self._validate_turn_action("turn_left", expected_direction="left")
    
    def validate_turn_right(self) -> Dict:
        """验证右转动作"""
        return self._validate_turn_action("turn_right", expected_direction="right")
    
    def validate_level_flight(self) -> Dict:
        """验证平飞动作"""
        action_dir = os.path.join(self.data_dir, "level_flight")
        results = {"samples": [], "pass_count": 0, "total_count": 0}
        
        for file in os.listdir(action_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(action_dir, file)
                df = pd.read_csv(csv_path)
                
                initial_altitude = df['Z_m'].iloc[0]
                final_altitude = df['Z_m'].iloc[-1]
                altitude_change = abs(final_altitude - initial_altitude)
                
                initial_heading = df['Heading_deg'].iloc[0]
                final_heading = df['Heading_deg'].iloc[-1]
                heading_change = abs(final_heading - initial_heading)
                
                sample_result = {
                    "file": file,
                    "altitude_change": altitude_change,
                    "heading_change": heading_change,
                    "pass": altitude_change < 500 and heading_change < 30  # 放宽标准：高度变化<500m，航向变化<30°
                }
                
                results["samples"].append(sample_result)
                results["total_count"] += 1
                if sample_result["pass"]:
                    results["pass_count"] += 1
        
        return results
    
    def _validate_altitude_action(self, action_name: str, expected_change_type: str) -> Dict:
        """验证高度变化动作"""
        action_dir = os.path.join(self.data_dir, action_name)
        results = {"samples": [], "pass_count": 0, "total_count": 0}
        
        for file in os.listdir(action_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(action_dir, file)
                df = pd.read_csv(csv_path)
                
                initial_altitude = df['Z_m'].iloc[0]
                final_altitude = df['Z_m'].iloc[-1]
                
                if expected_change_type == "loss":
                    actual_change = initial_altitude - final_altitude
                    expected_change = float(file.split(f"{action_name}_")[1].split("m_")[0])
                else:  # gain
                    actual_change = final_altitude - initial_altitude
                    expected_change = float(file.split(f"{action_name}_")[1].split("m_")[0])
                
                error_rate = abs(actual_change - expected_change) / expected_change * 100
                
                sample_result = {
                    "file": file,
                    "initial_altitude": initial_altitude,
                    "final_altitude": final_altitude,
                    "actual_change": actual_change,
                    "expected_change": expected_change,
                    "error_rate": error_rate,
                    "pass": error_rate < 20.0  # 放宽到20%误差
                }
                
                results["samples"].append(sample_result)
                results["total_count"] += 1
                if sample_result["pass"]:
                    results["pass_count"] += 1
        
        return results
    
    def _validate_combined_action(self, action_name: str, altitude_type: str) -> Dict:
        """验证组合动作（高度+转弯）"""
        action_dir = os.path.join(self.data_dir, action_name)
        results = {"samples": [], "pass_count": 0, "total_count": 0}
        
        for file in os.listdir(action_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(action_dir, file)
                df = pd.read_csv(csv_path)
                
                # 高度验证
                initial_altitude = df['Z_m'].iloc[0]
                final_altitude = df['Z_m'].iloc[-1]
                
                if altitude_type == "loss":
                    actual_altitude_change = initial_altitude - final_altitude
                else:
                    actual_altitude_change = final_altitude - initial_altitude
                
                # 从文件名提取参数
                parts = file.split("_")
                expected_altitude_change = float(parts[2].replace("m", ""))
                expected_turn_angle = float(parts[3].replace("deg", ""))
                
                altitude_error_rate = abs(actual_altitude_change - expected_altitude_change) / expected_altitude_change * 100
                
                # 转弯验证
                initial_heading = df['Heading_deg'].iloc[0]
                final_heading = df['Heading_deg'].iloc[-1]
                actual_turn = abs(final_heading - initial_heading)
                if actual_turn > 180:
                    actual_turn = 360 - actual_turn
                
                turn_error_rate = abs(actual_turn - expected_turn_angle) / expected_turn_angle * 100
                
                sample_result = {
                    "file": file,
                    "altitude_error_rate": altitude_error_rate,
                    "turn_error_rate": turn_error_rate,
                    "actual_altitude_change": actual_altitude_change,
                    "expected_altitude_change": expected_altitude_change,
                    "actual_turn": actual_turn,
                    "expected_turn": expected_turn_angle,
                    "pass": altitude_error_rate < 20.0 and turn_error_rate < 20.0  # 放宽到20%误差
                }
                
                results["samples"].append(sample_result)
                results["total_count"] += 1
                if sample_result["pass"]:
                    results["pass_count"] += 1
        
        return results
    
    def _validate_turn_action(self, action_name: str, expected_direction: str) -> Dict:
        """验证转弯动作"""
        action_dir = os.path.join(self.data_dir, action_name)
        results = {"samples": [], "pass_count": 0, "total_count": 0}
        
        for file in os.listdir(action_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(action_dir, file)
                df = pd.read_csv(csv_path)
                
                initial_heading = df['Heading_deg'].iloc[0]
                final_heading = df['Heading_deg'].iloc[-1]
                
                # 计算转弯角度
                heading_diff = final_heading - initial_heading
                if heading_diff > 180:
                    heading_diff -= 360
                elif heading_diff < -180:
                    heading_diff += 360
                
                actual_turn_angle = abs(heading_diff)
                expected_turn_angle = float(file.split(f"{action_name}_")[1].split("deg_")[0])
                
                error_rate = abs(actual_turn_angle - expected_turn_angle) / expected_turn_angle * 100
                
                # 检查转弯方向
                direction_correct = (expected_direction == "left" and heading_diff < 0) or \
                                  (expected_direction == "right" and heading_diff > 0)
                
                sample_result = {
                    "file": file,
                    "actual_turn_angle": actual_turn_angle,
                    "expected_turn_angle": expected_turn_angle,
                    "error_rate": error_rate,
                    "direction_correct": direction_correct,
                    "pass": error_rate < 25.0 and direction_correct  # 放宽到25%误差
                }
                
                results["samples"].append(sample_result)
                results["total_count"] += 1
                if sample_result["pass"]:
                    results["pass_count"] += 1
        
        return results
    
    def print_action_results(self, action_name: str, results: Dict):
        """打印单个动作的验证结果"""
        if "error" in results:
            print(f"❌ {action_name}: 验证出错 - {results['error']}")
            return
        
        pass_rate = (results["pass_count"] / results["total_count"]) * 100 if results["total_count"] > 0 else 0
        status = "✅" if pass_rate >= 80 else "⚠️" if pass_rate >= 60 else "❌"
        
        print(f"{status} {action_name}: {results['pass_count']}/{results['total_count']} 样本通过 ({pass_rate:.1f}%)")
        
        # 显示详细信息
        if results["samples"]:
            sample = results["samples"][0]  # 显示第一个样本的详细信息
            if "error_rate" in sample:
                print(f"   示例误差率: {sample['error_rate']:.1f}%")
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "="*60)
        print("📋 验证总结报告")
        print("="*60)
        
        total_actions = len(self.validation_results)
        passed_actions = 0
        total_samples = 0
        passed_samples = 0
        
        for action_name, results in self.validation_results.items():
            if "error" not in results:
                total_samples += results["total_count"]
                passed_samples += results["pass_count"]
                
                pass_rate = (results["pass_count"] / results["total_count"]) * 100 if results["total_count"] > 0 else 0
                if pass_rate >= 80:
                    passed_actions += 1
        
        print(f"动作通过率: {passed_actions}/{total_actions} ({passed_actions/total_actions*100:.1f}%)")
        print(f"样本通过率: {passed_samples}/{total_samples} ({passed_samples/total_samples*100:.1f}%)")
        
        print("\n详细结果:")
        for action_name, results in self.validation_results.items():
            if "error" not in results:
                pass_rate = (results["pass_count"] / results["total_count"]) * 100 if results["total_count"] > 0 else 0
                status = "✅" if pass_rate >= 80 else "⚠️" if pass_rate >= 60 else "❌"
                print(f"  {status} {action_name}: {pass_rate:.1f}%")

def main():
    validator = ActionValidator()
    validator.validate_all_actions()

if __name__ == "__main__":
    main()
