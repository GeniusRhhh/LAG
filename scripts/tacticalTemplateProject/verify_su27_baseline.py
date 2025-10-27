#!/usr/bin/env python3
"""
Baseline模型验证脚本
验证基本机动能力：平稳飞行、转弯、爬升、下降

使用方法:
  python verify_su27_baseline.py [aircraft_type]
  
示例:
  python verify_su27_baseline.py su27    # 测试SU27模型
  python verify_su27_baseline.py f16     # 测试F16模型
  python verify_su27_baseline.py         # 默认测试SU27
"""

import os
import sys
import numpy as np
import torch
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from envs.JSBSim.envs import SingleControlEnv
from envs.JSBSim.utils.utils import get_root_dir
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
from envs.JSBSim.model.baseline_actor import BaselineActor
from gymnasium import spaces


class BaselineVerifier:
    """Baseline模型验证器"""
    
    def __init__(self, aircraft_type='su27'):
        self.actor = None
        self.baseline_policy = None
        self.rnn_state = None
        self.env = None
        self.aircraft_type = aircraft_type.lower()
        
        # 设置模型路径和环境
        if self.aircraft_type == 'f16':
            self.model_path = get_root_dir() + '/model/baseline_model.pt'
            self.env_name = '1/heading'
            self.use_ppo = False  # F16使用BaselineActor
        elif self.aircraft_type == 'su27':
            self.model_path = get_root_dir() + '/model/su27_baseline.pt'
            self.env_name = '1/heading_su27'
            self.use_ppo = True   # SU27使用PPOActor
        else:
            raise ValueError(f"不支持的飞机类型: {aircraft_type}")
        
        # 高层指令数组 - 完全按照pure_maneuver_task的定义
        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0
        self.norm_delta_heading = np.array([
            -np.pi,           # -180°
            -2*np.pi/3,       # -120°
            -np.pi/2,         # -90°
            -5*np.pi/12,      # -75°
            -np.pi/3,         # -60°
            -np.pi/4,         # -45°
            -np.pi/6,         # -30°
            -np.pi/12,        # -15°
            0,                # 0°
            np.pi/12,         # 15°
            np.pi/6,          # 30°
            np.pi/4,          # 45°
            np.pi/3,          # 60°
            5*np.pi/12,       # 75°
            np.pi/2,          # 90°
            2*np.pi/3,        # 120°
            np.pi             # 180°
        ])
        self.norm_delta_velocity = np.array([-200, -150, -100, 0, 50, 100, 200]) / 5.0
        
    def load_model(self):
        """加载模型"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ 模型文件不存在: {self.model_path}")
                return False
            
            if self.use_ppo:
                # SU27使用PPO模型
                class Args:
                    hidden_size = "128 128"
                    act_hidden_size = "128 128"
                    activation_id = 1
                    use_feature_normalization = False
                    gain = 0.01
                    use_recurrent_policy = True
                    recurrent_hidden_size = 128
                    recurrent_hidden_layers = 1
                    use_naive_recurrent_policy = False
                    use_mlp_actlayer = True
                    use_prior = False
                
                args = Args()
                obs_space = spaces.Box(low=-10, high=10., shape=(12,), dtype=np.float32)
                act_space = spaces.MultiDiscrete([41, 41, 41, 30])
                
                self.actor = PPOActor(args, obs_space, act_space, device=torch.device('cpu'))
                checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
                self.actor.load_state_dict(checkpoint)
                self.actor.eval()
            else:
                # F16使用BaselineActor
                self.baseline_policy = BaselineActor()
                self.baseline_policy.load_state_dict(
                    torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
                )
                self.baseline_policy.eval()
            
            self.rnn_state = np.zeros((1, 1, 128))
            
            print(f"✅ 成功加载{self.aircraft_type.upper()}模型: {self.model_path}\n")
            return True
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_env(self):
        """创建环境"""
        try:
            self.env = SingleControlEnv(self.env_name)
            self.env.reset()
            print(f"✅ {self.aircraft_type.upper()}测试环境创建成功\n")
            return True
        except Exception as e:
            print(f"❌ 创建环境失败: {e}")
            return False
    
    def get_state(self):
        """获取飞机状态"""
        agent_id = list(self.env.agents.keys())[0]
        aircraft = self.env.agents[agent_id]
        
        return {
            'altitude': aircraft.get_property_value(c.position_h_sl_m),
            'heading': np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad)),
            'velocity': aircraft.get_property_value(c.velocities_u_mps),
            'roll': np.rad2deg(aircraft.get_property_value(c.attitude_phi_rad)),
            'pitch': np.rad2deg(aircraft.get_property_value(c.attitude_theta_rad))
        }
    
    def get_obs(self):
        """获取观测 - 完全按照战术项目的方式"""
        agent_id = list(self.env.agents.keys())[0]
        aircraft = self.env.agents[agent_id]
        
        position = aircraft.get_position()
        velocity = aircraft.get_velocity()
        
        # 完全按照run_drag_shoot_simulation.py的get_aircraft_obs
        obs = np.array([
            position[2] / 10000.0,  # 归一化高度
            np.linalg.norm(velocity) / 300.0,  # 归一化总速度
            aircraft.get_property_value(c.attitude_phi_rad),  # roll
            aircraft.get_property_value(c.attitude_theta_rad),  # pitch
            aircraft.get_property_value(c.attitude_psi_rad),  # yaw
            velocity[0] / 300.0,  # x速度
            velocity[1] / 300.0,  # y速度
            velocity[2] / 300.0,  # z速度
            0.0  # 填充
        ])
        
        return obs
    
    
    def verify_maneuver(self, name, altitude_diff, heading_diff, velocity_diff, duration_steps=200):
        """验证单个机动 - 使用baseline模型"""
        print(f"\n{'='*60}")
        print(f"验证机动: {name}")
        print(f"指令 - 高度差: {altitude_diff}m, 航向差: {heading_diff}°, 速度差: {velocity_diff}m/s")
        print(f"{'='*60}")
        
        initial = self.get_state()
        print(f"初始状态:")
        print(f"  高度: {initial['altitude']:.1f}m, 航向: {initial['heading']:.1f}°, 速度: {initial['velocity']:.1f}m/s")
        
        agent_id = list(self.env.agents.keys())[0]
        terminated = False
        
        for step in range(duration_steps):
            # 获取高层指令索引 - 完全按照pure_maneuver_task的转换方式
            # 高度：直接用米数匹配
            altitude_values = self.norm_delta_altitude * 1000.0
            altitude_cmd_id = np.argmin(np.abs(altitude_values - altitude_diff))
            
            # 航向：先转弧度，再匹配
            heading_diff_rad = np.deg2rad(heading_diff)
            heading_cmd_id = np.argmin(np.abs(self.norm_delta_heading - heading_diff_rad))
            
            # 速度：直接用m/s匹配
            velocity_values = self.norm_delta_velocity * 5.0
            velocity_cmd_id = np.argmin(np.abs(velocity_values - velocity_diff))
            
            # 获取当前观测（9维飞机状态）
            raw_obs = self.get_obs()
            
            # 构建12维输入：[指令1, 指令2, 指令3, 观测1-9]
            input_obs = np.zeros(12)
            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.nan_to_num(input_obs, nan=0.0)
            
            # 调用baseline模型获取底层控制
            if self.use_ppo:
                # SU27: PPO模型
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(input_obs).unsqueeze(0)
                    rnn_states_tensor = torch.FloatTensor(self.rnn_state)
                    masks_tensor = torch.ones((1, 1))
                    
                    action, _, rnn_states_out = self.actor(
                        obs_tensor,
                        rnn_states_tensor,
                        masks_tensor,
                        deterministic=True
                    )
                    
                    self.rnn_state = rnn_states_out.cpu().numpy()
                    action_indices = action.cpu().numpy()[0]
                
                # 转换为归一化动作
                norm_act = np.zeros(4)
                norm_act[0] = action_indices[0] / 20 - 1.
                norm_act[1] = action_indices[1] / 20 - 1.
                norm_act[2] = action_indices[2] / 20 - 1.
                norm_act[3] = action_indices[3] / 58 + 0.4
            else:
                # F16: BaselineActor模型
                input_obs_expanded = np.expand_dims(input_obs, axis=0)
                
                _action, _rnn_states = self.baseline_policy(
                    torch.FloatTensor(input_obs_expanded),
                    torch.FloatTensor(self.rnn_state)
                )
                
                action_output = _action.detach().cpu().numpy().squeeze(0)
                self.rnn_state = _rnn_states.detach().cpu().numpy()
                
                # 转换为归一化动作
                norm_act = np.zeros(4)
                norm_act[0] = action_output[0] / 20 - 1.
                norm_act[1] = action_output[1] / 20 - 1.
                norm_act[2] = action_output[2] / 20 - 1.
                norm_act[3] = action_output[3] / 58 + 0.4
            
            # 应用底层控制
            aircraft = self.env.agents[agent_id]
            aircraft.set_property_value(c.fcs_aileron_cmd_norm, norm_act[0])
            aircraft.set_property_value(c.fcs_elevator_cmd_norm, norm_act[1])
            aircraft.set_property_value(c.fcs_rudder_cmd_norm, norm_act[2])
            aircraft.set_property_value(c.fcs_throttle_cmd_norm, norm_act[3])
            
            # 运行仿真
            for _ in range(self.env.agent_interaction_steps):
                aircraft.run()
            
            # 更新环境步数
            self.env.current_step += 1
            
            # 检查终止条件
            done, info = self.env.task.get_termination(self.env, agent_id, {})
            
            if done:
                print(f"⚠️  飞机在第{step+1}步终止")
                terminated = True
                break
            
            if (step + 1) % 50 == 0:
                current = self.get_state()
                print(f"步数 {step+1}: 高度 {current['altitude']:.1f}m, 航向 {current['heading']:.1f}°, 速度 {current['velocity']:.1f}m/s")
        
        final = self.get_state()
        print(f"\n最终状态:")
        print(f"  高度: {final['altitude']:.1f}m (变化: {final['altitude']-initial['altitude']:.1f}m)")
        print(f"  航向: {final['heading']:.1f}° (变化: {final['heading']-initial['heading']:.1f}°)")
        print(f"  速度: {final['velocity']:.1f}m/s (变化: {final['velocity']-initial['velocity']:.1f}m/s)")
        
        # 评估结果
        result = {
            'name': name,
            'target_alt': altitude_diff,
            'target_hdg': heading_diff,
            'target_vel': velocity_diff,
            'actual_alt_change': final['altitude'] - initial['altitude'],
            'actual_hdg_change': final['heading'] - initial['heading'],
            'actual_vel_change': final['velocity'] - initial['velocity'],
            'terminated': terminated
        }
        
        return result
    
    def run_verification(self):
        """运行验证"""
        print("\n" + "="*60)
        print(f"{self.aircraft_type.upper()} Baseline模型验证")
        print("="*60)
        
        if not self.load_model():
            return False
        
        if not self.create_env():
            return False
        
        test_cases = [
            ("平稳飞行", 0, 0, 0, 200),
            ("右转弯 45°", 0, 45, 0, 200),
            ("左转弯 -45°", 0, -45, 0, 200),
            ("爬升 +500m", 500, 0, 0, 300),
            ("下降 -200m", -200, 0, 0, 300),
            ("组合机动（右转+爬升）", 500, 45, 0, 300),
        ]
        
        results = []
        for name, alt_diff, hdg_diff, vel_diff, duration in test_cases:
            try:
                result = self.verify_maneuver(name, alt_diff, hdg_diff, vel_diff, duration)
                results.append(result)
                self.env.reset()
                self.rnn_state = np.zeros((1, 1, 128))
            except Exception as e:
                print(f"❌ 验证 {name} 失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({'name': name, 'terminated': True, 'error': str(e)})
        
        # 打印汇总报告
        self.print_summary(results)
        
        return True
    
    def print_summary(self, results):
        """打印验证结果汇总"""
        print("\n" + "="*60)
        print("📊 验证结果汇总")
        print("="*60)
        
        success_count = 0
        fail_count = 0
        
        for r in results:
            if 'error' in r:
                continue
                
            name = r['name']
            target_alt = r['target_alt']
            target_hdg = r['target_hdg']
            actual_alt = r['actual_alt_change']
            actual_hdg = r['actual_hdg_change']
            terminated = r['terminated']
            
            # 评估是否成功
            alt_success = abs(actual_alt - target_alt) < 200 if target_alt != 0 else abs(actual_alt) < 500
            hdg_success = abs(actual_hdg - target_hdg) < 20 if target_hdg != 0 else abs(actual_hdg) < 30
            
            if alt_success and hdg_success and not terminated:
                status = "✅ 通过"
                success_count += 1
            else:
                status = "❌ 失败"
                fail_count += 1
            
            print(f"\n{name}: {status}")
            if target_alt != 0:
                print(f"  高度控制: 目标{target_alt:+.0f}m, 实际{actual_alt:+.0f}m, 误差{abs(actual_alt-target_alt):.0f}m")
            if target_hdg != 0:
                print(f"  航向控制: 目标{target_hdg:+.0f}°, 实际{actual_hdg:+.0f}°, 误差{abs(actual_hdg-target_hdg):.0f}°")
            if terminated:
                print(f"  ⚠️  飞机提前终止（坠毁或极端状态）")
        
        print("\n" + "="*60)
        print(f"总计: {len(results)}项测试, {success_count}项通过, {fail_count}项失败")
        
        # 总体评价
        if success_count == 0:
            print("\n❌ 模型验证失败！模型无法执行基本机动动作。")
            print("建议:")
            print("  1. 检查模型训练是否充分")
            print("  2. 检查模型是否与环境匹配")
            print("  3. 考虑重新训练或调整训练参数")
        elif success_count < len(results) / 2:
            print("\n⚠️  模型表现较差，大部分机动无法正确执行。")
            print("建议继续训练或调整模型。")
        elif success_count < len(results):
            print("\n⚠️  模型基本可用，但部分机动存在问题。")
            print("建议针对失败的机动进行优化。")
        else:
            print("\n✅ 模型验证通过！可以执行各种基本机动动作。")
        
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline模型验证测试')
    parser.add_argument('aircraft', nargs='?', default='f16',
                        choices=['su27', 'f16'],
                        help='飞机类型 (su27 或 f16，默认: su27)')
    args = parser.parse_args()
    
    verifier = BaselineVerifier(aircraft_type=args.aircraft)
    verifier.run_verification()
