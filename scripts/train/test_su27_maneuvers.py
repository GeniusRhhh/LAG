#!/usr/bin/env python
"""
SU-27机动动作测试脚本
测试训练好的模型能否执行各种机动动作
"""
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from envs.JSBSim.envs import SingleControlEnv
from algorithms.ppo.ppo_actor import PPOActor
from gymnasium import spaces


class ManeuverTester:
    """机动动作测试器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.actor = None
        self.load_model()
        
    def load_model(self):
        """加载模型"""
        print(f"📥 加载模型: {self.model_path}")
        
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
        print("✅ 模型加载成功\n")
    
    def test_maneuver(self, maneuver_name, duration=300, plot=True):
        """测试单个机动"""
        print("="*80)
        print(f"🎯 测试机动: {maneuver_name}")
        print("="*80)
        
        env = SingleControlEnv('1/heading_su27')
        obs = env.reset()
        
        rnn_states = np.zeros((1, 1, 128))
        masks = np.ones((1, 1))
        
        # 记录数据
        history = {
            'time': [],
            'altitude': [],
            'heading': [],
            'roll': [],
            'pitch': [],
            'velocity': [],
            'actions': []
        }
        
        for step in range(duration):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                rnn_states_tensor = torch.FloatTensor(rnn_states)
                masks_tensor = torch.FloatTensor(masks)
                
                action, _, rnn_states_out = self.actor(
                    obs_tensor,
                    rnn_states_tensor,
                    masks_tensor,
                    deterministic=True
                )
                
                rnn_states = rnn_states_out.cpu().numpy()
            
            obs, reward, done, info = env.step(action.cpu().numpy())
            
            agent_id = list(env.agents.keys())[0]
            agent = env.agents[agent_id]
            
            # 记录状态
            history['time'].append(step * 0.1)  # 假设每步0.1秒
            history['altitude'].append(agent.get_position()[2])
            history['heading'].append(agent.jsbsim_exec['attitude/psi-deg'])
            history['roll'].append(agent.jsbsim_exec['attitude/phi-deg'])
            history['pitch'].append(agent.jsbsim_exec['attitude/theta-deg'])
            history['velocity'].append(np.linalg.norm(agent.get_velocity()))
            history['actions'].append(action.cpu().numpy()[0])
            
            if done:
                break
        
        env.close()
        
        # 分析结果
        self.analyze_maneuver(maneuver_name, history)
        
        # 绘图
        if plot:
            self.plot_maneuver(maneuver_name, history)
        
        return history
    
    def analyze_maneuver(self, name, history):
        """分析机动结果"""
        alt = np.array(history['altitude'])
        hdg = np.array(history['heading'])
        roll = np.array(history['roll'])
        pitch = np.array(history['pitch'])
        vel = np.array(history['velocity'])
        
        print(f"\n📊 {name} 分析结果:")
        print(f"\n高度:")
        print(f"  范围: {alt.min():.1f}m ~ {alt.max():.1f}m")
        print(f"  变化: {alt.max() - alt.min():.1f}m")
        print(f"  标准差: {alt.std():.1f}m")
        
        print(f"\n航向:")
        print(f"  范围: {hdg.min():.1f}° ~ {hdg.max():.1f}°")
        print(f"  变化: {abs(hdg[-1] - hdg[0]):.1f}°")
        print(f"  标准差: {hdg.std():.1f}°")
        
        print(f"\n滚转角:")
        print(f"  最大: {abs(roll).max():.1f}°")
        print(f"  平均: {abs(roll).mean():.1f}°")
        
        print(f"\n俯仰角:")
        print(f"  范围: {pitch.min():.1f}° ~ {pitch.max():.1f}°")
        
        print(f"\n速度:")
        print(f"  范围: {vel.min():.1f}m/s ~ {vel.max():.1f}m/s")
        print(f"  平均: {vel.mean():.1f}m/s")
        
        # 评估机动能力
        print(f"\n✨ 机动能力评估:")
        
        # 高度控制
        if alt.max() - alt.min() > 1000:
            print("  ✅ 高度控制: 良好 (能够大幅度改变高度)")
        elif alt.max() - alt.min() > 500:
            print("  ⚠️  高度控制: 一般 (高度变化有限)")
        else:
            print("  ❌ 高度控制: 较差 (高度变化很小)")
        
        # 航向控制
        hdg_change = abs(hdg[-1] - hdg[0])
        if hdg_change > 90:
            print("  ✅ 航向控制: 良好 (能够大角度转向)")
        elif hdg_change > 45:
            print("  ⚠️  航向控制: 一般 (转向角度有限)")
        else:
            print("  ❌ 航向控制: 较差 (几乎不转向)")
        
        # 滚转能力
        max_roll = abs(roll).max()
        if max_roll > 30:
            print("  ✅ 滚转能力: 良好 (能够大角度滚转)")
        elif max_roll > 15:
            print("  ⚠️  滚转能力: 一般 (滚转角度有限)")
        else:
            print("  ❌ 滚转能力: 较差 (几乎不滚转)")
        
        # 稳定性
        if alt.std() < 200 and hdg.std() < 15:
            print("  ✅ 飞行稳定性: 良好")
        elif alt.std() < 400 and hdg.std() < 30:
            print("  ⚠️  飞行稳定性: 一般")
        else:
            print("  ❌ 飞行稳定性: 较差 (波动过大)")
    
    def plot_maneuver(self, name, history):
        """绘制机动轨迹"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'SU-27 Maneuver Test: {name}', fontsize=16)
        
        time = history['time']
        
        # 高度
        axes[0, 0].plot(time, history['altitude'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Altitude (m)')
        axes[0, 0].set_title('Altitude')
        axes[0, 0].grid(True)
        
        # 航向
        axes[0, 1].plot(time, history['heading'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Heading (deg)')
        axes[0, 1].set_title('Heading')
        axes[0, 1].grid(True)
        
        # 滚转角
        axes[1, 0].plot(time, history['roll'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Roll (deg)')
        axes[1, 0].set_title('Roll Angle')
        axes[1, 0].grid(True)
        
        # 俯仰角
        axes[1, 1].plot(time, history['pitch'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Pitch (deg)')
        axes[1, 1].set_title('Pitch Angle')
        axes[1, 1].grid(True)
        
        # 速度
        axes[2, 0].plot(time, history['velocity'], 'c-', linewidth=2)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Velocity (m/s)')
        axes[2, 0].set_title('Velocity')
        axes[2, 0].grid(True)
        
        # 动作
        actions = np.array(history['actions'])
        axes[2, 1].plot(time, actions[:, 0], label='Alt Cmd', alpha=0.7)
        axes[2, 1].plot(time, actions[:, 1], label='Hdg Cmd', alpha=0.7)
        axes[2, 1].plot(time, actions[:, 2], label='Vel Cmd', alpha=0.7)
        axes[2, 1].plot(time, actions[:, 3], label='Throttle', alpha=0.7)
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Action Value')
        axes[2, 1].set_title('Control Actions')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        save_dir = Path(__file__).parent / 'maneuver_plots'
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f'{name.replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 图表已保存: {save_path}")
        
        plt.show()
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*80)
        print("🚀 开始全面机动测试")
        print("="*80 + "\n")
        
        maneuvers = [
            ("平稳飞行", 200),
            ("高度变化", 300),
            ("航向转弯", 300),
            ("爬升机动", 250),
            ("俯冲机动", 250),
            ("盘旋机动", 400),
        ]
        
        results = {}
        for name, duration in maneuvers:
            results[name] = self.test_maneuver(name, duration, plot=True)
            print("\n")
        
        print("="*80)
        print("✅ 所有测试完成")
        print("="*80)
        
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SU-27机动测试')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--maneuver', type=str, choices=['stable', 'altitude', 'heading', 'climb', 'dive', 'turn', 'all'], 
                        default='all', help='测试的机动类型')
    parser.add_argument('--duration', type=int, default=300, help='测试持续时间(步数)')
    
    args = parser.parse_args()
    
    # 如果没有指定模型，使用最新的
    if not args.model:
        # 先找改进版本
        results_dir = Path(__file__).parent.parent / 'results' / 'SU27_Improved' / '1' / 'heading_su27' / 'ppo'
        if not results_dir.exists():
            # 找基础版本
            results_dir = Path(__file__).parent.parent / 'results' / 'SU27_Baseline' / '1' / 'heading_su27' / 'ppo' / 'check' / 'run2'
        
        if results_dir.exists():
            actor_files = list(results_dir.glob('**/actor_*.pt'))
            actor_files = [f for f in actor_files if f.stem != 'actor_latest']
            if actor_files:
                actor_files.sort(key=lambda x: int(x.stem.split('_')[1]))
                args.model = str(actor_files[-1])
                print(f"🔍 自动找到最新模型: {args.model}\n")
    
    if not args.model:
        print("❌ 未找到模型文件")
        print("请指定模型路径: --model <path>")
        return
    
    tester = ManeuverTester(args.model)
    
    if args.maneuver == 'all':
        tester.run_all_tests()
    else:
        maneuver_map = {
            'stable': '平稳飞行',
            'altitude': '高度变化',
            'heading': '航向转弯',
            'climb': '爬升机动',
            'dive': '俯冲机动',
            'turn': '盘旋机动'
        }
        tester.test_maneuver(maneuver_map[args.maneuver], args.duration)


if __name__ == '__main__':
    main()
