#!/usr/bin/env python
"""
SU-27训练诊断脚本
分析训练过程中的问题，给出改进建议
"""
import sys
import os
import torch
import numpy as np
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from envs.JSBSim.envs import SingleControlEnv
from algorithms.ppo.ppo_actor import PPOActor
from gymnasium import spaces

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class TrainingDiagnostics:
    """训练诊断工具"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.actor = None
        self.env = None
        
    def load_model(self):
        """加载模型"""
        logging.info(f"📥 加载模型: {self.model_path}")
        
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
        
        try:
            self.actor = PPOActor(args, obs_space, act_space, device=torch.device('cpu'))
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
            self.actor.load_state_dict(checkpoint)
            self.actor.eval()
            logging.info("✅ 模型加载成功")
            return True
        except Exception as e:
            logging.error(f"❌ 模型加载失败: {e}")
            return False
    
    def analyze_action_distribution(self, num_samples=1000):
        """分析动作分布"""
        logging.info("\n" + "="*80)
        logging.info("📊 分析1: 动作分布")
        logging.info("="*80)
        
        env = SingleControlEnv('1/heading_su27')
        obs = env.reset()
        
        actions_history = []
        # rnn_states shape: (batch_size=1, num_layers=1, hidden_size=128)
        rnn_states = np.zeros((1, 1, 128))
        masks = np.ones((1, 1))
        
        for _ in range(num_samples):
            with torch.no_grad():
                # obs直接使用，不需要unsqueeze
                obs_tensor = torch.FloatTensor(obs)
                rnn_states_tensor = torch.FloatTensor(rnn_states)
                masks_tensor = torch.FloatTensor(masks)
                
                action, _, rnn_states_out = self.actor(
                    obs_tensor,
                    rnn_states_tensor,
                    masks_tensor,
                    deterministic=True
                )
                
                # 更新rnn_states
                rnn_states = rnn_states_out.cpu().numpy()
                actions_history.append(action.cpu().numpy()[0])
            
            obs, _, done, _ = env.step(action.cpu().numpy())
            if done:
                obs = env.reset()
                rnn_states = np.zeros((1, 1, 128))
        
        env.close()
        
        actions_history = np.array(actions_history)
        
        # 分析每个动作维度
        action_names = ['高度控制', '航向控制', '速度控制', '油门控制']
        action_ranges = [41, 41, 41, 30]
        
        logging.info("\n动作分布统计:")
        for i, (name, max_val) in enumerate(zip(action_names, action_ranges)):
            actions = actions_history[:, i]
            mean_action = np.mean(actions)
            std_action = np.std(actions)
            unique_actions = len(np.unique(actions))
            
            logging.info(f"\n{name}:")
            logging.info(f"  范围: 0-{max_val-1}")
            logging.info(f"  平均值: {mean_action:.2f}")
            logging.info(f"  标准差: {std_action:.2f}")
            logging.info(f"  使用的不同动作数: {unique_actions}/{max_val}")
            logging.info(f"  多样性: {unique_actions/max_val*100:.1f}%")
            
            # 判断是否有问题
            if std_action < 2.0:
                logging.warning(f"  ⚠️  动作变化太小，可能陷入局部最优")
            if unique_actions < max_val * 0.3:
                logging.warning(f"  ⚠️  动作空间利用不足")
            if abs(mean_action - max_val/2) > max_val * 0.3:
                logging.warning(f"  ⚠️  动作偏向一侧，可能存在偏差")
    
    def analyze_stability(self, duration=500):
        """分析稳定性"""
        logging.info("\n" + "="*80)
        logging.info("📊 分析2: 控制稳定性")
        logging.info("="*80)
        
        env = SingleControlEnv('1/heading_su27')
        obs = env.reset()
        
        # rnn_states shape: (batch_size=1, num_layers=1, hidden_size=128)
        rnn_states = np.zeros((1, 1, 128))
        masks = np.ones((1, 1))
        
        states_history = []
        
        for step in range(duration):
            with torch.no_grad():
                # obs直接使用，不需要unsqueeze
                obs_tensor = torch.FloatTensor(obs)
                rnn_states_tensor = torch.FloatTensor(rnn_states)
                masks_tensor = torch.FloatTensor(masks)
                
                action, _, rnn_states_out = self.actor(
                    obs_tensor,
                    rnn_states_tensor,
                    masks_tensor,
                    deterministic=True
                )
                
                # 更新rnn_states
                rnn_states = rnn_states_out.cpu().numpy()
            
            obs, _, done, _ = env.step(action.cpu().numpy())
            
            agent_id = list(env.agents.keys())[0]
            agent = env.agents[agent_id]
            
            state = {
                'altitude': agent.get_position()[2],
                'heading': agent.jsbsim_exec['attitude/psi-deg'],
                'velocity': np.linalg.norm(agent.get_velocity())
            }
            states_history.append(state)
            
            if done:
                break
        
        env.close()
        
        # 计算稳定性指标
        altitudes = [s['altitude'] for s in states_history]
        headings = [s['heading'] for s in states_history]
        velocities = [s['velocity'] for s in states_history]
        
        alt_std = np.std(altitudes)
        hdg_std = np.std(headings)
        vel_std = np.std(velocities)
        
        alt_drift = abs(altitudes[-1] - altitudes[0])
        hdg_drift = abs(headings[-1] - headings[0])
        vel_drift = abs(velocities[-1] - velocities[0])
        
        logging.info("\n稳定性分析:")
        logging.info(f"高度:")
        logging.info(f"  标准差: {alt_std:.1f}m")
        logging.info(f"  漂移: {alt_drift:.1f}m")
        if alt_std > 300:
            logging.warning(f"  ⚠️  高度波动过大")
        if alt_drift > 1000:
            logging.warning(f"  ⚠️  高度持续漂移")
        
        logging.info(f"\n航向:")
        logging.info(f"  标准差: {hdg_std:.1f}°")
        logging.info(f"  漂移: {hdg_drift:.1f}°")
        if hdg_std > 20:
            logging.warning(f"  ⚠️  航向波动过大")
        if hdg_drift > 50:
            logging.warning(f"  ⚠️  航向持续漂移")
        
        logging.info(f"\n速度:")
        logging.info(f"  标准差: {vel_std:.1f}m/s")
        logging.info(f"  漂移: {vel_drift:.1f}m/s")
        if vel_std > 30:
            logging.warning(f"  ⚠️  速度波动过大")
        if vel_drift > 80:
            logging.warning(f"  ⚠️  速度持续漂移")
    
    def compare_with_f16(self):
        """与F-16模型对比"""
        logging.info("\n" + "="*80)
        logging.info("📊 分析3: 与F-16模型对比")
        logging.info("="*80)
        
        f16_model_path = Path(__file__).parent.parent.parent / 'envs' / 'JSBSim' / 'model' / 'baseline_model.pt'
        
        if not f16_model_path.exists():
            logging.warning("⚠️  未找到F-16模型，跳过对比")
            return
        
        logging.info(f"F-16模型: {f16_model_path}")
        logging.info(f"SU-27模型: {self.model_path}")
        
        # 加载F-16模型
        from envs.JSBSim.model.baseline_actor import BaselineActor
        f16_actor = BaselineActor()
        try:
            checkpoint = torch.load(f16_model_path, map_location=torch.device('cpu'))
            f16_actor.load_state_dict(checkpoint)
            f16_actor.eval()
            logging.info("✅ F-16模型加载成功")
        except Exception as e:
            logging.error(f"❌ F-16模型加载失败: {e}")
            return
        
        # 对比模型结构
        su27_params = sum(p.numel() for p in self.actor.parameters())
        f16_params = sum(p.numel() for p in f16_actor.parameters())
        
        logging.info(f"\n模型参数量:")
        logging.info(f"  SU-27: {su27_params:,}")
        logging.info(f"  F-16: {f16_params:,}")
        
        if su27_params != f16_params:
            logging.warning(f"  ⚠️  参数量不同，模型结构可能不兼容")
    
    def generate_recommendations(self):
        """生成改进建议"""
        logging.info("\n" + "="*80)
        logging.info("💡 改进建议")
        logging.info("="*80)
        
        logging.info("\n基于诊断结果，建议:")
        logging.info("\n1. 训练策略:")
        logging.info("   - 当前训练5583个episodes可能不够")
        logging.info("   - 建议继续训练到至少10000个episodes")
        logging.info("   - 或增加总训练步数到2000万步")
        
        logging.info("\n2. 超参数调整:")
        logging.info("   - 提高学习率: 3e-4 -> 5e-4")
        logging.info("   - 增加PPO更新轮数: 10 -> 15")
        logging.info("   - 增加熵系数: 0.01 (鼓励探索)")
        
        logging.info("\n3. 环境配置:")
        logging.info("   - 增加并行环境数: 8 -> 16")
        logging.info("   - 检查SU-27配置文件是否合理")
        logging.info("   - 确保奖励函数适合SU-27")
        
        logging.info("\n4. 其他建议:")
        logging.info("   - 使用课程学习: 先训练简单任务，再训练复杂任务")
        logging.info("   - 数据增强: 随机化初始状态")
        logging.info("   - 定期评估: 每50个episode测试一次")
        
        logging.info("\n5. 快速改进方案:")
        logging.info("   运行: train_su27_improved.bat")
        logging.info("   该脚本会自动应用上述改进")
    
    def run_full_diagnosis(self):
        """运行完整诊断"""
        logging.info("="*80)
        logging.info("🔍 SU-27训练诊断")
        logging.info("="*80)
        
        if not self.load_model():
            return
        
        self.analyze_action_distribution()
        self.analyze_stability()
        self.compare_with_f16()
        self.generate_recommendations()
        
        logging.info("\n" + "="*80)
        logging.info("✅ 诊断完成")
        logging.info("="*80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SU-27训练诊断')
    parser.add_argument('--model', type=str, help='模型路径')
    
    args = parser.parse_args()
    
    # 如果没有指定模型，使用最新的
    if not args.model:
        results_dir = Path(__file__).parent.parent / 'results' / 'SU27_Baseline' / '1' / 'heading_su27' / 'ppo' / 'check' / 'run2'
        if results_dir.exists():
            actor_files = list(results_dir.glob('actor_*.pt'))
            actor_files = [f for f in actor_files if f.stem != 'actor_latest']
            if actor_files:
                actor_files.sort(key=lambda x: int(x.stem.split('_')[1]))
                args.model = str(actor_files[-1])
                logging.info(f"🔍 自动找到最新模型: {args.model}")
    
    if not args.model:
        logging.error("❌ 未找到模型文件")
        logging.info("请指定模型路径: --model <path>")
        return
    
    diagnostics = TrainingDiagnostics(args.model)
    diagnostics.run_full_diagnosis()


if __name__ == '__main__':
    main()
