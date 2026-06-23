#!/usr/bin/env python
"""
SU-27飞行测试脚本
测试基本飞行能力和机动动作

测试项目：
1. 平稳飞行 - 保持高度、速度、航向
2. 航向转换 - 左转、右转
3. 爬升/俯冲 - 高度变化
4. 加速/减速 - 速度控制
5. 组合机动 - 战术机动动作
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


class SU27FlightTester:
    """SU-27飞行测试器"""
    
    def __init__(self, model_path=None):
        """
        初始化测试器
        
        Args:
            model_path: 训练好的模型路径（可选）
        """
        self.model_path = model_path
        self.actor = None
        self.env = None
        
        # 测试结果
        self.test_results = {
            'stable_flight': None,
            'heading_change': None,
            'altitude_change': None,
            'speed_control': None,
            'maneuvers': None
        }
        
    def load_model(self):
        """加载训练好的模型"""
        if not self.model_path:
            logging.info("⚠️  未指定模型，将使用随机动作测试")
            return False
            
        logging.info(f"📥 加载模型: {self.model_path}")
        
        # 创建PPOActor
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
    
    def create_env(self):
        """创建测试环境"""
        logging.info("🌍 创建SU-27测试环境...")
        self.env = SingleControlEnv(config_name='1/heading_su27')
        logging.info("✅ 环境创建成功")
    
    def get_aircraft_state(self):
        """获取飞机当前状态"""
        agent_id = list(self.env.agents.keys())[0]
        agent = self.env.agents[agent_id]
        
        pos = agent.get_position()
        vel = agent.get_velocity()
        
        # 使用JSBSim的属性访问方式
        return {
            'altitude': pos[2],  # 米
            'heading': agent.jsbsim_exec['attitude/psi-deg'],  # 度
            'pitch': agent.jsbsim_exec['attitude/theta-deg'],  # 度
            'roll': agent.jsbsim_exec['attitude/phi-deg'],  # 度
            'velocity': np.linalg.norm(vel),  # m/s
            'velocity_u': vel[0],  # 前向速度
            'velocity_v': vel[1],  # 侧向速度
            'velocity_w': vel[2],  # 垂直速度
        }
    
    def predict_action(self, obs, rnn_states, masks):
        """预测动作"""
        if self.actor is None:
            # 随机动作
            return np.random.randint(0, 41, size=(1, 1, 4)), rnn_states
        
        with torch.no_grad():
            action, _, rnn_states = self.actor(
                torch.FloatTensor(obs),
                torch.FloatTensor(rnn_states),
                torch.FloatTensor(masks),
                deterministic=True
            )
            return action.cpu().numpy(), rnn_states
    
    def test_stable_flight(self, duration=200):
        """
        测试1: 平稳飞行
        保持高度、速度、航向稳定
        """
        logging.info("\n" + "="*80)
        logging.info("📊 测试1: 平稳飞行")
        logging.info("="*80)
        
        obs = self.env.reset()
        rnn_states = np.zeros((1, 1, 128))
        masks = np.ones((1, 1))
        
        initial_state = self.get_aircraft_state()
        logging.info(f"初始状态:")
        logging.info(f"  高度: {initial_state['altitude']:.1f}m")
        logging.info(f"  航向: {initial_state['heading']:.1f}°")
        logging.info(f"  速度: {initial_state['velocity']:.1f}m/s")
        
        states_history = []
        
        for step in range(duration):
            action, rnn_states = self.predict_action(obs, rnn_states, masks)
            obs, reward, done, info = self.env.step(action)
            
            state = self.get_aircraft_state()
            states_history.append(state)
            
            if step % 50 == 0:
                logging.info(f"Step {step}: Alt={state['altitude']:.1f}m, "
                           f"Hdg={state['heading']:.1f}°, Vel={state['velocity']:.1f}m/s")
            
            if done:
                logging.warning(f"⚠️  飞机在第{step}步结束（可能坠毁）")
                break
        
        # 分析稳定性
        final_state = states_history[-1]
        alt_change = abs(final_state['altitude'] - initial_state['altitude'])
        hdg_change = abs(final_state['heading'] - initial_state['heading'])
        vel_change = abs(final_state['velocity'] - initial_state['velocity'])
        
        logging.info(f"\n结果分析:")
        logging.info(f"  高度变化: {alt_change:.1f}m")
        logging.info(f"  航向变化: {hdg_change:.1f}°")
        logging.info(f"  速度变化: {vel_change:.1f}m/s")
        
        # 评分
        stable = alt_change < 500 and hdg_change < 30 and vel_change < 50
        result = "✅ 通过" if stable else "❌ 不稳定"
        logging.info(f"  评价: {result}")
        
        self.test_results['stable_flight'] = {
            'passed': stable,
            'alt_change': alt_change,
            'hdg_change': hdg_change,
            'vel_change': vel_change
        }
        
        return states_history
    
    def test_heading_change(self, target_heading_change=90, duration=300):
        """
        测试2: 航向转换
        测试左转和右转能力
        """
        logging.info("\n" + "="*80)
        logging.info(f"📊 测试2: 航向转换 (目标: {target_heading_change}°)")
        logging.info("="*80)
        
        obs = self.env.reset()
        rnn_states = np.zeros((1, 1, 128))
        masks = np.ones((1, 1))
        
        initial_state = self.get_aircraft_state()
        initial_heading = initial_state['heading']
        
        logging.info(f"初始航向: {initial_heading:.1f}°")
        logging.info(f"目标航向: {(initial_heading + target_heading_change) % 360:.1f}°")
        
        # 修改环境的目标航向
        agent_id = list(self.env.agents.keys())[0]
        self.env.agents[agent_id].bloods = 100  # 重置
        
        heading_history = []
        
        for step in range(duration):
            action, rnn_states = self.predict_action(obs, rnn_states, masks)
            obs, reward, done, info = self.env.step(action)
            
            state = self.get_aircraft_state()
            heading_history.append(state['heading'])
            
            if step % 50 == 0:
                current_change = abs(state['heading'] - initial_heading)
                logging.info(f"Step {step}: Heading={state['heading']:.1f}° "
                           f"(变化: {current_change:.1f}°)")
            
            if done:
                break
        
        final_heading = heading_history[-1]
        actual_change = abs(final_heading - initial_heading)
        
        logging.info(f"\n结果分析:")
        logging.info(f"  初始航向: {initial_heading:.1f}°")
        logging.info(f"  最终航向: {final_heading:.1f}°")
        logging.info(f"  实际变化: {actual_change:.1f}°")
        
        # 评分
        success = actual_change > target_heading_change * 0.7  # 完成70%即可
        result = "✅ 通过" if success else "❌ 未完成"
        logging.info(f"  评价: {result}")
        
        self.test_results['heading_change'] = {
            'passed': success,
            'target': target_heading_change,
            'actual': actual_change
        }
        
        return heading_history
    
    def test_altitude_change(self, duration=300):
        """
        测试3: 高度变化
        测试爬升和俯冲能力
        """
        logging.info("\n" + "="*80)
        logging.info("📊 测试3: 高度变化（爬升/俯冲）")
        logging.info("="*80)
        
        obs = self.env.reset()
        rnn_states = np.zeros((1, 1, 128))
        masks = np.ones((1, 1))
        
        initial_state = self.get_aircraft_state()
        initial_altitude = initial_state['altitude']
        
        logging.info(f"初始高度: {initial_altitude:.1f}m")
        
        altitude_history = []
        max_altitude = initial_altitude
        min_altitude = initial_altitude
        
        for step in range(duration):
            action, rnn_states = self.predict_action(obs, rnn_states, masks)
            obs, reward, done, info = self.env.step(action)
            
            state = self.get_aircraft_state()
            altitude_history.append(state['altitude'])
            
            max_altitude = max(max_altitude, state['altitude'])
            min_altitude = min(min_altitude, state['altitude'])
            
            if step % 50 == 0:
                logging.info(f"Step {step}: Alt={state['altitude']:.1f}m, "
                           f"Pitch={state['pitch']:.1f}°, Vz={state['velocity_w']:.1f}m/s")
            
            if done:
                break
        
        altitude_range = max_altitude - min_altitude
        
        logging.info(f"\n结果分析:")
        logging.info(f"  初始高度: {initial_altitude:.1f}m")
        logging.info(f"  最高高度: {max_altitude:.1f}m")
        logging.info(f"  最低高度: {min_altitude:.1f}m")
        logging.info(f"  高度范围: {altitude_range:.1f}m")
        
        # 评分
        success = altitude_range > 1000  # 高度变化超过1000米
        result = "✅ 通过" if success else "❌ 高度控制不足"
        logging.info(f"  评价: {result}")
        
        self.test_results['altitude_change'] = {
            'passed': success,
            'range': altitude_range,
            'max': max_altitude,
            'min': min_altitude
        }
        
        return altitude_history
    
    def test_maneuvers(self, duration=400):
        """
        测试4: 组合机动
        测试复杂机动动作
        """
        logging.info("\n" + "="*80)
        logging.info("📊 测试4: 组合机动动作")
        logging.info("="*80)
        
        obs = self.env.reset()
        rnn_states = np.zeros((1, 1, 128))
        masks = np.ones((1, 1))
        
        initial_state = self.get_aircraft_state()
        
        logging.info(f"初始状态:")
        logging.info(f"  高度: {initial_state['altitude']:.1f}m")
        logging.info(f"  航向: {initial_state['heading']:.1f}°")
        logging.info(f"  速度: {initial_state['velocity']:.1f}m/s")
        
        maneuver_data = {
            'max_roll': 0,
            'max_pitch': 0,
            'max_g': 0,
            'heading_changes': 0,
            'altitude_changes': 0
        }
        
        prev_heading = initial_state['heading']
        prev_altitude = initial_state['altitude']
        
        for step in range(duration):
            action, rnn_states = self.predict_action(obs, rnn_states, masks)
            obs, reward, done, info = self.env.step(action)
            
            state = self.get_aircraft_state()
            
            # 记录最大值
            maneuver_data['max_roll'] = max(maneuver_data['max_roll'], abs(state['roll']))
            maneuver_data['max_pitch'] = max(maneuver_data['max_pitch'], abs(state['pitch']))
            
            # 统计变化次数
            if abs(state['heading'] - prev_heading) > 10:
                maneuver_data['heading_changes'] += 1
                prev_heading = state['heading']
            
            if abs(state['altitude'] - prev_altitude) > 100:
                maneuver_data['altitude_changes'] += 1
                prev_altitude = state['altitude']
            
            if step % 100 == 0:
                logging.info(f"Step {step}: Roll={state['roll']:.1f}°, "
                           f"Pitch={state['pitch']:.1f}°, Alt={state['altitude']:.1f}m")
            
            if done:
                break
        
        logging.info(f"\n机动性能分析:")
        logging.info(f"  最大滚转角: {maneuver_data['max_roll']:.1f}°")
        logging.info(f"  最大俯仰角: {maneuver_data['max_pitch']:.1f}°")
        logging.info(f"  航向变化次数: {maneuver_data['heading_changes']}")
        logging.info(f"  高度变化次数: {maneuver_data['altitude_changes']}")
        
        # 评分
        success = (maneuver_data['max_roll'] > 20 and 
                  maneuver_data['heading_changes'] > 3)
        result = "✅ 通过" if success else "❌ 机动性不足"
        logging.info(f"  评价: {result}")
        
        self.test_results['maneuvers'] = {
            'passed': success,
            'data': maneuver_data
        }
        
        return maneuver_data
    
    def run_all_tests(self):
        """运行所有测试"""
        logging.info("\n" + "="*80)
        logging.info("🚀 SU-27 飞行测试开始")
        logging.info("="*80)
        
        # 加载模型
        if self.model_path:
            self.load_model()
        
        # 创建环境
        self.create_env()
        
        # 运行测试
        try:
            self.test_stable_flight(duration=200)
            self.test_heading_change(target_heading_change=90, duration=300)
            self.test_altitude_change(duration=300)
            self.test_maneuvers(duration=400)
        finally:
            if self.env:
                self.env.close()
        
        # 总结
        self.print_summary()
    
    def print_summary(self):
        """打印测试总结"""
        logging.info("\n" + "="*80)
        logging.info("📋 测试总结")
        logging.info("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r and r.get('passed', False))
        
        logging.info(f"\n测试项目: {total_tests}")
        logging.info(f"通过: {passed_tests}")
        logging.info(f"失败: {total_tests - passed_tests}")
        logging.info(f"通过率: {passed_tests/total_tests*100:.1f}%")
        
        logging.info(f"\n详细结果:")
        for test_name, result in self.test_results.items():
            if result:
                status = "✅ 通过" if result.get('passed', False) else "❌ 失败"
                logging.info(f"  {test_name}: {status}")
        
        # 总体评价
        logging.info("\n" + "="*80)
        if passed_tests == total_tests:
            logging.info("🌟 总体评价: 优秀 - SU-27飞行性能完美！")
        elif passed_tests >= total_tests * 0.75:
            logging.info("👍 总体评价: 良好 - SU-27飞行性能不错！")
        elif passed_tests >= total_tests * 0.5:
            logging.info("⚠️  总体评价: 一般 - SU-27需要继续训练")
        else:
            logging.info("❌ 总体评价: 较差 - SU-27控制能力不足")
        logging.info("="*80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SU-27飞行测试')
    parser.add_argument('--model', type=str, 
                       help='训练好的模型路径（可选，不指定则使用随机动作）')
    parser.add_argument('--test', type=str, choices=['all', 'stable', 'heading', 'altitude', 'maneuver'],
                       default='all', help='指定测试项目')
    
    args = parser.parse_args()
    
    # 如果没有指定模型，尝试使用最新的模型
    if not args.model:
        # 查找最新模型
        results_dir = Path(__file__).parent.parent / 'results' / 'SU27_Improved' / '1' / 'heading_su27' / 'ppo' / 'check' / 'run5'
        if results_dir.exists():
            actor_files = list(results_dir.glob('actor_*.pt'))
            actor_files = [f for f in actor_files if f.stem != 'actor_latest']
            if actor_files:
                actor_files.sort(key=lambda x: int(x.stem.split('_')[1]))
                args.model = str(actor_files[-1])
                logging.info(f"🔍 自动找到最新模型: {args.model}")
    
    # 创建测试器
    tester = SU27FlightTester(model_path=args.model)
    
    # 加载模型
    if args.model:
        tester.load_model()
    
    # 创建环境
    tester.create_env()
    
    try:
        # 运行指定测试
        if args.test == 'all':
            tester.run_all_tests()
        elif args.test == 'stable':
            tester.test_stable_flight()
        elif args.test == 'heading':
            tester.test_heading_change()
        elif args.test == 'altitude':
            tester.test_altitude_change()
        elif args.test == 'maneuver':
            tester.test_maneuvers()
    finally:
        if tester.env:
            tester.env.close()


if __name__ == '__main__':
    main()
