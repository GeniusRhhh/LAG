#!/usr/bin/env python3
"""
SU27 Baseline模型验证测试
测试基本机动能力：平稳飞行、转弯、爬升、下降

使用方法:
  python run_su27_baseline.py [模型路径]
  
示例:
  python run_su27_baseline.py ../train/runs/su27_xxx/actor_latest.pt
  python run_su27_baseline.py  # 默认使用 envs/JSBSim/model/baseline_model.pt
"""

import os
import sys
import logging
import numpy as np
import torch
import argparse

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from envs.JSBSim.envs import SingleControlEnv
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.JSBSim.core.catalog import Catalog as c

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SU27BaselineTest:
    """SU27 Baseline模型测试类"""
    
    def __init__(self, model_path=None):
        self.baseline_policy = None
        self.rnn_state = None
        self.env = None
        
        # 设置模型路径
        if model_path is None:
            # 默认使用F16的baseline模型
            self.model_path = get_root_dir() + '/model/baseline_model.pt'
        else:
            self.model_path = model_path
        
        # 高层指令数组（与F16一致）
        self.norm_delta_altitude = np.array([-1000, -500, -200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200, 500, 1000]) / 1000.0
        self.norm_delta_heading = np.array([-180, -90, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 90, 180]) / 180.0
        self.norm_delta_velocity = np.array([-50, -20, -10, 0, 10, 20, 50]) / 50.0
        
    def load_model(self):
        """加载baseline模型"""
        try:
            self.baseline_policy = BaselineActor()
            
            if not os.path.exists(self.model_path):
                logging.error(f"❌ 模型文件不存在: {self.model_path}")
                logging.info("\n提示：")
                logging.info("1. 如果你已经训练好SU27模型，请指定正确的模型路径")
                logging.info("2. 或者先训练模型: python scripts/train/train_su27_improved.py")
                logging.info("3. 也可以使用F16的baseline模型测试: 不指定参数即可\n")
                return False
                
            self.baseline_policy.load_state_dict(
                torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
            )
            self.baseline_policy.eval()
            
            # 初始化RNN状态
            self.rnn_state = np.zeros((1, 1, 128))
            
            logging.info(f"✅ 成功加载模型: {self.model_path}")
            return True
        except Exception as e:
            logging.error(f"❌ 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_env(self):
        """创建单机测试环境"""
        try:
            # 使用SU27的heading任务
            self.env = SingleControlEnv("1/heading_su27")
            self.env.reset()
            logging.info("✅ 测试环境创建成功")
            return True
        except Exception as e:
            logging.error(f"❌ 创建环境失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_aircraft_state(self):
        """获取飞机当前状态"""
        agent_id = list(self.env.agents.keys())[0]
        aircraft = self.env.agents[agent_id]
        
        altitude = aircraft.get_property_value(c.position_h_sl_m)
        heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
        velocity = aircraft.get_property_value(c.velocities_u_mps)
        roll = np.rad2deg(aircraft.get_property_value(c.attitude_phi_rad))
        pitch = np.rad2deg(aircraft.get_property_value(c.attitude_theta_rad))
        
        return {
            'altitude': altitude,
            'heading': heading,
            'velocity': velocity,
            'roll': roll,
            'pitch': pitch
        }
    
    def get_aircraft_obs(self):
        """获取飞机观测向量"""
        agent_id = list(self.env.agents.keys())[0]
        aircraft = self.env.agents[agent_id]
        
        position = aircraft.get_position()
        velocity = aircraft.get_velocity()
        
        roll = aircraft.get_property_value(c.attitude_phi_rad)
        pitch = aircraft.get_property_value(c.attitude_theta_rad)
        yaw = aircraft.get_property_value(c.attitude_psi_rad)
        
        obs = np.array([
            position[2] / 10000.0,
            np.linalg.norm(velocity) / 300.0,
            roll,
            pitch,
            yaw,
            velocity[0] / 300.0,
            velocity[1] / 300.0,
            velocity[2] / 300.0,
            0.0
        ])
        
        return obs
    
    def execute_command(self, altitude_diff, heading_diff, velocity_diff):
        """执行高层指令"""
        # 转换为指令索引
        altitude_cmd_id = np.argmin(np.abs(self.norm_delta_altitude * 1000.0 - altitude_diff))
        heading_cmd_id = np.argmin(np.abs(self.norm_delta_heading * 180.0 - heading_diff))
        velocity_cmd_id = np.argmin(np.abs(self.norm_delta_velocity * 50.0 - velocity_diff))
        
        # 获取观测
        raw_obs = self.get_aircraft_obs()
        input_obs = np.zeros(12)
        
        # 构建输入
        input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
        input_obs[1] = self.norm_delta_heading[heading_cmd_id]
        input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.nan_to_num(input_obs, nan=0.0)
        input_obs = np.expand_dims(input_obs, axis=0)
        
        # 调用baseline模型
        _action, _rnn_states = self.baseline_policy(
            torch.FloatTensor(input_obs),
            torch.FloatTensor(self.rnn_state)
        )
        
        action_output = _action.detach().cpu().numpy().squeeze(0)
        self.rnn_state = _rnn_states.detach().cpu().numpy()
        
        # 动作归一化
        norm_act = np.zeros(4)
        norm_act[0] = action_output[0] / 20 - 1.  # aileron
        norm_act[1] = action_output[1] / 20 - 1.  # elevator
        norm_act[2] = action_output[2] / 20 - 1.  # rudder
        norm_act[3] = action_output[3] / 58 + 0.4  # throttle
        
        return norm_act
    
    def apply_action(self, action):
        """应用控制动作"""
        agent_id = list(self.env.agents.keys())[0]
        aircraft = self.env.agents[agent_id]
        
        aircraft.set_property_value(c.fcs_aileron_cmd_norm, action[0])
        aircraft.set_property_value(c.fcs_elevator_cmd_norm, action[1])
        aircraft.set_property_value(c.fcs_rudder_cmd_norm, action[2])
        aircraft.set_property_value(c.fcs_throttle_cmd_norm, action[3])
    
    def test_maneuver(self, name, altitude_diff, heading_diff, velocity_diff, duration_steps=200):
        """测试单个机动动作"""
        logging.info(f"\n{'='*60}")
        logging.info(f"测试机动: {name}")
        logging.info(f"指令 - 高度差: {altitude_diff}m, 航向差: {heading_diff}°, 速度差: {velocity_diff}m/s")
        logging.info(f"{'='*60}")
        
        # 记录初始状态
        initial_state = self.get_aircraft_state()
        logging.info(f"初始状态:")
        logging.info(f"  高度: {initial_state['altitude']:.1f}m")
        logging.info(f"  航向: {initial_state['heading']:.1f}°")
        logging.info(f"  速度: {initial_state['velocity']:.1f}m/s")
        logging.info(f"  滚转: {initial_state['roll']:.1f}°")
        logging.info(f"  俯仰: {initial_state['pitch']:.1f}°")
        
        # 执行机动
        for step in range(duration_steps):
            # 生成控制指令
            action = self.execute_command(altitude_diff, heading_diff, velocity_diff)
            
            # 应用控制
            self.apply_action(action)
            
            # 环境步进
            self.env.step({})
            
            # 每50步输出一次状态
            if (step + 1) % 50 == 0:
                current_state = self.get_aircraft_state()
                logging.info(f"步数 {step+1}/{duration_steps}:")
                logging.info(f"  高度: {current_state['altitude']:.1f}m, 航向: {current_state['heading']:.1f}°, 速度: {current_state['velocity']:.1f}m/s")
        
        # 记录最终状态
        final_state = self.get_aircraft_state()
        logging.info(f"\n最终状态:")
        logging.info(f"  高度: {final_state['altitude']:.1f}m (变化: {final_state['altitude']-initial_state['altitude']:.1f}m)")
        logging.info(f"  航向: {final_state['heading']:.1f}° (变化: {final_state['heading']-initial_state['heading']:.1f}°)")
        logging.info(f"  速度: {final_state['velocity']:.1f}m/s (变化: {final_state['velocity']-initial_state['velocity']:.1f}m/s)")
        logging.info(f"  滚转: {final_state['roll']:.1f}°")
        logging.info(f"  俯仰: {final_state['pitch']:.1f}°")
        
        return initial_state, final_state
    
    def run_all_tests(self):
        """运行所有测试"""
        logging.info("\n" + "="*60)
        logging.info("开始Baseline模型验证测试")
        logging.info(f"模型: {self.model_path}")
        logging.info("="*60)
        
        # 加载模型
        if not self.load_model():
            return False
        
        # 创建环境
        if not self.create_env():
            return False
        
        # 定义测试用例
        test_cases = [
            ("平稳飞行", 0, 0, 0, 200),
            ("右转弯 45°", 0, 45, 0, 200),
            ("左转弯 -45°", 0, -45, 0, 200),
            ("爬升 +500m", 500, 0, 0, 300),
            ("下降 -200m", -200, 0, 0, 300),
            ("加速 +20m/s", 0, 0, 20, 200),
            ("减速 -20m/s", 0, 0, -20, 200),
            ("组合机动（右转45° + 爬升500m）", 500, 45, 0, 300),
        ]
        
        # 执行所有测试
        for name, alt_diff, hdg_diff, vel_diff, duration in test_cases:
            try:
                self.test_maneuver(
                    name=name,
                    altitude_diff=alt_diff,
                    heading_diff=hdg_diff,
                    velocity_diff=vel_diff,
                    duration_steps=duration
                )
                
                # 重置环境
                self.env.reset()
                self.rnn_state = np.zeros((1, 1, 128))
                
            except Exception as e:
                logging.error(f"测试 {name} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        logging.info("\n" + "="*60)
        logging.info("✅ 所有测试完成")
        logging.info("="*60)
        
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SU27 Baseline模型验证测试')
    parser.add_argument('model_path', nargs='?', default=None,
                        help='模型文件路径 (默认: envs/JSBSim/model/baseline_model.pt)')
    args = parser.parse_args()
    
    tester = SU27BaselineTest(model_path=args.model_path)
    tester.run_all_tests()
