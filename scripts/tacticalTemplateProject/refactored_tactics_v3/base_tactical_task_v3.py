#!/usr/bin/env python3
"""
战术任务基类 V3 - 最小化设计

设计原则：
- 只提供真正共享的基础设施
- 不干涉战术逻辑
- 让子类保留完整的战术实现

作者：重构优化
日期：2025-10-19
版本：3.0 (最终版)
"""

import logging
import numpy as np
import torch

from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir


class BaseTacticalTask(MultipleCombatTask):
    """
    战术任务基类 - 最小化设计
    
    只提供基础设施，不干涉战术逻辑：
    1. 雷达和RWR系统集成
    2. baseline模型加载
    3. 动作空间定义
    4. 基础工具方法
    
    子类需要自己实现：
    - normalize_action() - 完整的战术逻辑
    - step() - 如果需要自定义
    - 所有战术特有的方法
    """
    
    def __init__(self, config):
        """初始化基础设施"""
        super().__init__(config)
        
        # 1. 初始化雷达和RWR系统
        self._init_radar_and_rwr()
        
        # 2. 加载baseline模型
        self._load_baseline_model()
        
        # 3. 初始化动作空间
        self._init_action_space()
        
        # 4. 初始化状态跟踪
        self.initial_heading = {}
        self.initial_altitude = {}
        self._inner_rnn_states = {}
        
        logging.info(f"✅ {self.__class__.__name__} 基础设施初始化完成")
    
    # ==================== 基础设施方法 ====================
    
    def _init_radar_and_rwr(self):
        """初始化雷达和RWR系统"""
        try:
            from radar_manager import (
                get_unified_radar_manager, 
                get_rwr_threat_level, 
                get_rwr_threat_sources
            )
            self.radar_manager = get_unified_radar_manager()
            self.get_rwr_threat_level = get_rwr_threat_level
            self.get_rwr_threat_sources = get_rwr_threat_sources
            logging.info(f"📡 统一雷达管理系统已集成到{self.__class__.__name__}")
            logging.info("🚨 RWR威胁检测系统已集成")
        except ImportError as e:
            logging.error(f"❌ 雷达/RWR系统导入失败: {e}")
            self.radar_manager = None
    
    def _load_baseline_model(self):
        """加载baseline模型"""
        try:
            self.my_lowlevel_policy = BaselineActor()
            
            model_path = get_root_dir() + '/model/baseline_model.pt'
            logging.info(f"📦 加载baseline模型: {model_path}")
            
            if torch.cuda.is_available():
                device = torch.device("cuda")
                checkpoint = torch.load(model_path, weights_only=True)
            else:
                device = torch.device("cpu")
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            
            self.my_lowlevel_policy.load_state_dict(checkpoint)
            self.my_lowlevel_policy.eval()
            logging.info("✅ baseline模型加载成功")
        except Exception as e:
            logging.error(f"❌ baseline模型加载失败: {e}")
            self.my_lowlevel_policy = None
    
    def _init_action_space(self):
        """初始化动作空间定义"""
        # 高度指令数组 (15个选项)
        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0
        
        # 航向指令数组 (17个选项)
        self.norm_delta_heading = np.array([
            -np.pi, -2*np.pi/3, -np.pi/2, -5*np.pi/12, -np.pi/3, -np.pi/4, -np.pi/6, -np.pi/12,
            0,
            np.pi/12, np.pi/6, np.pi/4, np.pi/3, 5*np.pi/12, np.pi/2, 2*np.pi/3, np.pi
        ])
        
        # 速度指令数组 (7个选项)
        self.norm_delta_velocity = np.array([-100, -50, -20, 0, 20, 50, 100])
        
        logging.info("🎮 动作空间已初始化: [15, 17, 7]")
    
    # ==================== 雷达和RWR辅助方法 ====================
    
    def _update_radar_and_rwr(self, env, agent_id, current_time):
        """更新雷达和RWR状态（子类可调用）"""
        if agent_id == "A0100" and self.radar_manager is not None:
            self.radar_manager.update_friendly_radar_states(env, current_time)
            self.radar_manager.update_enemy_radar_states(env, current_time)
            
            if env.current_step % 25 == 0:
                self._check_rwr_threats(env)
    
    def _check_rwr_threats(self, env):
        """检查RWR威胁状态"""
        try:
            for agent_id in ["A0100", "A0200", "B0100", "B0200"]:
                if agent_id not in env.agents or not env.agents[agent_id].is_alive:
                    continue
                
                threat_level = self.get_rwr_threat_level(agent_id)
                
                if threat_level > 0:
                    threat_sources = self.get_rwr_threat_sources(agent_id)
                    threat_str = ""
                    for threat in threat_sources:
                        threat_str += f"{threat['source']}(等级{threat['level']},方位{threat['bearing']:.0f}°) "
                    
                    if threat_level >= 3:
                        logging.warning(f"🚨 {agent_id} RWR告警: 等级{threat_level} - {threat_str}")
                    elif threat_level == 2:
                        logging.info(f"⚠️  {agent_id} RWR告警: 等级{threat_level} - {threat_str}")
                    else:
                        logging.debug(f"📡 {agent_id} RWR告警: 等级{threat_level} - {threat_str}")
        except Exception as e:
            logging.error(f"❌ RWR威胁检测错误: {e}")
    
    def get_radar_states(self):
        """获取雷达状态"""
        try:
            if self.radar_manager is None:
                return {"A0100": "SEARCH", "A0200": "SEARCH", "B0100": "SEARCH", "B0200": "SEARCH"}
            
            radar_summary = self.radar_manager.get_radar_performance_summary()
            states = {}
            for agent_id, radar_info in radar_summary["friendly_radars"].items():
                states[agent_id] = radar_info["status"]
            for agent_id, radar_info in radar_summary["enemy_radars"].items():
                states[agent_id] = radar_info["status"]
            return states
        except Exception as e:
            logging.error(f"❌ 获取雷达状态错误: {e}")
            return {"A0100": "SEARCH", "A0200": "SEARCH", "B0100": "SEARCH", "B0200": "SEARCH"}
    
    # ==================== 底层策略方法（子类可直接使用） ====================
    
    def _use_lowlevel_policy(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """使用底层策略网络"""
        if self.my_lowlevel_policy is None:
            return np.array([0.0, 0.0, 0.0, 0.7])

        try:
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)

            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)

            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.nan_to_num(input_obs, nan=0.0)
            input_obs = np.expand_dims(input_obs, axis=0)

            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))

            _action, _rnn_states = self.my_lowlevel_policy(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()

            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4

            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 1000:
                norm_act[1] = max(norm_act[1], 0.0)
                norm_act[3] = max(norm_act[3], 0.8)

            return norm_act
        except Exception as e:
            logging.error(f"底层策略执行失败 {agent_id}: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])
    
    # ==================== 通用工具方法 ====================
    
    def _calculate_distance(self, aircraft1, aircraft2):
        """计算两架飞机之间的距离（米）"""
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        return np.linalg.norm(pos1 - pos2)
    
    def _calculate_bearing_to_target(self, my_pos, target_pos):
        """计算到目标的方位角（度）"""
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        bearing = np.rad2deg(np.arctan2(dy, dx))
        return (bearing + 360) % 360


if __name__ == "__main__":
    print("BaseTacticalTask V3 - 最小化基类")
    print("只提供基础设施，不干涉战术逻辑")
