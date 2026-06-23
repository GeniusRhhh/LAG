#!/usr/bin/env python3
"""
回转射击战术任务 - 基于drag_shoot_tactical_task架构
双机在DOR前做short skate，到达安全回转距离后回转重新交战，类钳形包夹
"""

import logging
import numpy as np
import torch
from enum import Enum
from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.termination_conditions.termination_condition_base import BaseTerminationCondition
from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers, CompositeManeuverExecutor


class TurnAroundShootingTermination(BaseTerminationCondition):
    """回转射击专用终止条件"""

    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 1000)
        self.max_steps = getattr(config, 'max_steps', 2500)

    def get_termination(self, task, env, agent_id, info={}):
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= self.altitude_limit:
            logging.warning(f"⚠️ TERMINATION: {agent_id} altitude too low: {current_alt:.1f}m")
            return True, False, info

        if env.current_step >= self.max_steps:
            logging.info(f"⏰ TERMINATION: Time limit reached (step {env.current_step})")
            return True, False, info

        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            logging.warning(f"⚠️ TERMINATION: {agent_id} extreme state detected")
            return True, False, info

        if (abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0):
            logging.warning(f"⚠️ TERMINATION: {agent_id} overload detected")
            return True, False, info

        red_alive = [aid for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive]
        blue_alive = [aid for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive]

        if len(red_alive) == 0:
            logging.info(f"🏆 TERMINATION: Red eliminated, Blue wins!")
            return True, True, {"termination_reason": "red_eliminated", "winner": "blue"}
        elif len(blue_alive) == 0:
            logging.info(f"🏆 TERMINATION: Blue eliminated, Red wins!")
            return True, True, {"termination_reason": "blue_eliminated", "winner": "red"}

        return False, False, info


class TacticalPhase(Enum):
    """回转射击战术阶段"""
    NLT_MELD = "NLT_MELD"
    MELD_MTR = "MELD_MTR"
    MTR_TR = "MTR_TR"
    TR_DOR = "TR_DOR"
    DOR_DR = "DOR_DR"


class TurnAroundShootingTacticalTask(MultipleCombatTask):
    """回转射击战术任务"""

    def __init__(self, config):
        super().__init__(config)

        self.termination_conditions = [TurnAroundShootingTermination(self.config)]

        self.tactical_distances = {
            'NLT_MELD_min': 81000,
            'MELD_MTR_min': 45000,
            'MTR_TR_min': 41000,
            'TR_DOR_min': 19600,
            'DOR_DR_min': 14500,
        }

        self.turn_around_config = {
            'first_short_skate_distance': 35000,
            'turn_back_distance': 25000,
            'second_short_skate_distance': 19600,
        }

        self.current_phase = TacticalPhase.NLT_MELD
        self.phase_start_time = 0.0
        self.phase_start_step = 0

        self.turn_around_states = {
            'A0100': {'first_skate_done': False, 'turned_back': False, 'second_skate_done': False},
            'A0200': {'first_skate_done': False, 'turned_back': False, 'second_skate_done': False}
        }

        self.last_missile_launch_time = {"A0100": -999, "A0200": -999, "B0100": -999, "B0200": -999}
        self.friendly_missile_cooldown = 2.0
        self.enemy_missile_cooldown = 10.0
        self.friendly_burst_launch = {"A0100": 0, "A0200": 0}
        self.missile_launched = {"A0100": False, "A0200": False, "B0100": False, "B0200": False}

        from radar_manager import get_unified_radar_manager, get_rwr_threat_level, get_rwr_threat_sources
        self.radar_manager = get_unified_radar_manager()
        self.get_rwr_threat_level = get_rwr_threat_level
        self.get_rwr_threat_sources = get_rwr_threat_sources
        logging.info(" 统一雷达管理系统已集成到回转射击任务")
        logging.info(" RWR威胁检测系统已集成")

        self.unified_enemy_ai = None
        self.initial_heading = {}
        self.initial_altitude = {}
        self.short_skate_states = {}
        self.short_skate_start_time = {}

        self.my_lowlevel_policy = BaselineActor()
        self._inner_rnn_states = {}

        self.basic_maneuvers = BasicManeuvers()
        self.composite_executor = CompositeManeuverExecutor()
        self.active_maneuvers = {}
        self.maneuver_start_times = {}
        self.enable_precise_maneuvers = True

        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0

        self.norm_delta_heading = np.array([
            -np.pi, -2*np.pi/3, -np.pi/2, -5*np.pi/12, -np.pi/3, -np.pi/4, -np.pi/6, -np.pi/12,
            0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, 5*np.pi/12, np.pi/2, 2*np.pi/3, np.pi
        ])

        self.norm_delta_velocity = np.array([-150, -100, -50, 0, 50, 100, 150]) / 100.0

        self._load_baseline_models()
        logging.info("TurnAroundShootingTacticalTask initialized")

    def reset(self, env):
        super().reset(env)
        self.current_phase = TacticalPhase.NLT_MELD
        self.phase_start_time = 0.0
        self.phase_start_step = 0
        
        self.turn_around_states = {
            'A0100': {'first_skate_done': False, 'turned_back': False, 'second_skate_done': False},
            'A0200': {'first_skate_done': False, 'turned_back': False, 'second_skate_done': False}
        }
        
        # ✅ 全局同步状态 - 确保长机僚机同时触发
        self.global_skate_triggers = {
            'first_triggered': False,
            'first_trigger_time': 0,
            'second_triggered': False, 
            'second_trigger_time': 0,
            'third_triggered': False,
            'third_trigger_time': 0
        }
        
        self.last_missile_launch_time = {"A0100": -999, "A0200": -999, "B0100": -999, "B0200": -999}
        self.friendly_burst_launch = {"A0100": 0, "A0200": 0}
        self.missile_launched = {"A0100": False, "A0200": False, "B0100": False, "B0200": False}
        
        self.initial_heading.clear()
        self.initial_altitude.clear()
        self.short_skate_states.clear()
        self.short_skate_start_time.clear()
        self.active_maneuvers.clear()
        self.maneuver_start_times.clear()
        
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        
        if hasattr(self, 'unified_enemy_ai') and self.unified_enemy_ai is not None:
            self.unified_enemy_ai.reset_for_new_episode()
        
        logging.info("TurnAroundShootingTacticalTask reset completed")
        return super().reset(env)

    def get_termination(self, env, agent_id, info={}):
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= 1000:
            logging.info(f"{agent_id} altitude too low: {current_alt:.1f}m")
            return True, {"termination_reason": "low_altitude"}

        if env.current_step >= 2500:
            logging.info(f"Time limit reached")
            return True, {"termination_reason": "timeout"}

        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            logging.info(f"{agent_id} extreme state")
            return True, {"termination_reason": "extreme_state"}

        if (abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0):
            logging.info(f"{agent_id} overload")
            return True, {"termination_reason": "overload"}

        red_alive = [aid for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive]
        blue_alive = [aid for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive]

        if env.current_step % 50 == 0:
            current_time = env.current_step * env.time_interval
            logging.info(f"📊 CHECK - 时间{current_time:.1f}s Step{env.current_step}: Red存活={len(red_alive)}{red_alive}, Blue存活={len(blue_alive)}{blue_alive}")

        if len(red_alive) == 0:
            logging.warning(f"🔴 GAME OVER: Red全灭，Blue获胜！Step={env.current_step}")
            logging.warning(f"   Red状态: A0100={env.agents.get('A0100', None) and env.agents['A0100'].is_alive}, A0200={env.agents.get('A0200', None) and env.agents['A0200'].is_alive}")
            return True, {"termination_reason": "red_eliminated", "winner": "blue"}
        elif len(blue_alive) == 0:
            logging.warning(f"🔵 GAME OVER: Blue全灭，Red获胜！Step={env.current_step}")
            logging.warning(f"   Blue状态: B0100={env.agents.get('B0100', None) and env.agents['B0100'].is_alive}, B0200={env.agents.get('B0200', None) and env.agents['B0200'].is_alive}")
            return True, {"termination_reason": "blue_eliminated", "winner": "red"}

        return False, info

    def _load_baseline_models(self):
        try:
            model_path = get_root_dir() + '/model/baseline_model.pt'
            logging.info(f"Loading baseline model from: {model_path}")
            if torch.cuda.is_available():
                device = torch.device("cuda")
                checkpoint = torch.load(model_path, weights_only=True)
            else:
                device = torch.device("cpu")
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            self.my_lowlevel_policy.load_state_dict(checkpoint)
            self.my_lowlevel_policy.eval()
            logging.info(" Successfully loaded baseline model")
        except Exception as e:
            logging.error(f" 加载baseline模型失败: {e}")
            self.my_lowlevel_policy = None
    
    def normalize_action(self, env, agent_id, action):
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

        current_time = env.current_step * env.time_interval
        
        # 更新雷达状态（每个时间步都更新）
        if agent_id == "A0100":  # 只在第一个飞机时更新一次，避免重复
            self.radar_manager.update_friendly_radar_states(env, current_time)
            self.radar_manager.update_enemy_radar_states(env, current_time)
            
            # RWR威胁检测（每5秒打印一次）
            if env.current_step % 25 == 0:
                self._check_rwr_threats(env)

        if agent_id not in self.initial_heading:
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
            self.initial_heading[agent_id] = np.rad2deg(current_heading)
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            self.initial_altitude[agent_id] = current_altitude
            logging.info(f"{agent_id} 初始: 航向{self.initial_heading[agent_id]:.1f}°, 高度{self.initial_altitude[agent_id]:.1f}m")

        result = self._process_turn_around_tactics(env, agent_id, current_time)
        self._handle_missile_launch(env, agent_id, current_time)

        return result

    def step(self, env):
        """执行回转射击战术步骤 - 完全复制drag_shoot实现"""
        # 更新战术阶段
        self._update_tactical_phase(env)

        # 处理导弹发射
        current_time = env.current_step * env.time_interval
        for agent_id in env._jsbsims.keys():
            if env._jsbsims[agent_id].is_alive:
                self._handle_missile_launch(env, agent_id, current_time)

        # 更新统一雷达系统状态
        self.radar_manager.update_friendly_radar_states(env, current_time)
        self.radar_manager.update_enemy_radar_states(env, current_time)

        # 为每个智能体生成战术动作
        obs = {}
        share_obs = {}
        rewards = {}
        dones = {}
        infos = {}

        for agent_id in env._jsbsims.keys():
            if not env._jsbsims[agent_id].is_alive:
                obs[agent_id] = np.zeros(self.obs_length)
                share_obs[agent_id] = np.zeros(self.obs_length)
                rewards[agent_id] = [-10.0]
                dones[agent_id] = [True]
                infos[agent_id] = {"agent_id": agent_id, "alive": False}
                continue

            # 获取观测
            agent_obs = self.get_obs(env, agent_id)
            obs[agent_id] = agent_obs
            share_obs[agent_id] = agent_obs

            # 计算奖励
            reward = self._calculate_reward(env, agent_id)
            rewards[agent_id] = [reward]

            # 检查终止条件
            done, info = self.get_termination(env, agent_id, {})
            dones[agent_id] = [done]
            infos[agent_id] = {"agent_id": agent_id, "alive": True, "phase": self.current_phase.value}

        return obs, share_obs, rewards, dones, infos

    def _process_turn_around_tactics(self, env, agent_id, current_time):
        try:
            self._update_tactical_phase(env)
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_tactical_command_indices(env, agent_id)
            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
        except Exception as e:
            logging.error(f"{agent_id} 战术执行错误: {e}")
            return np.array([0.0, 0.1, 0.0, 0.8])

    def _update_tactical_phase(self, env):
        leader_red = env._jsbsims.get("A0100")
        leader_blue = env._jsbsims.get("B0100")

        if not leader_red or not leader_blue or not leader_red.is_alive or not leader_blue.is_alive:
            return

        distance = self._calculate_distance(leader_red, leader_blue)
        old_phase = self.current_phase
        
        # 每50步输出一次距离和阶段判断
        if env.current_step % 50 == 0:
            logging.info(f"📏 距离={distance/1000:.1f}km, 当前阶段={self.current_phase.value}")

        # ✅ 阶段只能前进，不能后退（防止循环）
        phase_order = [
            TacticalPhase.NLT_MELD,
            TacticalPhase.MELD_MTR,
            TacticalPhase.MTR_TR,
            TacticalPhase.TR_DOR,
            TacticalPhase.DOR_DR
        ]
        current_phase_index = phase_order.index(self.current_phase)
        
        # 根据距离判断新阶段
        new_phase = self.current_phase
        
        if distance >= self.tactical_distances['NLT_MELD_min']:
            new_phase = TacticalPhase.NLT_MELD
        elif distance >= self.tactical_distances['MELD_MTR_min']:
            new_phase = TacticalPhase.MELD_MTR
        elif distance >= self.tactical_distances['MTR_TR_min']:
            new_phase = TacticalPhase.MTR_TR
        elif distance >= self.tactical_distances['TR_DOR_min']:
            new_phase = TacticalPhase.TR_DOR
        else:
            new_phase = TacticalPhase.DOR_DR
        
        # 只允许前进到更后的阶段
        new_phase_index = phase_order.index(new_phase)
        if new_phase_index > current_phase_index:
            self.current_phase = new_phase
            current_time = env.current_step * env.time_interval
            self.phase_start_time = current_time
            self.phase_start_step = env.current_step
            logging.info(f"🔄 阶段切换: {old_phase.value} → {self.current_phase.value} (距离{distance/1000:.1f}km)")

    def _get_tactical_command_indices(self, env, agent_id):
        if agent_id.startswith('A'):
            if agent_id == "A0100":
                return self._get_leader_command_indices(env, agent_id)
            else:
                return self._get_wingman_command_indices(env, agent_id)
        else:
            return self._get_enemy_command_indices(env, agent_id)

    def _get_leader_command_indices(self, env, agent_id: str):
        """长机指令 - 使用全局同步触发机制"""
        # 更新全局触发状态（只在长机时计算，避免重复）
        if agent_id == "A0100":
            self._update_global_triggers(env)
        
        # 获取距离用于日志显示
        leader_blue = None
        for blue_id in ["B0100", "B0200"]:
            if blue_id in env._jsbsims and env._jsbsims[blue_id].is_alive:
                leader_blue = env._jsbsims[blue_id]
                break
        
        if leader_blue:
            distance = self._calculate_distance(env.agents[agent_id], leader_blue)
        else:
            distance = 100000  # 敌机全灭，设置大距离
        
        # 调试日志：每50步打印一次距离和阶段
        if env.current_step % 50 == 0:
            logging.info(f"🔍 {agent_id} 阶段={self.current_phase.value}, 距离={distance/1000:.1f}km, 状态={self.turn_around_states[agent_id]}")

        # 阶段1-3: 保持0度平稳飞行
        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR]:
            return self._maintain_heading_precise(env, agent_id, 0.0)

        # 阶段4: TR-DOR - 使用全局同步触发
        elif self.current_phase == TacticalPhase.TR_DOR:
            state = self.turn_around_states[agent_id]
            
            # 第1次short skate: 基于全局触发
            if not state['first_skate_done']:
                if self.global_skate_triggers['first_triggered']:
                    if agent_id not in self.short_skate_states:
                        logging.info(f"🔄 {agent_id}(长机) 执行第1次short skate（朝外侧左转）")
                    return self._execute_short_skate_precise(env, agent_id, "left")
                else:
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            
            # 第2次short skate: 基于全局触发（朝内侧回转交战）
            elif not state['turned_back']:
                if self.global_skate_triggers['second_triggered']:
                    if agent_id not in self.short_skate_states:
                        logging.info(f"🔄 {agent_id}(长机) 执行第2次short skate（朝内侧左转回转交战）")
                    return self._execute_short_skate_precise(env, agent_id, "left")
                else:
                    # 保持180度等待全局触发
                    return self._maintain_heading_precise(env, agent_id, 180.0)
            
            else:
                # 回转完成，检查是否需要第3次short skate
                if not state['second_skate_done']:
                    # 第3次short skate: 基于全局触发
                    if self.global_skate_triggers['third_triggered']:
                        if agent_id not in self.short_skate_states:
                            logging.info(f"🔄 {agent_id}(长机) 执行第3次short skate（朝外侧左转返航）")
                        return self._execute_short_skate_precise(env, agent_id, "left")
                    else:
                        # 保持0度交战，等待第3次触发
                        return self._maintain_heading_precise(env, agent_id, 0.0)
                else:
                    # 第3次完成，保持180度返航
                    return self._maintain_heading_precise(env, agent_id, 180.0)

        # 阶段5: DOR-DR - 继续返航
        elif self.current_phase == TacticalPhase.DOR_DR:
            # 保持180度返航
            return self._maintain_heading_precise(env, agent_id, 180.0)
        
        return self._maintain_heading_precise(env, agent_id, 0.0)

    def _update_global_triggers(self, env):
        """更新全局触发状态 - 确保长机僚机同步"""
        current_time = env.current_step * env.time_interval
        
        # 获取平均距离用于触发判定
        total_distance = 0
        count = 0
        for red_id in ["A0100", "A0200"]:
            if red_id in env.agents and env.agents[red_id].is_alive:
                for blue_id in ["B0100", "B0200"]:
                    if blue_id in env._jsbsims and env._jsbsims[blue_id].is_alive:
                        distance = self._calculate_distance(env.agents[red_id], env._jsbsims[blue_id])
                        total_distance += distance
                        count += 1
        
        avg_distance = total_distance / count if count > 0 else 100000
        
        # 第1次触发：基于距离
        if not self.global_skate_triggers['first_triggered']:
            if avg_distance < self.turn_around_config['first_short_skate_distance']:
                self.global_skate_triggers['first_triggered'] = True
                self.global_skate_triggers['first_trigger_time'] = current_time
                logging.info(f"🌐 全局第1次short skate触发，平均距离{avg_distance/1000:.1f}km，时间{current_time:.1f}s")
        
        # 第2次触发：基于时间
        if self.global_skate_triggers['first_triggered'] and not self.global_skate_triggers['second_triggered']:
            time_since_first = current_time - self.global_skate_triggers['first_trigger_time']
            if time_since_first > 15.0:
                self.global_skate_triggers['second_triggered'] = True
                self.global_skate_triggers['second_trigger_time'] = current_time
                logging.info(f"🌐 全局第2次short skate触发，时间{time_since_first:.1f}s后")
        
        # 第3次触发：基于时间（第2次后10秒）
        if self.global_skate_triggers['second_triggered'] and not self.global_skate_triggers['third_triggered']:
            time_since_second = current_time - self.global_skate_triggers['second_trigger_time']
            if time_since_second > 10.0:
                self.global_skate_triggers['third_triggered'] = True
                self.global_skate_triggers['third_trigger_time'] = current_time
                logging.info(f"🌐 全局第3次short skate触发，第2次后{time_since_second:.1f}s")

    def _get_wingman_command_indices(self, env, agent_id: str):
        """僚机指令 - 完全重写战术回转逻辑"""
        # 寻找存活的敌机
        leader_blue = None
        for blue_id in ["B0100", "B0200"]:
            if blue_id in env._jsbsims and env._jsbsims[blue_id].is_alive:
                leader_blue = env._jsbsims[blue_id]
                break
        
        if leader_blue:
            distance = self._calculate_distance(env.agents[agent_id], leader_blue)
        else:
            distance = 100000  # 敌机全灭，设置大距离
        
        # 调试日志：每50步打印一次距离和阶段
        if env.current_step % 50 == 0:
            logging.info(f"🔍 {agent_id} 阶段={self.current_phase.value}, 距离={distance/1000:.1f}km, 状态={self.turn_around_states[agent_id]}")

        # 阶段1-3: 保持0度平稳飞行
        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR]:
            return self._maintain_heading_precise(env, agent_id, 0.0)

        # 阶段4: TR-DOR - 使用全局同步触发
        elif self.current_phase == TacticalPhase.TR_DOR:
            state = self.turn_around_states[agent_id]
            
            # 第1次short skate: 基于全局触发
            if not state['first_skate_done']:
                if self.global_skate_triggers['first_triggered']:
                    if agent_id not in self.short_skate_states:
                        logging.info(f"🔄 {agent_id}(僚机) 执行第1次short skate（朝外侧右转）")
                    return self._execute_short_skate_precise(env, agent_id, "right")
                else:
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            
            # 第2次short skate: 基于全局触发（朝内侧回转交战）
            elif not state['turned_back']:
                if self.global_skate_triggers['second_triggered']:
                    if agent_id not in self.short_skate_states:
                        logging.info(f"🔄 {agent_id}(僚机) 执行第2次short skate（朝内侧右转回转交战）")
                    return self._execute_short_skate_precise(env, agent_id, "right")
                else:
                    # 保持180度等待全局触发
                    return self._maintain_heading_precise(env, agent_id, 180.0)
            
            else:
                # 回转完成，检查是否需要第3次short skate
                if not state['second_skate_done']:
                    # 第3次short skate: 基于全局触发
                    if self.global_skate_triggers['third_triggered']:
                        if agent_id not in self.short_skate_states:
                            logging.info(f"🔄 {agent_id}(僚机) 执行第3次short skate（朝外侧右转返航）")
                        return self._execute_short_skate_precise(env, agent_id, "right")
                    else:
                        # 保持0度交战，等待第3次触发
                        return self._maintain_heading_precise(env, agent_id, 0.0)
                else:
                    # 第3次完成，保持180度返航
                    return self._maintain_heading_precise(env, agent_id, 180.0)

        # 阶段5: DOR-DR - 继续返航
        elif self.current_phase == TacticalPhase.DOR_DR:
            # 保持180度返航
            return self._maintain_heading_precise(env, agent_id, 180.0)
        
        return self._maintain_heading_precise(env, agent_id, 0.0)

    def _execute_short_skate_precise(self, env, agent_id, direction):
        """执行精确的Short Skate机动 - 完全复制pincer_attack的实现
        
        Args:
            direction: "left" 左转, "right" 右转
        
        流程：
            1. 前10秒：转向中间航向（左转315°，右转45°）
            2. 10秒后：转向最终航向（180°或0°）
        """
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        
        # 高度保护
        if current_altitude < 2000.0:
            altitude_cmd_id = self._convert_altitude_to_index(500.0)
            if env.current_step % 50 == 0:
                logging.warning(f" {agent_id} 高度{current_altitude:.0f}m过低，强制爬升")
            return altitude_cmd_id, 8, 3
        
        # 初始化Short Skate状态
        if agent_id not in self.short_skate_states:
            # 确定中间航向 - 减小角度，不要太弧
            if direction == "right":
                target_heading = 30.0  # 右转30°（减小弧度）
            elif direction == "left":
                target_heading = 330.0  # 左转30°（减小弧度）
            else:
                target_heading = 0.0
            
            self.short_skate_states[agent_id] = {
                'start_time': env.current_step * env.time_interval,
                'phase': 'turn',
                'target_heading': target_heading
            }
        
        # 执行Short Skate机动
        skate_state = self.short_skate_states[agent_id]
        current_time = env.current_step * env.time_interval
        elapsed_time = current_time - skate_state['start_time']
        
        if elapsed_time < 6.0:  # 前6秒执行转弯（缩短时间）
            return self._maintain_heading_precise(env, agent_id, skate_state['target_heading'])
        else:  # 6秒后转向最终航向
            # 确定最终航向
            turn_around_state = self.turn_around_states[agent_id]
            if not turn_around_state['first_skate_done']:
                final_heading = 180.0  # 第1次：脱离到180°
            elif not turn_around_state['turned_back']:
                final_heading = 0.0    # 第2次：回转到0°
            else:
                final_heading = 180.0  # 第3次：返航到180°
            
            # 检查是否完成
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            heading_diff = self._normalize_angle_diff(final_heading - current_heading)
            
            if abs(heading_diff) < 5.0:
                # 机动完成，更新状态
                if not turn_around_state['first_skate_done']:
                    self.turn_around_states[agent_id]['first_skate_done'] = True
                    self.turn_around_states[agent_id]['first_skate_time'] = current_time
                    logging.info(f"✅ {agent_id} 完成第1次short skate，最终航向{current_heading:.1f}°")
                elif not turn_around_state['turned_back']:
                    self.turn_around_states[agent_id]['turned_back'] = True
                    self.turn_around_states[agent_id]['turned_back_time'] = current_time
                    logging.info(f"✅ {agent_id} 完成第2次short skate，最终航向{current_heading:.1f}°")
                else:
                    self.turn_around_states[agent_id]['second_skate_done'] = True
                    self.turn_around_states[agent_id]['second_skate_time'] = current_time
                    logging.info(f"✅ {agent_id} 完成第3次short skate，最终航向{current_heading:.1f}°")
                
                del self.short_skate_states[agent_id]
            
            return self._maintain_heading_precise(env, agent_id, final_heading)

    def _maintain_heading_precise(self, env, agent_id, target_heading):
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        
        heading_diff = self._normalize_angle_diff(target_heading - current_heading)

        altitude_cmd_id = 7
        heading_cmd_id = 8
        velocity_cmd_id = 3

        if current_altitude < 2000.0:
            altitude_cmd_id = self._convert_altitude_to_index(500.0)
        elif current_altitude < 3000.0:
            altitude_cmd_id = self._convert_altitude_to_index(200.0)

        if abs(heading_diff) > 5.0:
            if abs(heading_diff) > 30.0:
                heading_cmd_id = 5 if heading_diff < 0 else 11
            elif abs(heading_diff) > 10.0:
                heading_cmd_id = 6 if heading_diff < 0 else 10
            else:
                heading_cmd_id = 7 if heading_diff < 0 else 9

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _get_enemy_command_indices(self, env, agent_id: str):
        try:
            if hasattr(self, 'unified_enemy_ai') and self.unified_enemy_ai is not None:
                current_time = env.current_step * env.time_interval
                return self.unified_enemy_ai.get_enemy_command_indices(env, agent_id, current_time)
            else:
                return self._maintain_heading_precise(env, agent_id, 180.0)
        except Exception as e:
            logging.error(f"敌方AI错误 {agent_id}: {e}")
            return self._maintain_heading_precise(env, agent_id, 180.0)

    def _handle_missile_launch(self, env, agent_id, current_time):
        try:
            if not env.agents[agent_id].is_alive:
                return

            if self.current_phase not in [TacticalPhase.MTR_TR, TacticalPhase.TR_DOR]:
                return

            last_launch = self.last_missile_launch_time[agent_id]
            if agent_id.startswith('A'):
                cooldown = self.friendly_missile_cooldown
            else:
                cooldown = self.enemy_missile_cooldown
            
            if current_time - last_launch < cooldown:
                return

            if env.agents[agent_id].num_missiles <= 0:
                return

            if agent_id.startswith('A'):
                locked_targets = self.radar_manager.friendly_radar_targets.get(agent_id, {})
            else:
                locked_targets = self.radar_manager.enemy_radar_targets.get(agent_id, {})
            
            if not locked_targets:
                return

            target_id = list(locked_targets.keys())[0]
            if target_id in env.agents and env.agents[target_id].is_alive:
                target = env.agents[target_id]
                self._launch_missile(env, agent_id, target, current_time)
                self.last_missile_launch_time[agent_id] = current_time

        except Exception as e:
            logging.error(f"导弹发射错误 {agent_id}: {e}")

    def _normalize_angle_diff(self, angle_diff):
        """标准化角度差到[-180, 180]范围"""
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        return angle_diff
    
    def _convert_altitude_to_index(self, altitude_cmd):
        if altitude_cmd < 0:
            logging.warning(f"🛡️ 俯冲指令{altitude_cmd:.0f}m已禁用")
            altitude_cmd = 0

        altitude_values = np.array([0, 50, 150, 300, 500, 750, 1000, 1500])
        distances = np.abs(altitude_values - altitude_cmd)
        return np.argmin(distances) + 7

    def _convert_heading_to_index(self, heading_cmd):
        heading_cmd = np.clip(heading_cmd, -np.pi, np.pi)
        distances = np.abs(self.norm_delta_heading - heading_cmd)
        return np.argmin(distances)

    def _convert_velocity_to_index(self, velocity_cmd):
        distances = np.abs(self.norm_delta_velocity - velocity_cmd)
        return np.argmin(distances)

    def _use_lowlevel_policy(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        if self.my_lowlevel_policy is None:
            return np.array([0.0, 0.1, 0.0, 0.8])

        try:
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)

            # 安全索引访问
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
            logging.error(f"低级策略错误: {e}")
            return np.array([0.0, 0.1, 0.0, 0.8])

    def _calculate_distance(self, aircraft1, aircraft2):
        """计算两架飞机之间的距离（米）- 完全复制drag_shoot实现"""
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        return np.linalg.norm(pos1 - pos2)

    def _calculate_reward(self, env, agent_id):
        if not env.agents[agent_id].is_alive:
            return -10.0
        return 0.1
    
    def _check_rwr_threats(self, env):
        """检查RWR威胁状态（文档3.7.1节）"""
        try:
            for agent_id in ["A0100", "A0200", "B0100", "B0200"]:
                if agent_id not in env.agents or not env.agents[agent_id].is_alive:
                    continue
                
                # 获取RWR威胁等级
                threat_level = self.get_rwr_threat_level(agent_id)
                
                if threat_level > 0:
                    # 获取威胁源详情
                    threat_sources = self.get_rwr_threat_sources(agent_id)
                    
                    # 格式化威胁信息
                    threat_str = ""
                    for threat in threat_sources:
                        threat_str += f"{threat['source']}(等级{threat['level']},方位{threat['bearing']:.0f}°) "
                    
                    # 根据威胁等级显示不同级别的警告
                    if threat_level >= 3:
                        logging.warning(f"🚨 {agent_id} RWR告警: 等级{threat_level} - {threat_str}")
                    elif threat_level == 2:
                        logging.info(f"⚠️  {agent_id} RWR告警: 等级{threat_level} - {threat_str}")
                    else:
                        logging.debug(f"📡 {agent_id} RWR告警: 等级{threat_level} - {threat_str}")
                        
        except Exception as e:
            logging.error(f"❌ RWR威胁检测错误: {e}")
    
    def _launch_missile(self, env, agent_id: str, target, current_time: float):
        """发射导弹 - 根据发射平台选择导弹类型"""
        try:
            aircraft = env.agents[agent_id]
            
            # 检查导弹数量
            if aircraft.num_missiles <= 0:
                logging.warning(f"⚠️ {agent_id} 导弹已用尽，无法发射")
                return

            # 创建导弹ID
            missile_count = 2 - aircraft.num_missiles + 1
            base_id = agent_id[0] + agent_id[2:]
            missile_uid = f"{base_id}{missile_count}"
            
            # 检查是否已存在（防止重复发射）
            if missile_uid in env._tempsims:
                logging.warning(f"⚠️ 导弹{missile_uid}已存在，跳过发射")
                return

            # 根据发射平台选择导弹类型
            if agent_id.startswith('A'):  # 我方飞机 - 使用AIM-120C7
                from envs.JSBSim.core.simulatior import MissileSimulator
                missile = MissileSimulator.create(
                    parent=aircraft,
                    target=target,
                    uid=missile_uid
                )
                missile_type = "AIM-120C-7"
            else:  # 敌方飞机 - 使用R-27ER
                from r27er_missile import R27ERMissileSimulator
                missile = R27ERMissileSimulator.create(
                    parent=aircraft,
                    target=target,
                    uid=missile_uid
                )
                missile_type = "R-27ER"

            # 添加到环境
            env.add_temp_simulator(missile)

            # 初始化导弹记录
            if not hasattr(env, '_missile_records'):
                env._missile_records = {}
            
            env._missile_records[missile_uid] = {
                'launcher': agent_id,
                'target': target.uid,
                'type': missile_type,
                'status': 'LAUNCHED',
                'launch_time': current_time,
                'launch_position': aircraft.get_position().copy(),
                'launch_velocity': aircraft.get_velocity().copy()
            }

            # 更新状态
            aircraft.num_missiles -= 1
            logging.info(f"✈️  {agent_id} 发射{missile_type}导弹 {missile_uid} -> {target.uid} at t={current_time:.1f}s")

        except Exception as e:
            logging.error(f"❌ 导弹发射失败 {agent_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("回转射击战术任务模块")
    print("请使用 run_turn_around_shooting.py 脚本运行仿真")
