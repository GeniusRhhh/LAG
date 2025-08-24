#!/usr/bin/env python3
"""
前后攻击战术任务 - 基于拖曳射击架构
僚机隐藏在长机后方，利用攻击的突然性和时间差压制敌机
"""

import logging
import numpy as np
import torch
from enum import Enum
from gym import spaces
from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.termination_conditions.termination_condition_base import BaseTerminationCondition
from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers, CompositeManeuverExecutor


class DragShootTermination(BaseTerminationCondition):
    """拖曳射击专用终止条件 - 只有双方全灭才终止"""

    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 1000)  # 1000米
        self.max_steps = getattr(config, 'max_steps', 2500)

    def get_termination(self, task, env, agent_id, info={}):
        """
        自定义终止条件：
        1. 高度过低
        2. 时间限制
        3. 极端状态
        4. 过载
        5. 双方全灭（不是单架飞机被击落）
        """
        # 检查高度过低
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= self.altitude_limit:
            self.log(f"{agent_id} altitude too low: {current_alt:.1f}m")
            return True, False, info

        # 检查时间限制
        if env.current_step >= self.max_steps:
            self.log(f"Time limit reached: {env.current_step} steps")
            return True, False, info

        # 检查极端状态
        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            self.log(f"{agent_id} extreme state detected")
            return True, False, info

        # 检查过载
        g_force = env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)
        if abs(g_force) > 12:
            self.log(f"{agent_id} excessive G-force: {g_force:.1f}G")
            return True, False, info

        # 检查双方全灭（不是单架飞机被击落）
        alive_agents = [agent for agent in env.agents.values() if agent.is_alive]
        if len(alive_agents) == 0:
            self.log("All agents destroyed")
            return True, False, info

        # 检查一方全灭
        alive_blue = [agent for agent in env.agents.values() if agent.is_alive and agent.side == 0]
        alive_red = [agent for agent in env.agents.values() if agent.is_alive and agent.side == 1]

        if len(alive_blue) == 0:
            self.log("Blue team eliminated")
            return True, False, info
        elif len(alive_red) == 0:
            self.log("Red team eliminated")
            return True, False, info

        return False, False, info


class TacticalPhase(Enum):
    """战术阶段枚举"""
    NLT_MELD = "NLT_MELD"      # 81km+ 平稳飞行
    MELD_MTR = "MELD_MTR"      # 50-81km 僚机调整到长机后方
    MTR_TR = "MTR_TR"          # 40-50km 保持一字队形，长机准备发射
    TR_DOR = "TR_DOR"          # 19.6-40km 长机发射后返航，僚机准备发射
    DOR_DR = "DOR_DR"          # 14.5-19.6km 僚机发射后返航
    DISENGAGEMENT = "DISENGAGEMENT"  # <14.5km 脱离


class FrontBackAttackTacticalTask(MultipleCombatTask):
    """
    前后攻击战术任务 - 基于拖曳射击的成功架构
    
    核心概念：僚机隐藏在长机后方，误导敌方雷达识别，
    利用攻击的突然性和时间差压制敌机
    
    动作空间架构：
    - 定义的动作空间: [41, 41, 41, 30] (继承自MultipleCombatTask，但实际不使用)
    - 实际使用的动作空间: [12, 12, 6] (高度, 航向, 速度)
    - 通过_convert_to_jsbsim_action转换为JSBSim可识别的动作
    """

    def __init__(self, config):
        super().__init__(config)
        
        # 战术状态管理
        self.current_phase = TacticalPhase.NLT_MELD
        self.phase_start_time = 0.0
        self.last_phase_update_time = 0.0
        
        # 前后攻击特有：队形管理状态
        self.formation_states = {
            "A0100": {  # 长机 - 前方攻击者
                "role": "leader",
                "target_y_offset": 0.0,  # 长机在中心线
                "formation_established": True
            },
            "A0200": {  # 僚机 - 后方攻击者
                "role": "wingman", 
                "target_y_offset": 0.0,  # 僚机也在中心线，但在长机后方
                "target_x_offset": -5556.0,  # 后方3海里 (1海里=1852m)
                "formation_established": False
            }
        }
        
        # 前后攻击时间线管理
        self.attack_timeline = {
            "leader_attack_phase": None,  # 长机攻击阶段开始时间
            "wingman_attack_delay": 8.0,  # 僚机攻击滞后8秒
            "leader_missile_launched": False,
            "wingman_attack_ready": False
        }
        
        # 导弹发射管理
        self.last_missile_launch_time = {}
        self.missile_launch_count = {}
        
        # Short Skate机动状态管理
        self.short_skate_states = {}
        
        # 距离计算缓存
        self.distance_cache = {}
        self.distance_cache_time = -1
        
        # 雷达状态管理
        self.radar_states = {"A0100": "SEARCH", "A0200": "SEARCH", "B0100": "SEARCH", "B0200": "SEARCH"}
        
        # 机动执行器
        self.maneuver_executor = CompositeManeuverExecutor()
        
        # 战术日志
        self.tactical_log = []
        
        logging.info("🎯 前后攻击战术任务初始化完成")

    def load_variables(self):
        """加载环境变量 - 与MultipleCombatTask保持一致"""
        self.state_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
            c.velocities_v_north_mps,
            c.velocities_v_east_mps,
            c.velocities_v_down_mps,
            c.velocities_u_mps,
            c.velocities_v_mps,
            c.velocities_w_mps,
            c.velocities_vc_mps,
            c.accelerations_n_pilot_x_norm,
            c.accelerations_n_pilot_y_norm,
            c.accelerations_n_pilot_z_norm,
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        """定义观测空间 - 与MultipleCombatTask保持一致"""
        self.obs_length = 9 + (self.num_agents - 1) * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        """定义动作空间 - 与MultipleCombatTask保持一致"""
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def reset(self, env):
        """重置任务状态"""
        # 重置战术状态
        self.current_phase = TacticalPhase.NLT_MELD
        self.phase_start_time = 0.0
        self.last_phase_update_time = 0.0
        
        # 重置前后攻击状态
        self.formation_states["A0200"]["formation_established"] = False
        self.attack_timeline = {
            "leader_attack_phase": None,
            "wingman_attack_delay": 8.0,
            "leader_missile_launched": False,
            "wingman_attack_ready": False
        }
        
        # 重置其他状态
        self.last_missile_launch_time.clear()
        self.missile_launch_count.clear()
        self.short_skate_states.clear()
        self.distance_cache.clear()
        self.distance_cache_time = -1
        self.radar_states = {"A0100": "SEARCH", "A0200": "SEARCH", "B0100": "SEARCH", "B0200": "SEARCH"}
        self.tactical_log.clear()
        
        logging.info("🎯 前后攻击战术任务重置完成")
        return super().reset(env)

    def get_obs(self, env, agent_id):
        """获取观测"""
        # 更新战术阶段
        self._update_tactical_phase(env)
        
        # 更新距离缓存
        self._update_distance_cache(env)
        
        # 更新雷达状态
        self._update_radar_states(env)
        
        # 执行导弹发射逻辑
        current_time = env.current_step * env.time_interval
        self._handle_missile_launch(env, agent_id, current_time)
        
        # 获取基础观测
        obs = super().get_obs(env, agent_id)
        
        return obs

    def step(self, env):
        """执行前后攻击战术步骤"""
        # 更新战术阶段
        self._update_tactical_phase(env)

        # 处理导弹发射
        current_time = env.current_step * env.time_interval
        for agent_id in env._jsbsims.keys():
            if env._jsbsims[agent_id].is_alive:
                self._handle_missile_launch(env, agent_id, current_time)

        # 详细状态信息 - 每5秒打印一次
        if env.current_step % 25 == 0:
            self._print_detailed_status(env, current_time)

        # 获取观测
        obs = {agent_id: self.get_obs(env, agent_id) for agent_id in env.agents.keys()}

        # 构建共享观测
        all_obs = np.stack([obs[agent_id] for agent_id in sorted(env.agents.keys())], axis=0)
        share_obs = np.tile(all_obs.flatten(), (len(env.agents.keys()), 1))
        share_obs = {agent_id: share_obs[i] for i, agent_id in enumerate(sorted(env.agents.keys()))}

        # 计算奖励
        rewards = {}
        dones = {}
        infos = {}

        for agent_id in env.agents.keys():
            reward_sum = 0.0
            reward_details = {}

            # 简化的奖励计算
            if env.agents[agent_id].is_alive:
                reward_sum = 0.1  # 存活奖励
            else:
                reward_sum = -1.0  # 被击落惩罚

            rewards[agent_id] = reward_sum

            # 终止条件检查
            done = not env.agents[agent_id].is_alive
            dones[agent_id] = done

            infos[agent_id] = {
                'reward_details': reward_details,
                'tactical_phase': self.current_phase.value,
                'formation_established': self.formation_states.get(agent_id, {}).get('formation_established', False)
            }

        return obs, share_obs, rewards, dones, infos

    def _print_detailed_status(self, env, current_time: float):
        """打印详细的战术状态信息"""
        try:
            # 获取主要对抗双方
            leader_red = env._jsbsims.get("A0100")
            leader_blue = env._jsbsims.get("B0100")

            if not leader_red or not leader_blue:
                return

            # 计算距离
            distance = self._calculate_distance(leader_red, leader_blue)

            # 使用传入的current_time参数
            logging.info(f"🎯 t={current_time:.1f}s 阶段={self.current_phase.value} 距离={distance/1000:.1f}km")

            # 打印队形状态
            if self.formation_states["A0200"]["formation_established"]:
                logging.info("✅ 僚机后方队形已建立")
            else:
                logging.info("⏳ 僚机正在建立后方队形")

            # 打印攻击时间线
            if self.attack_timeline["leader_missile_launched"]:
                logging.info("✅ 长机已先行发射")
            if self.attack_timeline["wingman_attack_ready"]:
                logging.info("✅ 僚机攻击就绪")

        except Exception as e:
            logging.error(f"❌ 详细状态打印失败: {e}")

    def get_action(self, env, agent_id, obs):
        """获取动作 - 前后攻击战术核心逻辑"""
        try:
            # 获取战术指令索引
            if agent_id == "A0100":  # 长机
                altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_leader_command_indices(env, agent_id)
            elif agent_id == "A0200":  # 僚机
                altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_wingman_command_indices(env, agent_id)
            else:
                # 敌机简单AI - 朝友方前进
                return self._get_enemy_simple_ai(env, agent_id)

            # 转换为JSBSim动作
            action = self._convert_cmd_to_action(altitude_cmd_id, heading_cmd_id, velocity_cmd_id, 0)

            # 记录战术日志
            current_time = env.current_step * env.time_interval
            if current_time - self.last_phase_update_time > 5.0:  # 每5秒记录一次
                self._log_tactical_status(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
                self.last_phase_update_time = current_time

            return action

        except Exception as e:
            logging.error(f"❌ {agent_id} 获取动作失败: {e}")
            return np.array([0.0, 0.0, 0.0, 0.8], dtype=np.float32)

    def _get_leader_command_indices(self, env, agent_id: str):
        """长机战术指令索引 - 前后攻击战术：前方攻击者"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_time = env.current_step * env.time_interval

        # 条件1：已经开始short_skate机动（防止中断）
        if agent_id in self.short_skate_states:
            return self._execute_short_skate_precise(env, agent_id, current_time)

        # 前后攻击长机战术：作为前方攻击者
        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR]:
            # 长机平稳接敌，保持中心线飞行
            return self._maintain_steady_approach(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.MTR_TR:
            # 长机准备先行攻击 - 保持稳定，精确瞄准
            return self._maintain_steady_approach(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 长机先行攻击阶段
            self.attack_timeline["leader_attack_phase"] = current_time

            # 如果已经发射过导弹，执行左侧脱离
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                return self._execute_left_rtb_maneuver(env, agent_id, current_time)
            else:
                # 继续攻击态势
                return self._maintain_steady_approach(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.DOR_DR:
            # 长机执行左侧返航脱离
            return self._execute_left_rtb_maneuver(env, agent_id, current_time)
        else:
            return 7, 8, 3

    def _get_wingman_command_indices(self, env, agent_id: str):
        """僚机战术指令索引 - 前后攻击战术：后方攻击者"""
        current_time = env.current_step * env.time_interval

        # 计算当前阶段（僚机可能有滞后）
        wingman_phase = self._get_wingman_phase()

        # 条件1：已经开始short_skate机动（防止中断）
        if agent_id in self.short_skate_states:
            return self._execute_short_skate_precise(env, agent_id, current_time)

        # 前后攻击僚机战术：作为后方攻击者
        if wingman_phase == TacticalPhase.NLT_MELD:
            # 僚机平稳飞行，保持与长机相同航向
            return self._maintain_steady_approach(env, agent_id, 0.0)

        elif wingman_phase == TacticalPhase.MELD_MTR:
            # 僚机调整到长机后方 - 执行左侧crank机动
            if not self.formation_states[agent_id]["formation_established"]:
                return self._establish_rear_formation(env, agent_id)
            else:
                # 队形已建立，保持位置
                return self._maintain_rear_formation(env, agent_id)

        elif wingman_phase == TacticalPhase.MTR_TR:
            # 僚机保持后方队形，跟随长机
            return self._maintain_rear_formation(env, agent_id)

        elif wingman_phase == TacticalPhase.TR_DOR:
            # 僚机后方攻击阶段 - 等待长机攻击后再行动
            leader_attack_time = self.attack_timeline.get("leader_attack_phase")
            if leader_attack_time and (current_time - leader_attack_time) >= self.attack_timeline["wingman_attack_delay"]:
                # 长机已攻击足够时间，僚机开始攻击
                self.attack_timeline["wingman_attack_ready"] = True

                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    # 发射导弹后执行右侧脱离
                    return self._execute_right_rtb_maneuver(env, agent_id, current_time)
                else:
                    # 执行攻击
                    return self._execute_rear_attack(env, agent_id)
            else:
                # 等待长机先攻击，保持后方位置
                return self._maintain_rear_formation(env, agent_id)

        elif wingman_phase == TacticalPhase.DOR_DR:
            # 僚机执行右侧返航脱离
            return self._execute_right_rtb_maneuver(env, agent_id, current_time)
        else:
            return 7, 8, 3

    # ==================== 前后攻击专用战术函数 ====================

    def _maintain_steady_approach(self, env, agent_id: str, target_heading: float):
        """保持稳定接敌飞行 - 修复速度控制"""
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        # 保持初始高度
        target_altitude = 6096.0  # 20000英尺

        # 默认索引 - 使用MultipleCombatTask的动作空间中点
        altitude_cmd_id = 20  # 保持高度 (41个选项的中点)
        heading_cmd_id = 20   # 保持航向 (41个选项的中点)

        # 修复速度控制 - 根据阶段调整速度
        if self.current_phase == TacticalPhase.NLT_MELD:
            velocity_cmd_id = 25  # 加速接敌
        elif self.current_phase == TacticalPhase.MELD_MTR:
            velocity_cmd_id = 22  # 继续前进
        else:
            velocity_cmd_id = 20  # 其他阶段保持速度

        # 高度控制
        altitude_diff = target_altitude - current_altitude
        if abs(altitude_diff) > 50.0:  # 50m容差
            altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        # 航向控制
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        if abs(heading_diff) > 1.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _establish_rear_formation(self, env, agent_id: str):
        """僚机建立后方队形 - 简化版本参考拖曳射击"""
        leader = env.agents.get("A0100")
        wingman = env.agents.get(agent_id)

        if not leader or not wingman or not leader.is_alive or not wingman.is_alive:
            return 7, 8, 3

        # 计算与长机的距离
        distance_to_leader = self._calculate_distance(wingman, leader)
        target_distance = 3 * 1852  # 3海里转换为米

        # 获取长机航向
        leader_heading = np.rad2deg(leader.get_property_value(c.attitude_psi_rad))
        current_heading = np.rad2deg(wingman.get_property_value(c.attitude_psi_rad))
        current_altitude = wingman.get_property_value(c.position_h_sl_m)

        # 默认索引
        altitude_cmd_id = 20  # 保持高度
        heading_cmd_id = 20   # 保持航向
        velocity_cmd_id = 20  # 保持速度

        # 队形建立逻辑
        distance_error = distance_to_leader - target_distance

        if abs(distance_error) > 500:  # 500米容差
            # 需要调整位置
            if distance_error > 0:
                # 距离太远，向长机靠近
                velocity_cmd_id = 5  # 加速
            else:
                # 距离太近，减速
                velocity_cmd_id = 1  # 减速

            # 朝向长机方向
            heading_diff = leader_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360

            if abs(heading_diff) > 5.0:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
        else:
            # 队形建立成功
            if not self.formation_states[agent_id]["formation_established"]:
                self.formation_states[agent_id]["formation_established"] = True
                logging.info(f"✅ {agent_id}僚机后方队形建立成功，距离={distance_to_leader/1852:.1f}海里")

            # 保持队形 - 跟随长机航向
            heading_diff = leader_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360

            if abs(heading_diff) > 3.0:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id



    def _maintain_rear_formation(self, env, agent_id: str):
        """僚机保持后方队形 - 简化版本"""
        leader = env.agents.get("A0100")
        wingman = env.agents.get(agent_id)

        if not leader or not wingman or not leader.is_alive or not wingman.is_alive:
            return 7, 8, 3

        # 获取长机航向，跟随长机
        leader_heading = np.rad2deg(leader.get_property_value(c.attitude_psi_rad))
        leader_altitude = leader.get_property_value(c.position_h_sl_m)

        current_heading = np.rad2deg(wingman.get_property_value(c.attitude_psi_rad))
        current_altitude = wingman.get_property_value(c.position_h_sl_m)

        # 默认索引
        altitude_cmd_id = 7  # 保持高度
        heading_cmd_id = 8   # 保持航向
        velocity_cmd_id = 3  # 保持速度

        # 高度控制 - 与长机保持相同高度
        altitude_diff = leader_altitude - current_altitude
        if abs(altitude_diff) > 50.0:
            altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        # 航向控制 - 跟随长机航向
        heading_diff = leader_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        if abs(heading_diff) > 2.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _execute_rear_attack(self, env, agent_id: str):
        """僚机执行后方攻击"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

        # 获取敌机位置进行瞄准
        enemy_agent = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
        if enemy_agent and enemy_agent.is_alive:
            enemy_pos = np.array([
                enemy_agent.get_property_value(c.position_long_gc_deg),
                enemy_agent.get_property_value(c.position_lat_geod_deg)
            ])
            own_pos = np.array([
                env.agents[agent_id].get_property_value(c.position_long_gc_deg),
                env.agents[agent_id].get_property_value(c.position_lat_geod_deg)
            ])

            # 计算指向敌机的航向
            delta_pos = enemy_pos - own_pos
            target_heading = np.rad2deg(np.arctan2(delta_pos[0], delta_pos[1]))
            if target_heading < 0:
                target_heading += 360
        else:
            target_heading = 0.0  # 默认朝北

        # 默认索引
        altitude_cmd_id = 7  # 保持高度
        heading_cmd_id = 8   # 保持航向
        velocity_cmd_id = 4  # 加速攻击

        # 航向控制 - 精确瞄准敌机
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        if abs(heading_diff) > 1.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _execute_left_rtb_maneuver(self, env, agent_id: str, current_time: float):
        """长机执行左侧返航机动"""
        if agent_id not in self.short_skate_states:
            # 初始化左侧返航机动
            self.short_skate_states[agent_id] = {
                "phase": "left_rtb",
                "phase_start_time": current_time,
                "total_start_time": current_time,
                "target_heading": 270.0,  # 左转90度返航
                "initial_heading": np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad)),
                "initial_altitude": env.agents[agent_id].get_property_value(c.position_h_sl_m)
            }

        state = self.short_skate_states[agent_id]
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

        # 默认索引
        altitude_cmd_id = 7  # 保持高度
        heading_cmd_id = 8   # 保持航向
        velocity_cmd_id = 5  # 加速脱离

        # 高度控制 - 保持当前高度
        if "initial_altitude" in state:
            altitude_diff = state["initial_altitude"] - current_altitude
            if abs(altitude_diff) > 100.0:
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        # 航向控制 - 左转返航
        target_heading = state["target_heading"]
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        if abs(heading_diff) > 2.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _execute_right_rtb_maneuver(self, env, agent_id: str, current_time: float):
        """僚机执行右侧返航机动"""
        if agent_id not in self.short_skate_states:
            # 初始化右侧返航机动
            self.short_skate_states[agent_id] = {
                "phase": "right_rtb",
                "phase_start_time": current_time,
                "total_start_time": current_time,
                "target_heading": 90.0,  # 右转90度返航
                "initial_heading": np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad)),
                "initial_altitude": env.agents[agent_id].get_property_value(c.position_h_sl_m)
            }

        state = self.short_skate_states[agent_id]
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

        # 默认索引
        altitude_cmd_id = 7  # 保持高度
        heading_cmd_id = 8   # 保持航向
        velocity_cmd_id = 5  # 加速脱离

        # 高度控制 - 保持当前高度
        if "initial_altitude" in state:
            altitude_diff = state["initial_altitude"] - current_altitude
            if abs(altitude_diff) > 100.0:
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        # 航向控制 - 右转返航
        target_heading = state["target_heading"]
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        if abs(heading_diff) > 2.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _get_wingman_phase(self):
        """获取僚机当前阶段（可能有滞后）"""
        # 前后攻击中，僚机在MELD_MTR阶段需要调整队形，可能有轻微滞后
        if self.current_phase == TacticalPhase.MELD_MTR:
            if not self.formation_states["A0200"]["formation_established"]:
                return TacticalPhase.MELD_MTR  # 继续调整队形

        return self.current_phase

    # ==================== 辅助函数 ====================

    def _convert_altitude_to_index(self, altitude_diff: float) -> int:
        """将高度差转换为动作索引"""
        if altitude_diff > 200:
            return 11  # +300m
        elif altitude_diff > 100:
            return 10  # +150m
        elif altitude_diff > 25:
            return 9   # +50m
        elif altitude_diff < -200:
            return 5   # -300m
        elif altitude_diff < -100:
            return 6   # -150m
        elif altitude_diff < -25:
            return 7   # -50m
        else:
            return 7   # 保持高度

    def _convert_heading_to_index(self, heading_diff_rad: float) -> int:
        """将航向差转换为动作索引"""
        heading_diff_deg = np.rad2deg(heading_diff_rad)

        if heading_diff_deg > 15:
            return 11  # 大右转
        elif heading_diff_deg > 5:
            return 10  # 中右转
        elif heading_diff_deg > 1:
            return 9   # 小右转
        elif heading_diff_deg < -15:
            return 5   # 大左转
        elif heading_diff_deg < -5:
            return 6   # 中左转
        elif heading_diff_deg < -1:
            return 7   # 小左转
        else:
            return 8   # 保持航向



    def _convert_cmd_to_action(self, altitude_cmd_id: int, heading_cmd_id: int, velocity_cmd_id: int, shoot_cmd: int):
        """将指令索引转换为JSBSim动作 - 参考拖曳射击实现"""
        # 使用MultipleCombatTask的normalize_action方法
        # 动作空间是MultiDiscrete([41, 41, 41, 30])
        action_indices = [altitude_cmd_id, heading_cmd_id, velocity_cmd_id, shoot_cmd]

        # 确保索引在有效范围内
        action_indices[0] = np.clip(action_indices[0], 0, 40)  # 高度
        action_indices[1] = np.clip(action_indices[1], 0, 40)  # 航向
        action_indices[2] = np.clip(action_indices[2], 0, 40)  # 速度
        action_indices[3] = np.clip(action_indices[3], 0, 29)  # 射击

        # 转换为连续动作 - 参考MultipleCombatTask.normalize_action
        norm_act = np.zeros(4)
        norm_act[0] = action_indices[0] * 2. / (41 - 1.) - 1.  # [-1, 1]
        norm_act[1] = action_indices[1] * 2. / (41 - 1.) - 1.  # [-1, 1]
        norm_act[2] = action_indices[2] * 2. / (41 - 1.) - 1.  # [-1, 1]
        norm_act[3] = action_indices[3] * 0.5 / (30 - 1.) + 0.4  # [0.4, 0.9]

        return norm_act

    # ==================== 核心辅助函数 ====================

    def _update_tactical_phase(self, env):
        """更新战术阶段"""
        # 获取主要对抗双方
        leader_red = env._jsbsims.get("A0100")
        leader_blue = env._jsbsims.get("B0100")

        if not leader_red or not leader_blue or not leader_red.is_alive or not leader_blue.is_alive:
            return

        # 计算距离
        distance = self._calculate_distance(leader_red, leader_blue)

        # 确定当前阶段
        new_phase = self._get_phase_by_distance(distance)

        if new_phase != self.current_phase:
            current_time = env.current_step * env.time_interval
            logging.info(f"🎯 阶段转换: {self.current_phase.value} -> {new_phase.value} "
                        f"at t={current_time:.1f}s, distance={distance/1000:.1f}km")
            self.current_phase = new_phase

    def _get_phase_by_distance(self, distance: float) -> TacticalPhase:
        """根据距离确定战术阶段 - 调整阈值确保正确转换"""
        if distance > 75000:  # 75km
            return TacticalPhase.NLT_MELD
        elif distance > 55000:  # 55km
            return TacticalPhase.MELD_MTR
        elif distance > 45000:  # 45km
            return TacticalPhase.MTR_TR
        elif distance > 35000:  # 35km
            return TacticalPhase.TR_DOR
        elif distance > 20000:  # 20km
            return TacticalPhase.DOR_DR
        else:
            return TacticalPhase.DISENGAGEMENT

    def _calculate_distance(self, aircraft1, aircraft2) -> float:
        """计算两架飞机之间的距离 - 参考拖曳射击实现"""
        try:
            pos1 = aircraft1.get_position()
            pos2 = aircraft2.get_position()
            return np.linalg.norm(pos1 - pos2)
        except Exception as e:
            logging.error(f"❌ 距离计算错误: {e}")
            return float('inf')

    def _update_distance_cache(self, env):
        """更新距离缓存"""
        current_time = env.current_step * env.time_interval
        if current_time == self.distance_cache_time:
            return

        self.distance_cache.clear()
        self.distance_cache_time = current_time

        # 计算所有智能体之间的距离
        for agent1_id in env.agents:
            if not env.agents[agent1_id].is_alive:
                continue
            for agent2_id in env.agents:
                if agent1_id != agent2_id and env.agents[agent2_id].is_alive:
                    key = f"{agent1_id}_{agent2_id}"
                    self.distance_cache[key] = self._calculate_distance(
                        env.agents[agent1_id], env.agents[agent2_id]
                    )

    def _update_radar_states(self, env):
        """更新雷达状态"""
        try:
            for agent_id in self.radar_states:
                if agent_id in env.agents and env.agents[agent_id].is_alive:
                    # 简化的雷达状态逻辑
                    if agent_id.startswith('A'):  # 友方
                        self.radar_states[agent_id] = "SEARCH"
                    else:  # 敌方
                        self.radar_states[agent_id] = "SEARCH"
        except Exception as e:
            logging.error(f"❌ 获取雷达状态错误: {e}")

    def _log_tactical_status(self, env, agent_id: str, alt_cmd: int, hdg_cmd: int, vel_cmd: int):
        """记录战术状态"""
        try:
            current_time = env.current_step * env.time_interval
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

            logging.info(f"🎯 {agent_id} t={current_time:.1f}s 阶段={self.current_phase.value} "
                        f"高度={current_altitude:.0f}m 航向={current_heading:.0f}° "
                        f"指令=[{alt_cmd},{hdg_cmd},{vel_cmd}]")
        except Exception as e:
            logging.error(f"❌ {agent_id} 战术状态记录失败: {e}")

    def _handle_missile_launch(self, env, agent_id: str, current_time: float):
        """处理导弹发射 - 前后攻击专用逻辑"""
        if not env.agents[agent_id].is_alive:
            return

        # 检查导弹数量
        if env.agents[agent_id].num_missiles <= 0:
            return

        current_time = env.current_step * env.time_interval

        # 检查发射冷却时间
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if current_time - last_launch < 2.0:  # 2秒冷却时间
            return

        # 前后攻击战术：长机在DOR_DR阶段已经返航，禁止发射
        if agent_id.startswith('A') and self.current_phase == TacticalPhase.DOR_DR:
            if agent_id == "A0100":  # 长机在DOR_DR阶段已经返航，禁止发射
                return
            # 僚机在DOR_DR阶段可以发射（前后攻击特殊逻辑）

        # 获取最近的敌机目标
        target_agent = None
        min_distance = float('inf')

        for enemy_id in ["B0100", "B0200"]:
            if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                distance = self._calculate_distance(env.agents[agent_id], env.agents[enemy_id])
                if distance < min_distance:
                    min_distance = distance
                    target_agent = env.agents[enemy_id]

        if not target_agent:
            return

        distance = min_distance
        should_launch = False

        # 前后攻击专用发射逻辑
        if agent_id == "A0100":  # 长机：前方攻击者，先行发射
            # 长机在MTR_TR和TR_DOR阶段先行发射
            in_launch_phase = self.current_phase in [TacticalPhase.MTR_TR, TacticalPhase.TR_DOR]
            in_launch_range = 35000 <= distance <= 50000  # 调整发射距离

            should_launch = in_launch_phase and in_launch_range

            if should_launch:
                # 记录长机攻击时间
                self.attack_timeline["leader_missile_launched"] = True
                self.attack_timeline["leader_attack_phase"] = current_time
                logging.info(f"🎯 A0100长机先行发射: 阶段={self.current_phase.value}, 距离={distance/1000:.1f}km, 时间={current_time:.1f}s")

        elif agent_id == "A0200":  # 僚机：后方攻击者，滞后发射
            # 僚机必须等待长机先发射，且有足够的时间间隔
            leader_launched = self.attack_timeline.get("leader_missile_launched", False)
            leader_attack_time = self.attack_timeline.get("leader_attack_phase")

            # 僚机发射条件：
            # 1. 长机已经发射
            # 2. 距离长机攻击开始已经过了足够时间
            # 3. 在合适的阶段和距离
            if leader_launched and leader_attack_time:
                time_since_leader_attack = current_time - leader_attack_time
                if time_since_leader_attack >= self.attack_timeline["wingman_attack_delay"]:
                    # 僚机发射阶段和距离
                    in_launch_phase = self.current_phase in [TacticalPhase.TR_DOR, TacticalPhase.DOR_DR]
                    in_launch_range = 25000 <= distance <= 40000  # 僚机从后方攻击，距离更近

                    should_launch = in_launch_phase and in_launch_range

                    if should_launch:
                        logging.info(f"🎯 A0200僚机后方攻击发射: 阶段={self.current_phase.value}, "
                                   f"距离={distance/1000:.1f}km, 延迟={time_since_leader_attack:.1f}s")
                else:
                    logging.debug(f"A0200僚机等待攻击延迟: {time_since_leader_attack:.1f}s < {self.attack_timeline['wingman_attack_delay']}s")
            else:
                logging.debug(f"A0200僚机等待长机先攻击: leader_launched={leader_launched}, leader_attack_time={leader_attack_time}")

        # 执行发射
        if should_launch:
            try:
                # 发射导弹
                env.agents[agent_id].launch_missile()
                self.last_missile_launch_time[agent_id] = current_time

                # 更新发射计数
                if agent_id not in self.missile_launch_count:
                    self.missile_launch_count[agent_id] = 0
                self.missile_launch_count[agent_id] += 1

                logging.info(f"🚀 {agent_id} 发射导弹 #{self.missile_launch_count[agent_id]} "
                           f"目标距离={distance/1000:.1f}km 阶段={self.current_phase.value}")

            except Exception as e:
                logging.error(f"❌ {agent_id} 导弹发射失败: {e}")

    def _execute_short_skate_precise(self, env, agent_id: str, current_time: float):
        """执行精确的Short Skate机动"""
        if agent_id not in self.short_skate_states:
            # 初始化Short Skate状态
            self.short_skate_states[agent_id] = {
                "phase": "crank",
                "phase_start_time": current_time,
                "total_start_time": current_time,
                "initial_heading": None,
                "initial_altitude": None
            }

        state = self.short_skate_states[agent_id]
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

        if state["initial_heading"] is None:
            state["initial_heading"] = current_heading
            state["initial_altitude"] = current_altitude

        phase_time = current_time - state["phase_start_time"]

        # Short Skate机动阶段
        if state["phase"] == "crank":
            # Crank阶段：侧向机动
            if phase_time < 8.0:
                if agent_id == "A0100":  # 长机左转
                    target_heading = state["initial_heading"] - 45.0
                else:  # 僚机右转
                    target_heading = state["initial_heading"] + 45.0
            else:
                state["phase"] = "turn_cold"
                state["phase_start_time"] = current_time

        elif state["phase"] == "turn_cold":
            # Turn Cold阶段：转向返航
            if phase_time < 12.0:
                target_heading = 180.0  # 南向返航
            else:
                state["phase"] = "escape"
                state["phase_start_time"] = current_time

        else:  # escape阶段
            # Escape阶段：加速脱离
            target_heading = 180.0  # 继续南向

        # 标准化航向
        if target_heading < 0:
            target_heading += 360
        elif target_heading >= 360:
            target_heading -= 360

        # 默认索引
        altitude_cmd_id = 7  # 保持高度
        heading_cmd_id = 8   # 保持航向
        velocity_cmd_id = 5  # 加速脱离

        # 高度控制 - 保持初始高度
        if state["initial_altitude"]:
            altitude_diff = state["initial_altitude"] - current_altitude
            if abs(altitude_diff) > 100.0:
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

        # 航向控制
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        if abs(heading_diff) > 2.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _get_enemy_simple_ai(self, env, agent_id: str):
        """敌机简单AI - 朝友方前进并发射导弹"""
        try:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

            # 默认动作索引
            altitude_cmd_id = 7  # 保持高度
            heading_cmd_id = 8   # 保持航向（朝南）
            velocity_cmd_id = 4  # 前进

            # 高度控制 - 保持20000英尺
            target_altitude = 6096.0
            altitude_diff = target_altitude - current_altitude
            if abs(altitude_diff) > 100.0:
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)

            # 航向控制 - 朝南（180度）
            target_heading = 180.0
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360

            if abs(heading_diff) > 5.0:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))

            # 速度控制 - 根据距离调整
            # 找到最近的友方飞机
            min_distance = float('inf')
            for friendly_id in ["A0100", "A0200"]:
                if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                    distance = self._calculate_distance(env.agents[agent_id], env.agents[friendly_id])
                    if distance < min_distance:
                        min_distance = distance

            # 根据距离调整速度
            if min_distance > 60000:  # 60km以上
                velocity_cmd_id = 5  # 加速接敌
            elif min_distance > 40000:  # 40km以上
                velocity_cmd_id = 4  # 正常前进
            else:
                velocity_cmd_id = 3  # 保持速度

            # 转换为动作
            action = self._convert_cmd_to_action(altitude_cmd_id, heading_cmd_id, velocity_cmd_id, 0)
            return np.array(action, dtype=np.float32)

        except Exception as e:
            logging.error(f"❌ {agent_id} 敌机AI失败: {e}")
            return np.array([0.0, 0.0, 0.0, 0.8], dtype=np.float32)
