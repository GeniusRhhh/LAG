#!/usr/bin/env python3
"""
前后攻击战术任务V2 - 完全基于拖曳射击的成功架构
僚机隐藏在长机后方3海里，形成前后纵队攻击
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
    """拖曳射击终止条件"""
    def __init__(self):
        super().__init__()
        self.name = "DragShootTermination"

    def get_termination(self, task, env, agent_id, info={}):
        """检查终止条件"""
        if not env.agents[agent_id].is_alive:
            return True
        return False


class TacticalPhase(Enum):
    """前后攻击战术阶段 - 复制拖曳射击"""
    NLT_MELD = "NLT_MELD"    # 90-81km
    MELD_MTR = "MELD_MTR"    # 81-45km
    MTR_TR = "MTR_TR"        # 45-41km
    TR_DOR = "TR_DOR"        # 41-19.6km
    DOR_DR = "DOR_DR"        # 19.6-14.5km


class FrontBackAttackTacticalTaskV2(MultipleCombatTask):
    """
    前后攻击战术任务V2 - 完全基于拖曳射击的成功架构

    动作空间架构：
    - 定义的动作空间: [41, 41, 41, 30] (继承自MultipleCombatTask，但实际不使用)
    - 实际使用的动作空间: [15, 17, 7] (高层战术指令)
    - 转换机制: 高层指令 → baseline模型 → 底层飞行控制

    主要函数调用链：
    normalize_action() → _process_front_back_tactics() → _get_tactical_command_indices()
    → _get_leader_command_indices() / _get_wingman_command_indices() / _get_enemy_command_indices()
    → _use_lowlevel_policy() → baseline模型输出底层控制指令

    平稳飞行指令: [7, 8, 3] = [高度0m变化, 航向0°变化, 速度0m/s变化]
    """

    def __init__(self, config):
        super().__init__(config)
        
        # 战术状态
        self.current_phase = TacticalPhase.NLT_MELD
        self.phase_start_time = 0.0
        self.phase_start_step = 0
        
        # 前后攻击特有状态
        self.formation_states = {
            "A0200": {
                "formation_established": False,
                "target_distance": 3 * 1852,  # 3海里
                "formation_tolerance": 1000.0  # 1000米容差
            }
        }
        
        # 攻击时间线
        self.attack_timeline = {
            "leader_missile_launched": False,
            "leader_attack_phase": None,
            "wingman_attack_delay": 8.0,  # 僚机延迟8秒攻击
            "wingman_attack_ready": False
        }
        
        # 导弹发射管理
        self.last_missile_launch_time = {}
        self.missile_launch_count = {}
        self.friendly_missile_cooldown = 2.0  # 友方2秒冷却
        self.enemy_missile_cooldown = 10.0    # 敌方10秒冷却
        
        # Short Skate机动状态
        self.short_skate_states = {}
        
        # 距离缓存
        self.distance_cache = {}
        self.distance_cache_time = -1
        
        # 雷达状态
        self.radar_states = {}
        
        # 初始化低级策略模型 - 完全复制拖曳射击
        self._init_lowlevel_policy()

        # 初始化机动模块 - 复制拖曳射击
        self.basic_maneuvers = BasicManeuvers()
        self.composite_maneuver_executor = CompositeManeuverExecutor()
        self.active_maneuvers = {}

        logging.info("✅ 前后攻击战术任务V2初始化完成")

    def _init_lowlevel_policy(self):
        """初始化低级策略模型 - 完全复制拖曳射击"""
        try:
            # 加载低级策略网络 - 完全照抄pure_maneuver_task
            model_path = get_root_dir() + '/envs/JSBSim/model/baseline_actor.pt'
            self.my_lowlevel_policy = BaselineActor(
                obs_space=spaces.Box(low=-10, high=10., shape=(12,)),
                act_space=spaces.Box(low=-1, high=1., shape=(4,))
            )
            self.my_lowlevel_policy.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.my_lowlevel_policy.eval()

            # 初始化动作映射表 - 完全照抄pure_maneuver_task
            self._init_action_mappings()

            logging.info("✅ 低级策略模型加载成功")
        except Exception as e:
            logging.error(f"❌ 低级策略模型加载失败: {e}")
            self.my_lowlevel_policy = None

    def _init_action_mappings(self):
        """初始化动作映射表 - 完全照抄pure_maneuver_task"""
        # 高度变化映射 (15个选项)
        self.norm_delta_altitude = np.array([
            -1.0, -0.8, -0.6, -0.4, -0.2, -0.1, -0.05,  # 下降
            0.0,  # 保持
            0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0  # 爬升
        ])

        # 航向变化映射 (17个选项)
        self.norm_delta_heading = np.array([
            -1.0, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, -0.05, # 左转
            0.0,  # 保持
            0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0  # 右转
        ])

        # 速度变化映射 (7个选项)
        self.norm_delta_velocity = np.array([
            -1.0, -0.5, -0.2,  # 减速
            0.0,  # 保持
            0.2, 0.5, 1.0  # 加速
        ])

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

    def get_termination(self, env, agent_id, info={}):
        """获取终止条件"""
        termination = DragShootTermination()
        return termination.get_termination(self, env, agent_id, info)

    def normalize_action(self, env, agent_id, action):
        """
        动作标准化 - 完全复制拖曳射击的成功实现
        这是核心函数，将高层战术指令转换为底层飞行控制
        """
        try:
            current_time = env.current_step * env.time_interval
            
            # 处理前后攻击战术
            return self._process_front_back_tactics(env, agent_id, current_time)
            
        except Exception as e:
            logging.error(f"❌ {agent_id} normalize_action失败: {e}")
            # 返回默认的平稳飞行动作
            return np.array([0.0, 0.0, 0.0, 0.8], dtype=np.float32)

    def _process_front_back_tactics(self, env, agent_id, current_time):
        """处理前后攻击战术 - 学习拖曳射击的处理模式"""
        try:
            # 更新战术阶段
            self._update_tactical_phase(env)

            # 根据智能体角色和当前阶段生成指令索引
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_tactical_command_indices(env, agent_id)

            # 使用baseline模型 - 完全学习拖曳射击的_use_lowlevel_policy
            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

        except Exception as e:
            logging.error(f"❌ {agent_id} 前后攻击战术处理失败: {e}")
            return np.array([0.0, 0.0, 0.0, 0.8], dtype=np.float32)

    def _get_tactical_command_indices(self, env, agent_id):
        """获取战术指令索引 - 复制拖曳射击架构"""
        if agent_id == "A0100":  # 长机
            return self._get_leader_command_indices(env, agent_id)
        elif agent_id == "A0200":  # 僚机
            return self._get_wingman_command_indices(env, agent_id)
        else:  # 敌机
            return self._get_enemy_command_indices(env, agent_id)

    def _get_leader_command_indices(self, env, agent_id: str):
        """长机战术指令索引 - 基于前后攻击战术"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        # 条件1：已经开始short_skate机动（防止中断）
        if agent_id in self.short_skate_states:
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate_precise(env, agent_id, current_time)

        if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR]:
            # 长机平稳飞行 - 朝北接敌（0°）- 使用精确航向保持
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 长机在TR_DOR阶段：检查是否应该执行short_skate机动
            current_time = env.current_step * env.time_interval
            # 如果已经发射过导弹且距离上次发射超过5秒，执行short_skate机动
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) > 5.0:
                # 发射导弹后执行精确的short_skate机动
                return self._execute_short_skate_precise(env, agent_id, current_time)
            else:
                # 继续精确的平稳飞行等待发射时机
                return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.DOR_DR:
            # DOR_DR阶段：长机执行精确的short_skate机动
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate_precise(env, agent_id, current_time)
        else:
            return 7, 8, 3

    def _get_wingman_command_indices(self, env, agent_id: str):
        """僚机战术指令索引 - 基于前后攻击战术（后方队形）"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        # 条件1：已经开始short_skate机动（防止中断）
        if agent_id in self.short_skate_states:
            current_time = env.current_step * env.time_interval
            return self._execute_short_skate_precise(env, agent_id, current_time)

        # 前后攻击特有逻辑：僚机建立后方队形
        if not self.formation_states["A0200"]["formation_established"]:
            return self._establish_rear_formation(env, agent_id)
        else:
            return self._maintain_rear_formation(env, agent_id)

    def _get_enemy_command_indices(self, env, agent_id: str):
        """敌机战术指令索引 - 简单AI朝友方前进"""
        return self._get_enemy_simple_ai(env, agent_id)

    # ==================== 核心辅助函数 - 完全复制拖曳射击 ====================

    def _use_lowlevel_policy(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """使用低级策略网络 - 完全照抄拖曳射击的实现"""
        if self.my_lowlevel_policy is None:
            return np.array([0.0, 0.0, 0.0, 0.7])

        try:
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)

            # 安全索引访问，防止越界
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)

            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]

            # 填充其他观测数据
            for i in range(min(9, len(raw_obs))):
                input_obs[i + 3] = raw_obs[i]

            # 使用低级策略网络
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_obs).unsqueeze(0)
                norm_act = self.my_lowlevel_policy(input_tensor).squeeze(0).numpy()

            return norm_act
        except Exception as e:
            logging.error(f"低级策略错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def _maintain_heading_precise(self, env, agent_id: str, target_heading: float):
        """精确航向保持 - 复制拖曳射击"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

        # 默认指令索引
        altitude_cmd_id = 7  # 保持高度
        heading_cmd_id = 8   # 保持航向
        velocity_cmd_id = 3  # 保持速度

        # 高度控制 - 保持20000英尺
        target_altitude = 6096.0
        altitude_diff = target_altitude - current_altitude
        if abs(altitude_diff) > 100.0:
            if altitude_diff > 0:
                altitude_cmd_id = 9  # 爬升
            else:
                altitude_cmd_id = 5  # 下降

        # 航向控制
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        if abs(heading_diff) > 2.0:
            if heading_diff > 0:
                heading_cmd_id = 10  # 右转
            else:
                heading_cmd_id = 6   # 左转

        # 速度控制 - 根据阶段调整
        if self.current_phase == TacticalPhase.NLT_MELD:
            velocity_cmd_id = 4  # 加速接敌
        elif self.current_phase == TacticalPhase.MELD_MTR:
            velocity_cmd_id = 4  # 继续前进
        else:
            velocity_cmd_id = 3  # 保持速度

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _establish_rear_formation(self, env, agent_id: str):
        """僚机建立后方队形 - 前后攻击专用"""
        leader = env.agents.get("A0100")
        wingman = env.agents.get(agent_id)

        if not leader or not wingman or not leader.is_alive or not wingman.is_alive:
            return 7, 8, 3

        # 计算与长机的距离
        distance_to_leader = self._calculate_distance(leader, wingman)
        target_distance = self.formation_states["A0200"]["target_distance"]
        formation_tolerance = self.formation_states["A0200"]["formation_tolerance"]

        # 获取航向信息
        leader_heading = np.rad2deg(leader.get_property_value(c.attitude_psi_rad))
        current_heading = np.rad2deg(wingman.get_property_value(c.attitude_psi_rad))
        current_altitude = wingman.get_property_value(c.position_h_sl_m)
        leader_altitude = leader.get_property_value(c.position_h_sl_m)

        # 默认索引
        altitude_cmd_id = 7  # 保持高度
        heading_cmd_id = 8   # 保持航向
        velocity_cmd_id = 3  # 保持速度

        # 高度控制 - 与长机保持相同高度
        altitude_diff = leader_altitude - current_altitude
        if abs(altitude_diff) > 100.0:
            if altitude_diff > 0:
                altitude_cmd_id = 9  # 爬升
            else:
                altitude_cmd_id = 5  # 下降

        # 队形建立逻辑
        distance_error = distance_to_leader - target_distance

        if abs(distance_error) > formation_tolerance:
            # 需要调整位置
            if distance_error > 0:
                # 距离太远，加速向长机靠近
                velocity_cmd_id = 4  # 加速
            else:
                # 距离太近，减速
                velocity_cmd_id = 2  # 减速

            # 朝向长机方向
            heading_diff = leader_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360

            if abs(heading_diff) > 5.0:
                if heading_diff > 0:
                    heading_cmd_id = 10  # 右转
                else:
                    heading_cmd_id = 6   # 左转
        else:
            # 队形建立成功
            if not self.formation_states["A0200"]["formation_established"]:
                self.formation_states["A0200"]["formation_established"] = True
                logging.info(f"✅ {agent_id}僚机后方队形建立成功，距离={distance_to_leader/1852:.1f}海里")

            # 保持队形 - 跟随长机航向
            heading_diff = leader_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360

            if abs(heading_diff) > 3.0:
                if heading_diff > 0:
                    heading_cmd_id = 10  # 右转
                else:
                    heading_cmd_id = 6   # 左转

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _maintain_rear_formation(self, env, agent_id: str):
        """僚机保持后方队形"""
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
            if altitude_diff > 0:
                altitude_cmd_id = 9  # 爬升
            else:
                altitude_cmd_id = 5  # 下降

        # 航向控制 - 跟随长机
        heading_diff = leader_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360

        if abs(heading_diff) > 2.0:
            if heading_diff > 0:
                heading_cmd_id = 10  # 右转
            else:
                heading_cmd_id = 6   # 左转

        # 速度控制 - 根据阶段调整
        if self.current_phase == TacticalPhase.NLT_MELD:
            velocity_cmd_id = 4  # 加速接敌
        elif self.current_phase == TacticalPhase.MELD_MTR:
            velocity_cmd_id = 4  # 继续前进
        else:
            velocity_cmd_id = 3  # 保持速度

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

    def _get_enemy_simple_ai(self, env, agent_id: str):
        """敌机简单AI - 朝友方前进"""
        try:
            current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)

            # 默认索引
            altitude_cmd_id = 7  # 保持高度
            heading_cmd_id = 8   # 保持航向
            velocity_cmd_id = 4  # 前进

            # 高度控制 - 保持20000英尺
            target_altitude = 6096.0
            altitude_diff = target_altitude - current_altitude
            if abs(altitude_diff) > 100.0:
                if altitude_diff > 0:
                    altitude_cmd_id = 9  # 爬升
                else:
                    altitude_cmd_id = 5  # 下降

            # 航向控制 - 朝南（180度）
            target_heading = 180.0
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360

            if abs(heading_diff) > 5.0:
                if heading_diff > 0:
                    heading_cmd_id = 10  # 右转
                else:
                    heading_cmd_id = 6   # 左转

            # 速度控制 - 根据距离调整
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

            return altitude_cmd_id, heading_cmd_id, velocity_cmd_id

        except Exception as e:
            logging.error(f"❌ {agent_id} 敌机AI失败: {e}")
            return 7, 8, 3

    def _execute_short_skate_precise(self, env, agent_id: str, current_time: float):
        """执行精确的short_skate机动 - 复制拖曳射击"""
        # 简化版本，返回脱离机动
        return 7, 10, 4  # 保持高度，右转，加速

    # ==================== 战术阶段和距离管理 ====================

    def _update_tactical_phase(self, env):
        """更新战术阶段 - 完全复制拖曳射击"""
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
            self.phase_start_time = current_time
            self.phase_start_step = env.current_step

    def _get_phase_by_distance(self, distance: float) -> TacticalPhase:
        """根据距离确定战术阶段 - 复制拖曳射击"""
        distance_km = distance / 1000.0

        if distance_km > 81:
            return TacticalPhase.NLT_MELD
        elif distance_km > 45:
            return TacticalPhase.MELD_MTR
        elif distance_km > 41:
            return TacticalPhase.MTR_TR
        elif distance_km > 19.6:
            return TacticalPhase.TR_DOR
        else:
            return TacticalPhase.DOR_DR

    def _calculate_distance(self, aircraft1, aircraft2) -> float:
        """计算两架飞机之间的距离 - 复制拖曳射击"""
        try:
            pos1 = aircraft1.get_position()
            pos2 = aircraft2.get_position()
            return np.linalg.norm(pos1 - pos2)
        except Exception as e:
            logging.error(f"❌ 距离计算错误: {e}")
            return float('inf')

    # ==================== Step函数和状态管理 ====================

    def step(self, env):
        """执行前后攻击战术步骤 - 复制拖曳射击架构"""
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

        # 队形监控 - 每3秒打印一次
        if env.current_step % 15 == 0:
            self._print_formation_status(env, current_time)

        # 调用父类step方法
        obs, share_obs, rewards, dones, infos = super().step(env)

        return obs, share_obs, rewards, dones, infos

    def _handle_missile_launch(self, env, agent_id: str, current_time: float):
        """处理导弹发射 - 前后攻击专用逻辑"""
        if not env.agents[agent_id].is_alive:
            return

        # 检查导弹数量
        if env.agents[agent_id].num_missiles <= 0:
            return

        # 友方返航期间禁止发射导弹
        if agent_id.startswith('A') and self.current_phase == TacticalPhase.DOR_DR:
            return

        # 差异化冷却时间检查
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if agent_id.startswith('A'):  # 友方使用短冷却时间
            cooldown = self.friendly_missile_cooldown
        else:  # 敌方使用长冷却时间
            cooldown = self.enemy_missile_cooldown

        if current_time - last_launch < cooldown:
            return

        # 前后攻击专用发射逻辑
        should_launch = False

        if agent_id == "A0100":  # 长机：先行发射
            # 长机在MTR_TR和TR_DOR阶段先行发射
            in_launch_phase = self.current_phase in [TacticalPhase.MTR_TR, TacticalPhase.TR_DOR]

            # 计算与敌机距离
            distance = float('inf')
            for enemy_id in ["B0100", "B0200"]:
                if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                    dist = self._calculate_distance(env.agents[agent_id], env.agents[enemy_id])
                    if dist < distance:
                        distance = dist

            in_launch_range = 35000 <= distance <= 50000  # 35-50km
            should_launch = in_launch_phase and in_launch_range

            if should_launch:
                # 记录长机攻击时间
                self.attack_timeline["leader_missile_launched"] = True
                self.attack_timeline["leader_attack_phase"] = current_time
                logging.info(f"🎯 A0100长机先行发射: 阶段={self.current_phase.value}, 距离={distance/1000:.1f}km, 时间={current_time:.1f}s")

        elif agent_id == "A0200":  # 僚机：延迟发射
            # 僚机需要等待长机发射后再发射
            if (self.attack_timeline["leader_missile_launched"] and
                self.attack_timeline["leader_attack_phase"] is not None):

                # 检查是否达到延迟时间
                delay_time = current_time - self.attack_timeline["leader_attack_phase"]
                if delay_time >= self.attack_timeline["wingman_attack_delay"]:

                    # 僚机发射阶段和距离
                    in_launch_phase = self.current_phase in [TacticalPhase.TR_DOR, TacticalPhase.DOR_DR]

                    # 计算与敌机距离
                    distance = float('inf')
                    for enemy_id in ["B0100", "B0200"]:
                        if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                            dist = self._calculate_distance(env.agents[agent_id], env.agents[enemy_id])
                            if dist < distance:
                                distance = dist

                    in_launch_range = 25000 <= distance <= 40000  # 25-40km
                    should_launch = in_launch_phase and in_launch_range

                    if should_launch:
                        self.attack_timeline["wingman_attack_ready"] = True
                        logging.info(f"🎯 A0200僚机后方攻击: 延迟={delay_time:.1f}s, 距离={distance/1000:.1f}km, 时间={current_time:.1f}s")

        else:  # 敌机发射逻辑
            # 计算与友方距离
            distance = float('inf')
            for friendly_id in ["A0100", "A0200"]:
                if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                    dist = self._calculate_distance(env.agents[agent_id], env.agents[friendly_id])
                    if dist < distance:
                        distance = dist

            # 敌机在合适距离发射
            should_launch = 30000 <= distance <= 45000

        # 执行发射
        if should_launch:
            try:
                env.agents[agent_id].launch_missile()
                self.last_missile_launch_time[agent_id] = current_time
                self.missile_launch_count[agent_id] = self.missile_launch_count.get(agent_id, 0) + 1
                logging.info(f"🚀 {agent_id} 发射导弹 #{self.missile_launch_count[agent_id]} at t={current_time:.1f}s")
            except Exception as e:
                logging.error(f"❌ {agent_id} 导弹发射失败: {e}")

    def _print_detailed_status(self, env, current_time: float):
        """打印详细状态信息"""
        try:
            # 计算主要距离
            leader_red = env._jsbsims.get("A0100")
            leader_blue = env._jsbsims.get("B0100")

            if leader_red and leader_blue and leader_red.is_alive and leader_blue.is_alive:
                distance = self._calculate_distance(leader_red, leader_blue)
                logging.info(f"🎯 t={current_time:.1f}s 阶段={self.current_phase.value} 距离={distance/1000:.1f}km")

        except Exception as e:
            logging.error(f"❌ 状态打印失败: {e}")

    def _print_formation_status(self, env, current_time: float):
        """打印队形状态"""
        try:
            if not self.formation_states["A0200"]["formation_established"]:
                logging.info("⏳ 僚机正在建立后方队形")
            else:
                # 计算僚机与长机距离
                leader = env.agents.get("A0100")
                wingman = env.agents.get("A0200")
                if leader and wingman and leader.is_alive and wingman.is_alive:
                    distance = self._calculate_distance(leader, wingman)
                    logging.info(f"✅ 僚机后方队形保持中，距离长机={distance/1852:.1f}海里")

        except Exception as e:
            logging.error(f"❌ 队形状态打印失败: {e}")
