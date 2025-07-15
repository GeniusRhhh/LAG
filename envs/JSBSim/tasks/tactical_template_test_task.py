# tasks/tactical_template_test_task.py
import logging
import os

import numpy as np
import torch
from typing import Dict, Any, Tuple

import yaml
from gymnasium import spaces
from .multiplecombat_task import MultipleCombatTask,HierarchicalMultipleCombatShootTask,HierarchicalMultipleCombatTask
from ..core.simulatior import MissileSimulator
from ..tasks.TacticalTemplate import EnhancedTacticalTemplate
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor
from ..reward_functions import AltitudeRewardNew, PostureRewardNew, EventDrivenRewardNew
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.RadarModel import RadarModel

class TacticalTemplateTestTask(MultipleCombatTask):
    """战术模板测试任务 """

    def __init__(self, config):
        logging.info(f"TacticalTemplateTestTask config: {config}, type: {type(config)}")
        self.env = None
        self.initial_enemy_altitude = None  # 存储敌方初始高度
        if not isinstance(config, str):
            raise ValueError(f"config must be a string, got {type(config)}")
        self.config_path = config

        self.config = self._load_config()  # 解析后的配置对象

        self.test_template_id = getattr(config, 'test_template_id', 1)
        self.enemy_behavior = getattr(config, 'enemy_behavior', 'intelligent_baseline')
        # 调用父类初始化
        super().__init__(self.config)
        # 加载baseline策略网络用于敌方
        self.enemy_baseline_policy = BaselineActor()
        self.my_lowlevel_policy = BaselineActor()
        self.radar_model = RadarModel(max_range=120000, h_beamwidth=60, v_beamwidth=30)
        try:
            policy_path = get_root_dir() + '/model/baseline_model.pt'
            self.enemy_baseline_policy.load_state_dict(
                torch.load(policy_path, map_location=torch.device('cpu')))
            self.my_lowlevel_policy.load_state_dict(
                torch.load(policy_path, map_location=torch.device('cpu')))
            self.enemy_baseline_policy.eval()
            self.my_lowlevel_policy.eval()
            logging.info("Loaded baseline policies for both sides")
        except Exception as e:
            logging.error(f"Failed to load baseline policies: {e}")
            self.enemy_baseline_policy = None
            self.my_lowlevel_policy = None

        # 三层架构的参数映射
        self.norm_delta_altitude = np.array([-1000, -500, -200, 0, 200, 500, 1000]) / 1000.0
        self.norm_delta_heading = np.array(
            [-np.pi, -np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi])
        self.norm_delta_velocity = np.array([-150, -100, -50, 0, 50, 100, 150]) / 100.0

        # 针对不同战术模板的测试场景配置
        self.test_scenarios = self._init_test_scenarios()

        # 状态管理
        self._inner_rnn_states = {}
        self._enemy_rnn_states = {}
        self._last_shoot_time = {}
        self._remaining_missiles = {}
        self._shoot_action = {}
        self._last_action = {}
        self._maneuver_history = []
        self._target_allocation = {}
        self.tactical_templates = {}
        self.current_phases = {}

        # 测试指标收集
        self.test_metrics = {
            "template_execution_count": 0,
            "maneuver_quality_scores": [],
            "distance_management": [],
            "altitude_management": [],
            "heading_changes": [],
            "tactical_effectiveness": [],
            "enemy_response_data": [],
            "missile_events": [],
            "phase_transitions": [],
            "crank_angle_accuracy": [],  # Crank机动角度精度
            "beam_perpendicular_accuracy": [],  # Beam机动垂直度精度
            "notch_altitude_effectiveness": [],  # Notch高度变化效果
        }

        # 敌方行为状态
        self.enemy_behavior_state = {
            "current_strategy": "baseline_cruise",
            "threat_timer": 0,
            "engagement_phase": "approach",
            "last_missile_launch": -1000,
            "target_heading": 0.0,
            "formation_offset": 0.0
        }

        logging.info(f"Enhanced TacticalTemplateTestTask initialized: test_template={self.test_template_id}")

    def _load_config(self):
        """从原始字符串路径加载并解析配置文件"""
        full_path = os.path.join('E:/Pycharm/LAG/envs/JSBSim/configs', f'{self.config_path}.yaml')
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"配置文件不存在: {full_path}")
        with open(full_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)  # 返回字典格式的配置对象

    def _init_test_scenarios(self):
        """初始化不同战术模板的测试场景"""
        scenarios = {
            # 单机规避机动
            1: {  # Crank
                "name": "Crank_Test",
                "description": "测试30-60度偏航维持雷达锁定",
                "enemy_strategy": "steady_approach",
                "initial_distance": 45000,
                "enemy_heading_offset": 0,
                "missile_threat": True,
                "expected_maneuver": "30-60度偏航，维持锁定"
            },
            2: {  # Beam
                "name": "Beam_Test",
                "description": "测试90度横向机动消耗导弹动能",
                "enemy_strategy": "missile_threat_pursuit",
                "initial_distance": 45000,
                "enemy_heading_offset": 0,
                "missile_threat": True,
                "expected_maneuver": "90度横向，最小化径向速度"
            },
            3: {  # Notch
                "name": "Notch_Test",
                "description": "测试地面杂波遮蔽机动",
                "enemy_strategy": "missile_threat_pursuit",
                "initial_distance": 40000,
                "enemy_heading_offset": 0,
                "missile_threat": True,
                "expected_maneuver": "90度横向+下降高度"
            },
            4: {  # Skate
                "name": "Skate_Test",
                "description": "测试复杂攻击序列",
                "enemy_strategy": "defensive_maneuvering",
                "initial_distance": 50000,
                "enemy_heading_offset": 10,
                "missile_threat": False,
                "expected_maneuver": "发射-机动-转冷-再攻击序列"
            },
            5: {  # Short_Skate
                "name": "Short_Skate_Test",
                "description": "测试发射后快速脱离",
                "enemy_strategy": "defensive_maneuvering",
                "initial_distance": 45000,
                "enemy_heading_offset": 0,
                "missile_threat": False,
                "expected_maneuver": "发射后立即转冷脱离"
            },
            6: {  # Banzai
                "name": "Banzai_Test",
                "description": "测试发射后决策交汇",
                "enemy_strategy": "aggressive_approach",
                "initial_distance": 40000,
                "enemy_heading_offset": 0,
                "missile_threat": False,
                "expected_maneuver": "发射后继续接敌"
            },
            7: {  # Simple_F_Pole
                "name": "Simple_F_Pole_Test",
                "description": "测试保持锁定最大化F-pole",
                "enemy_strategy": "steady_approach",
                "initial_distance": 70000,
                "enemy_heading_offset": 0,
                "missile_threat": False,
                "expected_maneuver": "保持正向锁定"
            },
            8: {  # Advanced_F_Pole
                "name": "Advanced_F_Pole_Test",
                "description": "测试动态F-pole优化",
                "enemy_strategy": "evasive_approach",
                "initial_distance": 65000,
                "enemy_heading_offset": 15,
                "missile_threat": False,
                "expected_maneuver": "动态调整保持最佳F-pole"
            },
            14: {  # Defensive_Sequence
                "name": "Defensive_Sequence_Test",
                "description": "测试连续防御机动序列",
                "enemy_strategy": "multi_threat_attack",
                "initial_distance": 50000,
                "enemy_heading_offset": 0,
                "missile_threat": True,
                "expected_maneuver": "Beam->Notch->Crank->Turn_Cold序列"
            }
        }
        return scenarios

    @property
    def num_agents(self) -> int:
        return 2

    def load_observation_space(self):
        """加载观测空间 - 增强版包含更多战术信息"""
        # 基础观测15 + 战术状态5 = 20
        self.obs_length = 20
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        """加载动作空间 - 第一层：模板选择 + 射击决策"""
        self.action_space = spaces.MultiDiscrete([15, 2])

    def get_obs(self, env, agent_id):
        """获取增强观测 - 包含更多战术相关信息"""
        norm_obs = np.zeros(self.obs_length)

        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return norm_obs

        try:
            # 获取自身状态
            ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
            if np.any(np.isnan(ego_state)):
                ego_state = np.nan_to_num(ego_state, nan=0.0)

            # 计算位置
            ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
            ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])

            # 基础状态观测 (0-8)
            norm_obs[0] = ego_state[2] / 5000  # 高度
            norm_obs[1] = np.sin(ego_state[3])  # roll_sin
            norm_obs[2] = np.cos(ego_state[3])  # roll_cos
            norm_obs[3] = np.sin(ego_state[4])  # pitch_sin
            norm_obs[4] = np.cos(ego_state[4])  # pitch_cos
            norm_obs[5] = ego_state[9] / 340  # v_body_x
            norm_obs[6] = ego_state[10] / 340  # v_body_y
            norm_obs[7] = ego_state[11] / 340  # v_body_z
            norm_obs[8] = ego_state[12] / 340  # vc

            # 敌机相对状态 (9-14)
            enemy_id = "B0100" if agent_id == "A0100" else "A0100"
            enemy = env.agents.get(enemy_id)

            if enemy and enemy.is_alive:
                enemy_state = np.array(enemy.get_property_values(self.state_var))
                if np.any(np.isnan(enemy_state)):
                    enemy_state = np.nan_to_num(enemy_state, nan=0.0)

                cur_ned = LLA2NEU(*enemy_state[:3], env.center_lon, env.center_lat, env.center_alt)
                feature = np.array([*cur_ned, *(enemy_state[6:9])])

                AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)

                norm_obs[9] = (enemy_state[9] - ego_state[9]) / 340
                norm_obs[10] = (enemy_state[2] - ego_state[2]) / 1000
                norm_obs[11] = AO
                norm_obs[12] = TA
                norm_obs[13] = R / 10000
                norm_obs[14] = side_flag
            else:
                norm_obs[9:15] = [0, 0, 0, 0, 10, 0]

            # 战术状态观测 (15-19)
            state_dict = self.get_basic_state_dict(env, agent_id)

            norm_obs[15] = 1 if state_dict.get("radar_lock", False) else 0
            norm_obs[16] = 1 if state_dict.get("has_warning", False) else 0
            norm_obs[17] = self.test_template_id / 14.0  # 当前测试模板
            norm_obs[18] = len([m for m in env.agents[agent_id].under_missiles if m.is_alive]) / 5.0  # 威胁导弹数量
            norm_obs[19] = state_dict.get("enemy_distance", 50000) / 100000  # 归一化距离

        except Exception as e:
            logging.error(f"Error in get_obs for {agent_id}: {e}")
            return np.zeros(self.obs_length)

        norm_obs = np.nan_to_num(norm_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """动作归一化 - 智能的敌我双方行为"""

        # 我方使用完整的三层架构测试指定模板
        if agent_id == "A0100":
            return self._process_my_tactical_action(env, agent_id, action)

        # 敌方使用智能baseline行为配合测试
        elif agent_id == "B0100":
            return self._process_enemy_intelligent_behavior(env, agent_id)

        return np.array([0.0, 0.0, 0.0, 0.7])

    def _process_my_tactical_action(self, env, agent_id, action):
        """处理我方的战术动作 - 强制使用测试模板"""

        # 使用测试模板
        template_id = self.test_template_id
        shoot = action[1] > 0

        self._shoot_action[agent_id] = shoot
        self._last_action[agent_id] = [template_id, shoot]

        # 记录模板执行
        self.test_metrics["template_execution_count"] += 1

        # 生成高层指令
        if agent_id in self.tactical_templates:
            state_dict = self.get_basic_state_dict(env, agent_id)
            tactical_action = self.tactical_templates[agent_id].get_tactical_action(template_id, state_dict)

            # 提取战术指令
            altitude_cmd = tactical_action.get("altitude_cmd", 0)
            heading_cmd = tactical_action.get("heading_cmd", 0)
            velocity_cmd = tactical_action.get("velocity_cmd", 600)

            # 记录机动质量
            self._record_maneuver_quality(template_id, state_dict, tactical_action)

        else:
            altitude_cmd, heading_cmd, velocity_cmd = 0, 0, 600

        # 转换为索引
        altitude_cmd_id =self._convert_altitude_to_index(altitude_cmd)
        heading_cmd_id = self._convert_heading_to_index(heading_cmd)
        velocity_cmd_id = self._convert_velocity_to_index(velocity_cmd - 600)

        # 第三层：使用低级策略网络
        return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

    def _process_enemy_intelligent_behavior(self, env, agent_id):
        """修复后的敌方行为 - 统一使用baseline策略网络"""

        # 获取双方状态
        my_agent = env.agents.get("A0100")
        enemy_agent = env.agents.get("B0100")

        if not (my_agent and enemy_agent and my_agent.is_alive and enemy_agent.is_alive):
            return np.array([0.0, 0.0, 0.0, 0.7])

        distance = np.linalg.norm(my_agent.get_position() - enemy_agent.get_position())

        # 使用baseline策略网络
        enemy_action = self._enemy_baseline_policy_behavior(env, agent_id)

        # 在baseline行为基础上叠加威胁行为
        scenario = self.test_scenarios.get(self.test_template_id, {})
        enemy_strategy = scenario.get("enemy_strategy", "steady_approach")

        # 威胁导弹发射逻辑
        if enemy_strategy in ["steady_approach", "missile_threat_pursuit"]:
            self._handle_enemy_threat_generation(env, agent_id, distance)

        return enemy_action

    def _handle_enemy_threat_generation(self, env, agent_id, distance):
        """处理敌方威胁导弹发射"""
        enemy_agent = env.agents[agent_id]
        my_agent = env.agents["A0100"]

        # 更新威胁计时器
        self.enemy_behavior_state["threat_timer"] += 1

        # 威胁发射条件
        should_launch = (
                distance <= 60000 and  # 60km内
                enemy_agent.num_left_missiles > 0 and
                self.enemy_behavior_state["threat_timer"] > 100 and  # 100步后
                (self.enemy_behavior_state["threat_timer"] -
                 self.enemy_behavior_state.get("last_missile_launch", 0)) > 150  # 间隔150步
        )

        if should_launch:
            try:
                missile_uid = f"{self.enemy_behavior_state['threat_timer']}"
                threat_missile = MissileSimulator.create(
                    parent=enemy_agent,
                    target=my_agent,
                    uid=missile_uid
                )
                env.add_temp_simulator(threat_missile)
                enemy_agent.num_left_missiles -= 1
                self.enemy_behavior_state["last_missile_launch"] = self.enemy_behavior_state["threat_timer"]
                self.test_metrics["missile_events"].append({
                    "step": self.enemy_behavior_state["threat_timer"],
                    "type": "threat_missile_launched",
                    "distance": distance
                })
                logging.info(f"*** THREAT MISSILE LAUNCHED *** Distance: {distance:.0f}m")
            except Exception as e:
                logging.error(f"Failed to launch threat missile: {e}")

    def _enemy_baseline_policy_behavior(self, env, agent_id):
        """修复后的敌方baseline策略 - 使用正确的12维输入"""

        if self.enemy_baseline_policy is None:
            return np.array([0.0, 0.05, 0.0, 0.7])  # 安全备用

        try:
            # 构造12维输入
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)

            # 为敌方生成简单的高层指令
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._generate_enemy_high_level_commands(
                env, agent_id
            )

            # 构造和我方一致的12维输入
            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]  # 只取前9个基础观测

            input_obs = np.nan_to_num(input_obs, nan=0.0)
            input_obs = np.expand_dims(input_obs, axis=0)

            # 确保RNN状态
            if agent_id not in self._enemy_rnn_states:
                self._enemy_rnn_states[agent_id] = torch.zeros(1, 1, 128)

            # 调用baseline策略网络
            with torch.no_grad():
                action, rnn_state = self.enemy_baseline_policy(
                    torch.FloatTensor(input_obs),
                    self._enemy_rnn_states[agent_id]
                )
                action_np = action.numpy().squeeze()
                self._enemy_rnn_states[agent_id] = rnn_state

            # 转换为控制指令
            norm_act = np.zeros(4)
            norm_act[0] = action_np[0] / 20 - 1.  # 副翼
            norm_act[1] = action_np[1] / 20 - 1.  # 升降舵
            norm_act[2] = action_np[2] / 20 - 1.  # 方向舵
            norm_act[3] = action_np[3] / 58 + 0.4  # 油门
            return norm_act

        except Exception as e:
            logging.error(f"Enemy baseline policy error: {e}")
            return np.array([0.0, 0.05, 0.0, 0.7])

    def _generate_enemy_high_level_commands(self, env, agent_id):
        """为敌方生成简单的高层指令"""

        my_agent = env.agents.get("A0100")
        enemy_agent = env.agents.get(agent_id)

        if not (my_agent and enemy_agent and my_agent.is_alive and enemy_agent.is_alive):
            return 3, 4, 3  # 默认指令：保持高度，直飞，正常速度

        # 计算相对位置和距离
        relative_pos = my_agent.get_position() - enemy_agent.get_position()
        distance = np.linalg.norm(relative_pos)

        enemy_pos = enemy_agent.get_position()
        my_pos = my_agent.get_position()

        current_altitude = enemy_pos[2]
        target_altitude = 5000  # 保持初始高度
        # target_altitude = self.initial_enemy_altitude if self.initial_enemy_altitude is not None else 6096.0
        logging.info(f"Enemy {agent_id} altitude: current={current_altitude:.1f}m, target={target_altitude:.1f}m")
        if abs(current_altitude - target_altitude) < 100:
            altitude_cmd_id = 3  # 保持
        elif current_altitude < target_altitude:
            altitude_cmd_id = 4  # 上升
        else:
            altitude_cmd_id = 2  # 下降
        logging.info(
            f"Enemy {agent_id} altitude: current={current_altitude:.1f}m, target={target_altitude:.1f}m, cmd={altitude_cmd_id}")
        # 航向指令：朝向我方
        target_heading = np.arctan2(relative_pos[1], relative_pos[0])
        current_heading = enemy_agent.get_rpy()[2]
        heading_error = target_heading - current_heading

        # 角度归一化
        if heading_error > np.pi:
            heading_error -= 2 * np.pi
        elif heading_error < -np.pi:
            heading_error += 2 * np.pi

        # 根据航向误差选择指令
        if abs(heading_error) < np.pi / 12:  # <15度
            heading_cmd_id = 4  # 直飞
        elif heading_error > 0:
            if heading_error > np.pi / 3:  # >60度
                heading_cmd_id = 7  # 大幅左转
            else:
                heading_cmd_id = 6  # 小幅左转
        else:
            if heading_error < -np.pi / 3:  # <-60度
                heading_cmd_id = 1  # 大幅右转
            else:
                heading_cmd_id = 2  # 小幅右转

        # 速度指令：根据距离调整
        if distance > 70000:
            velocity_cmd_id = 6  # 高速接近
        elif distance > 45000:
            velocity_cmd_id = 5  # 中速接近
        elif distance > 30000:
            velocity_cmd_id = 4  # 正常速度
        else:
            velocity_cmd_id = 3  # 减速

        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id


    def _check_threat_launch_conditions(self, enemy_agent, distance):
        """检查威胁导弹发射条件"""
        return (
                distance <= 60000 and  # 扩大发射距离到60km
                enemy_agent.num_left_missiles > 0 and
                self.enemy_behavior_state["threat_timer"] > 100 and  # 减少等待时间
                (self.enemy_behavior_state["threat_timer"] -
                 self.enemy_behavior_state.get("last_missile_launch", 0)) > 150  # 发射间隔
        )

    def _record_maneuver_quality(self, template_id, state_dict, tactical_action):
        """记录机动质量指标"""

        if template_id == 1:  # Crank
            heading_cmd = tactical_action.get("heading_cmd", 0)
            target_angle = np.deg2rad(45)  # 理想Crank角度
            angle_error = abs(abs(heading_cmd) - target_angle)
            accuracy = max(0, 1 - angle_error / np.deg2rad(30))  # 30度容差
            self.test_metrics["crank_angle_accuracy"].append(accuracy)

        elif template_id == 2:  # Beam
            heading_cmd = tactical_action.get("heading_cmd", 0)
            target_angle = np.deg2rad(90)  # 理想Beam角度
            angle_error = abs(abs(heading_cmd) - target_angle)
            accuracy = max(0, 1 - angle_error / np.deg2rad(20))  # 20度容差
            self.test_metrics["beam_perpendicular_accuracy"].append(accuracy)

        elif template_id == 3:  # Notch
            altitude_cmd = tactical_action.get("altitude_cmd", 0)
            if altitude_cmd < -200:  # 下降超过200米
                effectiveness = min(1.0, abs(altitude_cmd) / 1000)
            else:
                effectiveness = 0.0
            self.test_metrics["notch_altitude_effectiveness"].append(effectiveness)

        # 通用机动质量
        maneuver_intensity = abs(tactical_action.get("heading_cmd", 0)) + abs(
            tactical_action.get("altitude_cmd", 0)) / 1000
        self.test_metrics["maneuver_quality_scores"].append(maneuver_intensity)

    def _convert_altitude_to_index(self, altitude_cmd):
        """高度指令转索引"""
        altitude_values = np.array([-1000, -500, -200, 0, 200, 500, 1000])
        distances = np.abs(altitude_values - altitude_cmd)
        return np.argmin(distances)

    def _convert_heading_to_index(self, heading_cmd):
        """航向指令转索引"""
        heading_cmd = np.clip(heading_cmd, -np.pi, np.pi)
        heading_values = np.array(
            [-np.pi, -np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi])
        distances = np.abs(heading_values - heading_cmd)
        return np.argmin(distances)

    def _convert_velocity_to_index(self, velocity_offset):
        """速度偏移转索引"""
        velocity_values = np.array([-150, -100, -50, 0, 50, 100, 150])
        distances = np.abs(velocity_values - velocity_offset)
        return np.argmin(distances)

    def _use_lowlevel_policy(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """使用低级策略网络"""

        if self.my_lowlevel_policy is None:
            return self._direct_control_mapping(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

        try:
            # 构建输入
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)

            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]

            input_obs = np.nan_to_num(input_obs, nan=0.0)
            input_obs = np.expand_dims(input_obs, axis=0)

            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))

            # 调用策略
            _action, _rnn_states = self.my_lowlevel_policy(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()

            # 转换为控制指令
            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4

            # 安全限制
            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 1000:
                norm_act[1] = max(norm_act[1], 0.0)
                norm_act[3] = max(norm_act[3], 0.8)

            return norm_act

        except Exception as e:
            logging.error(f"Lowlevel policy error: {e}")
            return self._direct_control_mapping(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

    def _direct_control_mapping(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """直接控制映射（备用）"""

        # 获取连续指令
        altitude_cmd = self.norm_delta_altitude[altitude_cmd_id] * 1000
        heading_cmd = self.norm_delta_heading[heading_cmd_id]
        velocity_cmd = self.norm_delta_velocity[velocity_cmd_id] * 100

        # 获取当前状态
        current_alt = env.agents[agent_id].get_position()[2]
        current_vel = np.linalg.norm(env.agents[agent_id].get_velocity())

        # 控制映射
        norm_act = np.zeros(4)

        # 副翼（滚转）
        norm_act[0] = np.clip(heading_cmd * 0.3, -0.4, 0.4)

        # 升降舵（俯仰）
        if current_alt < 500:
            norm_act[1] = 0.3  # 紧急拉起
        else:
            norm_act[1] = np.clip(altitude_cmd / 5000, -0.3, 0.3)

        # 方向舵（偏航）
        norm_act[2] = np.clip(heading_cmd * 0.2, -0.3, 0.3)

        # 油门
        target_velocity = 600 + velocity_cmd
        velocity_error = target_velocity - current_vel
        norm_act[3] = np.clip(0.7 + velocity_error / 1000, 0.4, 0.9)

        return norm_act

    def reset(self, env):
        """重置任务"""
        self.step_count = 0
        self.env = env
        # 重置测试指标
        self.test_metrics = {
            "template_execution_count": 0,
            "maneuver_quality_scores": [],
            "distance_management": [],
            "altitude_management": [],
            "heading_changes": [],
            "tactical_effectiveness": [],
            "enemy_response_data": [],
            "missile_events": [],
            "phase_transitions": [],
            "crank_angle_accuracy": [],
            "beam_perpendicular_accuracy": [],
            "notch_altitude_effectiveness": [],
        }

        # 重置敌方行为状态
        self.enemy_behavior_state = {
            "current_strategy": "baseline_cruise",
            "threat_timer": 0,
            "engagement_phase": "approach",
            "last_missile_launch": -1000,
            "target_heading": 0.0,
            "formation_offset": 0.0
        }

        # 获取智能体
        available_agents = list(env.agents.keys())
        logging.info(
            f"Testing template {self.test_template_id}: {self.test_scenarios.get(self.test_template_id, {}).get('name', 'Unknown')}"
        )
        # 初始化状态
        self._inner_rnn_states = {aid: np.zeros((1, 1, 128)) for aid in available_agents}
        self._enemy_rnn_states = {aid: torch.zeros(1, 1, 128) for aid in available_agents}
        self._last_shoot_time = {aid: -30 for aid in available_agents}
        self._remaining_missiles = {aid: agent.num_missiles for aid, agent in env.agents.items()}
        self._shoot_action = {aid: False for aid in available_agents}
        self._last_action = {aid: [0, 0] for aid in available_agents}
        self._maneuver_history = []
        self._target_allocation = {}
        self.current_phases = {aid: "contact_guidance" for aid in available_agents}

        for agent_id in available_agents:
            if not hasattr(env.agents[agent_id], 'under_missiles'):
                env.agents[agent_id].under_missiles = []
                logging.info(f"Initialized under_missiles for agent {agent_id}")
            elif env.agents[agent_id].under_missiles is None:
                env.agents[agent_id].under_missiles = []
                logging.info(f"Reset under_missiles to empty list for agent {agent_id}")
        # 初始化战术模板（仅我方）
        self.tactical_templates = {}
        if "A0100" in available_agents:
            self.tactical_templates["A0100"] = EnhancedTacticalTemplate(
                is_enemy=False, env=env, agent_id="A0100"
            )
        if "B0100" in env.agents:
            self.initial_enemy_altitude = env.agents["B0100"].get_position()[2]
            logging.info(f"Enemy B0100 initial altitude: {self.initial_enemy_altitude:.1f}m")

        # 设置角色
        for agent_id in available_agents:
            is_leader = agent_id == "A0100"
            env.agents[agent_id].set_leader(is_leader)
            logging.info(f"Agent {agent_id} initial position: {env.agents[agent_id].get_position()}")

        # 重置奖励函数
        for func in self.reward_functions:
            if hasattr(func, 'reset'):
                func.reset(self, env)

        logging.info(
            f"Enhanced TacticalTemplateTestTask reset complete: testing {self.test_scenarios.get(self.test_template_id, {}).get('description', 'Unknown template')}"
        )
        return self

    def get_basic_state_dict(self, env, agent_id):
        """增加威胁检测"""
        if agent_id not in env.agents:
            logging.warning(f"Agent {agent_id} not found in env.agents")
            return {}

        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enemy_id = "B0100" if agent_id == "A0100" else "A0100"
        enemy = env.agents.get(enemy_id)

        enemy_distance = 100000
        enemy_angle_off = 0.0
        enemy_velocity = 340.0
        if enemy and enemy.is_alive:
            distance = np.linalg.norm(enemy.get_position() - env.agents[agent_id].get_position())
            enemy_distance = distance

            ego_pos = env.agents[agent_id].get_position()
            ego_vel = env.agents[agent_id].get_velocity()
            enemy_pos = enemy.get_position()
            enemy_velocity = np.linalg.norm(enemy.get_velocity())
            relative_vec = enemy_pos - ego_pos

            if np.linalg.norm(relative_vec) > 0 and np.linalg.norm(ego_vel) > 0:
                angle = np.arccos(np.clip(
                    np.dot(relative_vec, ego_vel) /
                    (np.linalg.norm(relative_vec) * np.linalg.norm(ego_vel)),
                    -1, 1))
                enemy_angle_off = np.rad2deg(angle)
        missiles_incoming = []
        try:
            if not hasattr(env.agents[agent_id], 'under_missiles') or env.agents[agent_id].under_missiles is None:
                logging.warning(f"Agent {agent_id} has no valid under_missiles attribute, defaulting to empty list")
                missiles_incoming = []
            else:
                missiles_incoming = [m for m in env.agents[agent_id].under_missiles if m.is_alive]
        except Exception as e:
            logging.error(f"Error processing missiles_incoming for {agent_id}: {e}")

        # 计算最近导弹距离
        missile_distance = np.inf
        if missiles_incoming:
            try:
                closest_missile = min(missiles_incoming,
                                      key=lambda m: np.linalg.norm(
                                          m.get_position() - env.agents[agent_id].get_position()))
                missile_distance = np.linalg.norm(closest_missile.get_position() - env.agents[agent_id].get_position())
            except Exception as e:
                logging.error(f"Error calculating missile_distance for {agent_id}: {e}")
                missile_distance = np.inf

        # 构建雷达输入
        radar_state_input = {
            "enemy_distance": enemy_distance if enemy_distance is not None else 100000,
            "enemy_angle_off": enemy_angle_off if enemy_angle_off is not None else 0.0,
            "enemy_velocity": enemy_velocity if enemy_velocity is not None else 340.0,
            "current_altitude": ego_state[2] if ego_state[2] is not None else 5000,
            "missile_distance": missile_distance if missile_distance is not None else np.inf,
            "missiles_incoming": missiles_incoming
        }

        # 日志检查输入
        logging.debug(f"Agent {agent_id} radar_state_input: {radar_state_input}")

        try:
            radar_state = self.radar_model.get_radar_state(radar_state_input, env, agent_id)
        except Exception as e:
            logging.error(f"RadarModel error for {agent_id}: {e}")
            radar_state = self.radar_model._default_radar_state()

        state_dict = {
            "agent_id": agent_id,
            "current_altitude": ego_state[2],
            "enemy_distance": enemy_distance,
            "enemy_angle_off": enemy_angle_off,
            "enemy_velocity": enemy_velocity,
            "missile_distance": missile_distance,
            "radar_lock": radar_state["radar_lock"],
            "has_warning": radar_state["has_warning"],
            "snr": radar_state["snr"],
            "doppler_shift": radar_state["doppler_shift"],
            "ground_clutter": radar_state["ground_clutter"],
            "lock_quality": radar_state["lock_quality"],
            "beam_angle": radar_state["beam_angle"],
            "missile_launched": self._shoot_action.get(agent_id, False),
            "missile_active": bool(env.agents[agent_id].launch_missiles),
            "missile_hit": any(m.is_success for m in env.agents[agent_id].launch_missiles),
            "is_leader": env.agents[agent_id].is_leader(),
            "targets_assigned": bool(self._target_allocation.get(agent_id)),
            "attack_decided": False,
            "shoot_probability": 0.1,
            "current_phase": self.current_phases.get(agent_id, "contact_guidance"),
            "tactical_distances": getattr(self, 'tactical_distances', {}),
            "current_heading": env.agents[agent_id].get_rpy()[2],
            "missiles_incoming": missiles_incoming,
            "threat_level": "HIGH" if missile_distance < 25000 else "MEDIUM" if missile_distance < 40000 else "LOW"
        }

        # 记录雷达状态日志
        if env.current_step % 100 == 0:
            logging.info(f"Agent {agent_id} radar state: lock={radar_state['radar_lock']}, "
                         f"SNR={radar_state['snr']:.1f}dB, doppler={radar_state['doppler_shift']:.1f}m/s, "
                         f"clutter={radar_state['ground_clutter']:.1f}dB")
        return state_dict

    def step(self, env):
        """执行一步"""
        self.step_count += 1

        # 目标分配
        if "A0100" in env.agents and "B0100" in env.agents:
            self._target_allocation["A0100"] = [env.agents["B0100"]]
            self._target_allocation["B0100"] = [env.agents["A0100"]]

        # 收集测试数据
        self._collect_detailed_test_metrics(env)

        # 获取观测
        obs = {}
        for agent_id in env.agents.keys():
            obs[agent_id] = self.get_obs(env, agent_id)

        # 构建共享观测
        all_obs = np.stack([obs[agent_id] for agent_id in sorted(env.agents.keys())], axis=0)
        share_obs = np.tile(all_obs.flatten(), (len(env.agents.keys()), 1))
        share_obs = {agent_id: share_obs[i] for i, agent_id in enumerate(sorted(env.agents.keys()))}

        # 计算奖励
        rewards = {}
        dones = {}
        infos = {}

        for agent_id in env.agents.keys():
            if env.agents[agent_id].is_alive:
                reward_sum = 0.1  # 基础奖励

                # 战术模板执行奖励
                if agent_id == "A0100" and self.test_metrics["template_execution_count"] > 0:
                    reward_sum += 0.5

                # 获取状态字典
                state_dict = self.get_basic_state_dict(env, agent_id)

                # 使用奖励函数
                for reward_func in self.reward_functions:
                    try:
                        # 检查 get_reward 方法是否接受 state_dict
                        from inspect import signature
                        func_sig = signature(reward_func.get_reward)
                        if 'state_dict' in func_sig.parameters:
                            reward_value = reward_func.get_reward(self, env, agent_id, state_dict)
                        else:
                            reward_value = reward_func.get_reward(self, env, agent_id)
                        if isinstance(reward_value, (tuple, list)):
                            reward_value = reward_value[0]
                        reward_sum += reward_value
                    except Exception as e:
                        logging.warning(f"Reward function {reward_func.__class__.__name__} failed: {e}")
            else:
                reward_sum = -10.0

            rewards[agent_id] = np.array([np.clip(reward_sum, -20, 20)])

            done, info = self.get_termination(env, agent_id, {})
            dones[agent_id] = [done]
            if done:
                logging.info(
                    f"Agent {agent_id} marked as done at step {self.step_count}. Reason: {info.get('reason', 'unknown')}")

            infos[agent_id] = {
                "current_phase": self.current_phases.get(agent_id, "contact_guidance"),
                "step_count": self.step_count,
                "test_template_id": self.test_template_id,
                "test_scenario": self.test_scenarios.get(self.test_template_id, {}),
                "test_metrics": self.test_metrics.copy()
            }

        return obs, share_obs, rewards, dones, infos

    def _collect_detailed_test_metrics(self, env):
        my_agent = env.agents.get("A0100")
        enemy_agent = env.agents.get("B0100")
        if not (my_agent and enemy_agent and my_agent.is_alive):
            return
        distance = np.linalg.norm(my_agent.get_position() - enemy_agent.get_position())
        my_altitude = my_agent.get_position()[2]
        my_heading = my_agent.get_rpy()[2]
        self.test_metrics["distance_management"].append(distance)
        self.test_metrics["altitude_management"].append(my_altitude)
        self.test_metrics["heading_changes"].append(my_heading)
        enemy_heading = enemy_agent.get_rpy()[2]
        enemy_altitude = enemy_agent.get_position()[2]
        self.test_metrics["enemy_response_data"].append({
            "step": self.step_count,
            "distance": distance,
            "enemy_heading": enemy_heading,
            "enemy_altitude": enemy_altitude,
            "my_heading": my_heading,
            "my_altitude": my_altitude
        })
        # 添加周期性日志
        if self.step_count % 100 == 0:
            template_name = self.env._get_template_name(self.test_template_id)
            logging.debug(f"Step {self.step_count} - {template_name} Test: Distance={distance:.0f}m")
            if self.test_metrics["maneuver_quality_scores"]:
                quality = self.test_metrics["maneuver_quality_scores"][-1]
                logging.debug(f"Step {self.step_count} - {template_name} Maneuver Quality: {quality:.3f}")

    def get_test_summary(self):
        """获取详细的测试总结"""
        template_names = [
            "No_Template", "Crank", "Beam", "Notch", "Skate", "Short_Skate",
            "Banzai", "Simple_F_Pole", "Advanced_F_Pole", "Pincer",
            "Defensive_Split", "High_Low", "Engaging_Trail", "Loose_Deuce", "Defensive_Sequence"
        ]

        template_name = template_names[self.test_template_id] if self.test_template_id < len(
            template_names) else "Unknown"
        scenario = self.test_scenarios.get(self.test_template_id, {})

        # 计算专项指标
        summary = {
            "template_id": self.test_template_id,
            "template_name": template_name,
            "scenario_name": scenario.get("name", "Unknown"),
            "expected_maneuver": scenario.get("expected_maneuver", "Unknown"),
            "total_steps": self.step_count,
            "template_executions": self.test_metrics["template_execution_count"],

            # 战术效果评估
            "maneuver_quality": {
                "avg_quality_score": np.mean(self.test_metrics["maneuver_quality_scores"]) if self.test_metrics[
                    "maneuver_quality_scores"] else 0,
                "maneuver_consistency": np.std(self.test_metrics["maneuver_quality_scores"]) if len(
                    self.test_metrics["maneuver_quality_scores"]) > 1 else 0
            },

            # 距离管理
            "distance_management": {
                "min_distance": min(self.test_metrics["distance_management"]) if self.test_metrics[
                    "distance_management"] else 0,
                "max_distance": max(self.test_metrics["distance_management"]) if self.test_metrics[
                    "distance_management"] else 0,
                "avg_distance": np.mean(self.test_metrics["distance_management"]) if self.test_metrics[
                    "distance_management"] else 0,
                "distance_variance": np.var(self.test_metrics["distance_management"]) if self.test_metrics[
                    "distance_management"] else 0
            },

            # 高度管理
            "altitude_management": {
                "min_altitude": min(self.test_metrics["altitude_management"]) if self.test_metrics[
                    "altitude_management"] else 0,
                "max_altitude": max(self.test_metrics["altitude_management"]) if self.test_metrics[
                    "altitude_management"] else 0,
                "avg_altitude": np.mean(self.test_metrics["altitude_management"]) if self.test_metrics[
                    "altitude_management"] else 0,
                "altitude_change_rate": np.std(self.test_metrics["altitude_management"]) if len(
                    self.test_metrics["altitude_management"]) > 1 else 0
            },

            # 导弹事件
            "missile_events": {
                "total_missiles": len(self.test_metrics["missile_events"]),
                "missile_details": self.test_metrics["missile_events"]
            },

            # 专项精度指标
            "template_specific_metrics": {},
            "analysis": {
                "overall_score": 0,
                "maneuver_execution": 0,
                "tactical_effectiveness": 0,
                "recommendations": []
            }
        }

        # 计算评分
        maneuver_score = 5.0
        if summary["maneuver_quality"]["avg_quality_score"]:
            maneuver_score = min(10, max(0, summary["maneuver_quality"]["avg_quality_score"] * 10))
        tactical_score = 5.0
        if self.test_template_id == 1 and self.test_metrics["crank_angle_accuracy"]:
            summary["template_specific_metrics"]["crank_angle_accuracy"] = {
                "avg_accuracy": np.mean(self.test_metrics["crank_angle_accuracy"]),
                "min_accuracy": min(self.test_metrics["crank_angle_accuracy"]),
                "max_accuracy": max(self.test_metrics["crank_angle_accuracy"])
            }
            tactical_score = summary["template_specific_metrics"]["crank_angle_accuracy"]["avg_accuracy"] * 10
        elif self.test_template_id == 2 and self.test_metrics["beam_perpendicular_accuracy"]:
            summary["template_specific_metrics"]["beam_perpendicular_accuracy"] = {
                "avg_accuracy": np.mean(self.test_metrics["beam_perpendicular_accuracy"]),
                "min_accuracy": min(self.test_metrics["beam_perpendicular_accuracy"]),
                "max_accuracy": max(self.test_metrics["beam_perpendicular_accuracy"])
            }
            tactical_score = summary["template_specific_metrics"]["beam_perpendicular_accuracy"]["avg_accuracy"] * 10
        elif self.test_template_id == 3 and self.test_metrics["notch_altitude_effectiveness"]:
            summary["template_specific_metrics"]["notch_altitude_effectiveness"] = {
                "avg_effectiveness": np.mean(self.test_metrics["notch_altitude_effectiveness"]),
                "max_descent": max(self.test_metrics["notch_altitude_effectiveness"]) if self.test_metrics[
                    "notch_altitude_effectiveness"] else 0
            }
            tactical_score = summary["template_specific_metrics"]["notch_altitude_effectiveness"][
                                 "avg_effectiveness"] * 10
        overall_score = (maneuver_score + tactical_score) / 2
        recommendations = []
        if maneuver_score < 6:
            recommendations.append("Improve maneuver execution quality")
        if tactical_score < 6:
            recommendations.append("Enhance tactical effectiveness")
        if overall_score > 8:
            recommendations.append("Excellent performance - try more complex scenarios")
        summary["analysis"] = {
            "overall_score": overall_score,
            "maneuver_execution": maneuver_score,
            "tactical_effectiveness": tactical_score,
            "recommendations": recommendations
        }

        return summary