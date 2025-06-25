import numpy as np
import logging
from typing import Dict, Tuple, Any
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir

import numpy as np
import logging
from typing import Dict, Tuple, Any
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir


# 修改 TacticalRewardNew 类
class TacticalRewardNew(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        # 大幅简化奖励结构
        self.phase_rewards = {
            "contact_guidance": 0.0,
            "target_search": 0.01,
            "target_identification": 0.02,
            "threat_assessment": 0.02,
            "target_allocation": 0.03,
            "tactical_decision": 0.03,
            "missile_launch": 0.05,
            "mid_guidance_defense": 0.02,
            "terminal_guidance": 0.02,
            "effect_assessment": 0.03
        }

        # 简化模板奖励 - 只奖励关键模板
        self.template_rewards = {
            1: 0.02,  # Crank
            2: 0.03,  # Beam - 防御重要
            3: 0.05,  # Notch - 最重要规避
            9: 0.04,  # Pincer - 协同重要
            14: 0.03  # Defensive_Sequence
        }

    def get_reward(self, task, env, agent_id, state_dict: Dict[str, Any] = None):
        if not env.agents[agent_id].is_alive:
            return -1.0  # 死亡大幅惩罚

        total_reward = 0.01  # 基础生存奖励

        # 1. 阶段奖励
        current_phase = state_dict.get("current_phase", "contact_guidance")
        total_reward += self.phase_rewards.get(current_phase, 0.0)

        # 2. 简化模板奖励
        if hasattr(task, '_last_action') and agent_id in task._last_action:
            template_id = task._last_action[agent_id][0]
            if template_id in self.template_rewards:
                # 只在适当条件下给奖励
                if self._is_template_effective(template_id, state_dict):
                    total_reward += self.template_rewards[template_id]

        # 3. 距离管理（简化）
        enemy_distance = state_dict.get("enemy_distance", 50000)
        if 30000 <= enemy_distance <= 50000:  # 理想距离
            total_reward += 0.01
        elif enemy_distance < 15000:  # 危险接近
            total_reward -= 0.02

        # 4. 射击质量
        if state_dict.get("missile_launched", False):
            if (state_dict.get("radar_lock", False) and
                    25000 <= enemy_distance <= 40000 and
                    abs(state_dict.get("enemy_angle_off", 0)) < 30):
                total_reward += 0.05  # 好射击
            else:
                total_reward -= 0.03  # 差射击惩罚

        # 严格限制奖励范围
        return np.clip(total_reward, -1.0, 0.1)

    def _is_template_effective(self, template_id: int, state_dict: Dict[str, Any]) -> bool:
        """简化的有效性检查"""
        has_warning = state_dict.get("has_warning", False)
        radar_lock = state_dict.get("radar_lock", False)

        if template_id in [2, 3, 14] and has_warning:  # 防御战术
            return True
        elif template_id == 1 and radar_lock:  # Crank需要锁定
            return True
        elif template_id == 9:  # Pincer协同
            return state_dict.get("is_leader", False)
        return False


class TemplateRewardNew(BaseRewardFunction):
    """模板奖励：专门针对14种战术模板的使用效果评估"""

    def __init__(self, config):
        super().__init__(config)
        self.template_usage_history = {}
        self.template_success_rates = {}

    def get_reward(self, task, env, agent_id):
        """计算模板使用奖励"""
        if not env.agents[agent_id].is_alive:
            return 0.0

        if not hasattr(task, '_last_action') or agent_id not in task._last_action:
            return 0.0

        template_id = task._last_action[agent_id][0]
        if template_id == 0:  # 无模板
            return 0.0

        state_dict = task.get_state_dict(env, agent_id) if hasattr(task, 'get_state_dict') else {}

        # 记录模板使用
        if agent_id not in self.template_usage_history:
            self.template_usage_history[agent_id] = []
        self.template_usage_history[agent_id].append(template_id)

        reward = 0.0

        # 特定模板奖励逻辑
        if template_id == 3 and state_dict.get("has_warning", False):  # Notch规避导弹
            if state_dict.get("current_altitude", 5000) < 2000:
                reward += 2.0  # 成功进入地面杂波
            else:
                reward += 1.0

        elif template_id == 9:  # Pincer钳形机动
            partner_angle = state_dict.get("partner_angle", 0)
            if 35 <= abs(partner_angle) <= 55:  # 理想夹角
                reward += 1.5
                if state_dict.get("enemy_distance", 50000) < state_dict.get("prev_enemy_distance", 60000):
                    reward += 1.0  # 协同成功逼近

        elif template_id == 7 and state_dict.get("enemy_distance", 0) > 40000:  # Simple F-Pole超视距
            if state_dict.get("radar_lock", False):
                reward += 1.2

        elif template_id in [4, 5, 6] and state_dict.get("missile_hit", False):  # 攻击战术命中
            reward += 3.0

        elif template_id == 2:  # Beam机动
            if state_dict.get("has_warning", False):
                missile_distance = state_dict.get("missile_distance", np.inf)
                if missile_distance > 25000:  # 成功拉开距离
                    reward += 1.8

        elif template_id == 14:  # 连续防御机动
            if state_dict.get("has_warning", False):
                if state_dict.get("missile_distance", np.inf) > 35000:
                    reward += 2.5  # 成功防御

        # 模板切换连贯性奖励
        if len(self.template_usage_history[agent_id]) >= 2:
            prev_template = self.template_usage_history[agent_id][-2]
            current_template = template_id

            # 合理的模板切换序列
            good_sequences = [
                (7, 4),  # F-Pole -> Skate
                (1, 2),  # Crank -> Beam
                (2, 3),  # Beam -> Notch
                (6, 1),  # Banzai -> Crank
                (10, 9),  # Defensive_Split -> Pincer
            ]

            if (prev_template, current_template) in good_sequences:
                reward += 0.8

        return self._process(reward, agent_id)


class RadarLockRewardNew(BaseRewardFunction):
    """雷达锁定奖励：基于精细雷达建模的锁定奖励"""

    def __init__(self, config):
        super().__init__(config)
        self.scale = getattr(config, 'radar_lock_reward_scale', 1.0)
        self.lock_duration = {}
        self.lock_quality_history = {}

    def get_reward(self, task, env, agent_id: str, state_dict: Dict[str, Any]) -> Tuple[float, Dict]:
        """计算雷达锁定奖励"""
        radar_lock = state_dict.get("radar_lock", False)
        distance = state_dict.get("enemy_distance", np.inf)

        if agent_id not in self.lock_duration:
            self.lock_duration[agent_id] = 0
        if agent_id not in self.lock_quality_history:
            self.lock_quality_history[agent_id] = []

        reward = 0.0

        if radar_lock:
            self.lock_duration[agent_id] += 1

            # 基础锁定奖励
            base_reward = self.scale * (1 - min(distance / 80000, 0.1))

            # 持续锁定奖励
            duration_bonus = min(self.lock_duration[agent_id] / 30, 0.2)

            # 距离奖励调整
            if 30000 <= distance <= 60000:  # 理想锁定距离
                distance_bonus = 0.15
            elif distance <= 30000:  # 近距离锁定更困难
                distance_bonus = 0.2
            else:
                distance_bonus = 0.1

            # 雷达质量奖励（如果有雷达状态信息）
            radar_quality = 1.0
            if 'snr' in state_dict:
                snr = state_dict['snr']
                if snr > 15:
                    radar_quality = 0.13
                elif snr > 10:
                    radar_quality = 0.11

            reward = base_reward * duration_bonus * distance_bonus * radar_quality

        else:
            self.lock_duration[agent_id] = max(0, self.lock_duration[agent_id] - 2)

        # 记录锁定质量历史
        self.lock_quality_history[agent_id].append(1 if radar_lock else 0)
        if len(self.lock_quality_history[agent_id]) > 50:
            self.lock_quality_history[agent_id].pop(0)

        # 锁定稳定性奖励
        if len(self.lock_quality_history[agent_id]) >= 10:
            stability = np.mean(self.lock_quality_history[agent_id][-10:])
            if stability > 0.8:
                reward *= 1.2

        info = {
            "lock_duration": self.lock_duration[agent_id],
            "lock_stability": np.mean(self.lock_quality_history[agent_id]) if self.lock_quality_history[agent_id] else 0
        }

        return reward, info


class MissileHitRewardNew(BaseRewardFunction):
    """导弹命中奖励：基于导弹性能和战术使用的命中奖励"""

    def __init__(self, config):
        super().__init__(config)
        self.scale = getattr(config, 'missile_hit_reward_scale', 1.0)
        self.launch_conditions = {}
        self.hit_history = {}

    def get_reward(self, task, env, agent_id: str, state_dict: Dict[str, Any]) -> Tuple[float, Dict]:
        """计算导弹命中奖励"""
        missile_hit = state_dict.get("missile_hit", False)
        missile_launched = state_dict.get("missile_launched", False)

        if agent_id not in self.hit_history:
            self.hit_history[agent_id] = {"launches": 0, "hits": 0}

        reward = 0.0

        # 记录发射
        if missile_launched:
            self.hit_history[agent_id]["launches"] += 1
            # 记录发射条件
            self.launch_conditions[f"{agent_id}_{self.hit_history[agent_id]['launches']}"] = {
                "distance": state_dict.get("enemy_distance", 0),
                "radar_lock": state_dict.get("radar_lock", False),
                "enemy_angle": state_dict.get("enemy_angle_off", 0),
                "altitude": state_dict.get("current_altitude", 0)
            }

        # 命中奖励
        if missile_hit:
            self.hit_history[agent_id]["hits"] += 1

            # 基础命中奖励
            base_reward = self.scale

            # 根据发射条件调整奖励
            launch_key = f"{agent_id}_{self.hit_history[agent_id]['hits']}"
            if launch_key in self.launch_conditions:
                conditions = self.launch_conditions[launch_key]

                # 距离奖励系数
                distance = conditions["distance"]
                if distance > 50000:
                    distance_bonus = 2.0  # 远距离命中更困难
                elif distance > 30000:
                    distance_bonus = 1.5
                else:
                    distance_bonus = 1.0

                # 角度奖励系数
                angle = abs(conditions["enemy_angle"])
                if angle > 30:
                    angle_bonus = 1.5  # 大角度命中更困难
                else:
                    angle_bonus = 1.0

                # 锁定质量奖励
                lock_bonus = 1.2 if conditions["radar_lock"] else 0.8

                reward = base_reward * distance_bonus * angle_bonus * lock_bonus
            else:
                reward = base_reward

        # 命中率奖励
        hit_rate_bonus = 0.0
        if self.hit_history[agent_id]["launches"] > 0:
            hit_rate = self.hit_history[agent_id]["hits"] / self.hit_history[agent_id]["launches"]
            if hit_rate > 0.7:
                hit_rate_bonus = 10.0
            elif hit_rate > 0.5:
                hit_rate_bonus = 5.0

        total_reward = reward + hit_rate_bonus

        info = {
            "missile_hit": missile_hit,
            "hit_rate": self.hit_history[agent_id]["hits"] / max(self.hit_history[agent_id]["launches"], 1),
            "total_launches": self.hit_history[agent_id]["launches"],
            "total_hits": self.hit_history[agent_id]["hits"]
        }

        return total_reward, info


class AltitudeRewardNew(BaseRewardFunction):
    """高度奖励：增强版高度管理奖励，支持战术高度选择"""

    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(config, f'{self.__class__.__name__}_safe_altitude', 5.0)
        self.danger_altitude = getattr(config, f'{self.__class__.__name__}_danger_altitude', 1.5)
        self.Kv = getattr(config, f'{self.__class__.__name__}_Kv', 0.2)

        # 战术高度区间
        self.tactical_altitudes = {
            "high_patrol": (12000, 15000, 1.2),  # 高空巡逻
            "medium_combat": (8000, 12000, 1.0),  # 中空作战
            "low_penetration": (3000, 6000, 0.8),  # 低空突防
            "nap_earth": (500, 2000, 1.5),  # 超低空（地面杂波）
        }

    def get_reward(self, task, env, agent_id):
        """计算高度奖励"""
        if not env.agents[agent_id].is_alive:
            return 0.0

        ego_z = env.agents[agent_id].get_position()[-1] / 1000  # km
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340  # normalized vz

        # 基础高度奖励
        Pv = -0.03 * np.clip(ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude, 0.0,
                             1.0) if ego_z <= self.safe_altitude else 0.0
        PH = -0.05 * (1.0 - np.clip(ego_z / self.danger_altitude, 0.0, 1.0)) if ego_z <= self.danger_altitude else 0.0

        # 战术高度奖励
        tactical_bonus = 0.0
        altitude_m = ego_z * 1000

        for zone_name, (min_alt, max_alt, bonus) in self.tactical_altitudes.items():
            if min_alt <= altitude_m <= max_alt:
                # 检查战术背景
                state_dict = task.get_state_dict(env, agent_id) if hasattr(task, 'get_state_dict') else {}

                if zone_name == "nap_earth" and state_dict.get("has_warning", False):
                    tactical_bonus = bonus * 0.5  # 规避时的超低空奖励
                elif zone_name == "high_patrol" and state_dict.get("enemy_distance", 0) > 80000:
                    tactical_bonus = bonus * 0.3  # 远距离巡逻奖励
                elif zone_name == "medium_combat" and 30000 <= state_dict.get("enemy_distance", 0) <= 80000:
                    tactical_bonus = bonus * 0.4  # 作战高度奖励
                break

        # 高度变化率惩罚
        Pvz = -0.005 * abs(ego_vz) if abs(ego_vz) > 0.8 else 0.0

        total_reward = Pv + PH + tactical_bonus + Pvz

        if task.step_count % 200 == 0:
            logging.debug(f"Agent {agent_id} AltitudeReward: total={total_reward:.4f}, "
                          f"Pv={Pv:.4f}, PH={PH:.4f}, tactical={tactical_bonus:.4f}, "
                          f"altitude={altitude_m:.0f}m")

        return self._process(total_reward, agent_id)


class PostureRewardNew(BaseRewardFunction):
    """姿态奖励：增强版姿态管理，支持战术机动评估"""

    def __init__(self, config):
        super().__init__(config)
        self.orientation_version = getattr(config, f'{self.__class__.__name__}_orientation_version', 'v2')
        self.range_version = getattr(config, f'{self.__class__.__name__}_range_version', 'v3')
        self.target_dist = getattr(config, f'{self.__class__.__name__}_target_dist', 55.0)

        # 战术姿态奖励
        self.tactical_postures = {
            "offensive": {"AO_optimal": (0, 30), "range_optimal": (40000, 80000), "bonus": 1.2},
            "defensive": {"AO_optimal": (60, 120), "range_optimal": (80000, 120000), "bonus": 1.0},
            "neutral": {"AO_optimal": (30, 60), "range_optimal": (60000, 100000), "bonus": 0.8}
        }

        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)

    def get_reward(self, task, env, agent_id):
        """计算姿态奖励"""
        if not env.agents[agent_id].is_alive:
            return 0.0

        new_reward = 0
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        for enm in env.agents[agent_id].enemies:
            if not enm.is_alive:
                continue

            enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)

            # 基础姿态奖励
            orientation_reward = self.orientation_fn(AO, TA)
            range_reward = self.range_fn(R / 1000)

            # 战术姿态调整
            state_dict = task.get_state_dict(env, agent_id) if hasattr(task, 'get_state_dict') else {}
            tactical_multiplier = self._get_tactical_posture_multiplier(AO, R, state_dict)

            new_reward += orientation_reward * range_reward * tactical_multiplier

        return self._process(new_reward, agent_id)

    def _get_tactical_posture_multiplier(self, AO: float, R: float, state_dict: Dict[str, Any]) -> float:
        """获取战术姿态乘数"""
        AO_deg = np.rad2deg(AO)

        # 根据当前威胁状况确定理想姿态
        has_warning = state_dict.get("has_warning", False)
        radar_lock = state_dict.get("radar_lock", False)

        if has_warning:
            # 防御姿态：大角度偏离
            posture = "defensive"
        elif radar_lock and R < 50000:
            # 攻击姿态：正向或小角度
            posture = "offensive"
        else:
            # 中性姿态：保持灵活
            posture = "neutral"

        optimal_config = self.tactical_postures[posture]
        AO_min, AO_max = optimal_config["AO_optimal"]
        R_min, R_max = optimal_config["range_optimal"]
        bonus = optimal_config["bonus"]

        # 角度匹配度
        angle_match = 1.0
        if AO_min <= AO_deg <= AO_max:
            angle_match = 1.2
        elif abs(AO_deg - (AO_min + AO_max) / 2) > 45:
            angle_match = 0.8

        # 距离匹配度
        range_match = 1.0
        if R_min <= R <= R_max:
            range_match = 1.1

        return bonus * angle_match * range_match

    def get_orientation_function(self, version):
        """获取方向奖励函数"""
        if version == 'v2':
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                                  + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        else:
            return lambda AO, TA: 1.0

    def get_range_funtion(self, version):
        """获取距离奖励函数"""
        if version == 'v3':
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.05 * R ** 2 + 0.3 * R + 0.4, 0, 1)
        else:
            return lambda R: 1.0


class EventDrivenRewardNew(BaseRewardFunction):
    """简化的事件奖励"""

    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        reward = 0.0
        agent = env.agents[agent_id]

        # 关键事件奖励
        if agent.is_shotdown:
            reward -= 5.0  # 主要负面事件
        elif agent.is_crash:
            reward -= 5.0  # 主要负面事件

        # 导弹命中奖励
        for missile in agent.launch_missiles:
            if missile.is_success:
                reward += 2.0  # 重要正面事件

        # 严格限制范围
        return np.clip(reward, -5.0, 2.0)

class MissilePostureRewardNew(BaseRewardFunction):
    """导弹姿态奖励：增强版导弹规避奖励"""

    def __init__(self, config):
        super().__init__(config)
        self.previous_missile_states = {}

    def reset(self, task, env):
        self.previous_missile_states = {}
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """计算导弹姿态奖励"""
        reward = 0
        agent = env.agents[agent_id]

        # 获取所有威胁导弹
        threatening_missiles = agent.check_missile_warning(multi=True)

        if not threatening_missiles:
            # 清空记录
            self.previous_missile_states[agent_id] = {}
            return 0

        if agent_id not in self.previous_missile_states:
            self.previous_missile_states[agent_id] = {}

        current_states = self.previous_missile_states[agent_id]

        for missile in threatening_missiles:
            missile_id = missile.uid
            current_distance = np.linalg.norm(missile.get_position() - agent.get_position())
            current_velocity = np.linalg.norm(missile.get_velocity())

            if missile_id in current_states:
                prev_distance = current_states[missile_id]["distance"]
                prev_velocity = current_states[missile_id]["velocity"]

                # 距离增加奖励
                distance_change = current_distance - prev_distance
                if distance_change > 0:
                    reward += 0.1 * (distance_change / 1000)  # 每公里0.1奖励

                # 导弹速度降低奖励
                velocity_decrease = prev_velocity - current_velocity
                if velocity_decrease > 0:
                    reward += 0.05 * (velocity_decrease / 100)  # 每100m/s减速0.05奖励

                # 角度规避奖励
                missile_heading = missile.get_velocity()
                agent_heading = agent.get_velocity()

                if (np.linalg.norm(missile_heading) > 0 and
                        np.linalg.norm(agent_heading) > 0):
                    angle = np.dot(missile_heading, agent_heading) / (
                            np.linalg.norm(missile_heading) * np.linalg.norm(agent_heading))

                    # 垂直角度（beam机动）奖励
                    if abs(angle) < 0.3:  # 接近90度
                        reward += 0.2

            # 更新状态
            current_states[missile_id] = {
                "distance": current_distance,
                "velocity": current_velocity
            }

        return self._process(reward, agent_id)


class BasicFlightReward(BaseRewardFunction):
    """基础飞行稳定性奖励"""

    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        if not env.agents[agent_id].is_alive:
            return -1.0

        agent = env.agents[agent_id]
        reward = 0.0

        # 基础存活
        reward += 0.005

        # 速度检查
        current_speed = np.linalg.norm(agent.get_velocity())
        if current_speed < 100:  # 失速风险
            reward -= 0.1
        elif 150 <= current_speed <= 600:  # 正常速度
            reward += 0.005

        # 高度检查
        current_alt = agent.get_position()[2]
        if current_alt < 500:  # 极低空
            reward -= 0.2
        elif 3000 <= current_alt <= 12000:  # 正常高度
            reward += 0.005

        return np.clip(reward, -1.0, 0.1)