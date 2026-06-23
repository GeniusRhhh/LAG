"""Termination conditions for the tactical task."""

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.termination_conditions.termination_condition_base import BaseTerminationCondition


class TacticalTermination(BaseTerminationCondition):
    """战术任务终止条件"""

    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 1500)
        self.max_steps = getattr(config, 'max_steps', 3300)

    def get_termination(self, task, env, agent_id, info={}):
        """终止条件检查"""
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= self.altitude_limit:
            self.log(f"{agent_id} 高度过低: {current_alt:.1f}m")
            return True, False, info

        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            self.log(f"{agent_id} 检测到极端状态")
            return True, False, info

        if (
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0
            or abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0
            or abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0
        ):
            self.log(f"{agent_id} 过载")
            return True, False, info

        red_alive = [aid for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive]
        blue_alive = [aid for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive]

        if len(red_alive) == 0:
            self.log("红方全灭")
            return True, True, {"termination_reason": "red_eliminated", "winner": "blue"}
        if len(blue_alive) == 0:
            self.log("蓝方全灭")
            return True, True, {"termination_reason": "blue_eliminated", "winner": "red"}

        max_time_steps = 3300
        max_steps_limit = getattr(env, 'max_steps', max_time_steps)
        if env.current_step >= max_steps_limit:
            current_time = env.current_step * getattr(env, 'time_interval', 0.2)
            self.log(f"达到最大时间限制: {current_time:.0f}秒 ({env.current_step} 步)")
            return True, False, {"termination_reason": "time_limit", "winner": "draw"}

        return False, False, info
