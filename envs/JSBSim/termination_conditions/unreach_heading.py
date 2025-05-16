import math
import logging
from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition

class UnreachHeading(BaseTerminationCondition):
    def __init__(self, config):
        super().__init__(config)
        uid = list(config.aircraft_configs.keys())[0]
        aircraft_config = config.aircraft_configs[uid]
        self.max_heading_increment = aircraft_config['max_heading_increment']
        self.max_altitude_increment = aircraft_config['max_altitude_increment']
        self.max_velocities_u_increment = aircraft_config['max_velocities_u_increment']
        self.check_interval = aircraft_config['check_interval']
        self.increment_size = [0.5] * 20  # 固定增量 0.5
        self.heading_threshold = 30.0  # 放宽到 30°
        self.consecutive_steps = 50  # 连续 50 步超过阈值才终止

    def get_termination(self, task, env, agent_id, info={}):
        done = False
        success = False
        cur_step = info.get('current_step', 0)
        check_time = env.agents[agent_id].get_property_value(c.heading_check_time)
        if env.agents[agent_id].get_property_value(c.simulation_sim_time_sec) >= check_time:
            delta_heading = math.fabs(env.agents[agent_id].get_property_value(c.delta_heading))
            if delta_heading > self.heading_threshold:
                if 'consecutive_unreach' not in info:
                    info['consecutive_unreach'] = 0
                info['consecutive_unreach'] += 1
                if info['consecutive_unreach'] >= self.consecutive_steps:
                    done = True
            else:
                info['consecutive_unreach'] = 0
                delta = self.increment_size[env.heading_turn_counts % len(self.increment_size)]
                new_heading = env.agents[agent_id].get_property_value(c.target_heading_deg) + delta * self.max_heading_increment
                new_heading = (new_heading + 360) % 360
                new_altitude = env.agents[agent_id].get_property_value(c.target_altitude_ft) + delta * self.max_altitude_increment
                new_altitude = max(new_altitude, 15000)
                new_velocities_u = env.agents[agent_id].get_property_value(c.target_velocities_u_mps) + delta * self.max_velocities_u_increment
                new_velocities_u = max(new_velocities_u, 120.)
                env.agents[agent_id].set_property_value(c.target_heading_deg, new_heading)
                env.agents[agent_id].set_property_value(c.target_altitude_ft, new_altitude)
                env.agents[agent_id].set_property_value(c.target_velocities_u_mps, new_velocities_u)
                env.agents[agent_id].set_property_value(c.heading_check_time, check_time + self.check_interval)
                env.heading_turn_counts += 1
                self.log(f'current_step:{cur_step} target_heading:{new_heading} '
                         f'target_altitude_ft:{new_altitude} target_velocities_u_mps:{new_velocities_u}')
        if done:
            self.log(f'agent[{agent_id}] unreached heading. Total Steps={env.current_step}, '
                     f'delta_heading={delta_heading:.2f}°')
            info['heading_turn_counts'] = env.heading_turn_counts
        success = False
        return done, success, info