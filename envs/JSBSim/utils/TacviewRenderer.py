import logging
from typing import Dict

import numpy as np
from ..core.simulatior import AircraftSimulator, MissileSimulator
from ..core.catalog import Catalog as c

class TacviewRenderer:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.created = False

    def initialize_acmi(self, filepath):
        """初始化 ACMI 文件。

        Args:
            filepath: 文件路径，必须为有效字符串。
        """
        if not filepath:
            raise ValueError("Filepath must be specified for ACMI initialization")
        with open(filepath, mode='w', encoding='utf-8-sig') as f:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")
            f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
        self.created = True
        logging.info(f"Initialized ACMI file: {filepath}")

    def render(self, env, mode="txt", tacview=None, filepath=None):
        """渲染环境状态。

        Args:
            env: 环境对象，提供模拟器和时间信息。
            mode: 渲染模式，'txt' 或 'real_time'。
            tacview: 实时 Tacview 客户端。
            filepath: 输出文件路径，优先于 self.filepath。
        """
        target_filepath = filepath if filepath else self.filepath
        if not target_filepath:
            raise ValueError("No valid filepath provided for rendering")
        timestamp = env.current_step * env.time_interval
        logging.debug(f"Rendering step {env.current_step}, timestamp={timestamp:.2f}")
        data = [f"#{timestamp:.2f}\n"]
        for agent_id, sim in env._jsbsims.items():
            log_msg = sim.log()
            if log_msg:
                data.append(log_msg + "\n")
            else:
                logging.warning(f"Empty log for agent {agent_id} at step {env.current_step}")
        for sim in env._tempsims.values():
            log_msg = sim.log()
            if log_msg:
                data.append(log_msg + "\n")
        if mode == "txt":
            if not self.created:
                self.initialize_acmi(target_filepath)
            with open(target_filepath, mode='a', encoding='utf-8-sig') as f:
                f.writelines(data)
                f.flush()
            logging.debug(f"Wrote {len(data)} lines to {target_filepath}")
        elif mode == "real_time" and tacview:
            tacview.send_data_to_client("".join(data))

    def generate_radar_cone(self, env, sim: AircraftSimulator, state: Dict) -> str:
        """生成雷达锥数据（保留但不使用）。"""
        radar_state = env.task.tactical_templates[sim.uid].get_radar_state(state)
        if not radar_state["radar_lock"]:
            return ""
        pos = sim.get_position()
        heading = sim.get_property_values([c.attitude_heading_true_rad])[0]
        max_range = 100000
        h_beamwidth = np.deg2rad(60)
        points = []
        for angle in np.linspace(-h_beamwidth/2, h_beamwidth/2, 5):
            dx = max_range * np.cos(heading + angle)
            dy = max_range * np.sin(heading + angle)
            points.append(f"{dx:.1f},{dy:.1f},0")
        return "|".join(points)

    def close(self):
        logging.info("TacviewRenderer closed")