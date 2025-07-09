import logging
import os
from ..core.simulatior import AircraftSimulator, MissileSimulator
from ..core.catalog import Catalog as c

class TacviewRenderer:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.created = False
        self.file = None  # 添加文件句柄

    def initialize_acmi(self, filepath):
        """初始化 ACMI 文件"""
        if not filepath:
            raise ValueError("Filepath must be specified for ACMI initialization")
        self.filepath = os.path.abspath(filepath)
        target_dir = os.path.dirname(self.filepath)
        os.makedirs(target_dir, exist_ok=True)  # 确保目录存在
        self.file = open(self.filepath, mode='w', encoding='utf-8-sig')
        self.file.write("FileType=text/acmi/tacview\n")
        self.file.write("FileVersion=2.1\n")
        self.file.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
        self.file.flush()
        self.created = True
        logging.info(f"Initialized ACMI file: {self.filepath}")

    def render(self, env, mode="txt", tacview=None, filepath=None):
        """渲染环境状态"""
        target_filepath = os.path.abspath(filepath if filepath else self.filepath)
        if not target_filepath:
            raise ValueError("No valid filepath provided for rendering")
        if not self.created:
            self.initialize_acmi(target_filepath)

        timestamp = env.current_step * env.time_interval
        data = [f"#{timestamp:.2f}\n"]
        for agent_id, sim in env._jsbsims.items():
            log_msg = sim.log()
            if log_msg:
                data.append(log_msg + "\n")
                logging.debug(f"Agent {agent_id} log at step {env.current_step}: {log_msg}")
            else:
                logging.warning(f"Empty log for agent {agent_id} at step {env.current_step}")
        for sim in env._tempsims.values():
            log_msg = sim.log()
            if log_msg:
                data.append(log_msg + "\n")
                logging.debug(f"Temp sim {sim.uid} log at step {env.current_step}: {log_msg}")

        if mode == "txt":
            with open(target_filepath, mode='a', encoding='utf-8-sig') as f:
                f.writelines(data)
                f.flush()
            logging.info(f"Wrote {len(data)} lines to {target_filepath} at step {env.current_step}")
        elif mode == "real_time" and tacview:
            tacview.send_data_to_client("".join(data))

    def close(self):
        """关闭文件句柄"""
        if self.file and not self.file.closed:
            self.file.close()
        self.created = False
        logging.info("TacviewRenderer closed")