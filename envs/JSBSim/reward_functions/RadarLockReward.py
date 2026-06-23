import logging
from typing import Dict, Tuple, Any
import numpy as np
from ..core.catalog import Catalog as c
from .reward_function_base import BaseRewardFunction
from ..tasks.TacticalTemplate import TacticalTemplate

class RadarLockReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.scale = getattr(config, 'radar_lock_reward_scale', 5.0)
        self.lock_duration = 0
        logging.info(f"RadarLockReward initialized with scale={self.scale}")

    def get_reward(self, task: Any, env: Any, agent_id: str, state: Dict[str, Any]) -> Tuple[float, Dict]:
        try:
            radar_lock = state.get("radar_lock", False)
            distance = state.get("enemy_distance", np.inf)

            if not isinstance(radar_lock, (bool, np.bool_)):  # 添加 np.bool_
                logging.warning(f"Invalid radar_lock type for {agent_id}: {type(radar_lock)}, using False")
                radar_lock = False
            if not np.isfinite(distance):
                logging.warning(f"Invalid distance for {agent_id}: {distance}, using np.inf")
                distance = np.inf

            if radar_lock:
                self.lock_duration += 1
                reward = self.scale * (1 - distance / 50000) * min(self.lock_duration / 20, 1.0)
            else:
                self.lock_duration = 0
                reward = 0.0

            logging.debug(f"Agent {agent_id} RadarLockReward: lock={radar_lock}, distance={distance:.1f}m, "
                          f"duration={self.lock_duration}, reward={reward:.3f}")
            return reward, {"lock_duration": self.lock_duration}
        except Exception as e:
            logging.error(f"Error in RadarLockReward.get_reward for {agent_id}: {str(e)}, state={state}")
            raise ValueError(f"RadarLockReward failed for {agent_id}: {str(e)}")