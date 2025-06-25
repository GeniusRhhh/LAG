import logging
from typing import Dict, Tuple
import numpy as np
from ..core.catalog import Catalog as c
from .reward_function_base import BaseRewardFunction
from ..tasks.TacticalTemplate import TacticalTemplate

class MissileHitReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.scale = getattr(config, 'missile_hit_reward_scale', 30.0)
        logging.info(f"MissileHitReward initialized with scale={self.scale}")

    def get_reward(self, task: any, env: any, agent_id: str, state: Dict[str, any]) -> Tuple[float, Dict]:
        try:
            missile_hit = state.get("missile_hit", False)

            if not isinstance(missile_hit, (bool, np.bool_)):  # 添加 np.bool_
                logging.warning(f"Invalid missile_hit type for {agent_id}: {type(missile_hit)}, using False")
                missile_hit = False

            reward = self.scale * 1.0 if missile_hit else 0.0
            logging.debug(f"Agent {agent_id} MissileHitReward: hit={missile_hit}, reward={reward:.3f}")
            return reward, {"missile_hit": missile_hit}
        except Exception as e:
            logging.error(f"Error in MissileHitReward.get_reward for {agent_id}: {str(e)}, state={state}")
            raise ValueError(f"MissileHitReward failed for {agent_id}: {str(e)}")