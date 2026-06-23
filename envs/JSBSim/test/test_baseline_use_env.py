import logging
from abc import ABC
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Literal
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import in_range_rad, get_root_dir
from envs.JSBSim.envs import SingleCombatEnv
from envs.JSBSim.model.baseline_actor import BaselineActor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class BaselineAgent(ABC):
    def __init__(self, agent_id) -> None:
        self.model_path = get_root_dir() + '/model/baseline_model.pt'
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.actor.eval()
        self.agent_id = agent_id
        self.state_var = [
            c.delta_altitude, c.delta_heading, c.delta_velocities_u,
            c.attitude_roll_rad, c.attitude_pitch_rad, c.velocities_u_mps,
            c.velocities_v_mps, c.velocities_w_mps, c.velocities_vc_mps,
            c.position_h_sl_m
        ]
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    @abstractmethod
    def set_delta_value(self, env, task):
        raise NotImplementedError

    def get_observation(self, env, task, delta_value):
        uid = list(env.agents.keys())[self.agent_id]
        obs = env.agents[uid].get_property_values(self.state_var)
        norm_obs = np.zeros(12)
        norm_obs[0] = delta_value[0] / 1000
        norm_obs[1] = in_range_rad(delta_value[1])
        norm_obs[2] = delta_value[2] / 340
        norm_obs[3] = obs[9] / 5000
        norm_obs[4] = np.sin(obs[3])
        norm_obs[5] = np.cos(obs[3])
        norm_obs[6] = np.sin(obs[4])
        norm_obs[7] = np.cos(obs[4])
        norm_obs[8] = obs[5] / 340
        norm_obs[9] = obs[6] / 340
        norm_obs[10] = obs[7] / 340
        norm_obs[11] = obs[8] / 340
        norm_obs = np.expand_dims(norm_obs, axis=0)
        return norm_obs

    def get_action(self, env, task):
        delta_value = self.set_delta_value(env, task)
        observation = self.get_observation(env, task, delta_value)
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        logging.info(f"Raw action shape: {_action.shape}, Squeezed action shape: {action.shape}")

        # 确保action是一维数组且有4个元素
        if action.ndim == 0:
            action = np.array([action])
        if len(action) != 4:
            logging.error(f"Action length {len(action)} != 4, padding or truncating")
            if len(action) < 4:
                action = np.pad(action, (0, 4 - len(action)), mode='constant', constant_values=20)
            else:
                action = action[:4]

        # 预归一化动作，模拟 normalize_action
        norm_act = np.zeros(4)
        try:
            norm_act[0] = float(action[0]) / 20 - 1.
            norm_act[1] = float(action[1]) / 20 - 1.
            norm_act[2] = float(action[2]) / 20 - 1.
            norm_act[3] = float(action[3]) / 58 + 0.4
        except (ValueError, TypeError) as e:
            logging.error(f"Error normalizing action {action}: {e}")
            norm_act = np.array([0.0, 0.0, 0.0, 0.7])
        norm_act[0] = np.clip(norm_act[0], -1.0, 1.0)
        norm_act[1] = np.clip(norm_act[1], -1.0, 1.0)
        norm_act[2] = np.clip(norm_act[2], -1.0, 1.0)
        norm_act[3] = np.clip(norm_act[3], 0.4, 0.9)

        logging.info(f"Raw action: {action}, Normalized action: {norm_act}")
        return norm_act  # 返回归一化动作
class PursueAgent(BaselineAgent):
    def __init__(self, agent_id) -> None:
        super().__init__(agent_id)

    def set_delta_value(self, env, task):
        ego_uid, enm_uid = list(env.agents.keys())[self.agent_id], list(env.agents.keys())[(self.agent_id + 1) % 2]
        ego_x, ego_y, ego_z = env.agents[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.agents[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.agents[enm_uid].get_position()
        delta_altitude = enm_z - ego_z
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        delta_velocity = env.agents[enm_uid].get_property_value(c.velocities_u_mps) - \
                         env.agents[ego_uid].get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])


class ManeuverAgent(BaselineAgent):
    def __init__(self, agent_id, maneuver: Literal['l', 'r', 'n', 'triangle']) -> None:
        super().__init__(agent_id)
        self.turn_interval = 30
        self.dodge_missile = False
        if maneuver == 'l':
            self.target_heading_list = [0]
        elif maneuver == 'r':
            self.target_heading_list = [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]
        elif maneuver == 'n':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
        elif maneuver == 'triangle':
            self.target_heading_list = [np.pi / 3, np.pi, -np.pi / 3] * 2
        self.target_altitude_list = [6000] * 6
        self.target_velocity_list = [243] * 6

    def reset(self):
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))
        self.init_heading = None

    def set_delta_value(self, env, task):
        step_list = np.arange(1, len(self.target_heading_list) + 1) * self.turn_interval / env.time_interval
        uid = list(env.agents.keys())[self.agent_id]
        cur_heading = env.agents[uid].get_property_value(c.attitude_heading_true_rad)
        if self.init_heading is None:
            self.init_heading = cur_heading

        # 检查导弹威胁（这里需要适配你的环境API）
        has_missile_warning = False
        try:
            # 尝试调用检查导弹威胁的方法
            missile_warning = getattr(env.agents[uid], 'check_missile_warning', lambda: None)()
            has_missile_warning = missile_warning is not None
        except:
            has_missile_warning = False

        if not self.dodge_missile or has_missile_warning:
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            delta_heading = self.init_heading + self.target_heading_list[i] - cur_heading
            delta_altitude = self.target_altitude_list[i] - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = self.target_velocity_list[i] - env.agents[uid].get_property_value(c.velocities_u_mps)
            self.step += 1
        else:
            delta_heading = self.init_heading - cur_heading
            delta_altitude = 6000 - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = 243 - env.agents[uid].get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])


def test_maneuver():
    try:
        env = SingleCombatEnv(config_name='1v1/NoWeapon/test/opposite')
        obs = env.reset()
        logging.info(f"Initial obs shape: {obs.shape}")
        env.render(filepath="control.txt.acmi")

        agent0 = ManeuverAgent(agent_id=0, maneuver='triangle')
        agent1 = PursueAgent(agent_id=1)
        reward_list = []
        agent_ids = list(env.agents.keys())
        step_count = 0
        max_steps = 1000  # 添加最大步数限制防止无限循环

        while step_count < max_steps:
            try:
                action0 = agent0.get_action(env, env.task)
                action1 = agent1.get_action(env, env.task)

                # 确保动作格式正确
                action0 = np.array(action0, dtype=np.float32)
                action1 = np.array(action1, dtype=np.float32)

                logging.info(f"Action0: {action0}, Action1: {action1}")

                # 构建正确的动作格式 - 检查环境期望的动作格式
                if hasattr(env, 'action_space'):
                    logging.info(f"Action space: {env.action_space}")

                # 根据你的代码，似乎需要 (n_rollout_threads, n_agents, action_dim) 格式
                actions = np.array([action0, action1])  # (2, 4)
                actions = actions[np.newaxis, :, :]  # (1, 2, 4)

                logging.info(f"Actions shape: {actions.shape}, Actions: {actions}")

                obs, reward, done, info = env.step(actions)
                env.render(filepath="control.txt.acmi")

                # 处理奖励 - 确保是标量
                if isinstance(reward, (list, np.ndarray)):
                    reward_scalar = reward[0] if len(reward) > 0 else 0.0
                else:
                    reward_scalar = reward

                reward_list.append(reward_scalar)
                step_count += 1

                logging.info(f"Step {step_count}: Obs shape: {obs.shape}, Reward: {reward_scalar}, Done: {done}")

                # 检查终止条件
                if isinstance(done, (list, np.ndarray)):
                    if np.array(done).all():
                        logging.info(f"Episode finished at step {step_count}")
                        print(info)
                        break
                else:
                    if done:
                        logging.info(f"Episode finished at step {step_count}")
                        print(info)
                        break

            except Exception as e:
                logging.error(f"Error in step {step_count}: {e}")
                import traceback
                traceback.print_exc()
                break

        # 绘制奖励曲线
        if reward_list:
            plt.figure(figsize=(10, 6))
            plt.plot(reward_list)
            plt.title('Reward over time')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.savefig('rewards.png')
            plt.show()
            logging.info(f"Saved reward plot with {len(reward_list)} data points")
        else:
            logging.warning("No rewards to plot")

    except Exception as e:
        logging.error(f"Error in test_maneuver: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass


if __name__ == '__main__':
    test_maneuver()