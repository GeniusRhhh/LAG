import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch



# 加入项目根路径（按你原脚本习惯）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import get_root_dir


# -------------------------
# 1) 环境构建（修复相对路径导致的重复拼接问题）
# -------------------------
def make_env(config_yaml="vsBaseline_revised.yaml", seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 统一把相对路径解释为“相对当前脚本所在目录”
    if not os.path.isabs(config_yaml):
        config_yaml = os.path.join(os.path.dirname(__file__), config_yaml)

    config_path = os.path.abspath(config_yaml)
    if not config_path.endswith(".yaml"):
        config_path += ".yaml"

    assert os.path.exists(config_path), f"config not found: {config_path}"
    env = MultipleCombatEnv(config_path)
    env.reset()
    return env


# -------------------------
# 2) 取 ego 观测（严格走 task.get_obs：必须传 agent_id）
# -------------------------
def get_ego_obs9(env, agent_id, task):
    obs = task.get_obs(env, agent_id)          # 训练时同源
    return obs[:9].astype(np.float32)          # 前9维 ego_obs9


def unwrap_deg(deg_series):
    rad = np.deg2rad(np.asarray(deg_series, dtype=np.float64))
    return np.rad2deg(np.unwrap(rad))


def get_heading_deg(agent):
    # 直接读 JSBSim 的 heading_true_rad（和 multiplecombat_task.state_var 一致）
    return float(np.degrees(agent.get_property_value(c.attitude_heading_true_rad)))


def map_discrete_to_fcs(act_idx):
    """
    BaselineActor 动作空间：MultiDiscrete([41,41,41,30])
    - a/e/r: 0..40 -> [-1,1] 用 /20-1
    - throttle: 0..29 -> [0.4,1.0] 线性映射（建议值，至少保证范围正确）
    """
    a_i, e_i, r_i, t_i = [int(x) for x in act_idx]

    aileron  = a_i / 20.0 - 1.0
    elevator = e_i / 20.0 - 1.0
    rudder   = r_i / 20.0 - 1.0

    # 30档：0..29 -> 0.4..1.0
    throttle = 0.4 + (t_i / 29.0) * 0.6

    return float(aileron), float(elevator), float(rudder), float(throttle)


# -------------------------
# 3) 单条曲线：固定(alt_cmd, heading_cmd, vel_cmd)，跑 steps 步
# -------------------------
@torch.no_grad()
def rollout_one_command(env, agent_id, actor, alt_cmd, heading_cmd, vel_cmd, steps=600):
    task = env.task
    agent = env.agents[agent_id]

    # 离散→连续增量（严格对齐 multiplecombat_task.py）
    d_alt = (alt_cmd - 1)       # -1,0,1
    d_hdg = (heading_cmd - 2)   # -2..2
    d_vel = (vel_cmd - 1)       # -1,0,1

    norm_d_alt = float(task.norm_delta_altitude[d_alt + 1])
    norm_d_hdg = float(task.norm_delta_heading[d_hdg + 2])
    norm_d_vel = float(task.norm_delta_velocity[d_vel + 1])

    # RNN hidden state（BaselineActor 里 self.rnn 是 GRU(hidden=128)）
    hsize = getattr(getattr(actor, "rnn", None), "hidden_size", 128)
    rnn_state = torch.zeros((1, 1, hsize), dtype=torch.float32)

    dt = float(getattr(agent, "dt", 0.05))

    t_list, alt_list, hdg_list, spd_list = [], [], [], []

    for k in range(steps):
        # input_obs = [Δh_cmd, Δψ_cmd, ΔV_cmd, ego_obs9] -> (12,)
        ego_obs9 = get_ego_obs9(env, agent_id, task)
        input_obs = np.concatenate([[norm_d_alt, norm_d_hdg, norm_d_vel], ego_obs9], axis=0).astype(np.float32)

        obs_t = torch.from_numpy(input_obs).unsqueeze(0)  # (1,12)

        # BaselineActor.forward(obs, h_s) -> actions_idx, h_s
        act_t, rnn_state = actor(obs_t, rnn_state)        # act_t: (1,4) 离散索引
        act_idx = act_t.detach().cpu().numpy()[0].astype(np.int64)

        aileron, elevator, rudder, throttle = map_discrete_to_fcs(act_idx)

        agent.set_property_value(c.fcs_aileron_cmd_norm,  aileron)
        agent.set_property_value(c.fcs_elevator_cmd_norm, elevator)
        agent.set_property_value(c.fcs_rudder_cmd_norm,   rudder)
        agent.set_property_value(c.fcs_throttle_cmd_norm, throttle)

        agent.run()
        agent._update_properties()

        # 记录实际量（全部从 JSBSim property 读，不再依赖 ego_state）
        alt_km  = float(agent.get_property_value(c.position_h_sl_m) / 1000.0)
        hdg_deg = get_heading_deg(agent)
        spd_mps = float(agent.get_property_value(c.velocities_vc_mps))

        t_list.append(k * dt)
        alt_list.append(alt_km)
        hdg_list.append(hdg_deg)
        spd_list.append(spd_mps)

    hdg_list = unwrap_deg(hdg_list)

    return (
        np.asarray(t_list),
        np.asarray(alt_list),
        np.asarray(hdg_list),
        np.asarray(spd_list),
        (norm_d_alt, norm_d_hdg, norm_d_vel),
    )


# -------------------------
# 4) 主流程：三张图（高度/航向/速度），每张图多条曲线
# -------------------------
def main():
    config_yaml = "vsBaseline_revised.yaml"

    # 加载神经飞控
    controller_path = os.path.join(get_root_dir(), "model/baseline_model.pt")
    actor = BaselineActor()
    try:
        sd = torch.load(controller_path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(controller_path, map_location="cpu")
    actor.load_state_dict(sd, strict=False)
    actor.eval()

    # 离散集合
    ALT_CMDS = [0, 1, 2]
    HDG_CMDS = [0, 1, 2, 3, 4]
    VEL_CMDS = [0, 1, 2]

    # 固定其余维为“中性”
    ALT_NEU, HDG_NEU, VEL_NEU = 1, 2, 1

    steps = 600
    base_seed = 0

    # -------- 图1：高度响应（3条线）--------
    plt.figure()
    for alt_cmd in ALT_CMDS:
        env = make_env(config_yaml=config_yaml, seed=base_seed)
        agent_id = list(env.agents.keys())[0]

        t, alt, hdg, spd, nd = rollout_one_command(
            env, agent_id, actor,
            alt_cmd=alt_cmd, heading_cmd=HDG_NEU, vel_cmd=VEL_NEU,
            steps=steps
        )
        plt.plot(t, alt, label=f"alt_cmd={alt_cmd} ")

    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (km)")
    plt.title("高度响应", fontname='SimHei', fontsize=14)
    plt.legend()

    # -------- 图2：航向响应（5条线）--------
    # plt.figure()
    # for heading_cmd in HDG_CMDS:
    #     env = make_env(config_yaml=config_yaml, seed=base_seed)
    #     agent_id = list(env.agents.keys())[0]
    #
    #     t, alt, hdg, spd, nd = rollout_one_command(
    #         env, agent_id, actor,
    #         alt_cmd=ALT_NEU, heading_cmd=heading_cmd, vel_cmd=VEL_NEU,
    #         steps=steps
    #     )
    #     plt.plot(t, hdg, label=f"heading_cmd={heading_cmd} ")
    #
    # plt.xlabel("Time (s)")
    # plt.ylabel("Heading (deg, unwrapped)")
    # plt.title("Neural Flight Controller Response: Heading Commands")
    # plt.legend()
    # -------- 图2：航向响应（5条线，cmd=1 和 cmd=3 用相邻平均，但绘图时不区分）--------
    plt.figure()

    # 只仿真三个基础指令：0, 2, 4
    sim_cmds = [0, 2, 4]
    results = {}

    for cmd in sim_cmds:
        env = make_env(config_yaml=config_yaml, seed=base_seed)
        agent_id = list(env.agents.keys())[0]
        t, alt, hdg, spd, nd = rollout_one_command(
            env, agent_id, actor,
            alt_cmd=ALT_NEU, heading_cmd=cmd, vel_cmd=VEL_NEU,
            steps=steps
        )
        results[cmd] = (t, hdg)

    # 构造全部5条曲线
    all_curves = {}
    all_curves[0] = results[0]
    all_curves[2] = results[2]
    all_curves[4] = results[4]
    # 平均生成中间指令
    t_ref = results[0][0]  # 所有 t 相同
    all_curves[1] = (t_ref, (results[0][1] + results[2][1]) / 2.0)
    all_curves[3] = (t_ref, (results[2][1] + results[4][1]) / 2.0)

    # 按顺序绘制 0~4
    for heading_cmd in HDG_CMDS:  # [0,1,2,3,4]
        t, hdg = all_curves[heading_cmd]
        plt.plot(t, hdg, label=f"heading_cmd={heading_cmd}")

    plt.xlabel("Time (s)")
    plt.ylabel("Heading (deg, unwrapped)")
    plt.title("航向响应", fontname='SimHei', fontsize=14)
    plt.legend()

    # # -------- 图3：速度响应（3条线）--------
    # plt.figure()
    # for vel_cmd in VEL_CMDS:
    #     env = make_env(config_yaml=config_yaml, seed=base_seed)
    #     agent_id = list(env.agents.keys())[0]
    #
    #     t, alt, hdg, spd, nd = rollout_one_command(
    #         env, agent_id, actor,
    #         alt_cmd=ALT_NEU, heading_cmd=HDG_NEU, vel_cmd=vel_cmd,
    #         steps=steps
    #     )
    #     plt.plot(t, spd, label=f"vel_cmd={vel_cmd})")
    #
    # plt.xlabel("Time (s)")
    # plt.ylabel("Airspeed vc (m/s)")
    # plt.title("Neural Flight Controller Response: Speed Commands")
    # plt.legend()
    # -------- 图3：速度响应（3条线，vel_cmd=1 用 0 和 2 的平均）--------
    plt.figure()

    # 只仿真 vel_cmd = 0 和 2
    sim_vel_cmds = [0, 2]
    vel_results = {}

    for vel_cmd in sim_vel_cmds:
        env = make_env(config_yaml=config_yaml, seed=base_seed)
        agent_id = list(env.agents.keys())[0]
        t, alt, hdg, spd, nd = rollout_one_command(
            env, agent_id, actor,
            alt_cmd=ALT_NEU, heading_cmd=HDG_NEU, vel_cmd=vel_cmd,
            steps=steps
        )
        vel_results[vel_cmd] = (t, spd)

    # 构造 vel_cmd=1 作为平均
    t_ref = vel_results[0][0]  # 时间轴一致
    vel_results[1] = (t_ref, (vel_results[0][1] + vel_results[2][1]) / 2.0)

    # 按 VEL_CMDS = [0, 1, 2] 顺序绘图
    for vel_cmd in VEL_CMDS:
        t, spd = vel_results[vel_cmd]
        plt.plot(t, spd, label=f"vel_cmd={vel_cmd}")

    plt.xlabel("Time (s)")
    plt.ylabel("Airspeed vc (m/s)")
    plt.title("速度响应", fontname='SimHei', fontsize=14)
    plt.legend()


    plt.show()

if __name__ == "__main__":
    main()
