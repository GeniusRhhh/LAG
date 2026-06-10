import os
import numpy as np
import torch

from envs.JSBSim.envs import MultipleCombatEnv
from envs.JSBSim.utils.utils import parse_config
from config import get_config
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.core.radar import RadarModel
from algorithms.ppo.ppo_policy import PPOPolicy

import tactical_templates            # 你已有的单机模板
import tactical_templates_v2         # 你新增的T1/T2/T3/T4与Pincer

from rl_template_driver import step_red_agent_pure_rl, list_templates, max_param_dim, reset_agent_state
from algorithms.hybrid.ppo_policy_hybrid import PPOPolicyHybrid

def safe_load_matching_state_dict(model, ckpt):
    """
    只加载“名字相同且shape完全一致”的权重，其它一概忽略（保持随机初始化）。
    解决：2v2环境(新输入维度/新命名) + 1v1权重(旧维度/旧命名) 的不匹配问题。
    """
    cur = model.state_dict()
    ok = {k: v for k, v in ckpt.items() if (k in cur and v.shape == cur[k].shape)}
    missed = [k for k in cur.keys() if k not in ok]
    extra  = [k for k in ckpt.keys() if k not in ok]
    model.load_state_dict(ok, strict=False)
    print(f"[Actor] partial loaded: matched={len(ok)}  missed={len(missed)}  extra={len(extra)}")

USE_RED_PINCER = True   # 先测红方双机模板
USE_BLUE_RL    = True  # 先关 PPO，确认能动后再开

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # 2v2 配置 + 多机环境
    config_path = os.path.join(project_root, "envs/JSBSim/configs/2v2/ShootMissile/HierarchySelfplay.yaml")
    env = MultipleCombatEnv(config_path)
    env.reset()

    # 取一帧观测(不推进环境，用来拿观测维度)
    boot_obs = env.get_obs()
    aid0 = (env.ego_ids + env.enm_ids)[0]  # 任取一个agent
    obs0_any = boot_obs[aid0]  # 这就是一条观测向量

    red_policy = PPOPolicyHybrid(
        obs_dim=len(obs0_any),
        tpl_num=len(list_templates()),
        z_dim=max_param_dim(),
        device="cpu"
    ).eval()

    # 新增：两侧共享上下文（尤其是 Grinder 要用）
    ctx_A = {"GR": {}}  # 红方
    ctx_B = {"GR": {}}  # 蓝方（一般用不上模板，但留着不出错）

    # （可选）蓝方 PPO，先别开
    if USE_BLUE_RL:
        parser = get_config()
        train_args = parser.parse_args([])
        policy = PPOPolicy(args=train_args,
                           obs_space=env.observation_space,
                           act_space=env.action_space,
                           device=device)
        # 旧 1v1 权重可能不完全匹配，不加载也能跑
        actor_path = os.path.join(project_root,
            "scripts/results/SingleCombat/1v1/ShootMissile/HierarchySelfplay/ppo/shoot_missile_train/run13/actor_latest.pt")
        try:
            sd = torch.load(actor_path, map_location=device)
            # 只要能 load 就行；真不行直接注释掉
            policy.actor.load_state_dict(sd, strict=False)
        except Exception as e:
            print("[Actor] load fail, use random init:", e)
        policy.actor.to(device).eval()
        rnn_states = {}

    radar = RadarModel()
    pincer_ctx = {"initialized": False, "lead": None, "wing": None, "target": None}

    # 关键：确定这一局“任务层会收哪些飞机的动作，且按什么顺序收”
    control_ids = (env.ego_ids + env.enm_ids)[:env.num_agents]
    print("[control_ids used by task]:", control_ids)

    # 清红方 A* 的调度状态（每回合一次）
    for aid in [k for k in env.agents.keys() if k.startswith("A")]:
        reset_agent_state(aid)

    for step in range(3000):
        obs = env.get_obs()
        actions = {}

        # 只给 control_ids 这几架分配动作；别给多了（会被丢）
        for aid in control_ids:
            if aid.startswith("A"):
                # —— RL头：根据 obs[aid] 输出 模板ID 与 连续参数z ——
                tpl_id, z_vec, logp, v = red_policy.act(obs[aid])     # 训练用 act()，评测可用 act_deterministic()
                #print(tpl_id, z_vec)
                # —— 通过“桥接层”把 (模板ID, z) 落到 YAML 参数并执行模板 ——
                actions[aid] = step_red_agent_pure_rl(env, radar, aid, tpl_id, z_vec)
                if (step <= 400):
                    actions[aid] = tactical_templates_v2._dispatch_template("PINCER", env, aid, radar,ctx=ctx_A)
                elif(400 < step <= 600 ):
                    if (aid == "A0100"):
                        actions[aid] = tactical_templates_v2._dispatch_template("T3_DEF", env, aid, radar, ctx=ctx_A)
                    if (aid == "A0200"):
                        actions[aid] = tactical_templates_v2._dispatch_template("T2_LAD", env, aid, radar, ctx=ctx_A)
                elif(600 <step):
                    if (aid == "A0100"):
                        actions[aid] = tactical_templates_v2._dispatch_template("T3_DEF", env, aid, radar, ctx=ctx_A)
                    if (aid == "A0200"):
                        actions[aid] = tactical_templates_v2._dispatch_template("T2_LAD", env, aid, radar, ctx=ctx_A)

            else:   # 蓝方
                if USE_BLUE_RL:
                    ob = torch.as_tensor(obs[aid], dtype=torch.float32, device=device).unsqueeze(0)
                    st = rnn_states.get(aid, torch.zeros(1, 1, 128, device=device))
                    masks = torch.ones(1, 1, device=device)
                    act_tensor, _, st2 = policy.actor(ob, st, masks)
                    rnn_states[aid] = st2
                    act=act_tensor.squeeze(0).detach().cpu().numpy()
                    #actions[aid] = act
                    if step<=200:
                        # 假设环境里“act[-1] > 阈值”才会真正发射导弹，
                        # 这里把它改成明显的“不开火”值（-1 或 0 都可以，一般更保险用 -1）
                        act= [1,2,1,0]
                    if step>=600:
                        act[2]=0
                    actions[aid] = act
                else:
                    # 先用规则，让它确定动起来
                    actions[aid] = [1, 2, 1, 0]  # 保持高度，直行，中速，不开火

        # 按 control_ids 的顺序打包（必须严格这个顺序与长度）
        ordered_actions = [actions[aid] for aid in control_ids]

        # MultipleCombatEnv.step 一般返回 5 个量：obs, share_obs, rewards, dones, info
        ret = env.step(ordered_actions)
        if len(ret) == 5:
            obs, share_obs, rewards, dones, info = ret
        else:
            obs, rewards, dones, info = ret  # 兼容万一返回 4 个

        # 画 ACMI（可关闭）
        env.render(mode='txt', filepath="C:/Users/ww/Desktop/simulation_log.acmi")

        # 收尾判定
        done_all = False
        if isinstance(dones, dict):
            done_all = bool(dones.get("__all__", False))
        elif isinstance(dones, (list, tuple, np.ndarray)):
            done_all = bool(np.all(dones))
        else:
            done_all = bool(dones)

        if done_all:
            # 新局清上下文
            pincer_ctx = {"initialized": False, "lead": None, "wing": None, "target": None}
            if USE_BLUE_RL:
                rnn_states.clear()
            env.reset()
            break  # 或者继续下一局


import unittest
class SmokeTest2v2(unittest.TestCase):
    def test_run_main(self):
        # 不做断言，只为用 unittest 触发一次 main() 跑 2v2
        main()

if __name__ == "__main__":
    import unittest
    unittest.main()