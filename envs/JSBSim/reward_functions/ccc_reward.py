# Tactical_Rule_Template/tscc/ccc_reward.py
from Tactical_Rule_Template.ruleset.rl_template_driver import get_current_tpl

# 配置参数：你也可以单独弄一个 config 文件
DUAL_TEMPLATES = {"GRINDER", "PINCER"}
ALPHA_CC = 0.1   # 可选：协同良好时奖励
BETA_CC  = 0.1   # 协同破裂时惩罚
# 统计量
_CCC_TOTAL   = 0   # 触发计数：g_cc > 0.5 且 pair 非空
_CCC_SUCCESS = 0   # 成功计数：双机同模板那一次

def get_ccc_stats(reset=False):
    """返回 (success, total, rate)，可选重置"""
    global _CCC_TOTAL, _CCC_SUCCESS
    if _CCC_TOTAL > 0:
        rate = _CCC_SUCCESS / float(_CCC_TOTAL)
    else:
        rate = 0.0
    stats = (_CCC_SUCCESS, _CCC_TOTAL, rate)
    if reset:
        _CCC_TOTAL = 0
        _CCC_SUCCESS = 0
    return stats

def _ccc_trigger(env, red_ids):
    """
    内部触发器 g(s_t):
      - 返回 g, pair, info，与之前逻辑相同。
    """
    alive = [aid for aid in red_ids if getattr(env.agents[aid], "is_alive", True)]
    if len(alive) < 2:
        return 0.0, None, {}

    a0, a1 = alive[0], alive[1]
    tpl0 = get_current_tpl(a0) or "None"
    tpl1 = get_current_tpl(a1) or "None"

    d0 = tpl0 in DUAL_TEMPLATES
    d1 = tpl1 in DUAL_TEMPLATES

    info = {"A0_tpl": tpl0, "A1_tpl": tpl1, "d0": d0, "d1": d1}

    # 一双一单
    if d0 ^ d1:
        return 1.0, (a0, a1), info

    # 两机都是双机模板，但模板不一致
    if d0 and d1 and tpl0 != tpl1:
        return 1.0, (a0, a1), info

    return 0.0, (a0, a1), info


def apply_ccc_reward(env, rew_dict, red_ids):
    """
    对当前步的 rew_dict 做 CCC 奖励塑形。
    参数：
      env          : 当前环境
      rew_dict     : {agent_id: r_t}，会在原地被修改
      red_ids      : 红方两机 ID 列表（如 ["A0100","A0200"]）
      use_ccc_rshape : 外部开关（方便在训练脚本里一键关掉）
    返回：
      rew_dict（同一个对象，便于链式调用）
    """
    global _CCC_TOTAL, _CCC_SUCCESS

    g_cc, pair, info = _ccc_trigger(env, red_ids)
    if g_cc <= 0.5 or pair is None:
        return rew_dict

    aid0, aid1 = pair
    tpl0, tpl1 = info["A0_tpl"], info["A1_tpl"]
    d0, d1     = info["d0"], info["d1"]

    _CCC_TOTAL += 1  # 只要进入了“需要协同”的阶段就 +1
    if _CCC_TOTAL <= 60:  # 避免刷屏
        print(f"[CCC-debug] tpl0={tpl0}, tpl1={tpl1}, d0={d0}, d1={d1}")

    extra = 0.0
    # 一双一单 / 双机不同 -> 惩罚
    if d0 ^ d1:
        extra = -BETA_CC
    elif d0 and d1 and tpl0 != tpl1:
        extra = -BETA_CC
    #可选：两机都在同一双机模板时给一点正奖励
    elif d0 and d1 and tpl0 == tpl1:
        extra = +ALPHA_CC
        _CCC_SUCCESS += 1

    if abs(extra) > 1e-8:
        # 情况 1：环境按 agent_id 给奖励（dict）
        if isinstance(rew_dict, dict):
            rew_dict[aid0] += extra
            rew_dict[aid1] += extra

        # 情况 2：训练脚本把红方奖励拆成数组 / list，顺序与 red_ids 对齐
        else:
            # red_ids 里一般是 ["A0100", "A0200"]
            idx0 = red_ids.index(aid0)
            idx1 = red_ids.index(aid1)
            rew_dict[idx0] += extra
            rew_dict[idx1] += extra

    return rew_dict
