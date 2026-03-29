# -*- coding: utf-8 -*-
import os, yaml, numpy as np
from Tactical_Rule_Template.ruleset import tactical_templates_v2

# === 模板名 <-> YAML 文件 ===
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_NAME2FILE = {
    "T1_LAL":   "T1_LAL.yaml",
    "T2_LAD":   "T2_LAD.yaml",
    "T3_DEF":   "T3_DEF.yaml",
    "T4_DRAG":  "T4_DRAG.yaml",
    "T5_BEAM":  "T5_BEAM.yaml",
    "GRINDER":  "Grinder_2ship.yaml",
    "PINCER":   "Pincer_2ship.yaml",
}

# === 切换护栏参数（可按需调整）===
_MIN_DWELL_DEFAULT = 25     # 默认最小驻留步数（同一模板至少跑这么多 env steps 才允许离开）
_SWITCH_COOLDOWN   = 15     # 切换后冷却期（这段时间内禁止再次切换）
_HYSTERESIS_K      = 2      # 迟滞：需要连续 K 次都提议同一个新模板才认可切换
_MIN_DWELL_BY_TPL  = {      # 针对不同模板的驻留步数（覆盖默认值）
    "GRINDER": 20,
    "PINCER":  20,
    "T1_LAL":  30,
    "T2_LAD":  30,
    "T3_DEF":  20,
    "T4_DRAG": 10,
    "T5_BEAM": 15,
    "FREE":    10,          # 若你加入了无模板选项，可给短一点以保证灵活
}

# 仅保留目录里真实存在的模板（动作空间大小=这个列表长度）
_TEMPLATE_LIST = [k for k, v in _NAME2FILE.items()
                  if os.path.isfile(os.path.join(_TEMPLATE_DIR, v))]
_NAME2IDX = {n: i for i, n in enumerate(_TEMPLATE_LIST)}
_IDX2NAME = {i: n for n, i in _NAME2IDX.items()}

# === 每 team 的共享上下文（双机模板共用；A队/B队各一份）===
_TEAM_CTX = {}  # {"A": {...}, "B": {...}}
def _team_of(agent_id: str) -> str:
    # 约定 agent_id[0] == 'A' or 'B'
    return str(agent_id)[0]
def _get_team_ctx(agent_id: str) -> dict:
    team = _team_of(agent_id)
    return _TEAM_CTX.setdefault(team, {})
# rl_template_driver.py 末尾附近加
def get_current_tpl(agent_id: str):
    st = _SCHED.get(agent_id)
    return None if st is None else st.get("tpl", None)

def list_templates():
    """返回当前目录下能找到 YAML 的模板名列表（按上面的顺序过滤不存在者）"""
    return list(_TEMPLATE_LIST)

def name2idx():
    return dict(_NAME2IDX)

def idx2name():
    return dict(_IDX2NAME)

def _yaml_path(tpl_name: str) -> str:
    f = _NAME2FILE.get(tpl_name)
    if not f:
        raise FileNotFoundError(f"No YAML mapping for template: {tpl_name}")
    p = os.path.join(_TEMPLATE_DIR, f)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Template YAML not found: {p}")
    return p

def _load_param_def(tpl_name: str) -> dict:
    """读取该模板 YAML 的 params 字段，返回 {key: {low, high, default}}"""
    with open(_yaml_path(tpl_name), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("params", {})

def get_param_keys(tpl_name: str):
    """按 YAML 中出现的顺序返回参数名列表"""
    return list(_load_param_def(tpl_name).keys())

# 参数类型归类（整型/布尔）
_INT_KEYS  = {"crank_ticks_max", "short_defend_ticks_max", "cooldown_steps"}
_BOOL_KEYS = {"require_lock_BVR", "debug"}

def _squash_to_range(x: float, lo: float, hi: float) -> float:
    """把任意实数 x 经 tanh 压到 [lo, hi]"""
    y = np.tanh(float(x))             # (-inf, inf) -> (-1, 1)
    return lo + 0.5 * (y + 1.0) * (hi - lo)

def z_to_param_overrides(tpl_name: str, z_vec):
    """
    把连续向量 z 映射成该模板的 param_overrides（遵循 YAML 的 low/high/default）
    - 连续值：tanh 压缩到 [low, high]
    - 整型键：四舍五入并截断
    - 布尔键：映射到[0,1]后 0.5 阈值 -> 0/1
    """
    spec = _load_param_def(tpl_name)
    keys = list(spec.keys())
    z = np.asarray(z_vec if z_vec is not None else [], dtype=float)

    overrides = {}
    for i, k in enumerate(keys):
        bounds = spec[k]
        lo, hi = float(bounds["low"]), float(bounds["high"])
        dflt   = bounds.get("default", lo)

        if i >= len(z):
            val = float(dflt)
        else:
            val = _squash_to_range(z[i], lo, hi)

        if k in _INT_KEYS:
            val = int(min(max(round(val), int(lo)), int(hi)))
        elif k in _BOOL_KEYS:
            p = (val - lo) / (hi - lo + 1e-9)
            val = int(p >= 0.5)

        overrides[k] = val
    return overrides

def reset_team_ctx(team: str = None):
    """回合开始时清团队共享上下文；team=None 则清所有队。"""
    if team is None:
        _TEAM_CTX.clear()
    else:
        _TEAM_CTX.pop(str(team)[0], None)

def max_param_dim():
    """统计所有可用模板的最大参数维度（给动作空间 Box 预留长度）"""
    dims = []
    for n in list_templates():
        try:
            dims.append(len(get_param_keys(n)))
        except:
            dims.append(0)
    return max(dims) if dims else 0

# === 每 agent 的持久上下文（模板内部用；episode 结束前不要清）===
_SCHED = {}  # {agent_id: {"ctx": {...}}}

def reset_agent_state(agent_id: str):
    """回合开始或复位时调用，清空该机的模板上下文，避免相位串线"""
    _SCHED.pop(agent_id, None)

def _apply_template(agent_id, tpl_name, overrides, env, radar):
    """统一调用模板执行器"""
    st = _SCHED.setdefault(agent_id, {"ctx": {}})
    # # ★ 在这里偷偷 alias ★
    # if tpl_name == "PINCER":
    #     tpl_name = "GRINDER"
    return tactical_templates_v2._dispatch_template(
        tpl_name, env, agent_id, radar,
        ctx=st["ctx"], param_overrides=overrides
    )
    # 双机模板共用“队级 ctx”，确保两机看到同一个 GR/PINCER 状态机
    # if tpl_name in ("GRINDER", "PINCER"):
    #     ctx_ref = _get_team_ctx(agent_id)
    # else:
    #     ctx_ref = st["ctx"]
    # return tactical_templates_v2._dispatch_template(
    #     tpl_name, env, agent_id, radar,
    #     ctx=ctx_ref, param_overrides=overrides
    # )

# ===== 方案 A：连续参数（Box，高斯头/PPO 常用）=====
def step_red_agent_pure_rl(env, radar, agent_id: str, tpl_idx: int, z_vec,forced_tpl_name: str=None):
    """
    tpl_idx : RL 输出的模板索引（对应 list_templates() 顺序）
    z_vec   : 连续参数向量（长度不足则用 default）
    return  : [alt_cmd, heading_cmd, vel_cmd, shoot]
    """
    if tpl_idx not in _IDX2NAME:
        tpl_idx = 0
    proposed = _IDX2NAME[tpl_idx]
    # # —— 接入护栏：可能拒绝切换，继续旧模板 ——
    st = _ensure_sched(agent_id)
    chosen, _ = _allow_switch(st, proposed)
    # —— 保持你现有的参数映射与模板执行 ——
    overrides = z_to_param_overrides(chosen, z_vec)  # 你文件里已有该函数
    return _apply_template(agent_id, chosen, overrides, env, radar)  # 你文件里已有该函数
    # 若给了强制模板（来自队内仲裁），直接“硬切”
    # 如果强制切换模板且尚未切换，则强制切换并开始执行
    # if forced_tpl_name is not None and not st.get("force_switched", False):
    #     st["tpl"] = forced_tpl_name
    #     st["steps_on_tpl"] = 0  # 重置步数
    #     st["cooldown"] = 0  # 不设置冷却
    #     st["proposed_hist"].clear()
    #     st["force_switched"] = True  # 标记已经强制切换
    # # 如果当前模板是强制切换的模板，则只要步数未满足护栏条件就继续执行
    # if st.get("force_switched", False):
    #     min_dwell = _MIN_DWELL_BY_TPL.get(st["tpl"], _MIN_DWELL_DEFAULT)
    #     if st["steps_on_tpl"] < min_dwell:
    #         # 如果步数不够，继续执行当前模板
    #         chosen = st["tpl"]
    #         st["steps_on_tpl"] += 1  # 增加步数
    #         # 使用 overrides，而不是直接传递 z_vec
    #         overrides = z_to_param_overrides(chosen, z_vec)
    #         return _apply_template(agent_id, chosen, overrides, env, radar)
    # # 否则进入正常的模板切换判断
    # chosen, _ = _allow_switch(st, proposed)
    # # 在完成强制切换模板后，恢复正常切换逻辑
    # if st.get("force_switched", False):
    #     st["force_switched"] = False
    #
    # overrides = z_to_param_overrides(chosen, z_vec)
    # return _apply_template(agent_id, chosen, overrides, env, radar)

# # ===== 方案 B：离散档位（MultiDiscrete，最省事）=====
# def step_red_agent_pure_rl_bins(env, radar, agent_id: str, action_vec, B: int):
#     """
#     action_vec: [tpl_id, b1, b2, ..., bM]  其中每个 b ∈ [0, B-1]
#     B         : 档位数
#     """
#     if len(action_vec) < 1:
#         raise ValueError("action_vec 至少包含模板ID")
#     tpl_id = int(action_vec[0])
#     if tpl_id not in _IDX2NAME:
#         tpl_id = 0
#     proposed = _IDX2NAME[tpl_id]
#
#     # —— 护栏 ——
#     st = _ensure_sched(agent_id)
#     chosen, _ = _allow_switch(st, proposed)
#
#     # —— 以下保持你原有“档位 -> 参数区间”的映射逻辑 ——
#     spec = _load_param_def(chosen)      # 你文件里已有
#     keys = list(spec.keys())
#     overrides = {}
#     for i, k in enumerate(keys):
#         lo = float(spec[k]["low"]); hi = float(spec[k]["high"])
#         dflt = spec[k].get("default", lo)
#
#         if (i + 1) >= len(action_vec):
#             val = float(dflt); b = None
#         else:
#             b = int(action_vec[i + 1]); b = max(0, min(B - 1, b))
#             val = lo + (b / max(1, B - 1)) * (hi - lo)
#
#         if k in _INT_KEYS:
#             val = int(min(max(round(val), int(lo)), int(hi)))
#         elif k in _BOOL_KEYS:
#             if b is None:
#                 p = (float(dflt) - lo) / (hi - lo + 1e-9)
#                 val = int(p >= 0.5)
#             else:
#                 val = int(b >= (B - 1) // 2)
#
#         overrides[k] = val
#
#     return _apply_template(agent_id, chosen, overrides, env, radar)

# ===（可选）给策略/训练器的动作规格速查 ===
def get_action_spec():
    """
    返回动作空间元信息，构建 Actor 头部/日志时使用：
    {
      "template_names": [...],
      "n_templates":    K,
      "param_count_per_template": {"GRINDER": 8, ...},
      "max_param_dim":  M,
      "name2idx": {...}, "idx2name": {...}
    }
    """
    names = list_templates()
    per_count = {}
    mx = 0
    for n in names:
        try:
            c = len(get_param_keys(n))
        except:
            c = 0
        per_count[n] = c
        mx = max(mx, c)
    return {
        "template_names": names,
        "n_templates": len(names),
        "param_count_per_template": per_count,
        "max_param_dim": mx,
        "name2idx": name2idx(),
        "idx2name": idx2name(),
    }

def _ensure_sched(agent_id: str):
    """
    确保存在该 agent 的调度状态并返回引用。
    字段含义：
      - ctx:             模板执行器上下文（传给 _dispatch_template 的那个 ctx）
      - tpl:             当前实际执行的模板名（护栏可能拒绝 RL 提议，从而维持旧模板）
      - steps_on_tpl:    已在当前模板上连续驻留的步数
      - cooldown:        切换后剩余冷却步数
      - proposed_hist:   最近的提议模板序列（用于迟滞一致性检查）
    """
    st = _SCHED.setdefault(agent_id, {
        "ctx": {},
        "tpl": None,
        "steps_on_tpl": 0,
        "cooldown": 0,
        "proposed_hist": [],
    })
    st["agent_id"] = agent_id
    # 老代码兼容：确保字段存在
    st.setdefault("ctx", {})
    st.setdefault("tpl", None)
    st.setdefault("steps_on_tpl", 0)
    st.setdefault("cooldown", 0)
    st.setdefault("proposed_hist", [])
    return st

def _allow_switch(st, proposed_tpl: str):
    """
    按护栏规则判断是否允许从 st['tpl'] 切到 proposed_tpl。
    返回 (exec_tpl: str, switched: bool)
    """
    # 首次选模板：直接采用
    if st["tpl"] is None:
        st["tpl"] = proposed_tpl
        st["steps_on_tpl"] = 0
        st["cooldown"] = 0
        st["proposed_hist"].clear()
        return st["tpl"], True

    # 驻留步数+冷却倒计时推进
    st["steps_on_tpl"] += 1
    if st["cooldown"] > 0:
        st["cooldown"] -= 1

    # 提议 == 当前，维持不切换，清提议历史
    if proposed_tpl == st["tpl"]:
        st["proposed_hist"].clear()
        return st["tpl"], False

    # 迟滞：连续 K 次都提同一个新模板
    st["proposed_hist"].append(proposed_tpl)
    st["proposed_hist"] = st["proposed_hist"][-_HYSTERESIS_K:]
    hyst_ok = (len(st["proposed_hist"]) == _HYSTERESIS_K
               and all(x == proposed_tpl for x in st["proposed_hist"]))

    # 最小驻留判断（按模板定制覆盖默认）
    min_dwell = _MIN_DWELL_BY_TPL.get(st["tpl"], _MIN_DWELL_DEFAULT)
    dwell_ok = (st["steps_on_tpl"] >= min_dwell)

    # 冷却是否结束
    cd_ok = (st["cooldown"] == 0)

    # aid = st.get("agent_id", "?")  # 取出上面存的 ID
    # if hyst_ok and dwell_ok and cd_ok:
    #     print(f"[{aid}] [SWITCH] {proposed_tpl} ✓  (from={st['tpl']}, dwell={st['steps_on_tpl']}, cd_ok={cd_ok})")
    #     ...
    # else:
    #     print(f"[{aid}] [HOLD] keep={st['tpl']}  propose={proposed_tpl}  "
    #           f"dwell={st['steps_on_tpl']}  cd={st['cooldown']}  "
    #           f"hyst={st['proposed_hist']}")

    if hyst_ok and dwell_ok and cd_ok:
        # 允许切换：重置驻留计数，设置冷却
        st["tpl"] = proposed_tpl
        st["steps_on_tpl"] = 0
        st["cooldown"] = _SWITCH_COOLDOWN
        st["proposed_hist"].clear()
        return st["tpl"], True
    else:
        # 拒绝切换：继续跑旧模板
        return st["tpl"], False

# === 参数维度工具（掩码用） ===
def get_param_dim_by_id(tpl_idx: int) -> int:
    """
    返回该模板需要的连续参数维度 K（按 YAML 中 params 键数）
    """
    if tpl_idx not in _IDX2NAME:
        return 0
    tpl_name = _IDX2NAME[tpl_idx]
    spec = _load_param_def(tpl_name)  # {key:{low,high,default}, ...}
    return len(list(spec.keys()))

def param_mask_by_id(tpl_idx: int, M: int):
    """
    生成一个长度为 M 的 0/1 掩码：前 K 位=1，后面=0
    """
    import numpy as _np
    k = get_param_dim_by_id(tpl_idx)
    m = _np.zeros((M,), dtype=_np.float32)
    m[:k] = 1.0
    return m
