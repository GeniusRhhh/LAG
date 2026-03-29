import numpy as np
from Tactical_Rule_Template.ruleset import tactical_templates

# === v2层：传感器快照 + 模板掩码 + 派发 ===
import numpy as np
from Tactical_Rule_Template.ruleset import tactical_templates as tpl

# ---- 1) 传感器快照：只用本机可见的信息 ----
def _sensor_snapshot(env, agent_id, radar, track_ttl_steps=40):
    ego = env.agents[agent_id]
    enemy = ego.enemies[0] if ego.enemies else None

    locked = False
    rel_dist = 1e9
    last_rel_dir = None

    if enemy is not None:
        ego_pos = np.array(ego.get_position())
        ego_vel = np.array(ego.get_velocity())
        roll, pitch, yaw = ego.get_rpy() * 180/np.pi
        tar_pos = np.array(enemy.get_position())
        tar_vel = np.array(enemy.get_velocity())

        locked, _ = radar.detect(
            ego_pos=ego_pos, ego_vel=ego_vel, ego_yaw=yaw, ego_pitch=pitch,
            tar_pos=tar_pos, tar_vel=tar_vel
        )
        if locked:
            # 记录一次“轨迹记忆”，供未锁定时参考（非作弊，全由本机雷达触发）
            rel = tar_pos - ego_pos
            ego._trk_rel = rel
            ego._trk_step = getattr(env, "current_step", 0)
        rel_dist = np.linalg.norm(tar_pos - ego_pos)

    # 轨迹是否可用（最近 TTL 步内）
    trk_ok = hasattr(ego, "_trk_rel") and hasattr(ego, "_trk_step") \
             and (getattr(env, "current_step", 0) - ego._trk_step <= track_ttl_steps)
    if trk_ok:
        last_rel_dir = ego._trk_rel / (np.linalg.norm(ego._trk_rel) + 1e-6)

    incoming = [m for m in getattr(ego, "under_missiles", [])
                if getattr(m, "is_alive", False)]

    ammo_ok = getattr(ego, "num_missiles", 0) > 0
    spd = np.linalg.norm(np.array(ego.get_velocity()))
    energy_ok = (spd > 120.0)  # 简易能量门，按需调整

    return dict(
        locked=locked,
        rel_dist=rel_dist,
        last_rel_dir=last_rel_dir,
        incoming=incoming,
        ammo_ok=ammo_ok,
        energy_ok=energy_ok
    )
def _should_emergency_defend(sensor, MAR=20000.0):
    """最小“硬急停”护栏：只有生存级紧急才短路到 T3_DEF。"""
    return (len(sensor.get("incoming", [])) > 0) or (sensor.get("rel_dist", 1e9) < 0.6 * MAR)

# 约定：模板ID自己定义；这里给 3 个你常用的：T1(Launch-and-Leave=short_skate)、T2(LAD)、T3(防御)
TEMPLATE_ID = {
    "T1_LAL":   0,
    "T2_LAD":   1,
    "T3_DEF":   2,
    "T4_DRAG":  3,
    "T5_BEAM":  4,
    "PINCER":   5,
    "GRINDER":  6,
}

# ---- 3) 无锁定兜底：背离 or 搜索（不改模板）----
def _fallback_when_no_lock(env, agent_id, sensor, cold_angle=150.0):
    """无锁定且无来弹时的兜底几何：
       - 若有最近轨迹：判断大致方位，优先冷转拉开
       - 否则：保持能量（加速）+ 轻微蛇形促成再锁定
    """
    ego = env.agents[agent_id]
    ego_vel = np.array(ego.get_velocity())
    ego_dir = ego_vel / (np.linalg.norm(ego_vel) + 1e-6)

    alt_cmd, heading_cmd, vel_cmd, shoot = 1, 2, 2, 0  # 保高、加速

    if sensor["last_rel_dir"] is not None:
        # 敌大致方位（依据最近一次锁定），判断“是否在后”
        dotv = float(np.clip(np.dot(ego_dir, sensor["last_rel_dir"]), -1.0, 1.0))
        # 敌在右/左（用于选择背离方向）
        cross_z = np.cross(ego_dir, sensor["last_rel_dir"])[2]
        enemy_on_right = (cross_z > 0)
        # 冷转：把相对夹角推大到 ~cold_angle
        heading_cmd = 1 if enemy_on_right else 3
        # 不发射，先拉开
        return [alt_cmd, heading_cmd, vel_cmd, shoot]
    else:
        # 没任何轨迹：小蛇形搜索（轻微左右摆头），避免直飞
        if not hasattr(ego, "_weave_dir"): ego._weave_dir = 1
        # 每 2 秒换向（按仿真步长自己折算，也可直接每步翻转）
        if getattr(env, "current_step", 0) % 40 == 0:
            ego._weave_dir = -ego._weave_dir
        heading_cmd = 1 if ego._weave_dir < 0 else 3
        return [alt_cmd, heading_cmd, vel_cmd, shoot]

# ---- 4) 统一的“根据RL选择/或规则选择 → 调用模板”入口（不改模板内部）----
def run_template_manager(env, agent_id, radar, rl_choice=None, MAR=20000.0, ctx=None):
    """顶层只做两件事：
       1) 生存级紧急（来弹/贴脸）→ 强制 T3_DEF；
       2) 否则无条件执行 RL/上层选择（或用默认）。其它约束交给模板 guards/内部状态机。
    """
    s = _sensor_snapshot(env, agent_id, radar)

    # 1) 硬急停（仅此处可“抢控制权”）
    if _should_emergency_defend(s, MAR=MAR):
        if ctx is not None:
            ctx.setdefault("DECISION_LOG", []).append(
                (getattr(env, "current_step", 0), agent_id, "FORCE_T3_DEF",
                 {"rel_dist": s.get("rel_dist"), "incoming": len(s.get("incoming", []))})
            )
        return _dispatch_template("T3_DEF", env, agent_id, radar, ctx=ctx)

    # 2) 执行 RL/上层的选择；未给则用默认模板（例如 T1_LAL）
    if rl_choice is None:
        name = "T1_LAL"
    else:
        if isinstance(rl_choice, int):
            inv = {v: k for k, v in TEMPLATE_ID.items()}
            name = inv.get(rl_choice, "T1_LAL")
        else:
            name = rl_choice

    if ctx is not None:
        ctx.setdefault("DECISION_LOG", []).append(
            (getattr(env, "current_step", 0), agent_id, "EXECUTE", {"template": name})
        )

    return _dispatch_template(name, env, agent_id, radar, ctx=ctx)


# tactical_templates_v2.py (片段)
from Tactical_Rule_Template.ruleset.template_runtime import run_yaml_step
import os

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

def _dispatch_template(name, env, agent_id, radar, ctx=None, param_overrides=None):
    mapping = {
        "T1_LAL": "T1_LAL.yaml",
        "T2_LAD": "T2_LAD.yaml",
        "T3_DEF": "T3_DEF.yaml",
        "T4_DRAG": "T4_DRAG.yaml",
        "T5_BEAM": "T5_BEAM.yaml",
        "GRINDER": "Grinder_2ship.yaml",
        "PINCER": "Pincer_2ship.yaml",
    }

    # —— 不再回退到旧模块；name 未注册就直接报错，避免静默跑错策略 ——
    if name not in mapping:
        raise KeyError(f"Unknown template '{name}'. Known: {list(mapping.keys())}")

    fpath = os.path.join(TEMPLATE_DIR, mapping[name])
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Template file not found: {fpath}")

    return run_yaml_step(
        fpath, env, agent_id, radar, ctx=ctx,
        action_module="tactical_templates_v2",
        param_overrides=param_overrides or {}
    )



def Launch_and_Leave(env, agent_id, radar,
                     R_fire_min=10000.0,   # 首射最小距离（进入此窗且LOCKED才开火）
                     R_fire_max=20000.0,   # 首射最大距离
                     crank_enable=False,   # 是否在转冷前插入“1~2拍短Crank”横移
                     cold_angle=150.0,     # 判定“已背离”的相对夹角阈值（度）
                     DOR=30000.0,          # 期望脱离距离（Distance of Regret / Recommit判断线）
                     MAR=20000.0           # 最小脱离距离 / 来弹高威胁护栏
                     ):
    """
    T1: 发射-脱离（LAL / Launch-and-Leave / F-Pole / Skate）
    --------------------------------------------------------
    战术意图（论文术语对齐）:
      - guards: 仅当 `LOCKED` 且 敌距 ∈ [R_fire_min, R_fire_max] 才首射；
                若出现强威胁（来弹/距离<MAR），优先插入T3防御一拍（beam/orth/maxG），再回本模板。
      - params: cold_angle / DOR / MAR / (crank_enable)
      - phases:
          Phase 0（首射压制）：满足 guards → shoot=1 → Phase 1
          Phase 1（转冷背离）：将相对夹角推到 cold_angle 以上（左/右按叉积判别）→ Phase 2
          Phase 2（直线冷态脱离）：维持背离航向，直到敌距 ≥ DOR → 退出（交由上层调度）
      - events: LOCKED, 距离阈值（R_fire_min/max, MAR, DOR）, 来弹/under_missiles

    输入/输出:
      输入: env, agent_id, radar（与现有接口一致）
      输出: [alt_cmd, heading_cmd, vel_cmd, shoot]
             alt_cmd: 0上/1保/2下
             heading_cmd: 0左大/1左小/2保/3右小/4右大
             vel_cmd: 0减速/1中速/2加速（按你工程定义）
             shoot: 1发射导弹 / 0不发射

    复用的已有函数:
      - crank_strategy_3d(env, agent_id, radar): 短横移/减闭合（非导引照射）
      - beam_strategy_3d(env, agent_id), ortesc_strategy_3d(env, agent_id), max_g_break_strategy_3d(env, agent_id)
        （T3防御插段：来弹或近威胁时先防一拍，再回本模板继续）

    备注:
      - 本模板是“Short Skate”（不含回头二次接敌）；若需要“完整Skate”可继续用你原有 skate_strategy_3d。
      - 不在函数内部覆盖参数常量（例如不再写死 MAR=20000），全部由签名传入，便于RL/配置统一管理。
    """

    ego = env.agents[agent_id]

    # ========== 0) 初始化本模板内部状态 ==========
    # 继续沿用你现有的字段，避免接口破坏
    if not hasattr(ego, "skate_phase"):
        ego.skate_phase = 0          # 0:首射, 1:转冷, 2:脱离
    if not hasattr(ego, "skate_flag_shot"):
        ego.skate_flag_shot = False  # 是否已发射第一枚导弹

    # 可选: 若启用短Crank，需要一个小计数器（只在 crank_enable=True 时才创建与使用）
    if crank_enable and (not hasattr(ego, "_lal_crank_ticks")):
        ego._lal_crank_ticks = 0     # 仅用于“短Crank 1~2拍”计次；不与原字段重叠

    # ========== 1) 选择最近敌机，计算几何量 ==========
    ego_pos = np.array(ego.get_position())
    ego_vel = np.array(ego.get_velocity())
    ego_speed = np.linalg.norm(ego_vel) + 1e-6
    ego_dir = ego_vel / ego_speed

    best_enemy = None
    min_R = float("inf")
    for enemy in ego.enemies:
        R = np.linalg.norm(np.array(enemy.get_position()) - ego_pos)
        if R < min_R:
            min_R = R
            best_enemy = enemy

    # 无敌目标：保持平飞中速、不射击
    if best_enemy is None:
        return [1, 2, 1, 0]

    tar_pos = np.array(best_enemy.get_position())
    tar_vel = np.array(best_enemy.get_velocity())

    rel_vec = tar_pos - ego_pos
    rel_dist = np.linalg.norm(rel_vec) + 1e-6
    rel_dir = rel_vec / rel_dist

    # 与目标的相对夹角（用于判定是否“已背离”）
    cos_theta = np.clip(np.dot(ego_dir, rel_dir), -1.0, 1.0)
    rel_angle = np.rad2deg(np.arccos(cos_theta))

    # 敌在左/右？（右手系z分量判向）
    cross_z = np.cross(ego_dir, rel_dir)[2]
    # TODO: 若要允许“大转(0/4)”，可在 rel_angle 较小时使用0/4；默认用小转(1/3)

    # ========== 2) 雷达锁定判据（硬guards之一） ==========
    # 你的雷达模型带波束+多普勒门限；这里必须调用 detect 作为“是否可射击”的硬条件。
    roll, pitch, yaw = ego.get_rpy() * 180 / np.pi
    is_locked, _ = radar.detect(
        ego_pos=ego_pos, ego_vel=ego_vel, ego_yaw=yaw, ego_pitch=pitch,
        tar_pos=tar_pos, tar_vel=tar_vel
    )

    # ========== 3) 高优先级护栏：强威胁则先插T3防御一拍（再回本模板） ==========
    # 条件：敌距 < MAR，或来弹追踪（under_missiles存在）
    # 说明：这是“事件驱动回流”的关键：先防一拍，下一步仍走本模板的 phase 流程。
    under_missiles = getattr(ego, "under_missiles", [])
    incoming = [
        m for m in getattr(ego, "under_missiles", [])
        if getattr(m, "is_alive", False)
           and getattr(m, "target_aircraft", None)
           and m.target_aircraft.uid == ego.uid
    ]
    high_threat = (rel_dist < MAR) or (len(incoming) > 0)

    if high_threat:
        # 敌已发射 / 有来弹 / 太近 → 抗弹几何
        if len(incoming) > 0 and rel_dist < 0.6 * MAR:
            # print("max_g_break_strategy")
            return tactical_templates.max_g_break_strategy_3d(env, agent_id)
        else:
            # print("ortesc_strategy")
            return tactical_templates.ortesc_strategy_3d(env, agent_id)

    # ========== 4) 默认动作：保持高度/航向/中速，不射击 ==========
    alt_cmd = 1       # 高度保持
    heading_cmd = 2   # 航向保持
    vel_cmd = 1       # 中速
    shoot = 0
    #print("ego.skate_phase",ego.skate_phase)
    # ========== 5) 相位执行（skate_phase: 0→1→2） ==========
    # Phase 0：首射压制
    if ego.skate_phase == 0:
        in_window = (R_fire_min <= rel_dist <= R_fire_max)
        if is_locked and in_window:
            shoot = 1
            ego.skate_flag_shot = True
            ego.skate_phase = 1  # 进入转冷阶段
        # 未满足射击窗：保持正前飞，等待条件；不做别的几何修正
        return [alt_cmd, heading_cmd, vel_cmd, shoot]

    # Phase 1：转冷背离（可选：短Crank 1~2拍）
    elif ego.skate_phase == 1:
        # （可选）短Crank：只在启用且计数未满时
        if crank_enable and getattr(ego, "_lal_crank_ticks", 0) < 2:
            alt_cmd, heading_cmd, vel_cmd, shoot = tactical_templates.crank_strategy_3d(env, agent_id, radar)
            ego._lal_crank_ticks += 1
            return [alt_cmd, heading_cmd, vel_cmd, shoot]

        if rel_angle < cold_angle:
            # 第一次决定转向：敌在右( cross_z>0 ) → 我向左（左大=0）；敌在左 → 我向右（右大=4）
            if not hasattr(ego, "_lal_turn_dir"):
                ego._lal_turn_dir = 0 if (cross_z > 0) else 4
            heading_cmd = ego._lal_turn_dir
            return [alt_cmd, heading_cmd, vel_cmd, shoot]
        else:
            # 背离达到阈值 → 进入 Phase 2
            ego.skate_phase = 2
            if hasattr(ego, "_lal_turn_dir"):
                delattr(ego, "_lal_turn_dir")
            return [alt_cmd, 2, vel_cmd, shoot]

    # Phase 2：直线冷态脱离直到 DOR
    elif ego.skate_phase == 2:
        # 已背离：保持当前航向（heading_cmd=2），直线延伸
        heading_cmd = 2
        # 脱离距离达标：本模板“完成”，是否重置状态交给上层控制（此处不强制重置，以便上层汇总管理）
        # if rel_dist >= DOR:
        #     ego.skate_phase = 0
        #     ego.skate_flag_shot = False
        #     if hasattr(ego, "_lal_crank_ticks"): delattr(ego, "_lal_crank_ticks")
        return [alt_cmd, heading_cmd, vel_cmd, shoot]

    # 兜底：未知相位 → 保持飞行
    return [alt_cmd, heading_cmd, vel_cmd, shoot]


def Launch_and_Decide(
    env, agent_id, radar,
    # —— 触发与门限（可被RL/配置覆盖）——
    R_fire_min=10000.0,     # 首射最小距离
    R_fire_max=20000.0,     # 首射最大距离
    second_shot_dist=22000.0, # 二次射击触发距离（入窗且LOCKED）
    merge_gate_dist=9000.0, # 并入WVR的阈值（若你暂不做WVR，可只作提示/保持压进）
    MAR=20000.0,            # 来弹/近威胁护栏
    # —— 支援与插段参数 ——
    crank_enable=True,      # 首射后短Crank支援（不转冷）
    crank_ticks_max=2,      # Crank最多插入拍数（“一拍” = 一次 step）
    short_defend_ticks_max=3 # 短时防御插段最长拍数（Notch/Orth/Max-G）
):
    """
    T2: 发射-决断压进（LAD / Launch-and-Decide）
    目标：首射后不转冷，最多做短时防御插段，然后继续压进、寻机二次射击；若逼近到WVR门限则准备并入近距。

    phases（内部状态机，最小三段）：
      _lad_phase=0  首射与支援：满足射击窗&LOCKED -> shoot=1；可插1~2拍 crank 减闭合；转 Phase 1
      _lad_phase=1  短时防御插段：若侦测到来弹/近威胁 -> 调用 ortesc/max_g_break（不再用beam对已来弹）
                     插段不超过 short_defend_ticks_max 拍；结束后转 Phase 2
      _lad_phase=2  压进与二次射击：朝向HOT压进；入 second_shot_dist & LOCKED -> shoot=1
                     若距离 < merge_gate_dist 则继续HOT（并入WVR由上层决定）

    返回: [alt_cmd, heading_cmd, vel_cmd, shoot]
      编码沿用你的约定：heading_cmd: 0左大/1左小/2保持/3右小/4右大；alt_cmd: 0上/1保/2下；vel_cmd: 0减速/1中速/2加速
    """
    ego = env.agents[agent_id]

    # ---------- 初始化 LAD 本地状态 ----------
    if not hasattr(ego, "_lad_phase"):            ego._lad_phase = 0
    if not hasattr(ego, "_lad_flag_shot"):        ego._lad_flag_shot = False  # 是否已首射
    if not hasattr(ego, "_lad_crank_ticks"):      ego._lad_crank_ticks = 0
    if not hasattr(ego, "_lad_defend_ticks"):     ego._lad_defend_ticks = 0

    # ---------- 基本几何 ----------
    ego_pos = np.array(ego.get_position())
    ego_vel = np.array(ego.get_velocity())
    ego_spd = np.linalg.norm(ego_vel) + 1e-6
    ego_dir = ego_vel / ego_spd

    # 选最近敌机
    best_enemy, rel_dist = None, float("inf")
    for enemy in ego.enemies:
        d = np.linalg.norm(np.array(enemy.get_position()) - ego_pos)
        if d < rel_dist:
            rel_dist = d
            best_enemy = enemy
    if best_enemy is None:
        return [1, 2, 1, 0]

    tar_pos = np.array(best_enemy.get_position())
    tar_vel = np.array(best_enemy.get_velocity())
    rel_vec = tar_pos - ego_pos
    rel_dir = rel_vec / (np.linalg.norm(rel_vec) + 1e-6)

    # 相对夹角（我机速度方向与目标方向）
    cos_th = np.clip(np.dot(ego_dir, rel_dir), -1.0, 1.0)
    rel_angle = np.rad2deg(np.arccos(cos_th))
    cross_z = np.cross(ego_dir, rel_dir)[2]     # 敌在右(<0) / 左(>0)

    # ---------- 雷达锁定（硬条件） ----------
    roll, pitch, yaw = ego.get_rpy() * 180 / np.pi
    locked, _ = radar.detect(
        ego_pos=ego_pos, ego_vel=ego_vel, ego_yaw=yaw, ego_pitch=pitch,
        tar_pos=tar_pos, tar_vel=tar_vel
    )

    # ---------- 威胁评估 ----------
    incoming = [
        m for m in getattr(ego, "under_missiles", [])
        if getattr(m, "is_alive", False)
           and getattr(m, "target_aircraft", None)
           and m.target_aircraft.uid == ego.uid
    ]
    high_threat = (rel_dist < MAR) or (len(incoming) > 0)

    # ---------- 默认动作 ----------
    alt_cmd, heading_cmd, vel_cmd, shoot = 1, 2, 1, 0
    # ============ Phase 0：首射与支援 ============
    if ego._lad_phase == 0:
        in_window = (R_fire_min <= rel_dist <= R_fire_max)
        if locked and in_window and not ego._lad_flag_shot:
            shoot = 1
            ego._lad_flag_shot = True
            # 进入支援/插段阶段
            ego._lad_phase = 1
            ego._lad_crank_ticks = 0
            ego._lad_defend_ticks = 0
            return [alt_cmd, heading_cmd, vel_cmd, shoot]

        # 未入窗或未锁定：朝向HOT压进（小转纠正朝向）
        if rel_angle > 5.0:
            heading_cmd = 3 if (cross_z > 0) else 1  # 敌在右->左小转；在左->右小转
        return [alt_cmd, heading_cmd, vel_cmd, shoot]

    # ============ Phase 1：短Crank/短防御插段后回热 ============
    elif ego._lad_phase == 1:
        # 1) 可选短Crank（首射后横移/减闭合），最多 crank_ticks_max 拍
        if crank_enable and ego._lad_crank_ticks < crank_ticks_max:
            alt_cmd, heading_cmd, vel_cmd, _ = tactical_templates.crank_strategy_3d(env, agent_id, radar)
            ego._lad_crank_ticks += 1
            return [alt_cmd, heading_cmd, vel_cmd, shoot]

        # 2) 若出现高威胁：短时防御插段（不使用beam对已来弹；近末段优先Max-G，否则Orth-Esc）
        if high_threat and ego._lad_defend_ticks < short_defend_ticks_max:
            ego._lad_defend_ticks += 1
            if len(incoming) > 0 and rel_dist < 0.6 * MAR:
                return tactical_templates.max_g_break_strategy_3d(env, agent_id)
            else:
                return tactical_templates.ortesc_strategy_3d(env, agent_id)

        # 3) 插段完成 → 转入压进
        ego._lad_phase = 2
        return [alt_cmd, heading_cmd, vel_cmd, shoot]

    # ============ Phase 2：压进与二次射击 ============
    elif ego._lad_phase == 2:
        # 持续HOT压进
        vel_cmd = 2  # 建议加速压进（
        if rel_angle > 3.0:
            heading_cmd = 3 if (cross_z > 0) else 1

        # 若再次进入高威胁，可短暂回到 Phase 1 做1~N拍防御，然后再回到 Phase 2
        if high_threat:
            ego._lad_phase = 1
            ego._lad_defend_ticks = 0
            return [alt_cmd, heading_cmd, vel_cmd, shoot]

        # 二次射击窗口
        in_second = (rel_dist <= second_shot_dist)
        if locked and in_second:
            shoot = 1  # 二次射击
            # 仍保持压进；是否进入WVR由上层或后续模板决定
            return [alt_cmd, heading_cmd, vel_cmd, shoot]
        # 接近WVR门限：保持HOT；真正并入/切模板交给上层（例如切到“WVR”或“继续LAD”）
        # 这里不强制切换，只给出几何意图
        return [alt_cmd, heading_cmd, vel_cmd, shoot]

    # ============ 兜底 ============
    return [alt_cmd, heading_cmd, vel_cmd, shoot]

def ortesc_strategy_3d(env, agent_id, radar):
    """
    垂直机动
    OrtEsc（Orthogonal Escape）机动策略：
    保持自身速度方向垂直于敌方导弹 LOS，最大化横向机动，消耗导弹能量。
    """
    ego = env.agents[agent_id]
    ego_pos = np.array(ego.get_position())
    ego_vel = np.array(ego.get_velocity())
    ego_speed = np.linalg.norm(ego_vel)
    ego_normvec = ego_vel / (ego_speed + 1e-6)

    # 查找所有导向ego的敌方导弹
    threat_missiles = [
        m for m in ego.under_missiles
        if m.is_alive and m.target_aircraft and m.target_aircraft.uid == ego.uid
    ]

    shoot = 0  # 默认不发射
    if len(threat_missiles) == 0:
        return [1, 2, 1,shoot]  # 无导弹威胁，保持平飞中速

    # 获取目标 LOS 方向（共线或不共线）
    if len(threat_missiles) == 1:
        # 单导弹：与 LOS 垂直即可
        los_vec = np.array(threat_missiles[0].get_position()) - ego_pos
        los_unit = los_vec / (np.linalg.norm(los_vec) + 1e-6)
    else:
        # 多导弹（>=2）：构建张成平面
        los1 = np.array(threat_missiles[0].get_position()) - ego_pos
        los2 = np.array(threat_missiles[1].get_position()) - ego_pos
        # 计算每个导弹的距离
        distance1 = np.linalg.norm(los1)
        distance2 = np.linalg.norm(los2)
        # 选择距离最近的导弹
        if distance1 < distance2:
            los_unit = los1 / (np.linalg.norm(los1) + 1e-6)
        else:
            los_unit = los2 / (np.linalg.norm(los2) + 1e-6)

    # 计算速度方向与 los_unit 的夹角
    if ego_speed < 1e-3 or np.linalg.norm(los_unit) < 1e-3:
        return [1, 2, 1]  # 停滞或方向异常，保持当前

    cos_theta = np.clip(np.dot(ego_normvec, los_unit), -1.0, 1.0)
    theta = np.rad2deg(np.arccos(cos_theta))

    # 控制目标夹角为 90°（即速度与 LOS 垂直）
    target_theta = 90.0
    tolerance = 5.0  # 容差范围
    cross = np.cross(ego_normvec, los_unit)
    sign = np.sign(cross[2])  # 判断左右转（右为正，左为负）
    # 判断是朝左还是朝右转以逼近 90 度
    if sign >= 0:
        # 敌人在右侧，目标夹角是 90°
        if theta < target_theta - tolerance:
            heading_cmd = 1
        elif theta > target_theta + tolerance:
            heading_cmd = 3
        else:
            heading_cmd = 2  # 保持
    else:
        # 敌人在左侧，目标夹角是 90°
        if theta < target_theta - tolerance:
            heading_cmd = 3
        elif theta > target_theta + tolerance:
            heading_cmd = 1
        else:
            heading_cmd = 2  # 保持

    alt_cmd = 1  # 保持高度
    vel_cmd = 2  # 加速

    # print(f"[OrtEsc] θ = {theta:.2f}°, heading_cmd = {heading_cmd}")
    return [alt_cmd, heading_cmd, vel_cmd,shoot]

def Drag_and_Recommit(
    env, agent_id, radar,
    cold_angle=150.0,      # 判定“已背离”的相对夹角阈值（度）
    DOR=30000.0,           # 期望脱离距离（到达即认为可重接/退出）
    MAR=20000.0,           # 来弹/近威胁护栏（仅用本机可见）
    auto_recommit=True,   # 到达 DOR 后是否自动回热（否则等待上层/RL切模板）
    hot_angle=30.0,        # 若 auto_recommit=True，回热时将相对夹角收敛到该角度以下
    allow_T3=True          # 脱离过程中允许插入 T3 防御
):
    """
    T4: 战术脱离与重接（Drag/Pump）
    目的：在不利时果断冷态脱离；拉开到 DOR 后再回热（或交由上层切模板）。
    返回: [alt_cmd, heading_cmd, vel_cmd, shoot]
      heading_cmd: 0左大/1左小/2保持/3右小/4右大；alt_cmd: 0上/1保/2下；vel_cmd: 0减速/1中速/2加速
    仅用本机可见信息：敌距/相对角、ego.under_missiles；不猜测敌是否开火。
    """
    #print(f"[DEBUG] Drag called for {agent_id}")
    ego = env.agents[agent_id]

    # --- 初始化本模板局部状态 ---
    if not hasattr(ego, "_pump_phase"):         ego._pump_phase = 0   # 0:转冷 1:延伸 2:DOR达成（可回热/交由上层）
    if not hasattr(ego, "_pump_defend_lock"):   ego._pump_defend_lock = False  # 可选：是否正插入一次T3防御
    if not hasattr(ego, "_pump_shot"):
        ego._pump_shot = False  # 回热阶段是否已打过一发（防重复）

    # --- 基本几何 ---
    ego_pos = np.array(ego.get_position())
    ego_vel = np.array(ego.get_velocity())
    ego_spd = np.linalg.norm(ego_vel) + 1e-6
    ego_dir = ego_vel / ego_spd

    # 选最近敌机
    best_enemy, rel_dist = None, float("inf")
    for enemy in ego.enemies:
        d = np.linalg.norm(np.array(enemy.get_position()) - ego_pos)
        if d < rel_dist:
            rel_dist = d
            best_enemy = enemy
    if best_enemy is None:
        return [1, 2, 1, 0]  # 无敌目标：保持平飞中速

    tar_pos = np.array(best_enemy.get_position())
    rel_vec = tar_pos - ego_pos
    rel_dir = rel_vec / (np.linalg.norm(rel_vec) + 1e-6)

    # 取水平投影并单位化
    ego_xy = ego_dir.copy()
    ego_xy[2] = 0
    ego_xy /= (np.linalg.norm(ego_xy) + 1e-6)
    rel_xy = rel_dir.copy()
    rel_xy[2] = 0
    rel_xy /= (np.linalg.norm(rel_xy) + 1e-6)

    signed = np.degrees(np.arctan2(np.cross(ego_xy, rel_xy)[2], np.dot(ego_xy, rel_xy)))
    rel_angle = abs(signed)

    # --- 仅用本机可见的威胁评估 ---
    incoming = [
        m for m in getattr(ego, "under_missiles", [])
        if getattr(m, "is_alive", False)
           and getattr(m, "target_aircraft", None)
           and m.target_aircraft.uid == ego.uid
    ]
    # 稍放宽距离阈值，避免“高威胁”过度触发
    high_threat = (len(incoming) > 0) or (rel_dist < 0.8 * MAR)

    # --- 默认动作 ---
    alt_cmd, heading_cmd, vel_cmd, shoot = 1, 2, 2, 0   # 脱离阶段建议加速（vel_cmd=0）
    # print("ego._pump_phase",ego._pump_phase)
    # print("rel_angle  cold_angle",rel_angle , cold_angle)
    # print("ego_xy:, rel_xy:",ego_xy, rel_xy)
    # =============== Phase 0：Turn Cold（转冷到足够背离） ===============
    if ego._pump_phase == 0:
        # 插入防御（仅在确有来弹时，且允许）
        # if allow_T3 and len(incoming) > 0:
        #     # print("处理威胁")
        #     # 末段近威胁优先 Max-G，否则 OrtEsc
        #     if rel_dist < 0.6 * MAR:
        #         return tactical_templates.max_g_break_strategy_3d(env, agent_id)
        #     else:
        #         return tactical_templates.ortesc_strategy_3d(env, agent_id)

        if rel_angle < cold_angle:
            # print("转冷")
            #左负右正
            heading_cmd = 1 if np.cross(ego_xy, rel_xy)[2] > 0 else 3  # 小幅持续转冷
            return [alt_cmd, heading_cmd, vel_cmd, shoot]
        else:
            # 已背离到阈值，进入延伸
            ego._pump_phase = 1
            return [alt_cmd, heading_cmd, vel_cmd, shoot]

    # =============== Phase 1：Extend Cold（冷态高速延伸到 DOR） ===============
    elif ego._pump_phase == 1:
        # 背离延伸：保持当前航向直线拉开
        heading_cmd = 2
        vel_cmd = 0  # 建议全加力延伸

        # 必要时插入一次 T3 防御（只用本机可见）
        if allow_T3 and len(incoming) > 0:
            if rel_dist < 0.6 * MAR:
                return tactical_templates.max_g_break_strategy_3d(env, agent_id)
            else:
                return tactical_templates.ortesc_strategy_3d(env, agent_id)

        # 达到 DOR：可重接/退出
        if rel_dist >= DOR:
            ego._pump_phase = 2
            # —— 关键：清发射相关一次性标记，保证重接后能再次射击 ——
            if hasattr(ego, "skate_flag_shot"):
                ego.skate_flag_shot = False
            if hasattr(ego, "_fired_once"):
                ego._fired_once = False
            # 可选：告诉上层“已达DOR可重接”
            ego._ready_recommit = True
            ego._pump_shot = False  # 刚达 DOR，准备回热时允许“T4内回热即开火”再触发一次
        return [alt_cmd, heading_cmd, vel_cmd, shoot]

    # =============== Phase 2：Recommit / 交由上层 ===============
    elif ego._pump_phase == 2:
        # 默认：保持冷态飞行，由上层（RL）决定切回 LAL/LAD/T3 或退出战区
        heading_cmd = 2
        vel_cmd = 1  # 进入评估期可降到中速，节油/恢复能量

        # 可选：自动回热（若你希望 T4 自行回到 HOT 态势）
        if auto_recommit:
            # 反向把 rel_angle 收到 hot_angle 以下（HOT）
            if rel_angle > hot_angle:
                # print("转热")
                # 敌在右→向右转热；在左→向左转热（与 Phase0 相反）
                heading_cmd = 3 if np.cross(ego_xy, rel_xy)[2] >0 else 1
                vel_cmd = 0
            else:
                heading_cmd = 2  # 已基本 HOT，可交由上层切 LAL/LAD

        # print("auto_recommit",auto_recommit)
        # print("(rel_angle <= hot_angle)",(rel_angle <= hot_angle))
        # print("not ego._pump_shot",not ego._pump_shot)

            # —— 回热达标就开火（一次性）——
        if auto_recommit and (rel_angle <= hot_angle) and not ego._pump_shot:
            shoot = 1
            ego._pump_shot = True
        return [alt_cmd, heading_cmd, vel_cmd, shoot]

    # 兜底
    return [alt_cmd, heading_cmd, vel_cmd, shoot]


def beam_strategy_3d(env, agent_id, radar):
    ego = env.agents[agent_id]
    # === 早退：若已有来弹，Beam 不再适用，直接转垂直规避 ===
    incoming = [m for m in getattr(ego, "under_missiles", [])
                if getattr(m, "is_alive", False)
                and getattr(m, "target_aircraft", None)
                and m.target_aircraft.uid == ego.uid]
    if len(incoming) > 0:
        # 这里不要横向 Beaming 了，直接用你已验证过的垂直规避
        return tactical_templates.ortesc_strategy_3d(env, agent_id)

    # 获取自身位置/速度
    ego_pos = np.array(ego.get_position())
    ego_vel = np.array(ego.get_velocity())
    ego_heading = ego_vel / (np.linalg.norm(ego_vel) + 1e-6)

    # 寻找最近的敌人
    best_enemy = None
    min_R = float("inf")
    for enemy in ego.enemies:
        R = np.linalg.norm(np.array(enemy.get_position()) - ego_pos)
        if R < min_R:
            min_R = R
            best_enemy = enemy

    # for enemy in ego.enemies:
    #     if not enemy.is_alive:
    #         print("被击毁")

    shoot = 0  # 默认不发射
    if best_enemy is None:
        return [1, 2, 1,shoot]  # 没有敌人，保持当前飞行状态

    # 计算相对位置和单位方向
    enemy_pos = np.array(best_enemy.get_position())
    relative_vec = enemy_pos - ego_pos
    rel_dir = relative_vec / (np.linalg.norm(relative_vec) + 1e-6)

    rel_vel = best_enemy.get_velocity() - ego_vel
    rel_vel_unit = rel_vel / (np.linalg.norm(rel_vel) + 1e-6)
    # 计算水平面投影的方向
    rel_vel_xy = rel_vel_unit.copy()

    rel_vel_xy[2] = 0
    los_xy = rel_dir.copy()
    los_xy[2] = 0

    if np.linalg.norm(rel_vel_xy) < 1e-3 or np.linalg.norm(los_xy) < 1e-3:
        theta_h = 0
    else:
        theta_h = np.rad2deg(np.arccos(np.clip(
            np.dot(rel_vel_xy, los_xy) /
            (np.linalg.norm(rel_vel_xy) * np.linalg.norm(los_xy) + 1e-6),
            -1, 1)))
        cross_h = np.cross(rel_vel_xy, los_xy)
        theta_h *= np.sign(cross_h[2])  # 右为正，左为负

    # Beam 理想角度是 90 度（即相对方向与航向垂直）
    target_theta = 90.0
    tolerance = 5.0  # 容差

    # 判断是朝左还是朝右转以逼近 90 度
    if theta_h >= 0:
        # 相对速度在 LOS 右侧，目标夹角是 +90°
        if theta_h < target_theta - tolerance:
            heading_cmd = 3  # 右转
        elif theta_h > target_theta + tolerance:
            heading_cmd = 1  # 左转回来
        else:
            heading_cmd = 2  # 保持
    else:
        # 相对速度在 LOS 左侧，目标夹角是 -90°
        if theta_h > -target_theta + tolerance:
            heading_cmd = 1  # 左转靠近 -90°
        elif theta_h < -target_theta - tolerance:
            heading_cmd = 3  # 右转回来
        else:
            heading_cmd = 2  # 保持

    # 仅控制水平偏航，不控制高度
    alt_cmd = 1  # 保持高度
    vel_cmd = 1  # 保持中速（避免太快/太慢）

    #print(f"[BEAM] θ_h = {theta_h:.2f}°, heading_cmd = {heading_cmd}")
    return [alt_cmd, heading_cmd, vel_cmd,shoot]

def Pincer_2ship(
    env, agent_id, radar,
    role='auto',
    R_fire=80000.0,
    SEP_GATE=12000.0,    # 敌两机相距小于此阈值才做钳形，否则直接HOT压进
    L_split=9000.0,      # Phase0 侧向分离目标(米)
    WRAP_SIDE=12000.0,    # Phase1 包抄侧向偏置(米)
    WRAP_BEHIND=6000.0,  # Phase1 包抄“敌后”纵向偏置(米，沿编队轴)
    MAR=20000.0,
    hot_gate=60.0,
    allow_T3=True
):
    import numpy as np
    ego = env.agents[agent_id]

    # ---------- 小工具 ----------
    def unit_xy(v):
        u = v.copy(); u[2] = 0.0
        n = np.linalg.norm(u) + 1e-6
        return u / n
    def turn_toward(cur_xy, tgt_xy):
        cz = np.cross(cur_xy, tgt_xy)[2]
        return 1 if cz < 0 else 3

    # ---------- 局部状态 ----------
    if not hasattr(ego, "_pincer_phase"): ego._pincer_phase = 0   # 0:分离 1:包抄 2:攻击
    if not hasattr(ego, "_pincer_role"):  ego._pincer_role  = None
    if not hasattr(ego, "_pincer_shot"):  ego._pincer_shot  = False

    # 自动分配左右（基于几何：相对“我→敌”基线的左右半平面）
    if ego._pincer_role is None:
        # 若外部显式指定，则直接用（lead/bait 映射到 left/right）
        if role in ("left", "right", "lead", "bait"):
            ego._pincer_role = role if role in ("left", "right") else ("left" if role == "lead" else "right")
        else:
            # 1) 分阵营
            team_tag = agent_id[0]  # 'A' 或 'B'
            team_ids = [k for k in env.agents.keys() if k.startswith(team_tag)]
            opp_ids = [k for k in env.agents.keys() if not k.startswith(team_tag)]

            # 2) 各自质心
            team_cent = np.mean([np.array(env.agents[k].get_position()) for k in team_ids], axis=0)
            opp_cent = np.mean([np.array(env.agents[k].get_position()) for k in opp_ids], axis=0)

            # 3) 基线与“左法向”（NEU 平面左转 90°：(-y, x, 0)）
            to_opp = opp_cent - team_cent
            to_opp[2] = 0.0
            to_opp = to_opp / (np.linalg.norm(to_opp) + 1e-6)
            left_norm = np.array([-to_opp[1], to_opp[0], 0.0])  # 左半平面法向

            # 4) 看自己在左/右半平面
            sign = np.dot(np.array(ego.get_position()) - team_cent, left_norm)
            ego._pincer_role = "left" if sign >= 0 else "right"

    # ---------- 基本量 ----------
    ego_pos = np.array(ego.get_position())
    ego_vel = np.array(ego.get_velocity())
    ego_xy  = unit_xy(ego_vel)

    my_side   = agent_id[0]
    my_ids    = [k for k in env.agents.keys() if k.startswith(my_side)]
    opp_ids   = [k for k in env.agents.keys() if not k.startswith(my_side)]
    my_center   = np.mean([np.array(env.agents[k].get_position()) for k in my_ids], axis=0)
    opp_center  = np.mean([np.array(env.agents[k].get_position()) for k in opp_ids], axis=0)

    # 敌分散度
    opp_positions = [np.array(env.agents[k].get_position()) for k in opp_ids]
    sep = np.linalg.norm(opp_positions[0] - opp_positions[1]) if len(opp_positions) >= 2 else 0.0

    # 编队轴（我方质心 -> 敌方质心）
    axis_xy = unit_xy(opp_center - my_center)

    # 左右基向量；并“自检”以匹配 cross>0=左 的约定
    left_xy  = np.array([-axis_xy[1], axis_xy[0], 0.0])
    right_xy = -left_xy

    # 最近敌（用于LOS/距离/锁定）
    best_enemy, rel_dist = None, float("inf")
    for e in ego.enemies:
        d = np.linalg.norm(np.array(e.get_position()) - ego_pos)
        if d < rel_dist: rel_dist, best_enemy = d, e
    if best_enemy is None:
        return [1, 2, 1, 0]
    tar_pos = np.array(best_enemy.get_position())
    los_xy  = unit_xy(tar_pos - ego_pos)

    # HOT角（与LOS夹角）
    hot = np.degrees(np.arccos(np.clip(np.dot(ego_xy, los_xy), -1.0, 1.0)))

    # 威胁
    incoming = [
        m for m in getattr(ego, "under_missiles", [])
        if getattr(m, "is_alive", False)
           and getattr(m, "target_aircraft", None)
           and m.target_aircraft.uid == ego.uid
    ]
    high_threat = (len(incoming) > 0) #or (rel_dist < 0.8*MAR)

    # 雷达锁定（仅开火时用）
    locked = False
    if radar is not None:
        roll, pitch, yaw = ego.get_rpy() * 180/np.pi
        locked, _ = radar.detect(
            ego_pos=ego_pos, ego_vel=ego_vel, ego_yaw=yaw, ego_pitch=pitch,
            tar_pos=tar_pos, tar_vel=np.array(best_enemy.get_velocity())
        )
    #单机情况
    team = agent_id[0]
    alive = [k for k in env.agents if k.startswith(team) and getattr(env.agents[k], "is_alive", True)]
    if len(alive) < 2:
        ego = env.agents[agent_id]
        ego_pos = np.array(ego.get_position());ego_vel = np.array(ego.get_velocity());ego_xy = ego_vel.copy();ego_xy[2] = 0;ego_xy /= (np.linalg.norm(ego_xy) + 1e-6)
        e = min(ego.enemies, key=lambda x: np.linalg.norm(np.array(x.get_position()) - ego_pos))
        tar_pos = np.array(e.get_position());los_xy = tar_pos - ego_pos;los_xy[2] = 0;los_xy /= (np.linalg.norm(los_xy) + 1e-6)
        hot = np.degrees(np.arccos(np.clip(np.dot(ego_xy, los_xy), -1, 1)))
        locked = False
        if radar is not None:
            _, _, yaw = ego.get_rpy();yaw *= 180 / np.pi
            locked, _ = radar.detect(ego_pos=ego_pos, ego_vel=np.array(ego.get_velocity()),
                                     ego_yaw=yaw, ego_pitch=ego.get_rpy()[1] * 180 / np.pi,
                                     tar_pos=tar_pos, tar_vel=np.array(e.get_velocity()))
        now = getattr(env, "current_step", 0)
        ammo_ok = (getattr(ego, "num_missiles", None) is None) or (getattr(ego, "num_missiles", 1) > 0)
        cd_ok = (now - getattr(ego, "_last_single_shot", -10 ** 9)) >= 80
        rel_dist = np.linalg.norm(tar_pos - ego_pos)
        if locked and (rel_dist <= R_fire) and (hot <= hot_gate) and ammo_ok and cd_ok:
            ego._last_single_shot = now
            return [1, 2, 2, 1]  # 本拍打1发，下一拍继续撤
        heading_cmd = (1 if np.cross(ego_xy, los_xy)[2] > 0 else 3)  # 敌在右→向左
        return [1, heading_cmd, 2, 0]

    # ---------- 默认动作 ----------
    if rel_dist<10000 and (hot <= hot_gate):
        alt_cmd, heading_cmd, vel_cmd, shoot = 1, 2, 2, 1
    else:
        alt_cmd, heading_cmd, vel_cmd, shoot = 1, 2, 2, 0   # vel: 2=加速

    # 敌分很开：不做钳形，直接 HOT 压进（也能射击）
    if sep > SEP_GATE:
        goal_dir = axis_xy
        heading_cmd = turn_toward(ego_xy, goal_dir)
        if locked and (rel_dist <= R_fire) and (hot <= hot_gate) and not ego._pincer_shot:
            shoot = 1; ego._pincer_shot = True
        return [alt_cmd, heading_cmd, vel_cmd, shoot]

    # print("ego._pincer_phase",ego._pincer_phase,"agent_id",agent_id,"high_threat",high_threat)
    # ---------- Phase 0：前向加权的侧向分离 ----------
    if ego._pincer_phase == 0:
        side_xy = left_xy if (ego._pincer_role == "left") else right_xy
        # ☆ 关键：目标朝向带前向分量，避免“横着撤退”
        goal_dir = unit_xy(0.8*axis_xy + 0.6*side_xy)
        # 保证永远不朝反向：若与轴向点积为负，强制用轴向
        if np.dot(goal_dir, axis_xy) < 0: goal_dir = axis_xy

        heading_cmd = turn_toward(ego_xy, goal_dir)
        vel_cmd = 2

        # # 以侧向投影判定是否达到分离量
        # lat = np.dot(ego_pos - my_center, side_xy)   # 左正右负
        # if (ego._pincer_role == "left" and lat >= L_split) or \
        #    (ego._pincer_role == "right" and lat <= -L_split):
        #     ego._pincer_phase = 1
        # return [alt_cmd, heading_cmd, vel_cmd, shoot]
        # 1) 自适应分离目标：敌近→目标侧向小一些；敌远→大一些（不超过 L_split）
        L_MIN = 3000.0
        BRACKET = 50.0  # 侧向开角（度）用于估算合理侧向
        L_need = np.clip(rel_dist * np.sin(np.radians(BRACKET)), L_MIN, L_split)

        lat = np.dot(ego_pos - my_center, side_xy)  # 左正右负

        # 2) 敌已很近 / 高威胁 → 立刻不再分离，转包抄
        if (rel_dist <= 0.6 * MAR) or high_threat:
            ego._pincer_phase = 1
        else:
            # 仅在“达到自适应侧向目标”后再进 Phase 1
            if (ego._pincer_role == "left" and lat >= L_need) or \
                    (ego._pincer_role == "right" and lat <= -L_need):
                ego._pincer_phase = 1
        return [alt_cmd, heading_cmd, vel_cmd, shoot]

    # ---------- Phase 1：飞向包夹点（同样带前向逼近约束） ----------
    elif ego._pincer_phase == 1:
        if allow_T3 and high_threat:
            return tactical_templates.max_g_break_strategy_3d(env, agent_id) if (rel_dist < 0.6*MAR) \
                   else tactical_templates.ortesc_strategy_3d(env, agent_id)

        side_xy = left_xy if (ego._pincer_role == "left") else right_xy
        #goal_point = opp_center - axis_xy * WRAP_BEHIND + side_xy * WRAP_SIDE
        goal_point = opp_center + axis_xy * WRAP_BEHIND + side_xy * WRAP_SIDE
        # 解释：axis_xy 指向 “我方质心 -> 敌方质心”；
        #   真正的“敌后方(远侧)”应该是 opp_center + (+axis_xy)*距离，而不是减。

        goal_xy    = unit_xy(goal_point - ego_pos)

        # 不允许“倒着飞向目标点”：若与轴向点积为负，加入轴向修正
        if np.dot(goal_xy, axis_xy) < 0:
            goal_xy = unit_xy(0.7*goal_xy + 0.5*axis_xy)

        heading_cmd = turn_toward(ego_xy, goal_xy)
        vel_cmd = 2

        # if np.linalg.norm(goal_point - ego_pos) < 200000.0:
        #     ego._pincer_phase = 2
        vec_to_opp = ego_pos - opp_center
        lat_now = abs(np.dot(vec_to_opp, side_xy))  # 侧向达标
        behind_now = np.dot(opp_center - ego_pos, -axis_xy)  # 注意：axis_xy 是我->敌；“在敌后”=朝 +axis 方向更远
        # 为直观，这里重写 behind_now：等价于在“敌后侧”的投影量
        behind_now = np.dot(ego_pos - opp_center, axis_xy)  # >0 表示在敌后（远侧）
        SIDE_goal = min(WRAP_SIDE, 0.5 * rel_dist)
        BEHIND_goal = min(WRAP_BEHIND, 0.5 * rel_dist)
        arrived = (lat_now >= 0.8 * SIDE_goal) and (behind_now >= 0.6 * BEHIND_goal)
        if arrived:
            ego._pincer_phase = 2
        # lat_now = abs(np.dot(ego_pos - opp_center, side_xy))
        # # 新增条件：当敌我距离达到 R_fire 的 70% ~ 90% 时，直接转入攻击
        # R_ENTRY = 0.8 * R_fire  # 例如 80% 射程作为进入攻击阶段的距离门限
        # print("behind_now,lat_now,lat_now:", behind_now, lat_now, lat_now)
        # # 满足侧向，且进入攻击距离
        # if (lat_now >= 0.4 * SIDE_goal):
        #     ego._pincer_phase = 2

        return [alt_cmd, heading_cmd, vel_cmd, shoot]

    # ---------- Phase 2：收回到 HOT 并射击 ----------
    elif ego._pincer_phase == 2:
        heading_cmd = turn_toward(ego_xy, los_xy)
        vel_cmd = 2
        if (rel_dist <= R_fire) and (hot <= hot_gate) and not ego._pincer_shot:
            shoot = 1; ego._pincer_shot = True

        # print(("rel_dist <= R_fire", rel_dist <= R_fire))
        # print("rel_dist",rel_dist)
        # print("(hot <= hot_gate)", (hot <= hot_gate))
        # print("not ego._pincer_shot", not ego._pincer_shot)
        # print(alt_cmd, heading_cmd, vel_cmd, shoot)

        return [alt_cmd, heading_cmd, vel_cmd, shoot]

    return [alt_cmd, heading_cmd, vel_cmd, shoot]

def Grinder_2ship(
    env, agent_id, radar, ctx,
    profile="BVR",                 # Grinder 默认用 BVR 连续压制
    # —— IR 包线（按武器改）——
    R_fire_IR=9000.0, FOV_IR=30.0,
    # —— BVR 包线（按武器改）——
    R_fire_BVR=30000.0, HOT_gate_BVR=60.0, require_lock_BVR=True,
    # —— 护栏与节奏 ——
    cold_angle=150.0, DOR=30000.0, MAR=20000.0,
    crank_ticks_max=2,             # 首射后 crank 1~2 拍
    cooldown_steps=80,             # 同机两次发射最小间隔
    owner_timeout=180,             # 当班超时自动让位
    debug=False
):
    """
    Grinder（双单元轮流压制）：
      当班机：PRESS→(命中门)→FIRE→CRANK(1~2拍)→PUMP至DOR→WAIT，并将owner交给另一机；
      非当班机：HOT接近、在更好窗口时“竞标”接班；遭高威胁先防御。
    约定：
      heading_cmd: 1=左小转, 2=直, 3=右小转
      vel_cmd:     0=减速, 1=中速, 2=加速
    坐标系：左手系，cz=f_x*los_y - f_y*los_x；cz>0 在右，cz<0 在左。
    """
    ego = env.agents[agent_id]

    # ——— 小工具（左手系：cz>0 右 / cz<0 左） ———
    def turn_toward(f_xy, los_xy):
        cz = f_xy[0]*los_xy[1] - f_xy[1]*los_xy[0]
        if   cz >  1e-8: return 3  # 右小转
        elif cz < -1e-8: return 1  # 左小转
        else:             return 2  # 正前/正后
    def turn_away(f_xy, los_xy):
        cz = f_xy[0]*los_xy[1] - f_xy[1]*los_xy[0]
        if   cz >  1e-8: return 1  # 目标在右→向左背离
        elif cz < -1e-8: return 3  # 目标在左→向右背离
        else:             return 2
    def hot_angle_deg(f_xy, los_xy):
        dotc = float(np.clip(np.dot(f_xy, los_xy), -1.0, 1.0))
        return float(np.degrees(np.arccos(dotc)))

    # ——— 机内局部状态 ———
    if not hasattr(ego, "_gr_phase"):         ego._gr_phase = 0   # 0INIT 1PRESS 2CRANK 3PUMP 4WAIT
    if not hasattr(ego, "_gr_shot_step"):     ego._gr_shot_step = -10**9
    if not hasattr(ego, "_gr_crank_ticks"):   ego._gr_crank_ticks = 0
    if not hasattr(ego, "_gr_owner_since"):   ego._gr_owner_since = -10**9

    # ——— 阵营上下文（必须两机共用同一个 ctx） ———
    team = agent_id[0]  # 'A' 或 'B'
    same_side_ids = sorted([k for k in env.agents.keys() if k.startswith(team)])
    my_idx   = same_side_ids.index(agent_id)
    my_elem  = f"E{my_idx}"
    other_el = "E1" if my_elem == "E0" else "E0"

    if "GR" not in ctx or team not in ctx.get("GR", {}):
        ctx.setdefault("GR", {})
        # 近者先当班
        d0 = min(np.linalg.norm(np.array(e.get_position()) - np.array(env.agents[same_side_ids[0]].get_position()))
                 for e in env.agents[same_side_ids[0]].enemies)
        d1 = min(np.linalg.norm(np.array(e.get_position()) - np.array(env.agents[same_side_ids[1]].get_position()))
                 for e in env.agents[same_side_ids[1]].enemies)
        owner = "E0" if d0 <= d1 else "E1"
        ctx["GR"][team] = {"owner": owner, "last_shot_step": -10**9, "owner_since": getattr(env, "current_step", 0)}

    owner_is_me = (ctx["GR"][team]["owner"] == my_elem)

    # ——— 基本几何（最近敌） ———
    ego_pos = np.array(ego.get_position())
    ego_vel = np.array(ego.get_velocity()); spd = np.linalg.norm(ego_vel) + 1e-6
    fwd = ego_vel / spd

    best_enemy, rel_dist = None, float("inf")
    for enemy in ego.enemies:
        R = np.linalg.norm(np.array(enemy.get_position()) - ego_pos)
        if R < rel_dist: rel_dist, best_enemy = R, enemy
    if best_enemy is None:
        return [1, 2, 1, 0]

    tar_pos = np.array(best_enemy.get_position())
    tar_vel = np.array(best_enemy.get_velocity())
    los     = tar_pos - ego_pos
    los_dir = los / (np.linalg.norm(los) + 1e-6)

    # 水平投影
    f_xy   = fwd.copy();    f_xy[2] = 0;    f_xy /= (np.linalg.norm(f_xy)+1e-6)
    los_xy = los_dir.copy(); los_xy[2] = 0; los_xy /= (np.linalg.norm(los_xy)+1e-6)
    hot = hot_angle_deg(f_xy, los_xy)  # 与目标夹角（度）

    # 雷达锁（BVR使用）
    locked = True
    if profile.upper() == "BVR":
        roll, pitch, yaw = ego.get_rpy() * 180/np.pi
        locked, _ = radar.detect(
            ego_pos=ego_pos, ego_vel=ego_vel, ego_yaw=yaw, ego_pitch=pitch,
            tar_pos=tar_pos, tar_vel=tar_vel
        )
        if not require_lock_BVR:
            locked = True

    # 威胁评估
    incoming = [
        m for m in getattr(ego, "under_missiles", [])
        if getattr(m, "is_alive", False)
           and getattr(m, "target_aircraft", None)
           and m.target_aircraft.uid == ego.uid
    ]
    high_threat = (len(incoming) > 0) or (rel_dist < 0.6*MAR)

    # 发射门限（弹药 + 冷却 + 包线）
    now = getattr(env, "current_step", 0)
    ammo = getattr(ego, "num_missiles", None)
    ammo_ok = (ammo is None) or (ammo > 0)
    cd_ok   = (now - ego._gr_shot_step) >= cooldown_steps

    if profile.upper() == "IR":
        dist_ok  = (rel_dist <= R_fire_IR)
        angle_ok = (hot <= FOV_IR)
        fire_gate_ok = dist_ok and angle_ok and ammo_ok and cd_ok
    else:  # BVR
        dist_ok  = (rel_dist <= R_fire_BVR)
        angle_ok = (hot <= HOT_gate_BVR)
        #fire_gate_ok = dist_ok and angle_ok and locked and ammo_ok and cd_ok
        fire_gate_ok = dist_ok and angle_ok

    # 默认动作
    if rel_dist<10000 and (hot <= HOT_gate_BVR):
        alt_cmd, heading_cmd, vel_cmd, shoot = 1, 2, 2, 1
    else:
        alt_cmd, heading_cmd, vel_cmd, shoot = 1, 2, 1, 0

    # print("ego._gr_phase",ego._gr_phase,"agent_id",agent_id)
    # if(fire_gate_ok==True):
    #     print("发射")

    # ========== 我当班：PRESS / CRANK / PUMP ==========
    if owner_is_me:
        if ego._gr_phase in (0, 4):  # INIT/WAIT -> PRESS
            ego._gr_phase = 1
            # ego._gr_crank_ticks = 0
            ego._gr_owner_since = now
            ctx["GR"][team]["owner_since"] = now

        # 高威胁优先防御
        if high_threat:
            if len(incoming) > 0 and rel_dist < 0.6*MAR:
                return tactical_templates.max_g_break_strategy_3d(env, agent_id)
            else:
                return tactical_templates.ortesc_strategy_3d(env, agent_id)

        # 当班超时自动让位（避免死锁）
        if (now - ctx["GR"][team]["owner_since"]) > owner_timeout and ego._gr_phase == 1:
            ctx["GR"][team]["owner"] = other_el
            ctx["GR"][team]["owner_since"] = now
            ego._gr_phase = 4  # WAIT
            if debug: print(f"[{agent_id}] owner timeout -> {other_el}")
            return [alt_cmd, 2, 1, 0]

        # Phase 1: PRESS（压进）
        if ego._gr_phase == 1:
            heading_cmd = turn_toward(f_xy, los_xy)
            vel_cmd = 2  # 加速压进
            if fire_gate_ok:
                shoot = 1                         # 单拍脉冲
                ego._gr_shot_step = now
                ego._gr_phase = 3
                ego._gr_crank_ticks = 0
                # 立刻换班（磨盘节奏关键）
                ctx["GR"][team]["owner"] = other_el
                ctx["GR"][team]["last_shot_step"] = now
                ctx["GR"][team]["owner_since"] = now
                if debug: print(f"[{agent_id}] FIRE @R={int(rel_dist)} hot={int(hot)}")
                return [alt_cmd, heading_cmd, vel_cmd, shoot]
            return [alt_cmd, heading_cmd, vel_cmd, shoot]

        # Phase 2: CRANK（横移 1~2 拍）
        # if ego._gr_phase == 2:
            # if ego._gr_crank_ticks < crank_ticks_max:
            #     # 你已有的 crank 策略（输出 [alt_cmd, heading_cmd, vel_cmd, shoot]）
            #     alt_cmd, heading_cmd, vel_cmd, _ = tactical_templates.crank_strategy_3d(env, agent_id, radar)
            #     ego._gr_crank_ticks += 1
            #     return [alt_cmd, heading_cmd, vel_cmd, 0]
            # else:
            #     ego._gr_phase = 3  # 进入 PUMP

        # Phase 3: PUMP（转冷到 DOR）
        if ego._gr_phase == 3:
            # 直到“冷态阈值”达成（hot >= cold_angle）前持续背离
            if hot < cold_angle:
                heading_cmd = turn_away(f_xy, los_xy)
            else:
                heading_cmd = 2
            vel_cmd = 2
            if rel_dist >= DOR:
                ego._gr_phase = 4  # WAIT
            return [alt_cmd, heading_cmd, vel_cmd, 0]

    # ========== 我不当班：支援 / 竞标接力 ==========
    else:
        vel_cmd = 1
        heading_cmd = turn_toward(f_xy, los_xy) if hot > 20.0 else 2

        # 在“更好窗口”时发起竞标接班（避免对方迟迟不开火）
        owner_age = now - ctx["GR"][team]["owner_since"]
        i_have_window = (profile.upper()=="BVR" and dist_ok and angle_ok and locked) or \
                        (profile.upper()=="IR"  and dist_ok and angle_ok)
        if i_have_window and owner_age > 10:     # 给当班机一个短暂窗口
            ctx["GR"][team]["owner"] = my_elem
            ctx["GR"][team]["owner_since"] = now
            ego._gr_phase = 1
            if debug: print(f"[{agent_id}] take over owner (better window)")
            return [alt_cmd, heading_cmd, vel_cmd, 0]

        # 防御优先
        if high_threat:
            if len(incoming) > 0 and rel_dist < 0.6*MAR:
                return tactical_templates.max_g_break_strategy_3d(env, agent_id)
            else:
                return tactical_templates.ortesc_strategy_3d(env, agent_id)

        if ego._gr_phase == 0:
            ego._gr_phase = 4  # WAIT
        return [alt_cmd, heading_cmd, vel_cmd, 0]

    return [alt_cmd, heading_cmd, vel_cmd, 0]