"""
战术选择算法 - 实现项目说明.md Algorithm 6-1 和 Algorithm 6-1.1

Algorithm 6-1: Tactical Template Selection
- 紧急中断检查（导弹来袭、RWR告警）
- 决策表查询
- 战术连续性检查
- ComputeTacticalFitness评分选择

Algorithm 6-1.1: ComputeTacticalFitness
- 威胁等级评估（权重0.3）
- 队形切换代价（权重0.25）
- 任务完成概率（权重0.30）
- 战术匹配度（权重0.15）
"""
import logging
import numpy as np

try:
    from utils.trace_logger import trace_throttle
except Exception:  # pragma: no cover
    trace_throttle = None


class TacticalSelectorAlgorithm:
    """战术选择算法（完整实现Algorithm 6-1）"""
    
    def __init__(self, threat_evaluator, decision_table):
        """
        初始化战术选择算法
        
        Args:
            threat_evaluator: 威胁评估器
            decision_table: 决策表系统
        """
        self.threat_evaluator = threat_evaluator
        self.decision_table = decision_table
        
        # MAR距离（40km）
        self.MAR_DISTANCE = 40000
        
        # 当前战术（用于连续性检查）
        self.current_tactic = None
        self.last_tactic_change_time = 0.0
        self.disable_randomization = False
        
        logging.info("✅ 战术选择算法初始化完成")
    
    def select_tactic(self, control_distance, my_intent, enemy_intent, situation,
                     my_aircraft, enemy_aircraft, env, current_tactic=None):
        """
        战术模板选择算法（Algorithm 6-1）
        
        Args:
            control_distance: 控制距离 ('NLT', 'MELD', 'DOR', 'DR', etc.)
            my_intent: 我方意图
            enemy_intent: 敌方意图
            situation: 战场态势
            my_aircraft: 我方飞机列表
            enemy_aircraft: 敌方飞机列表
            env: 环境对象
            current_tactic: 当前战术（用于连续性检查）
            
        Returns:
            str: 选定的战术模板名称
        """
        try:
            current_time = getattr(env, 'current_step', 0) * getattr(env, 'time_interval', 0.2)

            # ✅ 中文文件日志：模板选择入口（节流）
            try:
                from utils.trace_logger import trace_throttle, trace_block, trace_span
                me = None
                # 威胁/雷达状态概览
                threat_overview = {}
                snapshot = {}
                try:
                    if isinstance(my_aircraft, (list, tuple)) and my_aircraft:
                        for ac in my_aircraft:
                            if ac is not None and getattr(ac, 'is_alive', True):
                                me = getattr(ac, 'uid', None) or getattr(ac, 'name', None)
                                break
                except Exception:
                    me = None

                try:
                    rep = None
                    if isinstance(my_aircraft, (list, tuple)) and my_aircraft:
                        for ac in my_aircraft:
                            if ac is not None and getattr(ac, 'is_alive', True):
                                rep = ac
                                break

                    # 距离最近敌机
                    closest_enemy = None
                    closest_dist = None
                    if rep is not None and isinstance(enemy_aircraft, (list, tuple)):
                        my_pos = rep.get_position()
                        for ea in enemy_aircraft:
                            if ea is None or not getattr(ea, 'is_alive', False):
                                continue
                            d = float(np.linalg.norm(np.array(ea.get_position()) - np.array(my_pos)))
                            if closest_dist is None or d < closest_dist:
                                closest_dist = d
                                closest_enemy = ea

                    incoming = False
                    rwr_level = None
                    try:
                        if rep is not None:
                            incoming = bool(self.threat_evaluator.detect_incoming_missiles(rep, env))
                    except Exception:
                        incoming = False
                    try:
                        if rep is not None:
                            rwr_level = int(self.threat_evaluator.calculate_rwr_level(rep, enemy_aircraft, env))
                    except Exception:
                        rwr_level = None

                    # 雷达/ECM状态
                    radar_state = None
                    lock_target = None
                    ecm_active = None
                    ecm_type = None
                    ecm_duration = None
                    rwr_bearing = None
                    track_count = None
                    try:
                        from simulation.radar_manager import get_unified_radar_manager
                        rm = get_unified_radar_manager()
                        if rm is not None:
                            st = getattr(rm, 'friendly_radar_states', {}).get(str(me)) if hasattr(rm, 'friendly_radar_states') else None
                            radar_state = st.value if hasattr(st, 'value') else (str(st) if st is not None else None)
                            lock_target = getattr(rm, 'friendly_lock_targets', {}).get(str(me)) if hasattr(rm, 'friendly_lock_targets') else None
                            es = getattr(rm, 'ecm_states', {}).get(str(me)) if hasattr(rm, 'ecm_states') else None
                            if isinstance(es, dict):
                                ecm_active = bool(es.get('active', False))
                                et = es.get('type')
                                ecm_type = et.value if hasattr(et, 'value') else (str(et) if et is not None else None)
                                ecm_duration = float(es.get('duration', 0.0)) if es.get('duration') is not None else None
                            try:
                                if hasattr(rm, 'get_rwr_max_threat_bearing'):
                                    rwr_bearing = rm.get_rwr_max_threat_bearing(str(me))
                            except Exception:
                                rwr_bearing = None
                            try:
                                tgs = getattr(rm, 'friendly_radar_targets', {}).get(str(me), {}) if hasattr(rm, 'friendly_radar_targets') else {}
                                if isinstance(tgs, dict):
                                    track_count = int(len(tgs))
                            except Exception:
                                track_count = None
                    except Exception:
                        pass

                    threat_overview = {
                        "incoming_missile": incoming,
                        "rwr_level": rwr_level,
                        "closest_enemy": getattr(closest_enemy, 'uid', None) if closest_enemy is not None else None,
                        "closest_dist_km": (float(closest_dist) / 1000.0) if closest_dist is not None else None,
                        "radar_state": radar_state,
                        "lock_target": lock_target,
                        "ecm_active": ecm_active,
                        "ecm_type": ecm_type,
                        "ecm_duration": ecm_duration,
                        "rwr_max_bearing_deg": rwr_bearing,
                        "track_count": track_count,
                    }

                    # 详细快照：我/僚/敌 航迹与相对几何 + 态势/威胁分项（尽量能算就算）
                    try:
                        from envs.JSBSim.core.catalog import Catalog as c
                        def _ac_state(ac):
                            if ac is None or not getattr(ac, 'is_alive', True):
                                return None
                            p = ac.get_position()
                            try:
                                hdg = float(ac.get_property_value(c.attitude_psi_deg))
                            except Exception:
                                hdg = None
                            try:
                                spd = float(ac.get_property_value(c.velocities_vc_kts))
                            except Exception:
                                spd = None
                            return {
                                "pos_km": (float(p[0]) / 1000.0, float(p[1]) / 1000.0),
                                "alt_m": float(p[2]),
                                "heading_deg": hdg,
                                "speed_kts": spd,
                                "missiles_left": int(getattr(ac, 'num_missiles', -1)),
                            }

                        snapshot["my"] = _ac_state(rep)
                        # 僚机
                        try:
                            wing = None
                            if isinstance(my_aircraft, (list, tuple)):
                                for ac in my_aircraft:
                                    if ac is None or not getattr(ac, 'is_alive', True):
                                        continue
                                    uid = getattr(ac, 'uid', None) or getattr(ac, 'name', None)
                                    if str(uid) != str(me):
                                        wing = ac
                                        break
                            snapshot["wingman"] = _ac_state(wing)
                        except Exception:
                            snapshot["wingman"] = None

                        # 敌机列表（最多2架）
                        enemies = {}
                        if isinstance(enemy_aircraft, (list, tuple)) and rep is not None:
                            myp = np.array(rep.get_position(), dtype=float)
                            for ea in enemy_aircraft:
                                if ea is None or not getattr(ea, 'is_alive', False):
                                    continue
                                eid = getattr(ea, 'uid', None) or getattr(ea, 'name', None)
                                ep = np.array(ea.get_position(), dtype=float)
                                rel = ep - myp
                                rng = float(np.linalg.norm(rel))
                                bearing = float(np.degrees(np.arctan2(rel[1], rel[0])))
                                bearing = (bearing + 360.0) % 360.0
                                enemies[str(eid)] = {
                                    "state": _ac_state(ea),
                                    "range_km": rng / 1000.0,
                                    "bearing_deg": bearing,
                                }
                        snapshot["enemies"] = enemies
                    except Exception:
                        snapshot = {}

                    # 态势评估分项（选最近敌机）
                    situation_detail = {}
                    try:
                        if rep is not None and closest_enemy is not None:
                            from core.situation_evaluator import SituationEvaluator, TacticalPhase as SEPhase
                            se = SituationEvaluator()
                            phase_map = {
                                "NLT": SEPhase.NLT_MELD,
                                "MELD": SEPhase.MELD_MTR,
                                "MTR": SEPhase.MTR_LR,
                                "LR": SEPhase.LR_TR,
                                "TR": SEPhase.TR_DOR,
                                "DOR": SEPhase.DOR_DR,
                                "DR": SEPhase.DR_MAR,
                                "MAR": SEPhase.BEYOND_MAR,
                            }
                            se_phase = phase_map.get(str(control_distance), SEPhase.NLT_MELD)
                            s = se.evaluate_situation(rep, closest_enemy, se_phase)
                            try:
                                weights = getattr(se, 'phase_weights', {}).get(se_phase, {})
                            except Exception:
                                weights = {}
                            situation_detail = {
                                "situation_score.angle": float(getattr(s, 'angle', 0.0)),
                                "situation_score.distance": float(getattr(s, 'distance', 0.0)),
                                "situation_score.altitude": float(getattr(s, 'altitude', 0.0)),
                                "situation_score.speed": float(getattr(s, 'speed', 0.0)),
                                "situation_score.detection": float(getattr(s, 'detection', 0.0)),
                                "situation_score.total": float(getattr(s, 'total', 0.0)),
                                "situation_weights": weights,
                            }
                    except Exception:
                        situation_detail = {}

                    # 威胁评估分项（选最近敌机）
                    threat_detail = {}
                    try:
                        if rep is not None and closest_enemy is not None:
                            te = self.threat_evaluator
                            # 若是CompleteThreatEvaluator，优先读取其分项
                            parts = {}
                            for name, fn in (
                                ("rwr", "_calculate_rwr_threat"),
                                ("missile", "_calculate_missile_threat"),
                                ("radar", "_calculate_radar_threat"),
                                ("range", "_calculate_range_threat"),
                            ):
                                try:
                                    if hasattr(te, fn):
                                        parts[name] = float(getattr(te, fn)(rep, closest_enemy if name != 'missile' else env,))
                                except Exception:
                                    pass
                            try:
                                total = float(te.calculate_total_threat(rep, closest_enemy, env)) if hasattr(te, 'calculate_total_threat') else None
                            except Exception:
                                total = None
                            threat_detail = {
                                "threat_parts": parts,
                                "threat_total": total,
                                "threat_weights": {
                                    "w_rwr": float(getattr(te, 'w_rwr', 0.0)) if hasattr(te, 'w_rwr') else None,
                                    "w_missile": float(getattr(te, 'w_missile', 0.0)) if hasattr(te, 'w_missile') else None,
                                    "w_radar": float(getattr(te, 'w_radar', 0.0)) if hasattr(te, 'w_radar') else None,
                                    "w_range": float(getattr(te, 'w_range', 0.0)) if hasattr(te, 'w_range') else None,
                                },
                            }
                    except Exception:
                        threat_detail = {}

                    # 合并到snapshot
                    snapshot = {
                        **snapshot,
                        "threat_overview": threat_overview,
                        "situation_detail": situation_detail,
                        "threat_detail": threat_detail,
                    }
                except Exception:
                    threat_overview = {}

                trace_throttle(
                    key=f"tactic_select:enter:{me or 'team'}:{control_distance}",
                    min_steps=60,
                    标题="原因链-模板选择入口",
                    env=env,
                    模块="tactical_selector",
                    类型="DECISION",
                    状态="INPUT",
                    我机=str(me) if me else None,
                    阶段=str(control_distance),
                    战术=str(current_tactic or self.current_tactic),
                    说明="开始战术模板选择（候选→评分→选择）",
                    要点=[
                        f"I_self={my_intent} I_enemy={enemy_intent} situation={situation}",
                        "威胁概览: 来袭导弹/RWR/最近敌机距离/雷达状态/ECM",
                    ],
                    数据=threat_overview,
                )

                # ✅ 更详细的“输入快照”块（节流但更频繁一点）：用于写分析报告
                try:
                    trace_throttle(
                        key=f"tactic_select:snapshot:{me or 'team'}:{control_distance}",
                        min_steps=30,
                        标题="原因链-模板选择输入快照",
                        env=env,
                        模块="tactical_selector",
                        类型="DECISION",
                        状态="INPUT",
                        我机=str(me) if me else None,
                        阶段=str(control_distance),
                        战术=str(current_tactic or self.current_tactic),
                        说明="记录战术选择的主要输入：航迹/几何/态势分项/威胁分项/雷达ECM",
                        要点=[
                            f"I_self={my_intent} I_enemy={enemy_intent} situation={situation}",
                            "若字段为空：表示当前对象/接口在此阶段不可用或计算失败",
                        ],
                        数据={
                            "current_time_s": float(current_time),
                            "snapshot": snapshot,
                        },
                    )
                except Exception:
                    pass
            except Exception:
                pass
            # 更新当前战术
            if current_tactic:
                self.current_tactic = current_tactic
            
            # ===== 1. 紧急中断检查（生存优先） =====
            
            # 检查导弹来袭
            for my_ac in my_aircraft:
                if my_ac and my_ac.is_alive:
                    if self.threat_evaluator.detect_incoming_missiles(my_ac, env):
                        logging.warning(f"🚨 检测到导弹来袭！立即执行战术规避")
                        try:
                            from utils.trace_logger import trace_block
                            trace_block(
                                标题="原因链-紧急中断",
                                env=env,
                                我机=str(getattr(my_ac, 'uid', None) or getattr(my_ac, 'name', None)),
                                阶段=str(control_distance),
                                战术=str(current_tactic or self.current_tactic),
                                说明="检测到导弹来袭，生存优先强制规避",
                                要点=["触发条件=detect_incoming_missiles=True", "返回战术=TACTICAL_EVASION"],
                            )
                        except Exception:
                            pass
                        return 'TACTICAL_EVASION'  # T6
            
            # 检查RWR告警等级（传入env参数以检测导弹）
            for my_ac in my_aircraft:
                if my_ac and my_ac.is_alive:
                    rwr_level = self.threat_evaluator.calculate_rwr_level(my_ac, enemy_aircraft, env)
                    
                    # 计算距离
                    if enemy_aircraft:
                        enemy_pos = enemy_aircraft[0].get_position() if enemy_aircraft[0].is_alive else enemy_aircraft[1].get_position()
                        my_pos = my_ac.get_position()
                        distance = np.linalg.norm(np.array(enemy_pos) - np.array(my_pos))
                    else:
                        distance = 100000
                    
                    # RWR≥4 或 距离≤MAR
                    if rwr_level >= 4 or distance <= self.MAR_DISTANCE:
                        if my_intent == 'AGGRESSIVE_CLEAR':
                            # 激进策略：继续当前战术，但在机动层调整参数
                            logging.info(f"⚠️ RWR告警{rwr_level}级或距离≤MAR，激进策略继续当前战术")
                            # continue（不返回，继续后续决策）
                        else:
                            # 保守/防御策略：执行战术回转
                            logging.warning(f"🚨 RWR告警{rwr_level}级或距离≤MAR！执行战术回转")
                            try:
                                from utils.trace_logger import trace_block
                                trace_block(
                                    标题="原因链-紧急中断",
                                    env=env,
                                    我机=str(getattr(my_ac, 'uid', None) or getattr(my_ac, 'name', None)),
                                    阶段=str(control_distance),
                                    战术=str(current_tactic or self.current_tactic),
                                    说明="RWR高告警或进入MAR，非激进意图则强制回转/返航语义",
                                    要点=[
                                        f"RWR={rwr_level}",
                                        f"distance_km={distance/1000:.1f}",
                                        f"I_self={my_intent}",
                                        "返回战术=TACTICAL_TURN",
                                    ],
                                )
                            except Exception:
                                pass
                            return 'TACTICAL_TURN'  # T7
            
            # ===== 2. 从决策表查询候选战术集合 =====
            candidates = self.decision_table.query_candidates(
                control_distance, my_intent, enemy_intent, situation
            )

            # ✅ 中文文件日志：候选集（变化触发）
            try:
                from utils.trace_logger import trace_if_changed
                me = None
                try:
                    if isinstance(my_aircraft, (list, tuple)) and my_aircraft:
                        for ac in my_aircraft:
                            if ac is not None and getattr(ac, 'is_alive', True):
                                me = getattr(ac, 'uid', None) or getattr(ac, 'name', None)
                                break
                except Exception:
                    me = None
                trace_if_changed(
                    key=f"tactic_select:candidates:{me or 'team'}:{control_distance}",
                    value=tuple(candidates) if candidates else tuple(),
                    标题="原因链-模板候选集",
                    env=env,
                    我机=str(me) if me else None,
                    阶段=str(control_distance),
                    战术=str(current_tactic or self.current_tactic),
                    说明="决策表查询得到的候选模板",
                    要点=[
                        f"I_self={my_intent} I_enemy={enemy_intent} situation={situation}",
                        f"candidates={list(candidates) if candidates else []}",
                    ],
                )
            except Exception:
                pass
            
            if not candidates:
                logging.warning("决策表查询无结果，使用默认战术")
                return 'SIDE_BY_SIDE'
            
            # ===== 3. 战术连续性检查（增强版 - 允许条件切换） =====
            # 检查是否应该强制切换战术
            should_force_switch = False
            
            # 条件1：敌机数量变化（有敌机被击落）
            alive_enemies = sum(1 for e in enemy_aircraft if e and e.is_alive)
            if hasattr(self, 'last_enemy_count') and alive_enemies < self.last_enemy_count:
                should_force_switch = True
                logging.info(f"🔄 敌机数量变化 ({self.last_enemy_count}→{alive_enemies})，允许战术切换")
            self.last_enemy_count = alive_enemies
            
            # 条件2：态势显著变化
            if hasattr(self, 'last_situation') and self.last_situation != situation:
                if (self.last_situation == 'DISADVANTAGE' and situation == 'ADVANTAGE') or \
                   (self.last_situation == 'ADVANTAGE' and situation == 'DISADVANTAGE'):
                    should_force_switch = True
                    logging.info(f"🔄 态势显著变化 ({self.last_situation}→{situation})，允许战术切换")
            self.last_situation = situation
            
            # 条件3：在特定节点（DOR、DR）允许切换
            if control_distance in ['DOR', 'DR']:
                should_force_switch = True
                logging.debug(f"🔄 控制距离{control_distance}，允许战术评估")
            
            # 如果当前战术在候选集中，且没有强制切换条件，保持当前战术
            if self.current_tactic and self.current_tactic in candidates and not should_force_switch:
                # 🔥 只为我方输出日志（通过检查my_aircraft列表）
                is_friendly = False
                try:
                    if isinstance(my_aircraft, (list, tuple)) and my_aircraft:
                        for ac in my_aircraft:
                            if ac is not None and hasattr(ac, 'uid'):
                                if str(ac.uid).startswith('A'):
                                    is_friendly = True
                                    break
                except:
                    pass
                
                if is_friendly:
                    logging.info(f"[战术连续性] 保持当前战术 {self.current_tactic}")
                return self.current_tactic
            
            # ===== 4. 基于态势的战术评分与选择（改进为加权随机选择）=====
            tactic_scores = []
            
            for tactic in candidates:
                score = self._compute_tactical_fitness(
                    tactic, situation, my_intent, enemy_intent,
                    my_aircraft, enemy_aircraft, env
                )
                tactic_scores.append((tactic, score))
                logging.info(f"   战术 {tactic}: 适应度 = {score:.3f}")
            
            if tactic_scores:
                # 🔧 修复：移除硬编码的前3名限制，确保所有战术都能被评估
                tactic_scores.sort(key=lambda x: x[1], reverse=True)
                
                # 🔥 任务2：修复战术选择多样性问题 - 降低阈值，增加选择多样性
                # ✅ 修复：阈值必须基于“最高分”，不能用 max(tuple)（会按字符串比较导致阈值错误）
                best_score = float(tactic_scores[0][1])
                threshold = best_score * 0.5  # 🔥 降低阈值：70% → 50%，让更多战术有机会被选中
                valid_tactics = [(t, s) for t, s in tactic_scores if s >= threshold]

                # 记录“被筛掉”的原因（主要是低于阈值）
                excluded = [(t, float(s)) for (t, s) in tactic_scores if float(s) < float(threshold)]

                # 🔥 任务2：确保所有战术都有机会被选中，移除数量限制
                if len(valid_tactics) < 2:
                    # 如果阈值过严，使用所有战术而不是限制为3个
                    valid_tactics = tactic_scores  # 🔥 移除min(3, len(tactic_scores))限制
                
                # file-only: 结果将由 trace_throttle 统一记录（避免控制台刷屏）
                
                # 从合格战术中加权随机选择（分数越高，被选中概率越大）
                import random
                selection_mode = "best_fitness"
                selection_detail = {}
                if (
                    not self.disable_randomization
                    and len(valid_tactics) >= 2
                    and (valid_tactics[0][1] - valid_tactics[-1][1] < 0.20)
                ):
                    # 如果适应度相近（差距<0.20），增加随机性
                    selection_mode = "weighted_random_close_scores"
                    weights = [1.0 * (0.8 ** i) for i in range(len(valid_tactics))]  # 递减权重
                    selected_tactic = random.choices([t[0] for t in valid_tactics], 
                                                    weights=weights)[0]
                    selected_score = next(score for tactic, score in valid_tactics if tactic == selected_tactic)
                    # file-only: 结果将由 trace_throttle 统一记录（避免控制台刷屏）
                    try:
                        selection_detail = {
                            "delta_top_bottom": float(valid_tactics[0][1]) - float(valid_tactics[-1][1]),
                            "eligible": [f"{t}={float(s):.3f}" for t, s in valid_tactics[:8]],
                            "weights": [float(w) for w in weights[:8]],
                        }
                    except Exception:
                        selection_detail = {}
                else:
                    # 否则选择最佳战术
                    selected_tactic = valid_tactics[0][0]
                    selected_score = valid_tactics[0][1]
                    # file-only: 结果将由 trace_throttle 统一记录（避免控制台刷屏）
                
                # 记录切换时间
                if selected_tactic != self.current_tactic:
                    self.last_tactic_change_time = current_time
                self.current_tactic = selected_tactic

                # ✅ 中文文件日志：评分与最终选择（节流）
                try:
                    me = None
                    try:
                        if isinstance(my_aircraft, (list, tuple)) and my_aircraft:
                            for ac in my_aircraft:
                                if ac is not None and getattr(ac, 'is_alive', True):
                                    me = getattr(ac, 'uid', None) or getattr(ac, 'name', None)
                                    break
                    except Exception:
                        me = None

                    top_lines = []
                    try:
                        for t, s in tactic_scores[:6]:
                            top_lines.append(f"{t}={float(s):.3f}")
                    except Exception:
                        top_lines = [str(tactic_scores)]

                    trace_throttle(
                        key=f"tactic_select:result:{me or 'team'}:{control_distance}",
                        min_steps=60,
                        标题="原因链-模板选择结果",
                        env=env,
                        我机=str(me) if me else None,
                        阶段=str(control_distance),
                        战术=str(selected_tactic),
                        说明="候选适应度评分→阈值筛选→最终模板",
                        要点=[
                            f"I_self={my_intent} I_enemy={enemy_intent} situation={situation}",
                            f"候选数={len(tactic_scores)} 合格数={len(valid_tactics)} 阈值≈{float(threshold):.3f} (基于best={best_score:.3f})",
                            f"top_scores={'; '.join(top_lines)}",
                            f"selected={selected_tactic} score≈{float(selected_score):.3f}",
                            ("未选最高分原因=分差较小启用随机多样性" if (selection_mode == 'weighted_random_close_scores' and selected_tactic != valid_tactics[0][0]) else ""),
                        ],
                        数据={
                            "selection_mode": selection_mode,
                            "selection_detail": selection_detail or None,
                            "excluded_below_threshold": [f"{t}={s:.3f}" for (t, s) in excluded[:8]] if excluded else [],
                            "note": "若出现‘高分未选’，优先检查 selection_mode=weighted_random_close_scores；若在excluded列表则是低于阈值淘汰",
                        },
                    )
                except Exception:
                    pass
                return selected_tactic
            else:
                logging.warning("未找到最佳战术，使用默认战术")
                return 'SIDE_BY_SIDE'
            
        except Exception as e:
            logging.error(f"战术选择错误: {e}")
            try:
                return current_tactic if current_tactic else 'SIDE_BY_SIDE'
            except Exception:
                return 'SIDE_BY_SIDE'
    
    def _compute_tactical_fitness(self, tactic, situation, my_intent, enemy_intent,
                                  my_aircraft, enemy_aircraft, env):
        """
        战术适应性评估函数（Algorithm 6-1.1）
        
        Args:
            tactic: 候选战术模板
            situation: 战场态势
            my_intent: 我方意图
            enemy_intent: 敌方意图
            my_aircraft: 我方飞机列表
            enemy_aircraft: 敌方飞机列表
            env: 环境对象
            
        Returns:
            float: 适应度分数 [0, 1]
        """
        try:
            # 1. 威胁等级评估（权重0.35 - 提高威胁评估权重）
            threat_fitness = self._evaluate_threat_fitness(tactic, my_aircraft, enemy_aircraft, env)

            # 2. 队形切换代价（权重0.20 - 降低队形切换权重）
            formation_fitness = self._evaluate_formation_fitness(tactic)

            # 3. 任务完成概率（权重0.30 - 保持不变）
            success_prob = self._evaluate_success_probability(tactic, situation, my_intent, enemy_intent)

            # 4. 战术匹配度（权重0.15 - 保持不变）
            tactical_match = self._evaluate_tactical_match(tactic, my_intent, enemy_intent, situation)

            # 加权求和（优化后的权重：0.35 + 0.20 + 0.30 + 0.15 = 1.0）
            fitness_score = (
                0.35 * threat_fitness +
                0.20 * formation_fitness +
                0.30 * success_prob +
                0.15 * tactical_match
            )

            # 记录详细评分信息
            logging.debug(f"[战术评分] {tactic}: 威胁={threat_fitness:.2f}, 队形={formation_fitness:.2f}, "
                         f"成功率={success_prob:.2f}, 匹配度={tactical_match:.2f}, 总分={fitness_score:.2f}")
            
            return np.clip(fitness_score, 0.0, 1.0)

        except Exception as e:
            logging.error(f"适应度计算错误: {e}")
            return 0.5

    def _evaluate_threat_fitness(self, tactic, my_aircraft, enemy_aircraft, env):
        """
        威胁等级评估（权重0.3）

        根据项目说明.md Algorithm 6-1.1:
        - 防御性战术（T6, T7）：威胁越高，适应性越强
        - 攻击性战术：威胁越低，适应性越强
        """
        try:
            # 计算动态威胁等级（基于距离、数量、RWR等）
            total_threat = 0.0
            count = 0
            current_time = getattr(env, 'current_step', 0) * getattr(env, 'time_interval', 0.2)

            for my_ac in my_aircraft:
                if my_ac and my_ac.is_alive:
                    # 获取RWR威胁等级
                    try:
                        from simulation.radar_manager import get_unified_radar_manager
                        rm = get_unified_radar_manager()
                        agent_id = getattr(my_ac, 'agent_id', None) or getattr(my_ac, 'uid', 'unknown')
                        rwr_level = rm.get_rwr_threat_level(agent_id)
                        rwr_threat = rwr_level / 5.0  # 归一化到[0,1]
                    except:
                        rwr_threat = 0.0

                    for enemy_ac in enemy_aircraft:
                        if enemy_ac and enemy_ac.is_alive:
                            # 计算基础威胁（距离相关）
                            try:
                                my_pos = my_ac.get_position()
                                enemy_pos = enemy_ac.get_position()
                                distance = ((my_pos[0] - enemy_pos[0])**2 + (my_pos[1] - enemy_pos[1])**2)**0.5
                                # 距离威胁：近距离威胁高
                                distance_threat = max(0, (80000 - distance) / 80000)
                                
                                # 角度威胁
                                angle_threat = self.threat_evaluator.calculate_angle_threat(my_ac, enemy_ac, env)
                                
                                # 综合威胁
                                individual_threat = (distance_threat + angle_threat + rwr_threat) / 3.0
                                total_threat += individual_threat
                                count += 1
                            except:
                                # 回退到简单计算
                                threat = self.threat_evaluator.calculate_total_threat(my_ac, enemy_ac, env)
                                total_threat += threat
                                count += 1

            if count > 0:
                avg_threat = total_threat / count
            else:
                avg_threat = 0.5

            # 添加时间因素（仿真进行越久，威胁可能变化）
            time_factor = min(1.0, current_time / 300.0) * 0.1  # 5分钟内从0增长到0.1
            avg_threat += time_factor

            # 归一化威胁等级
            normalized_threat = np.clip(avg_threat, 0.0, 1.0)

            # 根据战术类型计算适应度 - 平衡各战术选择
            if tactic in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                # 防御性战术：威胁越高，适应性越强
                threat_fitness = normalized_threat  # 威胁高 → 分数高
            else:
                # 攻击性战术：威胁评估更平衡，避免DRAG_SHOOT过度优势
                base_fitness = 1.0 - normalized_threat  # 威胁低 → 分数高
                
                # 为不同攻击性战术添加特色调整（平衡化权重）
                if tactic == 'DRAG_SHOOT':
                    # 拖曳射击：在中等威胁时表现更佳
                    if 0.3 <= normalized_threat <= 0.7:
                        adjustment = 0.08  # 中等威胁时小幅加成（降低：0.1→0.08）
                    else:
                        adjustment = -0.02  # 其他情况小幅减分（减少：-0.05→-0.02）
                elif tactic == 'PINCER_ATTACK':
                    # 钳形攻击：在低威胁时表现更佳（降低权重）
                    if normalized_threat <= 0.4:
                        adjustment = 0.08  # 低威胁时加成（大幅降低：0.15→0.08）
                    else:
                        adjustment = -0.05  # 其他情况减分（新增）
                elif tactic == 'HIGH_LOW_ATTACK':
                    # 高低攻击：在高威胁时表现更佳
                    if normalized_threat >= 0.6:
                        adjustment = 0.10  # 高威胁时加成（轻微降低：0.12→0.10）
                    else:
                        adjustment = 0.0
                elif tactic == 'SEQUENTIAL_ATTACK' or tactic == 'FRONT_BACK':
                    # 🔥 任务2：序列攻击（FRONT_BACK）：大幅提升适应度，确保能被选中
                    if 0.3 <= normalized_threat <= 0.9:
                        adjustment = 0.15  # 🔥 大幅提高评分：0.12 → 0.15
                    elif 0.1 <= normalized_threat <= 0.95:
                        adjustment = 0.12  # 🔥 扩大适用范围：0.08 → 0.12
                    else:
                        adjustment = 0.08  # 🔥 提高基础分：0.02 → 0.08
                elif tactic == 'SIDE_BY_SIDE':
                    # 🔥 任务2：并排攻击：大幅提升适应度，确保能被选中
                    if normalized_threat <= 0.7:
                        adjustment = 0.14  # 🔥 大幅提高评分：0.10 → 0.14
                    elif normalized_threat <= 0.9:
                        adjustment = 0.10  # 🔥 扩大适用范围和提高评分：0.06 → 0.10
                    else:
                        adjustment = 0.06  # 🔥 提高基础分：0.02 → 0.06
                else:
                    adjustment = 0.0
                    
                threat_fitness = base_fitness + adjustment

            # 减少随机扰动，但保持一定变化
            import random
            random_factor = random.uniform(0.95, 1.05)  # 减小随机范围
            threat_fitness *= random_factor

            return np.clip(threat_fitness, 0.0, 1.0)

        except Exception as e:
            logging.error(f"威胁适应度评估错误: {e}")
            return 0.5

    def _evaluate_formation_fitness(self, tactic):
        """
        队形切换代价评估（权重0.25）- 平衡各战术选择概率

        计算从当前队形到目标队形的转换成本
        """
        try:
            # 简化版本：不同战术的队形切换代价
            # 如果当前战术与目标战术相同，代价为0
            if self.current_tactic == tactic:
                return 1.0  # 无需切换，适应度最高

            # 🔥 任务2：进一步优化队形切换代价矩阵，增加FRONT_BACK和SIDE_BY_SIDE选择概率
            base_costs = {
                'DRAG_SHOOT': 0.20,  # 🔥 轻微提高拖曳射击代价：0.18 → 0.20
                'PINCER_ATTACK': 0.28,  # 🔥 进一步提高钳形攻击代价：0.25 → 0.28
                'HIGH_LOW_ATTACK': 0.24,  # 🔥 提高高低攻击代价：0.20 → 0.24
                'FRONT_BACK': 0.12,  # 🔥 大幅降低前后攻击代价：0.17 → 0.12
                'SIDE_BY_SIDE': 0.14,  # 🔥 大幅降低并排攻击代价：0.19 → 0.14
                'SEQUENTIAL_ATTACK': 0.12,  # 🔥 确保SEQUENTIAL_ATTACK也有低代价
            }

            base_cost = base_costs.get(tactic, 0.20)
            
            # 增加随机扰动幅度，增加选择多样性
            import random
            random_factor = random.uniform(0.85, 1.15)  # 扩大随机范围：±15%
            cost = base_cost * random_factor

            # 适应度 = 1 - 代价
            formation_fitness = max(0.0, 1.0 - cost)

            return formation_fitness

        except Exception as e:
            logging.error(f"队形适应度评估错误: {e}")
            return 0.5

    def _evaluate_success_probability(self, tactic, situation, my_intent, enemy_intent):
        """
        任务完成概率评估（权重0.30）

        基于敌我态势预测战术成功率
        """
        try:
            # 基础成功率（根据态势）
            if situation == 'ADVANTAGE':
                base_prob = 0.8
            elif situation == 'NEUTRAL':
                base_prob = 0.6
            else:  # DISADVANTAGE
                base_prob = 0.4

            # 根据战术类型调整
            if tactic in ['PINCER_ATTACK', 'HIGH_LOW_ATTACK']:
                # 协同战术在优势态势下成功率更高
                if situation == 'ADVANTAGE':
                    tactic_modifier = 0.1
                else:
                    tactic_modifier = -0.1
            elif tactic in ['SEQUENTIAL_ATTACK', 'DRAG_SHOOT']:
                # 时序战术在均势下成功率较高
                if situation == 'NEUTRAL':
                    tactic_modifier = 0.1
                else:
                    tactic_modifier = 0.0
            elif tactic == 'SIDE_BY_SIDE':
                # 并排战术在劣势下成功率相对较高（简单直接）
                if situation == 'DISADVANTAGE':
                    tactic_modifier = 0.1
                else:
                    tactic_modifier = 0.0
            elif tactic in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                # 防御战术在劣势下成功率较高
                if situation == 'DISADVANTAGE':
                    tactic_modifier = 0.2
                else:
                    tactic_modifier = 0.0
            else:
                tactic_modifier = 0.0

            # 根据敌方意图调整
            if enemy_intent == 'ATTACK' and tactic in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                intent_modifier = 0.1  # 敌方攻击时，防御战术成功率提高
            elif enemy_intent == 'RETREAT' and tactic in ['SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE']:
                intent_modifier = 0.1  # 敌方撤退时，追击战术成功率提高
            else:
                intent_modifier = 0.0

            # 综合成功概率
            success_prob = base_prob + tactic_modifier + intent_modifier

            return np.clip(success_prob, 0.0, 1.0)

        except Exception as e:
            logging.error(f"成功概率评估错误: {e}")
            return 0.5

    def _evaluate_tactical_match(self, tactic, my_intent, enemy_intent, situation):
        """
        战术匹配度评估（权重0.15）

        衡量战术与当前意图和态势的契合程度
        """
        try:
            match_score = 0.5  # 基础匹配度

            # 🔥 [已禁用our_intent] 不再基于我方意图对战术进行加减分，让5种进攻战术公平竞争
            pass

            # 根据态势调整
            if situation == 'ADVANTAGE' and tactic in ['PINCER_ATTACK', 'HIGH_LOW_ATTACK']:
                match_score += 0.1
            elif situation == 'DISADVANTAGE' and tactic in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                match_score += 0.1

            return np.clip(match_score, 0.0, 1.0)

        except Exception as e:
            logging.error(f"战术匹配度评估错误: {e}")
            return 0.5

