"""Extracted CAP tactic-selection and situation-log helpers."""

import logging
import os
from typing import Dict, List, Optional

import numpy as np

from envs.JSBSim.core.catalog import Catalog as c

try:
    from .cap_state_machine import CAPState, StateContext
    from .control_ranges import ControlRanges, DEFAULT_RANGES
    from .faor_manager import RiskZone
    from .mission_evaluator import MissionResult
    from .tactic_selector import TacticAssignment, TacticType
except ImportError:
    from cap_state_machine import CAPState, StateContext
    from control_ranges import ControlRanges, DEFAULT_RANGES
    from faor_manager import RiskZone
    from mission_evaluator import MissionResult
    from tactic_selector import TacticAssignment, TacticType

log = logging.getLogger(__name__)

_CAP_DEBUG_PRINT = os.environ.get('CAP_DEBUG_PRINT') == '1'
_CAP_VERIFY_TABLE = os.environ.get('CAP_VERIFY_TABLE', '0').strip().lower() not in ('0', 'false', 'no', 'off')


def _cap_table_print(*args, **kwargs):
    if _CAP_VERIFY_TABLE or _CAP_DEBUG_PRINT:
        print(*args, **kwargs)


def _should_emit_tactic_log(self, current_time: float, signature, interval_s: float = 20.0) -> bool:
    last_time = float(getattr(self, "_last_tactic_diag_log_time", -9999.0))
    last_sig = getattr(self, "_last_tactic_diag_log_signature", None)
    if (current_time - last_time) >= float(interval_s) or signature != last_sig:
        self._last_tactic_diag_log_time = float(current_time)
        self._last_tactic_diag_log_signature = signature
        return True
    return False

def _log_control_ranges_once(self, initial_distance: float):
    """启动时一次性打印控制距离节点（含二次进攻节点）。"""
    base = DEFAULT_RANGES.get_all()
    base_text = (
        f"NLT={base['NLT']:.0f} MELD={base['MELD']:.0f} MTR={base['MTR']:.0f} "
        f"LR={base['LR']:.0f} TR={base['TR']:.0f} DOR={base['DOR']:.0f} DR={base['DR']:.0f} MAR={base['MAR']:.0f}"
    )
    second_text = (
        f"MTR'={base['MTR_PRIME']:.0f} LR'={base['LR_PRIME']:.0f} TR'={base['TR_PRIME']:.0f}"
    )

    compressed = DEFAULT_RANGES.get_compressed_ranges(initial_distance)
    if compressed:
        ratio = min(1.0, initial_distance / DEFAULT_RANGES.COMPRESSION_BASE)
        comp_text = (
            f"NLT={compressed['NLT']:.1f} MELD={compressed['MELD']:.1f} MTR={compressed['MTR']:.1f} "
            f"LR={compressed['LR']:.1f} TR={compressed['TR']:.1f} DOR={compressed['DOR']:.1f} "
            f"DR={compressed['DR']:.1f} MAR={compressed['MAR']:.1f} "
            f"MTR'={compressed['MTR_PRIME']:.1f} LR'={compressed['LR_PRIME']:.1f} TR'={compressed['TR_PRIME']:.1f}"
        )
        log.info(f"[节点参数] 理论节点(km): {base_text} | 二次进攻: {second_text}")
        log.info(f"[节点参数] 初始距离={initial_distance:.1f}km 压缩比={ratio:.0%} -> 动态节点(km): {comp_text}")
    else:
        log.info(f"[节点参数] 理论节点(km): {base_text} | 二次进攻: {second_text}")
        log.info(f"[节点参数] 初始距离={initial_distance:.1f}km < MAR，直接进入紧急规避区")

def _get_second_attack_subnode(self, distance: float, ranges_dict: Dict[str, float] = None) -> Optional[str]:
    """在DR~MAR区间内细分二次进攻子节点。"""
    if not DEFAULT_RANGES.is_second_attack_zone(distance):
        return None

    def get_val(key: str) -> float:
        if ranges_dict and key in ranges_dict:
            return float(ranges_dict[key])
        return float(getattr(DEFAULT_RANGES, key))

    mtr_p = get_val('MTR_PRIME')
    lr_p = get_val('LR_PRIME')
    tr_p = get_val('TR_PRIME')

    if distance > mtr_p:
        return "MTRP_ENTRY"
    if distance > lr_p:
        return "MTRP_LRP"
    if distance > tr_p:
        return "LRP_TRP"
    return "TRP_MAR"

def _node_to_progress(self, node: str) -> int:
    """模板节点顺序编码（用于禁止阶段回退）。"""
    order = {
        'BEYOND_NLT': 0,
        'NLT_MELD': 1,
        'MELD_MTR': 2,
        'MTR_LR': 3,
        'LR_TR': 4,
        'TR_DOR': 5,
        'DOR_DR': 6,
        'DR_MAR': 7,
        'BELOW_MAR': 8,
    }
    return order.get(node, 0)

def _lock_template_node(self, agent_id: str, target_id: str, raw_node: str) -> str:
    """按机按目标锁定模板节点，禁止回退（对齐旧版phase_lock思想）。"""
    if not target_id:
        return raw_node

    prev_target = self._agent_template_target.get(agent_id)
    if prev_target != target_id:
        self._agent_template_target[agent_id] = target_id
        self._agent_template_node_progress[agent_id] = self._node_to_progress(raw_node)
        return raw_node

    old_prog = self._agent_template_node_progress.get(agent_id, self._node_to_progress(raw_node))
    new_prog = self._node_to_progress(raw_node)
    if new_prog < old_prog:
        if str(agent_id).startswith('A'):
            self._agent_template_node_progress[agent_id] = new_prog
            return raw_node
        for node_name in ('BEYOND_NLT', 'NLT_MELD', 'MELD_MTR', 'MTR_LR', 'LR_TR', 'TR_DOR', 'DOR_DR', 'DR_MAR', 'BELOW_MAR'):
            if self._node_to_progress(node_name) == old_prog:
                return node_name
        return raw_node

    self._agent_template_node_progress[agent_id] = new_prog
    return raw_node

def _pick_pair_target_with_stickiness(self, pair_name: str, alive_pair: List[str], remaining_threats: List):
    """为2v2对子选择目标：优先保持原目标，避免每步重分配导致节点回跳。"""
    if not remaining_threats or not alive_pair:
        return None

    by_id = {t.track_id: t for t in remaining_threats}
    lock_id = self._pair_target_memory.get(pair_name)
    lock_until = self._pair_target_lock_until_step.get(pair_name, -1)

    # 目标仍有效时持续保持（强粘滞），避免对子在交战中频繁换目标导致阶段回跳
    if lock_id in by_id:
        return by_id[lock_id]

    pair_center = np.mean([self._get_battlefield_pos_placeholder(aid) for aid in alive_pair], axis=0)
    best = min(
        remaining_threats,
        key=lambda t: np.linalg.norm(np.array([t.x, t.y], dtype=float) - pair_center)
    )
    self._pair_target_memory[pair_name] = best.track_id
    self._pair_target_lock_until_step[pair_name] = self.step_count + 900
    return best

def _get_pair_name_for_agent(self, agent_id: str) -> Optional[str]:
    if agent_id in ('A0100', 'A0200', 'B0100', 'B0200'):
        return 'left'
    if agent_id in ('A0300', 'A0400', 'B0300', 'B0400'):
        return 'right'
    return None

def _get_pair_enemy_ids(self, agent_id: str) -> List[str]:
    pair_name = self._get_pair_name_for_agent(agent_id)
    if pair_name == 'left':
        return ['B0100', 'B0200'] if agent_id.startswith('A') else ['A0100', 'A0200']
    if pair_name == 'right':
        return ['B0300', 'B0400'] if agent_id.startswith('A') else ['A0300', 'A0400']
    return []

def _get_pair_tactic_assignment(self, agent_id: str) -> Optional[TacticAssignment]:
    assignments = getattr(self, '_tactic_assignments_by_agent', {})
    assign = assignments.get(agent_id)
    if assign is not None:
        return assign

    pair_name = self._get_pair_name_for_agent(agent_id)
    if pair_name is None:
        return getattr(self, '_last_tactic_assignment', None)

    pair_agents = ('A0100', 'A0200') if pair_name == 'left' else ('A0300', 'A0400')
    for teammate_id in pair_agents:
        cand = assignments.get(teammate_id)
        if cand is not None:
            return cand

    return self._last_tactic_assignments_by_pair.get(pair_name)

def _get_nearest_visible_pair_threat(self, agent_id: str, x: float, y: float):
    enemy_ids = set(self._get_pair_enemy_ids(agent_id))
    if not enemy_ids:
        return None

    pair_threats = [threat for threat in self._get_picture_threats(purpose="decision") if threat.track_id in enemy_ids]
    if not pair_threats:
        return None

    return min(
        pair_threats,
        key=lambda threat: (threat.x - x) ** 2 + (threat.y - y) ** 2
    )

def _get_battlefield_pos_placeholder(self, aid: str) -> np.ndarray:
    """战术分配阶段使用已计算的friendlies位置占位（防止重复地理转换）。"""
    p = getattr(self, '_friendly_pos_cache_for_tactic', {}).get(aid)
    if p is not None:
        return np.array([float(p[0]), float(p[1])], dtype=float)
    return np.array([0.0, 0.0], dtype=float)

def _build_pair_node_status(self, env) -> str:
    """输出左右2v2对子各自目标节点，避免与全局最近距离节点混淆。"""
    if not hasattr(self, '_tactic_assignments_by_agent') or not self._tactic_assignments_by_agent:
        return ""

    pair_defs = {
        'L': ('A0100', 'A0200'),
        'R': ('A0300', 'A0400'),
    }
    items = []
    for tag, pair in pair_defs.items():
        asg = None
        for aid in pair:
            cand = self._tactic_assignments_by_agent.get(aid)
            if cand is not None:
                asg = cand
                break
        if asg is None or not asg.target_id or asg.shooter_id not in env.agents:
            continue
        if not env.agents[asg.shooter_id].is_alive:
            continue

        threat = None
        threat = self._get_picture_threat_by_id(asg.target_id, purpose="decision")
        if threat is None:
            continue

        # sx, sy = self._get_battlefield_pos(env, asg.shooter_id)
        # dist = float(np.sqrt((threat.x - sx) ** 2 + (threat.y - sy) ** 2))
        
        import pymap3d
        s_geo = env.agents[asg.shooter_id].get_geodetic()
        t_lon, t_lat = self.coord_sys.battlefield_to_geodetic(threat.x, threat.y)
        t_alt_m = threat.altitude * 1000.0
        n, e, d = pymap3d.geodetic2ned(t_lat, t_lon, t_alt_m, s_geo[1], s_geo[0], s_geo[2])
        dist = float(np.sqrt(n**2 + e**2 + d**2) / 1000.0)
        
        node = DEFAULT_RANGES.get_current_node(dist, getattr(self, '_locked_compression', None))
        node = self._lock_template_node(asg.shooter_id, asg.target_id, node)
        items.append(f"{tag}:{asg.target_id}@{node}/{dist:.0f}km")

    return " ".join(items)

def _log_situation(self, env, ctx: StateContext, cap_state: CAPState, current_time: float):
    """输出关键态势日志（简化版）"""
    if not self._initial_nodes_logged:
        self._initial_nodes_logged = True
        self._log_control_ranges_once(ctx.min_threat_distance)

    # [节点] 动态压缩逻辑 (移到日志前以确保状态正确)
    if not hasattr(self, '_compression_locked_at_distance'):
        self._compression_locked_at_distance = None
        self._locked_compression = None  # 确保初始化
        
    # 首次雷达探测到目标时锁定节点压缩
    total_radar_tracks = sum(len(self.cap_radar.get_tracks(aid)) for aid in self.cap_radar._radar_tracks)
    threat_count = len(self._get_picture_threats(current_time=current_time, purpose="decision"))
    
    if self._locked_compression is None and total_radar_tracks > 0:
        self._compression_locked_at_distance = ctx.min_threat_distance
        self._locked_compression = DEFAULT_RANGES.get_compressed_ranges(ctx.min_threat_distance)
        if self._locked_compression:
            # 🔥 关键修复：实例化为真正的 ControlRanges 对象，打通到 TacticalBridge 的传导链路
            self.ranges = ControlRanges(**self._locked_compression)
            
            ratio = min(1.0, ctx.min_threat_distance / 180.0)
            nodes_str = " ".join([f"{k}:{v:.0f}" for k, v in list(self._locked_compression.items())[:6]])
            _cap_table_print(f"[节点] 首次雷达探测@{ctx.min_threat_distance:.0f}km | 压缩锁定:{ratio:.0%} | {nodes_str}")

    # 判断雷达是否已开机（有任何飞机激活雷达且有雷达航迹）
    radar_range_entry_times = getattr(self, '_radar_range_entry_times', {})
    radar_is_on = bool(radar_range_entry_times) and total_radar_tracks > 0

    # 获取当前节点 (传入动态压缩后的范围)
    node = self.cap_state_machine.get_current_control_range(
        ctx.min_threat_distance, 
        self._locked_compression
    )
    # 🔥 修复：优先使用已锁定的最远阶段进度（避免EVADE时回显NLT_MELD）
    if hasattr(self, '_agent_template_node_progress') and self._agent_template_node_progress:
        # 取所有我方飞机中进度最远的阶段
        phase_order = ['BEYOND_NLT', 'NLT_MELD', 'MELD_MTR', 'MTR_LR', 'LR_TR', 'TR_DOR', 'DOR_DR', 'DR_MAR', 'BELOW_MAR']
        best_idx = -1
        for aid, progress_node in self._agent_template_node_progress.items():
            if aid.startswith('A') and progress_node in phase_order:
                idx = phase_order.index(progress_node)
                best_idx = max(best_idx, idx)
        if best_idx > 0:
            node = phase_order[best_idx]
    second_node = self._get_second_attack_subnode(ctx.min_threat_distance, self._locked_compression)
    if second_node:
        node = f"{node}/{second_node}"
    
    high_cnt = self._get_picture_zone_count(RiskZone.HIGH, current_time=current_time, purpose="decision")
    
    # [Coop] 协同探测详细日志
    mode = self.coop_detection.mode.value
    
    # 雷达状态诊断：以“是否已激活且有航迹”为准（避免仅按距离导致误判）
    radar_status_str = "ON" if radar_is_on else "OFF"
    
    # 添加has_awacs_info诊断
    awacs_flag = "预警机在线" if ctx.has_awacs_info else "预警机离线"
    # ---------- 改进综合状态展示 ----------
    # 理清距离概念：
    # - state_machine_dist: 用于CAP大阶段(INTERCEPT->ENGAGE->RTB)的综合威胁距离
    # - real_dist: 实际对子的真实距离（决定战术节点的NLT/MELD等）
    state_machine_dist = ctx.min_threat_distance
    
    # 提取真实的对子和对应的战术信息
    pair_nodes = self._build_pair_node_status(env)
    
    # 仅在 ENGAGE 阶段才显示战术和具体的战术节点信息
    def _normalize_tactic_name(name: str) -> str:
        name = str(name or '').strip()
        return '' if name.lower() in ('', 'unknown', 'none', '未决策') else name

    def _resolve_pair_tactic(agent_id: str) -> str:
        bridge_local = getattr(self, 'tactical_bridge', None)
        if bridge_local and getattr(bridge_local, 'initialized', False):
            tactic_name = _normalize_tactic_name(bridge_local.get_agent_tactic_info(agent_id).get('tactic', ''))
            if tactic_name:
                return tactic_name
        assign = self._get_pair_tactic_assignment(agent_id) if hasattr(self, '_get_pair_tactic_assignment') else None
        if assign is not None and getattr(assign, 'tactic', None) is not None:
            return str(assign.tactic.value)
        return ''

    tactic_display = ""
    if cap_state == CAPState.ENGAGE:
        bridge = getattr(self, 'tactical_bridge', None)
        if bridge and bridge.initialized:
            # 尝试从 bridge 获取两个对子的真实战术（legacy 引擎的决策）
            t1 = _resolve_pair_tactic('A0100')
            t2 = _resolve_pair_tactic('A0300')
            
            # 美化
            t1 = t1.lower() if t1 else '未知'
            t2 = t2.lower() if t2 else '未知'
            
            if t1 == t2:
                tactic_display = f" | 战术:{t1}"
            else:
                tactic_display = f" | 战术:A队={t1},B队={t2}"
        else:
            # 如果 bridge 还没有决策（或者初始化失败），获取 fallback
            last_assign = getattr(self, '_last_tactic_assignment', None)
            if last_assign:
                tactic_display = f" | 战术(fallback):{last_assign.tactic.value}"
    
    # 组合打印
    log_msg = f"[状态] [{current_time:.0f}s] {cap_state.value}{tactic_display} | 全局最小距离:{state_machine_dist:.1f}km | {awacs_flag} | 模式:{mode} | 威胁:{threat_count} (雷达:{total_radar_tracks}/{radar_status_str})"
    if pair_nodes:
        log_msg += f" | 对子:{pair_nodes}"
        
    log.info(log_msg)

    for aid in sorted(env.agents.keys()):
        if not aid.startswith('B') or not getattr(env.agents[aid], 'is_alive', False):
            continue
        try:
            bx, by = self._get_battlefield_pos(env, aid)
            zone = self.faor.get_risk_zone(float(bx), float(by))
        except Exception:
            bx, by = 0.0, 0.0
            zone = RiskZone.OUTSIDE
        zone_name = getattr(zone, 'value', str(zone))
        prev_zone = self._zone_transition_last.get(aid)
        if prev_zone != zone_name:
            self._zone_transition_last[aid] = zone_name
            self._zone_transition_history.setdefault(aid, []).append((float(current_time), zone_name))
            if prev_zone is None:
                log.info(
                    "[区域事件] t=%.0fs target=%s init=%s pos=(%.1f,%.1f)",
                    current_time,
                    aid,
                    zone_name,
                    bx,
                    by,
                )
            else:
                log.info(
                    "[区域事件] t=%.0fs target=%s %s->%s pos=(%.1f,%.1f)",
                    current_time,
                    aid,
                    prev_zone,
                    zone_name,
                    bx,
                    by,
                )

    if current_time - float(getattr(self, '_last_battle_snapshot_time', -999.0)) >= 60.0:
        zone_counts = {
            RiskZone.HIGH.value: 0,
            RiskZone.MEDIUM.value: 0,
            RiskZone.LOW.value: 0,
            RiskZone.OUTSIDE.value: 0,
        }
        for aid in env.agents:
            if not aid.startswith('B') or not getattr(env.agents[aid], 'is_alive', False):
                continue
            try:
                bx, by = self._get_battlefield_pos(env, aid)
                zone = self.faor.get_risk_zone(float(bx), float(by))
            except Exception:
                zone = RiskZone.OUTSIDE
            zone_counts[getattr(zone, 'value', RiskZone.OUTSIDE.value)] += 1

        def _team_alive(prefix: str) -> int:
            return sum(
                1
                for aid in env.agents
                if aid.startswith(prefix) and getattr(env.agents[aid], 'is_alive', False)
            )

        def _team_missiles(prefix: str) -> int:
            total = 0
            for aid, aircraft in env.agents.items():
                if not aid.startswith(prefix) or not getattr(aircraft, 'is_alive', False):
                    continue
                try:
                    total += int(getattr(aircraft, 'num_left_missiles'))
                except Exception:
                    total += int(getattr(aircraft, 'num_missiles', 0))
            return total

        def _normalize_tactic_name(name: str) -> str:
            name = str(name or '').strip()
            return '' if name.lower() in ('', 'unknown', 'none', '未决策') else name

        bridge = getattr(self, 'tactical_bridge', None)
        left_tactic = right_tactic = None
        try:
            if bridge and getattr(bridge, 'initialized', False):
                left_tactic = _normalize_tactic_name(bridge.get_agent_tactic_info('A0100').get('tactic'))
                right_tactic = _normalize_tactic_name(bridge.get_agent_tactic_info('A0300').get('tactic'))
        except Exception:
            left_tactic = right_tactic = None
        if hasattr(self, '_get_agent_tactic'):
            try:
                direct_left_tactic = _normalize_tactic_name(self._get_agent_tactic('A0100'))
                if direct_left_tactic:
                    left_tactic = direct_left_tactic
            except Exception:
                pass
            try:
                direct_right_tactic = _normalize_tactic_name(self._get_agent_tactic('A0300'))
                if direct_right_tactic:
                    right_tactic = direct_right_tactic
            except Exception:
                pass
        if not left_tactic and hasattr(self, '_get_pair_tactic_assignment'):
            left_assign = self._get_pair_tactic_assignment('A0100')
            if left_assign is not None and getattr(left_assign, 'tactic', None) is not None:
                left_tactic = str(left_assign.tactic.value)
        if not right_tactic and hasattr(self, '_get_pair_tactic_assignment'):
            right_assign = self._get_pair_tactic_assignment('A0300')
            if right_assign is not None and getattr(right_assign, 'tactic', None) is not None:
                right_tactic = str(right_assign.tactic.value)

        verify = getattr(self, '_guidance_verify', {}) or {}
        awacs_detected = len(getattr(self, '_target_first_awacs_time', {}) or {})
        fcr_detected = len(getattr(self, '_target_first_fcr_time', {}) or {})
        ready_targets = len(getattr(self, '_target_first_ready_time', {}) or {})
        log.info(
            "[BATTLE_SNAPSHOT] t=%.0fs A_alive=%d B_alive=%d A_ms=%d B_ms=%d "
            "zones(H/M/L/O)=%d/%d/%d/%d tactic(L/R)=%s/%s tracking=%s relay=%s gate=%s/%s relay_ok=%s/%s sense(A/F/R)=%d/%d/%d",
            current_time,
            _team_alive('A'),
            _team_alive('B'),
            _team_missiles('A'),
            _team_missiles('B'),
            zone_counts[RiskZone.HIGH.value],
            zone_counts[RiskZone.MEDIUM.value],
            zone_counts[RiskZone.LOW.value],
            zone_counts[RiskZone.OUTSIDE.value],
            left_tactic or '-',
            right_tactic or '-',
            "ON" if getattr(self, '_cooperative_tracking_active', False) else "OFF",
            "ON" if getattr(self, '_relay_guidance_active', False) else "OFF",
            int(verify.get('prelaunch_gate_pass_count', 0)),
            int(verify.get('prelaunch_gate_block_count', 0)),
            int(verify.get('relay_success_count', 0)),
            int(verify.get('relay_attempt_count', 0)),
            awacs_detected,
            fcr_detected,
            ready_targets,
        )
        self._last_battle_snapshot_time = current_time
    
    # [Verify] 真值对比 (Ground Truth Verification)
    # 获取所有存活敌机的真实位置
    true_positions = {}
    for aid in env.agents:
        if aid.startswith('B') and env.agents[aid].is_alive:
            pos = self._get_battlefield_pos(env, aid)
            true_positions[aid] = pos
    
    # radar_is_on 已在上方计算
    
    # 详细对比表仅在协同探测阶段打印，避免交战后刷屏
    show_verify_table = (
        current_time % 60 < 1.0
        and true_positions
        and threat_count > 0
        and cap_state in (CAPState.PATROL, CAPState.INTERCEPT)
        and not radar_is_on
    )
    if show_verify_table:
        # 根据雷达状态选择表头
        if radar_is_on:
            header = f"\n[对比] ===== 真值 vs 融合位置对比表 (t={current_time:.0f}s) [雷达已开机] ====="
            col_label = "融合"
        else:
            header = f"\n[对比] ===== 真值 vs 预警机位置对比表 (t={current_time:.0f}s) [仅预警机] ====="
            col_label = "预警机"
        
        _cap_table_print(header)
        _cap_table_print(f"{'目标ID':<8} {'真实X':>8} {'真实Y':>8} {col_label+'X':>8} {col_label+'Y':>8} {'误差km':>8} {'首探延迟':>12}")
        _cap_table_print("-" * 85)
        
        total_error = 0
        matched_count = 0
        for tid, true_pos in true_positions.items():
            # 查找对应的融合航迹
            fused_track = None
            for t in self._get_picture_threats(current_time=current_time, purpose="decision"):
                if t.track_id == tid:
                    fused_track = t
                    break
            
            if fused_track:
                error = np.linalg.norm(np.array([fused_track.x, fused_track.y]) - np.array(true_pos[:2]))
                
                # 🔥 修复：只显示最快的探测延迟
                detect_str = "未探测"
                if tid in self._first_radar_detection_time:
                    min_delay = float('inf')
                    fastest_agent = None
                    
                    for aid, first_detect in self._first_radar_detection_time[tid].items():
                        # 获取该飞机的雷达激活时间
                        if aid in self._radar_range_entry_times:
                            activation_time = self._radar_range_entry_times[aid]
                            delay = first_detect - activation_time
                            # 修复负数和-0问题
                            if delay < 0.1:
                                delay = max(0.2, delay)
                            
                            if delay < min_delay:
                                min_delay = delay
                                fastest_agent = aid
                    
                    if fastest_agent:
                        detect_str = f"{fastest_agent[-4:]}:{min_delay:.1f}s"
                
                _cap_table_print(f"{tid:<8} {true_pos[0]:>8.1f} {true_pos[1]:>8.1f} {fused_track.x:>8.1f} {fused_track.y:>8.1f} {error:>8.2f} {detect_str:>12}")
                total_error += error
                matched_count += 1
            else:
                _cap_table_print(f"{tid:<8} {true_pos[0]:>8.1f} {true_pos[1]:>8.1f} {'--':>8} {'--':>8} {'--':>8} {'无数据':>12}")
        
        _cap_table_print("-" * 85)
        if matched_count > 0:
            avg_err = total_error / matched_count
            quality = "[优]" if avg_err < 3.0 else ("[良]" if avg_err < 5.0 else "[差]")
            
            # 🔥 修复：显示每架飞机的探测效率摘要
            if radar_is_on and self._first_radar_detection_time:
                _cap_table_print(f"\n[探测效率摘要] 误差: {avg_err:.2f}km {quality}")
                _cap_table_print(f"{'飞机ID':<8} {'激活时间':>10} {'首探延迟':>10} {'全探延迟':>10} {'探测目标数':>12}")
                _cap_table_print("-" * 60)
                
                for aid in sorted(self._radar_range_entry_times.keys()):
                    activation = self._radar_range_entry_times[aid]
                    
                    # 计算该飞机的首探和全探延迟
                    delays = []
                    for tid in self._first_radar_detection_time:
                        if aid in self._first_radar_detection_time[tid]:
                            first_detect = self._first_radar_detection_time[tid][aid]
                            delay = max(0.2, first_detect - activation)
                            delays.append(delay)
                    
                    if delays:
                        first_delay = min(delays)
                        last_delay = max(delays)
                        target_count = len(delays)
                        _cap_table_print(f"{aid:<8} {activation:>10.1f}s {first_delay:>9.1f}s {last_delay:>9.1f}s {target_count:>12}")
                    else:
                        _cap_table_print(f"{aid:<8} {activation:>10.1f}s {'--':>10} {'--':>10} {0:>12}")
                
                _cap_table_print()
            else:
                _cap_table_print(f"[预警机] 仅预警机数据 | 误差: {avg_err:.2f}km {quality}")
        _cap_table_print()

def _update_mission_evaluation(self, env, current_time: float):
    """更新任务评估"""
    # 统计我方存活数
    friendly_alive = sum(1 for aid in env.agents 
                        if aid.startswith('A') and env.agents[aid].is_alive)
    
    # 更新评估器
    self.mission_evaluator.update(
        self.picture,
        friendly_alive,
        current_time - self.mission_start_time,
        current_time=current_time,
        track_filter_kwargs=self._get_picture_filter_kwargs("decision"),
    )
    
    # 检查任务结果（只在状态变化时输出日志）
    result = self.mission_evaluator.evaluate()
    if not hasattr(self, '_last_mission_result'):
        self._last_mission_result = result
    
    if result != self._last_mission_result:
        if result == MissionResult.FAILURE:
            log.warning(f"[评估] 任务失败! 原因: 损失过大或高风险区被突破")
        elif result == MissionResult.SUCCESS:
            log.info(f"[评估] 任务成功完成!")
        self._last_mission_result = result

def _update_tactic_selection(self, env, current_time: float):
    """更新战术选择 - 仅在 ENGAGE 状态下运行"""
    # 只在 ENGAGE 状态进行战术选择（INTERCEPT 阶段是协同探测，不需要战术模板）
    if self.cap_state_machine.state != CAPState.ENGAGE:
        self._tactic_assignments_by_agent = {}
        return
    
    # 获取我方位置
    friendly_positions = {}
    for aid in env.agents:
        if aid.startswith('A') and env.agents[aid].is_alive:
            ac = env.agents[aid]
            lon = ac.get_property_value(c.position_long_gc_deg)
            lat = ac.get_property_value(c.position_lat_geod_deg)
            alt_ft = ac.get_property_value(c.position_h_sl_ft)
            x, y = self.coord_sys.geodetic_to_battlefield(lon, lat)
            z = alt_ft * 0.3048 / 1000.0
            friendly_positions[aid] = np.array([x, y, z])
    
    threats = self._get_picture_threats(current_time=current_time, purpose="decision")
    threat_source = "decision"
    if not threats:
        fallback_threats = self._get_picture_threats(current_time=current_time, purpose="search")
        if fallback_threats:
            threats = fallback_threats
            threat_source = "search"
            if self.step_count % 20 == 0:
                log.warning(
                    "[TACTIC_SELECT_FALLBACK] t=%.0fs decision=0 -> use search=%d",
                    current_time,
                    len(threats),
                )
        else:
            self._tactic_assignments_by_agent = {}
            self._last_tactic_assignment = None
            return

    # 供"目标粘滞选择器"使用
    self._friendly_pos_cache_for_tactic = friendly_positions

    # 全局战术类型：仅作为 fallback（底层旧引擎会自主选择真实战术）
    global_assignment = self.tactic_selector.select_tactic(
        self.picture, friendly_positions, current_time, threats=threats
    )
    tactic = global_assignment.tactic if global_assignment else TacticType.T_FB

    # 调试/验证：支持强制指定战术模板，便于对照旧 run_simulation 行为
    forced = os.environ.get('CAP_FORCE_TACTIC', '').strip().lower()
    forced_map = {
        'drag_shoot': TacticType.T_DS,
        't_ds': TacticType.T_DS,
        'pincer_attack': TacticType.T_PA,
        'pincer': TacticType.T_PA,
        't_pa': TacticType.T_PA,
        'high_low': TacticType.T_HL,
        'high_low_attack': TacticType.T_HL,
        't_hl': TacticType.T_HL,
        'side_by_side': TacticType.T_SBS,
        'sbs': TacticType.T_SBS,
        't_sbs': TacticType.T_SBS,
        'front_back': TacticType.T_FB,
        'fb': TacticType.T_FB,
        't_fb': TacticType.T_FB,
        'tactical_evasion': TacticType.T_TE,
        'te': TacticType.T_TE,
        't_te': TacticType.T_TE,
        'tactical_turn': TacticType.T_TT,
        'tt': TacticType.T_TT,
        't_tt': TacticType.T_TT,
    }
    if forced in forced_map:
        tactic = forced_map[forced]
        if self.step_count % 120 == 0:
            log.info(f"  强制战术: {tactic.value} (环境变量CAP_FORCE_TACTIC={forced})")

    pair_defs = {
        'left': {
            'friends': ['A0100', 'A0200'],
            'enemies': ['B0100', 'B0200'],
        },
        'right': {
            'friends': ['A0300', 'A0400'],
            'enemies': ['B0300', 'B0400'],
        },
    }

    for pair_name, pair_info in pair_defs.items():
        if not any(
            enemy_id in env.agents and env.agents[enemy_id].is_alive
            for enemy_id in pair_info['enemies']
        ):
            self._last_tactic_assignments_by_pair.pop(pair_name, None)
            self._pair_target_memory.pop(pair_name, None)
            self._pair_target_lock_until_step.pop(pair_name, None)

    pair_targets = {}
    threat_by_id = {threat.track_id: threat for threat in threats}

    for pair_name, pair_info in pair_defs.items():
        pair_agents = pair_info['friends']
        enemy_ids = pair_info['enemies']
        alive_pair = [aid for aid in pair_agents if aid in friendly_positions]
        if not alive_pair:
            continue

        candidate_threats = [threat_by_id[enemy_id] for enemy_id in enemy_ids if enemy_id in threat_by_id]
        if not candidate_threats:
            continue

        best_threat = self._pick_pair_target_with_stickiness(pair_name, alive_pair, candidate_threats)
        if best_threat is None:
            continue
        pair_targets[pair_name] = best_threat

    tactic_signature = (
        getattr(global_assignment.tactic, "value", None) if global_assignment else None,
        tuple(sorted((pair_name, threat.track_id) for pair_name, threat in pair_targets.items())),
        len(threats),
        threat_source,
    )
    if _should_emit_tactic_log(self, current_time, tactic_signature, interval_s=18.0):
        log.info(
            "[TACTIC_SELECT] t=%.0fs threats=%d src=%s global=%s left=%s right=%s",
            current_time,
            len(threats),
            threat_source,
            getattr(getattr(global_assignment, "tactic", None), "value", "-"),
            getattr(pair_targets.get("left"), "track_id", "-"),
            getattr(pair_targets.get("right"), "track_id", "-"),
        )

    assignments_by_agent: Dict[str, TacticAssignment] = {}
    primary_assignment = None
    for pair_name, pair_info in pair_defs.items():
        pair_agents = pair_info['friends']
        threat = pair_targets.get(pair_name)
        alive_pair = [aid for aid in pair_agents if aid in friendly_positions]
        if threat is None or not alive_pair:
            continue

        # 🔥 动态反查底层的真实战术分配（真正的意图识别与战术决策在旧版算法里计算）
        pair_tactic = tactic
        bridge = getattr(self, 'tactical_bridge', None)
        if bridge and bridge.initialized:
            real_tactic_str = bridge.get_agent_tactic_info(alive_pair[0]).get('tactic', '')
            tactic_map = {
                'DRAG_SHOOT': TacticType.T_DS,
                'PINCER_ATTACK': TacticType.T_PA,
                'HIGH_LOW_ATTACK': TacticType.T_HL,
                'SIDE_BY_SIDE': TacticType.T_SBS,
                'FRONT_BACK': TacticType.T_FB,
                'TACTICAL_EVASION': TacticType.T_TE,
                'TACTICAL_TURN': TacticType.T_TT,
            }
            pair_enemy_alive = any(
                enemy_id in env.agents and env.agents[enemy_id].is_alive
                for enemy_id in enemy_ids
            )
            if real_tactic_str == 'TACTICAL_TURN' and any(aid.startswith('A') for aid in alive_pair) and pair_enemy_alive:
                if self.step_count % 25 == 0:
                    log.warning(f"🛑 [CAP战术返航拦截-{pair_name}] bridge tactic=TACTICAL_TURN 但敌机仍存活，保持CAP选择战术{pair_tactic.value}")
            elif real_tactic_str in tactic_map:
                pair_tactic = tactic_map[real_tactic_str]

        sorted_pair = sorted(
            alive_pair,
            key=lambda aid: np.linalg.norm(friendly_positions[aid][:2] - np.array([threat.x, threat.y], dtype=float))
        )
        shooter = sorted_pair[0]
        support = sorted_pair[1] if len(sorted_pair) > 1 else None
        pair_assignment = TacticAssignment(
            tactic=pair_tactic,
            target_id=threat.track_id,
            shooter_id=shooter,
            support_id=support,
            priority=global_assignment.priority if global_assignment else 1,
        )
        self._last_tactic_assignments_by_pair[pair_name] = pair_assignment
        assignments_by_agent[shooter] = pair_assignment
        if support:
            assignments_by_agent[support] = pair_assignment
        if primary_assignment is None:
            primary_assignment = pair_assignment

    self._tactic_assignments_by_agent = assignments_by_agent
    self._last_tactic_assignment = primary_assignment
