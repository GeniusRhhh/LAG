"""
导弹管理模块
管理导弹发射、追踪、状态更新等功能
从tactical_task.py中提取，提高代码可维护性
"""
import logging
import numpy as np
from envs.JSBSim.core.catalog import Catalog as c
from tactical_types import TacticalPhase, get_target_with_fallback
from simulation.radar_manager import check_missile_launch_conditions


class MissileManager:
    """导弹管理器 - 负责导弹相关操作"""
    
    def __init__(self, state_manager):
        """
        Args:
            state_manager: TacticalStateManager实例
        """
        self.state_manager = state_manager
        self.missile_tracks = {}  # 导弹追踪信息
        self.aircraft_missile_counts = {}  # 每架飞机的导弹发射计数 {agent_id: fired_count}
        self.MAX_MISSILES_PER_AIRCRAFT = 4  # 每架飞机最大导弹数量
        # 最近一次“发射门限判定”记录（供真实发射事件复盘引用）
        # {agent_id: {allowed, reason, target_id, phase, time_s, details}}
        self.last_gate_decision = {}

    def _get_aircraft_missiles_left(self, aircraft) -> int:
        try:
            if hasattr(aircraft, 'num_left_missiles'):
                return max(0, int(getattr(aircraft, 'num_left_missiles')))
        except Exception:
            pass
        return max(0, int(getattr(aircraft, 'num_missiles', 0)))

    def _count_alive_enemy_aircraft(self, env, agent_id: str) -> int:
        enemy_prefix = 'B' if str(agent_id).startswith('A') else 'A'
        return sum(
            1
            for other_id, other_aircraft in getattr(env, 'agents', {}).items()
            if str(other_id).startswith(enemy_prefix) and getattr(other_aircraft, 'is_alive', False)
        )

    def _get_team_remaining_missiles(self, env, agent_id: str) -> int:
        team_prefix = str(agent_id)[:1]
        total = 0
        for other_id, other_aircraft in getattr(env, 'agents', {}).items():
            if not str(other_id).startswith(team_prefix) or not getattr(other_aircraft, 'is_alive', False):
                continue
            total += self._get_aircraft_missiles_left(other_aircraft)
        return total

    def _get_team_fired_missiles(self, agent_id: str) -> int:
        team_prefix = str(agent_id)[:1]
        total = 0
        for other_id, fired_count in self.aircraft_missile_counts.items():
            if str(other_id).startswith(team_prefix):
                total += max(0, int(fired_count))
        return total

    def _resolve_target_aircraft(self, env, target_id: str = None, target_aircraft=None):
        if target_aircraft is not None:
            return target_aircraft
        if target_id:
            for collection_name in ("agents", "_jsbsims"):
                collection = getattr(env, collection_name, None)
                if isinstance(collection, dict) and target_id in collection:
                    return collection.get(target_id)
        return None

    def _resolve_target_aliases(self, env, target_id: str = None, target_aircraft=None) -> set:
        aliases = set()
        if isinstance(target_id, str) and target_id:
            aliases.add(target_id)
        target_aircraft = self._resolve_target_aircraft(env, target_id=target_id, target_aircraft=target_aircraft)
        if target_aircraft is not None:
            for attr_name in ("real_id", "uid", "target_id"):
                value = getattr(target_aircraft, attr_name, None)
                if isinstance(value, str) and value:
                    aliases.add(value)
        return aliases

    def _resolve_track_target_id(self, env, target_id: str = None, target_aircraft=None) -> str:
        target_aircraft = self._resolve_target_aircraft(env, target_id=target_id, target_aircraft=target_aircraft)
        if target_aircraft is not None:
            for attr_name in ("real_id", "uid", "target_id"):
                value = getattr(target_aircraft, attr_name, None)
                if isinstance(value, str) and value:
                    return value
        return str(target_id or "")

    def _get_target_battlefield_y_km(self, env, target_id: str = None, target_aircraft=None) -> float:
        target_aircraft = self._resolve_target_aircraft(env, target_id=target_id, target_aircraft=target_aircraft)
        if target_aircraft is None:
            return float("nan")

        task = getattr(env, "task", None)
        if task is not None and hasattr(task, "coord_sys"):
            try:
                geo = target_aircraft.get_geodetic()
                _, y_km = task.coord_sys.geodetic_to_battlefield(geo[0], geo[1])
                return float(y_km)
            except Exception:
                pass

        try:
            pos = target_aircraft.get_position()
            return float(pos[0]) / 1000.0
        except Exception:
            return float("nan")

    def _classify_target_risk_zone(self, env, target_id: str = None, target_aircraft=None) -> tuple[str, float]:
        y_km = self._get_target_battlefield_y_km(env, target_id=target_id, target_aircraft=target_aircraft)
        if not np.isfinite(y_km):
            return "UNKNOWN", y_km
        if y_km < 0.0 or y_km > 300.0:
            return "OUTSIDE", y_km
        if y_km < 100.0:
            return "HIGH", y_km
        if y_km < 200.0:
            return "MEDIUM", y_km
        return "LOW", y_km

    def _get_coop_prelaunch_state(
        self,
        env,
        target_id: str,
        current_time: float,
        shooter_id: str = None,
        distance_km: float = None,
        phase_name: str = "",
        is_second_attack: bool = False,
        target_zone: str = None,
    ) -> tuple[bool, dict]:
        owner_task = getattr(env, "task", None)
        if owner_task is None or not hasattr(owner_task, "check_external_prelaunch_gate"):
            return False, {}
        try:
            passed, details = owner_task.check_external_prelaunch_gate(
                target_id,
                current_time,
                shooter_id=shooter_id,
                distance_km=distance_km,
                phase_name=phase_name,
                is_second_attack=is_second_attack,
                target_zone=target_zone,
            )
            if not isinstance(details, dict):
                details = {}
            return bool(passed), details
        except Exception:
            return False, {}

    def _get_friendly_launch_window_policy(
        self,
        current_phase,
        *,
        is_second_attack: bool,
        priority_target: bool,
        coop_prelaunch_ready: bool,
        radar_passed: bool,
        radar_locked: bool,
        heading_error: float,
        active_on_target: int,
        effective_active_on_target: int,
        team_fired_count: int,
    ) -> dict:
        phase = current_phase
        policy = {
            "phase_allowed": False,
            "phase_role": "blocked",
            "phase_reason": "phase_not_authorized",
            "window_low_m": 0.0,
            "window_high_m": 0.0,
            "window_heading_limit": 0.0,
            "hard_backwards_limit": 108.0 if not is_second_attack else 114.0,
            "no_lock_backwards_limit": 74.0 if not is_second_attack else 82.0,
        }

        if is_second_attack:
            policy["phase_allowed"] = phase in (
                TacticalPhase.LR_TR,
                TacticalPhase.TR_DOR,
                TacticalPhase.DOR_DR,
                TacticalPhase.DR_MAR,
            )
            policy["phase_role"] = "second_attack"
            policy["phase_reason"] = "second_attack_window"
            if phase == TacticalPhase.LR_TR:
                policy["window_low_m"] = 42000.0
                policy["window_high_m"] = 66000.0 if coop_prelaunch_ready else 64000.0
                policy["window_heading_limit"] = 50.0 if radar_locked else 46.0
            elif phase == TacticalPhase.TR_DOR:
                policy["window_low_m"] = 40000.0
                policy["window_high_m"] = 64000.0 if coop_prelaunch_ready else 62000.0
                policy["window_heading_limit"] = 52.0 if radar_locked else 48.0
            elif phase == TacticalPhase.DOR_DR:
                policy["window_low_m"] = 38000.0
                policy["window_high_m"] = 60000.0 if coop_prelaunch_ready else 58000.0
                policy["window_heading_limit"] = 54.0 if radar_locked else 50.0
            elif phase == TacticalPhase.DR_MAR:
                policy["window_low_m"] = 34000.0 if coop_prelaunch_ready else 36000.0
                policy["window_high_m"] = 56000.0 if coop_prelaunch_ready else 52000.0
                policy["window_heading_limit"] = 56.0 if radar_locked else 52.0
            return policy

        if phase == TacticalPhase.LR_TR:
            policy["phase_allowed"] = True
            policy["phase_role"] = "primary_lr"
            policy["phase_reason"] = "primary_lr_window"
            policy["window_low_m"] = 50000.0 if priority_target else 52000.0
            if coop_prelaunch_ready and radar_passed and radar_locked and heading_error <= 32.0:
                policy["window_high_m"] = 80000.0 if priority_target else 79000.0
                policy["window_heading_limit"] = 50.0 if radar_locked else 46.0
            else:
                policy["window_high_m"] = 78000.0 if priority_target else 76000.0
                policy["window_heading_limit"] = 46.0 if radar_locked else 44.0
            policy["hard_backwards_limit"] = 110.0 if coop_prelaunch_ready else 106.0
            policy["no_lock_backwards_limit"] = 76.0 if coop_prelaunch_ready else 72.0
            return policy

        if phase == TacticalPhase.TR_DOR:
            late_primary_ready = bool(priority_target or coop_prelaunch_ready or radar_locked)
            policy["phase_allowed"] = late_primary_ready
            policy["phase_role"] = "late_primary" if late_primary_ready else "blocked"
            policy["phase_reason"] = "late_primary_recovery" if late_primary_ready else "late_primary_not_ready"
            if late_primary_ready:
                policy["window_low_m"] = 46000.0 if priority_target else 48000.0
                policy["window_high_m"] = 72000.0 if priority_target else 70000.0
                if coop_prelaunch_ready and radar_passed and heading_error <= 32.0:
                    policy["window_high_m"] += 2000.0
                policy["window_heading_limit"] = 48.0 if radar_locked else 44.0
                policy["hard_backwards_limit"] = 112.0 if coop_prelaunch_ready else 108.0
                policy["no_lock_backwards_limit"] = 78.0 if coop_prelaunch_ready else 74.0
            return policy

        if phase == TacticalPhase.DOR_DR:
            close_followup_ready = bool(
                priority_target and (coop_prelaunch_ready or active_on_target > 0 or effective_active_on_target > 0 or team_fired_count > 0)
            )
            policy["phase_allowed"] = close_followup_ready
            policy["phase_role"] = "close_followup" if close_followup_ready else "blocked"
            policy["phase_reason"] = "close_followup_window" if close_followup_ready else "close_followup_not_ready"
            if close_followup_ready:
                policy["window_low_m"] = 44000.0 if priority_target else 46000.0
                policy["window_high_m"] = 62000.0 if coop_prelaunch_ready else 60000.0
                policy["window_heading_limit"] = 46.0 if radar_locked else 42.0
                policy["hard_backwards_limit"] = 114.0 if coop_prelaunch_ready else 110.0
                policy["no_lock_backwards_limit"] = 80.0 if coop_prelaunch_ready else 76.0
            return policy

        if phase == TacticalPhase.DR_MAR:
            close_commit_ready = bool(
                priority_target and (coop_prelaunch_ready or active_on_target > 0 or effective_active_on_target > 0 or team_fired_count > 0)
            )
            policy["phase_allowed"] = close_commit_ready
            policy["phase_role"] = "close_commit" if close_commit_ready else "blocked"
            policy["phase_reason"] = "close_commit_window" if close_commit_ready else "close_commit_not_ready"
            if close_commit_ready:
                policy["window_low_m"] = 40000.0 if coop_prelaunch_ready else 42000.0
                policy["window_high_m"] = 56000.0 if coop_prelaunch_ready else 54000.0
                policy["window_heading_limit"] = 44.0 if radar_locked else 40.0
                policy["hard_backwards_limit"] = 116.0 if coop_prelaunch_ready else 112.0
                policy["no_lock_backwards_limit"] = 82.0 if coop_prelaunch_ready else 78.0
            return policy

        return policy

    def _get_effective_missiles_targeting(
        self,
        target_id: str,
        env=None,
        current_time: float = 0.0,
        max_age_s: float | None = None,
    ) -> int:
        target_aliases = {str(target_id)} if target_id else set()
        if env is not None:
            target_aliases |= self._resolve_target_aliases(env, target_id=str(target_id) if target_id else None)
        count = 0
        for track in self.missile_tracks.values():
            if track.get("status") != "active":
                continue
            aliases = set(track.get("target_aliases", []))
            aliases.add(str(track.get("target", "")))
            aliases.add(str(track.get("target_key", "")))
            if not (aliases & target_aliases):
                continue
            if max_age_s is not None:
                launch_time = float(track.get("launch_time", current_time))
                if (float(current_time) - launch_time) > float(max_age_s):
                    continue
            count += 1
        return count

    def record_external_missile_launch(
        self,
        env,
        launcher_id: str,
        missile_id: str,
        target_id: str = None,
        target_aircraft=None,
        current_time: float = 0.0,
    ) -> None:
        if not missile_id:
            return
        track_target_id = self._resolve_track_target_id(env, target_id=target_id, target_aircraft=target_aircraft)
        aliases = sorted(self._resolve_target_aliases(env, target_id=target_id, target_aircraft=target_aircraft))
        self.missile_tracks[str(missile_id)] = {
            "launcher": str(launcher_id),
            "target": str(track_target_id),
            "target_key": str(target_id or track_target_id),
            "target_aliases": aliases,
            "launch_time": float(current_time),
            "status": "active",
        }
    
    def should_launch_missile(self, env, agent_id: str, current_phase, target_id: str = None, 
                           salvo_mode: str = "single", is_second_attack: bool = False, 
                           return_reason: bool = False) -> tuple:
        """
        判断是否应该发射导弹 - 支持分批发射 + 严格4导弹限制
        
        Args:
            env: 环境
            agent_id: 飞机ID
            current_phase: 当前战术阶段
            target_id: 目标ID
            salvo_mode: 发射模式 ("single", "double", "salvo")
            
        Returns:
            True表示应该发射
        """

        def _phase_str(p):
            try:
                return p.value if hasattr(p, 'value') else str(p)
            except Exception:
                return None

        def _trace_gate(*, allowed: bool, reason: str, details: dict):
            # 只写文件，不污染控制台；变化触发 + 兜底节流
            try:
                from utils.trace_logger import trace_if_changed, trace_throttle

                # 记录最近一次门限判定，供“真实发射事件”补齐正向理由链
                try:
                    ti = float(getattr(env, 'time_interval', 0.2) or 0.2)
                    step = int(getattr(env, 'current_step', 0) or 0)
                    now_s = float(step) * float(ti)
                except Exception:
                    now_s = None
                try:
                    self.last_gate_decision[agent_id] = {
                        "allowed": bool(allowed),
                        "reason": str(reason),
                        "target_id": str(target_id) if target_id else None,
                        "phase": _phase_str(current_phase),
                        "time_s": float(now_s) if now_s is not None else None,
                        "details": details,
                    }
                except Exception:
                    pass

                phase_s = _phase_str(current_phase)
                sig = (bool(allowed), str(reason))
                trace_if_changed(
                    key=f"missile_gate:sig:{agent_id}:{target_id}",
                    value=sig,
                    标题="原因链-导弹发射门限",
                    env=env,
                    模块="missile_manager",
                    类型="GATE",
                    状态="ALLOW" if allowed else "DENY",
                    我机=agent_id,
                    敌机=str(target_id) if target_id else None,
                    阶段=phase_s,
                    战术=str(getattr(getattr(env, 'task', None), 'selected_tactic', None)),
                    说明="发射门限结果发生变化",
                    要点=[
                        f"allowed={allowed}",
                        f"reason={reason}",
                    ],
                    数据=details,
                )

                # 同时每60步给一个摘要（避免完全无日志）
                trace_throttle(
                    key=f"missile_gate:throttle:{agent_id}:{target_id}",
                    min_steps=60,
                    标题="原因链-导弹发射门限(摘要)",
                    env=env,
                    模块="missile_manager",
                    类型="GATE",
                    状态="ALLOW" if allowed else "DENY",
                    我机=agent_id,
                    敌机=str(target_id) if target_id else None,
                    阶段=phase_s,
                    战术=str(getattr(getattr(env, 'task', None), 'selected_tactic', None)),
                    说明="周期性摘要（节流）",
                    要点=[f"allowed={allowed} reason={reason}"],
                    数据=details,
                )
            except Exception:
                pass

        def _get_radar_status() -> dict:
            status = {
                "radar_state": None,
                "lock_target": None,
                "rwr_level": None,
                "ecm_active": None,
                "ecm_type": None,
                "ecm_duration": None,
            }
            try:
                from simulation.radar_manager import get_unified_radar_manager
                rm = get_unified_radar_manager()
                if rm is None:
                    return status

                # 雷达状态（SEARCH/TRACK/LOCK/...）
                try:
                    if agent_id.startswith('A') and hasattr(rm, 'friendly_radar_states'):
                        st = rm.friendly_radar_states.get(agent_id)
                    elif agent_id.startswith('B') and hasattr(rm, 'enemy_radar_states'):
                        st = rm.enemy_radar_states.get(agent_id)
                    else:
                        st = None
                    status["radar_state"] = st.value if hasattr(st, 'value') else (str(st) if st is not None else None)
                except Exception:
                    pass

                # 锁定目标
                try:
                    if agent_id.startswith('A') and hasattr(rm, 'friendly_lock_targets'):
                        status["lock_target"] = rm.friendly_lock_targets.get(agent_id)
                    elif agent_id.startswith('B') and hasattr(rm, 'enemy_lock_targets'):
                        status["lock_target"] = rm.enemy_lock_targets.get(agent_id)
                except Exception:
                    pass

                # RWR 威胁等级
                try:
                    if hasattr(rm, 'get_rwr_threat_level'):
                        status["rwr_level"] = int(rm.get_rwr_threat_level(agent_id))
                except Exception:
                    pass

                # ECM 状态
                try:
                    if hasattr(rm, 'ecm_states'):
                        es = rm.ecm_states.get(agent_id)
                        if isinstance(es, dict):
                            status["ecm_active"] = bool(es.get('active', False))
                            et = es.get('type')
                            status["ecm_type"] = et.value if hasattr(et, 'value') else (str(et) if et is not None else None)
                            status["ecm_duration"] = float(es.get('duration', 0.0)) if es.get('duration') is not None else None
                except Exception:
                    pass

            except Exception:
                return status
            return status

        # **关键修复：严格检查4导弹限制**
        if agent_id not in self.aircraft_missile_counts:
            self.aircraft_missile_counts[agent_id] = 0
        
        fired_count = self.aircraft_missile_counts[agent_id]
        
        # 根据发射模式检查导弹数量要求
        if salvo_mode == "single":
            required_missiles = 1
        elif salvo_mode == "double":
            required_missiles = 2
        elif salvo_mode == "salvo":
            required_missiles = min(2, self.MAX_MISSILES_PER_AIRCRAFT - fired_count)  # 最多一次性发射2枚
        else:
            required_missiles = 1
        
        # **关键检查：确保不超过4枚导弹限制**
        if fired_count + required_missiles > self.MAX_MISSILES_PER_AIRCRAFT:
            logging.debug(f"⛔ [{agent_id}] 导弹已达上限: 已发射{fired_count}枚, 需要{required_missiles}枚, 上限{self.MAX_MISSILES_PER_AIRCRAFT}枚")
            _trace_gate(
                allowed=False,
                reason="quota_exceeded",
                details={
                    "已发射": int(fired_count),
                    "本次需要": int(required_missiles),
                    "上限": int(self.MAX_MISSILES_PER_AIRCRAFT),
                    "salvo_mode": str(salvo_mode),
                },
            )
            return False
        
        # 检查剩余导弹
        current_time = env.current_step * env.time_interval
        aircraft = env.agents[agent_id]
        radar_status = _get_radar_status()

        # ✅ 统一剩弹口径：优先使用 num_left_missiles（AircraftSimulator.weapon_status 也来源于它）
        missiles_left = self._get_aircraft_missiles_left(aircraft)
        
        if missiles_left < required_missiles:
            logging.debug(f"⛔ [{agent_id}] 导弹不足: 需要{required_missiles}枚, 剩余{aircraft.num_missiles}枚")
            _trace_gate(
                allowed=False,
                reason="insufficient_missiles",
                details={
                    "需要": int(required_missiles),
                    "剩余": int(missiles_left),
                    "剩余口径": "num_left_missiles" if hasattr(aircraft, 'num_left_missiles') else "num_missiles",
                    **radar_status,
                },
            )
            return False
        
        if not self.state_manager.can_launch_missile(agent_id, current_time):
            _trace_gate(
                allowed=False,
                reason="cooldown_or_state_blocked",
                details={
                    "time_s": float(current_time),
                    "备注": "state_manager.can_launch_missile=False",
                    **radar_status,
                },
            )
            return False
        
        # 获取目标
        if target_id is None:
            target_id = get_target_with_fallback(agent_id, env)
        
        if target_id is None:
            _trace_gate(
                allowed=False,
                reason="no_target",
                details={
                    "备注": "get_target_with_fallback返回None",
                    **radar_status,
                },
            )
            return False
        
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft is None or not target_aircraft.is_alive:
            _trace_gate(
                allowed=False,
                reason="target_dead_or_missing",
                details={
                    "目标": str(target_id),
                    "target_exists": bool(target_aircraft is not None),
                    "target_alive": bool(getattr(target_aircraft, 'is_alive', False)) if target_aircraft is not None else False,
                    **radar_status,
                },
            )
            return False
        
        # 计算距离
        from tactical_utils import TacticalUtils
        distance = TacticalUtils.calculate_distance_between(env.agents[agent_id], target_aircraft)
        track_target_id = self._resolve_track_target_id(env, target_id=target_id, target_aircraft=target_aircraft)
        alive_enemy_count = self._count_alive_enemy_aircraft(env, agent_id)
        team_missiles_left = self._get_team_remaining_missiles(env, agent_id)
        team_fired_count = self._get_team_fired_missiles(agent_id)
        target_risk_zone, target_y_km = self._classify_target_risk_zone(
            env,
            target_id=target_id,
            target_aircraft=target_aircraft,
        )
        priority_target = target_risk_zone in ("HIGH", "MEDIUM")
        coop_prelaunch_ready, coop_prelaunch_details = self._get_coop_prelaunch_state(
            env,
            track_target_id,
            current_time,
            shooter_id=agent_id,
            distance_km=float(distance) / 1000.0,
            phase_name=_phase_str(current_phase),
            is_second_attack=bool(is_second_attack),
            target_zone=str(target_risk_zone),
        )
        coop_stable_seconds = float(coop_prelaunch_details.get("stable_seconds", 0.0) or 0.0)
        coop_trackers = coop_prelaunch_details.get("trackers", [])
        coop_tracker_count = len(coop_trackers) if isinstance(coop_trackers, (list, tuple, set)) else 0
        gate_zone_details = {
            "target_risk_zone": str(target_risk_zone),
            "target_y_km": None if not np.isfinite(target_y_km) else float(target_y_km),
            "coop_prelaunch_ready": bool(coop_prelaunch_ready),
            "coop_stable_seconds": float(coop_stable_seconds),
            "coop_tracker_count": int(coop_tracker_count),
        }

        saturation_limit = 2
        saturation_distance_m = 18000.0
        saturation_age_s = 46.0
        if target_risk_zone == "MEDIUM":
            saturation_limit = 3
            saturation_distance_m = 28000.0
            saturation_age_s = 38.0
        elif target_risk_zone == "HIGH":
            saturation_limit = 4
            saturation_distance_m = 36000.0
            saturation_age_s = 30.0
        if coop_prelaunch_ready:
            saturation_age_s = max(24.0, saturation_age_s - 6.0)
        active_on_target = self.get_missiles_targeting(track_target_id, env=env)
        effective_active_on_target = self._get_effective_missiles_targeting(
            track_target_id,
            env=env,
            current_time=current_time,
            max_age_s=saturation_age_s,
        )

        if (
            alive_enemy_count > 1
            and effective_active_on_target >= saturation_limit
            and distance > saturation_distance_m
        ):
            _trace_gate(
                allowed=False,
                reason="target_saturated",
                details={
                    "target_id": str(target_id),
                    "distance_km": float(distance) / 1000.0,
                    "active_on_target": int(active_on_target),
                    "effective_active_on_target": int(effective_active_on_target),
                    "saturation_age_s": float(saturation_age_s),
                    "saturation_limit": int(saturation_limit),
                    "alive_enemy_count": int(alive_enemy_count),
                    "team_missiles_left": int(team_missiles_left),
                    **gate_zone_details,
                    **radar_status,
                },
            )
            return False

        if (
            (not priority_target)
            and alive_enemy_count > 1
            and (missiles_left - required_missiles) < 1
            and team_missiles_left <= (alive_enemy_count + 1)
            and active_on_target <= 0
            and distance > 48000.0
        ):
            _trace_gate(
                allowed=False,
                reason="reserve_for_remaining_targets",
                details={
                    "distance_km": float(distance) / 1000.0,
                    "alive_enemy_count": int(alive_enemy_count),
                    "team_missiles_left": int(team_missiles_left),
                    "team_fired_count": int(team_fired_count),
                    "missiles_left": int(missiles_left),
                    "required_missiles": int(required_missiles),
                    "active_on_target": int(active_on_target),
                    **gate_zone_details,
                    **radar_status,
                },
            )
            return False

        if (
            (not priority_target)
            and alive_enemy_count > 1
            and team_missiles_left <= alive_enemy_count
            and active_on_target <= 0
            and distance > 55000.0
        ):
            _trace_gate(
                allowed=False,
                reason="team_reserve_hold",
                details={
                    "distance_km": float(distance) / 1000.0,
                    "alive_enemy_count": int(alive_enemy_count),
                    "team_missiles_left": int(team_missiles_left),
                    "team_fired_count": int(team_fired_count),
                    "active_on_target": int(active_on_target),
                    **gate_zone_details,
                    **radar_status,
                },
            )
            return False

        opening_hold_active = bool(
            (not is_second_attack)
            and alive_enemy_count > 1
            and team_fired_count <= 0
            and current_time <= 420.0
            and not priority_target
            and not coop_prelaunch_ready
        )
        if opening_hold_active and distance > 84000.0:
            _trace_gate(
                allowed=False,
                reason="opening_long_range_hold",
                details={
                    "distance_km": float(distance) / 1000.0,
                    "alive_enemy_count": int(alive_enemy_count),
                    "team_missiles_left": int(team_missiles_left),
                    "team_fired_count": int(team_fired_count),
                    "time_s": float(current_time),
                    "active_on_target": int(active_on_target),
                    **gate_zone_details,
                    **radar_status,
                },
            )
            return False

        if (
            (not is_second_attack)
            and alive_enemy_count > 1
            and team_fired_count <= 1
            and active_on_target >= 1
            and distance > 76000.0
            and not priority_target
            and not coop_prelaunch_ready
        ):
            _trace_gate(
                allowed=False,
                reason="followup_long_range_hold",
                details={
                    "distance_km": float(distance) / 1000.0,
                    "alive_enemy_count": int(alive_enemy_count),
                    "team_missiles_left": int(team_missiles_left),
                    "team_fired_count": int(team_fired_count),
                    "active_on_target": int(active_on_target),
                    **gate_zone_details,
                    **radar_status,
                },
            )
            return False

        # BVR 约束：默认在首轮近界内禁止继续新发射；二次进攻仅保留一个受控的近界，
        # 既允许必要时在较近 BVR 窗口补射，又避免滑入视距内空战。
        min_second_attack_km = 36.0
        is_friendly = str(agent_id).startswith("A")
        late_bvr_phases = (
            TacticalPhase.LR_TR,
            TacticalPhase.TR_DOR,
            TacticalPhase.DOR_DR,
            TacticalPhase.DR_MAR,
        )
        controlled_late_bvr_shot = bool(
            is_friendly
            and current_phase in late_bvr_phases
            and (priority_target or coop_prelaunch_ready)
        )
        min_first_attack_km = 41.0
        if controlled_late_bvr_shot and current_phase == TacticalPhase.DR_MAR:
            min_first_attack_km = 40.0
        min_second_attack_km = 35.0 if controlled_late_bvr_shot else 36.0
        if distance <= (min_first_attack_km * 1000.0 if not is_second_attack else 40000.0):
            if not is_second_attack:
                logging.debug(
                    f"⛔ [{agent_id}] 距离{distance/1000:.1f}km ≤ MAR({min_first_attack_km:.0f}km)，禁止发射"
                )
                _trace_gate(
                    allowed=False,
                    reason="mar_blocked",
                    details={
                        "距离_km": float(distance) / 1000.0,
                        "MAR_km": float(min_first_attack_km),
                        "二次进攻": bool(is_second_attack),
                        **radar_status,
                    },
                )
                return False
            if distance < (min_second_attack_km * 1000.0):
                logging.debug(f"⛔ [{agent_id}] 二次进攻距离过近({distance/1000:.1f}km < {min_second_attack_km:.0f}km)，禁止发射")
                _trace_gate(
                    allowed=False,
                    reason="second_attack_too_close",
                    details={
                        "距离_km": float(distance) / 1000.0,
                        "二次进攻最小_km": float(min_second_attack_km),
                        "二次进攻": bool(is_second_attack),
                        **radar_status,
                    },
                )
                return False

        # 3.7.2 导弹发射集成检查（雷达模式/锁定/跟踪质量/Notch/探测概率/范围）
        radar_passed = True
        radar_reason = None
        radar_conditions = {}
        try:
            launch_check = check_missile_launch_conditions(env, agent_id, target_id)
            if not launch_check.get('can_launch', False):
                reason = launch_check.get('reason', 'unknown')
                logging.debug(f"⛔ [{agent_id}] 发射条件不满足（3.7.2）: {reason}")
                radar_passed = False
                radar_reason = str(reason)
            try:
                cond = launch_check.get('conditions', {}) if isinstance(launch_check, dict) else {}
                if isinstance(cond, dict):
                    radar_conditions = {f"cond_{k}": bool(v) for k, v in cond.items()}
            except Exception:
                radar_conditions = {}
        except Exception as e:
            logging.debug(f"⚠️ [{agent_id}] 发射条件检查异常，回退到阶段窗口: {e}")
            radar_passed = False
            radar_reason = f"exception:{e}"
            radar_conditions = {}

        # 🔥 修复：参考敌方动态补偿发射机制，不限制在特定阶段
        # 在LR-TR及其附近范围内（MTR_LR, LR_TR, TR_DOR），只要有优势（雷达锁定、正对敌方等）就可以发射
        is_friendly = str(agent_id).startswith("A")
        
        # 🔥 修复：动态发射窗口，参考敌方逻辑（20-80km，朝向±45度）
        launch_allowed = False
        
        # 锁定状态仅作为内部决策（放宽朝向阈值）依据；日志以“雷达状态/锁定目标/ECM”等全字段表达
        radar_locked = False
        try:
            from simulation.radar_manager import get_unified_radar_manager
            radar_manager = get_unified_radar_manager()
            if is_friendly and hasattr(radar_manager, 'friendly_lock_targets'):
                radar_locked = radar_manager.friendly_lock_targets.get(agent_id) is not None
            elif (not is_friendly) and hasattr(radar_manager, 'enemy_lock_targets'):
                radar_locked = radar_manager.enemy_lock_targets.get(agent_id) is not None
        except Exception:
            pass
        
        # 🔥 修复：计算朝向偏差（参考敌方逻辑）
        from tactical_utils import TacticalUtils
        target_bearing = TacticalUtils.calculate_bearing(env.agents[agent_id], target_aircraft)
        current_heading_rad = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
        current_heading_deg = np.rad2deg(current_heading_rad)
        heading_error = abs(TacticalUtils.normalize_angle_diff(target_bearing - current_heading_deg))
        
        # 🔥 修复：动态发射条件（参考敌方逻辑）
        # 1. 距离条件：20-80km（与敌方一致）
        # 2. 朝向条件：±45度（与敌方一致），不允许背对或侧对
        # 3. 阶段条件：在LR-TR及其附近范围内（MTR_LR, LR_TR, TR_DOR, DOR_DR）
        # 4. 雷达锁定：如果有雷达锁定，可以放宽朝向限制到60度
        
        # 检查是否在LR-TR及其附近范围内
        nearby_phases = [
            TacticalPhase.MTR_LR, TacticalPhase.LR_TR, TacticalPhase.TR_DOR, TacticalPhase.DOR_DR
        ]
        in_nearby_phase = current_phase in nearby_phases if current_phase else False
        friendly_policy = None
        phase_gate_reason = ""
        phase_role = ""
        
        # 发射窗口按“首轮防御性攻击 / 二次攻击补射”区分：
        # 首轮优先在中远距BVR窗口开火；二次攻击允许稍近，但仍禁止落入WVR。
        if is_friendly:
            max_heading_error = 55.0 if radar_locked else 45.0
            if controlled_late_bvr_shot:
                max_heading_error = max(max_heading_error, 58.0 if coop_prelaunch_ready else 52.0)
        else:
            max_heading_error = 60.0 if radar_locked else 45.0
        
        # 🔥 修复：大幅放宽发射条件，降低要求
        # 1. 朝向条件：放宽到±90度（不允许完全背对>120度）
        # 2. 距离条件：放宽到15-100km
        # 3. 阶段条件：在LR-TR及其附近范围内更宽松
        
        # 🔥 修复：放宽朝向限制
        hard_backwards_limit = 120.0
        no_lock_backwards_limit = 90.0
        if controlled_late_bvr_shot:
            if current_phase == TacticalPhase.LR_TR:
                hard_backwards_limit = 123.0 if coop_prelaunch_ready else 121.0
                no_lock_backwards_limit = 95.0 if coop_prelaunch_ready else 92.0
            elif current_phase in (TacticalPhase.TR_DOR, TacticalPhase.DOR_DR):
                hard_backwards_limit = 126.0 if coop_prelaunch_ready else 123.0
                no_lock_backwards_limit = 98.0 if coop_prelaunch_ready else 95.0
            elif current_phase == TacticalPhase.DR_MAR:
                no_lock_backwards_limit = 96.0 if coop_prelaunch_ready else 94.0
        if heading_error > hard_backwards_limit:
            # 完全背对：禁止
            if env.current_step % 100 == 0:
                logging.warning(f"⛔ [{agent_id}] 完全背对禁止发射: 角度{heading_error:.1f}° > 120°")
            _trace_gate(
                allowed=False,
                reason="heading_backwards_blocked",
                details={
                    "距离_km": float(distance) / 1000.0,
                    "航向误差_deg": float(heading_error),
                    "radar_check_passed": bool(radar_passed),
                    "radar_check_reason": radar_reason,
                    **radar_conditions,
                    **radar_status,
                },
            )
            return False
        elif heading_error > no_lock_backwards_limit:
            # 背对但未完全背对：如果有雷达锁定，允许
            if not radar_locked:
                if env.current_step % 100 == 0:
                    logging.warning(f"⛔ [{agent_id}] 背对且无雷达锁定禁止发射: 角度{heading_error:.1f}° > 90°")
                _trace_gate(
                    allowed=False,
                    reason="heading_no_lock_blocked",
                    details={
                        "距离_km": float(distance) / 1000.0,
                        "航向误差_deg": float(heading_error),
                        "radar_check_passed": bool(radar_passed),
                        "radar_check_reason": radar_reason,
                        **radar_conditions,
                        **radar_status,
                    },
                )
                return False
        
        window_low_m = 0.0
        window_high_m = 0.0
        window_heading_limit = float(max_heading_error)
        if in_nearby_phase:
            if is_second_attack:
                window_low_m = 40000.0 if is_friendly else 42000.0
                window_high_m = 66000.0 if is_friendly else 68000.0
            else:
                window_low_m = 50000.0 if is_friendly else 48000.0
                window_high_m = 82000.0 if is_friendly else 82000.0
        else:
            if is_second_attack:
                window_low_m = 38000.0 if is_friendly else 38000.0
                window_high_m = 62000.0 if is_friendly else 62000.0
            else:
                window_low_m = 50000.0 if is_friendly else 50000.0
                window_high_m = 78000.0 if is_friendly else 78000.0

        if controlled_late_bvr_shot:
            if current_phase == TacticalPhase.LR_TR:
                window_low_m = min(window_low_m, 50000.0 if priority_target else 52000.0)
                if coop_prelaunch_ready and radar_passed and radar_locked and heading_error <= 35.0:
                    window_high_m = min(window_high_m, 82000.0)
                    window_heading_limit = min(max(window_heading_limit, 50.0), 54.0)
                else:
                    window_high_m = min(window_high_m, 78000.0 if priority_target else 76000.0)
                    window_heading_limit = min(max(window_heading_limit, 46.0), 50.0)
            elif current_phase == TacticalPhase.TR_DOR:
                window_low_m = min(window_low_m, 46000.0 if priority_target else 48000.0)
                if coop_prelaunch_ready and radar_passed and heading_error <= 35.0:
                    window_high_m = min(window_high_m, 76000.0 if priority_target else 74000.0)
                    window_heading_limit = min(max(window_heading_limit, 50.0), 56.0)
                else:
                    window_high_m = min(window_high_m, 74000.0 if priority_target else 72000.0)
                    window_heading_limit = min(max(window_heading_limit, 48.0), 52.0)
            elif current_phase == TacticalPhase.DOR_DR:
                window_low_m = min(window_low_m, 44000.0 if priority_target else 46000.0)
                if coop_prelaunch_ready and radar_passed and heading_error <= 30.0:
                    window_high_m = min(window_high_m, 70000.0 if priority_target else 68000.0)
                    window_heading_limit = min(max(window_heading_limit, 52.0), 58.0)
                else:
                    window_high_m = min(window_high_m, 68000.0 if priority_target else 66000.0)
                    window_heading_limit = min(max(window_heading_limit, 50.0), 55.0)
            elif current_phase == TacticalPhase.DR_MAR:
                if is_second_attack:
                    window_low_m = min(window_low_m, 34000.0 if coop_prelaunch_ready else 36000.0)
                    window_high_m = min(window_high_m, 66000.0 if coop_prelaunch_ready else 62000.0)
                    window_heading_limit = min(max(window_heading_limit, 58.0), 60.0)
                else:
                    window_low_m = min(window_low_m, 38000.0 if coop_prelaunch_ready else 39000.0)
                    window_high_m = min(window_high_m, 66000.0 if coop_prelaunch_ready else 62000.0)
                    window_heading_limit = min(max(window_heading_limit, 56.0), 58.0)

        if coop_prelaunch_ready and is_friendly and radar_passed and radar_locked and heading_error <= 30.0:
            window_high_m += 2000.0 if priority_target else 1000.0
            window_heading_limit = max(window_heading_limit, 50.0 if priority_target else 48.0)

        if window_low_m <= distance <= window_high_m and heading_error <= window_heading_limit:
            launch_allowed = True
            if env.current_step % (50 if is_second_attack else 100) == 0:
                mode_label = "二次进攻窗口" if is_second_attack else "首轮发射窗口"
                logging.info(
                    f"🎯 [{mode_label}] {agent_id} 距离{distance/1000:.1f}km, 角度{heading_error:.1f}°, "
                    f"阶段={current_phase.value if current_phase else 'None'}, 雷达锁定={radar_locked}"
                )
        
        # ✅ 雷达条件失败时：
        # - 友方：在 LR/TR/MTR窗口允许继续（强软通道）
        # - 二次进攻：进一步放宽，只要距离和朝向合适就允许发射（雷达条件作为参考）
        # - 敌方：仍保持严格雷达门限
        if (not radar_passed) and (not is_friendly):
            if env.current_step % 100 == 0:
                logging.warning(f"⛔ [{agent_id}] 雷达检查失败禁止发射: {radar_reason}")
            _trace_gate(
                allowed=False,
                reason="radar_gate_failed_enemy_strict",
                details={
                    "距离_km": float(distance) / 1000.0,
                    "航向误差_deg": float(heading_error),
                    "radar_check_passed": bool(radar_passed),
                    "radar_check_reason": radar_reason,
                    **radar_conditions,
                    "窗口允许": bool(launch_allowed),
                    "in_nearby_phase": bool(in_nearby_phase),
                    "二次进攻": bool(is_second_attack),
                    "phase": _phase_str(current_phase),
                    "phase_role": phase_role,
                    "phase_gate_reason": phase_gate_reason,
                    **radar_status,
                },
            )
            return False

        # ✅ 二次进攻：雷达条件失败时也允许发射（只要距离和朝向合适）
        if (not radar_passed) and is_friendly:
            if current_phase == TacticalPhase.LR_TR:
                max_relaxed_distance = 74000.0 if (priority_target and coop_prelaunch_ready) else 70000.0
                max_relaxed_heading = 52.0 if radar_locked else 48.0
            elif current_phase == TacticalPhase.TR_DOR:
                max_relaxed_distance = 76000.0 if priority_target else 72000.0
                max_relaxed_heading = 54.0 if radar_locked else 50.0
            elif current_phase == TacticalPhase.DOR_DR:
                max_relaxed_distance = 72000.0 if priority_target else 68000.0
                max_relaxed_heading = 56.0 if radar_locked else 52.0
            elif current_phase == TacticalPhase.DR_MAR:
                if is_second_attack:
                    max_relaxed_distance = 62000.0 if coop_prelaunch_ready else 58000.0
                    max_relaxed_heading = 56.0
                else:
                    max_relaxed_distance = 66000.0 if coop_prelaunch_ready else 62000.0
                    max_relaxed_heading = 54.0
            elif priority_target:
                max_relaxed_distance = 74000.0 if is_second_attack else 76000.0
                max_relaxed_heading = 56.0 if is_second_attack else 52.0
            else:
                max_relaxed_distance = 70000.0 if is_second_attack else 72000.0
                max_relaxed_heading = 54.0 if is_second_attack else 50.0
            relaxed_friendly_shot = bool(
                launch_allowed
                and distance <= max_relaxed_distance
                and heading_error <= max_relaxed_heading
            )
            coop_relaxed_distance = 0.0
            coop_relaxed_heading = 0.0
            coop_relaxed_shot = False
            if coop_prelaunch_ready and launch_allowed:
                if current_phase == TacticalPhase.LR_TR:
                    coop_relaxed_distance = 80000.0 if (radar_locked and heading_error <= 35.0) else 74000.0
                    coop_relaxed_heading = 54.0 if radar_locked else 50.0
                elif current_phase == TacticalPhase.TR_DOR:
                    coop_relaxed_distance = 78000.0 if (radar_locked and heading_error <= 35.0) else 74000.0
                    coop_relaxed_heading = 56.0 if radar_locked else 52.0
                elif current_phase == TacticalPhase.DOR_DR:
                    coop_relaxed_distance = 74000.0 if (radar_locked and heading_error <= 30.0) else 70000.0
                    coop_relaxed_heading = 58.0 if radar_locked else 54.0
                elif current_phase == TacticalPhase.DR_MAR:
                    coop_relaxed_distance = 68000.0 if is_second_attack else 70000.0
                    coop_relaxed_heading = 58.0 if is_second_attack else 56.0
                elif priority_target:
                    coop_relaxed_distance = 76000.0 if is_second_attack else 78000.0
                    coop_relaxed_heading = 56.0 if is_second_attack else 54.0
                else:
                    coop_relaxed_distance = 72000.0 if is_second_attack else 74000.0
                    coop_relaxed_heading = 54.0 if is_second_attack else 52.0
                coop_relaxed_shot = bool(
                    distance <= coop_relaxed_distance
                    and heading_error <= coop_relaxed_heading
                )
            if not (relaxed_friendly_shot or coop_relaxed_shot):
                _trace_gate(
                    allowed=False,
                    reason="friendly_radar_gate_failed",
                    details={
                        "distance_km": float(distance) / 1000.0,
                        "heading_error_deg": float(heading_error),
                        "radar_check_reason": radar_reason,
                        "track_target_id": str(track_target_id),
                        "phase": _phase_str(current_phase),
                        "second_attack": bool(is_second_attack),
                        "effective_active_on_target": int(effective_active_on_target),
                        "active_on_target": int(active_on_target),
                        "coop_relaxed_distance_km": float(coop_relaxed_distance) / 1000.0,
                        "coop_relaxed_heading_deg": float(coop_relaxed_heading),
                        **gate_zone_details,
                        **radar_conditions,
                        **radar_status,
                    },
                )
                return False
        if (not radar_passed) and is_friendly:
            if is_second_attack and launch_allowed:
                # 二次进攻：雷达条件失败也允许，只要距离和朝向合适
                if env.current_step % 50 == 0:
                    logging.info(f"🎯 [二次进攻-雷达放宽] {agent_id} radar_check_failed={radar_reason}，但距离和朝向合适，允许发射")
            elif launch_allowed:
                # 常规情况：强软发射通道
                if env.current_step % 100 == 0 and radar_reason:
                    logging.info(f"🎯 [{agent_id}] 强软发射通道: radar_check_failed={radar_reason}，但处于窗口内，继续朝向检查")

        # 打印不发射的详细原因
        if not launch_allowed:
            reasons = []
            if phase_gate_reason and window_high_m <= 0.0:
                reasons.append(f"phase_gate:{phase_gate_reason}")
            if distance < window_low_m:
                reasons.append(f"距离过近({distance/1000:.1f}km < {window_low_m/1000:.0f}km)")
            elif distance > window_high_m:
                reasons.append(f"距离过远({distance/1000:.1f}km > {window_high_m/1000:.0f}km)")
            if heading_error > window_heading_limit:
                reasons.append(f"朝向不满足(角度{heading_error:.1f}° > {window_heading_limit:.0f}°)")
            if is_second_attack and distance < (min_second_attack_km * 1000.0):
                reasons.append(f"低于二次进攻BVR下限({distance/1000:.1f}km < {min_second_attack_km:.0f}km)")
            if env.current_step % 100 == 0:
                logging.warning(f"⛔ [{agent_id}] 发射条件不满足: {', '.join(reasons) if reasons else '未知原因'}, 距离={distance/1000:.1f}km, 角度={heading_error:.1f}°, 阶段={current_phase.value if current_phase else 'None'}, 二次进攻={is_second_attack}")

            _trace_gate(
                allowed=False,
                reason="window_not_allowed",
                details={
                    "距离_km": float(distance) / 1000.0,
                    "航向误差_deg": float(heading_error),
                    "radar_check_passed": bool(radar_passed),
                    "radar_check_reason": radar_reason,
                    **radar_conditions,
                    "in_nearby_phase": bool(in_nearby_phase),
                    "phase": _phase_str(current_phase),
                    "phase_role": phase_role,
                    "phase_gate_reason": phase_gate_reason,
                    "二次进攻": bool(is_second_attack),
                    "窗口下界_km": float(window_low_m) / 1000.0,
                    "窗口上界_km": float(window_high_m) / 1000.0,
                    "窗口航向_deg": float(window_heading_limit),
                    "reasons": ",".join(reasons) if reasons else None,
                    "剩余": int(missiles_left),
                    "需要": int(required_missiles),
                    "已发射": int(fired_count),
                    **radar_status,
                },
            )
            
            # ✅ 保存拒绝发射的原因
            if not hasattr(self, 'last_gate_decision'):
                self.last_gate_decision = {}
            self.last_gate_decision[agent_id] = {
                'allowed': False,
                'reason': 'denied',
                'target_id': str(target_id) if target_id else None,
                'phase': _phase_str(current_phase),
                'time_s': float(current_time),
                'details': {
                    "距离_km": float(distance) / 1000.0,
                    "航向误差_deg": float(heading_error),
                    "radar_check_passed": bool(radar_passed),
                    "radar_check_reason": radar_reason,
                    **radar_conditions,
                    "in_nearby_phase": bool(in_nearby_phase),
                    "phase": _phase_str(current_phase),
                    "二次进攻": bool(is_second_attack),
                    "窗口下界_km": float(window_low_m) / 1000.0,
                    "窗口上界_km": float(window_high_m) / 1000.0,
                    "窗口航向_deg": float(window_heading_limit),
                    "reasons": ",".join(reasons) if reasons else None,
                    "剩余": int(missiles_left),
                    "需要": int(required_missiles),
                    "已发射": int(fired_count),
                    **radar_status,
                }
            }
        else:
            # ✅ 正向理由链：解释“为什么允许/触发发射”（不仅仅输出拒绝原因）
            allow_reasons = []
            try:
                allow_reasons.append(f"窗口内(distance={float(distance)/1000.0:.1f}km)")
                allow_reasons.append(f"朝向可用(误差={float(heading_error):.1f}°)")
                allow_reasons.append(f"冷却通过")
                allow_reasons.append(f"配额未超({int(fired_count)}/{int(self.MAX_MISSILES_PER_AIRCRAFT)})")
                allow_reasons.append(f"剩弹足够({int(missiles_left)}>=需要{int(required_missiles)})")
                if phase_role:
                    allow_reasons.append(f"phase_role={phase_role}")
                if radar_passed:
                    allow_reasons.append("雷达门限通过(3.7.2)")
                else:
                    if is_second_attack:
                        allow_reasons.append(f"二次进攻雷达放宽({radar_reason})")
                    else:
                        allow_reasons.append(f"强软发射通道(雷达未通过:{radar_reason})")
            except Exception:
                allow_reasons = None

            if radar_passed:
                allow_reason = "allowed_all_gates"
            elif is_second_attack:
                allow_reason = "allowed_second_attack_radar_relaxed"
            else:
                allow_reason = "allowed_soft_channel_radar_failed"

            _trace_gate(
                allowed=True,
                reason=str(allow_reason),
                details={
                    "距离_km": float(distance) / 1000.0,
                    "航向误差_deg": float(heading_error),
                    "radar_check_passed": bool(radar_passed),
                    "radar_check_reason": radar_reason,
                    **radar_conditions,
                    "in_nearby_phase": bool(in_nearby_phase),
                    "phase": _phase_str(current_phase),
                    "二次进攻": bool(is_second_attack),
                    "剩余": int(missiles_left),
                    "需要": int(required_missiles),
                    "已发射": int(fired_count),
                    "允许发射理由": allow_reasons,
                    **radar_status,
                },
            )
            
            # ✅ 保存发射决策信息供外部使用（用于记录发射原因）
            if not hasattr(self, 'last_gate_decision'):
                self.last_gate_decision = {}
            self.last_gate_decision[agent_id] = {
                'allowed': True,
                'reason': str(allow_reason),
                'target_id': str(target_id),
                'phase': _phase_str(current_phase),
                'time_s': float(current_time),
                'details': {
                    "距离_km": float(distance) / 1000.0,
                    "航向误差_deg": float(heading_error),
                    "radar_check_passed": bool(radar_passed),
                    "radar_check_reason": radar_reason,
                    **radar_conditions,
                    "in_nearby_phase": bool(in_nearby_phase),
                    "phase": _phase_str(current_phase),
                    "二次进攻": bool(is_second_attack),
                    "剩余": int(missiles_left),
                    "需要": int(required_missiles),
                    "已发射": int(fired_count),
                    "允许发射理由": allow_reasons,
                    **radar_status,
                }
            }
                
        return launch_allowed
    
    def execute_missile_launch(self, env, agent_id: str, target_id: str = None, 
                            salvo_mode: str = "single") -> bool:
        """
        执行导弹发射 - 支持分批发射 + 严格4导弹限制
        
        Args:
            env: 环境
            agent_id: 飞机ID
            target_id: 目标ID
            salvo_mode: 发射模式 ("single", "double", "salvo")
            
        Returns:
            True表示发射成功
        """
        
        # **关键修复：再次检查4导弹限制**
        if agent_id not in self.aircraft_missile_counts:
            self.aircraft_missile_counts[agent_id] = 0
        
        fired_count = self.aircraft_missile_counts[agent_id]
        
        # 获取目标
        if target_id is None:
            target_id = get_target_with_fallback(agent_id, env)
        
        if target_id is None:
            logging.warning(f"⚠️ [{agent_id}] 导弹发射失败: 无目标")
            return False
        
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft is None or not target_aircraft.is_alive:
            logging.warning(f"⚠️ [{agent_id}] 导弹发射失败: 目标不存在或已被击毁")
            return False
        
        # 确定发射数量
        aircraft = env.agents[agent_id]
        remaining_quota = self.MAX_MISSILES_PER_AIRCRAFT - fired_count

        missiles_left = self._get_aircraft_missiles_left(aircraft)
        
        if salvo_mode == "single":
            launch_count = min(1, remaining_quota, missiles_left)
        elif salvo_mode == "double":
            launch_count = min(2, remaining_quota, missiles_left)
        elif salvo_mode == "salvo":
            launch_count = min(2, remaining_quota, missiles_left)  # 限制一次最多发射2枚
        else:
            launch_count = min(1, remaining_quota, missiles_left)
        
        if launch_count <= 0:
            logging.warning(f"⚠️ [{agent_id}] 导弹发射失败: 已发射{fired_count}枚, 达到上限{self.MAX_MISSILES_PER_AIRCRAFT}枚")
            return False
        
        # 执行发射
        current_time = env.current_step * env.time_interval
        launched_count = 0
        
        try:
            for i in range(launch_count):
                missiles_left = self._get_aircraft_missiles_left(aircraft)

                if missiles_left <= 0 or self.aircraft_missile_counts[agent_id] >= self.MAX_MISSILES_PER_AIRCRAFT:
                    break
                    
                # **修复导弹ID生成逻辑**
                fired_sequence = self.aircraft_missile_counts[agent_id] + 1  # 1-4的序列号
                base_id = agent_id[0] + agent_id[2:]  # A0100 -> A100
                missile_uid = f"{base_id}{fired_sequence:0>2}"  # A10001, A10002, A10003, A10004
                
                # 发射导弹（这里应该调用环境的导弹发射接口）
                # env.launch_missile(agent_id, target_id, missile_uid)
                
                # 记录发射信息
                self.state_manager.record_missile_launch(agent_id, current_time, target_id)
                
                # 记录导弹轨迹追踪
                self.missile_tracks[missile_uid] = {
                    'launcher': agent_id,
                    'target': target_id,
                    'launch_time': current_time,
                    'status': 'active'
                }
                
                # **更新发射计数**
                self.aircraft_missile_counts[agent_id] += 1
                
                # 更新飞机导弹数量
                try:
                    if hasattr(aircraft, 'num_left_missiles'):
                        aircraft.num_left_missiles = max(0, int(getattr(aircraft, 'num_left_missiles', 0)) - 1)
                    else:
                        aircraft.num_missiles = max(0, int(getattr(aircraft, 'num_missiles', 0)) - 1)
                except Exception:
                    pass
                launched_count += 1
                
                logging.info(f"[MISSILE] {agent_id} fired {missile_uid} -> {target_id} ({self.aircraft_missile_counts[agent_id]}/{self.MAX_MISSILES_PER_AIRCRAFT})")
            
            if launched_count > 0:
                launch_mode_desc = {
                    "single": "单发",
                    "double": "双发", 
                    "salvo": "齐射"
                }
                logging.info(f"✅ [{agent_id}] {launch_mode_desc.get(salvo_mode, '发射')} {launched_count}枚导弹 → {target_id} (已发射总计:{self.aircraft_missile_counts[agent_id]}/{self.MAX_MISSILES_PER_AIRCRAFT})")
                return True
            
        except Exception as e:
            logging.error(f"❌ [{agent_id}] 导弹发射异常: {str(e)}")
            
        return False
    
    def update_missile_status(self, env, current_time: float):
        """
        更新所有导弹状态
        
        Args:
            env: 环境
            current_time: 当前时间
        """
        for missile_id, track in list(self.missile_tracks.items()):
            if track['status'] != 'active':
                continue
            
            # 检查导弹飞行时间
            flight_time = current_time - track['launch_time']
            
            # 检查目标状态
            target_lookup_id = track.get('target_key', track['target'])
            target_aircraft = self._resolve_target_aircraft(env, target_id=target_lookup_id)
            if target_aircraft is None:
                for alias in track.get('target_aliases', []):
                    target_aircraft = self._resolve_target_aircraft(env, target_id=str(alias))
                    if target_aircraft is not None:
                        break
            
            if target_aircraft is None or not target_aircraft.is_alive:
                track['status'] = 'target_destroyed'
                logging.info(f"🎯 导弹{missile_id}: 目标{track.get('target')}已被击毁")
                continue
            
            # 简化版：假设导弹飞行60秒后失效
            if flight_time > 60.0:
                track['status'] = 'expired'
                logging.info(f"⏱️ 导弹{missile_id}: 飞行超时失效")
                continue
    
    def get_active_missiles_count(self, agent_id: str = None) -> int:
        """
        获取激活导弹数量
        
        Args:
            agent_id: 飞机ID（None表示所有飞机）
            
        Returns:
            激活导弹数量
        """
        count = 0
        for track in self.missile_tracks.values():
            if track['status'] == 'active':
                if agent_id is None or track['launcher'] == agent_id:
                    count += 1
        return count
    
    def get_missiles_targeting(self, target_id: str, env=None) -> int:
        """
        获取瞄准某目标的导弹数量
        
        Args:
            target_id: 目标ID
            
        Returns:
            导弹数量
        """
        target_aliases = {str(target_id)} if target_id else set()
        if env is not None:
            target_aliases |= self._resolve_target_aliases(env, target_id=str(target_id) if target_id else None)
        count = 0
        for track in self.missile_tracks.values():
            if track['status'] != 'active':
                continue
            aliases = set(track.get('target_aliases', []))
            aliases.add(str(track.get('target', '')))
            aliases.add(str(track.get('target_key', '')))
            if aliases & target_aliases:
                count += 1
        return count
    
    def check_missile_threat(self, env, agent_id: str, current_time: float) -> dict:
        """
        检查导弹威胁
        
        Args:
            env: 环境
            agent_id: 飞机ID
            current_time: 当前时间
            
        Returns:
            威胁信息字典: {'threat_level': 0-3, 'incoming_count': int, 'closest_distance': float}
        """
        threat_info = {
            'threat_level': 0,  # 0-无威胁, 1-低, 2-中, 3-高
            'incoming_count': 0,
            'closest_distance': float('inf')
        }
        
        # 检查所有瞄准该飞机的导弹
        incoming = []
        agent_aliases = self._resolve_target_aliases(env, target_id=agent_id)
        for missile_id, track in self.missile_tracks.items():
            aliases = set(track.get('target_aliases', []))
            aliases.add(str(track.get('target', '')))
            aliases.add(str(track.get('target_key', '')))
            if track['status'] == 'active' and aliases & agent_aliases:
                flight_time = current_time - track['launch_time']
                incoming.append({
                    'missile_id': missile_id,
                    'launcher': track['launcher'],
                    'flight_time': flight_time
                })
        
        threat_info['incoming_count'] = len(incoming)
        
        # 评估威胁等级
        if len(incoming) == 0:
            threat_info['threat_level'] = 0
        elif len(incoming) == 1:
            # 单导弹威胁：根据飞行时间评估
            flight_time = incoming[0]['flight_time']
            if flight_time < 10:
                threat_info['threat_level'] = 1  # 低（刚发射）
            elif flight_time < 30:
                threat_info['threat_level'] = 2  # 中（接近）
            else:
                threat_info['threat_level'] = 3  # 高（即将命中）
        else:
            # 多导弹威胁：高威胁
            threat_info['threat_level'] = 3
        
        return threat_info
    
    def get_remaining_missiles(self, agent_id: str) -> int:
        """
        获取飞机剩余导弹配额
        
        Args:
            agent_id: 飞机ID
            
        Returns:
            剩余可发射导弹数量
        """
        if agent_id not in self.aircraft_missile_counts:
            self.aircraft_missile_counts[agent_id] = 0
        
        fired_count = self.aircraft_missile_counts[agent_id]
        return max(0, self.MAX_MISSILES_PER_AIRCRAFT - fired_count)
    
    def reset(self):
        """重置导弹管理器"""
        self.missile_tracks.clear()
        self.aircraft_missile_counts.clear()
