from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cap_state_machine import CAPState, StateContext
from .picture import RiskZone as PictureRiskZone
from .tactic_selector import TacticAssignment
from . import cap_picture_helpers as _cph
from . import cap_tactic_helpers as _ctah
from . import cap_guidance_helpers as _capgh
from . import cap_picture_helpers as _capph
from . import cap_tactic_helpers as _captach
from . import cap_engagement_helpers as _capegh
from . import cap_tracking_helpers as _captrk
from . import cap_rootcause_logging as _caproot
from . import cap_task_step_helpers as _capstep


class CAPTaskRuntimeMixin:
    def step(self, env):
        return _capstep.run_cap_step(self, env)

    def _update_awacs_data(self, env, current_time: float):
        return _cph._update_awacs_data(self, env, current_time)

    def _calculate_cap_context(self, env, current_time: float) -> StateContext:
        return _capph._calculate_cap_context(self, env, current_time)

    def _log_control_ranges_once(self, initial_distance: float):
        return _ctah._log_control_ranges_once(self, initial_distance)

    def _log_dynamic_control_ranges_once(self, initial_distance: float):
        return _ctah._log_dynamic_control_ranges_once(self, initial_distance)

    def _get_second_attack_subnode(self, distance: float, ranges_dict: Dict[str, float] = None) -> Optional[str]:
        return _captach._get_second_attack_subnode(self, distance, ranges_dict)

    def _node_to_progress(self, node: str) -> int:
        return _captach._node_to_progress(self, node)

    def _lock_template_node(self, agent_id: str, target_id: str, raw_node: str) -> str:
        return _captach._lock_template_node(self, agent_id, target_id, raw_node)

    def _pick_pair_target_with_stickiness(self, pair_name: str, alive_pair: List[str], remaining_threats: List):
        return _captach._pick_pair_target_with_stickiness(self, pair_name, alive_pair, remaining_threats)

    def _get_pair_name_for_agent(self, agent_id: str) -> Optional[str]:
        return _captach._get_pair_name_for_agent(self, agent_id)

    def _get_pair_enemy_ids(self, agent_id: str) -> List[str]:
        return _captach._get_pair_enemy_ids(self, agent_id)

    def _get_pair_tactic_assignment(self, agent_id: str) -> Optional[TacticAssignment]:
        return _captach._get_pair_tactic_assignment(self, agent_id)

    def _get_nearest_visible_pair_threat(self, agent_id: str, x: float, y: float):
        return _captach._get_nearest_visible_pair_threat(self, agent_id, x, y)

    def _get_battlefield_pos_placeholder(self, aid: str) -> np.ndarray:
        return _captach._get_battlefield_pos_placeholder(self, aid)

    def _build_pair_node_status(self, env) -> str:
        return _ctah._build_pair_node_status(self, env)

    def _log_situation(self, env, ctx: StateContext, cap_state: CAPState, current_time: float):
        return _ctah._log_situation(self, env, ctx, cap_state, current_time)

    def _update_mission_evaluation(self, env, current_time: float):
        return _ctah._update_mission_evaluation(self, env, current_time)

    def _update_tactic_selection(self, env, current_time: float):
        return _ctah._update_tactic_selection(self, env, current_time)

    def _refresh_dynamic_control_ranges(self, env, ctx: StateContext, current_time: float) -> Dict[str, float]:
        return _ctah._refresh_dynamic_control_ranges(self, env, ctx, current_time)

    def _update_cooperative_detection(self, env, current_time: float):
        return _capegh._update_cooperative_detection(self, env, current_time)

    def _update_radar_tracking_only(self, env, current_time: float):
        return _capegh._update_radar_tracking_only(self, env, current_time)

    def _update_engagement(self, env, current_time: float):
        return _capegh._update_engagement(self, env, current_time)

    def _update_stable_tracking_window(self, track_qualities: Dict[str, Dict[str, float]], current_time: float):
        return _capgh._update_stable_tracking_window(self, track_qualities, current_time)

    def get_guidance_verification_snapshot(self) -> Dict[str, object]:
        return _capgh.get_guidance_verification_snapshot(self)

    def record_external_launch_event(
        self,
        shooter_id: str,
        target_id: str,
        missile_id: str,
        current_time: float,
        source: str = 'external',
    ) -> None:
        return _capgh.record_external_launch_event(self, shooter_id, target_id, missile_id, current_time, source)

    def record_external_gate_block(
        self,
        shooter_id: str,
        target_id: str,
        current_time: float,
        reason: str = 'external',
    ) -> None:
        return _capgh.record_external_gate_block(self, shooter_id, target_id, current_time, reason)

    def _get_prelaunch_gate_profile(
        self,
        target_id: str,
        current_time: float,
        shooter_id: Optional[str] = None,
        distance_km: Optional[float] = None,
        phase_name: str = "",
        is_second_attack: bool = False,
        target_zone: Optional[str] = None,
    ) -> Dict[str, object]:
        return _capgh._get_prelaunch_gate_profile(
            self,
            target_id,
            current_time,
            shooter_id=shooter_id,
            distance_km=distance_km,
            phase_name=phase_name,
            is_second_attack=is_second_attack,
            target_zone=target_zone,
        )

    def _has_stable_tracking_ready(self, target_id: str, required_seconds: Optional[float] = None) -> bool:
        return _capgh._has_stable_tracking_ready(self, target_id, required_seconds=required_seconds)

    def _has_stable_dual_tracking(self, target_id: str, required_seconds: Optional[float] = None) -> bool:
        return _capgh._has_stable_tracking_ready(self, target_id, required_seconds=required_seconds)

    def check_external_prelaunch_gate(
        self,
        target_id: str,
        current_time: float,
        shooter_id: Optional[str] = None,
        distance_km: Optional[float] = None,
        phase_name: str = "",
        is_second_attack: bool = False,
        target_zone: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, object]]:
        return _capgh.check_external_prelaunch_gate(
            self,
            target_id,
            current_time,
            shooter_id=shooter_id,
            distance_km=distance_km,
            phase_name=phase_name,
            is_second_attack=is_second_attack,
            target_zone=target_zone,
        )

    def _is_agent_in_defensive_maneuver(self, env, agent_id: str) -> bool:
        return _capgh._is_agent_in_defensive_maneuver(self, env, agent_id)

    def _update_relay_guidance(self, env, current_time: float):
        return _capgh._update_relay_guidance(self, env, current_time)

    def _check_relay_guidance(self, env, agent_id: str, distance: float):
        return _capgh._check_relay_guidance(self, env, agent_id, distance)

    def _select_relay_candidate(
        self,
        env,
        exclude_agent: str,
        target_id: str,
        terminal_priority: bool = False,
    ) -> Optional[str]:
        return _capgh._select_relay_candidate(
            self,
            env,
            exclude_agent,
            target_id,
            terminal_priority=terminal_priority,
        )

    def _update_intent_recognition(self, env, current_time: float):
        return _captrk._update_intent_recognition(self, env, current_time)

    def _update_cooperative_tracking(self, env, current_time: float):
        return _captrk._update_cooperative_tracking(self, env, current_time)

    def _update_missiles(self, env, current_time: float):
        return _captrk._update_missiles(self, env, current_time)

    def _start_cooperative_tracking(self, env, target_id: str):
        return _captrk._start_cooperative_tracking(self, env, target_id)

    def _is_missile_incoming(self, env) -> bool:
        return _caproot._is_missile_incoming(self, env)

    def _is_fuel_critical(self, env) -> bool:
        return _caproot._is_fuel_critical(self, env)

    def _update_picture(self, env, current_time: float):
        return _cph._update_picture(self, env, current_time)

    def _build_state_context(self, env, current_time: float) -> StateContext:
        return _cph._build_state_context(self, env, current_time)

    def _get_current_sim_time(self) -> float:
        try:
            return float(getattr(self, "step_count", 0)) * 0.2
        except Exception:
            return 0.0

    def _get_picture_filter_kwargs(self, purpose: str = "decision") -> Dict[str, object]:
        purpose_name = str(purpose or "decision").strip().lower()
        if purpose_name == "raw":
            return {}
        if purpose_name == "search":
            return {
                "max_track_age": float(getattr(self, "_picture_search_track_max_age_s", 20.0)),
                "include_recent_lost": True,
                "max_lost_time": float(getattr(self, "_picture_search_lost_grace_s", 20.0)),
            }
        return {
            "max_track_age": float(getattr(self, "_picture_decision_track_max_age_s", 6.0)),
            "include_recent_lost": bool(getattr(self, "_picture_decision_include_recent_lost", True)),
            "max_lost_time": float(getattr(self, "_picture_decision_lost_grace_s", 2.5)),
        }

    def _resolve_picture_env(self):
        candidates = []
        for candidate in (
            getattr(self, "env", None),
            getattr(getattr(self, "_cap_task_sink", None), "env", None),
            getattr(getattr(self, "task", None), "env", None),
        ):
            if candidate is None:
                continue
            if any(candidate is existing for existing in candidates):
                continue
            candidates.append(candidate)
            original = getattr(candidate, "_original", None)
            if original is not None and not any(original is existing for existing in candidates):
                candidates.append(original)
        for candidate in candidates:
            if getattr(candidate, "agents", None):
                return candidate
        return candidates[0] if candidates else None

    def _get_picture_target_geometry(self, target_id: str) -> Dict[str, object]:
        info: Dict[str, object] = {
            "target_zone": "UNKNOWN",
            "target_depth_km": float("nan"),
        }
        tid = str(target_id or "").strip()
        if not tid:
            return info

        env = self._resolve_picture_env()
        if env is None:
            return info

        resolved_id = tid
        if hasattr(env, "resolve_real_agent_id"):
            try:
                candidate = env.resolve_real_agent_id(tid)
                if candidate:
                    resolved_id = str(candidate)
            except Exception:
                pass

        aircraft = None
        agents = getattr(env, "agents", None)
        if isinstance(agents, dict):
            aircraft = agents.get(resolved_id) or agents.get(tid)
        if aircraft is None:
            jsbsims = getattr(env, "_jsbsims", None)
            if isinstance(jsbsims, dict):
                aircraft = jsbsims.get(resolved_id) or jsbsims.get(tid)
        if aircraft is None:
            return info

        depth_km = float("nan")
        try:
            if hasattr(aircraft, "get_geodetic") and hasattr(self, "coord_sys"):
                geo = aircraft.get_geodetic()
                if geo is not None and len(geo) >= 2:
                    _, battlefield_y_km = self.coord_sys.geodetic_to_battlefield(
                        float(geo[0]),
                        float(geo[1]),
                    )
                    depth_km = float(battlefield_y_km)
        except Exception:
            depth_km = float("nan")

        if not np.isfinite(depth_km):
            try:
                position = aircraft.get_position()
                depth_km = float(position[1]) / 1000.0
            except Exception:
                return info

        info["target_depth_km"] = depth_km
        if not np.isfinite(depth_km):
            return info
        if depth_km < 0.0:
            info["target_zone"] = "OUTSIDE"
        elif depth_km <= 100.0:
            info["target_zone"] = "HIGH"
        elif depth_km <= 200.0:
            info["target_zone"] = "MEDIUM"
        elif depth_km <= 300.0:
            info["target_zone"] = "LOW"
        else:
            info["target_zone"] = "OUTSIDE"
        return info

    def _get_cap_pair_name(self, entity_id: Optional[str]) -> Optional[str]:
        eid = str(entity_id or "").strip().upper()
        if not eid:
            return None
        if eid.endswith("0100") or eid.endswith("0200"):
            return "left"
        if eid.endswith("0300") or eid.endswith("0400"):
            return "right"
        return None

    def _get_native_friendlies_for_target(self, target_id: Optional[str]) -> List[str]:
        pair_name = self._get_cap_pair_name(target_id)
        if pair_name == "left":
            return ["A0100", "A0200"]
        if pair_name == "right":
            return ["A0300", "A0400"]
        return []

    def _get_cap_agent_missiles_left(self, aircraft) -> int:
        if aircraft is None:
            return 0
        try:
            if hasattr(aircraft, "num_left_missiles"):
                return max(0, int(getattr(aircraft, "num_left_missiles")))
        except Exception:
            pass
        try:
            return max(0, int(getattr(aircraft, "num_missiles", 0)))
        except Exception:
            return 0

    def _get_cap_agent_depth_km(self, aircraft) -> float:
        if aircraft is None:
            return float("nan")
        try:
            if hasattr(aircraft, "get_geodetic") and hasattr(self, "coord_sys"):
                geo = aircraft.get_geodetic()
                if geo is not None and len(geo) >= 2:
                    _, battlefield_y_km = self.coord_sys.geodetic_to_battlefield(
                        float(geo[0]),
                        float(geo[1]),
                    )
                    return float(battlefield_y_km)
        except Exception:
            pass
        try:
            position = aircraft.get_position()
            return float(position[1]) / 1000.0
        except Exception:
            return float("nan")

    def _evaluate_cap_defensive_fire_window(
        self,
        agent_id: Optional[str],
        target_id: Optional[str],
        env=None,
        current_time: Optional[float] = None,
    ) -> Dict[str, object]:
        aid = str(agent_id or "").strip()
        tid = str(target_id or "").strip()
        info: Dict[str, object] = {
            "eligible": False,
            "distance_km": float("nan"),
            "target_zone": "UNKNOWN",
            "target_depth_km": float("nan"),
            "shooter_depth_km": float("nan"),
            "reason": "",
        }
        if not aid.startswith("A") or not tid.startswith("B"):
            return info

        env = env or self._resolve_picture_env()
        agents = getattr(env, "agents", None) if env is not None else None
        if not isinstance(agents, dict):
            return info

        shooter = agents.get(aid)
        if shooter is None or not getattr(shooter, "is_alive", False):
            return info
        if self._get_cap_agent_missiles_left(shooter) <= 0:
            return info

        target_status = self._get_picture_target_actionability(
            tid,
            current_time=current_time,
            purpose="decision",
        )
        if not bool(target_status.get("actionable", True)):
            return info

        target_zone = str(target_status.get("target_zone", "UNKNOWN") or "UNKNOWN").upper()
        target_depth_km = float(target_status.get("target_depth_km", float("nan")))
        info["target_zone"] = target_zone
        info["target_depth_km"] = target_depth_km
        if target_zone not in ("HIGH", "MEDIUM"):
            return info

        resolved_tid = tid
        if hasattr(env, "resolve_real_agent_id"):
            try:
                candidate = env.resolve_real_agent_id(tid)
                if candidate:
                    resolved_tid = str(candidate)
            except Exception:
                resolved_tid = tid
        target = agents.get(resolved_tid) or agents.get(tid)
        if target is None or not getattr(target, "is_alive", False):
            return info

        try:
            shooter_pos = np.asarray(shooter.get_position(), dtype=float)
            target_pos = np.asarray(target.get_position(), dtype=float)
            distance_km = float(np.linalg.norm(target_pos - shooter_pos)) / 1000.0
        except Exception:
            return info
        shooter_depth_km = self._get_cap_agent_depth_km(shooter)
        info["distance_km"] = distance_km
        info["shooter_depth_km"] = shooter_depth_km

        if target_zone == "HIGH":
            max_distance_km = 88.0
            max_shooter_depth_km = 110.0
            max_target_depth_km = 132.0
            retreat_depth_limit_km = 148.0
        else:
            max_distance_km = 72.0
            max_shooter_depth_km = 106.0
            max_target_depth_km = 168.0
            retreat_depth_limit_km = 155.0

        if distance_km > max_distance_km:
            return info
        if np.isfinite(shooter_depth_km) and shooter_depth_km > max_shooter_depth_km:
            return info
        if np.isfinite(target_depth_km) and target_depth_km > max_target_depth_km:
            return info

        group_phase = str(target_status.get("group_phase", "") or "").upper()
        if (
            group_phase in ("TURN_NORTH", "REGROUP_NORTH")
            and distance_km > 55.0
            and np.isfinite(target_depth_km)
            and target_depth_km > retreat_depth_limit_km
        ):
            return info

        info["eligible"] = True
        info["reason"] = "defensive_fire_window"
        return info

    def _should_cap_abort_tactical_turn(
        self,
        env,
        agent_id: Optional[str],
        *,
        threat=None,
        current_time: Optional[float] = None,
    ) -> bool:
        aid = str(agent_id or "").strip()
        if not aid.startswith("A"):
            return False

        env = env or self._resolve_picture_env()
        agents = getattr(env, "agents", None) if env is not None else None
        if not isinstance(agents, dict):
            return False

        shooter = agents.get(aid)
        if shooter is None or not getattr(shooter, "is_alive", False):
            return False
        if self._get_cap_agent_missiles_left(shooter) <= 0:
            return False

        threat_obj = threat
        if threat_obj is None:
            try:
                x = float(shooter.get_position()[0]) / 1000.0
                y = float(shooter.get_position()[1]) / 1000.0
            except Exception:
                return False
            try:
                threat_obj = self._get_nearest_visible_pair_threat(aid, x, y)
            except Exception:
                threat_obj = None
            if threat_obj is None:
                try:
                    threat_obj = self._get_picture_nearest_threat(np.array([x, y, 8.0]), purpose="decision")
                except Exception:
                    threat_obj = None
        if threat_obj is None:
            return False

        target_id = str(getattr(threat_obj, "track_id", "") or "").strip()
        if not target_id:
            return False

        window = self._evaluate_cap_defensive_fire_window(
            aid,
            target_id,
            env=env,
            current_time=current_time,
        )
        if not bool(window.get("eligible", False)):
            return False

        distance_km = float(window.get("distance_km", float("nan")))
        return np.isfinite(distance_km)

    def _is_agent_target_pair_compatible(
        self,
        agent_id: Optional[str],
        target_id: Optional[str],
        env=None,
        current_time: Optional[float] = None,
        immediate_cross_range_km: float = 45.0,
    ) -> bool:
        aid = str(agent_id or "").strip()
        tid = str(target_id or "").strip()
        if not aid.startswith("A") or not tid.startswith("B"):
            return True

        shooter_pair = self._get_cap_pair_name(aid)
        target_pair = self._get_cap_pair_name(tid)
        if shooter_pair is None or target_pair is None or shooter_pair == target_pair:
            return True

        env = env or self._resolve_picture_env()
        agents = getattr(env, "agents", None) if env is not None else None
        if not isinstance(agents, dict):
            return False

        native_friendlies = [
            fid
            for fid in self._get_native_friendlies_for_target(tid)
            if fid in agents and getattr(agents[fid], "is_alive", False)
        ]
        if not native_friendlies:
            return True

        target_status = self._get_picture_target_actionability(
            tid,
            current_time=current_time,
            purpose="decision",
        )
        if not bool(target_status.get("actionable", True)):
            return False

        support_window = self._evaluate_cap_defensive_fire_window(
            aid,
            tid,
            env=env,
            current_time=current_time,
        )
        target_zone = str(support_window.get("target_zone", target_status.get("target_zone", "UNKNOWN")) or "UNKNOWN").upper()
        native_armed_friendlies = [
            fid
            for fid in native_friendlies
            if self._get_cap_agent_missiles_left(agents.get(fid)) > 0
        ]
        if not native_armed_friendlies and bool(support_window.get("eligible", False)):
            support_distance_km = float(support_window.get("distance_km", float("nan")))
            cross_pair_limit_km = 66.0 if target_zone == "HIGH" else 60.0
            if np.isfinite(support_distance_km) and support_distance_km <= cross_pair_limit_km:
                return True
        if target_zone != "HIGH":
            return False

        resolved_tid = tid
        if hasattr(env, "resolve_real_agent_id"):
            try:
                candidate = env.resolve_real_agent_id(tid)
                if candidate:
                    resolved_tid = str(candidate)
            except Exception:
                resolved_tid = tid

        shooter = agents.get(aid)
        target = agents.get(resolved_tid) or agents.get(tid)
        if shooter is None or target is None:
            return False

        try:
            shooter_pos = np.asarray(shooter.get_position(), dtype=float)
            target_pos = np.asarray(target.get_position(), dtype=float)
            distance_km = float(np.linalg.norm(target_pos - shooter_pos)) / 1000.0
        except Exception:
            return False
        return np.isfinite(distance_km) and distance_km <= float(immediate_cross_range_km)

    def _get_picture_target_actionability(
        self,
        target_id: Optional[str],
        current_time: Optional[float] = None,
        purpose: str = "decision",
    ) -> Dict[str, object]:
        tid = str(target_id or "").strip()
        purpose_name = str(purpose or "decision").strip().lower()
        now = float(self._get_current_sim_time() if current_time is None else current_time)
        info: Dict[str, object] = {
            "target_id": tid,
            "purpose": purpose_name,
            "actionable": True,
            "enemy_phase": "",
            "group_id": "",
            "group_phase": "",
            "target_zone": "UNKNOWN",
            "target_depth_km": float("nan"),
            "reasons": [],
        }
        if not tid or not tid.startswith("B") or purpose_name == "raw":
            return info

        resolved_tid = tid
        env = self._resolve_picture_env()
        if env is not None and hasattr(env, "resolve_real_agent_id"):
            try:
                candidate = env.resolve_real_agent_id(tid)
                if candidate:
                    resolved_tid = str(candidate)
            except Exception:
                resolved_tid = tid

        adapter = getattr(self, "enemy_adapter", None)
        combat_ai = getattr(adapter, "_combat_ai", None) if adapter is not None else None
        phases = getattr(combat_ai, "_enemy_phases", None) if combat_ai is not None else None
        if isinstance(phases, dict):
            info["enemy_phase"] = str(phases.get(resolved_tid, phases.get(tid, "")) or "").upper()

        agent_contexts = getattr(adapter, "_agent_contexts", None) if adapter is not None else None
        context = None
        if isinstance(agent_contexts, dict):
            context = agent_contexts.get(resolved_tid) or agent_contexts.get(tid)
        if context is not None:
            info["group_id"] = str(getattr(context, "group_id", "") or "")

        group_states = getattr(adapter, "_group_states", None) if adapter is not None else None
        if not info.get("group_id") and adapter is not None and hasattr(adapter, "get_group_snapshot"):
            try:
                group_snapshot = adapter.get_group_snapshot(resolved_tid) or adapter.get_group_snapshot(tid)
            except Exception:
                group_snapshot = {}
            if isinstance(group_snapshot, dict):
                info["group_id"] = str(group_snapshot.get("group_id", "") or "")
                info["group_phase"] = str(group_snapshot.get("phase", "") or "").upper()
                info["group_pressure_level"] = int(group_snapshot.get("pressure_level", 0) or 0)
                info["group_wave_index"] = int(group_snapshot.get("wave_index", 0) or 0)
        group_id = str(info.get("group_id", "") or "")
        if group_id and isinstance(group_states, dict):
            group_state = group_states.get(group_id)
            if group_state is not None:
                info["group_phase"] = str(getattr(group_state, "phase", "") or "").upper()
                info["group_pressure_level"] = int(getattr(group_state, "pressure_level", 0) or 0)
                info["group_wave_index"] = int(getattr(group_state, "wave_index", 0) or 0)

        info.update(self._get_picture_target_geometry(tid))

        north_phase_status: Dict[str, object] = {}
        if adapter is not None and hasattr(adapter, "evaluate_north_phase_status"):
            try:
                north_phase_status = adapter.evaluate_north_phase_status(
                    resolved_tid,
                    target_y_km=float(info.get("target_depth_km", float("nan"))),
                    target_zone=str(info.get("target_zone", "UNKNOWN")),
                ) or {}
            except Exception:
                try:
                    north_phase_status = adapter.evaluate_north_phase_status(
                        tid,
                        target_y_km=float(info.get("target_depth_km", float("nan"))),
                        target_zone=str(info.get("target_zone", "UNKNOWN")),
                    ) or {}
                except Exception:
                    north_phase_status = {}
        if north_phase_status:
            north_phase = str(north_phase_status.get("phase", "") or "").upper()
            if north_phase and not str(info.get("group_phase", "") or ""):
                info["group_phase"] = north_phase
            if "group_pressure_level" not in info:
                info["group_pressure_level"] = int(north_phase_status.get("pressure_level", 0) or 0)
            if not str(info.get("group_id", "") or ""):
                info["group_id"] = str(north_phase_status.get("group_id", "") or "")
            info["group_forward_limit_km"] = float(north_phase_status.get("forward_limit_y_km", float("nan")))
            info["north_phase_disengaged"] = bool(north_phase_status.get("disengaged", False))
            info["north_phase_pressing_override"] = bool(north_phase_status.get("pressing_override", False))

        reasons: List[str] = []
        target_zone = str(info.get("target_zone", "UNKNOWN") or "UNKNOWN").upper()
        target_depth_km = float(info.get("target_depth_km", float("nan")))
        immediate_defense_zone = bool(
            target_zone == "HIGH"
            or (
                target_zone == "MEDIUM"
                and np.isfinite(target_depth_km)
                and target_depth_km <= 112.0
            )
        )
        info["immediate_defense_zone"] = immediate_defense_zone

        if str(info.get("enemy_phase", "") or "") == "RETURNING":
            reasons.append("enemy_returning")
        group_phase = str(
            north_phase_status.get("phase", "")
            or info.get("group_phase", "")
            or ""
        ).upper()
        group_pressure_level = int(info.get("group_pressure_level", 0) or 0)
        north_phase_distance_km = float(north_phase_status.get("distance_km", float("nan")) or float("nan"))
        regroup_pressure_release = bool(
            group_phase == "REGROUP_NORTH"
            and (
                bool(north_phase_status.get("pressing_override", False))
                or (
                    purpose_name == "decision"
                    and group_pressure_level >= 1
                    and np.isfinite(north_phase_distance_km)
                    and north_phase_distance_km <= 50.0
                    and target_zone in {"MEDIUM", "HIGH"}
                )
            )
        )
        if group_phase == "REGROUP_NORTH":
            if not immediate_defense_zone and not regroup_pressure_release:
                reasons.append("group_regroup_north")
        elif group_phase == "TURN_NORTH":
            if not immediate_defense_zone:
                reasons.append("group_turn_north")
            elif north_phase_status:
                if not bool(north_phase_status.get("pressing_override", False)) and group_pressure_level < 1:
                    reasons.append("group_turn_north")
            elif group_pressure_level < 2:
                reasons.append("group_turn_north")

        if reasons:
            info["actionable"] = False
            info["reasons"] = reasons

        log_state = getattr(self, "_picture_target_actionability_log", None)
        if not isinstance(log_state, dict):
            log_state = {}
            setattr(self, "_picture_target_actionability_log", log_state)
        key = (tid, purpose_name)
        signature = (
            bool(info.get("actionable", True)),
            str(info.get("enemy_phase", "")),
            str(info.get("group_id", "")),
            str(info.get("group_phase", "")),
            tuple(info.get("reasons", [])),
        )
        last_signature, last_time = log_state.get(key, (None, -1e9))
        should_log_filter = (
            not bool(info.get("actionable", True))
            and (signature != last_signature or (now - float(last_time)) >= 20.0)
        )
        should_log_restore = (
            last_signature is not None
            and bool(last_signature[0]) is False
            and bool(info.get("actionable", True))
        )
        if should_log_filter:
            logging.info(
                "[PICTURE_THREAT_FILTER] target=%s purpose=%s enemy_phase=%s group=%s group_phase=%s reasons=%s",
                tid,
                purpose_name,
                str(info.get("enemy_phase", "") or "-"),
                str(info.get("group_id", "") or "-"),
                str(info.get("group_phase", "") or "-"),
                ",".join(info.get("reasons", [])) or "-",
            )
            log_state[key] = (signature, now)
        elif should_log_restore:
            logging.info(
                "[PICTURE_THREAT_RESTORE] target=%s purpose=%s enemy_phase=%s group=%s group_phase=%s",
                tid,
                purpose_name,
                str(info.get("enemy_phase", "") or "-"),
                str(info.get("group_id", "") or "-"),
                str(info.get("group_phase", "") or "-"),
            )
            log_state[key] = (signature, now)
        elif signature != last_signature:
            log_state[key] = (signature, now)
        return info

    def _filter_picture_threats(
        self,
        threats,
        current_time: Optional[float] = None,
        purpose: str = "decision",
    ):
        purpose_name = str(purpose or "decision").strip().lower()
        if purpose_name == "raw":
            return list(threats)
        filtered = []
        for threat in list(threats):
            target_id = getattr(threat, "track_id", None)
            status = self._get_picture_target_actionability(
                target_id,
                current_time=current_time,
                purpose=purpose_name,
            )
            if bool(status.get("actionable", True)):
                filtered.append(threat)
        return filtered

    def _get_picture_threats(self, current_time: Optional[float] = None, purpose: str = "decision"):
        picture = getattr(self, "picture", None)
        if picture is None:
            return []
        if current_time is None:
            current_time = self._get_current_sim_time()
        kwargs = self._get_picture_filter_kwargs(purpose)
        if not kwargs:
            threats = list(picture.get_all_threats())
        else:
            threats = list(picture.get_all_threats(current_time=current_time, **kwargs))
        return self._filter_picture_threats(threats, current_time=current_time, purpose=purpose)

    def _get_picture_threat_by_id(
        self,
        track_id: Optional[str],
        current_time: Optional[float] = None,
        purpose: str = "decision",
    ):
        if not track_id:
            return None
        for threat in self._get_picture_threats(current_time=current_time, purpose=purpose):
            if getattr(threat, "track_id", None) == track_id:
                return threat
        if str(purpose or "").strip().lower() == "decision":
            for threat in self._get_picture_threats(current_time=current_time, purpose="search"):
                if getattr(threat, "track_id", None) == track_id:
                    return threat
        return None

    def _get_picture_nearest_threat(
        self,
        ref_pos: np.ndarray,
        current_time: Optional[float] = None,
        purpose: str = "decision",
    ):
        if current_time is None:
            current_time = self._get_current_sim_time()
        threats = self._get_picture_threats(current_time=current_time, purpose=purpose)
        if not threats and str(purpose or "").strip().lower() == "decision":
            threats = self._get_picture_threats(current_time=current_time, purpose="search")
        if not threats:
            return None

        ref_xy = np.asarray(ref_pos[:2], dtype=float)
        return min(
            threats,
            key=lambda threat: float(
                np.hypot(float(getattr(threat, "x", 0.0)) - ref_xy[0], float(getattr(threat, "y", 0.0)) - ref_xy[1])
            ),
        )

    def _normalize_picture_zone(self, zone) -> PictureRiskZone:
        if isinstance(zone, PictureRiskZone):
            return zone
        zone_name = getattr(zone, "name", None)
        if zone_name is None:
            zone_name = getattr(zone, "value", zone)
        zone_name = str(zone_name).split(".")[-1].strip().upper()
        return PictureRiskZone.__members__.get(zone_name, PictureRiskZone.OUTSIDE)

    def _get_picture_zone_threats(
        self,
        zone,
        current_time: Optional[float] = None,
        purpose: str = "decision",
    ):
        picture = getattr(self, "picture", None)
        if picture is None:
            return []
        if current_time is None:
            current_time = self._get_current_sim_time()
        kwargs = self._get_picture_filter_kwargs(purpose)
        zone_value = self._normalize_picture_zone(zone)
        if not kwargs:
            threats = list(picture.get_threats_in_zone(zone_value))
        else:
            threats = list(picture.get_threats_in_zone(zone_value, current_time=current_time, **kwargs))
        return self._filter_picture_threats(threats, current_time=current_time, purpose=purpose)

    def _get_picture_zone_count(
        self,
        zone,
        current_time: Optional[float] = None,
        purpose: str = "decision",
    ) -> int:
        return len(self._get_picture_zone_threats(zone, current_time=current_time, purpose=purpose))
