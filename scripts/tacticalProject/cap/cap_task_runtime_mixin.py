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

    def _has_stable_dual_tracking(self, target_id: str) -> bool:
        return _capgh._has_stable_dual_tracking(self, target_id)

    def check_external_prelaunch_gate(self, target_id: str, current_time: float) -> Tuple[bool, Dict[str, object]]:
        return _capgh.check_external_prelaunch_gate(self, target_id, current_time)

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
            "reasons": [],
        }
        if not tid or not tid.startswith("B") or purpose_name == "raw":
            return info

        adapter = getattr(self, "enemy_adapter", None)
        combat_ai = getattr(adapter, "_combat_ai", None) if adapter is not None else None
        phases = getattr(combat_ai, "_enemy_phases", None) if combat_ai is not None else None
        if isinstance(phases, dict):
            info["enemy_phase"] = str(phases.get(tid, "") or "").upper()

        agent_contexts = getattr(adapter, "_agent_contexts", None) if adapter is not None else None
        context = agent_contexts.get(tid) if isinstance(agent_contexts, dict) else None
        if context is not None:
            info["group_id"] = str(getattr(context, "group_id", "") or "")

        group_states = getattr(adapter, "_group_states", None) if adapter is not None else None
        group_id = str(info.get("group_id", "") or "")
        if group_id and isinstance(group_states, dict):
            group_state = group_states.get(group_id)
            if group_state is not None:
                info["group_phase"] = str(getattr(group_state, "phase", "") or "").upper()

        reasons: List[str] = []
        if str(info.get("enemy_phase", "") or "") == "RETURNING":
            reasons.append("enemy_returning")
        if str(info.get("group_phase", "") or "") in ("TURN_NORTH", "REGROUP_NORTH"):
            reasons.append(f"group_{str(info['group_phase']).lower()}")

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
