"""
敌方 AI 适配器。

设计目标：
1. 严格把敌机限制在 FAOR 200km x 300km 内；
2. 用确定性的“编队波次”控制敌机，而不是让其持续压入近距；
3. 保留原始双机编队优先，编队残缺时自动重编组；
4. 呈现清晰的战术节奏：南向逼近 -> 180° 北向拉开 -> 北向重整 ->
   180° 南向再攻击。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    from unified_enemy_tactical_ai import UnifiedEnemyTacticalAI
except Exception:
    UnifiedEnemyTacticalAI = None


try:
    from new_enemy_maneuver_ai import NewEnemyManeuverAI
except Exception:
    NewEnemyManeuverAI = None

try:
    from envs.JSBSim.core.catalog import Catalog as c
except ImportError:
    class _MockCatalog:
        attitude_heading_true_rad = "attitude/heading-true-rad"
        position_h_sl_m = "position/h-sl-m"

    c = _MockCatalog()


class PairAwareUnifiedEnemyAI(UnifiedEnemyTacticalAI if UnifiedEnemyTacticalAI is not None else object):
    """在保留原敌方战术AI的前提下，注入动态编队/目标约束。"""

    def __init__(self, context_provider, preferred_target_provider, target_pool_provider):
        self._context_provider = context_provider
        self._preferred_target_provider = preferred_target_provider
        self._target_pool_provider = target_pool_provider
        if UnifiedEnemyTacticalAI is not None:
            super().__init__()

    def _get_enemy_teammate_id(self, agent_id: str) -> Optional[str]:
        context = self._context_provider(agent_id)
        if context is not None and len(context.members) == 2:
            for member in context.members:
                if member != agent_id:
                    return member
        if UnifiedEnemyTacticalAI is None:
            return None
        return super()._get_enemy_teammate_id(agent_id)

    def _get_preferred_target_id(self, agent_id: str) -> Optional[str]:
        dynamic_target = self._preferred_target_provider(agent_id)
        if dynamic_target:
            return dynamic_target
        if UnifiedEnemyTacticalAI is None:
            return None
        return super()._get_preferred_target_id(agent_id)

    def _get_engagement_target_ids(self, env, agent_id: str) -> List[str]:
        candidate_ids = [
            target_id
            for target_id in self._target_pool_provider(agent_id)
            if target_id in env.agents and getattr(env.agents[target_id], "is_alive", False)
        ]
        close_target_id = self._get_close_priority_target_id(env, agent_id)
        if close_target_id is not None:
            return [close_target_id]
        preferred_target_id = self._get_preferred_target_id(agent_id)
        if preferred_target_id in candidate_ids:
            return [preferred_target_id]
        if candidate_ids:
            return candidate_ids
        if UnifiedEnemyTacticalAI is None:
            return []
        return super()._get_engagement_target_ids(env, agent_id)

    def _find_best_target(self, env, agent_id: str):
        candidate_ids = [
            target_id
            for target_id in self._target_pool_provider(agent_id)
            if target_id in env.agents and getattr(env.agents[target_id], "is_alive", False)
        ]
        close_target_id = self._get_close_priority_target_id(env, agent_id)
        if close_target_id is not None and close_target_id in env.agents:
            return env.agents[close_target_id]
        if not candidate_ids or UnifiedEnemyTacticalAI is None:
            if UnifiedEnemyTacticalAI is None:
                return None
            return super()._find_best_target(env, agent_id)

        current_pos = np.array(env.agents[agent_id].get_position(), dtype=float)
        preferred_target_id = self._get_preferred_target_id(agent_id)
        if preferred_target_id in candidate_ids:
            preferred_target = env.agents[preferred_target_id]
            distance = float(np.linalg.norm(current_pos - np.array(preferred_target.get_position(), dtype=float)))
            if 20000.0 <= distance <= 100000.0:
                return preferred_target

        best_target = None
        best_distance = float("inf")
        for target_id in candidate_ids:
            target = env.agents[target_id]
            distance = float(np.linalg.norm(current_pos - np.array(target.get_position(), dtype=float)))
            if distance < best_distance:
                best_distance = distance
                best_target = target
        return best_target

    def _get_close_priority_target_id(self, env, agent_id: str) -> Optional[str]:
        current_aircraft = env.agents.get(agent_id)
        if current_aircraft is None or not getattr(current_aircraft, "is_alive", False):
            return None
        current_pos = np.array(current_aircraft.get_position(), dtype=float)
        best_target_id = None
        best_distance = float("inf")
        for target_id in ("A0100", "A0200", "A0300", "A0400"):
            target = env.agents.get(target_id)
            if target is None or not getattr(target, "is_alive", False):
                continue
            distance = float(np.linalg.norm(current_pos - np.array(target.get_position(), dtype=float)))
            if distance < best_distance:
                best_distance = distance
                best_target_id = target_id
        if best_target_id is not None and best_distance <= 50000.0:
            return best_target_id
        return None


ENEMY_DEFAULT_PAIRS: Dict[str, Tuple[str, str]] = {
    "PAIR_LEFT": ("B0100", "B0200"),
    "PAIR_RIGHT": ("B0300", "B0400"),
}
FRIENDLY_DEFAULT_PAIRS: Dict[str, Tuple[str, str]] = {
    "LEFT": ("A0100", "A0200"),
    "RIGHT": ("A0300", "A0400"),
}
ENEMY_DEFAULT_SECTOR_BY_PAIR: Dict[str, str] = {
    "PAIR_LEFT": "LEFT",
    "PAIR_RIGHT": "RIGHT",
}
ENEMY_CALL_ORDER: Tuple[str, ...] = tuple(
    agent_id for members in ENEMY_DEFAULT_PAIRS.values() for agent_id in members
)
ENEMY_DEFAULT_PAIR_BY_AGENT: Dict[str, str] = {
    agent_id: pair_id
    for pair_id, members in ENEMY_DEFAULT_PAIRS.items()
    for agent_id in members
}
ENEMY_DEFAULT_SECTOR_BY_AGENT: Dict[str, str] = {
    agent_id: ENEMY_DEFAULT_SECTOR_BY_PAIR[pair_id]
    for pair_id, members in ENEMY_DEFAULT_PAIRS.items()
    for agent_id in members
}


@dataclass
class EnemyBounds:
    """敌方活动边界（km）。"""

    x_min: float = 0.0
    x_max: float = 200.0
    y_min: float = 0.0
    y_max: float = 300.0


@dataclass
class EnemyWaveState:
    """敌方编队波次状态。"""

    phase: str = "APPROACH"
    phase_enter_time: float = 0.0
    wave_index: int = 1
    trigger_reason: str = ""
    regroup_y_km: float = 0.0
    last_update_time: float = -1.0
    approach_hold_until: float = 0.0
    turn_entry_distance_km: float = 46.0
    resume_distance_km: float = 80.0
    recovery_hold_until: float = 0.0
    turn_north_entry_y_km: float = 0.0
    turn_south_entry_y_km: float = 0.0
    last_status_log_time: float = -999.0
    last_regroup_diag_log_time: float = -999.0
    forward_limit_y_km: float = 100.0
    regroup_timeout_s: float = 12.0
    regroup_min_hold_s: float = 8.0
    turn_south_min_hold_s: float = 6.0
    north_depart_margin_km: float = 16.0
    pressure_level: int = 0
    pressure_tag: str = "balanced"
    enemy_alive_total: int = 4
    friendly_alive_total: int = 4
    local_advantage: int = 0
    hard_standoff_distance_km: float = 36.0
    enemy_missiles_total: int = 16
    friendly_missiles_total: int = 16
    local_enemy_missiles: int = 8
    local_friendly_missiles: int = 8
    missile_advantage: int = 0
    missile_superiority: bool = False
    finish_exploit: bool = False


@dataclass
class EnemyGroupKinematics:
    """缂栭槦鑳介噺/濮挎€佸揩鐓э紝鐢ㄤ簬闃舵鍒囨崲涓庢仮澶嶅垽瀹氥€?"""

    min_altitude_m: float = float("inf")
    min_speed_mps: float = float("inf")
    min_vertical_speed_mps: float = float("inf")
    min_pitch_deg: float = float("inf")
    max_abs_roll_deg: float = 0.0


@dataclass
class EnemyFormationContext:
    """当前时刻的敌方编队快照。"""

    group_id: str
    members: Tuple[str, ...]
    anchor_x_km: float
    attack_anchor_x_km: float
    initial_center_y_km: float
    desired_spacing_km: float
    slot_x_by_agent: Dict[str, float]
    mean_x_km: float
    mean_y_km: float
    sector_id: str
    sector_x_min_km: float
    sector_x_max_km: float
    target_friendly_ids: Tuple[str, ...]
    is_dynamic_group: bool = False


class EnemyAIAdapter:
    """敌方机动适配器。"""

    def __init__(self):
        self.bounds = EnemyBounds()
        self._last_log_time: Dict[str, float] = {}
        self._last_pair_log_time: Dict[str, float] = {}
        self._initial_slots_km: Dict[str, Tuple[float, float]] = {}
        self._group_states: Dict[str, EnemyWaveState] = {}
        self._agent_contexts: Dict[str, EnemyFormationContext] = {}
        self._cached_group_time: float = -1.0
        self._last_seen_time: float = -1.0
        self._last_group_signature: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
        self._bounds_initialized = False
        self._rootcause_last_step: Dict[Tuple[str, str], int] = {}
        self._lowlevel_override_flags: Dict[str, bool] = {}

        self._boundary_margin_km = float(os.getenv("ENEMY_BOUNDARY_MARGIN_KM", "12"))
        self._turnback_distance_km = float(os.getenv("ENEMY_TURNBACK_DISTANCE_KM", "42"))
        self._hard_standoff_distance_km = float(os.getenv("ENEMY_HARD_STANDOFF_DISTANCE_KM", "36"))
        self._resume_distance_km = float(os.getenv("ENEMY_REATTACK_DISTANCE_KM", "68"))
        self._pullback_distance_km = float(os.getenv("ENEMY_PULLBACK_DISTANCE_KM", "28"))
        self._regroup_timeout_s = float(os.getenv("ENEMY_REGROUP_TIMEOUT_S", "12"))
        self._regroup_min_hold_s = float(os.getenv("ENEMY_REGROUP_MIN_HOLD_S", "8"))
        self._turn_phase_s = float(os.getenv("ENEMY_TURN_PHASE_S", "5"))
        self._reattack_hold_s = float(os.getenv("ENEMY_REATTACK_HOLD_S", "2"))
        self._turn_south_min_hold_s = float(os.getenv("ENEMY_TURN_SOUTH_MIN_HOLD_S", "6"))
        self._regroup_depart_margin_km = float(os.getenv("ENEMY_REGROUP_DEPART_MARGIN_KM", "16"))
        self._regroup_release_distance_km = float(os.getenv("ENEMY_REGROUP_RELEASE_DISTANCE_KM", "64"))
        self._turn_south_depart_margin_km = float(os.getenv("ENEMY_TURN_SOUTH_DEPART_MARGIN_KM", "3"))
        self._regroup_extend_step_km = float(os.getenv("ENEMY_REGROUP_EXTEND_STEP_KM", "5"))
        self._regroup_altitude_m = float(os.getenv("ENEMY_REGROUP_ALTITUDE_M", "8200"))
        self._desired_pair_spacing_km = float(os.getenv("ENEMY_FORMATION_SPACING_KM", "10"))
        # 默认启用双编队波次控制；进攻段仍然复用敌方原始战术 AI。
        self._recovery_hold_s = float(os.getenv("ENEMY_RECOVERY_HOLD_S", "5"))
        self._energy_turn_speed_mps = float(os.getenv("ENEMY_ENERGY_TURN_SPEED_MPS", "205"))
        self._energy_critical_speed_mps = float(os.getenv("ENEMY_ENERGY_CRITICAL_SPEED_MPS", "185"))
        self._energy_floor_altitude_m = float(os.getenv("ENEMY_ENERGY_FLOOR_ALTITUDE_M", "6800"))
        self._energy_descent_trigger_mps = float(os.getenv("ENEMY_ENERGY_DESCENT_TRIGGER_MPS", "-14"))
        self._energy_pitch_trigger_deg = float(os.getenv("ENEMY_ENERGY_PITCH_TRIGGER_DEG", "-8"))
        self._energy_roll_trigger_deg = float(os.getenv("ENEMY_ENERGY_ROLL_TRIGGER_DEG", "78"))
        self._disable_wave_mode = os.getenv("ENEMY_DISABLE_WAVE_MODE", "1") != "0"
        self._direct_rtb_enabled = os.getenv("CAP_ENEMY_DIRECT_RTB_ENABLED", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._direct_rtb_distance_km = float(
            os.getenv("CAP_ENEMY_DIRECT_RTB_DISTANCE_KM", str(self._turnback_distance_km))
        )
        self._use_new_enemy_ai = os.getenv("CAP_NEW_ENEMY_MANEUVER_AI_ENABLED", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._use_pairaware_unified = os.getenv("CAP_ENEMY_USE_PAIRAWARE_UNIFIED_ENABLED", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        shared_buffer_km = float(os.getenv("ENEMY_SHARED_BUFFER_KM", "40"))
        midline_y_km = 0.5 * (self.bounds.y_min + self.bounds.y_max)
        self._forward_limit_y_km = float(
            os.getenv(
                "ENEMY_FORWARD_LIMIT_Y_KM",
                str(midline_y_km + 0.5 * shared_buffer_km),
            )
        )
        if self._use_new_enemy_ai and NewEnemyManeuverAI is not None:
            self._combat_ai = NewEnemyManeuverAI(project_name="cap_tactical_project")
        else:
            self._combat_ai = (
                (
                    PairAwareUnifiedEnemyAI(
                        self._get_context_for_agent,
                        self._get_dynamic_preferred_target_id,
                        self._get_dynamic_target_pool,
                    )
                    if self._use_pairaware_unified
                    else UnifiedEnemyTacticalAI()
                )
                if UnifiedEnemyTacticalAI is not None
                else None
            )
        ai_name = type(self._combat_ai).__name__ if self._combat_ai is not None else "None"
        log.info(
            "[敌方战术AI] default_path=%s ai=%s bridge_enabled=%s wave_mode=%s unified_variant=%s",
            "new_enemy_ai" if self._use_new_enemy_ai and NewEnemyManeuverAI is not None else "unified_enemy_ai",
            ai_name,
            os.getenv("CAP_ENEMY_FRIENDLY_BRIDGE_ENABLED", "0"),
            "off" if self._disable_wave_mode else "on",
            "pairaware" if self._use_pairaware_unified else "raw",
        )

        self.norm_hdg = np.array(
            [
                -np.pi,
                -2 * np.pi / 3,
                -np.pi / 2,
                -5 * np.pi / 12,
                -np.pi / 3,
                -np.pi / 4,
                -np.pi / 6,
                -np.pi / 12,
                0,
                np.pi / 12,
                np.pi / 6,
                np.pi / 4,
                np.pi / 3,
                5 * np.pi / 12,
                np.pi / 2,
                2 * np.pi / 3,
                np.pi,
            ]
        )

    def _rootcause_enabled(self) -> bool:
        return str(os.getenv("CAP_ROOTCAUSE_TRACE", "0")).strip().lower() in ("1", "true", "yes", "on")

    def _rootcause_log(
        self,
        env,
        agent_id: str,
        key: str,
        message: str,
        interval_steps: int = 30,
        force: bool = False,
    ) -> None:
        if not self._rootcause_enabled():
            return
        step = int(getattr(env, "current_step", 0))
        throttle_key = (agent_id, key)
        last_step = self._rootcause_last_step.get(throttle_key, -10**9)
        if not force and step - last_step < interval_steps:
            return
        self._rootcause_last_step[throttle_key] = step
        log.warning("🧩 [T+%.1fs|step=%d] [根因链-%s] %s %s", step * 0.2, step, key, agent_id, message)

    def reset(self, env=None):
        """为新回合重置状态。"""
        self._last_log_time.clear()
        self._last_pair_log_time.clear()
        self._initial_slots_km.clear()
        self._group_states.clear()
        self._agent_contexts.clear()
        self._cached_group_time = -1.0
        self._last_seen_time = -1.0
        self._last_group_signature = ()
        self._bounds_initialized = False
        if self._combat_ai is not None:
            for agent_id in ENEMY_CALL_ORDER:
                try:
                    self._combat_ai.reset_agent(agent_id)
                except Exception:
                    pass

        if env is not None:
            self._ensure_initial_slots(env)

    def get_enemy_action(
        self,
        env,
        agent_id: str,
        current_time: float,
        threat_distance: float = 300.0,
        task=None,
    ) -> Tuple[int, int, int]:
        """获取敌机离散机动动作。"""
        aircraft = env.agents.get(agent_id)
        if aircraft is None or not getattr(aircraft, "is_alive", False):
            return 7, 8, 3

        if self._last_seen_time >= 0.0 and current_time + 1e-6 < self._last_seen_time:
            self.reset(env)
        self._last_seen_time = current_time

        self._ensure_initial_slots(env)

        if self._use_new_enemy_ai and NewEnemyManeuverAI is not None and isinstance(self._combat_ai, NewEnemyManeuverAI):
            try:
                cmd = self._combat_ai.get_enemy_command(env, agent_id, current_time, task)
            except TypeError:
                cmd = self._combat_ai.get_enemy_command(env, agent_id, current_time)
            except Exception as exc:
                log.warning("[敌方战术AI] %s 新AI调用失败: %s", agent_id, exc)
                cmd = self._default_action(env, agent_id)

            self._rootcause_log(
                env,
                agent_id,
                "适配器",
                f"path=new_ai raw_cmd=({int(cmd[0])},{int(cmd[1])},{int(cmd[2])}) ai={type(self._combat_ai).__name__}",
                interval_steps=20,
            )
            return self._apply_overlay_safety(env, agent_id, cmd, task, phase="NEW_AI")

        popup_cmd = self._maybe_execute_popup_behavior(aircraft, agent_id, threat_distance)
        if popup_cmd is not None:
            return popup_cmd

        self._sync_formations(env, current_time)
        if self._disable_wave_mode:
            self._maybe_force_direct_rtb(env, agent_id, current_time, threat_distance)
            self._rootcause_log(
                env,
                agent_id,
                "适配器",
                "path=unified_direct wave_mode=0",
                interval_steps=150,
            )
            return self._get_legacy_enemy_command(env, agent_id, current_time, task)

        context = self._agent_contexts.get(agent_id)
        if context is None:
            return self._default_action(env, agent_id)

        state = self._group_states.get(context.group_id)
        if state is None:
            initial_phase = self._initial_phase_for_group(env, context)
            state = EnemyWaveState(
                phase=initial_phase,
                phase_enter_time=current_time,
                regroup_y_km=self._compute_regroup_y_km(context),
            )
            self._refresh_wave_geometry(
                env,
                context,
                state,
                self._get_group_min_distance_km(env, context.members, context.target_friendly_ids),
            )
            self._group_states[context.group_id] = state
        """

        self._rootcause_log(
            env,
            agent_id,
            "閫傞厤鍣? ,
            f"path=wave group={context.group_id} phase={state.phase}",
            interval_steps=150,
        )

        """
        self._rootcause_log(
            env,
            agent_id,
            "adapter",
            f"path=wave group={context.group_id} phase={state.phase}",
            interval_steps=150,
        )
        self._update_group_state(env, context, state, current_time)
        if state.phase == "APPROACH":
            return self._get_combat_command(
                env,
                agent_id,
                current_time,
                task=task,
                context=context,
                state=state,
            )
        return self._build_group_command(env, agent_id, context, state, task)

    def get_lowlevel_override_raw(self, env, agent_id: str):
        aircraft = env.agents.get(agent_id)
        if aircraft is None or not getattr(aircraft, "is_alive", False):
            self._lowlevel_override_flags[agent_id] = False
            return None

        combat_ai = self._combat_ai
        if combat_ai is None:
            self._lowlevel_override_flags[agent_id] = False
            return None

        phases = getattr(combat_ai, "_enemy_phases", None)
        return_states = getattr(combat_ai, "return_states", None)
        if not isinstance(phases, dict) or not isinstance(return_states, dict):
            self._lowlevel_override_flags[agent_id] = False
            return None
        if phases.get(agent_id) != "RETURNING":
            self._lowlevel_override_flags[agent_id] = False
            return None

        state = return_states.get(agent_id)
        if not isinstance(state, dict) or state.get("phase") != "break_turn_180":
            self._lowlevel_override_flags[agent_id] = False
            return None

        current_heading = self._get_heading_deg(aircraft)
        target_heading = float(state.get("break_target_heading", 0.0))
        heading_diff = self._normalize_heading_error(target_heading - current_heading)
        if abs(heading_diff) <= 12.0:
            self._lowlevel_override_flags[agent_id] = False
            return None

        current_speed = self._get_speed_mps(aircraft)
        current_alt = self._get_altitude_m(aircraft)
        strong_turn = abs(heading_diff) > 60.0
        if heading_diff > 0.0:
            aileron_bin = 38 if strong_turn else 34
            rudder_bin = 32 if strong_turn else 28
        else:
            aileron_bin = 2 if strong_turn else 6
            rudder_bin = 8 if strong_turn else 12

        if current_alt > 4500.0 and current_speed > 170.0:
            elevator_bin = 18  # slight unload during the break turn
        elif current_speed < 155.0:
            elevator_bin = 20  # avoid over-unloading at low speed
        else:
            elevator_bin = 19
        throttle_bin = 29

        self._lowlevel_override_flags[agent_id] = True
        return np.array([aileron_bin, elevator_bin, rudder_bin, throttle_bin], dtype=np.int64)

    def consume_lowlevel_override_active(self, agent_id: str) -> bool:
        return bool(self._lowlevel_override_flags.pop(agent_id, False))

    def _maybe_force_direct_rtb(
        self,
        env,
        agent_id: str,
        current_time: float,
        threat_distance: float,
    ) -> bool:
        if not self._direct_rtb_enabled:
            return False
        if threat_distance is None or not np.isfinite(threat_distance):
            return False
        if float(threat_distance) > float(self._direct_rtb_distance_km):
            return False
        if current_time < 30.0:
            return False
        combat_ai = self._combat_ai
        if combat_ai is None:
            return False
        phases = getattr(combat_ai, "_enemy_phases", None)
        if phases is None:
            return False
        if phases.get(agent_id) == "RETURNING":
            return True

        phases[agent_id] = "RETURNING"
        if hasattr(combat_ai, "_enemy_return_start_time"):
            combat_ai._enemy_return_start_time[agent_id] = current_time
        if hasattr(combat_ai, "_enemy_second_attack_done"):
            try:
                combat_ai._enemy_second_attack_done.add(agent_id)
            except Exception:
                pass
        if hasattr(combat_ai, "return_states"):
            try:
                combat_ai.return_states.pop(agent_id, None)
            except Exception:
                pass
        if hasattr(combat_ai, "_init_return_to_base_unified"):
            try:
                combat_ai._init_return_to_base_unified(agent_id, current_time)
            except Exception:
                pass

        log.warning(
            "🧭 [敌方近距强制返航] %s min_dist=%.1fkm threshold=%.1fkm current_time=%.1fs",
            agent_id,
            float(threat_distance),
            float(self._direct_rtb_distance_km),
            float(current_time),
        )
        self._rootcause_log(
            env,
            agent_id,
            "适配器",
            f"forced_direct_rtb min_dist={float(threat_distance):.1f}km threshold={float(self._direct_rtb_distance_km):.1f}km",
            interval_steps=10,
            force=True,
        )
        return True

    def _get_legacy_enemy_command(
        self,
        env,
        agent_id: str,
        current_time: float,
        task=None,
        phase_label: Optional[str] = None,
    ) -> Tuple[int, int, int]:
        if self._combat_ai is None:
            return self._default_action(env, agent_id)
        try:
            cmd = self._combat_ai.get_enemy_command(env, agent_id, current_time, task)
        except TypeError:
            cmd = self._combat_ai.get_enemy_command(env, agent_id, current_time)
        except Exception as exc:
            log.warning("[敌方战术AI] %s 回退模式调用失败: %s", agent_id, exc)
            return self._default_action(env, agent_id)
        path_label = "unified_direct" if self._disable_wave_mode and not self._use_new_enemy_ai else "legacy"
        self._rootcause_log(
            env,
            agent_id,
            "适配器",
            f"path={path_label} raw_cmd=({int(cmd[0])},{int(cmd[1])},{int(cmd[2])}) ai={type(self._combat_ai).__name__}",
            interval_steps=20,
        )
        return self._apply_overlay_safety(env, agent_id, cmd, task, phase_label)

    def _maybe_execute_popup_behavior(
        self,
        aircraft,
        agent_id: str,
        threat_distance: float,
    ) -> Optional[Tuple[int, int, int]]:
        behavior_mode = os.environ.get("ENEMY_BEHAVIOR", "")
        if behavior_mode != "POPUP":
            return None

        try:
            current_alt = float(aircraft.get_property_value(c.position_h_sl_m))
            if threat_distance > 220.0:
                if current_alt > 4100.0:
                    return 2, 8, 3
                if current_alt < 3900.0:
                    return 4, 8, 3
                return 3, 8, 3

            if current_alt < 10500.0:
                return 6, 8, 6
            return 3, 8, 6
        except Exception:
            return None

    def _ensure_initial_slots(self, env):
        for agent_id in ENEMY_CALL_ORDER:
            aircraft = env.agents.get(agent_id)
            if aircraft is None:
                continue
            if agent_id not in self._initial_slots_km:
                self._initial_slots_km[agent_id] = self._get_agent_xy_km(aircraft)

        if not self._bounds_initialized and self._initial_slots_km:
            self._refresh_dynamic_geometry(env)

    def _refresh_dynamic_geometry(self, env):
        enemy_points = list(self._initial_slots_km.values())
        friendly_points = []
        for agent_id in FRIENDLY_DEFAULT_PAIRS["LEFT"] + FRIENDLY_DEFAULT_PAIRS["RIGHT"]:
            aircraft = env.agents.get(agent_id)
            if aircraft is None:
                continue
            friendly_points.append(self._get_agent_xy_km(aircraft))

        all_points = enemy_points + friendly_points
        if not enemy_points or not all_points:
            return

        xs = [point[0] for point in all_points]
        enemy_ys = [point[1] for point in enemy_points]
        friendly_ys = [point[1] for point in friendly_points] if friendly_points else [0.0]
        x_pad = max(self._boundary_margin_km, 18.0)
        enemy_y_pad = max(self._boundary_margin_km + 12.0, 28.0)
        friendly_y_pad = max(12.0, 0.5 * self._turnback_distance_km)

        x_min = min(xs) - x_pad
        x_max = max(xs) + x_pad
        y_min = min(friendly_ys) - friendly_y_pad
        y_max = max(enemy_ys) + enemy_y_pad

        if (x_max - x_min) < 120.0:
            mid_x = 0.5 * (x_min + x_max)
            x_min = mid_x - 60.0
            x_max = mid_x + 60.0

        if (y_max - y_min) < 180.0:
            mid_y = 0.5 * (y_min + y_max)
            y_min = mid_y - 90.0
            y_max = mid_y + 90.0

        self.bounds = EnemyBounds(
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
        )
        friendly_mean_y = float(np.mean(friendly_ys)) if friendly_ys else 0.0
        self._forward_limit_y_km = max(
            friendly_mean_y + max(self._turnback_distance_km + 2.0, self._hard_standoff_distance_km + 6.0),
            100.0,
        )
        self._bounds_initialized = True
        log.info(
            "[敌方边界] x=[%.1f, %.1f]km y=[%.1f, %.1f]km | 前出线=%.1fkm",
            self.bounds.x_min,
            self.bounds.x_max,
            self.bounds.y_min,
            self.bounds.y_max,
            self._forward_limit_y_km,
        )

    def _sync_formations(self, env, current_time: float):
        if abs(self._cached_group_time - current_time) <= 1e-6:
            return
        self._cached_group_time = current_time

        formations = self._build_alive_formations(env)
        contexts: Dict[str, EnemyFormationContext] = {}
        active_group_ids = set()
        unique_contexts: Dict[str, EnemyFormationContext] = {}

        for members in formations:
            context = self._build_context(env, members)
            contexts.update({member: context for member in members})
            active_group_ids.add(context.group_id)
            unique_contexts[context.group_id] = context

        self._agent_contexts = contexts

        stale_group_ids = [group_id for group_id in self._group_states if group_id not in active_group_ids]
        for group_id in stale_group_ids:
            self._group_states.pop(group_id, None)

        signature = tuple(sorted((group_id, context.members) for group_id, context in unique_contexts.items()))
        if signature != self._last_group_signature:
            log.info(
                "[敌方重编组] %s",
                " | ".join(f"{group_id}:{'/'.join(members)}" for group_id, members in signature) if signature else "无存活敌机",
            )
            self._last_group_signature = signature

    def _build_alive_formations(self, env) -> List[Tuple[str, ...]]:
        alive_ids = {
            agent_id
            for agent_id in ENEMY_CALL_ORDER
            if agent_id in env.agents and getattr(env.agents[agent_id], "is_alive", False)
        }
        groups: List[Tuple[str, ...]] = []
        leftovers: List[str] = []
        for pair_id, members in ENEMY_DEFAULT_PAIRS.items():
            alive_members = [member for member in members if member in alive_ids]
            if len(alive_members) == 2:
                groups.append(tuple(self._sort_by_initial_x(alive_members)))
            else:
                leftovers.extend(alive_members)

        leftovers = self._sort_by_initial_x(leftovers)
        while len(leftovers) >= 2:
            groups.append((leftovers.pop(0), leftovers.pop(0)))
        if leftovers:
            groups.append((leftovers[0],))

        return groups

    def _build_context(self, env, members: Tuple[str, ...]) -> EnemyFormationContext:
        members = tuple(self._sort_by_initial_x(members))
        current_positions = [self._get_agent_xy_km(env.agents[member]) for member in members]
        initial_positions = [self._initial_slots_km.get(member, current_positions[idx]) for idx, member in enumerate(members)]
        mean_x = float(np.mean([pos[0] for pos in current_positions]))
        mean_y = float(np.mean([pos[1] for pos in current_positions]))
        sector_id, target_friendly_ids = self._resolve_target_assignment(env, members, mean_x)
        sector_min_by_id, sector_max_by_id, _ = self._get_sector_layout()
        lane_min_km = sector_min_by_id[sector_id]
        lane_max_km = sector_max_by_id[sector_id]

        anchor_x = float(np.clip(
            np.mean([pos[0] for pos in initial_positions]),
            lane_min_km + 2.0,
            lane_max_km - 2.0,
        ))
        attack_anchor_x = anchor_x
        initial_center_y = float(np.clip(
            np.mean([pos[1] for pos in initial_positions]),
            self.bounds.y_min + 60.0,
            self.bounds.y_max - 2.0 * self._boundary_margin_km,
        ))

        if len(members) >= 2:
            desired_spacing = 9.26  # ~5nm
        else:
            desired_spacing = self._desired_pair_spacing_km
        slot_x_by_agent = self._build_slot_x_map(
            members,
            attack_anchor_x,
            desired_spacing,
            lane_min_km,
            lane_max_km,
        )

        group_id = self._make_group_id(members)

        return EnemyFormationContext(
            group_id=group_id,
            members=members,
            anchor_x_km=anchor_x,
            attack_anchor_x_km=attack_anchor_x,
            initial_center_y_km=initial_center_y,
            desired_spacing_km=desired_spacing,
            slot_x_by_agent=slot_x_by_agent,
            mean_x_km=mean_x,
            mean_y_km=mean_y,
            sector_id=sector_id,
            sector_x_min_km=lane_min_km,
            sector_x_max_km=lane_max_km,
            target_friendly_ids=target_friendly_ids,
            is_dynamic_group=self._is_dynamic_group_id(group_id),
        )

    def _get_sector_layout(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        left_center = self._get_initial_center_x_km(ENEMY_DEFAULT_PAIRS["PAIR_LEFT"])
        right_center = self._get_initial_center_x_km(ENEMY_DEFAULT_PAIRS["PAIR_RIGHT"])
        width = self.bounds.x_max - self.bounds.x_min

        if left_center is None:
            left_center = self.bounds.x_min + 0.25 * width
        if right_center is None:
            right_center = self.bounds.x_min + 0.75 * width
        if left_center > right_center:
            left_center, right_center = right_center, left_center

        split_x = float(np.clip(
            (left_center + right_center) / 2.0,
            self.bounds.x_min + 4.0 * self._boundary_margin_km,
            self.bounds.x_max - 4.0 * self._boundary_margin_km,
        ))
        lane_gap = max(2.5, 0.35 * self._desired_pair_spacing_km)
        left_min = self.bounds.x_min + self._boundary_margin_km
        left_max = max(left_min + 6.0, split_x - lane_gap)
        right_min = min(self.bounds.x_max - self._boundary_margin_km - 6.0, split_x + lane_gap)
        right_max = self.bounds.x_max - self._boundary_margin_km
        if left_max >= right_min:
            left_max = split_x - 2.0
            right_min = split_x + 2.0

        sector_min_by_id = {
            "LEFT": float(left_min),
            "RIGHT": float(right_min),
        }
        sector_max_by_id = {
            "LEFT": float(left_max),
            "RIGHT": float(right_max),
        }
        sector_center_by_id = {
            "LEFT": float(np.clip(left_center, sector_min_by_id["LEFT"] + 2.0, sector_max_by_id["LEFT"] - 2.0)),
            "RIGHT": float(np.clip(right_center, sector_min_by_id["RIGHT"] + 2.0, sector_max_by_id["RIGHT"] - 2.0)),
        }
        return sector_min_by_id, sector_max_by_id, sector_center_by_id

    def _get_initial_center_x_km(self, members: Sequence[str]) -> Optional[float]:
        xs = [
            self._initial_slots_km[member][0]
            for member in members
            if member in self._initial_slots_km
        ]
        if not xs:
            return None
        return float(np.mean(xs))

    def _infer_origin_sector(self, members: Tuple[str, ...], mean_x_km: float) -> str:
        votes = {"LEFT": 0, "RIGHT": 0}
        for member in members:
            sector_id = ENEMY_DEFAULT_SECTOR_BY_AGENT.get(member)
            if sector_id in votes:
                votes[sector_id] += 1

        if votes["LEFT"] > votes["RIGHT"]:
            return "LEFT"
        if votes["RIGHT"] > votes["LEFT"]:
            return "RIGHT"

        _, _, sector_centers = self._get_sector_layout()
        return min(sector_centers.keys(), key=lambda sector_id: abs(mean_x_km - sector_centers[sector_id]))

    def _resolve_target_assignment(
        self,
        env,
        members: Tuple[str, ...],
        mean_x_km: float,
    ) -> Tuple[str, Tuple[str, ...]]:
        origin_sector = self._infer_origin_sector(members, mean_x_km)
        close_target_id = None
        close_target_distance = float("inf")
        for member in members:
            member_aircraft = env.agents.get(member)
            if member_aircraft is None or not getattr(member_aircraft, "is_alive", False):
                continue
            member_pos = np.array(member_aircraft.get_position(), dtype=float)
            for target_id in ("A0100", "A0200", "A0300", "A0400"):
                target = env.agents.get(target_id)
                if target is None or not getattr(target, "is_alive", False):
                    continue
                distance = float(np.linalg.norm(member_pos - np.array(target.get_position(), dtype=float)))
                if distance < close_target_distance:
                    close_target_distance = distance
                    close_target_id = target_id

        if close_target_id is not None and close_target_distance <= 60000.0:
            for sector_id, friendly_ids in FRIENDLY_DEFAULT_PAIRS.items():
                if close_target_id in friendly_ids:
                    alive_pair_targets = tuple(
                        aid for aid in friendly_ids
                        if aid in env.agents and getattr(env.agents[aid], "is_alive", False)
                    )
                    if alive_pair_targets:
                        return sector_id, alive_pair_targets
                    return sector_id, (close_target_id,)

        alive_targets_by_sector = {
            sector_id: tuple(
                agent_id
                for agent_id in friendly_ids
                if agent_id in env.agents and getattr(env.agents[agent_id], "is_alive", False)
            )
            for sector_id, friendly_ids in FRIENDLY_DEFAULT_PAIRS.items()
        }

        member_sectors = {ENEMY_DEFAULT_SECTOR_BY_AGENT.get(member) for member in members}
        if member_sectors == {origin_sector} and alive_targets_by_sector.get(origin_sector):
            return origin_sector, alive_targets_by_sector[origin_sector]

        _, _, sector_centers = self._get_sector_layout()
        best_sector: Optional[str] = None
        best_score = -float("inf")
        for sector_id, alive_targets in alive_targets_by_sector.items():
            if not alive_targets:
                continue
            friendly_center_x = self._get_friendly_center_x_km(env, alive_targets, sector_id)
            score = (
                120.0 * len(alive_targets)
                + (24.0 if sector_id == origin_sector else 0.0)
                - 4.0 * abs(mean_x_km - friendly_center_x)
                - 2.0 * abs(mean_x_km - sector_centers[sector_id])
            )
            if score > best_score:
                best_sector = sector_id
                best_score = score

        if best_sector is not None:
            return best_sector, alive_targets_by_sector[best_sector]
        return origin_sector, tuple()

    def _get_friendly_center_x_km(
        self,
        env,
        friendly_ids: Sequence[str],
        fallback_sector: str,
    ) -> float:
        xs = [
            self._get_agent_xy_km(env.agents[agent_id])[0]
            for agent_id in friendly_ids
            if agent_id in env.agents and getattr(env.agents[agent_id], "is_alive", False)
        ]
        if xs:
            return float(np.mean(xs))
        _, _, sector_centers = self._get_sector_layout()
        return float(sector_centers[fallback_sector])

    def _build_slot_x_map(
        self,
        members: Tuple[str, ...],
        anchor_x_km: float,
        desired_spacing_km: float,
        lane_min_km: float,
        lane_max_km: float,
    ) -> Dict[str, float]:
        if len(members) <= 1:
            return {members[0]: float(np.clip(anchor_x_km, lane_min_km + 1.0, lane_max_km - 1.0))}

        usable_width = max(6.0, lane_max_km - lane_min_km - 2.0)
        spacing = float(np.clip(desired_spacing_km, 4.0, usable_width))
        center_x = float(np.clip(anchor_x_km, lane_min_km + spacing / 2.0, lane_max_km - spacing / 2.0))
        return {
            members[0]: center_x - spacing / 2.0,
            members[1]: center_x + spacing / 2.0,
        }

    def _get_context_for_agent(self, agent_id: str) -> Optional[EnemyFormationContext]:
        return self._agent_contexts.get(agent_id)

    def get_group_snapshot(self, agent_id: str) -> Dict[str, object]:
        context = self._agent_contexts.get(str(agent_id))
        if context is None:
            return {}
        state = self._group_states.get(context.group_id)
        return {
            "group_id": context.group_id,
            "members": tuple(context.members),
            "is_dynamic_group": bool(context.is_dynamic_group),
            "phase": str(state.phase) if state is not None else "UNKNOWN",
            "wave_index": int(state.wave_index) if state is not None else 0,
            "pressure_tag": str(state.pressure_tag) if state is not None else "",
            "turn_entry_distance_km": float(state.turn_entry_distance_km) if state is not None else float("nan"),
            "resume_distance_km": float(state.resume_distance_km) if state is not None else float("nan"),
            "regroup_y_km": float(state.regroup_y_km) if state is not None else float("nan"),
        }

    def _get_dynamic_target_pool(self, agent_id: str) -> Tuple[str, ...]:
        context = self._agent_contexts.get(agent_id)
        if context is None:
            return tuple()
        return tuple(context.target_friendly_ids)

    def _get_dynamic_preferred_target_id(self, agent_id: str) -> Optional[str]:
        context = self._agent_contexts.get(agent_id)
        if context is None or not context.target_friendly_ids:
            return None

        members = list(context.members)
        targets = list(context.target_friendly_ids)
        if len(targets) == 1:
            return targets[0]

        try:
            member_index = members.index(agent_id)
        except ValueError:
            return targets[0]

        member_index = min(member_index, len(targets) - 1)
        return targets[member_index]

    def _make_group_id(self, members: Tuple[str, ...]) -> str:
        if len(members) == 2:
            member_set = set(members)
            if member_set == set(ENEMY_DEFAULT_PAIRS["PAIR_LEFT"]):
                return "PAIR_LEFT"
            if member_set == set(ENEMY_DEFAULT_PAIRS["PAIR_RIGHT"]):
                return "PAIR_RIGHT"
            return f"PAIR_DYNAMIC_{members[0]}_{members[1]}"

        member = members[0]
        default_pair = ENEMY_DEFAULT_PAIR_BY_AGENT.get(member)
        if default_pair:
            return default_pair
        return f"SOLO_{member}"

    def _is_dynamic_group_id(self, group_id: str) -> bool:
        return str(group_id).startswith("PAIR_DYNAMIC_")

    def _initial_phase_for_group(self, env, context: EnemyFormationContext) -> str:
        min_distance = self._get_group_min_distance_km(env, context.members, context.target_friendly_ids)
        if min_distance < self._resume_distance_km:
            return "TURN_NORTH"
        return "APPROACH"

    def _update_group_state(
        self,
        env,
        context: EnemyFormationContext,
        state: EnemyWaveState,
        current_time: float,
    ):
        if abs(state.last_update_time - current_time) <= 1e-6:
            return
        state.last_update_time = current_time

        phase = state.phase
        elapsed = current_time - state.phase_enter_time
        min_distance = self._get_group_min_distance_km(env, context.members, context.target_friendly_ids)
        self._refresh_wave_geometry(env, context, state, min_distance)
        member_positions = [self._get_agent_xy_km(env.agents[member]) for member in context.members]
        forward_limit_y_km = float(state.forward_limit_y_km or self._forward_limit_y_km)
        hard_standoff_distance_km = float(state.hard_standoff_distance_km or self._hard_standoff_distance_km)
        forward_limit_reached = any(y_km <= forward_limit_y_km for _, y_km in member_positions)
        boundary_risk = any(self._is_outside_bounds(x_km, y_km) for x_km, y_km in member_positions)
        boundary_critical = any(
            (
                x_km < self.bounds.x_min - 8.0
                or x_km > self.bounds.x_max + 8.0
                or y_km < self.bounds.y_min - 8.0
                or y_km > self.bounds.y_max + 8.0
            )
            for x_km, y_km in member_positions
        )
        approach_hold_active = current_time < float(state.approach_hold_until)
        recovery_hold_active = current_time < float(state.recovery_hold_until)
        group_kinematics = self._collect_group_kinematics(env, context.members)
        energy_recovery_needed = self._needs_energy_recovery(group_kinematics)
        energy_critical = self._is_energy_critical(group_kinematics)
        missile_superiority = bool(state.missile_superiority)
        self._maybe_log_group_status(context, state, current_time, min_distance)

        if phase == "APPROACH":
            allow_boundary_phase_shift = min_distance > (float(state.turn_entry_distance_km or self._turnback_distance_km) + 12.0)
            if forward_limit_reached and elapsed >= 1.0 and not approach_hold_active:
                state.turn_north_entry_y_km = float(context.mean_y_km)
                self._set_group_phase(context.group_id, "TURN_NORTH", current_time, "forward_limit")
                return
            if boundary_critical and allow_boundary_phase_shift and elapsed >= 1.0:
                state.turn_north_entry_y_km = float(context.mean_y_km)
                self._set_group_phase(context.group_id, "TURN_NORTH", current_time, "boundary")
                return
            if min_distance <= hard_standoff_distance_km:
                state.turn_north_entry_y_km = float(context.mean_y_km)
                self._set_group_phase(context.group_id, "TURN_NORTH", current_time, f"hard_line_{min_distance:.1f}km")
                return
            if boundary_risk and allow_boundary_phase_shift and elapsed >= 2.0:
                if not approach_hold_active:
                    state.turn_north_entry_y_km = float(context.mean_y_km)
                    self._set_group_phase(context.group_id, "TURN_NORTH", current_time, "boundary")
                    return
            turnback_distance_km = float(state.turn_entry_distance_km or self._turnback_distance_km)
            if min_distance <= turnback_distance_km:
                state.turn_north_entry_y_km = float(context.mean_y_km)
                self._set_group_phase(context.group_id, "TURN_NORTH", current_time, f"close_{min_distance:.1f}km")
                return

        if phase == "TURN_NORTH":
            heading_ok = self._group_heading_ready(env, context.members, 0.0, 12.0)
            if (elapsed >= 2.5 and heading_ok) or elapsed >= (self._turn_phase_s + 1.5):
                self._set_group_phase(context.group_id, "REGROUP_NORTH", current_time, "northbound")
                return

        if phase == "REGROUP_NORTH":
            if state.regroup_y_km <= 0.0 or state.resume_distance_km <= 0.0:
                self._refresh_wave_geometry(env, context, state, min_distance)
            dynamic_group = bool(context.is_dynamic_group)
            north_depart_margin_km = max(6.0, float(state.north_depart_margin_km or self._regroup_depart_margin_km))
            turn_north_entry_y_km = float(state.turn_north_entry_y_km or context.mean_y_km)
            north_departed = context.mean_y_km >= (turn_north_entry_y_km + north_depart_margin_km)
            north_reached = context.mean_y_km >= state.regroup_y_km - 2.5
            formation_metrics = self._get_formation_metrics(env, context)
            formation_ready = self._formation_ready(env, context)
            regroup_timeout_s = float(state.regroup_timeout_s or self._regroup_timeout_s)
            regroup_min_hold_s = float(state.regroup_min_hold_s or self._regroup_min_hold_s)
            regroup_timeout_ready = elapsed >= regroup_timeout_s
            regroup_min_hold_ready = elapsed >= regroup_min_hold_s
            resume_gate_km = float(state.resume_distance_km or self._resume_distance_km)
            legacy_release_gate_km = float(self._regroup_release_distance_km)
            local_release_gate_km = max(
                hard_standoff_distance_km + (10.0 if missile_superiority else 8.0),
                float(state.turn_entry_distance_km or self._turnback_distance_km) + (2.0 if dynamic_group else 1.0),
            )
            # REGROUP_NORTH 结束应优先服从当前波次的 resume_distance 和局部安全间隔，
            # 而不是被全局 legacy release 门槛长期卡在北撤直飞。
            release_gate_km = max(resume_gate_km, local_release_gate_km)
            dynamic_depart_ready = bool(
                dynamic_group
                and (
                    north_reached
                    or north_departed
                    or elapsed >= max(5.0, regroup_min_hold_s + 0.5)
                )
            )
            release_ready = min_distance >= release_gate_km
            recovery_ready = self._is_recovery_ready(group_kinematics)
            dynamic_hold_cleared = bool(
                (not recovery_hold_active)
                or (not energy_recovery_needed)
                or (not energy_critical)
            )
            dynamic_regroup_ready = bool(
                dynamic_group
                and regroup_min_hold_ready
                and dynamic_depart_ready
                and release_ready
                and dynamic_hold_cleared
                and (
                    recovery_ready
                    or elapsed >= max(6.0, regroup_min_hold_s + 1.5)
                )
            )
            dynamic_timeout_ready = bool(
                dynamic_group
                and regroup_timeout_ready
                and dynamic_depart_ready
                and release_ready
                and (
                    recovery_ready
                    or (not energy_critical and elapsed >= max(regroup_timeout_s + 2.0, regroup_min_hold_s + 2.0))
                )
            )
            blocked_reasons = []
            if not north_departed:
                blocked_reasons.append("north_depart")
            if not north_reached:
                blocked_reasons.append("north_reached")
            if not release_ready:
                blocked_reasons.append("release")
            if not formation_ready:
                blocked_reasons.append("formation")
            if not recovery_ready:
                blocked_reasons.append("recovery")
            if recovery_hold_active:
                blocked_reasons.append("hold")
            if energy_recovery_needed:
                blocked_reasons.append("energy")
            self._maybe_log_regroup_diagnostics(
                context,
                state,
                current_time,
                min_distance,
                formation_metrics,
                {
                    "north_departed": north_departed,
                    "north_reached": north_reached,
                    "dynamic_depart_ready": dynamic_depart_ready,
                    "release_ready": release_ready,
                    "release_gate_km": release_gate_km,
                    "resume_gate_km": resume_gate_km,
                    "local_release_gate_km": local_release_gate_km,
                    "legacy_release_gate_km": legacy_release_gate_km,
                    "formation_ready": formation_ready,
                    "recovery_ready": recovery_ready,
                    "recovery_hold_active": recovery_hold_active,
                    "energy_recovery_needed": energy_recovery_needed,
                    "energy_critical": energy_critical,
                    "dynamic_regroup_ready": dynamic_regroup_ready,
                    "dynamic_timeout_ready": dynamic_timeout_ready,
                    "elapsed_s": elapsed,
                    "blocked_by": "|".join(blocked_reasons[:4]),
                },
            )
            missile_press_ready = bool(
                missile_superiority
                and regroup_min_hold_ready
                and north_departed
                and not recovery_hold_active
                and not energy_recovery_needed
                and min_distance >= max(
                    float(state.turn_entry_distance_km or self._turnback_distance_km) + 4.0,
                    hard_standoff_distance_km + 6.0,
                )
                and (formation_ready or len(context.members) <= 1 or elapsed >= max(5.0, regroup_timeout_s - 1.0))
            )
            finish_exploit_ready = bool(
                state.finish_exploit
                and regroup_min_hold_ready
                and north_departed
                and not recovery_hold_active
                and not energy_recovery_needed
                and min_distance >= max(
                    float(state.resume_distance_km or self._regroup_release_distance_km),
                    hard_standoff_distance_km + 12.0,
                )
                and (formation_ready or len(context.members) <= 1 or elapsed >= max(4.0, regroup_timeout_s - 1.5))
            )
            if (
                len(context.members) == 1
                and elapsed >= max(4.0, regroup_min_hold_s)
                and regroup_min_hold_ready
                and north_departed
                and release_ready
                and recovery_ready
                and not recovery_hold_active
            ):
                state.turn_south_entry_y_km = float(context.mean_y_km)
                self._set_group_phase(context.group_id, "TURN_SOUTH", current_time, "solo_pressure")
                return
            if dynamic_regroup_ready:
                state.turn_south_entry_y_km = float(context.mean_y_km)
                self._set_group_phase(context.group_id, "TURN_SOUTH", current_time, "dynamic_regroup_ready")
                return
            if (
                regroup_min_hold_ready
                and (north_reached or north_departed)
                and formation_ready
                and release_ready
                and recovery_ready
                and not recovery_hold_active
            ):
                reason = "regroup_complete"
                state.turn_south_entry_y_km = float(context.mean_y_km)
                self._set_group_phase(context.group_id, "TURN_SOUTH", current_time, reason)
                return
            if missile_press_ready:
                state.turn_south_entry_y_km = float(context.mean_y_km)
                self._set_group_phase(context.group_id, "TURN_SOUTH", current_time, "missile_superiority")
                return
            if finish_exploit_ready:
                state.turn_south_entry_y_km = float(context.mean_y_km)
                self._set_group_phase(context.group_id, "TURN_SOUTH", current_time, "finish_exploit")
                return
            if dynamic_timeout_ready:
                state.turn_south_entry_y_km = float(context.mean_y_km)
                self._set_group_phase(context.group_id, "TURN_SOUTH", current_time, "dynamic_regroup_timeout")
                return
            if regroup_timeout_ready and regroup_min_hold_ready:
                if (
                    release_ready
                    # Timeout is only a safety release after a real northward
                    # extension; touching regroup_y alone can cause a south
                    # turn to be immediately canceled by the hard-line guard.
                    and north_departed
                    and (recovery_ready or elapsed >= (regroup_timeout_s + 10.0))
                    and not recovery_hold_active
                ):
                    state.turn_south_entry_y_km = float(context.mean_y_km)
                    self._set_group_phase(context.group_id, "TURN_SOUTH", current_time, "regroup_timeout")
                    return
                relaxed_regroup_target = max(
                    float(state.turn_north_entry_y_km or context.mean_y_km) + max(4.0, north_depart_margin_km - 2.0),
                    context.mean_y_km + (4.0 if state.pressure_level >= 1 else 6.0),
                )
                state.regroup_y_km = float(np.clip(
                    min(state.regroup_y_km, relaxed_regroup_target),
                    self.bounds.y_min + 60.0,
                    self.bounds.y_max - 2.0 * self._boundary_margin_km,
                ))
                if energy_recovery_needed and (not dynamic_group or energy_critical):
                    state.recovery_hold_until = max(state.recovery_hold_until, current_time + self._recovery_hold_s)
                state.phase_enter_time = current_time
                state.last_update_time = current_time
                return

        if phase == "TURN_SOUTH":
            effective_hard_standoff_km = hard_standoff_distance_km
            if missile_superiority:
                effective_hard_standoff_km = max(
                    32.0,
                    hard_standoff_distance_km - (4.0 if state.friendly_missiles_total <= 0 else 2.0),
                )
            if state.finish_exploit:
                if state.enemy_missiles_total > 0:
                    effective_hard_standoff_km = max(40.0, effective_hard_standoff_km + 2.0)
                else:
                    effective_hard_standoff_km = max(34.0, effective_hard_standoff_km)
            south_depart_margin_km = max(3.0, float(self._turn_south_depart_margin_km))
            if missile_superiority:
                south_depart_margin_km = max(south_depart_margin_km, 4.0)
            if state.finish_exploit:
                south_depart_margin_km = max(south_depart_margin_km, 5.0)
            turn_south_entry_y_km = float(state.turn_south_entry_y_km or context.mean_y_km)
            south_departed = context.mean_y_km <= max(
                self.bounds.y_min + 20.0,
                turn_south_entry_y_km - south_depart_margin_km,
            )
            hard_line_margin_km = 2.5
            if missile_superiority:
                hard_line_margin_km = max(hard_line_margin_km, 3.0)
            if state.finish_exploit:
                hard_line_margin_km = max(hard_line_margin_km, 4.0)
            if len(context.members) <= 1 or state.friendly_alive_total <= 1:
                hard_line_margin_km = max(hard_line_margin_km, 4.0)
            hard_line_trigger_km = max(28.0, effective_hard_standoff_km - hard_line_margin_km)
            hard_line_guard_elapsed_s = 2.5 if missile_superiority else 3.5
            hard_line_rearm_s = max(
                hard_line_guard_elapsed_s + 4.0,
                float(state.turn_south_min_hold_s or self._turn_south_min_hold_s) + 1.0,
            )
            if (
                min_distance <= hard_line_trigger_km
                and elapsed >= hard_line_guard_elapsed_s
                and (
                    south_departed
                    or elapsed >= hard_line_rearm_s
                    or boundary_critical
                )
            ):
                state.turn_north_entry_y_km = float(context.mean_y_km)
                self._set_group_phase(context.group_id, "TURN_NORTH", current_time, f"hard_line_{min_distance:.1f}km")
                return
            resume_gate_km = float(state.resume_distance_km or self._resume_distance_km)
            if state.finish_exploit:
                south_reengage_gate_km = max(
                    float(state.turn_entry_distance_km or self._turnback_distance_km) + 6.0,
                    max(hard_standoff_distance_km + 4.0, resume_gate_km - 12.0),
                )
            elif missile_superiority:
                south_reengage_gate_km = max(
                    float(state.turn_entry_distance_km or self._turnback_distance_km) + 8.0,
                    max(hard_standoff_distance_km + 6.0, resume_gate_km - 8.0),
                )
            else:
                south_reengage_gate_km = max(
                    float(state.turn_entry_distance_km or self._turnback_distance_km) + 10.0,
                    max(hard_standoff_distance_km + 8.0, resume_gate_km - 6.0),
                )
            release_ready = min_distance >= south_reengage_gate_km
            heading_ready = self._group_heading_ready(env, context.members, 180.0, 18.0)
            southbound_hold_ready = elapsed >= float(state.turn_south_min_hold_s or self._turn_south_min_hold_s)
            if south_departed and release_ready and heading_ready and southbound_hold_ready:
                self._set_group_phase(
                    context.group_id,
                    "APPROACH",
                    current_time,
                    "reattack_finish" if missile_superiority else "reattack",
                )

    def _set_group_phase(self, group_id: str, new_phase: str, current_time: float, reason: str):
        state = self._group_states.setdefault(group_id, EnemyWaveState(phase_enter_time=current_time))
        if state.phase == new_phase:
            return

        prev_phase = state.phase
        if new_phase == "APPROACH" and prev_phase != "APPROACH":
            state.wave_index += 1
            state.approach_hold_until = current_time + self._reattack_hold_s
            state.recovery_hold_until = 0.0
        elif new_phase != "APPROACH":
            state.approach_hold_until = 0.0
        if new_phase != "TURN_SOUTH":
            state.turn_south_entry_y_km = 0.0
        if new_phase not in ("TURN_NORTH", "REGROUP_NORTH"):
            state.turn_north_entry_y_km = 0.0

        state.phase = new_phase
        state.phase_enter_time = current_time
        state.trigger_reason = reason
        state.last_update_time = current_time

        if current_time - self._last_pair_log_time.get(group_id, -999.0) >= 3.0:
            log.info(
                f"[敌方波次切换-{group_id}] {prev_phase} → {new_phase} | 原因:{reason} | 波次:{state.wave_index} "
                f"| turn={state.turn_entry_distance_km:.1f}km release={state.resume_distance_km:.1f}km regroup_y={state.regroup_y_km:.1f}km"
                f" forward={state.forward_limit_y_km:.1f}km hard={state.hard_standoff_distance_km:.1f}km pressure={state.pressure_tag}"
            )
            self._last_pair_log_time[group_id] = current_time

    def _maybe_log_group_status(
        self,
        context: EnemyFormationContext,
        state: EnemyWaveState,
        current_time: float,
        min_distance: float,
    ) -> None:
        if current_time - float(state.last_status_log_time) < 15.0:
            return
        state.last_status_log_time = current_time
        log.info(
            "[敌方波次状态-%s] phase=%s wave=%d members=%s min_dist=%.1fkm mean=(x=%.1fkm,y=%.1fkm) turn=%.1fkm release=%.1fkm regroup_y=%.1fkm front=%.1fkm hard=%.1fkm pressure=%s local_adv=%+d total=%dv%d ms=%d/%d local_ms=%d/%d hold_until=%.1fs",
            context.group_id,
            state.phase,
            int(state.wave_index),
            "/".join(context.members),
            float(min_distance),
            float(context.mean_x_km),
            float(context.mean_y_km),
            float(state.turn_entry_distance_km),
            float(state.resume_distance_km),
            float(state.regroup_y_km),
            float(state.forward_limit_y_km),
            float(state.hard_standoff_distance_km),
            str(state.pressure_tag),
            int(state.local_advantage),
            int(state.enemy_alive_total),
            int(state.friendly_alive_total),
            int(state.enemy_missiles_total),
            int(state.friendly_missiles_total),
            int(state.local_enemy_missiles),
            int(state.local_friendly_missiles),
            float(max(state.approach_hold_until, state.recovery_hold_until)),
        )

    def _maybe_log_regroup_diagnostics(
        self,
        context: EnemyFormationContext,
        state: EnemyWaveState,
        current_time: float,
        min_distance: float,
        formation_metrics: Dict[str, float],
        gates: Dict[str, object],
    ) -> None:
        if state.phase != "REGROUP_NORTH":
            return
        if current_time - float(state.last_regroup_diag_log_time) < 8.0:
            return
        state.last_regroup_diag_log_time = current_time
        log.info(
            "🧭 [敌方REGROUP诊断-%s] dynamic=%d min_dist=%.1fkm mean_y=%.1fkm "
            "departed=%d reached=%d dyn_depart=%d release=%d gate=%.1fkm resume=%.1fkm local=%.1fkm legacy=%.1fkm formation=%d recovery=%d hold=%d energy=%d critical=%d "
            "dyn_ready=%d dyn_timeout=%d elapsed=%.1fs x_err=%.1fkm spacing_err=%.1fkm y_spread=%.1fkm reason=%s",
            context.group_id,
            int(context.is_dynamic_group),
            float(min_distance),
            float(context.mean_y_km),
            int(bool(gates.get("north_departed", False))),
            int(bool(gates.get("north_reached", False))),
            int(bool(gates.get("dynamic_depart_ready", False))),
            int(bool(gates.get("release_ready", False))),
            float(gates.get("release_gate_km", float("nan"))),
            float(gates.get("resume_gate_km", float("nan"))),
            float(gates.get("local_release_gate_km", float("nan"))),
            float(gates.get("legacy_release_gate_km", float("nan"))),
            int(bool(gates.get("formation_ready", False))),
            int(bool(gates.get("recovery_ready", False))),
            int(bool(gates.get("recovery_hold_active", False))),
            int(bool(gates.get("energy_recovery_needed", False))),
            int(bool(gates.get("energy_critical", False))),
            int(bool(gates.get("dynamic_regroup_ready", False))),
            int(bool(gates.get("dynamic_timeout_ready", False))),
            float(gates.get("elapsed_s", 0.0)),
            float(formation_metrics.get("max_x_error_km", float("nan"))),
            float(formation_metrics.get("spacing_error_km", float("nan"))),
            float(formation_metrics.get("y_spread_km", float("nan"))),
            str(gates.get("blocked_by", "")),
        )

    def _get_friendly_mean_y_km(self, env, friendly_ids: Sequence[str]) -> float:
        ys = [
            self._get_agent_xy_km(env.agents[agent_id])[1]
            for agent_id in friendly_ids
            if agent_id in env.agents and getattr(env.agents[agent_id], "is_alive", False)
        ]
        if ys:
            return float(np.mean(ys))
        return self.bounds.y_min + 40.0

    def _count_alive_agents(self, env, agent_ids: Sequence[str]) -> int:
        return sum(
            1
            for agent_id in agent_ids
            if agent_id in env.agents and getattr(env.agents[agent_id], "is_alive", False)
        )

    def _get_aircraft_missiles_left(self, aircraft) -> int:
        try:
            if hasattr(aircraft, "num_left_missiles"):
                return max(0, int(getattr(aircraft, "num_left_missiles")))
        except Exception:
            pass
        return max(0, int(getattr(aircraft, "num_missiles", 0)))

    def _count_remaining_missiles(self, env, agent_ids: Sequence[str]) -> int:
        total = 0
        for agent_id in agent_ids:
            aircraft = env.agents.get(agent_id)
            if aircraft is None or not getattr(aircraft, "is_alive", False):
                continue
            total += self._get_aircraft_missiles_left(aircraft)
        return total

    def _build_pressure_profile(self, env, context: EnemyFormationContext) -> Dict[str, float | int | str]:
        friendly_ids = FRIENDLY_DEFAULT_PAIRS["LEFT"] + FRIENDLY_DEFAULT_PAIRS["RIGHT"]
        enemy_alive_total = self._count_alive_agents(env, ENEMY_CALL_ORDER)
        friendly_alive_total = self._count_alive_agents(env, friendly_ids)
        local_enemy_alive = self._count_alive_agents(env, context.members)
        local_friendly_alive = self._count_alive_agents(env, context.target_friendly_ids)
        enemy_missiles_total = self._count_remaining_missiles(env, ENEMY_CALL_ORDER)
        friendly_missiles_total = self._count_remaining_missiles(env, friendly_ids)
        local_enemy_missiles = self._count_remaining_missiles(env, context.members)
        local_friendly_missiles = self._count_remaining_missiles(env, context.target_friendly_ids)
        total_advantage = int(enemy_alive_total - friendly_alive_total)
        local_advantage = int(local_enemy_alive - local_friendly_alive)
        missile_advantage = int(enemy_missiles_total - friendly_missiles_total)
        local_missile_advantage = int(local_enemy_missiles - local_friendly_missiles)
        missile_superiority = bool(
            (enemy_missiles_total > 0 and friendly_missiles_total <= 0 and enemy_alive_total >= friendly_alive_total)
            or (local_enemy_missiles > 0 and local_friendly_missiles <= 0 and local_enemy_alive >= local_friendly_alive)
            or (missile_advantage >= 4 and enemy_alive_total >= friendly_alive_total)
            or (local_missile_advantage >= 3 and local_enemy_alive >= local_friendly_alive)
        )
        finish_exploit = bool(
            missile_superiority
            and enemy_missiles_total > 0
            and friendly_missiles_total <= 0
            and enemy_alive_total >= max(2, friendly_alive_total + 1)
            and friendly_alive_total <= 1
        )

        pressure_level = 0
        pressure_tag = "balanced"
        endgame_force_press = bool(
            friendly_alive_total <= 2
            and enemy_alive_total >= friendly_alive_total
        )
        if (
            total_advantage >= 2
            or endgame_force_press
            or (local_enemy_alive >= 2 and local_friendly_alive <= 1)
            or local_friendly_alive <= 0
        ):
            pressure_level = 2
            pressure_tag = "press_high"
        elif total_advantage >= 1 or local_advantage >= 1:
            pressure_level = 1
            pressure_tag = "press_mid"
        elif total_advantage < 0:
            pressure_tag = "cautious"
        if missile_superiority and pressure_level < 2:
            pressure_level = 2
            pressure_tag = "press_finish"
        elif missile_superiority:
            pressure_tag = "press_finish"
        if finish_exploit:
            pressure_level = 2
            pressure_tag = "press_finish_exploit"

        if pressure_level >= 2:
            hard_standoff_distance_km = max(30.0, self._hard_standoff_distance_km - 4.0)
            turn_entry_distance_km = max(hard_standoff_distance_km + 0.5, self._turnback_distance_km - 8.0)
            resume_distance_km = max(hard_standoff_distance_km + 16.0, self._resume_distance_km - 12.0)
            forward_floor_y_km = 72.0
            regroup_timeout_s = max(7.0, self._regroup_timeout_s - 5.0)
            regroup_min_hold_s = max(4.0, self._regroup_min_hold_s - 4.0)
            turn_south_min_hold_s = max(3.5, self._turn_south_min_hold_s - 2.5)
            north_depart_margin_km = max(6.0, self._regroup_depart_margin_km - 8.0)
            if missile_superiority:
                hard_standoff_distance_km = max(28.0, hard_standoff_distance_km - 3.0)
                turn_entry_distance_km = max(hard_standoff_distance_km + 0.5, turn_entry_distance_km - 2.0)
                resume_distance_km = max(hard_standoff_distance_km + 14.0, resume_distance_km - 10.0)
                forward_floor_y_km = min(forward_floor_y_km, 64.0)
                regroup_timeout_s = max(5.5, regroup_timeout_s - 1.5)
                regroup_min_hold_s = max(3.5, regroup_min_hold_s - 1.0)
                turn_south_min_hold_s = max(3.0, turn_south_min_hold_s - 0.5)
                north_depart_margin_km = max(5.0, north_depart_margin_km - 1.0)
            if finish_exploit:
                hard_standoff_distance_km = max(40.0, hard_standoff_distance_km + 2.0)
                turn_entry_distance_km = max(hard_standoff_distance_km + 4.0, turn_entry_distance_km + 2.0)
                resume_distance_km = max(hard_standoff_distance_km + 16.0, resume_distance_km + 4.0)
                forward_floor_y_km = min(forward_floor_y_km, 58.0)
                regroup_timeout_s = max(4.5, regroup_timeout_s - 0.5)
                regroup_min_hold_s = max(3.0, regroup_min_hold_s - 0.5)
                turn_south_min_hold_s = max(2.8, turn_south_min_hold_s - 0.2)
                north_depart_margin_km = max(4.0, north_depart_margin_km - 1.0)
            profile = {
                "pressure_level": pressure_level,
                "pressure_tag": pressure_tag,
                "hard_standoff_distance_km": hard_standoff_distance_km,
                "turn_entry_distance_km": turn_entry_distance_km,
                "resume_distance_km": resume_distance_km,
                "forward_floor_y_km": forward_floor_y_km,
                "regroup_timeout_s": regroup_timeout_s,
                "regroup_min_hold_s": regroup_min_hold_s,
                "turn_south_min_hold_s": turn_south_min_hold_s,
                "north_depart_margin_km": north_depart_margin_km,
                "enemy_alive_total": enemy_alive_total,
                "friendly_alive_total": friendly_alive_total,
                "local_advantage": local_advantage,
                "enemy_missiles_total": enemy_missiles_total,
                "friendly_missiles_total": friendly_missiles_total,
                "local_enemy_missiles": local_enemy_missiles,
                "local_friendly_missiles": local_friendly_missiles,
                "missile_advantage": missile_advantage,
                "missile_superiority": missile_superiority,
                "finish_exploit": finish_exploit,
            }
        elif pressure_level == 1:
            hard_standoff_distance_km = max(32.0, self._hard_standoff_distance_km - 2.0)
            profile = {
                "pressure_level": pressure_level,
                "pressure_tag": pressure_tag,
                "hard_standoff_distance_km": hard_standoff_distance_km,
                "turn_entry_distance_km": max(hard_standoff_distance_km + 1.5, self._turnback_distance_km - 5.0),
                "resume_distance_km": max(hard_standoff_distance_km + 20.0, self._resume_distance_km - 8.0),
                "forward_floor_y_km": 86.0,
                "regroup_timeout_s": max(9.0, self._regroup_timeout_s - 3.0),
                "regroup_min_hold_s": max(6.0, self._regroup_min_hold_s - 2.0),
                "turn_south_min_hold_s": max(4.5, self._turn_south_min_hold_s - 1.5),
                "north_depart_margin_km": max(8.0, self._regroup_depart_margin_km - 6.0),
                "enemy_alive_total": enemy_alive_total,
                "friendly_alive_total": friendly_alive_total,
                "local_advantage": local_advantage,
                "enemy_missiles_total": enemy_missiles_total,
                "friendly_missiles_total": friendly_missiles_total,
                "local_enemy_missiles": local_enemy_missiles,
                "local_friendly_missiles": local_friendly_missiles,
                "missile_advantage": missile_advantage,
                "missile_superiority": missile_superiority,
                "finish_exploit": finish_exploit,
            }
        else:
            profile = {
                "pressure_level": pressure_level,
                "pressure_tag": pressure_tag,
                "hard_standoff_distance_km": float(self._hard_standoff_distance_km),
                "turn_entry_distance_km": float(self._turnback_distance_km),
                "resume_distance_km": float(self._resume_distance_km),
                "forward_floor_y_km": 100.0,
                "regroup_timeout_s": float(self._regroup_timeout_s),
                "regroup_min_hold_s": float(self._regroup_min_hold_s),
                "turn_south_min_hold_s": float(self._turn_south_min_hold_s),
                "north_depart_margin_km": float(self._regroup_depart_margin_km),
                "enemy_alive_total": enemy_alive_total,
                "friendly_alive_total": friendly_alive_total,
                "local_advantage": local_advantage,
                "enemy_missiles_total": enemy_missiles_total,
                "friendly_missiles_total": friendly_missiles_total,
                "local_enemy_missiles": local_enemy_missiles,
                "local_friendly_missiles": local_friendly_missiles,
                "missile_advantage": missile_advantage,
                "missile_superiority": missile_superiority,
                "finish_exploit": finish_exploit,
            }
        return profile

    def _refresh_wave_geometry(
        self,
        env,
        context: EnemyFormationContext,
        state: EnemyWaveState,
        min_distance: float,
    ) -> None:
        pressure = self._build_pressure_profile(env, context)
        hard_standoff_distance_km = float(pressure["hard_standoff_distance_km"])
        turn_entry_distance = float(pressure["turn_entry_distance_km"])
        resume_distance = float(pressure["resume_distance_km"])
        if len(context.members) <= 1:
            turn_entry_distance = float(max(hard_standoff_distance_km + 0.5, turn_entry_distance - 2.0))
            resume_distance = float(max(hard_standoff_distance_km + 18.0, resume_distance - 6.0))
        friendly_mean_y = self._get_friendly_mean_y_km(env, context.target_friendly_ids)
        forward_limit_y_km = float(np.clip(
            max(
                friendly_mean_y + max(
                    turn_entry_distance + 2.0,
                    hard_standoff_distance_km + (4.0 if int(pressure["pressure_level"]) >= 2 else 6.0),
                ),
                float(pressure["forward_floor_y_km"]),
            ),
            self.bounds.y_min + 18.0,
            self.bounds.y_max - 30.0,
        ))
        regroup_floor = friendly_mean_y + resume_distance + (2.0 if int(pressure["pressure_level"]) >= 2 else 4.0 if int(pressure["pressure_level"]) == 1 else 6.0)
        pressure_level = int(pressure["pressure_level"])
        if len(context.members) <= 1:
            regroup_extension_km = 10.0 if pressure_level >= 2 else 14.0 if pressure_level >= 1 else 18.0
            regroup_cap_from_initial = context.initial_center_y_km - (18.0 if pressure_level >= 2 else 12.0 if pressure_level >= 1 else 8.0)
            regroup_target = max(
                regroup_floor,
                forward_limit_y_km + (8.0 if pressure_level >= 1 else 10.0),
                min(regroup_cap_from_initial, context.mean_y_km + regroup_extension_km),
            )
        else:
            regroup_extension_km = 12.0 if pressure_level >= 2 else 18.0 if pressure_level == 1 else 26.0
            regroup_cap_from_initial = context.initial_center_y_km - (22.0 if pressure_level >= 2 else 16.0 if pressure_level == 1 else 10.0)
            regroup_target = max(
                regroup_floor + (0.0 if pressure_level >= 2 else 2.0 if pressure_level == 1 else 6.0),
                forward_limit_y_km + (8.0 if pressure_level >= 2 else 12.0 if pressure_level == 1 else 16.0),
                min(regroup_cap_from_initial, context.mean_y_km + regroup_extension_km),
            )
        if state.phase == "REGROUP_NORTH" and state.regroup_y_km > 0.0:
            # REGROUP 阶段只允许把回转点放松到更近，不允许随着当前位置持续向北漂移。
            regroup_target = min(regroup_target, float(state.regroup_y_km))

        state.turn_entry_distance_km = turn_entry_distance
        state.resume_distance_km = resume_distance
        state.forward_limit_y_km = forward_limit_y_km
        state.regroup_timeout_s = float(pressure["regroup_timeout_s"])
        state.regroup_min_hold_s = float(pressure["regroup_min_hold_s"])
        state.turn_south_min_hold_s = float(pressure["turn_south_min_hold_s"])
        north_depart_margin_km = float(pressure["north_depart_margin_km"])
        if bool(pressure.get("finish_exploit", False)) or bool(pressure.get("missile_superiority", False)):
            north_depart_margin_km = min(north_depart_margin_km, 10.0 if len(context.members) <= 1 else 12.0)
        state.north_depart_margin_km = north_depart_margin_km
        state.pressure_level = int(pressure["pressure_level"])
        state.pressure_tag = str(pressure["pressure_tag"])
        state.enemy_alive_total = int(pressure["enemy_alive_total"])
        state.friendly_alive_total = int(pressure["friendly_alive_total"])
        state.local_advantage = int(pressure["local_advantage"])
        state.hard_standoff_distance_km = hard_standoff_distance_km
        state.enemy_missiles_total = int(pressure["enemy_missiles_total"])
        state.friendly_missiles_total = int(pressure["friendly_missiles_total"])
        state.local_enemy_missiles = int(pressure["local_enemy_missiles"])
        state.local_friendly_missiles = int(pressure["local_friendly_missiles"])
        state.missile_advantage = int(pressure["missile_advantage"])
        state.missile_superiority = bool(pressure["missile_superiority"])
        state.finish_exploit = bool(pressure.get("finish_exploit", False))
        state.regroup_y_km = float(np.clip(
            regroup_target,
            self.bounds.y_min + 70.0,
            self.bounds.y_max - 2.0 * self._boundary_margin_km,
        ))

    def _preferred_turn_sign(self, context: EnemyFormationContext, phase: str) -> int:
        if phase == "TURN_NORTH":
            return 1 if context.sector_id == "LEFT" else -1
        return -1 if context.sector_id == "LEFT" else 1

    def _slot_x_for_phase(
        self,
        context: EnemyFormationContext,
        agent_id: str,
        phase: str,
    ) -> float:
        if phase in ("TURN_NORTH", "REGROUP_NORTH"):
            anchor_x_km = context.anchor_x_km
        else:
            anchor_x_km = context.attack_anchor_x_km
        slot_map = self._build_slot_x_map(
            context.members,
            anchor_x_km,
            context.desired_spacing_km,
            context.sector_x_min_km,
            context.sector_x_max_km,
        )
        return float(np.clip(
            slot_map.get(agent_id, anchor_x_km),
            context.sector_x_min_km + 1.0,
            context.sector_x_max_km - 1.0,
        ))

    def _fast_turn_hdg_cmd(
        self,
        current_heading: float,
        target_heading: float,
        prefer_sign: int,
    ) -> int:
        diff = self._normalize_heading_error(target_heading - current_heading)
        if abs(diff) < 4.0:
            return 8
        if abs(diff) > 135.0:
            return 16 if prefer_sign > 0 else 0
        if abs(diff) > 95.0:
            return 15 if diff > 0 else 1
        if abs(diff) > 65.0:
            return 14 if diff > 0 else 2
        if abs(diff) > 38.0:
            return 12 if diff > 0 else 4
        if abs(diff) > 18.0:
            return 10 if diff > 0 else 6
        return 9 if diff > 0 else 7

    def _compute_regroup_y_km(self, context: EnemyFormationContext) -> float:
        preferred = max(
            context.initial_center_y_km - 18.0,
            self._forward_limit_y_km + 24.0,
        )
        return float(np.clip(
            preferred,
            self.bounds.y_min + 60.0,
            self.bounds.y_max - 2.0 * self._boundary_margin_km,
        ))

    def _get_combat_command(
        self,
        env,
        agent_id: str,
        current_time: float,
        task=None,
        context: Optional[EnemyFormationContext] = None,
        state: Optional[EnemyWaveState] = None,
    ) -> Tuple[int, int, int]:
        if self._combat_ai is None:
            return self._default_action(env, agent_id)
        try:
            raw_cmd = self._combat_ai.get_enemy_command(env, agent_id, current_time, task)
        except TypeError:
            raw_cmd = self._combat_ai.get_enemy_command(env, agent_id, current_time)
        except Exception as exc:
            log.warning("[敌方战术AI] %s 调用UnifiedEnemyTacticalAI失败: %s", agent_id, exc)
            return self._default_action(env, agent_id)

        if context is None or state is None:
            return self._default_action(env, agent_id)

        aircraft = env.agents.get(agent_id)
        if aircraft is None:
            return self._default_action(env, agent_id)

        advisory_cmd = self._clip_command(tuple(int(x) for x in raw_cmd))
        advisory_alt_cmd, advisory_hdg_cmd, advisory_vel_cmd = advisory_cmd
        current_x_km, current_y_km = self._get_agent_xy_km(aircraft)
        min_distance = self._get_group_min_distance_km(env, context.members, context.target_friendly_ids)
        approach_hold_active = current_time < float(state.approach_hold_until)
        current_altitude = self._get_altitude_m(aircraft)
        current_heading = self._get_heading_deg(aircraft)
        current_speed = self._get_speed_mps(aircraft)
        v_up_mps = self._get_vertical_speed_mps(aircraft)
        pitch_deg = self._get_pitch_deg(aircraft)
        roll_deg = self._get_roll_deg(aircraft)
        needs_boundary_guard = self._needs_boundary_correction(current_x_km, current_y_km)
        forward_limit_y_km = float(state.forward_limit_y_km or self._forward_limit_y_km)
        hard_standoff_distance_km = float(state.hard_standoff_distance_km or self._hard_standoff_distance_km)
        frontline_guard = current_y_km <= (forward_limit_y_km + 4.0)
        hard_standoff_guard = min_distance <= hard_standoff_distance_km
        turn_entry_distance_km = float(state.turn_entry_distance_km or self._turnback_distance_km)
        early_turn_guard = min_distance <= (turn_entry_distance_km + 4.0)
        stability_guard = (
            current_altitude < self._energy_floor_altitude_m
            or current_speed < self._energy_turn_speed_mps
            or v_up_mps < self._energy_descent_trigger_mps
            or pitch_deg < self._energy_pitch_trigger_deg
            or abs(roll_deg) > self._energy_roll_trigger_deg
        )
        attack_slot_x_km = self._slot_x_for_phase(context, agent_id, "APPROACH")
        friendly_center_x = self._get_friendly_center_x_km(env, context.target_friendly_ids, context.sector_id)
        if min_distance <= 60.0:
            attack_slot_x_km = float(np.clip(
                0.25 * attack_slot_x_km + 0.75 * friendly_center_x,
                context.sector_x_min_km + 1.0,
                context.sector_x_max_km - 1.0,
            ))
        else:
            attack_slot_x_km = float(np.clip(
                0.55 * attack_slot_x_km + 0.45 * friendly_center_x,
                context.sector_x_min_km + 1.0,
                context.sector_x_max_km - 1.0,
            ))
        friendly_mean_y = self._get_friendly_mean_y_km(env, context.target_friendly_ids)
        attack_line_y_km = float(np.clip(
            friendly_mean_y + max(state.turn_entry_distance_km + 2.0, hard_standoff_distance_km + 6.0),
            self.bounds.y_min + 10.0,
            self.bounds.y_max - 20.0,
        ))
        if hard_standoff_guard or frontline_guard or early_turn_guard:
            target_heading = self._bearing_deg(
                current_x_km,
                current_y_km,
                attack_slot_x_km,
                max(current_y_km - 14.0, attack_line_y_km),
            )
        else:
            target_heading = self._bearing_deg(current_x_km, current_y_km, attack_slot_x_km, attack_line_y_km)
        if needs_boundary_guard:
            target_heading = self._boundary_correction(current_x_km, current_y_km, target_heading)

        alt_cmd = self._altitude_command(current_altitude, "APPROACH")
        if hard_standoff_guard or frontline_guard:
            vel_cmd = max(advisory_vel_cmd, 6)
        else:
            if early_turn_guard:
                vel_cmd = max(advisory_vel_cmd, 6)
            else:
                vel_cmd = max(advisory_vel_cmd, 6 if (approach_hold_active or stability_guard) else 5)
        if stability_guard:
            if current_altitude < self._energy_floor_altitude_m:
                alt_cmd = max(alt_cmd, 9 if current_altitude >= 2600.0 else 10)
            vel_cmd = max(vel_cmd, 6)

        advisory_target_heading = self._cmd_to_target_heading(current_heading, advisory_hdg_cmd)
        advisory_heading_error = abs(self._normalize_heading_error(target_heading - advisory_target_heading))

        if min_distance <= (turn_entry_distance_km + 3.5):
            hdg_cmd = self._fast_turn_hdg_cmd(
                current_heading,
                target_heading,
                self._preferred_turn_sign(context, "TURN_NORTH"),
            )
        elif needs_boundary_guard or hard_standoff_guard or frontline_guard:
            hdg_cmd = self._hdg_to_cmd(current_heading, target_heading)
        elif stability_guard and advisory_heading_error > 55.0:
            hdg_cmd = self._hdg_to_cmd(current_heading, target_heading)
        elif early_turn_guard and advisory_heading_error > 80.0:
            hdg_cmd = self._hdg_to_cmd(current_heading, target_heading)
        else:
            hdg_cmd = advisory_hdg_cmd
        return self._apply_overlay_safety(env, agent_id, (alt_cmd, hdg_cmd, vel_cmd), task, "APPROACH")

    def _build_group_command(
        self,
        env,
        agent_id: str,
        context: EnemyFormationContext,
        state: EnemyWaveState,
        task=None,
    ) -> Tuple[int, int, int]:
        aircraft = env.agents.get(agent_id)
        if aircraft is None:
            return 7, 8, 3

        current_x_km, current_y_km = self._get_agent_xy_km(aircraft)
        current_heading = self._get_heading_deg(aircraft)
        current_altitude = self._get_altitude_m(aircraft)
        slot_x_km = self._slot_x_for_phase(context, agent_id, state.phase)
        if state.phase == "APPROACH":
            friendly_center_x = self._get_friendly_center_x_km(env, context.target_friendly_ids, context.sector_id)
            slot_x_km = float(np.clip(
                0.45 * slot_x_km + 0.55 * friendly_center_x,
                context.sector_x_min_km + 1.0,
                context.sector_x_max_km - 1.0,
            ))

        if state.phase == "APPROACH":
            target_y_km = max(float(state.forward_limit_y_km or self._forward_limit_y_km) + 2.0, current_y_km - 24.0)
            target_heading = self._bearing_deg(current_x_km, current_y_km, slot_x_km, target_y_km)
            min_speed_cmd = 5
        else:
            if state.phase == "TURN_NORTH":
                min_speed_cmd = 6
                target_y_km = max(state.regroup_y_km, current_y_km + 12.0)
            elif state.phase == "REGROUP_NORTH":
                min_speed_cmd = 6
                target_y_km = max(state.regroup_y_km, current_y_km + 10.0)
            else:
                min_speed_cmd = 6
                target_y_km = max(self.bounds.y_min + 20.0, current_y_km - 12.0)
            target_heading = self._bearing_deg(current_x_km, current_y_km, slot_x_km, target_y_km)

        if self._needs_boundary_correction(current_x_km, current_y_km):
            target_heading = self._bearing_deg(
                current_x_km,
                current_y_km,
                float(np.clip(
                    slot_x_km,
                    context.sector_x_min_km + 1.0,
                    context.sector_x_max_km - 1.0,
                )),
                float(np.clip(
                    target_y_km,
                    self.bounds.y_min + 20.0,
                    self.bounds.y_max - 2.0 * self._boundary_margin_km,
                )),
            )

        target_heading = self._boundary_correction(current_x_km, current_y_km, target_heading)
        alt_cmd = self._altitude_command(current_altitude, state.phase)
        vel_cmd = self._speed_command(current_altitude, min_speed_cmd)

        if state.phase in ("TURN_NORTH", "TURN_SOUTH"):
            hdg_cmd = self._fast_turn_hdg_cmd(
                current_heading,
                target_heading,
                self._preferred_turn_sign(context, state.phase),
            )
        else:
            hdg_cmd = self._hdg_to_cmd(current_heading, target_heading)

        if state.phase != "APPROACH" and current_altitude < self._energy_floor_altitude_m:
            alt_cmd = max(alt_cmd, 10 if current_altitude < (self._energy_floor_altitude_m - 800.0) else 9)
            vel_cmd = 6

        return self._apply_overlay_safety(env, agent_id, (alt_cmd, hdg_cmd, vel_cmd), task, state.phase)

    def _default_action(self, env, agent_id: str) -> Tuple[int, int, int]:
        aircraft = env.agents.get(agent_id)
        if aircraft is None:
            return 7, 8, 3
        current_heading = self._get_heading_deg(aircraft)
        hdg_cmd = self._hdg_to_cmd(current_heading, 180.0)
        return 7, hdg_cmd, 4

    def _get_formation_metrics(self, env, context: EnemyFormationContext) -> Dict[str, float]:
        metrics = {
            "max_x_error_km": 0.0,
            "spacing_error_km": 0.0,
            "y_spread_km": 0.0,
            "current_spacing_km": 0.0,
        }
        if len(context.members) < 2:
            return metrics

        x_errors = []
        y_values = []
        x_values = []
        for member in context.members:
            aircraft = env.agents.get(member)
            if aircraft is None or not getattr(aircraft, "is_alive", False):
                return {
                    "max_x_error_km": float("inf"),
                    "spacing_error_km": float("inf"),
                    "y_spread_km": float("inf"),
                    "current_spacing_km": float("inf"),
                }
            current_x_km, current_y_km = self._get_agent_xy_km(aircraft)
            x_values.append(current_x_km)
            y_values.append(current_y_km)
            x_errors.append(abs(current_x_km - context.slot_x_by_agent.get(member, current_x_km)))

        current_spacing = abs(x_values[1] - x_values[0]) if len(x_values) >= 2 else 0.0
        y_spread = max(y_values) - min(y_values) if y_values else 0.0
        metrics["max_x_error_km"] = float(max(x_errors) if x_errors else 0.0)
        metrics["current_spacing_km"] = float(current_spacing)
        metrics["spacing_error_km"] = float(abs(current_spacing - context.desired_spacing_km))
        metrics["y_spread_km"] = float(y_spread)
        return metrics

    def _formation_ready(self, env, context: EnemyFormationContext) -> bool:
        if len(context.members) < 2:
            return True
        metrics = self._get_formation_metrics(env, context)
        if context.is_dynamic_group:
            return (
                metrics["max_x_error_km"] <= 6.0
                and metrics["spacing_error_km"] <= 5.0
                and metrics["y_spread_km"] <= 5.0
            )
        return (
            metrics["max_x_error_km"] <= 3.5
            and metrics["spacing_error_km"] <= 2.0
            and metrics["y_spread_km"] <= 3.0
        )

    def _group_heading_ready(
        self,
        env,
        members: Tuple[str, ...],
        target_heading_deg: float,
        tolerance_deg: float,
    ) -> bool:
        errors = []
        for member in members:
            aircraft = env.agents.get(member)
            if aircraft is None or not getattr(aircraft, "is_alive", False):
                continue
            current_heading = self._get_heading_deg(aircraft)
            errors.append(abs(self._normalize_heading_error(target_heading_deg - current_heading)))
        return bool(errors) and max(errors) <= tolerance_deg

    def _sort_by_initial_x(self, agent_ids: Sequence[str]) -> List[str]:
        return sorted(
            agent_ids,
            key=lambda agent_id: self._initial_slots_km.get(agent_id, (float("inf"), 0.0))[0],
        )

    def _altitude_command(self, current_altitude_m: float, phase: str) -> int:
        if phase == "APPROACH":
            desired_altitude_m = max(7600.0, self._regroup_altitude_m - 500.0)
            if current_altitude_m < desired_altitude_m - 1800.0:
                return 12
            if current_altitude_m < desired_altitude_m - 900.0:
                return 11
            if current_altitude_m < desired_altitude_m - 300.0:
                return 10
            if current_altitude_m > desired_altitude_m + 1800.0:
                return 4
            if current_altitude_m > desired_altitude_m + 900.0:
                return 5
            if current_altitude_m > desired_altitude_m + 400.0:
                return 6
            return 7

        if phase in ("TURN_NORTH", "TURN_SOUTH"):
            desired_altitude_m = float(np.clip(max(current_altitude_m + 500.0, 7000.0), 7000.0, self._regroup_altitude_m))
            if current_altitude_m < desired_altitude_m - 1000.0:
                return 11
            if current_altitude_m < desired_altitude_m - 400.0:
                return 10
            if current_altitude_m < desired_altitude_m - 100.0:
                return 9
            if current_altitude_m > desired_altitude_m + 1100.0:
                return 6
            return 8

        desired_altitude_m = float(np.clip(max(current_altitude_m + 400.0, 7200.0), 7200.0, self._regroup_altitude_m))
        if current_altitude_m < desired_altitude_m - 1200.0:
            return 10
        if current_altitude_m < desired_altitude_m - 500.0:
            return 9
        if current_altitude_m > desired_altitude_m + 1000.0:
            return 6
        return 8

    def _speed_command(self, current_altitude_m: float, min_speed_cmd: int) -> int:
        if current_altitude_m < 7000.0:
            return max(min_speed_cmd, 6)
        return max(min_speed_cmd, 5)

    def _get_group_min_distance_km(
        self,
        env,
        members: Tuple[str, ...],
        candidate_friendly_ids: Sequence[str] = (),
    ) -> float:
        distances = [
            self._get_nearest_friendly_distance_km(env, member, candidate_friendly_ids)
            for member in members
        ]
        return min(distances) if distances else float("inf")

    def _get_nearest_friendly_distance_km(
        self,
        env,
        agent_id: str,
        candidate_friendly_ids: Sequence[str] = (),
    ) -> float:
        aircraft = env.agents.get(agent_id)
        if aircraft is None:
            return float("inf")

        current_pos = np.array(aircraft.get_position(), dtype=float)
        nearest = float("inf")
        friendly_pool = tuple(candidate_friendly_ids) if candidate_friendly_ids else ("A0100", "A0200", "A0300", "A0400")
        for friendly_id in friendly_pool:
            friendly = env.agents.get(friendly_id)
            if friendly is None or not getattr(friendly, "is_alive", False):
                continue
            dist_m = float(np.linalg.norm(current_pos - np.array(friendly.get_position(), dtype=float)))
            nearest = min(nearest, dist_m / 1000.0)
        return nearest

    def _get_agent_xy_km(self, aircraft) -> Tuple[float, float]:
        pos = aircraft.get_position()
        return float(pos[1]) / 1000.0, float(pos[0]) / 1000.0

    def _get_heading_deg(self, aircraft) -> float:
        try:
            return np.degrees(aircraft.get_property_value(c.attitude_heading_true_rad)) % 360.0
        except Exception:
            return 180.0

    def _get_altitude_m(self, aircraft) -> float:
        try:
            return float(aircraft.get_property_value(c.position_h_sl_m))
        except Exception:
            pos = aircraft.get_position()
            if len(pos) >= 3:
                return max(0.0, -float(pos[2]))
            return 0.0

    def _get_speed_mps(self, aircraft) -> float:
        try:
            return float(aircraft.get_property_value(c.velocities_vc_mps))
        except Exception:
            try:
                velocity = np.array(aircraft.get_velocity(), dtype=float)
                return float(np.linalg.norm(velocity))
            except Exception:
                return 0.0

    def _get_vertical_speed_mps(self, aircraft) -> float:
        try:
            return -float(aircraft.get_property_value(c.velocities_v_down_mps))
        except Exception:
            return 0.0

    def _get_pitch_deg(self, aircraft) -> float:
        try:
            return float(np.degrees(aircraft.get_property_value(c.attitude_theta_rad)))
        except Exception:
            return 0.0

    def _get_roll_deg(self, aircraft) -> float:
        try:
            return float(np.degrees(aircraft.get_property_value(c.attitude_phi_rad)))
        except Exception:
            return 0.0

    def _collect_group_kinematics(self, env, members: Sequence[str]) -> EnemyGroupKinematics:
        snapshot = EnemyGroupKinematics()
        for member in members:
            aircraft = env.agents.get(member)
            if aircraft is None or not getattr(aircraft, "is_alive", False):
                continue
            snapshot.min_altitude_m = min(snapshot.min_altitude_m, self._get_altitude_m(aircraft))
            snapshot.min_speed_mps = min(snapshot.min_speed_mps, self._get_speed_mps(aircraft))
            snapshot.min_vertical_speed_mps = min(snapshot.min_vertical_speed_mps, self._get_vertical_speed_mps(aircraft))
            snapshot.min_pitch_deg = min(snapshot.min_pitch_deg, self._get_pitch_deg(aircraft))
            snapshot.max_abs_roll_deg = max(snapshot.max_abs_roll_deg, abs(self._get_roll_deg(aircraft)))

        if snapshot.min_altitude_m == float("inf"):
            snapshot.min_altitude_m = 0.0
        if snapshot.min_speed_mps == float("inf"):
            snapshot.min_speed_mps = 0.0
        if snapshot.min_vertical_speed_mps == float("inf"):
            snapshot.min_vertical_speed_mps = 0.0
        if snapshot.min_pitch_deg == float("inf"):
            snapshot.min_pitch_deg = 0.0
        return snapshot

    def _needs_energy_recovery(self, snapshot: EnemyGroupKinematics) -> bool:
        attitude_unstable = (
            snapshot.min_vertical_speed_mps < self._energy_descent_trigger_mps
            or snapshot.min_pitch_deg < self._energy_pitch_trigger_deg
            or snapshot.max_abs_roll_deg > self._energy_roll_trigger_deg
        )
        return (
            snapshot.min_altitude_m < self._energy_floor_altitude_m
            or snapshot.min_speed_mps < self._energy_critical_speed_mps
            or (snapshot.min_speed_mps < self._energy_turn_speed_mps and attitude_unstable)
        )

    def _is_energy_critical(self, snapshot: EnemyGroupKinematics) -> bool:
        return (
            snapshot.min_speed_mps < self._energy_critical_speed_mps
            or snapshot.min_altitude_m < (self._energy_floor_altitude_m - 900.0)
            or snapshot.min_vertical_speed_mps < (self._energy_descent_trigger_mps - 10.0)
            or snapshot.min_pitch_deg < (self._energy_pitch_trigger_deg - 6.0)
            or snapshot.max_abs_roll_deg > (self._energy_roll_trigger_deg + 12.0)
        )

    def _is_recovery_ready(self, snapshot: EnemyGroupKinematics) -> bool:
        return (
            snapshot.min_speed_mps >= max(185.0, self._energy_critical_speed_mps)
            and snapshot.min_altitude_m >= max(6200.0, self._energy_floor_altitude_m - 600.0)
            and snapshot.min_vertical_speed_mps >= (self._energy_descent_trigger_mps + 2.0)
            and snapshot.min_pitch_deg >= (self._energy_pitch_trigger_deg - 3.0)
            and snapshot.max_abs_roll_deg <= max(68.0, self._energy_roll_trigger_deg - 5.0)
        )

    def _bearing_deg(
        self,
        current_x_km: float,
        current_y_km: float,
        target_x_km: float,
        target_y_km: float,
    ) -> float:
        dx = target_x_km - current_x_km
        dy = target_y_km - current_y_km
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return float(np.degrees(np.arctan2(dx, dy)) % 360.0)

    def _normalize_heading_error(self, diff_deg: float) -> float:
        if diff_deg > 180.0:
            diff_deg -= 360.0
        elif diff_deg < -180.0:
            diff_deg += 360.0
        return diff_deg

    def _needs_boundary_correction(self, x_km: float, y_km: float) -> bool:
        margin = self._boundary_margin_km
        return (
            x_km < self.bounds.x_min + margin
            or x_km > self.bounds.x_max - margin
            or y_km < self.bounds.y_min + margin
            or y_km > self.bounds.y_max - margin
        )

    def _is_outside_bounds(self, x_km: float, y_km: float) -> bool:
        return not (
            self.bounds.x_min <= x_km <= self.bounds.x_max
            and self.bounds.y_min <= y_km <= self.bounds.y_max
        )

    def _boundary_correction(self, x_km: float, y_km: float, target_hdg: float) -> float:
        margin = self._boundary_margin_km
        if x_km < self.bounds.x_min + margin:
            target_hdg = self._bearing_deg(x_km, y_km, self.bounds.x_min + margin + 8.0, y_km - 8.0)
        elif x_km > self.bounds.x_max - margin:
            target_hdg = self._bearing_deg(x_km, y_km, self.bounds.x_max - margin - 8.0, y_km - 8.0)

        if y_km > self.bounds.y_max - margin:
            target_hdg = self._bearing_deg(x_km, y_km, x_km, self.bounds.y_max - margin - 10.0)
        elif y_km < self.bounds.y_min + margin:
            target_hdg = self._bearing_deg(x_km, y_km, x_km, self.bounds.y_min + margin + 10.0)
        return target_hdg % 360.0

    def _hdg_to_cmd(self, current: float, target: float) -> int:
        diff = self._normalize_heading_error(target - current)
        if abs(diff) < 4.0:
            return 8
        return int(np.argmin(np.abs(self.norm_hdg - np.radians(diff))))

    def _cmd_to_target_heading(self, current_heading: float, hdg_cmd: int) -> float:
        try:
            hdg_cmd = int(np.clip(hdg_cmd, 0, len(self.norm_hdg) - 1))
            diff_deg = float(np.degrees(self.norm_hdg[hdg_cmd]))
            return (current_heading + diff_deg) % 360.0
        except Exception:
            return current_heading

    def _clip_command(self, cmd: Tuple[int, int, int]) -> Tuple[int, int, int]:
        alt_cmd, hdg_cmd, vel_cmd = cmd
        return (
            int(np.clip(alt_cmd, 0, 14)),
            int(np.clip(hdg_cmd, 0, 16)),
            int(np.clip(vel_cmd, 0, 6)),
        )

    def _apply_overlay_safety(
        self,
        env,
        agent_id: str,
        cmd: Tuple[int, int, int],
        task=None,
        phase: Optional[str] = None,
    ) -> Tuple[int, int, int]:
        """
        外层编队/边界修正不能绕过 UnifiedEnemyTacticalAI 的底层安全检查。
        根因修复：之前 TURN_NORTH/REGROUP/边界修正直接把命令下给飞机，
        低能量或持续下沉时没有再经过失速/高度保护，后期就会越救越掉。
        """
        clipped_cmd = self._clip_command(cmd)
        self._rootcause_log(
            env,
            agent_id,
            "外层透传",
            f"phase={phase or 'LEGACY'} cmd=({clipped_cmd[0]},{clipped_cmd[1]},{clipped_cmd[2]})",
            interval_steps=30,
        )
        if phase in ("APPROACH", "TURN_NORTH", "REGROUP_NORTH", "TURN_SOUTH"):
            try:
                aircraft = env.agents.get(agent_id)
                if aircraft is None:
                    return clipped_cmd
                current_alt = self._get_altitude_m(aircraft)
                current_speed = self._get_speed_mps(aircraft)
                v_up_mps = self._get_vertical_speed_mps(aircraft)
                pitch_deg = self._get_pitch_deg(aircraft)
                current_roll = self._get_roll_deg(aircraft)
                alt_cmd, hdg_cmd, vel_cmd = clipped_cmd
                if phase == "APPROACH" and current_alt > 10500.0:
                    alt_cmd = min(alt_cmd, 6)
                elif phase in ("TURN_NORTH", "REGROUP_NORTH", "TURN_SOUTH") and current_alt > (self._regroup_altitude_m + 1200.0):
                    alt_cmd = min(alt_cmd, 6)
                if current_alt < 1800.0:
                    alt_cmd = max(alt_cmd, 11)
                elif current_alt < self._energy_floor_altitude_m:
                    alt_cmd = max(alt_cmd, 9)
                if current_speed < self._energy_turn_speed_mps:
                    vel_cmd = max(vel_cmd, 6)
                low_energy = (
                    current_speed < self._energy_critical_speed_mps
                    or (
                        current_speed < self._energy_turn_speed_mps
                        and (v_up_mps < self._energy_descent_trigger_mps or pitch_deg < self._energy_pitch_trigger_deg)
                    )
                )
                unstable = (
                    abs(current_roll) > (self._energy_roll_trigger_deg + 10.0)
                    or pitch_deg < (self._energy_pitch_trigger_deg - 6.0)
                    or v_up_mps < (self._energy_descent_trigger_mps - 10.0)
                )
                if low_energy or unstable:
                    vel_cmd = 6
                    alt_cmd = max(alt_cmd, 9 if current_alt >= 2600.0 else 10)
                    hdg_cmd = 8
                elif abs(current_roll) > 70.0 and current_speed < (self._energy_turn_speed_mps + 5.0):
                    hdg_cmd = 8
                if current_alt < 900.0 and abs(current_roll) > 110.0 and current_speed < self._energy_critical_speed_mps:
                    hdg_cmd = 8
                return self._clip_command((alt_cmd, hdg_cmd, vel_cmd))
            except Exception:
                return clipped_cmd
        if self._combat_ai is None:
            return clipped_cmd

        safety_check = getattr(self._combat_ai, "_apply_global_safety_check", None)
        if safety_check is None:
            return clipped_cmd

        try:
            return self._clip_command(
                safety_check(
                    env,
                    agent_id,
                    int(clipped_cmd[0]),
                    int(clipped_cmd[1]),
                    int(clipped_cmd[2]),
                    task,
                )
            )
        except TypeError:
            try:
                return self._clip_command(
                    safety_check(
                        env,
                        agent_id,
                        int(clipped_cmd[0]),
                        int(clipped_cmd[1]),
                        int(clipped_cmd[2]),
                    )
                )
            except Exception:
                return clipped_cmd
        except Exception:
            return clipped_cmd






