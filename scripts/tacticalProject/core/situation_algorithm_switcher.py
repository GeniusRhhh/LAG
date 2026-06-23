"""Project-wide intent routing locked to bvr_intent_new."""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from .intent_algorithm_adapters import BVRIntentNewAdapter
from .intent_recognizer import EnemyIntent

try:
    from utils.trace_logger import trace_event, trace_if_changed, trace_throttle
except Exception:  # pragma: no cover
    trace_event = None
    trace_if_changed = None
    trace_throttle = None


LOG = logging.getLogger(__name__)


def _safe_get_id(aircraft: Any) -> Optional[str]:
    for attr in ("agent_id", "callsign", "name", "uid", "id"):
        try:
            value = getattr(aircraft, attr, None)
        except Exception:
            value = None
        if value:
            return str(value)
    return None


def _compute_intent_explain_features(enemy_aircraft: Any, my_aircraft: Any) -> Dict[str, Any]:
    explain: Dict[str, Any] = {}
    try:
        my_pos = np.asarray(my_aircraft.get_position(), dtype=np.float64)
        enemy_pos = np.asarray(enemy_aircraft.get_position(), dtype=np.float64)
        my_vel = np.asarray(my_aircraft.get_velocity(), dtype=np.float64)
        enemy_vel = np.asarray(enemy_aircraft.get_velocity(), dtype=np.float64)

        rel_pos = my_pos - enemy_pos
        distance = float(np.linalg.norm(rel_pos) + 1e-6)
        rel_vel = my_vel - enemy_vel
        closing_rate = float(-np.dot(rel_pos, rel_vel) / distance)

        heading_angle = None
        try:
            yaw = float(enemy_aircraft.get_rpy()[2])
            enemy_heading = np.asarray([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
            to_me = rel_pos / distance
            heading_angle = float(np.degrees(np.arccos(np.clip(np.dot(enemy_heading, to_me), -1.0, 1.0))))
        except Exception:
            heading_angle = None

        explain.update(
            {
                "distance_km": distance / 1000.0,
                "closing_m_s": closing_rate,
                "heading_to_me_deg": heading_angle,
                "alt_diff_m": float(my_pos[2] - enemy_pos[2]),
                "enemy_speed_m_s": float(np.linalg.norm(enemy_vel)),
                "own_speed_m_s": float(np.linalg.norm(my_vel)),
            }
        )
    except Exception:
        pass

    try:
        explain["enemy_radar_mode"] = str(getattr(enemy_aircraft, "radar_mode", None))
    except Exception:
        pass
    return explain


class SituationAlgorithmConfig(Enum):
    BVR_INTENT_NEW = "bvr_intent_new"


class SituationAlgorithmSwitcher:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_intent_explain_log_state"):
            self._intent_explain_log_state = {}
        if not hasattr(self, "_intent_model_state"):
            self._intent_model_state = {}
        if not hasattr(self, "_intent_stdlog_state"):
            self._intent_stdlog_state = {}
        if self._initialized:
            return

        self.current_algorithm = SituationAlgorithmConfig.BVR_INTENT_NEW
        self.bvr_intent_new = BVRIntentNewAdapter()
        self._initialized = True

        env_algo = os.getenv("TACTICAL_INTENT_ALGO")
        if env_algo:
            self.switch_algorithm(env_algo)

        LOG.info("Intent algorithm switcher initialised with bvr_intent_new only.")

    @property
    def algorithm_instance(self):
        return self.bvr_intent_new

    def get_algorithm_status(self) -> Dict[str, Any]:
        new_status = self.bvr_intent_new.status()
        return {
            "current_algorithm": self.current_algorithm.value,
            "note": {
                "bvr_intent_new": "IntentRecognition/bvr_intent_new online adapter (project-wide single backend).",
            },
            "algorithms": {
                "bvr_intent_new": new_status,
            },
        }

    def set_algorithm(self, algorithm: SituationAlgorithmConfig):
        if algorithm != SituationAlgorithmConfig.BVR_INTENT_NEW:
            LOG.warning("Intent algorithm %s requested but project is locked to bvr_intent_new", algorithm.value)
        self.current_algorithm = SituationAlgorithmConfig.BVR_INTENT_NEW
        LOG.info("Intent algorithm fixed to %s", self.current_algorithm.value)

    def switch_algorithm(self, algorithm):
        requested = str(getattr(algorithm, "value", algorithm))
        normalized = requested.strip().lower() if requested is not None else ""
        if normalized and normalized != "bvr_intent_new":
            LOG.warning("Unknown intent algorithm %s; forcing bvr_intent_new", requested)
        self.set_algorithm(SituationAlgorithmConfig.BVR_INTENT_NEW)

    def get_current_algorithm(self) -> str:
        return self.current_algorithm.value

    def recognize_intent(self, enemy_aircraft, my_aircraft, env) -> EnemyIntent:
        me = _safe_get_id(my_aircraft) or "ME"
        enemy = _safe_get_id(enemy_aircraft) or "ENEMY"

        explain = _compute_intent_explain_features(enemy_aircraft, my_aircraft)
        explain["algorithm"] = SituationAlgorithmConfig.BVR_INTENT_NEW.value

        try:
            intent = self.bvr_intent_new.recognize(enemy_aircraft, my_aircraft, env)
            model_meta = self.bvr_intent_new.get_last_infer_info(enemy_aircraft) or {}
        except Exception as exc:  # pragma: no cover
            if trace_event is not None:
                trace_event(
                    event="intent_algorithm_exception",
                    env=env,
                    module="situation_algorithm_switcher",
                    kind="INTENT",
                    status="NEUTRAL",
                    ownship=me,
                    enemy=enemy,
                    message="Intent algorithm exception; return neutral intent.",
                    data={"algorithm": explain["algorithm"], "exception": str(exc)},
                )
            intent = "NEUTRAL"
            model_meta = {
                "algorithm": explain["algorithm"],
                "used_model": False,
                "reason": f"exception_neutral: {exc}",
            }

        explain["model_used"] = model_meta.get("used_model")
        explain["model_reason"] = model_meta.get("reason")
        explain["model_confidence"] = model_meta.get("confidence")
        explain["model_buf_len"] = model_meta.get("buf_len")
        explain["model_window_size"] = model_meta.get("window_size")
        if model_meta.get("buf_len") is not None and model_meta.get("window_size") is not None:
            explain["model_window"] = f"{model_meta.get('buf_len')}/{model_meta.get('window_size')}"
        explain["model_pred_label"] = model_meta.get("pred_label")

        last_feat = model_meta.get("last_feat") or {}
        for key, value in last_feat.items():
            explain.setdefault(key, value)

        try:
            heading = explain.get("heading_to_me_deg")
            closing = explain.get("closing_m_s")
            if heading is not None and closing is not None:
                if heading < 60.0 and closing > 50.0:
                    explain["rule_hint"] = "Enemy noses toward us and keeps closing."
                elif heading > 120.0 and closing < -50.0:
                    explain["rule_hint"] = "Enemy points away and opens distance."
                else:
                    explain["rule_hint"] = "Geometry is mixed; intent tends to neutral."
        except Exception:
            pass

        try:
            if trace_throttle is not None and env is not None:
                step = int(getattr(env, "current_step", 0) or 0)
                dt = float(getattr(env, "time_interval", 0.2) or 0.2)
                interval_s = float(os.getenv("TACTICAL_INTENT_EXPLAIN_INTERVAL_S", "") or 20.0)
                min_steps = max(1, int(round(interval_s / dt)))
                pair_key = f"{me}->{enemy}"
                prev = self._intent_explain_log_state.get(pair_key) or {}
                prev_intent = prev.get("intent")
                prev_step = int(prev.get("step") or -10**9)
                intent_str = str(intent)
                if prev_intent != intent_str and (step - prev_step) >= min_steps:
                    self._intent_explain_log_state[pair_key] = {"intent": intent_str, "step": step}
                    trace_throttle(
                        key=f"intent_explain:{pair_key}",
                        min_steps=min_steps,
                        title="intent_explain",
                        env=env,
                        module="situation_algorithm_switcher",
                        kind="INTENT",
                        status=intent_str,
                        ownship=me,
                        enemy=enemy,
                        message="Log intent explain data only when intent changes.",
                        data={
                            "intent": intent_str,
                            "algorithm": explain.get("algorithm"),
                            "model_used": explain.get("model_used"),
                            "model_reason": explain.get("model_reason"),
                            "model_confidence": explain.get("model_confidence"),
                            "model_window": explain.get("model_window"),
                            "model_pred_label": explain.get("model_pred_label"),
                            "rule_hint": explain.get("rule_hint"),
                            "distance_km": explain.get("distance_km"),
                            "closing_m_s": explain.get("closing_m_s"),
                        },
                    )
        except Exception:
            pass

        try:
            if trace_if_changed is not None and env is not None:
                pair_key = f"{me}->{enemy}"
                signature = (
                    str(explain.get("algorithm")),
                    explain.get("model_used"),
                    str(explain.get("model_reason")),
                )
                trace_if_changed(
                    key=f"intent_model_state:{pair_key}",
                    value=signature,
                    title="intent_model_state",
                    env=env,
                    module="situation_algorithm_switcher",
                    kind="INTENT",
                    status="STATE",
                    ownship=me,
                    enemy=enemy,
                    message="Record model-state transitions for intent inference.",
                    data={
                        "algorithm": explain.get("algorithm"),
                        "model_used": explain.get("model_used"),
                        "model_reason": explain.get("model_reason"),
                        "model_confidence": explain.get("model_confidence"),
                        "model_window": explain.get("model_window"),
                        "model_pred_label": explain.get("model_pred_label"),
                    },
                )
        except Exception:
            pass

        try:
            self._maybe_log_intent_summary(
                my_aircraft=my_aircraft,
                enemy_aircraft=enemy_aircraft,
                env=env,
                intent=intent,
                explain=explain,
            )
        except Exception:
            pass

        return intent

    def _maybe_log_intent_summary(self, my_aircraft, enemy_aircraft, env, intent: str, explain: Dict[str, Any]) -> None:
        step = int(getattr(env, "current_step", 0) or 0) if env is not None else 0
        dt = float(getattr(env, "time_interval", 0.2) or 0.2) if env is not None else 0.2
        try:
            interval_s = float(os.getenv("TACTICAL_INTENT_STDLOG_INTERVAL_S", "") or 30.0)
        except Exception:
            interval_s = 30.0
        try:
            change_interval_s = float(os.getenv("TACTICAL_INTENT_CHANGE_LOG_INTERVAL_S", "") or 15.0)
        except Exception:
            change_interval_s = 15.0
        min_steps = max(1, int(round(interval_s / max(dt, 1e-6))))
        min_change_steps = max(1, int(round(change_interval_s / max(dt, 1e-6))))

        me = _safe_get_id(my_aircraft) or "ME"
        enemy = _safe_get_id(enemy_aircraft) or "ENEMY"
        pair_key = f"{me}->{enemy}"

        state = self._intent_stdlog_state.get(pair_key) or {}
        prev_intent = state.get("intent")
        prev_reason = state.get("model_reason")
        prev_used_model = state.get("model_used")
        prev_step = int(state.get("step") or -10**9)
        window_ready_logged = bool(state.get("window_ready_logged"))

        current_intent = str(intent)
        current_reason = str(explain.get("model_reason"))
        current_used_model = explain.get("model_used")
        algorithm = str(explain.get("algorithm") or SituationAlgorithmConfig.BVR_INTENT_NEW.value)

        buf_len = explain.get("model_buf_len")
        window_size = explain.get("model_window_size")
        try:
            window_ready = (
                buf_len is not None
                and window_size is not None
                and int(buf_len) >= max(1, int(window_size))
            )
        except Exception:
            window_ready = False

        should_log = False
        if not state:
            should_log = True
        elif prev_reason != current_reason or prev_used_model != current_used_model:
            should_log = True
        elif (step - prev_step) >= min_steps:
            should_log = True

        if window_ready and not window_ready_logged:
            online_parts = [
                f"[意图模型就绪] {pair_key}",
                f"algo={algorithm}",
                f"intent={current_intent}",
                f"model={'Y' if current_used_model else 'N'}",
                f"reason={current_reason}",
            ]
            if explain.get("model_pred_label"):
                online_parts.append(f"label={explain.get('model_pred_label')}")
            if explain.get("model_confidence") is not None:
                try:
                    online_parts.append(f"conf={float(explain.get('model_confidence')):.3f}")
                except Exception:
                    online_parts.append(f"conf={explain.get('model_confidence')}")
            if explain.get("model_window"):
                online_parts.append(f"window={explain.get('model_window')}")
            if explain.get("distance_km") is not None:
                try:
                    online_parts.append(f"dist={float(explain.get('distance_km')):.1f}km")
                except Exception:
                    pass
            if explain.get("closing_m_s") is not None:
                try:
                    online_parts.append(f"closing={float(explain.get('closing_m_s')):.1f}m/s")
                except Exception:
                    pass
            if explain.get("heading_to_me_deg") is not None:
                try:
                    online_parts.append(f"aspect={float(explain.get('heading_to_me_deg')):.1f}deg")
                except Exception:
                    pass
            LOG.info(" | ".join(online_parts))
            window_ready_logged = True

        confidence = explain.get("model_confidence")
        intent_change_ready = bool(
            prev_intent
            and prev_intent != current_intent
            and window_ready
            and (step - prev_step) >= min_change_steps
        )
        if intent_change_ready:
            try:
                intent_change_ready = float(confidence) >= 0.55
            except Exception:
                intent_change_ready = False

        if intent_change_ready:
            change_parts = [
                f"[意图变化] {pair_key}",
                f"algo={algorithm}",
                f"intent={prev_intent}->{current_intent}",
                f"model={'Y' if current_used_model else 'N'}",
                f"reason={current_reason}",
            ]
            if explain.get("model_pred_label"):
                change_parts.append(f"label={explain.get('model_pred_label')}")
            if confidence is not None:
                try:
                    change_parts.append(f"conf={float(confidence):.3f}")
                except Exception:
                    change_parts.append(f"conf={confidence}")
            if explain.get("model_window"):
                change_parts.append(f"window={explain.get('model_window')}")
            if explain.get("distance_km") is not None:
                try:
                    change_parts.append(f"dist={float(explain.get('distance_km')):.1f}km")
                except Exception:
                    pass
            if explain.get("closing_m_s") is not None:
                try:
                    change_parts.append(f"closing={float(explain.get('closing_m_s')):.1f}m/s")
                except Exception:
                    pass
            if explain.get("heading_to_me_deg") is not None:
                try:
                    change_parts.append(f"aspect={float(explain.get('heading_to_me_deg')):.1f}deg")
                except Exception:
                    pass
            LOG.info(" | ".join(change_parts))
            should_log = False

        if not should_log:
            self._intent_stdlog_state[pair_key] = {
                "intent": current_intent,
                "model_reason": current_reason,
                "model_used": current_used_model,
                "step": step,
                "window_ready_logged": window_ready_logged,
            }
            return

        self._intent_stdlog_state[pair_key] = {
            "intent": current_intent,
            "model_reason": current_reason,
            "model_used": current_used_model,
            "step": step,
            "window_ready_logged": window_ready_logged,
        }

        distance_km = explain.get("distance_km")
        closing_m_s = explain.get("closing_m_s")
        heading_to_me_deg = explain.get("heading_to_me_deg")
        enemy_radar_mode = explain.get("enemy_radar_mode")
        pred_label = explain.get("model_pred_label")
        model_window = explain.get("model_window")
        used_model = explain.get("model_used")
        reason = explain.get("model_reason")

        parts = [
            f"[意图识别] {pair_key}",
            f"algo={algorithm}",
            f"intent={current_intent}",
            f"model={'Y' if used_model else 'N'}",
            f"reason={reason}",
        ]
        if pred_label:
            parts.append(f"label={pred_label}")
        if confidence is not None:
            try:
                parts.append(f"conf={float(confidence):.3f}")
            except Exception:
                parts.append(f"conf={confidence}")
        if model_window:
            parts.append(f"window={model_window}")
        if distance_km is not None:
            try:
                parts.append(f"dist={float(distance_km):.1f}km")
            except Exception:
                parts.append(f"dist={distance_km}km")
        if closing_m_s is not None:
            try:
                parts.append(f"closing={float(closing_m_s):.1f}m/s")
            except Exception:
                parts.append(f"closing={closing_m_s}m/s")
        if heading_to_me_deg is not None:
            try:
                parts.append(f"aspect={float(heading_to_me_deg):.1f}deg")
            except Exception:
                parts.append(f"aspect={heading_to_me_deg}deg")
        if enemy_radar_mode not in (None, "", "None"):
            parts.append(f"radar={enemy_radar_mode}")

        LOG.info(" | ".join(parts))


_switcher = SituationAlgorithmSwitcher()


def get_situation_algorithm_switcher():
    return _switcher


def recognize_enemy_intent(enemy_aircraft, my_aircraft, env) -> EnemyIntent:
    return _switcher.recognize_intent(enemy_aircraft, my_aircraft, env)
