"""CAP intent adapter backed by bvr_intent_new only."""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from core.situation_algorithm_switcher import get_situation_algorithm_switcher


log = logging.getLogger(__name__)


class IntentType(Enum):
    UNKNOWN = "unknown"
    ATTACK = "attack"
    RECON = "recon"
    PENETRATE = "penetrate"
    EVADE = "evade"
    RETREAT = "retreat"


class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IntentAnalysis:
    track_id: str
    intent: IntentType
    confidence: float
    threat_level: ThreatLevel
    recommended_response: str
    raw_intent: str
    pred_label: Optional[str]
    model_used: bool
    model_reason: str
    model_window: Optional[str]
    distance_km: float


_LABEL_TO_INTENT = {
    "攻击": IntentType.ATTACK,
    "协同": IntentType.ATTACK,
    "侦察": IntentType.RECON,
    "防御": IntentType.EVADE,
    "规避": IntentType.EVADE,
    "撤退": IntentType.RETREAT,
}

_INTENT_TO_TYPE = {
    "ATTACK": IntentType.ATTACK,
    "NEUTRAL": IntentType.RECON,
    "RETREAT": IntentType.RETREAT,
}


class IntentAdapter:
    """Compatibility wrapper used by CAP telemetry and verification hooks."""

    def __init__(self):
        self._analyses: Dict[str, IntentAnalysis] = {}
        self._switcher = get_situation_algorithm_switcher()
        self._summary_log_state: Dict[str, object] = {}
        log.info("Intent adapter: using bvr_intent_new via SituationAlgorithmSwitcher")

    def analyze_intent(self, tracks: List[Dict], env=None, my_aircraft=None) -> Dict[str, IntentAnalysis]:
        results: Dict[str, IntentAnalysis] = {}

        for track in tracks:
            analysis = self._build_analysis(track, env=env, my_aircraft=my_aircraft)
            results[analysis.track_id] = analysis
            self._analyses[analysis.track_id] = analysis

        self._maybe_log_summary(results, env=env)
        return results

    def _build_analysis(self, track: Dict, env=None, my_aircraft=None) -> IntentAnalysis:
        track_id = str(track.get("track_id", "unknown") or "unknown")
        raw_intent = "NEUTRAL"
        pred_label = None
        confidence = 0.0
        model_used = False
        model_reason = "no_env"
        model_window = None

        if env is not None and my_aircraft is not None:
            enemy_aircraft = (getattr(env, "agents", {}) or {}).get(track_id)
            if enemy_aircraft is not None and getattr(enemy_aircraft, "is_alive", False):
                raw_intent = str(self._switcher.recognize_intent(enemy_aircraft, my_aircraft, env) or "NEUTRAL")
                model_meta = self._switcher.bvr_intent_new.get_last_infer_info(enemy_aircraft) or {}
                pred_label = model_meta.get("pred_label")
                confidence = float(model_meta.get("confidence") or 0.0)
                model_used = bool(model_meta.get("used_model"))
                model_reason = str(model_meta.get("reason") or "unknown")
                buf_len = model_meta.get("buf_len")
                window_size = model_meta.get("window_size")
                if buf_len is not None and window_size is not None:
                    model_window = f"{buf_len}/{window_size}"

        distance_km = float(track.get("distance", 300.0) or 300.0)
        intent = self._resolve_intent_type(raw_intent, pred_label)
        threat = self._assess_threat(distance_km, intent)
        response = self._recommend_response(intent, threat)
        return IntentAnalysis(
            track_id=track_id,
            intent=intent,
            confidence=confidence,
            threat_level=threat,
            recommended_response=response,
            raw_intent=raw_intent,
            pred_label=pred_label,
            model_used=model_used,
            model_reason=model_reason,
            model_window=model_window,
            distance_km=distance_km,
        )

    def _resolve_intent_type(self, raw_intent: str, pred_label: Optional[str]) -> IntentType:
        if pred_label:
            mapped = _LABEL_TO_INTENT.get(str(pred_label))
            if mapped is not None:
                return mapped
        return _INTENT_TO_TYPE.get(str(raw_intent), IntentType.UNKNOWN)

    def _assess_threat(self, distance: float, intent: IntentType) -> ThreatLevel:
        if distance < 50:
            base_threat = ThreatLevel.CRITICAL
        elif distance < 100:
            base_threat = ThreatLevel.HIGH
        elif distance < 150:
            base_threat = ThreatLevel.MEDIUM
        else:
            base_threat = ThreatLevel.LOW

        intent_modifier = {
            IntentType.ATTACK: 1,
            IntentType.PENETRATE: 1,
            IntentType.UNKNOWN: 0,
            IntentType.RECON: -1,
            IntentType.EVADE: -1,
            IntentType.RETREAT: -2,
        }
        threat_order = [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        base_idx = threat_order.index(base_threat)
        adjusted_idx = max(0, min(3, base_idx + intent_modifier.get(intent, 0)))
        return threat_order[adjusted_idx]

    def _recommend_response(self, intent: IntentType, threat: ThreatLevel) -> str:
        if threat == ThreatLevel.CRITICAL:
            return "immediate_engage"
        if threat == ThreatLevel.HIGH:
            if intent == IntentType.ATTACK:
                return "intercept_and_engage"
            return "defensive_posture"
        if threat == ThreatLevel.MEDIUM:
            return "monitor_and_prepare"
        return "continue_patrol"

    def get_threat_level(self, track_id: str) -> Optional[ThreatLevel]:
        analysis = self._analyses.get(track_id)
        return analysis.threat_level if analysis else None

    def get_analysis(self, track_id: str) -> Optional[IntentAnalysis]:
        return self._analyses.get(track_id)

    def get_highest_threat(self) -> Optional[IntentAnalysis]:
        if not self._analyses:
            return None

        threat_priority = {
            ThreatLevel.CRITICAL: 4,
            ThreatLevel.HIGH: 3,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.LOW: 1,
        }
        return max(self._analyses.values(), key=lambda analysis: threat_priority.get(analysis.threat_level, 0))

    def _maybe_log_summary(self, results: Dict[str, IntentAnalysis], env=None) -> None:
        if env is None or not results:
            return

        step = int(getattr(env, "current_step", 0) or 0)
        dt = float(getattr(env, "time_interval", 0.2) or 0.2)
        try:
            interval_s = float(os.getenv("CAP_INTENT_SNAPSHOT_INTERVAL_S", "") or 30.0)
        except Exception:
            interval_s = 30.0
        try:
            change_interval_s = float(os.getenv("CAP_INTENT_CHANGE_LOG_INTERVAL_S", "") or 12.0)
        except Exception:
            change_interval_s = 12.0

        min_steps = max(1, int(round(interval_s / max(dt, 1e-6))))
        min_change_steps = max(1, int(round(change_interval_s / max(dt, 1e-6))))
        state = self._summary_log_state or {}

        threat_priority = {
            ThreatLevel.CRITICAL: 4,
            ThreatLevel.HIGH: 3,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.LOW: 1,
        }
        ranked = sorted(
            results.values(),
            key=lambda analysis: (
                threat_priority.get(analysis.threat_level, 0),
                analysis.confidence,
                -analysis.distance_km,
            ),
            reverse=True,
        )
        ready_count = sum(1 for analysis in ranked if self._is_model_ready(analysis.model_window))
        top_analyses = ranked[:2]
        signature = tuple(
            (
                analysis.track_id,
                analysis.intent.value,
                analysis.threat_level.value,
                analysis.pred_label or "",
                self._is_model_ready(analysis.model_window),
            )
            for analysis in top_analyses
        )

        prev_signature = tuple(state.get("signature") or ())
        prev_step = int(state.get("step") or -10**9)
        prev_ready_count = int(state.get("ready_count") or -1)
        should_log = False
        if not state:
            should_log = True
        elif ready_count != prev_ready_count:
            should_log = True
        elif signature != prev_signature and (step - prev_step) >= min_change_steps:
            should_log = True
        elif (step - prev_step) >= min_steps:
            should_log = True

        if not should_log:
            return

        summary_parts = [
            "[意图态势]",
            "algo=bvr_intent_new",
            f"contacts={len(ranked)}",
            f"ready={ready_count}/{len(ranked)}",
        ]
        for analysis in top_analyses:
            summary_parts.append(self._format_summary_contact(analysis))
        log.info(" | ".join(summary_parts))
        self._summary_log_state = {
            "signature": signature,
            "step": step,
            "ready_count": ready_count,
        }

    def _format_summary_contact(self, analysis: IntentAnalysis) -> str:
        label = analysis.pred_label or analysis.intent.value
        confidence = f"{analysis.confidence:.3f}" if analysis.confidence > 0.0 else "0.000"
        model_window = analysis.model_window or "-"
        return (
            f"{analysis.track_id}:{label}/conf={confidence}/threat={analysis.threat_level.value}"
            f"/window={model_window}/dist={analysis.distance_km:.1f}km"
        )

    @staticmethod
    def _is_model_ready(model_window: Optional[str]) -> bool:
        if not model_window or "/" not in model_window:
            return False
        try:
            current, needed = model_window.split("/", 1)
            return int(current) >= max(1, int(needed))
        except Exception:
            return False
