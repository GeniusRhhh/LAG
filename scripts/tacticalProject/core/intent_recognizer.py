"""Project-wide intent compatibility layer backed by bvr_intent_new."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from intent_runtime_bridge import infer_project_intent, project_intent_to_core_enum_name


LOG = logging.getLogger(__name__)


class EnemyIntent(Enum):
    ATTACK = "ATTACK"
    COORDINATED = "COORDINATED"
    FEINT = "FEINT"
    JAMMING = "JAMMING"
    RECONNAISSANCE = "RECONNAISSANCE"
    DEFENSIVE = "DEFENSIVE"
    ESCAPE = "ESCAPE"


class FriendlyIntent(Enum):
    AGGRESSIVE_CLEAR = "AGGRESSIVE_CLEAR"
    CONSERVATIVE_CLEAR = "CONSERVATIVE_CLEAR"
    DEFENSIVE = "DEFENSIVE"


def _safe_agent_id(aircraft: Any, default: str) -> str:
    for attr in ("agent_id", "callsign", "name", "uid", "id"):
        try:
            value = getattr(aircraft, attr, None)
        except Exception:
            value = None
        if value:
            return str(value)
    return default


class ProjectIntentAdapter:
    """Project-wide object/env intent adapter backed by bvr_intent_new."""

    _logged_backend = False

    def __init__(self):
        self.intent_history: Dict[str, List[EnemyIntent]] = {}
        self.intent_confidence: Dict[str, float] = {}
        if not self.__class__._logged_backend:
            LOG.info("ProjectIntentAdapter: using bvr_intent_new")
            self.__class__._logged_backend = True

    def recognize_enemy_intent(self, my_aircraft, enemy_aircraft) -> str:
        enemy_id = _safe_agent_id(enemy_aircraft, "B0100")
        my_id = _safe_agent_id(my_aircraft, "A0100")
        intent, meta = infer_project_intent(
            my_aircraft,
            enemy_aircraft,
            my_agent_id=my_id,
            enemy_agent_id=enemy_id,
        )
        self.intent_confidence[enemy_id] = float(meta.get("confidence") or 0.0)
        return str(intent)

    def recognize_enemy_intent_env(
        self,
        env,
        enemy_id: str,
        my_aircraft_list: List,
    ) -> EnemyIntent:
        enemy_aircraft = (getattr(env, "agents", {}) or {}).get(enemy_id)
        if enemy_aircraft is None or not getattr(enemy_aircraft, "is_alive", False):
            return EnemyIntent.ESCAPE

        my_aircraft = self._select_reference_aircraft(my_aircraft_list)
        if my_aircraft is None:
            return EnemyIntent.RECONNAISSANCE

        my_id = _safe_agent_id(my_aircraft, "A0100")
        intent, meta = infer_project_intent(
            my_aircraft,
            enemy_aircraft,
            env=env,
            my_agent_id=my_id,
            enemy_agent_id=str(enemy_id),
        )
        mapped = EnemyIntent[project_intent_to_core_enum_name(intent)]
        self.intent_history.setdefault(str(enemy_id), []).append(mapped)
        self.intent_history[str(enemy_id)] = self.intent_history[str(enemy_id)][-5:]
        self.intent_confidence[str(enemy_id)] = float(meta.get("confidence") or 0.0)
        return mapped

    def get_intent_confidence(self, enemy_id: str) -> float:
        return float(self.intent_confidence.get(str(enemy_id), 0.5))

    def _select_reference_aircraft(self, my_aircraft_list: List) -> Optional[Any]:
        for aircraft in my_aircraft_list or []:
            if aircraft is not None and getattr(aircraft, "is_alive", False):
                return aircraft
        return None


class IntentRecognizer(ProjectIntentAdapter):
    """Compatibility alias for legacy imports."""
