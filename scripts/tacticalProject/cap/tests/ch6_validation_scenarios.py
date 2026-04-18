from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from scenario_generator import (
    AltitudeProfile,
    ApproachDirection,
    AwacsStatus,
    EnemyScenario,
    FormationType,
    ManeuverType,
    SpeedProfile,
)


@dataclass(frozen=True)
class Ch6ScenarioSpec:
    scenario_id: str
    title: str
    thesis_focus: str
    enemy_control_mode: str
    expected_tactic_keywords: Tuple[str, ...]
    expected_zone: str
    max_steps: int
    design_intent: str
    control_contract: Tuple[str, ...]
    scenario: EnemyScenario


def _build_specs() -> Dict[str, Ch6ScenarioSpec]:
    specs = [
        Ch6ScenarioSpec(
            scenario_id="S1",
            title="Low-Risk Long-Range Intercept",
            thesis_focus="Validate the low-risk decision branch, early warning fusion, stable tracking buildup, and the long-range attack chain under clean information conditions.",
            enemy_control_mode="scripted",
            expected_tactic_keywords=(
                "T_DS",
                "DRAG",
                "T_PA",
                "PINCER",
                "T_HL",
                "HIGH_LOW",
                "T_SBS",
                "SIDE",
            ),
            expected_zone="LOW",
            max_steps=6000,
            design_intent="Enemy remains predominantly in the low-risk zone, presents only weak tactical variation, and turns away before a deep sustained medium-risk push.",
            control_contract=(
                "standard four-ship front approach",
                "normal AWACS throughout",
                "mild split and re-center in low-risk zone",
                "early turn-away before deep medium-risk dwell",
            ),
            scenario=EnemyScenario(
                scenario_id="CH6_S1",
                formation=FormationType.STANDARD,
                altitude=AltitudeProfile.MEDIUM,
                direction=ApproachDirection.FRONT,
                speed=SpeedProfile.NORMAL,
                maneuver=ManeuverType.CH6_LOW_RISK_FEINT,
                awacs_status=AwacsStatus.NORMAL,
                difficulty=2,
                initial_distance=330.0,
                description="Chapter 6 Scene 1: deterministic low-risk long-range intercept presentation",
                validation_points="Detection, intent, stable tracking buildup, low-risk tactic choice, long-range attack chain.",
            ),
        ),
        Ch6ScenarioSpec(
            scenario_id="S2",
            title="Medium-Risk Sustained Pressure",
            thesis_focus="Validate search recovery, tracking continuity, prelaunch gate discipline, relay guidance, and the medium-risk decision branch under incomplete information.",
            enemy_control_mode="scripted",
            expected_tactic_keywords=(
                "T_FB",
                "FRONT_BACK",
                "T_TE",
                "TACTICAL_EVASION",
                "T_TT",
                "TACTICAL_TURN",
            ),
            expected_zone="MEDIUM",
            max_steps=6000,
            design_intent="Enemy deliberately presses into the medium-risk zone, stays there long enough to stress tracking and relay, then performs one controlled recommit instead of free native wandering.",
            control_contract=(
                "wide-front mixed-altitude approach",
                "intermittent AWACS availability",
                "repeatable medium-risk dwell with pairwise offset pressure",
                "controlled recommit without deep high-risk penetration",
            ),
            scenario=EnemyScenario(
                scenario_id="CH6_S2",
                formation=FormationType.WIDE,
                altitude=AltitudeProfile.MIXED,
                direction=ApproachDirection.FRONT,
                speed=SpeedProfile.HIGH_SPEED,
                maneuver=ManeuverType.CH6_MEDIUM_RISK_PRESSURE,
                awacs_status=AwacsStatus.INTERMITTENT,
                difficulty=4,
                initial_distance=285.0,
                description="Chapter 6 Scene 2: deterministic medium-risk sustained pressure with intermittent AWACS",
                validation_points="Search recovery, stable tracking, prelaunch gate, relay guidance, medium-risk tactic control.",
            ),
        ),
        Ch6ScenarioSpec(
            scenario_id="S3",
            title="High-Risk Low-Altitude Penetration",
            thesis_focus="Validate penetration intent recognition, urgent high-risk response, aircraft assignment under emergency pressure, and high-value asset protection.",
            enemy_control_mode="scripted",
            expected_tactic_keywords=(
                "T_TE",
                "TACTICAL_EVASION",
                "T_TT",
                "TACTICAL_TURN",
            ),
            expected_zone="HIGH",
            max_steps=6000,
            design_intent="One enemy pair stays low and keeps penetrating long enough to trigger the high-risk branch, while the other pair pops up later to increase urgency and vertical complexity.",
            control_contract=(
                "low-altitude start",
                "fast frontal penetration",
                "delayed popup by one pair",
                "brief high-risk trigger followed by turn-away",
            ),
            scenario=EnemyScenario(
                scenario_id="CH6_S3",
                formation=FormationType.WIDE,
                altitude=AltitudeProfile.POPUP_LOW,
                direction=ApproachDirection.FRONT,
                speed=SpeedProfile.HIGH_SPEED,
                maneuver=ManeuverType.CH6_HIGH_RISK_PENETRATION,
                awacs_status=AwacsStatus.NORMAL,
                difficulty=5,
                initial_distance=125.0,
                description="Chapter 6 Scene 3: deterministic low-altitude penetration with a delayed popup pair",
                validation_points="Penetration intent, urgent high-risk tactic, aircraft assignment, evasion, turn-back, mission protection effect.",
            ),
        ),
    ]
    return {spec.scenario_id: spec for spec in specs}


SCENARIO_SPECS = _build_specs()


def get_scenario_spec(scenario_id: str) -> Ch6ScenarioSpec:
    key = str(scenario_id or "").strip().upper()
    if key not in SCENARIO_SPECS:
        available = ", ".join(sorted(SCENARIO_SPECS))
        raise KeyError(f"Unknown Chapter 6 scenario '{scenario_id}'. Available: {available}")
    return SCENARIO_SPECS[key]


def iter_scenario_specs() -> Iterable[Ch6ScenarioSpec]:
    for key in sorted(SCENARIO_SPECS):
        yield SCENARIO_SPECS[key]
