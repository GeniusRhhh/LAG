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
            title="Low-Risk Frontal Probe And Early Turn-Away",
            thesis_focus="Validate the low-risk decision branch, early warning fusion, stable tracking buildup, and the long-range attack chain under clean information conditions.",
            enemy_control_mode="freeplay",
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
            design_intent="Scene 1 fixes a run_cap_simulation-like low-risk frontal initial geometry; once the run starts, both sides use the native AI freely.",
            control_contract=(
                "fixed run_cap_simulation-like frontal four-ship initial geometry",
                "normal AWACS throughout",
                "no post-start scripted enemy steering",
                "observe whether native interaction stays in the low-risk branch on its own",
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
                initial_distance=266.4,
                description="Chapter 6 Scene 1: clean low-risk frontal probe with re-center and early turn-away",
                validation_points="Detection, intent, stable tracking buildup, low-risk tactic choice, long-range attack chain.",
            ),
        ),
        Ch6ScenarioSpec(
            scenario_id="S2",
            title="Medium-Risk Sustained Pressure",
            thesis_focus="Validate search recovery, tracking continuity, prelaunch gate discipline, relay guidance, and the medium-risk decision branch under incomplete information.",
            enemy_control_mode="freeplay",
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
            design_intent="Scene 2 fixes a slightly closer mixed-altitude frontal geometry while preserving the native run_cap_simulation AI loop.",
            control_contract=(
                "fixed wide-front mixed-altitude initial geometry",
                "intermittent AWACS availability",
                "no post-start scripted enemy steering",
                "observe whether native interaction naturally develops a medium-risk picture",
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
                initial_distance=240.0,
                description="Chapter 6 Scene 2: medium-risk sustained pressure with controlled reset and single recommit",
                validation_points="Search recovery, stable tracking, prelaunch gate, relay guidance, medium-risk tactic control.",
            ),
        ),
        Ch6ScenarioSpec(
            scenario_id="S3",
            title="High-Risk Low-Altitude Penetration",
            thesis_focus="Validate penetration intent recognition, urgent high-risk response, aircraft assignment under emergency pressure, and high-value asset protection.",
            enemy_control_mode="hybrid_opening",
            expected_tactic_keywords=(
                "T_TE",
                "TACTICAL_EVASION",
                "T_TT",
                "TACTICAL_TURN",
            ),
            expected_zone="HIGH",
            max_steps=6000,
            design_intent="Scene 3 starts from a closer low-altitude penetration geometry, sustains repeated long AWACS loss, forces a short rapid popup climb toward the friendly altitude band, and then releases both sides to the native AI.",
            control_contract=(
                "fixed low-altitude penetration initial geometry",
                "repeated long AWACS loss throughout the run",
                "short opening low-altitude penetration plus rapid popup climb for the enemy only",
                "native AI takeover after the opening window",
                "no later scripted enemy steering",
            ),
            scenario=EnemyScenario(
                scenario_id="CH6_S3",
                formation=FormationType.WIDE,
                altitude=AltitudeProfile.POPUP_LOW,
                direction=ApproachDirection.FRONT,
                speed=SpeedProfile.HIGH_SPEED,
                maneuver=ManeuverType.CH6_HIGH_RISK_PENETRATION,
                awacs_status=AwacsStatus.INTERMITTENT_LONG,
                difficulty=5,
                initial_distance=150.0,
                description="Chapter 6 Scene 3: closer low-altitude penetration with repeated long AWACS loss and rapid popup climb into the main engagement altitude band",
                validation_points="High-risk low-altitude penetration, repeated long AWACS loss, rapid popup climb, urgent high-risk tactic, aircraft assignment, evasion, turn-back, mission protection effect.",
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
