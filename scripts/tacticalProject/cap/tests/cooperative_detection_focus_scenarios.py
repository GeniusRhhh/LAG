from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

try:
    from scenario_generator import (
        AltitudeProfile,
        ApproachDirection,
        AwacsStatus,
        EnemyScenario,
        FormationType,
        ManeuverType,
        SpeedProfile,
    )
except ImportError:
    from cap.tests.scenario_generator import (
        AltitudeProfile,
        ApproachDirection,
        AwacsStatus,
        EnemyScenario,
        FormationType,
        ManeuverType,
        SpeedProfile,
    )


@dataclass(frozen=True)
class DetectionFocusScenarioSpec:
    scenario_id: str
    title: str
    thesis_focus: str
    design_intent: str
    stop_window_s: float
    max_post_activation_s: float
    max_steps: int
    awacs_loss_windows_s: Tuple[Tuple[float, float], ...]
    maneuver_script: Tuple[Dict[str, Any], ...]
    scenario: EnemyScenario


def _build_specs() -> Dict[str, DetectionFocusScenarioSpec]:
    specs = [
        DetectionFocusScenarioSpec(
            scenario_id="D1",
            title="Front-wide layered distributed picture",
            thesis_focus=(
                "Validate whether the cooperative detection method can quickly complete "
                "sector division and full coverage under a fully frontal, wide, layered, "
                "and distributed target picture with complete AWACS support."
            ),
            design_intent=(
                "Keep the enemy strictly frontal, but maximize lateral spread, altitude "
                "layering, and pair-level asynchronous turning so the method is stressed "
                "by wide bearing span rather than by information loss."
            ),
            stop_window_s=18.0,
            max_post_activation_s=140.0,
            max_steps=2600,
            awacs_loss_windows_s=(),
            maneuver_script=(),
            scenario=EnemyScenario(
                scenario_id="D1",
                formation=FormationType.SPREAD,
                altitude=AltitudeProfile.STAIRCASE,
                direction=ApproachDirection.FRONT,
                speed=SpeedProfile.MIXED,
                maneuver=ManeuverType.EARLY_SPLIT_TURN_BACK,
                awacs_status=AwacsStatus.NORMAL,
                difficulty=4,
                initial_distance=320.0,
                description=(
                    "[D1] Strict frontal approach. Wide 2+2 distributed picture, staircase "
                    "altitude layering, mixed speed, and repeated split-cross-return "
                    "maneuver to stress sector allocation and edge-target coverage."
                ),
                validation_points=(
                    "Full coverage time, average coverage ratio, frontal wide-picture "
                    "sector allocation, and scan-mode switching stability."
                ),
            ),
        ),
        DetectionFocusScenarioSpec(
            scenario_id="D2",
            title="Front-only information-gap and maneuver-uncertainty picture",
            thesis_focus=(
                "Validate whether the method can recover search and keep coverage high when "
                "AWACS support is intermittent and the targets repeatedly change heading and "
                "altitude while still approaching from the front."
            ),
            design_intent=(
                "Keep the enemy strictly frontal, but introduce repeated short AWACS gaps, "
                "wide frontage, altitude mismatch, and moderate split-turn-back maneuver so "
                "uncertainty rises without destroying the basic intercept geometry."
            ),
            stop_window_s=18.0,
            max_post_activation_s=160.0,
            max_steps=3000,
            awacs_loss_windows_s=(
                (46.0, 48.4),
                (55.5, 58.0),
                (65.0, 67.6),
                (75.5, 78.3),
                (86.0, 88.6),
                (97.0, 100.0),
                (109.0, 111.8),
                (122.0, 124.7),
                (136.0, 139.0),
            ),
            maneuver_script=(
                {
                    "time_s": 0.0,
                    "heading_offsets_deg": (0.0, 0.0, 0.0, 0.0),
                    "target_altitudes_km": {
                        "B0100": 11.0,
                        "B0200": 4.5,
                        "B0300": 10.0,
                        "B0400": 4.5,
                    },
                },
                {
                    "time_s": 22.0,
                    "heading_offsets_deg": (2.5, 1.0, -1.0, -2.5),
                },
                {
                    "time_s": 48.0,
                    "heading_offsets_deg": (1.5, 0.5, -0.5, -1.5),
                },
                {
                    "time_s": 78.0,
                    "heading_offsets_deg": (0.0, 0.0, 0.0, 0.0),
                    "target_altitudes_km": {
                        "B0100": 10.0,
                        "B0200": 6.2,
                        "B0300": 9.4,
                        "B0400": 6.2,
                    },
                },
                {
                    "time_s": 108.0,
                    "heading_offsets_deg": (-2.5, -1.0, 1.0, 2.5),
                },
                {
                    "time_s": 136.0,
                    "heading_offsets_deg": (0.0, 0.0, 0.0, 0.0),
                    "target_altitudes_km": {
                        "B0100": 9.8,
                        "B0200": 6.8,
                        "B0300": 9.2,
                        "B0400": 6.8,
                    },
                },
            ),
            scenario=EnemyScenario(
                scenario_id="D2",
                formation=FormationType.STANDARD,
                altitude=AltitudeProfile.MIXED,
                direction=ApproachDirection.FRONT,
                speed=SpeedProfile.NORMAL,
                maneuver=ManeuverType.STRAIGHT,
                awacs_status=AwacsStatus.NORMAL,
                difficulty=5,
                initial_distance=320.0,
                description=(
                    "[D2] Strict frontal approach. Repeated short AWACS outages, mixed "
                    "high-low altitude start, moderate frontage, and mild frontal "
                    "split-return maneuver to stress uncertainty recovery without breaking the "
                    "intercept picture."
                ),
                validation_points=(
                    "Coverage recovery after AWACS loss, full coverage continuity, search/"
                    "directed switching rationality, and robustness of minimum-sufficient "
                    "scan-mode selection."
                ),
            ),
        ),
        DetectionFocusScenarioSpec(
            scenario_id="D3",
            title="Severe frontal low-altitude penetration picture",
            thesis_focus=(
                "Validate whether the method can still rapidly shrink the effective search "
                "sector and re-establish full coverage under severe low-altitude, high-speed, "
                "front-only penetration with repeated AWACS degradation."
            ),
            design_intent=(
                "Keep the enemy strictly frontal, but compress time and distance through "
                "low-altitude penetration, high speed, delayed popup climb, and repeated "
                "short AWACS gaps so this becomes the harshest detection-focused stress case "
                "without collapsing into immediate radar activation."
            ),
            stop_window_s=12.0,
            max_post_activation_s=150.0,
            max_steps=2600,
            awacs_loss_windows_s=(
                (12.0, 14.5),
                (20.0, 22.8),
                (29.0, 31.5),
                (38.0, 41.0),
                (49.0, 52.5),
                (61.0, 64.2),
                (75.0, 78.0),
                (92.0, 95.0),
            ),
            maneuver_script=(
                {
                    "time_s": 0.0,
                    "heading_offsets_deg": (0.0, 0.0, 0.0, 0.0),
                    "target_altitudes_km": {
                        "B0100": 3.4,
                        "B0200": 3.4,
                        "B0300": 3.7,
                        "B0400": 3.7,
                    },
                },
                {
                    "time_s": 18.0,
                    "heading_offsets_deg": (0.0, 0.0, -4.0, -8.0),
                    "target_altitudes_km": {
                        "B0100": 5.6,
                        "B0200": 5.6,
                        "B0300": 6.2,
                        "B0400": 6.4,
                    },
                },
                {
                    "time_s": 38.0,
                    "heading_offsets_deg": (0.0, 0.0, 3.0, 6.0),
                    "target_altitudes_km": {
                        "B0100": 7.8,
                        "B0200": 7.8,
                        "B0300": 8.4,
                        "B0400": 8.6,
                    },
                },
                {
                    "time_s": 60.0,
                    "heading_offsets_deg": (0.0, 0.0, 0.0, 0.0),
                    "target_altitudes_km": {
                        "B0100": 8.8,
                        "B0200": 8.8,
                        "B0300": 9.0,
                        "B0400": 9.2,
                    },
                },
            ),
            scenario=EnemyScenario(
                scenario_id="D3",
                formation=FormationType.WIDE,
                altitude=AltitudeProfile.POPUP_LOW,
                direction=ApproachDirection.FRONT,
                speed=SpeedProfile.HIGH_SPEED,
                maneuver=ManeuverType.STRAIGHT,
                awacs_status=AwacsStatus.NORMAL,
                difficulty=5,
                initial_distance=310.0,
                description=(
                    "[D3] Strict frontal approach. Low-altitude high-speed penetration, "
                    "repeated short AWACS outages, and delayed popup climb to stress "
                    "compressed-time coverage recovery after radar activation."
                ),
                validation_points=(
                    "Compressed-time first full coverage, low-altitude recovery ability, "
                    "coverage continuity under repeated information degradation, and target "
                    "responsibility redistribution under severe frontal penetration."
                ),
            ),
        ),
    ]
    return {spec.scenario_id: spec for spec in specs}


SCENARIO_SPECS = _build_specs()


def get_scenario_spec(scenario_id: str) -> DetectionFocusScenarioSpec:
    key = str(scenario_id or "").strip().upper()
    if key not in SCENARIO_SPECS:
        available = ", ".join(sorted(SCENARIO_SPECS))
        raise KeyError(f"Unknown detection focus scenario '{scenario_id}'. Available: {available}")
    return SCENARIO_SPECS[key]


def iter_scenario_specs() -> Iterable[DetectionFocusScenarioSpec]:
    for key in sorted(SCENARIO_SPECS):
        yield SCENARIO_SPECS[key]


def get_all_scenarios() -> List[DetectionFocusScenarioSpec]:
    return list(iter_scenario_specs())
