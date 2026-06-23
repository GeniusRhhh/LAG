from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent.parent
RESULT_BATCH_DIR = PROJECT_ROOT / "cap_results" / "Chapter6_validation" / "ALL_20260511_090228"
DOC_ASSET_DIR = DOCS_DIR / "第六章仿真验证文档0511_assets"
SCENE_OVERVIEW_SOURCES = {
    "S1": DOCS_DIR / "image.png",
    "S2": DOCS_DIR / "image-1.png",
    "S3": DOCS_DIR / "image-2.png",
}

SCENARIO_MAP = {
    "S1": RESULT_BATCH_DIR / "S1_20260511_090228",
    "S2": RESULT_BATCH_DIR / "S2_20260511_091756",
    "S3": RESULT_BATCH_DIR / "S3_20260511_093323",
}

CONTROL_PHASE_ORDER = (
    "BEYOND_NLT",
    "NLT_MELD",
    "MELD_MTR",
    "MTR_LR",
    "LR_TR",
    "TR_DOR",
    "DOR_DR",
    "DR_MAR",
    "BEYOND_MAR",
)
THREAT_LEVEL_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
INTENT_CLASS_ORDER = ("侦察", "防御", "撤退", "攻击")


def _safe_float(value: object, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _norm_label(value: object, default: str = "UNKNOWN") -> str:
    text = str(value or "").strip()
    return text if text else default


def _normalize_phase_label(value: object, default: str = "") -> str:
    text = _norm_label(value, default=default)
    if not text:
        return default
    return text.split(".")[-1]


def _series_float(rows: List[Dict[str, object]], key: str, default: float = float("nan")) -> List[float]:
    return [_safe_float(row.get(key), default) for row in rows]


def _series_int(rows: List[Dict[str, object]], key: str, default: int = 0) -> List[int]:
    return [_safe_int(row.get(key), default) for row in rows]


def _guidance_metric_series(
    timeline_rows: List[Dict[str, object]],
    *,
    base_key: str,
    prefer_unique: bool = True,
) -> List[int]:
    if prefer_unique:
        unique_key = f"{base_key}_unique_count"
        if any(unique_key in row for row in timeline_rows):
            return _series_int(timeline_rows, unique_key)
    if any(base_key in row for row in timeline_rows):
        return _series_int(timeline_rows, base_key)
    raw_key = f"{base_key}_raw_count"
    if any(raw_key in row for row in timeline_rows):
        return _series_int(timeline_rows, raw_key)
    return [0 for _ in timeline_rows]


def _running_ratio_series(numerator: Sequence[int], denominator: Sequence[int]) -> List[float]:
    result: List[float] = []
    for num, den in zip(numerator, denominator):
        result.append(float(num) / float(den) if den > 0 else 0.0)
    return result


def _plot_categorical(
    ax,
    times: List[float],
    values: List[str],
    title: str,
    *,
    preferred_order: Optional[Iterable[str]] = None,
) -> None:
    labels: List[str] = []
    preferred = [str(item) for item in list(preferred_order or []) if str(item or "").strip()]
    for label in preferred:
        if any(_norm_label(item, default="") == label for item in values) and label not in labels:
            labels.append(label)
    for item in values:
        norm = _norm_label(item, default="")
        if norm and norm not in labels:
            labels.append(norm)
    mapping = {label: idx for idx, label in enumerate(labels)}
    y_values = [mapping.get(_norm_label(item, default=""), np.nan) for item in values]
    ax.step(times, y_values, where="post", linewidth=1.8)
    ax.set_title(title)
    ax.set_yticks(list(mapping.values()))
    ax.set_yticklabels(labels)
    ax.grid(True, alpha=0.3)


def _plot_level_timeline(
    ax,
    times: List[float],
    values: List[object],
    title: str,
    level_order: Dict[str, int],
) -> None:
    ordered = sorted(level_order.items(), key=lambda item: item[1])
    labels = [item[0] for item in ordered]
    mapping = {str(label).lower(): int(level) for label, level in ordered}
    y_values = []
    for item in values:
        text = _norm_label(item, default="").lower()
        y_values.append(mapping.get(text, np.nan) if text else np.nan)
    ax.step(times, y_values, where="post", linewidth=1.8)
    ax.set_title(title)
    ax.set_yticks([level_order[label] for label in labels])
    ax.set_yticklabels(labels)
    ax.grid(True, alpha=0.3)


def _plot_multi_level_lines(
    ax,
    times: List[float],
    series_map: Dict[str, List[object]],
    title: str,
    level_order: Dict[str, int],
) -> None:
    for label, values in series_map.items():
        y_values = []
        for value in values:
            text = _norm_label(value, default="").lower()
            y_values.append(level_order.get(text, np.nan) if text else np.nan)
        ax.step(times, y_values, where="post", linewidth=1.5, label=label)
    ordered = sorted(level_order.items(), key=lambda item: item[1])
    ax.set_title(title)
    ax.set_yticks([level for _, level in ordered])
    ax.set_yticklabels([label for label, _ in ordered])
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_zone_series(rows: List[Dict[str, object]], enemy_id: str) -> List[str]:
    result: List[str] = []
    for row in rows:
        try:
            payload = json.loads(str(row.get("zone_by_enemy_json", "{}") or "{}"))
        except Exception:
            payload = {}
        result.append(_norm_label(payload.get(enemy_id), default="OUTSIDE"))
    return result


def _plot_batch_bars(
    ax,
    labels: List[str],
    values: List[float],
    title: str,
    *,
    suffix: str = "",
    color: str = "#2F6B9A",
) -> None:
    plot_values = [0.0 if not np.isfinite(value) else float(value) for value in values]
    bars = ax.bar(labels, plot_values, color=color, edgecolor="black", linewidth=0.4)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        label = "NA" if not np.isfinite(value) else f"{value:.2f}{suffix}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(plot_values + [1.0]) * 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _load_scenario_payload(scenario_key: str) -> Tuple[List[Dict[str, object]], Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    scenario_dir = SCENARIO_MAP[scenario_key]
    timeline_rows = _load_csv(scenario_dir / "timeline.csv")
    summary = _load_json(scenario_dir / "summary.json")
    thesis_csv_dir = scenario_dir / "thesis_assets" / "csv"
    missile_rows = _load_csv(thesis_csv_dir / "missile_end_reasons.csv")
    kill_rows = _load_csv(thesis_csv_dir / "kill_timeline.csv")
    return timeline_rows, summary, missile_rows, kill_rows


def _plot_overview_rows(timeline_rows: List[Dict[str, object]], asset_dir: Path) -> None:
    times = _series_float(timeline_rows, "time_s", 0.0)
    bullseye_series = _series_float(timeline_rows, "nearest_enemy_bullseye_km")
    friendly_distance_series = _series_float(timeline_rows, "nearest_enemy_to_friendly_km")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.step(times, _series_int(timeline_rows, "truth_low_count"), where="post", label="Truth low", linewidth=2.0)
    ax.step(times, _series_int(timeline_rows, "truth_medium_count"), where="post", label="Truth medium", linewidth=2.0)
    ax.step(times, _series_int(timeline_rows, "truth_high_count"), where="post", label="Truth high", linewidth=2.0)
    if any("picture_low_count" in row for row in timeline_rows):
        ax.plot(times, _series_int(timeline_rows, "picture_low_count"), label="Picture low", linewidth=1.2, alpha=0.9)
        ax.plot(times, _series_int(timeline_rows, "picture_medium_count"), label="Picture medium", linewidth=1.2, alpha=0.9)
        ax.plot(times, _series_int(timeline_rows, "picture_high_count"), label="Picture high", linewidth=1.2, alpha=0.9)
    ax.set_title("Risk-Zone Occupancy: truth vs decision picture")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "overview_a.png")

    fig, ax = plt.subplots(figsize=(14, 3.2))
    _plot_categorical(ax, times, [row.get("enemy_script_phase_name", "") for row in timeline_rows], "Enemy scene phase")
    _save(fig, asset_dir / "overview_b.png")

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.step(times, _series_int(timeline_rows, "enemy_script_awacs_available"), where="post", label="AWACS available", linewidth=1.8)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Enemy information condition and trigger distance")
    ax2 = ax.twinx()
    ax2.plot(times, _series_float(timeline_rows, "enemy_script_reference_distance_km"), label="Reference distance (km)", linewidth=1.3, color="#C0504D")
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "overview_c.png")

    fig, ax = plt.subplots(figsize=(14, 3.2))
    _plot_categorical(ax, times, [row.get("left_tactic", "") for row in timeline_rows], "Left pair tactic")
    _save(fig, asset_dir / "overview_d.png")

    fig, ax = plt.subplots(figsize=(14, 3.2))
    _plot_categorical(ax, times, [row.get("right_tactic", "") for row in timeline_rows], "Right pair tactic")
    _save(fig, asset_dir / "overview_e.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.step(times, _series_int(timeline_rows, "friendly_alive"), where="post", label="Friendly alive", linewidth=1.9)
    ax.step(times, _series_int(timeline_rows, "enemy_alive"), where="post", label="Enemy alive", linewidth=1.9)
    ax.set_ylabel("Alive count")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(times, bullseye_series, label="Nearest enemy to bullseye", linewidth=1.3, color="#C0504D")
    ax2.plot(times, friendly_distance_series, label="Nearest enemy to friendly", linewidth=1.3, color="#4F81BD")
    ax.set_title("Survivability and range compression")
    ax.set_xlabel("Time (s)")
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right", fontsize=8)
    _save(fig, asset_dir / "overview_f.png")


def _plot_truth_zone_geometry_panels(timeline_rows: List[Dict[str, object]], asset_dir: Path) -> None:
    times = _series_float(timeline_rows, "time_s", 0.0)

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.step(times, _series_int(timeline_rows, "truth_low_count"), where="post", label="Truth low", linewidth=2.0)
    ax.step(times, _series_int(timeline_rows, "truth_medium_count"), where="post", label="Truth medium", linewidth=2.0)
    ax.step(times, _series_int(timeline_rows, "truth_high_count"), where="post", label="Truth high", linewidth=2.0)
    ax.set_title("Truth risk-zone counts")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig01_truth_zone_geometry_a.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.step(times, _series_int(timeline_rows, "picture_low_count"), where="post", label="Picture low", linewidth=2.0)
    ax.step(times, _series_int(timeline_rows, "picture_medium_count"), where="post", label="Picture medium", linewidth=2.0)
    ax.step(times, _series_int(timeline_rows, "picture_high_count"), where="post", label="Picture high", linewidth=2.0)
    ax.set_title("Decision-picture risk-zone counts")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig01_truth_zone_geometry_b.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_int(timeline_rows, "picture_total_count"), label="Picture threats", linewidth=1.7)
    ax.plot(times, _series_int(timeline_rows, "mission_total_threats"), label="Mission-evaluator threats", linewidth=1.7)
    ax.set_title("Perceived hostile counts")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig01_truth_zone_geometry_c.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_float(timeline_rows, "nearest_enemy_bullseye_km"), label="Enemy to bullseye (km)", linewidth=1.7)
    ax.plot(times, _series_float(timeline_rows, "nearest_enemy_to_friendly_km"), label="Enemy to formation (km)", linewidth=1.7)
    ax.set_title("Geometric compression")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig01_truth_zone_geometry_d.png")


def _split_scene_overview_image(source_path: Path, asset_dir: Path) -> None:
    if not source_path.exists():
        return
    image = Image.open(source_path).convert("RGB")
    width, height = image.size
    crop_specs = [
        ("a", (0, 0, int(round(width * 0.45)), height)),
        ("b", (int(round(width * 0.25)), 0, int(round(width * 0.75)), height)),
        ("c", (int(round(width * 0.55)), 0, width, height)),
    ]
    for suffix, box in crop_specs:
        crop = image.crop(box)
        crop.save(asset_dir / f"fig00_scene_overview_{suffix}.png")


def _plot_detection_pipeline(timeline_rows: List[Dict[str, object]], asset_dir: Path) -> None:
    times = _series_float(timeline_rows, "time_s", 0.0)

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_int(timeline_rows, "awacs_track_count"), label="AWACS", linewidth=1.7)
    ax.plot(times, _series_int(timeline_rows, "radar_track_count"), label="Radar unique", linewidth=1.7)
    ax.plot(times, _series_int(timeline_rows, "stable_ready_count"), label="Stable ready", linewidth=1.7)
    ax.plot(times, _series_int(timeline_rows, "stable_tracking_target_count"), label="Stable targets", linewidth=1.3)
    ax.set_title("Tracking quality buildup")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig02_detection_tracking_pipeline_a.png")

    fig, ax = plt.subplots(figsize=(14, 3.2))
    _plot_categorical(ax, times, [row.get("detect_mode", "") for row in timeline_rows], "Detection mode")
    _save(fig, asset_dir / "fig02_detection_tracking_pipeline_b.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_float(timeline_rows, "scan_coverage_total_deg", 0.0), label="Coverage total (deg)", linewidth=1.6)
    ax.plot(times, _series_float(timeline_rows, "scan_overlap_deg", 0.0), label="Coverage overlap (deg)", linewidth=1.6)
    ax.plot(times, _series_float(timeline_rows, "confidence_radius_km", 0.0), label="Confidence radius (km)", linewidth=1.2)
    ax.set_title("Detection geometry")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig02_detection_tracking_pipeline_c.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_int(timeline_rows, "radar_track_count"), label="Radar unique tracks", linewidth=1.6)
    ax.plot(times, _series_int(timeline_rows, "scan_target_count"), label="Unique scan targets", linewidth=1.2)
    ax.plot(times, _series_int(timeline_rows, "intent_contact_count"), label="Intent contacts", linewidth=1.6)
    ax.plot(times, _series_int(timeline_rows, "intent_ready_count"), label="Intent model-ready contacts", linewidth=1.3)
    ax.set_title("Detection-to-intent pipeline load")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig02_detection_tracking_pipeline_d.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_int(timeline_rows, "scan_assignment_count"), label="Scan assignments", linewidth=1.6)
    ax.plot(times, _series_int(timeline_rows, "assignment_target_count"), label="Tactic-assigned targets", linewidth=1.6)
    ax.plot(times, _series_int(timeline_rows, "active_relay_flag"), label="Active relay", linewidth=1.2)
    ax.set_title("Assignment and relay activation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig02_detection_tracking_pipeline_e.png")


def _plot_coop_chain(timeline_rows: List[Dict[str, object]], summary: Dict[str, object], asset_dir: Path) -> None:
    times = _series_float(timeline_rows, "time_s", 0.0)
    first_detect_map = dict(summary.get("radar_first_detect_time_by_target_s", {}) or {})
    detect_times = sorted(
        float(value)
        for value in first_detect_map.values()
        if value not in (None, "") and np.isfinite(float(value))
    )
    cumulative_detected: List[int] = []
    detect_index = 0
    current_count = 0
    for time_s in times:
        while detect_index < len(detect_times) and float(time_s) >= float(detect_times[detect_index]) - 1e-9:
            current_count += 1
            detect_index += 1
        cumulative_detected.append(current_count)

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.step(times, cumulative_detected, where="post", label="Cumulative first-detected targets", linewidth=1.8)
    ax.plot(times, _series_int(timeline_rows, "radar_track_count"), label="Radar currently tracked targets", linewidth=1.4)
    ax.set_title("Cooperative detection closure")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig02b_cooperative_chain_timeline_a.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_int(timeline_rows, "stable_ready_count"), label="Stable ready", linewidth=1.7)
    ax.plot(times, _series_int(timeline_rows, "stable_tracking_target_count"), label="Stable tracked targets", linewidth=1.7)
    ax.plot(times, _series_int(timeline_rows, "assignment_target_count"), label="Assigned targets", linewidth=1.2)
    ax.set_title("Cooperative tracking buildup")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig02b_cooperative_chain_timeline_b.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.step(times, _guidance_metric_series(timeline_rows, base_key="gate_pass", prefer_unique=True), where="post", label="Gate pass (unique)", linewidth=1.7)
    ax.step(times, _guidance_metric_series(timeline_rows, base_key="relay_success", prefer_unique=True), where="post", label="Relay success (unique)", linewidth=1.7)
    ax.plot(times, _series_int(timeline_rows, "active_guided_missile_peak"), label="Active guided peak", linewidth=1.3)
    ax.set_title("Gate release and relay-guidance closure (deduplicated)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig02b_cooperative_chain_timeline_c.png")


def _plot_state_phase_rows(timeline_rows: List[Dict[str, object]], asset_dir: Path) -> None:
    times = _series_float(timeline_rows, "time_s", 0.0)

    fig, ax = plt.subplots(figsize=(14, 3.0))
    _plot_categorical(ax, times, [row.get("cap_state", "") for row in timeline_rows], "CAP state")
    _save(fig, asset_dir / "fig03_state_tactic_phase_timeline_a.png")

    fig, ax = plt.subplots(figsize=(14, 3.0))
    _plot_categorical(ax, times, [row.get("enemy_script_phase_name", "") for row in timeline_rows], "Enemy scene phase")
    _save(fig, asset_dir / "fig03_state_tactic_phase_timeline_b.png")

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.step(times, _series_int(timeline_rows, "enemy_script_awacs_available"), where="post", label="AWACS available", linewidth=1.7)
    ax2 = ax.twinx()
    ax2.plot(times, _series_float(timeline_rows, "enemy_script_reference_distance_km"), label="Reference distance (km)", linewidth=1.2, color="#C0504D")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Enemy information condition and trigger distance")
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig03_state_tactic_phase_timeline_c.png")

    fig, ax = plt.subplots(figsize=(14, 3.0))
    _plot_level_timeline(ax, times, [row.get("mission_threat_level", "") for row in timeline_rows], "Mission threat level", THREAT_LEVEL_ORDER)
    _save(fig, asset_dir / "fig03_state_tactic_phase_timeline_d.png")

    fig, ax = plt.subplots(figsize=(14, 3.0))
    _plot_categorical(ax, times, [row.get("left_tactic", "") for row in timeline_rows], "Left pair tactic")
    _save(fig, asset_dir / "fig03_state_tactic_phase_timeline_e.png")

    fig, ax = plt.subplots(figsize=(14, 3.0))
    _plot_categorical(ax, times, [row.get("right_tactic", "") for row in timeline_rows], "Right pair tactic")
    _save(fig, asset_dir / "fig03_state_tactic_phase_timeline_f.png")

    fig, ax = plt.subplots(figsize=(14, 3.0))
    _plot_categorical(ax, times, [_normalize_phase_label(row.get("left_phase", ""), default="") for row in timeline_rows], "Left pair phase", preferred_order=CONTROL_PHASE_ORDER)
    _save(fig, asset_dir / "fig03_state_tactic_phase_timeline_g.png")

    fig, ax = plt.subplots(figsize=(14, 3.0))
    _plot_categorical(ax, times, [_normalize_phase_label(row.get("right_phase", ""), default="") for row in timeline_rows], "Right pair phase", preferred_order=CONTROL_PHASE_ORDER)
    _save(fig, asset_dir / "fig03_state_tactic_phase_timeline_h.png")


def _plot_control_distance_nodes(timeline_rows: List[Dict[str, object]], summary: Dict[str, object], asset_dir: Path) -> None:
    times = _series_float(timeline_rows, "time_s", 0.0)
    node_rows = list(summary.get("control_distance_nodes", []) or [])
    route_nodes = {
        "left": [row for row in node_rows if str(row.get("route", "") or "") == "left"],
        "right": [row for row in node_rows if str(row.get("route", "") or "") == "right"],
    }
    for suffix, route_label, color in (
        ("a", "left", "#2F6B9A"),
        ("b", "right", "#D95F02"),
    ):
        fig, ax = plt.subplots(figsize=(14, 4.2))
        ax.plot(times, _series_float(timeline_rows, "nearest_enemy_to_friendly_km"), color="#4C78A8", linewidth=1.5)
        for row in route_nodes.get(route_label, []):
            time_s = _safe_float(row.get("first_time_s"), float("nan"))
            distance_km = _safe_float(row.get("distance_km"), float("nan"))
            if not np.isfinite(time_s) or not np.isfinite(distance_km):
                continue
            ax.scatter([time_s], [distance_km], color=color, s=32, zorder=4)
            ax.axvline(time_s, color=color, alpha=0.18, linewidth=0.9)
            ax.annotate(str(row.get("phase", "") or ""), xy=(time_s, distance_km), xytext=(4, 6), textcoords="offset points", fontsize=8, color=color)
        ax.set_title(f"{route_label.capitalize()} pair control-distance nodes")
        ax.set_ylabel("Nearest enemy-to-friendly (km)")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        _save(fig, asset_dir / f"fig03b_control_distance_timeline_{suffix}.png")


def _plot_intent_rows(timeline_rows: List[Dict[str, object]], asset_dir: Path) -> None:
    times = _series_float(timeline_rows, "time_s", 0.0)

    fig, ax = plt.subplots(figsize=(14, 3.0))
    _plot_level_timeline(ax, times, [row.get("intent_threat", "") for row in timeline_rows], "Highest intent threat", THREAT_LEVEL_ORDER)
    _save(fig, asset_dir / "fig04_intent_analysis_trace_a.png")

    fig, ax = plt.subplots(figsize=(14, 3.0))
    _plot_categorical(ax, times, [row.get("intent_class", "") for row in timeline_rows], "Highest classified intent", preferred_order=INTENT_CLASS_ORDER)
    _save(fig, asset_dir / "fig04_intent_analysis_trace_b.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.step(times, _series_int(timeline_rows, "intent_contact_count"), where="post", label="Intent contacts", linewidth=1.7)
    ax.step(times, _series_int(timeline_rows, "intent_classified_count"), where="post", label="Classified targets", linewidth=1.7)
    ax.step(times, _series_int(timeline_rows, "intent_ready_count"), where="post", label="Ready classified targets", linewidth=1.7)
    ax.step(times, _series_int(timeline_rows, "intent_high_count"), where="post", label="High-threat targets", linewidth=1.5)
    ax.step(times, _series_int(timeline_rows, "intent_critical_count"), where="post", label="Critical-threat targets", linewidth=1.5)
    ax.set_title("Intent-analysis coverage and high-threat target count")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig04_intent_analysis_trace_c.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.step(times, _series_int(timeline_rows, "intent_attack_count"), where="post", label="Attack", linewidth=1.7)
    ax.step(times, _series_int(timeline_rows, "intent_defense_count"), where="post", label="Defense", linewidth=1.7)
    ax.step(times, _series_int(timeline_rows, "intent_retreat_count"), where="post", label="Retreat", linewidth=1.7)
    ax.step(times, _series_int(timeline_rows, "intent_recon_count"), where="post", label="Reconnaissance", linewidth=1.3)
    ax.set_title("Intent-type target count")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig04_intent_analysis_trace_d.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_float(timeline_rows, "intent_confidence", 0.0), label="Highest confidence", linewidth=1.7)
    ax.plot(times, _series_int(timeline_rows, "intent_model_ready"), label="Highest ready", linewidth=1.2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Highest-threat confidence and readiness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig04_intent_analysis_trace_e.png")

    fig, ax = plt.subplots(figsize=(14, 3.4))
    _plot_categorical(ax, times, [row.get("enemy_left_intent_class", "") for row in timeline_rows], "Left enemy pair dominant intent", preferred_order=INTENT_CLASS_ORDER)
    ax2 = ax.twinx()
    ax2.step(times, _series_int(timeline_rows, "enemy_left_intent_ready_count"), where="post", label="Left ready count", linewidth=1.1, color="#C0504D")
    ax2.step(times, _series_int(timeline_rows, "enemy_right_intent_ready_count"), where="post", label="Right ready count", linewidth=1.1, color="#4F81BD")
    ax2.set_ylim(-0.1, 2.1)
    ax2.legend(fontsize=8, loc="upper right")
    _save(fig, asset_dir / "fig04_intent_analysis_trace_f.png")

    per_target_defs = [
        ("a", "enemy_left_intent_class", "Left enemy pair dominant intent"),
        ("b", "enemy_right_intent_class", "Right enemy pair dominant intent"),
        ("c", "B0100_intent_class", "B0100 intent"),
        ("d", "B0200_intent_class", "B0200 intent"),
        ("e", "B0300_intent_class", "B0300 intent"),
        ("f", "B0400_intent_class", "B0400 intent"),
    ]
    for suffix, key, title in per_target_defs:
        fig, ax = plt.subplots(figsize=(14, 3.0))
        _plot_categorical(ax, times, [row.get(key, "") for row in timeline_rows], title, preferred_order=INTENT_CLASS_ORDER)
        _save(fig, asset_dir / f"fig04b_intent_target_timeline_{suffix}.png")


def _plot_gate_and_engagement(timeline_rows: List[Dict[str, object]], asset_dir: Path) -> None:
    times = _series_float(timeline_rows, "time_s", 0.0)
    gate_pass = _guidance_metric_series(timeline_rows, base_key="gate_pass", prefer_unique=True)
    gate_block = _guidance_metric_series(timeline_rows, base_key="gate_block", prefer_unique=True)
    relay_attempt = _guidance_metric_series(timeline_rows, base_key="relay_attempt", prefer_unique=True)
    relay_success = _guidance_metric_series(timeline_rows, base_key="relay_success", prefer_unique=True)

    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.step(times, gate_pass, where="post", linewidth=2.3, label="gate_pass_unique", color="#2F5597")
    ax.step(times, gate_block, where="post", linewidth=2.3, label="gate_block_unique", color="#C0504D")
    ax.set_title("Prelaunch gate accumulation (unique)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, asset_dir / "fig05_gate_and_relay_a.png")

    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.step(times, relay_success, where="post", linewidth=2.3, label="relay_success_unique", color="#9BBB59")
    ax.step(times, relay_attempt, where="post", linewidth=2.3, label="relay_attempt_unique", color="#8064A2")
    ax.set_title("Relay guidance accumulation (unique)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, asset_dir / "fig05_gate_and_relay_b.png")

    gate_total = [p + b for p, b in zip(gate_pass, gate_block)]
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.step(times, gate_pass, where="post", label="Pass (unique)", linewidth=1.7)
    ax.step(times, gate_block, where="post", label="Block (unique)", linewidth=1.7)
    ax.plot(times, _running_ratio_series(gate_pass, gate_total), label="Pass rate (unique)", linewidth=1.2)
    ax.set_title("Prelaunch gate evolution (deduplicated)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig05_engagement_chain_a.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.step(times, relay_attempt, where="post", label="Attempt (unique)", linewidth=1.7)
    ax.step(times, relay_success, where="post", label="Success (unique)", linewidth=1.7)
    ax.plot(times, _running_ratio_series(relay_success, relay_attempt), label="Success rate (unique)", linewidth=1.2)
    ax.set_title("Relay guidance evolution (deduplicated)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig05_engagement_chain_b.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.step(times, _series_int(timeline_rows, "missile_launch_count"), where="post", label="Launches", linewidth=1.7)
    ax.step(times, _series_int(timeline_rows, "missile_outcome_count"), where="post", label="Outcomes", linewidth=1.7)
    ax.plot(times, _series_int(timeline_rows, "missile_state_guiding"), label="Guiding", linewidth=1.2)
    ax.plot(times, _series_int(timeline_rows, "missile_state_terminal"), label="Terminal", linewidth=1.2)
    ax.set_title("Missile-chain state counts")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig05_engagement_chain_c.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_int(timeline_rows, "active_guided_missile_peak"), label="Active guided peak", linewidth=1.6)
    ax.plot(times, _series_int(timeline_rows, "friendly_missiles_left"), label="Friendly missiles left", linewidth=1.6)
    ax.set_title("Guidance load and remaining inventory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig05_engagement_chain_d.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_int(timeline_rows, "enemy_kill_count"), label="Enemy kills", linewidth=1.7)
    ax.plot(times, _series_int(timeline_rows, "friendly_loss_count"), label="Friendly losses", linewidth=1.7)
    ax.set_title("Accumulated battle outcome")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig05_engagement_chain_e.png")


def _plot_survival_and_scan(timeline_rows: List[Dict[str, object]], asset_dir: Path) -> None:
    times = _series_float(timeline_rows, "time_s", 0.0)

    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.step(times, _series_int(timeline_rows, "friendly_alive"), where="post", label="Friendly alive", linewidth=1.8)
    ax.step(times, _series_int(timeline_rows, "enemy_alive"), where="post", label="Enemy alive", linewidth=1.8)
    ax.set_title("Alive aircraft count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig06_survival_distance_targeting_a.png")

    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.plot(times, _series_float(timeline_rows, "nearest_enemy_bullseye_km"), label="To bullseye", linewidth=1.6)
    ax.plot(times, _series_float(timeline_rows, "nearest_enemy_to_friendly_km"), label="To formation", linewidth=1.6)
    ax.set_title("Nearest-threat range")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig06_survival_distance_targeting_b.png")

    fig, ax = plt.subplots(figsize=(14, 3.0))
    _plot_categorical(ax, times, [row.get("left_target", "") for row in timeline_rows], "Left pair target")
    _save(fig, asset_dir / "fig06_survival_distance_targeting_c.png")

    fig, ax = plt.subplots(figsize=(14, 3.0))
    _plot_categorical(ax, times, [row.get("right_target", "") for row in timeline_rows], "Right pair target")
    _save(fig, asset_dir / "fig06_survival_distance_targeting_d.png")

    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.step(times, _series_int(timeline_rows, "friendly_alive"), where="post", linewidth=2.3, label="Friendly alive", color="#2F5597")
    ax.step(times, _series_int(timeline_rows, "enemy_alive"), where="post", linewidth=2.3, label="Enemy alive", color="#C0504D")
    ax.set_title("Alive aircraft count")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, asset_dir / "fig04_alive_and_missiles_a.png")

    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.step(times, _series_int(timeline_rows, "friendly_missiles_left"), where="post", linewidth=2.3, label="Friendly missiles left", color="#4F81BD")
    ax.step(times, _series_int(timeline_rows, "enemy_missiles_left"), where="post", linewidth=2.3, label="Enemy missiles left", color="#F79646")
    ax.set_title("Remaining missile inventory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, asset_dir / "fig04_alive_and_missiles_b.png")

    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(times, _series_int(timeline_rows, "scan_assignment_count"), label="Scan assignments", linewidth=1.8)
    ax.set_title("Scan assignment count")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, asset_dir / "fig08_scan_target_timeline_a.png")

    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(times, _series_float(timeline_rows, "scan_coverage_total_deg", 0.0), label="Coverage total (deg)", linewidth=1.8)
    ax.plot(times, _series_float(timeline_rows, "scan_overlap_deg", 0.0), label="Coverage overlap (deg)", linewidth=1.8)
    ax.set_title("Scan coverage and overlap")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, asset_dir / "fig08_scan_target_timeline_b.png")


def _plot_missile_end_reasons(missile_rows: List[Dict[str, object]], asset_dir: Path) -> None:
    friendly_counter = Counter()
    enemy_counter = Counter()
    for row in missile_rows:
        side = str(row.get("side", "") or "").strip()
        reason = str(row.get("reason", "") or "").strip()
        count = _safe_int(row.get("count"), 0)
        if not reason:
            continue
        if side == "friendly":
            friendly_counter[reason] += count
        elif side == "enemy":
            enemy_counter[reason] += count

    labels = sorted(set(friendly_counter) | set(enemy_counter))
    friendly_values = [friendly_counter.get(label, 0) for label in labels]
    enemy_values = [enemy_counter.get(label, 0) for label in labels]

    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.bar(labels, friendly_values, color="#4F81BD")
    ax.set_title("Friendly missile end reasons")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, asset_dir / "fig07_missile_end_reasons_a.png")

    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.bar(labels, enemy_values, color="#C0504D")
    ax.set_title("Enemy missile end reasons")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, asset_dir / "fig07_missile_end_reasons_b.png")


def _plot_flight_safety(timeline_rows: List[Dict[str, object]], asset_dir: Path) -> None:
    times = _series_float(timeline_rows, "time_s", 0.0)

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_float(timeline_rows, "friendly_min_altitude_m"), label="Friendly min alt", linewidth=1.7)
    ax.plot(times, _series_float(timeline_rows, "enemy_min_altitude_m"), label="Enemy min alt", linewidth=1.7)
    ax.plot(times, _series_float(timeline_rows, "friendly_mean_altitude_m"), label="Friendly mean alt", linewidth=1.1, alpha=0.85)
    ax.plot(times, _series_float(timeline_rows, "enemy_mean_altitude_m"), label="Enemy mean alt", linewidth=1.1, alpha=0.85)
    ax.set_title("Altitude envelope")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig09_flight_safety_envelope_a.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_float(timeline_rows, "friendly_min_speed_mps"), label="Friendly min speed", linewidth=1.7)
    ax.plot(times, _series_float(timeline_rows, "enemy_min_speed_mps"), label="Enemy min speed", linewidth=1.7)
    ax.plot(times, _series_float(timeline_rows, "friendly_mean_speed_mps"), label="Friendly mean speed", linewidth=1.1, alpha=0.85)
    ax.plot(times, _series_float(timeline_rows, "enemy_mean_speed_mps"), label="Enemy mean speed", linewidth=1.1, alpha=0.85)
    ax.set_title("Speed envelope")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig09_flight_safety_envelope_b.png")

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(times, _series_float(timeline_rows, "friendly_max_descent_rate_mps"), label="Friendly max descent", linewidth=1.7)
    ax.plot(times, _series_float(timeline_rows, "enemy_max_descent_rate_mps"), label="Enemy max descent", linewidth=1.7)
    ax.set_title("Maximum descent rate envelope")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, asset_dir / "fig09_flight_safety_envelope_c.png")


def _plot_batch_figures() -> None:
    batch_summary = _load_json(RESULT_BATCH_DIR / "batch_summary.json")
    summaries = list(batch_summary.get("summaries", []) or [])
    labels = [str(summary.get("scenario_id", "")) for summary in summaries]
    batch_dir = DOC_ASSET_DIR / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    effect_metrics = [
        ("comparison_effectiveness_a.png", "Enemy kills", "enemy_kill_count", 1.0, "#1B9E77", ""),
        ("comparison_effectiveness_b.png", "Friendly losses", "friendly_loss_count", 1.0, "#D95F02", ""),
        ("comparison_effectiveness_c.png", "Stable ready peak", "stable_ready_peak", 1.0, "#7570B3", ""),
        ("comparison_effectiveness_d.png", "Gate pass rate (unique)", "gate_pass_rate", 100.0, "#2F6B9A", "%"),
        ("comparison_effectiveness_e.png", "Relay success rate (unique)", "relay_success_rate", 100.0, "#66A61E", "%"),
        ("comparison_effectiveness_f.png", "Guided missile peak", "active_guided_missile_peak", 1.0, "#E7298A", ""),
    ]
    for filename, title, key, multiplier, color, suffix in effect_metrics:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        values = [_safe_float(summary.get(key), float("nan")) * multiplier for summary in summaries]
        _plot_batch_bars(ax, labels, values, title, suffix=suffix, color=color)
        _save(fig, batch_dir / filename)

    timing_metrics = [
        ("comparison_timing_a.png", "First radar track", "first_radar_track_time_s"),
        ("comparison_timing_b.png", "First stable ready", "first_stable_ready_time_s"),
        ("comparison_timing_c.png", "First gate pass", "first_gate_pass_time_s"),
        ("comparison_timing_d.png", "First missile launch", "first_missile_launch_time_s"),
        ("comparison_timing_e.png", "First enemy kill", "first_enemy_kill_time_s"),
        ("comparison_timing_f.png", "First high intent", "first_intent_high_time_s"),
    ]
    for filename, title, key in timing_metrics:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        values = [_safe_float(summary.get(key), float("nan")) for summary in summaries]
        _plot_batch_bars(ax, labels, values, title, suffix="s", color="#4C78A8")
        _save(fig, batch_dir / filename)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    low_truth = [_safe_float(summary.get("low_risk_enemy_time_s"), 0.0) for summary in summaries]
    medium_truth = [_safe_float(summary.get("medium_risk_enemy_time_s"), 0.0) for summary in summaries]
    high_truth = [_safe_float(summary.get("high_risk_breach_time_s"), 0.0) for summary in summaries]
    ax.bar(labels, low_truth, label="Low", color="#A6CEE3")
    ax.bar(labels, medium_truth, bottom=low_truth, label="Medium", color="#FDBF6F")
    ax.bar(labels, high_truth, bottom=[low + medium for low, medium in zip(low_truth, medium_truth)], label="High", color="#FB9A99")
    ax.set_title("Truth Zone Occupancy (s)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save(fig, batch_dir / "comparison_truth_zone_occupancy.png")


def _copy_scene_overview_images() -> None:
    for scenario_key, source_path in SCENE_OVERVIEW_SOURCES.items():
        if not source_path.exists():
            continue
        target_path = DOC_ASSET_DIR / scenario_key / "fig00_scene_overview.png"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def regenerate() -> None:
    _copy_scene_overview_images()
    for scenario_key in ("S1", "S2", "S3"):
        timeline_rows, summary, missile_rows, _kill_rows = _load_scenario_payload(scenario_key)
        asset_dir = DOC_ASSET_DIR / scenario_key
        _plot_overview_rows(timeline_rows, asset_dir)
        _plot_truth_zone_geometry_panels(timeline_rows, asset_dir)
        _split_scene_overview_image(SCENE_OVERVIEW_SOURCES[scenario_key], asset_dir)
        _plot_detection_pipeline(timeline_rows, asset_dir)
        _plot_coop_chain(timeline_rows, summary, asset_dir)
        _plot_state_phase_rows(timeline_rows, asset_dir)
        _plot_control_distance_nodes(timeline_rows, summary, asset_dir)
        if scenario_key == "S1":
            _plot_intent_rows(timeline_rows, asset_dir)
        _plot_gate_and_engagement(timeline_rows, asset_dir)
        _plot_survival_and_scan(timeline_rows, asset_dir)
        _plot_missile_end_reasons(missile_rows, asset_dir)
        _plot_flight_safety(timeline_rows, asset_dir)
    _plot_batch_figures()


if __name__ == "__main__":
    regenerate()
