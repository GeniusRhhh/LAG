from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DOC_ROOT = Path(r"D:\Pycharm\LAG\scripts\tacticalProject\cap\docs")
RESULT_ROOT = Path(r"D:\Pycharm\LAG\scripts\tacticalProject\cap_results\Chapter6_validation")
REPORT_ROOT = DOC_ROOT / "generated_compare_reports"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

BATCHES = [
    "COMPARE_ALL_20260521_225425",
    "COMPARE_ALL_20260521_225431",
    "COMPARE_ALL_20260521_225438",
    "COMPARE_ALL_20260521_225448",
]

SCENE_TITLE = {
    "S1": "场景一：低风险正面对进",
    "S2": "场景二：中风险持续压制",
    "S3": "场景三：高风险低空突防",
}

SCENE_PURPOSE = {
    "S1": "验证预警支撑完整、敌我几何较对称条件下，系统是否能够按照控制距离逐级推进，并把稳定跟踪、发射门释放、接力制导和规避回整组织成完整闭环。",
    "S2": "验证信息条件起伏和持续压制背景下，系统是否能够在探测模式切换、交战组织、规避打断与再次回整之间保持连续决策能力。",
    "S3": "验证高风险近距、低空突防背景下，系统是否能够更早形成 stable-ready、更早释放火力窗口，并用规避与回整逻辑压缩高风险区滞留时间。",
}

EXPECTED_ZONE = {
    "S1": "LOW",
    "S2": "MEDIUM",
    "S3": "HIGH",
}

ZONE_LABEL = {
    "LOW": "低风险区",
    "MEDIUM": "中风险区",
    "HIGH": "高风险区",
}

ZONE_COL = {
    "LOW": "truth_low_count",
    "MEDIUM": "truth_medium_count",
    "HIGH": "truth_high_count",
}

METRIC_LABEL = {
    "enemy_kill_count": "敌机击落数",
    "friendly_loss_count": "我方损失数",
    "radar_detection_coverage_ratio_post_activation": "雷达激活后平均探测覆盖率",
    "radar_full_coverage_continuity_post_activation": "雷达激活后全覆盖连续率",
    "first_stable_ready_time_s": "首次 stable-ready 时间",
    "first_gate_pass_time_s": "首次发射门通过时间",
    "gate_pass_rate": "发射门总体通过率（unique）",
    "gate_ready_pass_rate": "stable-ready 后发射门通过率（unique）",
    "relay_success_rate": "接力制导成功率（unique）",
    "high_risk_breach_time_s": "高风险区累计进入时长",
    "intent_truth_accuracy": "意图真值一致率",
}

PERCENT_0_TO_1_METRICS = {
    "radar_detection_coverage_ratio_post_activation",
    "radar_full_coverage_continuity_post_activation",
    "intent_truth_accuracy",
}

RATE_RATIO_METRICS = {
    "gate_pass_rate",
    "gate_ready_pass_rate",
    "relay_success_rate",
}

LOWER_IS_BETTER = {
    "first_stable_ready_time_s",
    "first_gate_pass_time_s",
    "high_risk_breach_time_s",
    "friendly_loss_count",
}

HIGHER_IS_BETTER = {
    "enemy_kill_count",
    "radar_detection_coverage_ratio_post_activation",
    "radar_full_coverage_continuity_post_activation",
    "gate_pass_rate",
    "gate_ready_pass_rate",
    "relay_success_rate",
    "intent_truth_accuracy",
}


@dataclass
class RunData:
    scene_id: str
    role: str
    output_dir: Path
    summary: Dict[str, float]
    milestones: Dict[str, float]
    timeline: pd.DataFrame
    decision: pd.DataFrame
    nodes: pd.DataFrame
    control_segments: pd.DataFrame
    summary_json: Dict[str, object]


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float):
        return value
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def _safe_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _json_load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metric_map(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    metric_col = "metric"
    if "metric_key" in df.columns:
        metric_col = "metric_key"
    return {str(row[metric_col]).strip(): _safe_float(row["value"]) for _, row in df.iterrows()}


def _read_milestones(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    return {str(row["key"]).strip(): _safe_float(row["time_s"]) for _, row in df.iterrows()}


def _load_run(scene_id: str, role: str, output_dir: Path) -> RunData:
    tables = output_dir / "tables"
    timeline = pd.read_csv(output_dir / "timeline.csv", low_memory=False)
    decision = pd.read_csv(tables / "decision_trace.csv", low_memory=False)
    nodes = pd.read_csv(tables / "control_distance_nodes.csv")
    control_segments = pd.read_csv(tables / "control_distance_segments.csv")
    summary = _read_metric_map(tables / "summary_metrics.csv")
    milestones = _read_milestones(tables / "milestone_times.csv")
    summary_json = _json_load(output_dir / "summary.json")
    return RunData(
        scene_id=scene_id,
        role=role,
        output_dir=output_dir,
        summary=summary,
        milestones=milestones,
        timeline=timeline,
        decision=decision,
        nodes=nodes,
        control_segments=control_segments,
        summary_json=summary_json,
    )


def _load_batch_scene_runs(batch_dir: Path) -> Dict[str, Dict[str, RunData]]:
    compare_df = pd.read_csv(batch_dir / "comparison_tables" / "proposed_vs_baseline.csv")
    result: Dict[str, Dict[str, RunData]] = {}
    for _, row in compare_df.iterrows():
        scene_id = str(row["scenario_id"]).strip()
        proposed_dir = Path(str(row["proposed_output_dir"]).strip())
        baseline_dir = Path(str(row["baseline_output_dir"]).strip())
        result[scene_id] = {
            "proposed": _load_run(scene_id, "proposed", proposed_dir),
            "baseline": _load_run(scene_id, "baseline", baseline_dir),
        }
    return result


def _format_metric(metric_key: str, value: float) -> str:
    if value is None or not math.isfinite(value):
        return "未触发"
    if metric_key in PERCENT_0_TO_1_METRICS:
        return f"{value * 100:.2f}%"
    if metric_key in RATE_RATIO_METRICS:
        return f"{value * 100:.2f}%"
    if metric_key.endswith("_time_s") or metric_key.endswith("_breach_time_s"):
        return f"{value:.1f} s"
    if metric_key in {"enemy_kill_count", "friendly_loss_count"}:
        return f"{int(round(value))}"
    return f"{value:.3f}"


def _metric_delta_text(metric_key: str, p_value: float, b_value: float) -> str:
    if not math.isfinite(p_value) or not math.isfinite(b_value):
        return "该指标缺少可比样本。"
    delta = p_value - b_value
    abs_delta = abs(delta)
    if metric_key in PERCENT_0_TO_1_METRICS:
        delta_text = f"{abs_delta * 100:.2f} 个百分点"
    elif metric_key in RATE_RATIO_METRICS:
        delta_text = f"{abs_delta * 100:.2f} 个百分点"
    elif metric_key.endswith("_time_s") or metric_key.endswith("_breach_time_s"):
        delta_text = f"{abs_delta:.1f} s"
    else:
        delta_text = f"{abs_delta:.2f}"

    if metric_key in LOWER_IS_BETTER:
        if p_value < b_value:
            return f"本文方法优于基线，优势幅度为 {delta_text}。"
        if p_value > b_value:
            return f"本文方法劣于基线，差距为 {delta_text}。"
        return "两种方法在该指标上持平。"
    if metric_key in HIGHER_IS_BETTER:
        if p_value > b_value:
            return f"本文方法优于基线，优势幅度为 {delta_text}。"
        if p_value < b_value:
            return f"本文方法劣于基线，差距为 {delta_text}。"
        return "两种方法在该指标上持平。"
    return "该指标仅作辅助展示。"


def _find_first(df: pd.DataFrame, predicate) -> Optional[pd.Series]:
    for _, row in df.iterrows():
        try:
            if predicate(row):
                return row
        except Exception:
            continue
    return None


def _find_state_row(run: RunData, state: str, after_time: Optional[float] = None) -> Optional[pd.Series]:
    prev = None
    for _, row in run.decision.iterrows():
        now = _safe_text(row.get("cap_state"))
        time_s = _safe_float(row.get("time_s"))
        if after_time is not None and (not math.isfinite(time_s) or time_s <= after_time):
            prev = now
            continue
        if now == state and prev != state:
            return row
        prev = now
    return None


def _find_detect_mode_change(run: RunData, mode: str) -> Optional[pd.Series]:
    prev = None
    for _, row in run.decision.iterrows():
        now = _safe_text(row.get("detect_mode"))
        if now == mode and prev != mode:
            return row
        prev = now
    return None


def _find_awacs_state_change(run: RunData, state: str) -> Optional[pd.Series]:
    prev = None
    for _, row in run.timeline.iterrows():
        now = _safe_text(row.get("enemy_script_awacs_state"))
        if now == state and prev != state:
            return row
        prev = now
    return None


def _find_first_non_unknown(run: RunData, col: str) -> Optional[pd.Series]:
    prev = None
    for _, row in run.decision.iterrows():
        now = _safe_text(row.get(col))
        if now and now != "UNKNOWN" and now != prev:
            return row
        prev = now
    return None


def _extract_pair_reason(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for part in text.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key] = value
    return result


def _extract_parameter_summary(text: str) -> str:
    if not text:
        return "参数层未记录。"
    stable = re.search(r"stable_required_s=([0-9.]+)", text)
    first_window = re.search(r"first_window=\[TR:([0-9.]+),LR:([0-9.]+)\]", text)
    second_window = re.search(r"second_window=\[TR':([0-9.]+),LR':([0-9.]+)\]", text)
    survival = re.search(r"survival=\[DOR:([0-9.]+),DR:([0-9.]+),MAR:([0-9.]+)\]", text)
    pieces: List[str] = []
    if stable:
        pieces.append(f"stable-ready 判稳窗长 {stable.group(1)} s")
    if first_window:
        pieces.append(f"首轮窗口 TR={first_window.group(1)} km、LR={first_window.group(2)} km")
    if second_window:
        pieces.append(f"二次窗口 TR'={second_window.group(1)} km、LR'={second_window.group(2)} km")
    if survival:
        pieces.append(
            f"生存约束 DOR={survival.group(1)} km、DR={survival.group(2)} km、MAR={survival.group(3)} km"
        )
    if not pieces:
        return text
    return "；".join(pieces) + "。"


def _format_tactic_name(name: str) -> str:
    return name.replace("_", "-") if name else "未记录"


def _ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _copy_single_figure(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_detection_compare(scene_id: str, proposed: RunData, baseline: RunData, out_path: Path) -> None:
    p = proposed.timeline.copy()
    b = baseline.timeline.copy()
    _ensure_numeric(p, ["time_s", "radar_track_count", "stable_ready_count"])
    _ensure_numeric(b, ["time_s", "radar_track_count", "stable_ready_count"])

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.step(p["time_s"], p["radar_track_count"], where="post", linewidth=2.0, color="#1f77b4", label="本文方法 雷达跟踪数")
    ax.step(p["time_s"], p["stable_ready_count"], where="post", linewidth=2.0, color="#d62728", label="本文方法 stable-ready 数")
    ax.step(
        b["time_s"],
        b["radar_track_count"],
        where="post",
        linewidth=1.6,
        linestyle="--",
        color="#1f77b4",
        label="基线 雷达跟踪数",
    )
    ax.step(
        b["time_s"],
        b["stable_ready_count"],
        where="post",
        linewidth=1.6,
        linestyle="--",
        color="#d62728",
        label="基线 stable-ready 数",
    )
    ax.set_title(f"{SCENE_TITLE[scene_id]}探测与 stable-ready 建立对比")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("目标数 / 个")
    ax.set_ylim(-0.1, 4.2)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=2)
    _save_figure(fig, out_path)


def _plot_control_distance(scene_id: str, proposed: RunData, out_path: Path) -> None:
    df = proposed.timeline.copy()
    _ensure_numeric(df, ["time_s", "nearest_enemy_to_friendly_km"])

    nodes = proposed.nodes.copy()
    _ensure_numeric(nodes, ["first_time_s", "distance_km"])

    first_evade = _find_state_row(proposed, "EVADE")
    first_return = None
    if first_evade is not None:
        first_return = _find_first(
            proposed.decision,
            lambda row: _safe_float(row.get("time_s")) > _safe_float(first_evade.get("time_s"))
            and _safe_text(row.get("cap_state")) in {"INTERCEPT", "ENGAGE"},
        )

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.plot(df["time_s"], df["nearest_enemy_to_friendly_km"], color="#1f77b4", linewidth=1.8, label="最近敌我距离")

    for route, color, marker in [("left", "#d62728", "o"), ("right", "#2ca02c", "s")]:
        sub = nodes[nodes["route"] == route]
        if sub.empty:
            continue
        ax.scatter(sub["first_time_s"], sub["distance_km"], color=color, s=42, marker=marker, label=f"{route} 路节点")
        for _, row in sub.iterrows():
            ax.annotate(
                _safe_text(row["phase"]),
                (_safe_float(row["first_time_s"]), _safe_float(row["distance_km"])),
                xytext=(4, 6 if route == "left" else -11),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

    key_marks = [
        ("stable-ready", proposed.milestones.get("first_stable_ready_time_s"), "#9467bd"),
        ("首发门通过", proposed.milestones.get("first_gate_pass_time_s"), "#ff7f0e"),
        ("首接力成功", proposed.milestones.get("first_relay_success_time_s"), "#8c564b"),
    ]
    if first_evade is not None:
        key_marks.append(("首规避", _safe_float(first_evade.get("time_s")), "#000000"))
    if first_return is not None:
        key_marks.append(("首回整", _safe_float(first_return.get("time_s")), "#7f7f7f"))

    ymax = float(df["nearest_enemy_to_friendly_km"].max()) if not df.empty else 260.0
    for label, time_s, color in key_marks:
        if time_s is None or not math.isfinite(_safe_float(time_s)):
            continue
        time_s = _safe_float(time_s)
        ax.axvline(time_s, color=color, linestyle="--", linewidth=1.1)
        ax.text(time_s, ymax * 0.98, label, rotation=90, va="top", ha="right", fontsize=8, color=color)

    ax.set_title(f"{SCENE_TITLE[scene_id]}动态控制距离推进与关键事件时间线（本文方法）")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("最近敌我距离 / km")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=3)
    _save_figure(fig, out_path)


def _plot_chain_compare(scene_id: str, proposed: RunData, baseline: RunData, out_path: Path) -> None:
    p = proposed.timeline.copy()
    b = baseline.timeline.copy()
    cols = ["time_s", "gate_ready_pass_count", "relay_success_count", "enemy_kill_count"]
    _ensure_numeric(p, cols)
    _ensure_numeric(b, cols)

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.step(p["time_s"], p["gate_ready_pass_count"], where="post", linewidth=2.0, color="#2ca02c", label="本文方法 发射门通过累计")
    ax.step(p["time_s"], p["relay_success_count"], where="post", linewidth=2.0, color="#1f77b4", label="本文方法 接力成功累计")
    ax.step(p["time_s"], p["enemy_kill_count"], where="post", linewidth=2.0, color="#d62728", label="本文方法 击落累计")
    ax.step(
        b["time_s"],
        b["gate_ready_pass_count"],
        where="post",
        linewidth=1.6,
        linestyle="--",
        color="#2ca02c",
        label="基线 发射门通过累计",
    )
    ax.step(
        b["time_s"],
        b["relay_success_count"],
        where="post",
        linewidth=1.6,
        linestyle="--",
        color="#1f77b4",
        label="基线 接力成功累计",
    )
    ax.step(
        b["time_s"],
        b["enemy_kill_count"],
        where="post",
        linewidth=1.6,
        linestyle="--",
        color="#d62728",
        label="基线 击落累计",
    )
    ax.set_title(f"{SCENE_TITLE[scene_id]}发射门、接力制导与击落结果对比")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("累计次数 / 次")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=2)
    _save_figure(fig, out_path)


def _plot_zone_compare(scene_id: str, proposed: RunData, baseline: RunData, out_path: Path) -> None:
    zone = EXPECTED_ZONE[scene_id]
    zone_col = ZONE_COL[zone]
    p = proposed.timeline.copy()
    b = baseline.timeline.copy()
    cols = ["time_s", zone_col, "truth_high_count"]
    _ensure_numeric(p, cols)
    _ensure_numeric(b, cols)

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.step(p["time_s"], p[zone_col], where="post", linewidth=2.0, color="#ff7f0e", label=f"本文方法 {ZONE_LABEL[zone]}目标数")
    ax.step(
        b["time_s"],
        b[zone_col],
        where="post",
        linewidth=1.6,
        linestyle="--",
        color="#ff7f0e",
        label=f"基线 {ZONE_LABEL[zone]}目标数",
    )
    if zone != "HIGH":
        ax.step(p["time_s"], p["truth_high_count"], where="post", linewidth=2.0, color="#d62728", label="本文方法 高风险目标数")
        ax.step(
            b["time_s"],
            b["truth_high_count"],
            where="post",
            linewidth=1.6,
            linestyle="--",
            color="#d62728",
            label="基线 高风险目标数",
        )
    ax.set_title(f"{SCENE_TITLE[scene_id]}责任区压力演化对比")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("敌机数量 / 架")
    ax.set_ylim(-0.1, 4.2)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=2)
    _save_figure(fig, out_path)


def _plot_batch_summary(batch_id: str, scenes: Dict[str, Dict[str, RunData]], out_path: Path) -> None:
    rows: List[Tuple[str, float]] = []
    labels: List[str] = []
    for scene_id in ["S1", "S2", "S3"]:
        p = scenes[scene_id]["proposed"]
        b = scenes[scene_id]["baseline"]
        p_value = _safe_float(p.summary.get("high_risk_breach_time_s"))
        b_value = _safe_float(b.summary.get("high_risk_breach_time_s"))
        labels.extend([f"{scene_id}-本文", f"{scene_id}-基线"])
        rows.extend(
            [
                (f"{scene_id}-本文", p_value if math.isfinite(p_value) else 0.0),
                (f"{scene_id}-基线", b_value if math.isfinite(b_value) else 0.0),
            ]
        )
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    values = [value for _, value in rows]
    colors = ["#d62728" if "本文" in label else "#7f7f7f" for label, _ in rows]
    ax.bar(labels, values, color=colors)
    for i, value in enumerate(values):
        ax.text(i, value + max(values + [1.0]) * 0.02, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{batch_id} 三场景高风险区累计进入时长对比")
    ax.set_ylabel("累计时长 / s")
    ax.grid(True, axis="y", alpha=0.28)
    _save_figure(fig, out_path)


def _build_figures(batch_id: str, scenes: Dict[str, Dict[str, RunData]], batch_asset_dir: Path) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    summary_path = batch_asset_dir / "fig00_batch_high_risk_compare.png"
    _plot_batch_summary(batch_id, scenes, summary_path)
    result["batch"] = {
        "high_risk_compare": str(summary_path.relative_to(DOC_ROOT)).replace("\\", "/"),
    }

    for scene_id, pair in scenes.items():
        proposed = pair["proposed"]
        baseline = pair["baseline"]
        scene_asset_dir = batch_asset_dir / scene_id
        scene_asset_dir.mkdir(parents=True, exist_ok=True)

        src_geometry = proposed.output_dir / "figures" / "fig01_risk_picture_geometry.png"
        dst_geometry = scene_asset_dir / "fig01_risk_picture_geometry.png"
        if src_geometry.exists():
            _copy_single_figure(src_geometry, dst_geometry)

        detection_path = scene_asset_dir / "fig02_detection_ready_compare.png"
        _plot_detection_compare(scene_id, proposed, baseline, detection_path)

        control_path = scene_asset_dir / "fig03_control_distance_timeline.png"
        _plot_control_distance(scene_id, proposed, control_path)

        chain_path = scene_asset_dir / "fig04_gate_relay_kill_compare.png"
        _plot_chain_compare(scene_id, proposed, baseline, chain_path)

        zone_path = scene_asset_dir / "fig05_zone_pressure_compare.png"
        _plot_zone_compare(scene_id, proposed, baseline, zone_path)

        result[scene_id] = {
            "geometry": str(dst_geometry.relative_to(DOC_ROOT)).replace("\\", "/"),
            "detection": str(detection_path.relative_to(DOC_ROOT)).replace("\\", "/"),
            "control": str(control_path.relative_to(DOC_ROOT)).replace("\\", "/"),
            "chain": str(chain_path.relative_to(DOC_ROOT)).replace("\\", "/"),
            "zone": str(zone_path.relative_to(DOC_ROOT)).replace("\\", "/"),
        }
    return result


def _scene_metric_rows(proposed: RunData, baseline: RunData) -> List[Tuple[str, str, str, str]]:
    keys = [
        "enemy_kill_count",
        "friendly_loss_count",
        "radar_detection_coverage_ratio_post_activation",
        "radar_full_coverage_continuity_post_activation",
        "first_stable_ready_time_s",
        "first_gate_pass_time_s",
        "gate_ready_pass_rate",
        "gate_pass_rate",
        "relay_success_rate",
        "high_risk_breach_time_s",
    ]
    rows: List[Tuple[str, str, str, str]] = []
    for key in keys:
        p_value = _safe_float(proposed.summary.get(key))
        b_value = _safe_float(baseline.summary.get(key))
        rows.append(
            (
                METRIC_LABEL[key],
                _format_metric(key, p_value),
                _format_metric(key, b_value),
                _metric_delta_text(key, p_value, b_value),
            )
        )
    return rows


def _render_markdown_table(headers: List[str], rows: List[Iterable[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _control_distance_table(run: RunData) -> str:
    df = run.nodes.copy()
    _ensure_numeric(df, ["first_time_s", "distance_km"])
    rows = []
    for _, row in df.iterrows():
        rows.append(
            (
                _safe_text(row["route"]),
                _safe_text(row["phase"]),
                f"{_safe_float(row['first_time_s']):.1f}",
                f"{_safe_float(row['distance_km']):.1f}",
                _safe_text(row["description"]),
            )
        )
    return _render_markdown_table(
        ["路径", "节点阶段", "首次进入时间 / s", "进入距离 / km", "说明"],
        rows,
    )


def _scene_event_snapshot(run: RunData) -> Dict[str, Optional[pd.Series]]:
    first_engage = _find_state_row(run, "ENGAGE")
    first_evade = _find_state_row(run, "EVADE")
    return_row = None
    if first_evade is not None:
        return_row = _find_first(
            run.decision,
            lambda row: _safe_float(row.get("time_s")) > _safe_float(first_evade.get("time_s"))
            and _safe_text(row.get("cap_state")) in {"INTERCEPT", "ENGAGE"},
        )
    return {
        "first_engage": first_engage,
        "first_evade": first_evade,
        "return_row": return_row,
        "first_left_tactic": _find_first_non_unknown(run, "left_tactic"),
        "first_right_tactic": _find_first_non_unknown(run, "right_tactic"),
        "first_sweep": _find_detect_mode_change(run, "SWEEP"),
        "first_awacs_denied": _find_awacs_state_change(run, "DENIED"),
    }


def _scene_key_event_table(scene_id: str, run: RunData) -> str:
    events = _scene_event_snapshot(run)
    milestones = run.milestones
    rows: List[Tuple[str, str, str, str]] = []

    engage = events["first_engage"]
    if engage is not None:
        rows.append(
            (
                "进入 ENGAGE",
                f"{_safe_float(engage['time_s']):.1f}",
                _safe_text(engage.get("cap_state_reason")),
                "系统确认预警图存在并开始把截获组织转入交战组织。",
            )
        )

    left_tactic = events["first_left_tactic"]
    right_tactic = events["first_right_tactic"]
    if left_tactic is not None and right_tactic is not None:
        rows.append(
            (
                "首次战术模板确定",
                f"{min(_safe_float(left_tactic['time_s']), _safe_float(right_tactic['time_s'])):.1f}",
                f"左路 {_safe_text(left_tactic.get('left_tactic'))} / 右路 {_safe_text(right_tactic.get('right_tactic'))}",
                "战术层开始给左右双机分配不同职责，而不是维持统一动作。",
            )
        )

    first_lr = run.nodes[run.nodes["phase"] == "LR_TR"]
    if not first_lr.empty:
        left_lr = first_lr[first_lr["route"] == "left"].head(1)
        right_lr = first_lr[first_lr["route"] == "right"].head(1)
        left_time = _safe_float(left_lr.iloc[0]["first_time_s"]) if not left_lr.empty else float("nan")
        right_time = _safe_float(right_lr.iloc[0]["first_time_s"]) if not right_lr.empty else float("nan")
        rows.append(
            (
                "进入 LR_TR 发射准备段",
                f"{min(v for v in [left_time, right_time] if math.isfinite(v)):.1f}",
                f"左路 {left_time:.1f}s / 右路 {right_time:.1f}s",
                "说明控制距离已经从中距压缩推进到首次发射准备段。",
            )
        )

    gate_time = milestones.get("first_gate_pass_time_s")
    if gate_time is not None and math.isfinite(_safe_float(gate_time)):
        rows.append(
            (
                "首次发射门通过",
                f"{_safe_float(gate_time):.1f}",
                f"stable-ready 后通过率 {_format_metric('gate_ready_pass_rate', _safe_float(run.summary.get('gate_ready_pass_rate')))}",
                "说明一旦进入 stable-ready，门禁逻辑并未继续无意义阻断。",
            )
        )

    relay_time = milestones.get("first_relay_success_time_s")
    if relay_time is not None and math.isfinite(_safe_float(relay_time)):
        rows.append(
            (
                "首次接力制导成功",
                f"{_safe_float(relay_time):.1f}",
                f"接力成功率 {_format_metric('relay_success_rate', _safe_float(run.summary.get('relay_success_rate')))}",
                "说明首轮放行后，中制导链条能够被另外一机接续，而不是发射后立即失链。",
            )
        )

    evade = events["first_evade"]
    if evade is not None:
        rows.append(
            (
                "首次规避",
                f"{_safe_float(evade['time_s']):.1f}",
                _safe_text(evade.get("cap_state_reason")),
                "导弹来袭触发最高优先级打断，系统强制从交战链切入规避链。",
            )
        )

    return_row = events["return_row"]
    if return_row is not None:
        rows.append(
            (
                "首次回整",
                f"{_safe_float(return_row['time_s']):.1f}",
                _safe_text(return_row.get("cap_state_reason")),
                "威胁短时下降后，系统重新回到截获或交战组织，而不是持续被动规避。",
            )
        )

    if scene_id == "S2":
        awacs_denied = events["first_awacs_denied"]
        if awacs_denied is not None:
            rows.append(
                (
                    "预警支撑受损",
                    f"{_safe_float(awacs_denied['time_s']):.1f}",
                    _safe_text(awacs_denied.get("enemy_script_awacs_state")),
                    "用于说明系统不是在理想信息条件下工作，而是在支撑受损后重新组织探测与压制。",
                )
            )

    return _render_markdown_table(["事件", "时间 / s", "直接证据", "分析含义"], rows)


def _scene_analysis_text(scene_id: str, proposed: RunData, baseline: RunData) -> str:
    events = _scene_event_snapshot(proposed)
    engage = events["first_engage"]
    left_tactic = events["first_left_tactic"]
    right_tactic = events["first_right_tactic"]
    first_evade = events["first_evade"]
    return_row = events["return_row"]

    engage_text = "未记录"
    tactic_text = "未记录"
    maneuver_text = "未记录"
    parameter_text = "参数层未记录。"
    if engage is not None:
        engage_text = _safe_text(engage.get("cap_state_reason"))
        tactic_text = _safe_text(engage.get("left_tactic_reason")) or _safe_text(engage.get("right_tactic_reason"))
        maneuver_text = _safe_text(engage.get("left_maneuver_reason")) or _safe_text(engage.get("right_maneuver_reason"))
        parameter_text = _extract_parameter_summary(_safe_text(engage.get("left_parameter_reason")) or _safe_text(engage.get("right_parameter_reason")))

    p_cov = _format_metric("radar_detection_coverage_ratio_post_activation", _safe_float(proposed.summary.get("radar_detection_coverage_ratio_post_activation")))
    b_cov = _format_metric("radar_detection_coverage_ratio_post_activation", _safe_float(baseline.summary.get("radar_detection_coverage_ratio_post_activation")))
    p_ready = _format_metric("first_stable_ready_time_s", _safe_float(proposed.summary.get("first_stable_ready_time_s")))
    b_ready = _format_metric("first_stable_ready_time_s", _safe_float(baseline.summary.get("first_stable_ready_time_s")))
    p_gate = _format_metric("gate_pass_rate", _safe_float(proposed.summary.get("gate_pass_rate")))
    p_gate_ready = _format_metric("gate_ready_pass_rate", _safe_float(proposed.summary.get("gate_ready_pass_rate")))
    p_relay = _format_metric("relay_success_rate", _safe_float(proposed.summary.get("relay_success_rate")))
    p_high = _format_metric("high_risk_breach_time_s", _safe_float(proposed.summary.get("high_risk_breach_time_s")))
    b_high = _format_metric("high_risk_breach_time_s", _safe_float(baseline.summary.get("high_risk_breach_time_s")))

    first_gate = _safe_float(proposed.milestones.get("first_gate_pass_time_s"))
    first_relay = _safe_float(proposed.milestones.get("first_relay_success_time_s"))
    first_kill = _safe_float(proposed.milestones.get("first_enemy_kill_time_s"))

    paragraphs: List[str] = []
    paragraphs.append(
        f"本文方法在 {SCENE_TITLE[scene_id]} 中并不是先固定模板后机械执行，而是先由阶段层判断是否进入交战组织，再由战术层给左右双机分派不同职责。首次进入 `ENGAGE` 的直接证据为：{engage_text}。在该时刻，左右路战术原因分别开始体现为 `{_safe_text(left_tactic.get('left_tactic_reason')) if left_tactic is not None else '未记录'}` 与 `{_safe_text(right_tactic.get('right_tactic_reason')) if right_tactic is not None else '未记录'}`，机动层则进一步给出 `{maneuver_text}` 这样的射手/支援组合解释。"
    )
    paragraphs.append(
        f"参数层日志表明，这一批结果已经不是旧版“按初始距离等比例压缩”的写法，而是围绕实时态势给出动态窗口。代表性参数记录为：{parameter_text} 这意味着控制距离节点是由当前收敛速度、首轮窗口和生存约束共同决定，而不是开局一次性写死。结合图中的控制距离推进曲线可见，系统先进入 `LR_TR` 准备段，再在 {first_gate:.1f} s 打开发射门，并于 {first_relay:.1f} s 完成首次接力制导成功；对应的总体发射门通过率为 {p_gate}，而 stable-ready 后通过率为 {p_gate_ready}。前者回答“系统整体有多保守”，后者回答“当目标真正达到 stable-ready 后，系统是否仍在错误阻断”。"
    )
    evade_text = _safe_text(first_evade.get("cap_state_reason")) if first_evade is not None else "未触发规避"
    return_text = _safe_text(return_row.get("cap_state_reason")) if return_row is not None else "未触发回整"
    paragraphs.append(
        f"从方法有效性看，本文方法在该场景下把雷达激活后平均探测覆盖率做到了 {p_cov}，高于或低于基线的情况已在表中给出；更关键的是，它把首次 stable-ready 建立时间压缩到 {p_ready}，而基线为 {b_ready}。随后系统在导弹来袭时依据 `{evade_text}` 进入规避链，并在威胁下降后依据 `{return_text}` 回到截获/交战组织。最终该场景的首次击落发生在 {first_kill:.1f} s，接力制导成功率为 {p_relay}，高风险区累计进入时长为 {p_high}，基线为 {b_high}。因此，这一场景真正证明的不是“过程看起来跑通了”，而是“阶段决策、战术模板、机动分工和动态控制距离是否把方法优势转化为更早的交战准备、更有解释性的规避回整，以及更好的责任区控制结果”。"
    )
    return "\n\n".join(paragraphs)


def _scene_conclusion(scene_id: str, proposed: RunData, baseline: RunData) -> str:
    p_kill = _safe_float(proposed.summary.get("enemy_kill_count"))
    b_kill = _safe_float(baseline.summary.get("enemy_kill_count"))
    p_loss = _safe_float(proposed.summary.get("friendly_loss_count"))
    b_loss = _safe_float(baseline.summary.get("friendly_loss_count"))
    p_high = _safe_float(proposed.summary.get("high_risk_breach_time_s"))
    b_high = _safe_float(baseline.summary.get("high_risk_breach_time_s"))
    p_ready = _safe_float(proposed.summary.get("first_stable_ready_time_s"))
    b_ready = _safe_float(baseline.summary.get("first_stable_ready_time_s"))

    improved_front = math.isfinite(p_ready) and math.isfinite(b_ready) and p_ready < b_ready
    improved_result = (
        (math.isfinite(p_kill) and math.isfinite(b_kill) and p_kill >= b_kill)
        and (math.isfinite(p_loss) and math.isfinite(b_loss) and p_loss <= b_loss)
    )
    improved_zone = math.isfinite(p_high) and math.isfinite(b_high) and p_high <= b_high

    if improved_front and improved_result and improved_zone:
        return "本场景中，本文方法同时取得了更早的交战准备、更好的责任区压制和不弱于基线的最终战果，可以认定该场景下的方法有效性成立。"
    if improved_front and (not improved_result or not improved_zone):
        return "本场景中，本文方法已经证明前端决策链和动态控制距离是有效的，但这种前端优势尚未完全稳定转化为更优的最终交换比或更短的高风险区滞留时间，说明后段火力释放和残局收口仍需继续优化。"
    if improved_result and not improved_front:
        return "本场景中，本文方法最终战果不差，但它并未在前端感知和交战准备时效上稳定领先，因此更适合表述为“结果可接受”，而不是“方法链条全面优于基线”。"
    return "本场景中，本文方法未能在关键结果上全面优于基线，因此报告中应如实表述其有效边界：方法链可以工作，但在该样本下并未形成全面优势。"


def _batch_overview_table(scenes: Dict[str, Dict[str, RunData]]) -> str:
    rows = []
    for scene_id in ["S1", "S2", "S3"]:
        proposed = scenes[scene_id]["proposed"]
        baseline = scenes[scene_id]["baseline"]
        rows.append(
            (
                SCENE_TITLE[scene_id],
                f"{_format_metric('enemy_kill_count', _safe_float(proposed.summary.get('enemy_kill_count')))} / {_format_metric('friendly_loss_count', _safe_float(proposed.summary.get('friendly_loss_count')))}",
                f"{_format_metric('enemy_kill_count', _safe_float(baseline.summary.get('enemy_kill_count')))} / {_format_metric('friendly_loss_count', _safe_float(baseline.summary.get('friendly_loss_count')))}",
                _format_metric(
                    "first_stable_ready_time_s",
                    _safe_float(proposed.summary.get("first_stable_ready_time_s")),
                ),
                _format_metric(
                    "first_stable_ready_time_s",
                    _safe_float(baseline.summary.get("first_stable_ready_time_s")),
                ),
                _format_metric(
                    "high_risk_breach_time_s",
                    _safe_float(proposed.summary.get("high_risk_breach_time_s")),
                ),
                _format_metric(
                    "high_risk_breach_time_s",
                    _safe_float(baseline.summary.get("high_risk_breach_time_s")),
                ),
            )
        )
    return _render_markdown_table(
        [
            "场景",
            "本文方法 击落/损失",
            "基线 击落/损失",
            "本文方法 首次 stable-ready",
            "基线 首次 stable-ready",
            "本文方法 高风险区时长",
            "基线 高风险区时长",
        ],
        rows,
    )


def _build_batch_doc(batch_id: str, scenes: Dict[str, Dict[str, RunData]], figures: Dict[str, Dict[str, str]]) -> str:
    lines: List[str] = []
    lines.append(f"# 第六章仿真验证文档（{batch_id}）")
    lines.append("")
    lines.append("## 6.3 仿真验证")
    lines.append("")
    lines.append(
        "本节面向 `unique` 口径下的三场景 CAP 对比结果，对当前批次的“本文方法”与“基线方法”进行正式仿真验证。文档不再堆砌过程性图组，而是围绕老师要求的三条证据链展开：第一，系统为什么在某一时刻进入 `ENGAGE`、`EVADE`、回整或再次压缩；第二，控制距离节点为什么推进到当前位置，战术模板为什么在此时切换；第三，这些决策是否真正改善了感知、火控、责任区控制和最终战果。"
    )
    lines.append("")
    lines.append("### 6.3.1 统计口径与证据来源")
    lines.append("")
    lines.append("本报告统一采用如下统计口径：")
    lines.append("")
    lines.append("- `发射门总体通过率（unique）= gate_pass_count / gate_total_request_count`。该指标描述系统整体上有多保守。")
    lines.append("- `stable-ready 后发射门通过率（unique）= gate_ready_pass_count / gate_ready_total_request_count`。该指标描述目标真正进入 stable-ready 后，门禁是否仍然错误阻断。")
    lines.append("- `接力制导成功率（unique）= relay_success_count / relay_attempt_count`。该指标用于评价中制导链是否能被持续接续。")
    lines.append("- 所有控制距离节点、阶段切换、模板切换与规避回整解释，均直接来自 `decision_trace.csv`、`timeline.csv`、`control_distance_nodes.csv` 和 `milestone_times.csv`。")
    lines.append("")
    lines.append("表6-1 三场景总览")
    lines.append("")
    lines.append(_batch_overview_table(scenes))
    lines.append("")
    lines.append(f"图6-1 {batch_id} 三场景高风险区累计进入时长对比")
    lines.append(f"![图6-1 {batch_id} 三场景高风险区累计进入时长对比](./{figures['batch']['high_risk_compare']})")
    lines.append("")
    lines.append(
        "图6-1只保留一个最直接的责任区结果指标：高风险区累计进入时长。老师要求验证的核心不是图多，而是是否能证明方法有效，因此本批次的总览部分只保留对结论最敏感的场景级结果；其余细节全部放到场景内逐项解释。"
    )
    lines.append("")

    scene_section_no = 2
    table_no = 2
    fig_no = 2
    for scene_id in ["S1", "S2", "S3"]:
        proposed = scenes[scene_id]["proposed"]
        baseline = scenes[scene_id]["baseline"]
        lines.append(f"### 6.3.{scene_section_no} {SCENE_TITLE[scene_id]}")
        lines.append("")
        lines.append("#### （1）场景目标")
        lines.append("")
        lines.append(SCENE_PURPOSE[scene_id])
        lines.append("")

        lines.append(f"表6-{table_no} {SCENE_TITLE[scene_id]}核心指标对比")
        lines.append("")
        lines.append(_render_markdown_table(["指标", "本文方法", "基线", "对比结论"], _scene_metric_rows(proposed, baseline)))
        lines.append("")
        table_no += 1

        lines.append(f"图6-{fig_no} {SCENE_TITLE[scene_id]}责任区风险几何图（本文方法）")
        lines.append(f"![图6-{fig_no} {SCENE_TITLE[scene_id]}责任区风险几何图（本文方法）](./{figures[scene_id]['geometry']})")
        lines.append("")
        lines.append(
            "该图用于交代场景几何与风险区边界，不承担算法有效性结论；真正的算法判断仍以控制距离、发射门、接力制导和责任区压制结果为准。"
        )
        lines.append("")
        fig_no += 1

        lines.append(f"图6-{fig_no} {SCENE_TITLE[scene_id]}探测与 stable-ready 建立对比")
        lines.append(f"![图6-{fig_no} {SCENE_TITLE[scene_id]}探测与 stable-ready 建立对比](./{figures[scene_id]['detection']})")
        lines.append("")
        lines.append(
            "该图直接回答“本文方法是否更早建立可交战窗口”。如果 stable-ready 建立更早，而最终结果没有同步改善，则说明问题不在探测链，而在后段放行与残局收口。"
        )
        lines.append("")
        fig_no += 1

        lines.append(f"图6-{fig_no} {SCENE_TITLE[scene_id]}动态控制距离推进与关键事件时间线（本文方法）")
        lines.append(f"![图6-{fig_no} {SCENE_TITLE[scene_id]}动态控制距离推进与关键事件时间线（本文方法）](./{figures[scene_id]['control']})")
        lines.append("")
        lines.append(
            "该图是本批次最关键的解释图之一。它不是按初始距离做等比例压缩，而是把左右路控制距离节点、stable-ready、首次发射门通过、首次接力成功、首次规避和首次回整放到同一时间轴上，从而回答“为什么在这一时刻推进、发射、规避或回整”。"
        )
        lines.append("")
        fig_no += 1

        lines.append(f"图6-{fig_no} {SCENE_TITLE[scene_id]}发射门、接力制导与击落结果对比")
        lines.append(f"![图6-{fig_no} {SCENE_TITLE[scene_id]}发射门、接力制导与击落结果对比](./{figures[scene_id]['chain']})")
        lines.append("")
        lines.append(
            "该图用于检验“放行链是否闭合”。老师质疑的重点之一是：稳定跟踪以后到底有没有合理地释放发射机会，并把导弹链继续接下去。这里用累计曲线直接展示门禁、接力和击落的先后顺序。"
        )
        lines.append("")
        fig_no += 1

        lines.append(f"图6-{fig_no} {SCENE_TITLE[scene_id]}责任区压力演化对比")
        lines.append(f"![图6-{fig_no} {SCENE_TITLE[scene_id]}责任区压力演化对比](./{figures[scene_id]['zone']})")
        lines.append("")
        lines.append(
            f"该图围绕 {ZONE_LABEL[EXPECTED_ZONE[scene_id]]} 展开，并同时展示高风险区压力变化。它直接用于回答“用了本文方法以后，敌机是否更难长期压在我方关键风险区内”。"
        )
        lines.append("")
        fig_no += 1

        lines.append(f"表6-{table_no} {SCENE_TITLE[scene_id]}控制距离节点（本文方法）")
        lines.append("")
        lines.append(_control_distance_table(proposed))
        lines.append("")
        table_no += 1

        lines.append(f"表6-{table_no} {SCENE_TITLE[scene_id]}关键事件解释")
        lines.append("")
        lines.append(_scene_key_event_table(scene_id, proposed))
        lines.append("")
        table_no += 1

        lines.append("#### （2）针对性分析")
        lines.append("")
        lines.append(_scene_analysis_text(scene_id, proposed, baseline))
        lines.append("")
        lines.append("#### （3）本场景结论")
        lines.append("")
        lines.append(_scene_conclusion(scene_id, proposed, baseline))
        lines.append("")
        scene_section_no += 1

    lines.append("### 6.3.5 综合结论与证据边界")
    lines.append("")
    lines.append(
        "综合本批次三场景结果，可以得出两个层面的结论。其一，当前系统已经能够把阶段决策、战术模板、机动分工和动态控制距离组织成完整链条，报告中已经可以清楚解释“何时进入交战、何时触发规避、何时回整、何时首次放行、为何推进到当前控制距离节点”。其二，方法是否全面优于基线，仍然必须逐场景、逐指标判断：有的样本证明了前端链条有效，但并未把优势完全转化为最终交换比；有的样本则同时改善了责任区控制和战果。"
    )
    lines.append("")
    lines.append(
        "需要特别说明的是，本批次与早先版本不同，已经能够从 `left_parameter_reason` 和 `right_parameter_reason` 中读出动态窗口、判稳时长和生存约束，因此控制距离不再是静态表值，也不是简单的等比例压缩。报告引用这些参数日志的目的，不是证明某个单独参数“绝对最优”，而是证明参数层确实在根据实时态势计算窗口，并驱动节点推进。若后续答辩还要继续回答“为什么这里一定是 10 秒而不是 20 秒”这类更强的最优性问题，还需要在此基础上补充候选参数集合、比较结果和最终选值理由。"
    )
    lines.append("")
    lines.append("#### 结果追溯路径")
    lines.append("")
    lines.append("- 批次目录：" + str((RESULT_ROOT / batch_id)).replace("\\", "/"))
    lines.append("- 本文方法与基线的具体场景目录、图表和日志，均可由本报告中的相对路径回溯。")
    lines.append("")
    return "\n".join(lines)


def build_reports() -> List[Path]:
    created_docs: List[Path] = []
    for batch_id in BATCHES:
        batch_dir = RESULT_ROOT / batch_id
        scenes = _load_batch_scene_runs(batch_dir)
        batch_asset_dir = REPORT_ROOT / batch_id / "figures"
        figures = _build_figures(batch_id, scenes, batch_asset_dir)
        doc_text = _build_batch_doc(batch_id, scenes, figures)
        doc_path = DOC_ROOT / f"第六章仿真验证文档_{batch_id}.md"
        doc_path.write_text(doc_text, encoding="utf-8")
        created_docs.append(doc_path)
    return created_docs


def main() -> int:
    docs = build_reports()
    for path in docs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
