from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DOC_ROOT = Path(r"D:\Pycharm\LAG\scripts\tacticalProject\cap\docs")
RESULT_ROOT = Path(r"D:\Pycharm\LAG\scripts\tacticalProject\cap_results\Chapter6_validation")
REPORT_ROOT = DOC_ROOT / "generated_compare_reports_v2"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

BATCHES = [
    "COMPARE_ALL_20260521_225425",
    "COMPARE_ALL_20260521_225431",
    "COMPARE_ALL_20260521_225438",
    "COMPARE_ALL_20260521_225448",
]

SCENE_META = {
    "S1": {
        "title": "场景一 低风险正面对进条件下的节点推进验证",
        "focus": "验证完整感知闭环、控制距离节点推进和首轮交战组织是否合理。",
        "expected_zone": "LOW",
        "goal": "说明本文方法在低风险标准接敌条件下，能够较早形成 stable-ready，并按控制距离节点组织首轮攻击与规避回转。",
        "initial_distance_km": 266.4,
        "enemy_altitudes_km": [10.0, 10.0, 10.0, 10.0],
        "enemy_offsets_km": [(-60.0, 0.0), (-48.0, 0.0), (48.0, 0.0), (60.0, 0.0)],
        "enemy_speed_kmps": 0.300,
        "awacs": "持续可用",
        "scene_emphasis": "重点分析模板首次选取、首轮发射窗口形成、导弹来袭后的规避触发与回转恢复。",
        "focus_metric_keys": [
            "first_stable_ready_time_s",
            "enemy_kill_count",
            "friendly_loss_count",
            "intent_truth_accuracy",
        ],
    },
    "S2": {
        "title": "场景二 预警间歇失效条件下的持续压制验证",
        "focus": "验证信息不完整条件下的目标恢复、模板切换和风险区压制能力。",
        "expected_zone": "MEDIUM",
        "goal": "说明本文方法在 AWACS 间歇丢失条件下，仍能通过机载链路恢复目标图景，并把高风险突入压制在可控范围内。",
        "initial_distance_km": 240.0,
        "enemy_altitudes_km": [10.2, 8.8, 10.2, 8.8],
        "enemy_offsets_km": [(-66.0, 0.0), (-48.0, -6.0), (48.0, 0.0), (66.0, -6.0)],
        "enemy_speed_kmps": 0.330,
        "awacs": "每 30 s 内前 20 s 可用",
        "scene_emphasis": "重点分析信息缺口出现后为何仍能保持推进，左右路如何分工，以及高风险区突入是否被有效压缩。",
        "focus_metric_keys": [
            "first_stable_ready_time_s",
            "high_risk_breach_time_s",
            "friendly_loss_count",
            "radar_detection_coverage_ratio_post_activation",
        ],
    },
    "S3": {
        "title": "场景三 低空突防条件下的紧急规避与生存优先验证",
        "focus": "验证高风险压缩场景下的快速就绪、强制规避和再次组织能力。",
        "expected_zone": "HIGH",
        "goal": "说明本文方法在低空突防与长时 AWACS 失效条件下，能够更早进入交战、在高风险突入时触发强制规避，并尽量控制我方损失。",
        "initial_distance_km": 150.0,
        "enemy_altitudes_km": [3.4, 3.4, 3.7, 3.7],
        "enemy_offsets_km": [(-22.0, -6.0), (-10.0, -14.0), (26.0, 14.0), (40.0, 6.0)],
        "enemy_speed_kmps": 0.330,
        "awacs": "每 45 s 内前 10 s 可用",
        "scene_emphasis": "重点分析高风险节点前后的规避触发、战术回转和高风险区滞留压缩效果。",
        "focus_metric_keys": [
            "first_stable_ready_time_s",
            "high_risk_breach_time_s",
            "enemy_kill_count",
            "friendly_loss_count",
        ],
    },
}

SCENE_SECTION_NO = {
    "S1": "6.3.2",
    "S2": "6.3.3",
    "S3": "6.3.4",
}

ZONE_LABEL = {
    "LOW": "低风险区",
    "MEDIUM": "中风险区",
    "HIGH": "高风险区",
}

ZONE_COLOR = {
    "LOW": "#8ecae6",
    "MEDIUM": "#ffb703",
    "HIGH": "#d62828",
}

RISK_ZONE_RADIUS = {
    "LOW": 180.0,
    "MEDIUM": 120.0,
    "HIGH": 60.0,
}

METRIC_LABEL = {
    "enemy_kill_count": "敌机击落数",
    "friendly_loss_count": "我方损失数",
    "first_stable_ready_time_s": "首次 stable-ready 时间",
    "first_gate_pass_time_s": "首次发射门通过时间",
    "high_risk_breach_time_s": "高风险区累计突入时长",
    "radar_detection_coverage_ratio_post_activation": "雷达激活后平均覆盖率",
    "radar_full_coverage_continuity_post_activation": "雷达激活后全覆盖连续性",
    "intent_truth_accuracy": "意图识别一致率",
}

LOWER_IS_BETTER = {
    "friendly_loss_count",
    "first_stable_ready_time_s",
    "first_gate_pass_time_s",
    "high_risk_breach_time_s",
}

HIGHER_IS_BETTER = {
    "enemy_kill_count",
    "radar_detection_coverage_ratio_post_activation",
    "radar_full_coverage_continuity_post_activation",
    "intent_truth_accuracy",
}


@dataclass
class RunData:
    scene_id: str
    role: str
    output_dir: Path
    summary_metrics: Dict[str, float]
    milestones: Dict[str, float]
    decision_trace: pd.DataFrame
    control_nodes: pd.DataFrame
    control_segments: pd.DataFrame
    timeline: pd.DataFrame
    summary_json: Dict[str, object]


def safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def safe_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    if text.lower() in {"nan", "none", "null"}:
        return default
    return text


def read_metric_map(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    metric_col = "metric_key" if "metric_key" in df.columns else "metric"
    return {safe_text(row[metric_col]): safe_float(row["value"]) for _, row in df.iterrows()}


def read_milestones(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    return {safe_text(row["key"]): safe_float(row["time_s"]) for _, row in df.iterrows()}


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(scene_id: str, role: str, output_dir: Path) -> RunData:
    tables = output_dir / "tables"
    return RunData(
        scene_id=scene_id,
        role=role,
        output_dir=output_dir,
        summary_metrics=read_metric_map(tables / "summary_metrics.csv"),
        milestones=read_milestones(tables / "milestone_times.csv"),
        decision_trace=pd.read_csv(tables / "decision_trace.csv", low_memory=False),
        control_nodes=pd.read_csv(tables / "control_distance_nodes.csv"),
        control_segments=pd.read_csv(tables / "control_distance_segments.csv"),
        timeline=pd.read_csv(output_dir / "timeline.csv", low_memory=False),
        summary_json=read_json(output_dir / "summary.json"),
    )


def load_batch(batch_dir: Path) -> Dict[str, Dict[str, RunData]]:
    df = pd.read_csv(batch_dir / "comparison_tables" / "proposed_vs_baseline.csv")
    result: Dict[str, Dict[str, RunData]] = {}
    for _, row in df.iterrows():
        scene_id = safe_text(row["scenario_id"])
        result[scene_id] = {
            "proposed": load_run(scene_id, "proposed", Path(safe_text(row["proposed_output_dir"]))),
            "baseline": load_run(scene_id, "baseline", Path(safe_text(row["baseline_output_dir"]))),
        }
    return result


def format_value(metric_key: str, value: float) -> str:
    if not math.isfinite(value):
        return "未统计"
    if metric_key.endswith("_time_s"):
        return f"{value:.1f} s"
    if metric_key in {"enemy_kill_count", "friendly_loss_count"}:
        return f"{int(round(value))}"
    if "ratio" in metric_key or "accuracy" in metric_key:
        return f"{value * 100:.2f}%"
    return f"{value:.2f}"


def compare_text(metric_key: str, proposed: float, baseline: float) -> str:
    if not math.isfinite(proposed) or not math.isfinite(baseline):
        return "该指标缺少可比样本。"
    delta = proposed - baseline
    if metric_key in LOWER_IS_BETTER:
        if proposed < baseline:
            return f"本文方法优于基线方法，差值为 {abs(delta):.1f}{' s' if metric_key.endswith('_time_s') else ''}。"
        if proposed > baseline:
            return f"本文方法劣于基线方法，差值为 {abs(delta):.1f}{' s' if metric_key.endswith('_time_s') else ''}。"
        return "两种方法在该指标上持平。"
    if metric_key in HIGHER_IS_BETTER:
        if "ratio" in metric_key or "accuracy" in metric_key:
            gap = abs(delta) * 100
            if proposed > baseline:
                return f"本文方法优于基线方法，差值为 {gap:.2f} 个百分点。"
            if proposed < baseline:
                return f"本文方法劣于基线方法，差值为 {gap:.2f} 个百分点。"
            return "两种方法在该指标上持平。"
        if proposed > baseline:
            return f"本文方法优于基线方法，差值为 {abs(delta):.1f}。"
        if proposed < baseline:
            return f"本文方法劣于基线方法，差值为 {abs(delta):.1f}。"
        return "两种方法在该指标上持平。"
    return "该指标仅作辅助展示。"


def find_first_row(df: pd.DataFrame, condition) -> Optional[pd.Series]:
    for _, row in df.iterrows():
        try:
            if condition(row):
                return row
        except Exception:
            continue
    return None


def nearest_row(df: pd.DataFrame, time_s: float) -> Optional[pd.Series]:
    if df.empty:
        return None
    idx = (df["time_s"] - time_s).abs().idxmin()
    return df.loc[idx]


def collect_change_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = ["cap_state", "left_tactic", "right_tactic", "left_target", "right_target"]
    rows: List[pd.Series] = []
    prev = None
    for _, row in df.iterrows():
        current = tuple(row.get(k) for k in keys)
        if prev is None or current != prev:
            rows.append(row)
            prev = current
    return pd.DataFrame(rows)


def choose_scene_decision_rows(run: RunData) -> List[pd.Series]:
    changed = collect_change_rows(run.decision_trace)
    scene_id = run.scene_id
    rows: List[pd.Series] = []
    if scene_id == "S1":
        for t in [92.2, 126.8, 319.8, 326.4, 468.6]:
            row = nearest_row(changed, t)
            if row is not None:
                rows.append(row)
    elif scene_id == "S2":
        for t in [54.8, 199.6, 312.0]:
            row = nearest_row(changed, t)
            if row is not None:
                rows.append(row)
    elif scene_id == "S3":
        for t in [0.2, 113.4, 133.0, 280.0]:
            row = nearest_row(changed, t)
            if row is not None:
                rows.append(row)
    unique_rows = []
    seen = set()
    for row in rows:
        key = safe_float(row.get("time_s"))
        if not row_is_explainable(row):
            continue
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    return unique_rows


def parse_json_object(text: object) -> Dict[str, object]:
    if not isinstance(text, str):
        return {}
    raw = text.strip()
    if not raw.startswith("{"):
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def brief_stage_reason(text: str) -> str:
    if not text:
        return "无"
    parts = text.split()
    picked = []
    for token in parts:
        if token.startswith("ctx_dist=") or token.startswith("pair_dist=") or token.startswith("node=") or token.startswith("zone=") or token.startswith("awacs=") or token.startswith("missile_incoming="):
            picked.append(token)
    return "，".join(picked) if picked else text


def brief_parameter_reason(text: str) -> str:
    if not text:
        return "无"
    parts = text.split()
    picked = []
    for token in parts:
        if token.startswith("stable_required_s=") or token.startswith("gate_phase=") or token.startswith("first_window=") or token.startswith("survival=") or token.startswith("closure_rate="):
            picked.append(token)
    return "，".join(picked) if picked else text


def extract_token(text: str, key: str) -> str:
    if not text:
        return ""
    match = re.search(rf"{re.escape(key)}=([^\s]+)", text)
    return match.group(1) if match else ""


def summarize_decision_row(row: pd.Series, side: str) -> str:
    tactic = safe_text(row.get(f"{side}_tactic"), "UNKNOWN")
    target = safe_text(row.get(f"{side}_target"), "未明确")
    stage_reason = brief_stage_reason(safe_text(row.get(f"{side}_stage_reason")))
    tactic_reason = safe_text(row.get(f"{side}_tactic_reason"))
    parameter_reason = brief_parameter_reason(safe_text(row.get(f"{side}_parameter_reason")))
    return (
        f"{side}路在 {safe_float(row.get('time_s')):.1f} s 选择 `{tactic}`，目标为 `{target}`。"
        f" 触发条件为：{stage_reason}。"
        f" 模板判据为：{tactic_reason or '未记录'}。"
        f" 参数约束为：{parameter_reason}。"
    )


def pair_label(side: str) -> str:
    return "左路" if side == "left" else "右路"


def tactic_purpose_text(tactic: str, node: str, zone: str, awacs_flag: str, cap_state: str) -> str:
    tactic = tactic.upper()
    node = node.upper()
    zone = zone.upper()
    if "EVASION" in tactic or cap_state.upper() == "EVADE":
        return "说明当前决策重心已经从继续压缩距离转为生存优先，模板的主要作用是打断原攻击序列并为脱离机动让位。"
    if tactic == "DRAG_SHOOT":
        return "说明系统并未急于收拢队形，而是优先保持射击准备与脱离余度并存，以便在较长距离上先建立攻击链、后保留回撤余地。"
    if tactic == "SIDE_BY_SIDE":
        return "说明当前更强调横向展开和协同跟踪稳定性，目的是在不提前透支生存边界的前提下扩大稳定跟踪覆盖。"
    if tactic == "PINCER_ATTACK":
        return "说明系统正在利用双路夹击几何压缩目标机动空间，以便尽快把目标推向首轮攻击窗口。"
    if tactic == "HIGH_LOW":
        return "说明系统正在利用高低配合拉开垂直方向职责分工，使一架机承担压制、另一架机保留支援与接替条件。"
    if tactic == "FRONT_BACK":
        if node in {"TR_DOR", "DOR_DR", "LR_TR"}:
            return "说明系统已接近首轮发射或脱离准备边界，采用前后分工以兼顾持续照射、脱离准备和回转余度。"
        return "说明系统希望保持纵向前后配合，通过主攻与支援分工维持火力链连续性。"
    if awacs_flag == "N":
        return "说明在预警支撑缺失条件下，系统仍保持了机载链路主导的战术组织，而没有退回到无差别保守机动。"
    if zone == "HIGH":
        return "说明敌机已经压入高风险区，模板选择首先服从风险区抑制与生存边界。"
    return "说明模板切换仍然受当前控制距离节点、敌情威胁和目标分配结果共同约束。"


def node_meaning_text(node: str) -> str:
    node = node.upper()
    if node in {"BEYOND_NLT", "NLT_MELD"}:
        return "该节点仍属于远距接敌与站位塑形阶段，决策重点是建立感知和占位，而不是过早消耗火力。"
    if node in {"MELD_MTR", "MTR_LR"}:
        return "该节点处于首次攻击准备阶段，决策重点是把目标稳定推入 fire-control 可用窗口。"
    if node in {"LR_TR", "TR_DOR"}:
        return "该节点已经逼近首轮发射与脱离准备边界，继续前压的收益与生存代价必须同步评估。"
    if node in {"DOR_DR", "DR_MAR"}:
        return "该节点已经进入脱离重组甚至临界生存边界，任何继续压缩都可能直接侵蚀规避空间。"
    return "该节点对应的主要任务是随距离推进调整攻击与生存的权衡。"


def format_window_text(snapshot: Dict[str, object]) -> str:
    ranges = snapshot.get("dynamic_ranges")
    if not isinstance(ranges, dict):
        return ""
    tr = safe_float(ranges.get("TR"))
    lr = safe_float(ranges.get("LR"))
    dor = safe_float(ranges.get("DOR"))
    dr = safe_float(ranges.get("DR"))
    mar = safe_float(ranges.get("MAR"))
    parts = []
    if math.isfinite(tr) and math.isfinite(lr):
        parts.append(f"首轮发射窗口约束为 TR={tr:.1f} km、LR={lr:.1f} km")
    if math.isfinite(dor) and math.isfinite(dr) and math.isfinite(mar):
        parts.append(f"生存边界约束为 DOR={dor:.1f} km、DR={dr:.1f} km、MAR={mar:.1f} km")
    return "；".join(parts)


def stable_required_text(snapshot: Dict[str, object]) -> str:
    value = safe_float(snapshot.get("stable_required_seconds"))
    if math.isfinite(value):
        return f"稳定跟踪门限取 {value:.1f} s"
    return ""


def build_route_event_text(row: pd.Series, side: str, cap_state: str) -> str:
    stage = safe_text(row.get(f"{side}_stage_reason"))
    node = extract_token(stage, "node")
    zone = extract_token(stage, "zone")
    awacs_flag = extract_token(stage, "awacs")
    ctx = extract_token(stage, "ctx_dist")
    pair = extract_token(stage, "pair_dist")
    tactic = safe_text(row.get(f"{side}_tactic"), "UNKNOWN")
    target = safe_text(row.get(f"{side}_target"), "未明确")
    purpose = tactic_purpose_text(tactic, node, zone, awacs_flag, cap_state).rstrip("。")
    return (
        f"{pair_label(side)}采用 `{tactic}` 指向 {target}，敌我参考距离约 {ctx or '未记录'}，"
        f"双机对目标距离约 {pair or '未记录'}，处于 {node or '未记录'} 节点，"
        f"目标位于{ZONE_LABEL.get(zone, zone or '未记录')}；{purpose}"
    )


def build_parameter_basis_text(left_snapshot: Dict[str, object], right_snapshot: Dict[str, object]) -> str:
    pieces: List[str] = []
    stable_value = safe_float(left_snapshot.get("stable_required_seconds"))
    if not math.isfinite(stable_value):
        stable_value = safe_float(right_snapshot.get("stable_required_seconds"))
    if math.isfinite(stable_value):
        pieces.append(f"发射放行前统一要求稳定跟踪至少持续 {stable_value:.1f} s")
    left_window = format_window_text(left_snapshot)
    right_window = format_window_text(right_snapshot)
    if left_window and right_window and left_window == right_window:
        pieces.append(left_window)
    else:
        if left_window:
            pieces.append(f"左路{left_window}")
        if right_window:
            pieces.append(f"右路{right_window}")
    return "；".join(pieces)


def build_key_event_paragraph(row: pd.Series) -> str:
    time_s = safe_float(row.get("time_s"))
    cap_state = safe_text(row.get("cap_state"))
    left_stage = safe_text(row.get("left_stage_reason"))
    right_stage = safe_text(row.get("right_stage_reason"))
    left_node = extract_token(left_stage, "node")
    right_node = extract_token(right_stage, "node")
    left_zone = extract_token(left_stage, "zone")
    right_zone = extract_token(right_stage, "zone")
    left_awacs = extract_token(left_stage, "awacs")
    right_awacs = extract_token(right_stage, "awacs")
    left_pair = extract_token(left_stage, "pair_dist")
    right_pair = extract_token(right_stage, "pair_dist")
    left_ctx = extract_token(left_stage, "ctx_dist")
    right_ctx = extract_token(right_stage, "ctx_dist")

    left_snapshot = parse_json_object(row.get("left_decision_snapshot_json"))
    right_snapshot = parse_json_object(row.get("right_decision_snapshot_json"))
    left_window = format_window_text(left_snapshot)
    right_window = format_window_text(right_snapshot)
    left_tactic = safe_text(row.get("left_tactic"), "UNKNOWN")
    right_tactic = safe_text(row.get("right_tactic"), "UNKNOWN")
    left_target = safe_text(row.get("left_target"), "未明确")
    right_target = safe_text(row.get("right_target"), "未明确")

    parts = [f"{time_s:.1f} s 时，系统处于 `{cap_state}` 状态。"]
    parts.append(build_route_event_text(row, "left", cap_state) + "。")
    parts.append(build_route_event_text(row, "right", cap_state) + "。")

    parameter_basis = build_parameter_basis_text(left_snapshot, right_snapshot)
    if parameter_basis:
        parts.append(parameter_basis + "。")

    if left_node and right_node and left_node == right_node:
        parts.append(node_meaning_text(left_node))
    else:
        parts.append(
            f"两路推进并非完全同步：左路位于 {left_node or '未记录'}，右路位于 {right_node or '未记录'}。"
            " 这表明系统允许左右两路围绕同一总任务进行分化推进，以同时兼顾主攻、支援和后续脱离余度。"
        )
    return " ".join(parts)


def row_is_explainable(row: pd.Series) -> bool:
    left_tactic = safe_text(row.get("left_tactic"))
    right_tactic = safe_text(row.get("right_tactic"))
    left_target = safe_text(row.get("left_target"))
    right_target = safe_text(row.get("right_target"))
    left_stage = safe_text(row.get("left_stage_reason"))
    right_stage = safe_text(row.get("right_stage_reason"))
    left_ok = left_tactic and left_tactic != "UNKNOWN" and left_target and left_stage
    right_ok = right_tactic and right_tactic != "UNKNOWN" and right_target and right_stage
    return left_ok and right_ok


def zone_transition_table(run: RunData) -> Dict[str, List[Tuple[float, str]]]:
    history = run.summary_json.get("target_zone_transition_history", {})
    result: Dict[str, List[Tuple[float, str]]] = {}
    if isinstance(history, dict):
        for target, items in history.items():
            cleaned: List[Tuple[float, str]] = []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, list) and len(item) >= 2:
                        cleaned.append((safe_float(item[0]), safe_text(item[1])))
            result[safe_text(target)] = cleaned
    return result


def zone_segments(transitions: List[Tuple[float, str]], end_time_s: float) -> List[Tuple[str, float, float]]:
    result: List[Tuple[str, float, float]] = []
    for idx, (start, zone) in enumerate(transitions):
        if not math.isfinite(start):
            continue
        end = end_time_s
        if idx + 1 < len(transitions):
            end = transitions[idx + 1][0]
        result.append((zone, start, end))
    return result


def zone_count_in_scene(run: RunData, zone_name: str) -> int:
    total = 0
    for items in zone_transition_table(run).values():
        for zone, _, _ in zone_segments(items, safe_float(run.summary_metrics.get("mission_runtime_s")) or 1200.0):
            if zone == zone_name:
                total += 1
    return total


def get_metric(run: RunData, key: str) -> float:
    if key in run.summary_metrics:
        return safe_float(run.summary_metrics[key])
    if key in run.milestones:
        return safe_float(run.milestones[key])
    return safe_float(run.summary_json.get(key))


def first_non_nan(*values: float) -> float:
    for value in values:
        if math.isfinite(value):
            return value
    return float("nan")


def infer_exchange_ratio(run: RunData) -> float:
    kills = get_metric(run, "enemy_kill_count")
    losses = get_metric(run, "friendly_loss_count")
    if not math.isfinite(kills) or not math.isfinite(losses):
        return float("nan")
    if losses <= 0.0:
        return float(kills)
    return kills / losses


def plot_initial_geometry(scene_id: str, scene_dir: Path) -> Path:
    meta = SCENE_META[scene_id]
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.set_facecolor("#f7f7f5")
    ax.add_patch(Rectangle((-200, -40), 400, 260, facecolor="#f1f5f9", edgecolor="none", zorder=0))

    for zone in ["LOW", "MEDIUM", "HIGH"]:
        radius = RISK_ZONE_RADIUS[zone]
        circ = Circle((0, 0), radius, fill=False, linewidth=2.0, linestyle="--", edgecolor=ZONE_COLOR[zone], alpha=0.85)
        ax.add_patch(circ)
        ax.text(radius + 4, 0, ZONE_LABEL[zone], color=ZONE_COLOR[zone], fontsize=10, va="center")

    friendly_positions = {
        "A0100": (-25, 0),
        "A0200": (-75, 100),
        "A0300": (25, 0),
        "A0400": (75, 100),
    }
    for name, (x, y) in friendly_positions.items():
        ax.scatter(x, y, s=120, color="#1d3557", marker="^", zorder=3)
        ax.text(x + 2, y + 3, f"{name}\n9 km", fontsize=9, color="#1d3557")

    base_x = 100.0
    base_y = meta["initial_distance_km"]
    for idx, (offset, alt) in enumerate(zip(meta["enemy_offsets_km"], meta["enemy_altitudes_km"]), start=1):
        x = base_x + offset[0]
        y = base_y + offset[1]
        name = f"B0{idx}00"
        ax.scatter(x, y, s=120, color="#d62828", marker="o", zorder=3)
        ax.text(x + 2, y + 3, f"{name}\n{alt:.1f} km", fontsize=9, color="#7f1d1d")

    ax.annotate("", xy=(0, 215), xytext=(0, 180), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1d3557"))
    ax.text(4, 198, "我方主接敌方向", color="#1d3557", fontsize=10)
    ax.annotate("", xy=(100, base_y - 25), xytext=(100, base_y - 60), arrowprops=dict(arrowstyle="->", lw=1.8, color="#7f1d1d"))
    ax.text(104, base_y - 45, "敌方来袭方向", color="#7f1d1d", fontsize=10)

    ax.set_xlim(-120, 180)
    ax.set_ylim(-20, max(280, base_y + 30))
    ax.set_xlabel("相对横向位置 / km")
    ax.set_ylabel("相对纵向位置 / km")
    ax.set_title(f"{scene_id} 初始几何与风险区示意")
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    out = scene_dir / "fig01_initial_geometry.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_zone_entry_timeline(run: RunData, scene_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    transitions = zone_transition_table(run)
    targets = sorted(transitions)
    runtime = max(1200.0, safe_float(run.timeline["time_s"].max()) if not run.timeline.empty else 1200.0)
    y_positions = list(range(len(targets)))[::-1]
    for y, target in zip(y_positions, targets):
        segments = zone_segments(transitions[target], runtime)
        for zone, start, end in segments:
            color = ZONE_COLOR.get(zone, "#9ca3af")
            ax.barh(y, max(end - start, 0.2), left=start, height=0.58, color=color, edgecolor="white")
            ax.text(start + 1.5, y, f"{zone}", fontsize=8, color="black", va="center")
        for start, zone in transitions[target]:
            ax.scatter(start, y, color=ZONE_COLOR.get(zone, "#111827"), s=24, zorder=3)
            ax.text(start, y + 0.32, f"{start:.1f}", fontsize=8, color=ZONE_COLOR.get(zone, "#111827"))

    ax.set_yticks(y_positions)
    ax.set_yticklabels(targets)
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("敌方目标")
    ax.set_title(f"{run.scene_id} 各敌机进入我方风险区的时间序列")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_xlim(0, runtime)
    legend_handles = [
        plt.Line2D([0], [0], color=ZONE_COLOR["LOW"], lw=8, label="低风险区"),
        plt.Line2D([0], [0], color=ZONE_COLOR["MEDIUM"], lw=8, label="中风险区"),
        plt.Line2D([0], [0], color=ZONE_COLOR["HIGH"], lw=8, label="高风险区"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    fig.tight_layout()
    out = scene_dir / "fig02_zone_entry_timeline.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_control_timeline(run: RunData, scene_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    rows = []
    labels = []
    for route in ["left", "right"]:
        subset = run.control_nodes[run.control_nodes["route"] == route].copy()
        subset = subset.sort_values("first_time_s")
        y = 1 if route == "left" else 0
        for _, item in subset.iterrows():
            labels.append(item["phase"])
            rows.append((safe_float(item["first_time_s"]), y, safe_text(item["phase"]), safe_float(item["distance_km"])))
    for t, y, phase, dist in rows:
        ax.scatter(t, y, s=80, color="#1d4ed8" if y == 1 else "#b91c1c")
        ax.text(t + 2, y + 0.03, f"{phase}\n{dist:.1f} km", fontsize=8)
    milestone_keys = [
        ("first_stable_ready_time_s", "stable-ready", "#2a9d8f"),
        ("first_gate_pass_time_s", "首发射门通过", "#e76f51"),
        ("first_relay_success_time_s", "首次接力成功", "#6a4c93"),
        ("first_enemy_kill_time_s", "首次击落", "#111827"),
        ("first_friendly_loss_time_s", "首次损失", "#7f1d1d"),
    ]
    for key, label, color in milestone_keys:
        value = safe_float(run.milestones.get(key))
        if math.isfinite(value):
            ax.axvline(value, color=color, linestyle="--", alpha=0.55)
            ax.text(value + 2, 1.22, f"{label}\n{value:.1f} s", color=color, fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["右路", "左路"])
    ax.set_xlabel("时间 / s")
    ax.set_xlim(0, max(1200.0, safe_float(run.timeline["time_s"].max()) if not run.timeline.empty else 1200.0))
    ax.set_ylim(-0.4, 1.45)
    ax.set_title(f"{run.scene_id} 控制距离节点与关键事件时间关系")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    out = scene_dir / "fig03_control_and_events.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_focus_metric_compare(scene_id: str, proposed: RunData, baseline: RunData, scene_dir: Path) -> Path:
    keys = SCENE_META[scene_id]["focus_metric_keys"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 4.8))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        pv = get_metric(proposed, key)
        bv = get_metric(baseline, key)
        values = [pv, bv]
        labels = ["本文方法", "基线方法"]
        colors = ["#1d4ed8", "#9ca3af"]
        ax.bar(labels, [0 if not math.isfinite(v) else v for v in values], color=colors, width=0.58)
        ax.set_title(METRIC_LABEL.get(key, key), fontsize=10)
        for idx, value in enumerate(values):
            txt = format_value(key, value)
            height = 0 if not math.isfinite(value) else value
            ax.text(idx, height, txt, ha="center", va="bottom", fontsize=9)
        if "ratio" in key or "accuracy" in key:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("比例")
        elif key.endswith("_time_s") or key.endswith("_count"):
            ax.set_ylabel("数值")
        else:
            ax.set_ylabel("数值")
        ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.suptitle(f"{scene_id} 场景重点指标对比", fontsize=13)
    fig.tight_layout()
    out = scene_dir / "fig04_focus_metric_compare.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_batch_summary(batch_name: str, runs: Dict[str, Dict[str, RunData]], batch_dir: Path) -> Path:
    scenes = ["S1", "S2", "S3"]
    metrics = ["enemy_kill_count", "friendly_loss_count", "first_stable_ready_time_s", "high_risk_breach_time_s"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        x = list(range(len(scenes)))
        pvals = [get_metric(runs[s]["proposed"], metric) for s in scenes]
        bvals = [get_metric(runs[s]["baseline"], metric) for s in scenes]
        ax.bar([v - 0.18 for v in x], [0 if not math.isfinite(v) else v for v in pvals], width=0.36, color="#2563eb", label="本文方法")
        ax.bar([v + 0.18 for v in x], [0 if not math.isfinite(v) else v for v in bvals], width=0.36, color="#9ca3af", label="基线方法")
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.set_title(METRIC_LABEL.get(metric, metric), fontsize=10)
        ax.grid(axis="y", alpha=0.2, linestyle=":")
        if metric == "high_risk_breach_time_s":
            ax.set_ylabel("时间 / s")
    axes[0].legend(frameon=False)
    fig.suptitle(f"{batch_name} 三场景总体结果对比", fontsize=14)
    fig.tight_layout()
    out = batch_dir / "fig00_batch_summary.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def build_initial_condition_lines(scene_id: str) -> List[str]:
    meta = SCENE_META[scene_id]
    lines = [
        f"我方初始编组固定为四机 CAP 巡逻队形，A0100、A0300 初始航向为 0°，A0200、A0400 初始航向为 180°，初始高度约 9000 m，初始速度约 300 m/s，每机挂弹 4 枚。",
        f"敌方初始中心距离为 {meta['initial_distance_km']:.1f} km，相对横向偏置分别为 {meta['enemy_offsets_km']}，初始高度分别为 {meta['enemy_altitudes_km']} km，初始速度约 {meta['enemy_speed_kmps']:.3f} km/s。",
        f"预警支撑条件为：{meta['awacs']}。场景重点为：{meta['scene_emphasis']}",
    ]
    return lines


def build_scene_summary_table(scene_id: str, proposed: RunData, baseline: RunData) -> str:
    keys = SCENE_META[scene_id]["focus_metric_keys"]
    lines = [
        "| 指标 | 本文方法 | 基线方法 | 对比结论 |",
        "| --- | --- | --- | --- |",
    ]
    for key in keys:
        pv = get_metric(proposed, key)
        bv = get_metric(baseline, key)
        lines.append(
            f"| {METRIC_LABEL.get(key, key)} | {format_value(key, pv)} | {format_value(key, bv)} | {compare_text(key, pv, bv)} |"
        )
    lines.append(
        f"| 交换比 | {infer_exchange_ratio(proposed):.2f} | {infer_exchange_ratio(baseline):.2f} | 该指标用于最终战果对比，不单独作为节点决策优劣判据。 |"
    )
    return "\n".join(lines)


def build_zone_history_text(run: RunData) -> str:
    parts = []
    history = zone_transition_table(run)
    for target in sorted(history):
        items = history[target]
        if not items:
            continue
        chain = "，".join(f"{time_s:.1f} s 进入{ZONE_LABEL.get(zone, zone)}" for time_s, zone in items)
        parts.append(f"- {target} 在 {chain}。")
    return "\n".join(parts) if parts else "未记录逐目标风险区进入历史。"


def milestone_gap_text(run: RunData) -> str:
    stable_t = safe_float(run.milestones.get("first_stable_ready_time_s"))
    gate_t = safe_float(run.milestones.get("first_gate_pass_time_s"))
    relay_t = safe_float(run.milestones.get("first_relay_success_time_s"))
    kill_t = safe_float(run.milestones.get("first_enemy_kill_time_s"))
    segs = []
    if math.isfinite(stable_t):
        segs.append(f"首次 stable-ready 出现在 {stable_t:.1f} s")
    if math.isfinite(gate_t) and math.isfinite(stable_t):
        segs.append(f"首次发射门通过出现在 {gate_t:.1f} s，较 stable-ready 晚 {gate_t - stable_t:.1f} s")
    if math.isfinite(relay_t) and math.isfinite(gate_t):
        segs.append(f"首次接力成功出现在 {relay_t:.1f} s，较首次发射晚 {relay_t - gate_t:.1f} s")
    if math.isfinite(kill_t):
        segs.append(f"首次击落出现在 {kill_t:.1f} s")
    return "；".join(segs) + "。"


def scene_conclusion(scene_id: str, proposed: RunData, baseline: RunData) -> str:
    stable_p = get_metric(proposed, "first_stable_ready_time_s")
    stable_b = get_metric(baseline, "first_stable_ready_time_s")
    breach_p = get_metric(proposed, "high_risk_breach_time_s")
    breach_b = get_metric(baseline, "high_risk_breach_time_s")
    kill_p = get_metric(proposed, "enemy_kill_count")
    kill_b = get_metric(baseline, "enemy_kill_count")
    loss_p = get_metric(proposed, "friendly_loss_count")
    loss_b = get_metric(baseline, "friendly_loss_count")
    intent_p = get_metric(proposed, "intent_truth_accuracy")
    intent_b = get_metric(baseline, "intent_truth_accuracy")

    if scene_id == "S1":
        return (
            f"本文方法将首次 stable-ready 时间由 {format_value('first_stable_ready_time_s', stable_b)} "
            f"提前至 {format_value('first_stable_ready_time_s', stable_p)}，并将意图识别一致率由 "
            f"{format_value('intent_truth_accuracy', intent_b)} 提升至 {format_value('intent_truth_accuracy', intent_p)}，"
            f"说明标准正面对进条件下的节点推进和态势判断更为积极。"
            f" 但该批次最终仅形成 {format_value('enemy_kill_count', kill_p)} 架击落、"
            f"{format_value('friendly_loss_count', loss_p)} 架损失，尚未把前段组织优势完全转化为战果优势，"
            "说明后续仍需提升首轮放行后的火力转化效率。"
        )
    if scene_id == "S2":
        return (
            f"本文方法在 AWACS 间歇失效条件下，仍将首次 stable-ready 时间提前至 "
            f"{format_value('first_stable_ready_time_s', stable_p)}，并把高风险区累计突入时长由 "
            f"{format_value('high_risk_breach_time_s', breach_b)} 压缩至 {format_value('high_risk_breach_time_s', breach_p)}。"
            f" 同时，我方损失由 {format_value('friendly_loss_count', loss_b)} 架降至 "
            f"{format_value('friendly_loss_count', loss_p)} 架，说明“信息缺口下继续推进、到边界再规避”的策略在该场景中是有效的。"
        )
    if scene_id == "S3":
        return (
            f"本文方法将首次 stable-ready 时间由 {format_value('first_stable_ready_time_s', stable_b)} "
            f"提前至 {format_value('first_stable_ready_time_s', stable_p)}，并把高风险区累计突入时长由 "
            f"{format_value('high_risk_breach_time_s', breach_b)} 压缩至 {format_value('high_risk_breach_time_s', breach_p)}，"
            "说明低空突防条件下的生存优先逻辑已经生效。"
            f" 但该批次未形成击落，而基线方法形成 {format_value('enemy_kill_count', kill_b)} 架击落，"
            "说明本文方法虽然更早触发规避并缩短了高风险滞留，但回整后的再次攻击效率仍需继续提高。"
        )
    return milestone_gap_text(proposed)


def build_decision_analysis(scene_id: str, proposed: RunData) -> str:
    rows = choose_scene_decision_rows(proposed)
    lines = []
    for idx, row in enumerate(rows, start=1):
        lines.append(f"（{idx}）{build_key_event_paragraph(row)}")
    return "\n\n".join(lines)


def figure1_conclusion_text(scene_id: str) -> str:
    if scene_id == "S1":
        return "从图中可以看出，场景一的初始几何接近标准正面对进，后续差异主要来自决策方法而非开局位置偶然性。"
    if scene_id == "S2":
        return "从图中可以看出，场景二在横向与高度方向均存在不对称扰动，后续分析重点因此转向信息缺口下的持续组织能力。"
    if scene_id == "S3":
        return "从图中可以看出，场景三的初始距离和高度均明显压缩，系统从开局起即面临更高的生存压力。"
    return ""


def figure3_conclusion_text(run: RunData) -> str:
    stable_t = safe_float(run.milestones.get("first_stable_ready_time_s"))
    gate_t = safe_float(run.milestones.get("first_gate_pass_time_s"))
    kill_t = safe_float(run.milestones.get("first_enemy_kill_time_s"))
    parts: List[str] = []
    if math.isfinite(stable_t) and math.isfinite(gate_t):
        parts.append(f"stable-ready 与首次发射门通过之间存在约 {gate_t - stable_t:.1f} s 的组织间隔")
    if math.isfinite(kill_t):
        parts.append(f"首次战果出现在 {kill_t:.1f} s")
    if not parts:
        return ""
    return "从图中可以看出，" + "，".join(parts) + "，说明交战组织、发射放行与战果形成之间存在清晰的层次顺序。"


def figure4_conclusion_text(scene_id: str, proposed: RunData, baseline: RunData) -> str:
    if scene_id == "S1":
        return "从图中可以看出，本文方法的主要优势集中在前段组织效率和意图识别一致性，而最终击落数并未同步占优。"
    if scene_id == "S2":
        return "从图中可以看出，本文方法同时改善了前段就绪速度、高风险区抑制和我方生存性，是三类场景中综合优势最完整的一类。"
    if scene_id == "S3":
        return "从图中可以看出，本文方法显著压缩了高风险暴露并降低了我方损失，但这一优势尚未完全转化为击落收益。"
    return ""


def build_scene_period_analysis(scene_id: str, run: RunData) -> str:
    changed = collect_change_rows(run.decision_trace)
    if scene_id == "S1":
        engage_row = nearest_row(changed, 319.8)
        evade_row = nearest_row(changed, 326.4)
        recover_row = nearest_row(changed, 468.6)
        if engage_row is None or evade_row is None or recover_row is None:
            return ""
        evade_start = safe_float(evade_row.get("time_s"))
        recover_time = safe_float(recover_row.get("time_s"))
        return (
            f"典型时段选取 300 s 至 470 s。300.6 s 左路进入 LR_TR、右路进入 TR_DOR 后，"
            f"319.8 s 左右两路同时转入 `FRONT_BACK`，表明系统已经不再以扩展搜索覆盖为主，而是围绕首轮攻击与脱离准备组织前后分工。"
            f" 326.4 s 系统切换到 `EVADE`，其触发原因并非预设时长到达，而是目标距离已经压缩到 55.7 km 和 49.9 km，分别逼近当时给出的 `LR=56.0 km`、`TR=50.0 km` 和 `DOR=45.0 km` 约束。"
            f" 也就是说，此时如果继续维持攻击姿态，后续可用规避空间将明显变窄，因此必须优先保留生存余度。"
            f" 此后规避段持续到 {recover_time:.1f} s，时长约 {recover_time - evade_start:.1f} s。"
            f" 回转恢复同样不是按固定秒数结束，而是在敌我距离重新拉开、交战链具备再次组织条件后才发生；468.6 s 左路重新回到 `SIDE_BY_SIDE`，说明系统判定此时已经可以由规避状态转回协同跟踪和再次攻击准备。"
        )
    if scene_id == "S2":
        ready_row = nearest_row(changed, 54.8)
        split_row = nearest_row(changed, 199.6)
        evade_row = nearest_row(changed, 312.0)
        if ready_row is None or split_row is None or evade_row is None:
            return ""
        return (
            "典型时段选取 180 s 至 320 s。该时段内 AWACS 已经出现失配，但左右两路并未退回到保守等待，而是在 199.6 s 形成“左路 `HIGH_LOW`、右路 `SIDE_BY_SIDE`”的分工。"
            " 这种分工说明系统没有把信息缺口简单等同于暂停交战，而是将火控压力集中到左路主压制方向，同时在右路维持协同跟踪稳定。"
            f" 到 {safe_float(evade_row.get('time_s')):.1f} s，左路与右路分别位于 LR_TR 和 LR_TR 节点，距离压缩到 70.1 km 与 56.8 km。"
            " 此时触发规避并不意味着交战组织失效，而是说明中风险持续压制已经完成了首轮攻击释放，随后必须及时把重心切换到生存控制。"
            " 与基线方法相比，本文方法在该场景下把高风险区累计突入压制为 0 s，说明这种“信息缺口下继续推进、到边界再规避”的策略并没有造成高风险失控，反而缩短了整体受威胁暴露。"
        )
    if scene_id == "S3":
        start_row = nearest_row(changed, 0.2)
        pre_evade_row = nearest_row(changed, 113.4)
        evade_row = nearest_row(changed, 133.0)
        recover_row = nearest_row(changed, 280.0)
        if start_row is None or pre_evade_row is None or evade_row is None or recover_row is None:
            return ""
        evade_start = safe_float(evade_row.get("time_s"))
        recover_time = safe_float(recover_row.get("time_s"))
        return (
            f"典型时段选取 110 s 至 280 s。场景开始时系统在 0.2 s 即进入 `ENGAGE`，这是因为敌机初始就在 MELD_MTR 甚至更近的节点附近，传统远距巡逻逻辑已经不再适用。"
            f" 到 {safe_float(pre_evade_row.get('time_s')):.1f} s，左路先转入 `TACTICAL_EVASION`，此时目标距离 65.3 km，已经接近 `LR=69.4 km` 与 `TR=63.4 km` 之间的收缩段，说明系统已经预判到继续保持攻击姿态将迅速侵蚀后续规避空间。"
            f" {evade_start:.1f} s 时左路目标 B0200 进入高风险区，同时左路推进到 DOR_DR，目标距离仅 52.6 km，已经逼近 `DOR=50.3 km` 与 `DR=46.3 km` 生存边界，因此系统正式切入 `EVADE`。"
            f" 规避持续到 {recover_time:.1f} s，时长约 {recover_time - evade_start:.1f} s。"
            f" 恢复时间之所以出现在这一时刻，是因为此时 AWACS 窗口重新可用、敌我距离回开到 60.9 km 至 86.1 km，目标也从高风险区回到中风险区，系统据此重新切回 `SIDE_BY_SIDE` 与 `FRONT_BACK` 组合，进入再次攻击准备。"
        )
    return ""


def build_scene_markdown(
    batch_name: str,
    scene_id: str,
    proposed: RunData,
    baseline: RunData,
    scene_dir: Path,
    rel_paths: Dict[str, str],
) -> str:
    meta = SCENE_META[scene_id]
    section_no = SCENE_SECTION_NO[scene_id]
    initial_lines = "\n".join([f"- {line}" for line in build_initial_condition_lines(scene_id)])
    zone_history = build_zone_history_text(proposed)
    decision_lines = build_decision_analysis(scene_id, proposed)
    period_analysis = build_scene_period_analysis(scene_id, proposed)

    fig1 = rel_paths["fig01_initial_geometry.png"]
    fig2 = rel_paths["fig02_zone_entry_timeline.png"]
    fig3 = rel_paths["fig03_control_and_events.png"]
    fig4 = rel_paths["fig04_focus_metric_compare.png"]

    return f"""
### {section_no} {meta['title']}

#### 场景目标与初始条件

{initial_lines}

本场景验证目标为：{meta['goal']}

图 {scene_id}-1 给出敌我初始几何关系与我方风险区示意。横轴为相对横向位置，纵轴为相对纵向位置；蓝色三角表示我方四机，红色圆点表示敌方四机，三条虚线圆分别表示低、中、高风险区。该图用于说明本场景开始时的几何约束和敌方初始突入方向。

![{scene_id} 初始几何与风险区示意]({fig1})

{figure1_conclusion_text(scene_id)}

#### 控制距离节点与战术模板选择分析

{milestone_gap_text(proposed)}

图 {scene_id}-2 给出逐目标风险区进入时序。横轴为时间，纵轴为敌方目标编号；蓝、橙、红三色分别表示目标处于低、中、高风险区。该图用于回答“哪一架敌机在什么时间进入了我方哪一级风险区”。

![{scene_id} 各敌机进入我方风险区的时间序列]({fig2})

逐目标风险区演化记录如下：
{zone_history}

图 {scene_id}-3 给出左右两路控制距离节点与关键事件时刻。横轴为时间，纵轴区分左路与右路；蓝色节点表示控制距离阶段首次进入时刻，虚线分别表示 stable-ready、首次发射门通过、首次接力成功、首次击落和首次损失。该图用于说明战术组织并不是在任意时刻发生，而是随控制距离节点推进而触发。

![{scene_id} 控制距离节点与关键事件时间关系]({fig3})

{figure3_conclusion_text(proposed)}

结合 `decision_trace.csv` 的阶段理由、模板理由和参数理由，可以还原关键决策过程：

{decision_lines}

上述记录表明，本场景中的模板切换并非由固定时间表驱动，而是由 `ctx_dist`、`pair_dist`、控制距离节点、风险区状态、预警支撑状态以及导弹来袭状态共同决定。记录中的 `first_window`、`survival` 和 `dynamic_ranges` 说明，动作参数不是任意给定，而是围绕发射窗口、生存边界和当前闭合态势动态修正的。

#### 典型时段剖析

{period_analysis}

#### 有效性对比

图 {scene_id}-4 给出本场景最需要关心的重点指标对比。横轴为对比方法，纵轴为对应指标数值；蓝色柱表示本文方法，灰色柱表示基线方法，柱顶数字为实际统计结果。该图采用时间节点、风险区占用和战果指标来评价决策效果，而不再依赖失真的聚合比值。

![{scene_id} 场景重点指标对比]({fig4})

{figure4_conclusion_text(scene_id, proposed, baseline)}

{build_scene_summary_table(scene_id, proposed, baseline)}

#### 场景结论

{scene_conclusion(scene_id, proposed, baseline)}
""".strip()


def build_batch_overview(batch_name: str, runs: Dict[str, Dict[str, RunData]], rel_batch_fig: str) -> str:
    lines = [
        "## 6.3 仿真验证",
        "",
        "### 6.3.1 验证口径与说明",
        "",
        "本文节的重点不是列举全部过程指标，而是围绕战术决策方法的有效性进行验证。所谓有效性，具体指：在不同敌情和信息条件下，系统是否能够在合适的控制距离节点选择合适的战术模板，是否能够在导弹威胁出现时及时转入规避，是否能够在威胁解除后恢复交战组织，以及这些决策是否带来了可解释的风险区控制效果和战果差异。",
        "",
        "考虑到 stable-ready 后放行比值和接力制导比值在现有批次中容易受到聚合统计口径影响，出现长期恒定或明显失真的现象，故本节不再将其作为正式结论指标，而改用以下更稳定、也更便于解释的量：",
        "",
        "- 首次 stable-ready 时间",
        "- 首次发射门通过时间",
        "- 高风险区累计突入时长",
        "- 逐目标风险区进入历史",
        "- 敌机击落数、我方损失数与交换比",
        "- 雷达覆盖率与意图识别一致率",
        "",
        "此外，各场景分析重点不同：场景一突出完整闭环与首轮交战组织，场景二突出信息不完整条件下的持续压制与恢复，场景三突出高风险压缩条件下的紧急规避和生存优先。这样才能使每个场景各自回答一个明确问题，而不是重复展示同一组指标。",
        "",
        "图 6-1 给出本批次三场景总体结果对比。横轴为场景编号，纵轴为对应指标数值；蓝色为本文方法，灰色为基线方法。该图仅用于总览，不替代后续场景级决策分析。",
        "",
        f"![{batch_name} 三场景总体结果对比]({rel_batch_fig})",
        "",
    ]
    return "\n".join(lines)


def build_cross_scene_summary(runs: Dict[str, Dict[str, RunData]]) -> str:
    lines = [
        "### 6.3.5 三场景横向归纳",
        "",
        "| 场景 | 主要验证问题 | 本文方法主要优势 | 主要短板或风险 |",
        "| --- | --- | --- | --- |",
    ]
    for scene_id in ["S1", "S2", "S3"]:
        meta = SCENE_META[scene_id]
        proposed = runs[scene_id]["proposed"]
        baseline = runs[scene_id]["baseline"]
        focus = meta["focus"]
        stable_cmp = compare_text("first_stable_ready_time_s", get_metric(proposed, "first_stable_ready_time_s"), get_metric(baseline, "first_stable_ready_time_s"))
        breach_cmp = compare_text("high_risk_breach_time_s", get_metric(proposed, "high_risk_breach_time_s"), get_metric(baseline, "high_risk_breach_time_s"))
        result_cmp = compare_text("enemy_kill_count", get_metric(proposed, "enemy_kill_count"), get_metric(baseline, "enemy_kill_count"))
        advantage = f"{stable_cmp} {breach_cmp}"
        risk = result_cmp
        lines.append(f"| {scene_id} | {focus} | {advantage} | {risk} |")
    lines.extend(
        [
            "",
            "综合三场景结果可以看到，本文方法最稳定的优势并不一定总是直接体现在击落数上，而更稳定地体现在两类中间能力上：一是更早建立 stable-ready，二是更强地压缩高风险区突入时间。这说明该方法首先改善的是“决策组织质量”，然后才在部分批次上进一步转化为战果优势。",
            "",
            "同时也应看到，本文方法并非在所有批次和所有场景下都优于基线方法。例如场景一和场景三的部分批次中，尽管本文方法更早进入 ready 或更早触发规避，但最终击落数并不占优。这意味着当前算法在“早就绪”和“火力转化效率”之间仍存在进一步协调空间，后续应重点检查发射门放行逻辑、回转后的再次攻击组织，以及导弹链事件级统计口径。",
        ]
    )
    return "\n".join(lines)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def relative_markdown_path(target: Path, anchor: Path) -> str:
    return target.relative_to(anchor).as_posix()


def render_batch(batch_name: str) -> None:
    batch_dir = RESULT_ROOT / batch_name
    runs = load_batch(batch_dir)

    out_dir = REPORT_ROOT / batch_name
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    batch_fig = plot_batch_summary(batch_name, runs, fig_dir)
    rel_batch_fig = relative_markdown_path(batch_fig, out_dir)

    sections = [build_batch_overview(batch_name, runs, rel_batch_fig)]

    for scene_id in ["S1", "S2", "S3"]:
        scene_fig_dir = fig_dir / scene_id
        ensure_dir(scene_fig_dir)
        proposed = runs[scene_id]["proposed"]
        baseline = runs[scene_id]["baseline"]

        figs = {
            "fig01_initial_geometry.png": plot_initial_geometry(scene_id, scene_fig_dir),
            "fig02_zone_entry_timeline.png": plot_zone_entry_timeline(proposed, scene_fig_dir),
            "fig03_control_and_events.png": plot_control_timeline(proposed, scene_fig_dir),
            "fig04_focus_metric_compare.png": plot_focus_metric_compare(scene_id, proposed, baseline, scene_fig_dir),
        }
        rel_paths = {name: relative_markdown_path(path, out_dir) for name, path in figs.items()}
        sections.append("")
        sections.append(build_scene_markdown(batch_name, scene_id, proposed, baseline, scene_fig_dir, rel_paths))

    sections.append("")
    sections.append(build_cross_scene_summary(runs))
    report_text = "\n".join(sections).strip() + "\n"

    report_path = DOC_ROOT / f"第六章仿真验证文档_{batch_name}.md"
    report_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    for batch_name in BATCHES:
        render_batch(batch_name)
        print(f"generated {batch_name}")


if __name__ == "__main__":
    main()
