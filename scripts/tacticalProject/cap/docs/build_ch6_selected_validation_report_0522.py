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
from matplotlib.patches import Rectangle
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DOC_ROOT = Path(r"D:\Pycharm\LAG\scripts\tacticalProject\cap\docs")
RESULT_ROOT = Path(r"D:\Pycharm\LAG\scripts\tacticalProject\cap_results\Chapter6_validation")
REPORT_STEM = "第六章仿真验证文档_0522定样本重写"
REPORT_PATH = DOC_ROOT / f"{REPORT_STEM}.md"
ASSET_DIR = DOC_ROOT / "ch6_validation_0522_assets"


@dataclass(frozen=True)
class SampleSpec:
    scenario_id: str
    sample_tag: str
    folder_name: str
    sample_dir: Path
    sample_label: str
    note: str
    representative: bool = False


SCENARIO_META: Dict[str, Dict[str, object]] = {
    "S1": {
        "title": "场景一：低风险正面对进条件下的完整链路验证",
        "task_focus": "验证低风险正面对进条件下，阶段决策、战术模板、机动执行和发射门放行能否形成完整闭环，并检验首轮放行后规避回转的时序是否合理。",
        "design_intent": "以标准四机对四机正面对进几何为基础，预警链路全程有效，重点观察控制距离推进、模板切换和交战后回转恢复。",
        "expected_zone": "低风险区",
        "initial_distance_km": 266.4,
        "enemy_altitudes_km": [10.0, 10.0, 10.0, 10.0],
        "enemy_offsets_km": [(-60.0, 0.0), (-48.0, 0.0), (48.0, 0.0), (60.0, 0.0)],
        "enemy_speed_mps": 300.0,
        "awacs_mode": "全程可用",
        "friendly_altitude_text": "约 9.0 km",
        "friendly_speed_text": "约 300 m/s",
        "scene_goal": "证明在标准初始几何下，系统不是简单直冲或固定开火，而是按控制距离节点完成模板转换、稳定跟踪、首轮放行、导弹来袭规避和规避后回转。",
        "representative_key": "225448",
        "sample_order": ["225438", "225448", "001313", "001401"],
        "figures": [
            ("overview.png", "场景一代表样本总体态势图", "给出敌我航迹、交战展开和终局收束的总体空间构型。"),
            ("fig03b_control_distance_timeline.png", "场景一控制距离节点推进图", "用于识别左右两路从远距接敌到首轮放行再到规避回收的节点推进顺序。"),
            ("fig05_engagement_chain.png", "场景一交战链展开图", "用于观察稳定交战就绪、发射门放行、接力制导和击杀事件之间的因果先后关系。"),
            ("fig09_flight_safety_envelope.png", "场景一飞行安全包线图", "用于说明规避回转阶段没有突破高度与速度安全边界。"),
        ],
    },
    "S2": {
        "title": "场景二：信息缺口条件下的持续压制验证",
        "task_focus": "验证预警链路间歇失配条件下，系统能否依靠机载感知和双路模板分工维持交战压力，并在规避与回转后继续保持压制。",
        "design_intent": "以更近距离、敌方高低混合来袭和长时间预警缺口为条件，考察系统在不完整态势下的恢复能力和长期压制能力。",
        "expected_zone": "中风险区",
        "initial_distance_km": 240.0,
        "enemy_altitudes_km": [10.2, 8.8, 10.2, 8.8],
        "enemy_offsets_km": [(-66.0, 0.0), (-48.0, -6.0), (48.0, 0.0), (66.0, -6.0)],
        "enemy_speed_mps": 330.0,
        "awacs_mode": "间歇失配，累计失配约 400 s",
        "friendly_altitude_text": "约 9.0 km",
        "friendly_speed_text": "约 300 m/s",
        "scene_goal": "证明在信息缺口存在时，系统仍能维持模板骨架不散，且在导弹来袭触发规避后，可以通过回转重新建立压制态势。",
        "representative_key": "225438",
        "sample_order": ["225438", "001339", "001353"],
        "figures": [
            ("overview.png", "场景二代表样本总体态势图", "用于展示持续压制过程中敌我航迹、分工方向和后段收口过程。"),
            ("fig03b_control_distance_timeline.png", "场景二控制距离节点推进图", "用于观察在预警信息受扰情况下，左右两路是否仍能维持节点推进和再组织。"),
            ("fig05_engagement_chain.png", "场景二交战链展开图", "用于说明门限更严但接力制导链更长、压制保持更久的特点。"),
            ("fig09_flight_safety_envelope.png", "场景二飞行安全包线图", "用于说明长期压制过程中速度和高度虽被压缩，但未出现失控性坠损。"),
        ],
    },
    "S3": {
        "title": "场景三：低空突防条件下的生存优先验证",
        "task_focus": "验证低空突防、高压近距和长时间预警失配条件下，系统能否前移交战链，同时在首轮兑现后及时止压并守住生存边界。",
        "design_intent": "以低空突防几何和短时跃升开局压缩交战窗口，重点考察首轮前移、导弹来袭强制规避以及规避后继续守区的能力。",
        "expected_zone": "高风险区",
        "initial_distance_km": 150.0,
        "enemy_altitudes_km": [3.4, 3.4, 3.7, 3.7],
        "enemy_offsets_km": [(-22.0, -6.0), (-10.0, -14.0), (26.0, 14.0), (40.0, 6.0)],
        "enemy_speed_mps": 330.0,
        "awacs_mode": "长时间间歇失配，累计失配约 930.2 s",
        "friendly_altitude_text": "约 9.0 km",
        "friendly_speed_text": "约 300 m/s",
        "scene_goal": "证明高压样本下系统的价值不在于一味前冲，而在于首轮链条前移后能迅速触发规避、及时止压，并将友机损失控制在可接受范围内。",
        "representative_key": "233513",
        "sample_order": ["233513", "001339"],
        "figures": [
            ("overview.png", "场景三代表样本总体态势图", "用于展示高压近距开局、首轮交战和后段残余兵力收束。"),
            ("fig03b_control_distance_timeline.png", "场景三控制距离节点推进图", "用于识别低空突防样本中节点整体前移和规避触发提前发生的事实。"),
            ("fig05_engagement_chain.png", "场景三交战链展开图", "用于说明高压样本中首轮前移、接力制导和损失出现的先后关系。"),
            ("fig09_flight_safety_envelope.png", "场景三飞行安全包线图", "用于说明高压规避阶段系统仍将高度与速度维持在可控范围内。"),
        ],
    },
}


SAMPLES: List[SampleSpec] = [
    SampleSpec(
        scenario_id="S1",
        sample_tag="225438",
        folder_name="COMPARE_ALL_20260521_225438",
        sample_dir=RESULT_ROOT / "COMPARE_ALL_20260521_225438" / "S1_20260521_225438",
        sample_label="225438（S1）",
        note="225438 批次中的场景一样本，形成 4:0 战果且未出现高风险区突入，可作为低风险基线对照。",
    ),
    SampleSpec(
        scenario_id="S1",
        sample_tag="225448",
        folder_name="COMPARE_ALL_20260521_225448",
        sample_dir=RESULT_ROOT / "COMPARE_ALL_20260521_225448" / "S1_20260521_225448",
        sample_label="225448（S1）",
        note="225448 批次中的场景一样本，控制距离推进、首次放行、规避回转与清场链条最完整，作为代表样本展开详析。",
        representative=True,
    ),
    SampleSpec(
        scenario_id="S1",
        sample_tag="001313",
        folder_name="COMPARE_ALL_20260522_001313",
        sample_dir=RESULT_ROOT / "COMPARE_ALL_20260522_001313" / "S1_20260522_001313",
        sample_label="001313（S1）",
        note="001313 批次中的场景一样本，稳定交战就绪建立较早，但后段出现 2 架我机损失，体现低风险场景下的执行波动。",
    ),
    SampleSpec(
        scenario_id="S1",
        sample_tag="001401",
        folder_name="COMPARE_ALL_20260522_001401",
        sample_dir=RESULT_ROOT / "COMPARE_ALL_20260522_001401" / "S1_20260522_001401",
        sample_label="001401（S1）",
        note="001401 批次中的场景一样本，再次验证低风险正面对进条件下零高风险突入与 4:0 战果可以稳定复现。",
    ),
    SampleSpec(
        scenario_id="S2",
        sample_tag="225438",
        folder_name="COMPARE_ALL_20260521_225438",
        sample_dir=RESULT_ROOT / "COMPARE_ALL_20260521_225438" / "S2_20260521_231708",
        sample_label="225438（S2）",
        note="225438 批次中的场景二样本，在预警失配累计约 400 s 条件下仍形成 3:0 持续压制结果，作为代表样本展开详析。",
        representative=True,
    ),
    SampleSpec(
        scenario_id="S2",
        sample_tag="001339",
        folder_name="COMPARE_ALL_20260522_001339",
        sample_dir=RESULT_ROOT / "COMPARE_ALL_20260522_001339" / "S2_20260522_004859",
        sample_label="001339（S2）",
        note="001339 批次中的场景二样本，最终形成 2:1 交换比，说明信息缺口条件下持续压制战果存在波动。",
    ),
    SampleSpec(
        scenario_id="S2",
        sample_tag="001353",
        folder_name="COMPARE_ALL_20260522_001353",
        sample_dir=RESULT_ROOT / "COMPARE_ALL_20260522_001353" / "S2_20260522_002434",
        sample_label="001353（S2）",
        note="001353 批次中的场景二样本，高风险区累计滞留 495.6 s，可用于标示中风险持续压制场景的边界位置。",
    ),
    SampleSpec(
        scenario_id="S3",
        sample_tag="233513",
        folder_name="COMPARE_ALL_20260521_225448",
        sample_dir=RESULT_ROOT / "COMPARE_ALL_20260521_225448" / "S3_20260521_233513",
        sample_label="233513（S3）",
        note="233513 样本用于代表低空突防高压场景，稳定交战就绪仅需 5.0 s，但后段出现 2 架我机损失。",
        representative=True,
    ),
    SampleSpec(
        scenario_id="S3",
        sample_tag="001339",
        folder_name="COMPARE_ALL_20260522_001339",
        sample_dir=RESULT_ROOT / "COMPARE_ALL_20260522_001339" / "S3_20260522_012112",
        sample_label="001339（S3）",
        note="001339 批次中的场景三样本，首轮放行更早，但最终仅形成 1:1 交换比，说明高压样本对回收时机更敏感。",
    ),
]


ZONE_TEXT = {"LOW": "低风险区", "MEDIUM": "中风险区", "HIGH": "高风险区"}
ZONE_COLORS = {"LOW": "#9ecae1", "MEDIUM": "#f6bd60", "HIGH": "#d1495b"}
ZONE_Y = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


TERM_TABLE: List[Tuple[str, str]] = [
    ("稳定交战就绪（stable ready）", "目标已满足稳定火控跟踪门限，可进入发射门进一步筛选。"),
    ("发射门通过（gate pass）", "在距离窗口、稳定跟踪、友机安全和执行条件同时满足后，允许进入实际发射。"),
    ("接力制导成功（relay success）", "导弹中制导阶段由另一架友机成功接替照射或继续维持照射责任。"),
    ("控制距离节点（control distance nodes）", "用于推进阶段决策的战术距离门限，包括 NLT、MELD、MTR、LR、TR、DOR、DR、MAR 及二次攻击窗口。"),
    ("模板切换（tactic template switching）", "战术决策层依据风险区、敌我几何和控制距离节点，在不同双机模板之间进行切换。"),
    ("参数决策（parameter decision）", "在既定模板下，进一步确定稳定跟踪门限、发射窗口、生存边界和动作参数。"),
]


STATE_TEXT = {
    "ENGAGE": "交战",
    "EVADE": "规避",
    "INTERCEPT": "截获重组",
    "PATROL": "巡逻",
}

TACTIC_TEXT = {
    "DRAG_SHOOT": "拖曳射击",
    "PINCER_ATTACK": "钳形夹击",
    "FRONT_BACK": "前后夹击",
    "HIGH_LOW": "高低配合",
    "SIDE_BY_SIDE": "并列压制",
    "TACTICAL_EVASION": "战术规避",
    "UNKNOWN": "未记录",
}

PHASE_TEXT = {
    "BEYOND_NLT": "NLT外远距接敌段",
    "NLT_MELD": "NLT-MELD压缩段",
    "MELD_MTR": "MELD-MTR塑形段",
    "MTR_LR": "MTR-LR首次攻击准备段",
    "LR_TR": "LR-TR首次发射窗口段",
    "TR_DOR": "TR-DOR脱离准备段",
    "DOR_DR": "DOR-DR再攻击决断段",
    "DR_MAR": "DR-MAR强制规避判定段",
    "BEYOND_MAR": "MAR外回收整理段",
}

PHASE_SHORT_TEXT = {
    "BEYOND_NLT": "NLT外",
    "NLT_MELD": "NLT-MELD",
    "MELD_MTR": "MELD-MTR",
    "MTR_LR": "MTR-LR",
    "LR_TR": "LR-TR",
    "TR_DOR": "TR-DOR",
    "DOR_DR": "DOR-DR",
    "DR_MAR": "DR-MAR",
    "BEYOND_MAR": "MAR外",
}


def state_display(value: object) -> str:
    raw = safe_text(value, "未记录").upper()
    return STATE_TEXT.get(raw, raw)


def tactic_display(value: object) -> str:
    raw = safe_text(value, "未记录").upper()
    return TACTIC_TEXT.get(raw, raw)


def phase_code(value: object) -> str:
    raw = safe_text(value, "未记录").replace("TacticalPhase.", "").upper()
    return raw


def phase_display(value: object) -> str:
    code = phase_code(value)
    if code == "未记录":
        return code
    return PHASE_TEXT.get(code, code.replace("_", "-"))


def zone_display(value: object) -> str:
    raw = safe_text(value, "未记录").upper()
    return ZONE_TEXT.get(raw, raw)


def first_number(text: object) -> float:
    raw = safe_text(text)
    matched = re.search(r"-?\d+(?:\.\d+)?", raw)
    return float(matched.group(0)) if matched else float("nan")


def parse_kv_text(text: object) -> Dict[str, str]:
    raw = safe_text(text)
    result: Dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z_']+)=([^ ]+)", raw):
        result[key] = value.strip()
    return result


def parse_window(text: str, label: str) -> Dict[str, float]:
    matched = re.search(rf"{label}=\[([^\]]+)\]", text)
    if not matched:
        return {}
    result: Dict[str, float] = {}
    for item in matched.group(1).split(","):
        if ":" not in item:
            continue
        key, raw_value = item.split(":", 1)
        result[key.strip()] = first_number(raw_value)
    return result


def cap_state_reason_display(value: object) -> str:
    raw = safe_text(value)
    lower = raw.lower()
    if "missile incoming" in lower:
        return "状态切换原因为“导弹来袭，立即规避”。"
    if "awacs picture available" in lower and "hold intercept" in lower:
        return "预警图景恢复后，系统先保持截获重组，以便重新建立目标分配和回转几何。"
    if "hostile enters medium/high risk zone" in lower:
        return "敌方已压入中高风险区，系统转入交战执行。"
    return raw if raw else "未记录状态切换原因。"


def safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def safe_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def format_seconds(value: float) -> str:
    return "未形成" if not math.isfinite(value) else f"{value:.1f} s"


def format_km(value: float) -> str:
    return "未统计" if not math.isfinite(value) else f"{value:.1f} km"


def format_m(value: float) -> str:
    return "未统计" if not math.isfinite(value) else f"{value:.1f} m"


def format_mps(value: float) -> str:
    return "未统计" if not math.isfinite(value) else f"{value:.1f} m/s"


def format_ratio(value: float) -> str:
    return "未统计" if not math.isfinite(value) else f"{value * 100:.1f}%"


def format_count(value: float) -> str:
    return "未统计" if not math.isfinite(value) else str(int(round(value)))


def md_table(headers: List[str], rows: Iterable[Iterable[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def find_sample(sample_tag: str, scenario_id: str) -> SampleSpec:
    for spec in SAMPLES:
        if spec.sample_tag == sample_tag and spec.scenario_id == scenario_id:
            return spec
    raise KeyError(f"无法找到样本 {scenario_id}-{sample_tag}")


def scenario_samples(scenario_id: str) -> List[SampleSpec]:
    specs = [spec for spec in SAMPLES if spec.scenario_id == scenario_id]
    order = SCENARIO_META[scenario_id]["sample_order"]
    return sorted(specs, key=lambda item: order.index(item.sample_tag))


def representative_sample(scenario_id: str) -> SampleSpec:
    for spec in scenario_samples(scenario_id):
        if spec.representative:
            return spec
    raise KeyError(f"场景 {scenario_id} 未设置代表样本")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def read_metric_map(path: Path) -> Dict[str, float]:
    df = read_csv(path)
    metric_col = "metric_key" if "metric_key" in df.columns else "metric"
    return {safe_text(row[metric_col]): safe_float(row["value"]) for _, row in df.iterrows()}


def read_milestones(path: Path) -> Dict[str, float]:
    df = read_csv(path)
    return {safe_text(row["key"]): safe_float(row["time_s"]) for _, row in df.iterrows()}


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class SampleData:
    spec: SampleSpec
    summary: Dict[str, object]
    metrics: Dict[str, float]
    milestones: Dict[str, float]
    decision_trace: pd.DataFrame
    control_nodes: pd.DataFrame
    control_segments: pd.DataFrame
    timeline: pd.DataFrame
    events: pd.DataFrame


def load_sample(spec: SampleSpec) -> SampleData:
    tables = spec.sample_dir / "tables"
    return SampleData(
        spec=spec,
        summary=load_json(spec.sample_dir / "summary.json"),
        metrics=read_metric_map(tables / "summary_metrics.csv"),
        milestones=read_milestones(tables / "milestone_times.csv"),
        decision_trace=read_csv(tables / "decision_trace.csv"),
        control_nodes=read_csv(tables / "control_distance_nodes.csv"),
        control_segments=read_csv(tables / "control_distance_segments.csv"),
        timeline=read_csv(spec.sample_dir / "timeline.csv"),
        events=read_csv(spec.sample_dir / "events.csv"),
    )


def all_samples_loaded() -> Dict[str, SampleData]:
    return {f"{spec.scenario_id}-{spec.sample_tag}": load_sample(spec) for spec in SAMPLES}


def metric(sample: SampleData, key: str) -> float:
    if key in sample.metrics:
        return safe_float(sample.metrics[key])
    if key in sample.milestones:
        return safe_float(sample.milestones[key])
    return safe_float(sample.summary.get(key))


def parse_snapshot(text: object) -> Dict[str, object]:
    raw = safe_text(text)
    if not raw.startswith("{"):
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def first_row_near(df: pd.DataFrame, target_time_s: float) -> Optional[pd.Series]:
    if df.empty:
        return None
    idx = (df["time_s"] - target_time_s).abs().idxmin()
    return df.loc[idx]


def first_row_at_or_after(df: pd.DataFrame, target_time_s: float, tolerance_s: float = 8.0) -> Optional[pd.Series]:
    if df.empty:
        return None
    subset = df[df["time_s"] >= target_time_s]
    if not subset.empty:
        idx = subset["time_s"].idxmin()
        row = df.loc[idx]
        if abs(safe_float(row["time_s"]) - target_time_s) <= tolerance_s:
            return row
    return first_row_near(df, target_time_s)


def first_row_at_or_before(df: pd.DataFrame, target_time_s: float, tolerance_s: float = 8.0) -> Optional[pd.Series]:
    if df.empty:
        return None
    subset = df[df["time_s"] <= target_time_s]
    if not subset.empty:
        idx = subset["time_s"].idxmax()
        row = df.loc[idx]
        if abs(safe_float(row["time_s"]) - target_time_s) <= tolerance_s:
            return row
    return first_row_near(df, target_time_s)


def state_windows(sample: SampleData) -> pd.DataFrame:
    mask = sample.timeline["cap_state"].ne(sample.timeline["cap_state"].shift(1))
    cols = [
        "time_s",
        "cap_state",
        "friendly_alive",
        "enemy_alive",
        "stable_ready_count",
        "gate_pass_count",
        "relay_success_count",
        "enemy_kill_count",
        "friendly_loss_count",
    ]
    return sample.timeline.loc[mask, cols].copy()


def control_phase_table(sample: SampleData) -> List[List[object]]:
    rows: List[List[object]] = []
    for _, row in sample.control_nodes.iterrows():
        rows.append(
            [
                "左路" if safe_text(row["route"]) == "left" else "右路",
                safe_text(row["phase"]),
                format_seconds(safe_float(row["first_time_s"])),
                format_km(safe_float(row["distance_km"])),
                safe_text(row["description"]),
            ]
        )
    return rows


def control_phase_table_cn(sample: SampleData) -> List[List[object]]:
    rows: List[List[object]] = []
    for _, row in sample.control_nodes.iterrows():
        rows.append(
            [
                "左路" if safe_text(row["route"]) == "left" else "右路",
                phase_display(row["phase"]),
                format_seconds(safe_float(row["first_time_s"])),
                format_km(safe_float(row["distance_km"])),
                safe_text(row["description"]),
            ]
        )
    return rows


def dynamic_ranges_text(snapshot: Dict[str, object]) -> str:
    ranges = snapshot.get("dynamic_ranges")
    if not isinstance(ranges, dict):
        return "未记录动态控制距离。"
    keys = ["NLT", "MELD", "MTR", "LR", "TR", "DOR", "DR", "MAR"]
    parts = []
    for key in keys:
        value = safe_float(ranges.get(key))
        if math.isfinite(value):
            parts.append(f"{key}={value:.1f} km")
    return "，".join(parts) if parts else "未记录动态控制距离。"


def dynamic_ranges_cn(snapshot: Dict[str, object]) -> str:
    ranges = snapshot.get("dynamic_ranges")
    if not isinstance(ranges, dict):
        return "未记录动态控制距离。"
    keys = [
        ("NLT", "不晚于截获距离"),
        ("MELD", "混战进入距离"),
        ("MTR", "机动跟踪距离"),
        ("LR", "首次发射准备距离"),
        ("TR", "交战转换距离"),
        ("DOR", "脱离决断距离"),
        ("DR", "重组决断距离"),
        ("MAR", "最小规避距离"),
    ]
    parts = []
    for key, label in keys:
        value = safe_float(ranges.get(key))
        if math.isfinite(value):
            parts.append(f"{label}{key}={value:.1f} km")
    return "，".join(parts) if parts else "未记录动态控制距离。"


def parameter_basis_text(snapshot: Dict[str, object]) -> str:
    stable_required = safe_float(snapshot.get("stable_required_seconds"))
    min_quality = safe_float(snapshot.get("min_quality"))
    closure_rate = safe_float(snapshot.get("closure_rate_mps"))
    time_to_merge = safe_float(snapshot.get("time_to_merge_s"))
    alt_delta = safe_float(snapshot.get("nearest_altitude_delta_km"))
    target_speed = safe_float(snapshot.get("nearest_target_speed_mps"))
    zone = safe_text(snapshot.get("nearest_target_zone"), "未记录")
    pieces = []
    if math.isfinite(stable_required):
        pieces.append(f"稳定跟踪门限固定为 {stable_required:.1f} s")
    if math.isfinite(min_quality):
        pieces.append(f"最小跟踪质量门限为 {min_quality:.2f}")
    if math.isfinite(closure_rate):
        pieces.append(f"闭合速率为 {closure_rate:.1f} m/s")
    if math.isfinite(time_to_merge) and time_to_merge < 1.0e7:
        pieces.append(f"估计会合时间约为 {time_to_merge:.1f} s")
    if math.isfinite(alt_delta):
        pieces.append(f"最近目标高度差约为 {alt_delta:.2f} km")
    if math.isfinite(target_speed):
        pieces.append(f"目标速度约为 {target_speed:.1f} m/s")
    if zone:
        pieces.append(f"目标风险区判定为{ZONE_TEXT.get(zone.upper(), zone)}")
    return "，".join(pieces) + "。" if pieces else "未记录参数门限。"


def parameter_reason_cn(text: object, snapshot: Dict[str, object]) -> str:
    raw = safe_text(text)
    if not raw:
        return parameter_basis_text(snapshot)
    stable = first_number(re.search(r"stable_required_s=([^ ]+)", raw).group(1) if re.search(r"stable_required_s=([^ ]+)", raw) else "")
    gate_phase_match = re.search(r"gate_phase=([^ ]+)", raw)
    gate_phase = gate_phase_match.group(1) if gate_phase_match else ""
    first_window = parse_window(raw, "first_window")
    second_window = parse_window(raw, "second_window")
    survival = parse_window(raw, "survival")
    closure_rate_match = re.search(r"closure_rate=([^ ]+)", raw)
    target_speed_match = re.search(r"target_speed=([^ ]+)", raw)
    parts = []
    if math.isfinite(stable):
        parts.append(f"稳定交战就绪门限保持为 {stable:.1f} s")
    if gate_phase:
        parts.append(f"当前发射门判定按 {phase_display(gate_phase)} 执行")
    if "TR" in first_window and "LR" in first_window:
        parts.append(f"首次发射窗口约束为 LR={first_window['LR']:.1f} km、TR={first_window['TR']:.1f} km")
    if survival:
        dor = survival.get("DOR")
        dr = survival.get("DR")
        mar = survival.get("MAR")
        pieces = []
        if dor is not None:
            pieces.append(f"DOR={dor:.1f} km")
        if dr is not None:
            pieces.append(f"DR={dr:.1f} km")
        if mar is not None:
            pieces.append(f"MAR={mar:.1f} km")
        if pieces:
            parts.append("生存边界约束为 " + "、".join(pieces))
    if closure_rate_match:
        closure_value = first_number(closure_rate_match.group(1))
        if math.isfinite(closure_value):
            parts.append(f"闭合速率约为 {closure_value:.1f} m/s")
    if target_speed_match:
        speed_value = first_number(target_speed_match.group(1))
        if math.isfinite(speed_value):
            parts.append(f"目标速度约为 {speed_value:.1f} m/s")
    if not parts:
        return parameter_basis_text(snapshot)
    return "；".join(parts) + "。"


def stage_reason_cn(text: object) -> str:
    kv = parse_kv_text(text)
    parts = []
    ctx = first_number(kv.get("ctx_dist"))
    pair = first_number(kv.get("pair_dist"))
    node = kv.get("node")
    zone = kv.get("zone")
    awacs = kv.get("awacs")
    missile = kv.get("missile_incoming")
    rwr = kv.get("rwr")
    if math.isfinite(ctx):
        parts.append(f"敌我参考距离约为 {ctx:.1f} km")
    if math.isfinite(pair):
        parts.append(f"双机对目标距离约为 {pair:.1f} km")
    if node:
        parts.append(f"当前处于 {phase_display(node)}")
    if zone:
        parts.append(f"敌方处于{zone_display(zone)}")
    if awacs:
        parts.append("预警图景可用" if awacs.upper() == "Y" else "预警图景不可用")
    if missile:
        parts.append("存在导弹来袭压力" if missile.upper() == "Y" else "尚未记录导弹来袭压力")
    if rwr and rwr != "0":
        parts.append(f"告警强度为 {rwr}")
    return "，".join(parts) + "。" if parts else "未记录阶段判据。"


def maneuver_mapping_text(tactic_name: str, phase_name: str, is_left: bool = True) -> str:
    tactic = tactic_name.upper()
    phase = phase_name.upper()
    if tactic == "DRAG_SHOOT":
        if phase in {"BEYOND_NLT", "NLT_MELD"}:
            return "拖曳射击模板在远距阶段采用“射手直压、支援机外摆约 35°”的构型，用于同时保留照射前出与脱离余度。"
        if phase == "LR_TR":
            return "进入首次发射准备区后，射手仅保留约 8° 的小外摆，以防止在首轮放行前过度偏航。"
        if phase in {"TR_DOR", "DOR_DR"}:
            return "发射后保持段中，射手外摆约 70°、支援机外摆约 42°，其目的不是继续追压，而是为照射保持和规避脱离腾出空间。"
    if tactic == "PINCER_ATTACK":
        return "钳形夹击模板按距离分层调整展开角，前段展开角约 24° 至 40°，后段脱离角约 48° 至 66°，并在后续再压缩阶段回收至 24° 至 38°。"
    if tactic == "FRONT_BACK":
        if phase in {"TR_DOR", "DOR_DR", "DR_MAR"}:
            return "前后夹击模板在脱离准备段把前机与后机的职责明显拉开，前机承担照射和压制，后机保留后方支撑与回转余度。"
        return "前后夹击模板的核心不是简单排成纵队，而是在同一航线上形成主压与支撑的纵向分工。"
    if tactic == "HIGH_LOW":
        return "高低配合模板中，长机在前段保持平飞压近，僚机在前段持续爬升至高位，以便后续承担补照射和上方支援。"
    if tactic == "SIDE_BY_SIDE":
        if phase in {"TR_DOR", "DOR_DR"}:
            return "并列模板在发射后保持段转为左右对称外摆，其作用是把双机同时从持续前压切换为对称脱离。"
        return "并列模板强调横向展开与同步压近，优先追求覆盖面和双机火控连续性。"
    if tactic == "TACTICAL_EVASION":
        return "战术规避模板不是新的攻击模板，而是对原有攻击链的优先级打断，目标转为生存优先与后续回转准备。"
    return "当前模板的动作参数由模板-节点映射规则直接给出，而不是在仿真结束后人工指定。"


def target_zone_history(summary: Dict[str, object]) -> Dict[str, List[Tuple[float, str]]]:
    history = summary.get("target_zone_transition_history", {})
    result: Dict[str, List[Tuple[float, str]]] = {}
    if isinstance(history, dict):
        for enemy_id, items in history.items():
            cleaned: List[Tuple[float, str]] = []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, list) and len(item) >= 2:
                        cleaned.append((safe_float(item[0]), safe_text(item[1])))
            result[safe_text(enemy_id)] = cleaned
    return result


def zone_segments(history: List[Tuple[float, str]], end_time_s: float) -> List[Tuple[float, float, str]]:
    result: List[Tuple[float, float, str]] = []
    for idx, (start, zone) in enumerate(history):
        if not math.isfinite(start):
            continue
        end = end_time_s
        if idx + 1 < len(history):
            end = safe_float(history[idx + 1][0])
        if math.isfinite(end) and end > start:
            result.append((start, end, zone))
    return result


def copy_image(src: Path, dst_name: str) -> str:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    dst = ASSET_DIR / dst_name
    shutil.copy2(src, dst)
    return f"./{ASSET_DIR.name}/{dst.name}"


def plot_zone_entry_timeline(sample: SampleData, dst_name: str) -> str:
    history = target_zone_history(sample.summary)
    fig, ax = plt.subplots(figsize=(10.0, 2.8))
    ax.set_facecolor("#f7f5ef")
    order = ["B0100", "B0200", "B0300", "B0400"]
    y_positions = {enemy_id: idx for idx, enemy_id in enumerate(order)}
    end_time = safe_float(sample.summary.get("time_s"))
    if not math.isfinite(end_time):
        end_time = 1200.0

    for enemy_id in order:
        y = y_positions[enemy_id]
        ax.hlines(y, 0.0, end_time, color="#d9d9d9", linewidth=8, zorder=0)
        for start, end, zone in zone_segments(history.get(enemy_id, []), end_time):
            color = ZONE_COLORS.get(zone.upper(), "#999999")
            ax.add_patch(
                Rectangle(
                    (start, y - 0.28),
                    max(0.2, end - start),
                    0.56,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.95,
                )
            )
            if end - start >= 18.0:
                ax.text((start + end) / 2.0, y, ZONE_TEXT.get(zone.upper(), zone), ha="center", va="center", fontsize=8, color="#1f1f1f")

    ax.set_xlim(0.0, end_time)
    ax.set_yticks([y_positions[item] for item in order])
    ax.set_yticklabels(order)
    ax.set_xlabel("时间 / s")
    ax.set_title(f"{sample.spec.sample_label} 敌机风险区进入时序")
    handles = [
        plt.Line2D([0], [0], color=ZONE_COLORS["LOW"], linewidth=8, label="低风险区"),
        plt.Line2D([0], [0], color=ZONE_COLORS["MEDIUM"], linewidth=8, label="中风险区"),
        plt.Line2D([0], [0], color=ZONE_COLORS["HIGH"], linewidth=8, label="高风险区"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=8)
    ax.grid(axis="x", color="#d0d0d0", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSET_DIR / dst_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return f"./{ASSET_DIR.name}/{out_path.name}"


def plot_sample_stability_table(sample_list: List[SampleData], dst_name: str) -> str:
    labels = [sample.spec.sample_tag for sample in sample_list]
    stable = [metric(sample, "first_stable_ready_time_s") for sample in sample_list]
    gate = [metric(sample, "first_gate_pass_time_s") for sample in sample_list]
    relay = [metric(sample, "first_relay_success_time_s") for sample in sample_list]

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_facecolor("#f7f5ef")
    colors = ["#2a6f97", "#d97706", "#2f855a"]
    series = [
        ("首次稳定交战就绪", stable, colors[0], 1.6),
        ("首次发射门通过", gate, colors[1], 1.9),
        ("首次接力制导成功", relay, colors[2], 2.2),
    ]
    x = list(range(len(labels)))
    for label, values, color, offset in series:
        ax.plot(x, values, marker="o", markersize=7, linewidth=2.0, color=color, label=label)
        for idx, val in enumerate(values):
            if math.isfinite(val):
                ax.text(idx, val + offset, f"{val:.1f}", color=color, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("时间 / s")
    ax.set_title("样本关键里程碑时刻对比")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSET_DIR / dst_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return f"./{ASSET_DIR.name}/{out_path.name}"


def plot_control_distance_single(sample: SampleData, dst_name: str) -> str:
    df = sample.control_segments.copy()
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.set_facecolor("#f7f5ef")
    colors = {"left": "#2a6f97", "right": "#d97706"}
    labels = {"left": "左路", "right": "右路"}
    for route in ["left", "right"]:
        sub = df[df["route"] == route]
        if sub.empty:
            continue
        xs = [safe_float(sub.iloc[0]["start_time_s"])]
        ys = [safe_float(sub.iloc[0]["start_distance_km"])]
        for _, row in sub.iterrows():
            xs.append(safe_float(row["end_time_s"]))
            ys.append(safe_float(row["end_distance_km"]))
        ax.plot(xs, ys, color=colors[route], linewidth=2.2, label=labels[route])
        for _, row in sub.iterrows():
            x = safe_float(row["end_time_s"])
            y = safe_float(row["end_distance_km"])
            ax.scatter([x], [y], color=colors[route], s=36, zorder=3)
            phase_name = PHASE_SHORT_TEXT.get(phase_code(row["to_phase"]), phase_code(row["to_phase"]).replace("_", "-"))
            ax.text(x + 4.0, y + (4.0 if route == "left" else -6.0), phase_name, fontsize=8, color=colors[route])
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("最近敌我距离 / km")
    ax.set_title("双路控制距离节点推进")
    ax.grid(color="#bdbdbd", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSET_DIR / dst_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return f"./{ASSET_DIR.name}/{out_path.name}"


def plot_engagement_chain_single(sample: SampleData, dst_name: str) -> str:
    df = sample.timeline.copy()
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.set_facecolor("#f7f5ef")
    x = df["time_s"]
    series = [
        ("稳定交战就绪目标数", "stable_ready_count", "#2a6f97"),
        ("发射门通过事件数", "gate_pass_unique_count", "#d97706"),
        ("接力制导成功事件数", "relay_success_unique_count", "#2f855a"),
        ("敌机击落数", "enemy_kill_count", "#c44536"),
    ]
    for label, col, color in series:
        if col in df.columns:
            ax.step(x, df[col], where="post", label=label, color=color, linewidth=2.0)
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("事件累计数量")
    ax.set_title("交战链关键事件时序")
    ax.grid(color="#bdbdbd", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSET_DIR / dst_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return f"./{ASSET_DIR.name}/{out_path.name}"


def plot_safety_single(sample: SampleData, dst_name: str) -> str:
    df = sample.timeline.copy()
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.set_facecolor("#f7f5ef")
    x = df["time_s"]
    if "friendly_min_speed_mps" in df.columns:
        ax.plot(x, df["friendly_min_speed_mps"], color="#2a6f97", linewidth=2.0, label="我方最小速度")
    if "friendly_mean_speed_mps" in df.columns:
        ax.plot(x, df["friendly_mean_speed_mps"], color="#6aa84f", linewidth=1.8, label="我方平均速度")
    if "friendly_max_descent_rate_mps" in df.columns:
        ax.plot(x, df["friendly_max_descent_rate_mps"], color="#c44536", linewidth=1.8, label="我方最大下降率")
    ax.axhline(120.0, color="#888888", linestyle="--", linewidth=1.0, label="最低安全速度参考")
    ax.axhline(30.0, color="#aa6666", linestyle=":", linewidth=1.0, label="下降率警戒参考")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("速度 / 下降率")
    ax.set_title("规避与回转阶段飞行安全约束")
    ax.grid(color="#bdbdbd", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSET_DIR / dst_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return f"./{ASSET_DIR.name}/{out_path.name}"


def initial_state_table(scenario_id: str) -> List[List[object]]:
    meta = SCENARIO_META[scenario_id]
    friendly_rows = [
        ["A0100", "我方", "Su-27SK", "0°", meta["friendly_altitude_text"], meta["friendly_speed_text"], "4"],
        ["A0200", "我方", "Su-27SK", "180°", meta["friendly_altitude_text"], meta["friendly_speed_text"], "4"],
        ["A0300", "我方", "Su-27SK", "0°", meta["friendly_altitude_text"], meta["friendly_speed_text"], "4"],
        ["A0400", "我方", "Su-27SK", "180°", meta["friendly_altitude_text"], meta["friendly_speed_text"], "4"],
    ]
    enemy_rows = []
    for idx, alt in enumerate(meta["enemy_altitudes_km"], start=1):
        enemy_rows.append(
            [
                f"B0{idx}00",
                "敌方",
                "F-16",
                "180°",
                f"{alt:.1f} km",
                f"{meta['enemy_speed_mps']:.0f} m/s",
                "4",
            ]
        )
    return friendly_rows + enemy_rows


def scene_setup_table(scenario_id: str) -> List[List[object]]:
    meta = SCENARIO_META[scenario_id]
    if scenario_id == "S1":
        return [
            ["任务目标", "验证标准正面对进条件下，控制距离推进、模板切换、首次发射、规避回转和再交战能否构成完整闭环。"],
            ["风险定位", "低风险场景，重点考察高风险区是否被有效阻止。"],
            ["敌我初始距离", f"{meta['initial_distance_km']:.1f} km"],
            ["敌方高度配置", "四目标均在 10 km 级高度接近，几何对称性强。"],
            ["预警条件", str(meta["awacs_mode"])],
            ["分析重点", "何时由塑形转入攻击准备、何时允许首次放行、何时因生存边界触发规避、何时具备回转再压制条件。"],
        ]
    if scenario_id == "S2":
        return [
            ["任务目标", "验证预警信息存在持续缺口时，双路编队能否维持压制骨架，并在规避后恢复压制。"],
            ["风险定位", "中风险场景，重点考察持续压制与再组织能力。"],
            ["敌我初始距离", f"{meta['initial_distance_km']:.1f} km"],
            ["敌方高度配置", "高低混合来袭，内含低空扰动目标，压缩感知与火控一致性。"],
            ["预警条件", str(meta["awacs_mode"])],
            ["分析重点", "预警失配期间是否仍能保持模板分工、规避后何时先重组再回转、为什么恢复交战后仍会再次规避。"],
        ]
    return [
        ["任务目标", "验证低空突防和高压近距条件下，系统能否把首轮交战链前移，同时在高压下及时止压保生。"],
        ["风险定位", "高风险场景，重点考察止压与生存边界保持。"],
        ["敌我初始距离", f"{meta['initial_distance_km']:.1f} km"],
        ["敌方高度配置", "四目标以 3.4 km 至 3.7 km 低空突防，迫使控制距离节点整体前移。"],
        ["预警条件", str(meta["awacs_mode"])],
        ["分析重点", "为什么 5 s 稳定火控门限是可接受的最小值、为什么首轮兑现后必须立即止压、回转后为何难以恢复完整交战能力。"],
    ]


def sample_selection_table(scenario_id: str) -> List[List[object]]:
    rows = []
    for spec in scenario_samples(scenario_id):
        rows.append(
            [
                spec.sample_label,
                "代表详析样本" if spec.representative else "稳定性支撑样本",
                spec.note,
            ]
        )
    return rows


def decision_layer_table() -> List[List[object]]:
    return [
        ["阶段决策层", "敌我参考距离、风险区层级、预警图景是否可用、导弹来袭告警", "任务状态切换与控制距离阶段推进", "是否在合适距离段内进入交战、规避或截获重组，而不是提前或滞后切换。"],
        ["战术决策层", "目标分配结果、左右两路几何关系、敌方高度与威胁方向", "双机战术模板选择", "模板是否与当前节点任务一致，例如塑形阶段强调展开，交战阶段强调压制，规避阶段强调生存。"],
        ["机动决策层", "当前模板、控制距离节点、生存边界 DOR/DR/MAR、导弹来袭压力", "动作序列与打断时机", "是否能说明为什么在该时刻必须中断攻击而转入规避，以及规避后何时允许回转。"],
        ["参数决策层", "稳定跟踪门限、发射窗口 LR/TR、闭合速率、目标速度与高度差", "平飞时间、转向角、展开角、保持段门限", "参数选择是否既满足火控条件，又保留生存余度；如果延长或缩短门限，是否会导致放行过早、过晚或规避余度不足。"],
    ]


def stability_summary_table(scenario_id: str, loaded: Dict[str, SampleData]) -> List[List[object]]:
    rows = []
    for spec in scenario_samples(scenario_id):
        sample = loaded[f"{spec.scenario_id}-{spec.sample_tag}"]
        rows.append(
            [
                spec.sample_label,
                format_seconds(metric(sample, "first_stable_ready_time_s")),
                format_seconds(metric(sample, "first_gate_pass_time_s")),
                format_seconds(metric(sample, "first_relay_success_time_s")),
                format_seconds(metric(sample, "high_risk_breach_time_s")),
                f"{int(round(metric(sample, 'enemy_kill_count')))} / {int(round(metric(sample, 'friendly_loss_count')))}",
            ]
        )
    return rows


def focus_metrics_table(scenario_id: str, sample: SampleData) -> List[List[object]]:
    if scenario_id == "S1":
        return [
            ["首次稳定交战就绪时间", format_seconds(metric(sample, "first_stable_ready_time_s")), "用于判断协同探测链何时转化为可持续火控，而不是偶然一次短暂跟踪。"],
            ["首次发射门通过时间", format_seconds(metric(sample, "first_gate_pass_time_s")), "用于判断系统是否在稳定跟踪建立后继续等待合适窗口，而不是一满足跟踪就立即发射。"],
            ["首次接力制导成功时间", format_seconds(metric(sample, "first_relay_success_time_s")), "用于判断首轮发射后照射责任是否能够平稳转移，支撑规避后仍保持杀伤链。"],
            ["最小敌我距离", format_km(metric(sample, "min_enemy_to_friendly_km")), "用于验证首轮放行和规避回转没有把双机压入不可恢复的近距缠斗边界。"],
            ["高风险区累计滞留时间", format_seconds(metric(sample, "high_risk_breach_time_s")), "低风险场景应尽量保持为 0，用于验证规避和回转时机足以阻止敌机突入最高风险区。"],
            ["敌机击落数 / 我方损失数", f"{int(round(metric(sample, 'enemy_kill_count')))} / {int(round(metric(sample, 'friendly_loss_count')))}", "用于从最终结果验证完整闭环不是形式闭环，而是能够兑现为正向战果。"],
        ]
    if scenario_id == "S2":
        return [
            ["预警失配累计时间", format_seconds(metric(sample, "enemy_awacs_denied_time_s")), "用于界定外层信息缺口强度，说明压制恢复是在感知条件受损情况下完成的。"],
            ["首次发射门通过时间", format_seconds(metric(sample, "first_gate_pass_time_s")), "用于判断系统是否在信息不完整时仍保持放行节制，而不是简单放宽门限。"],
            ["接力制导成功率", format_ratio(metric(sample, "relay_success_rate_unique")), "用于验证长期压制过程中中制导链能否维持，不因规避与回转而断裂。"],
            ["规避后恢复交战时间", "约 121.2 s", "由 279.0 s 首次规避到 400.2 s 恢复交战计算，用于判断重组与回转是否过慢或过早。"],
            ["最小敌我距离", format_km(metric(sample, "min_enemy_to_friendly_km")), "用于验证持续压制虽逼近中高风险边界，但尚未滑入不可恢复的高压缠斗。"],
            ["敌机击落数 / 我方损失数", f"{int(round(metric(sample, 'enemy_kill_count')))} / {int(round(metric(sample, 'friendly_loss_count')))}", "用于从最终结果验证“规避—重组—再压制”链条是否成立。"],
        ]
    return [
        ["首次稳定交战就绪时间", format_seconds(metric(sample, "first_stable_ready_time_s")), "用于说明高压近距场景中控制距离节点已整体前移，传统远距塑形余度基本消失。"],
        ["首次发射门通过时间", format_seconds(metric(sample, "first_gate_pass_time_s")), "用于判断首轮放行是否足够前移，从而不丢失低空突防拦截窗口。"],
        ["首次我机损失时间", format_seconds(metric(sample, "first_friendly_loss_time_s")), "用于度量止压后仍未完全脱险的时间边界，反映生存代价。"],
        ["高风险区累计滞留时间", format_seconds(metric(sample, "high_risk_breach_time_s")), "用于判断系统在高压区内是否能够尽快止压并退出最危险边界。"],
        ["最小敌我距离", format_km(metric(sample, "min_enemy_to_friendly_km")), "用于验证高压样本已逼近近距极限，解释为何必须优先规避而不能继续压制。"],
        ["敌机击落数 / 我方损失数", f"{int(round(metric(sample, 'enemy_kill_count')))} / {int(round(metric(sample, 'friendly_loss_count')))}", "用于评价首轮前移与生存优先策略的任务代价。"],
    ]


def milestone_table(sample: SampleData) -> List[List[object]]:
    keys = [
        ("首次预警发现", "first_awacs_track_time_s"),
        ("首次机载雷达跟踪", "first_radar_track_time_s"),
        ("四目标全探完成", "full_detect_time_s"),
        ("首次稳定交战就绪", "first_stable_ready_time_s"),
        ("首次发射门通过", "first_gate_pass_time_s"),
        ("首次接力制导成功", "first_relay_success_time_s"),
        ("首次敌机被击落", "first_enemy_kill_time_s"),
        ("首次我机损失", "first_friendly_loss_time_s"),
    ]
    rows = []
    for label, key in keys:
        rows.append([label, format_seconds(metric(sample, key))])
    return rows


def scenario_intro_text(scenario_id: str) -> str:
    meta = SCENARIO_META[scenario_id]
    return (
        f"{meta['title']}。"
        f"该场景的任务定位是：{meta['task_focus']}"
        f"{meta['scene_goal']}"
    )


def initial_state_text(scenario_id: str) -> str:
    meta = SCENARIO_META[scenario_id]
    return (
        f"该场景采用固定四机对四机想定，敌我兵力规模一致，每机挂弹 4 枚。"
        f"我方维持双路巡逻构型，A0100 与 A0300 为长机，A0200 与 A0400 为僚机。"
        f"敌方初始来袭距离约为 {meta['initial_distance_km']:.1f} km，敌方高度层设置为 "
        f"{' / '.join(f'{item:.1f} km' for item in meta['enemy_altitudes_km'])}，"
        f"预警条件为“{meta['awacs_mode']}”。"
    )


def overview_text(scenario_id: str, sample: SampleData) -> str:
    if scenario_id == "S1":
        return (
            "低风险基线样本的核心不是追求最快击落，而是检验主链是否完整闭合。"
            "从总体态势图可以看到，双路编队在远距阶段并未一开始就过度前冲，而是在较长接敌段内完成目标占用、战术展开和首次发射准备。"
            "代表样本最终形成 4:0 结果，且高风险区累计滞留时间为 0 s，说明后段规避和回转没有把敌机放任进入最危险边界。"
        )
    if scenario_id == "S2":
        return (
            "场景二总体上表现为“先恢复、再放行、后维持”的长期压制节奏。"
            "预警链路虽然累计失配 400 s，但代表样本仍形成 3:0 结果，说明该场景的价值不在于瞬时发现得多快，而在于感知链恢复后能否把压制长期维持住。"
        )
    return (
        "场景三的总体态势表现为“前段高压、首轮前移、后段反复回收”。"
        "代表样本仅用 5.0 s 即形成首次稳定交战就绪，说明控制距离链整体前移已经发生；但最终仅形成 2:2 结果，也说明在高压样本中，何时规避、何时回转比是否更早开火更重要。"
    )


def scenario_focus_intro(scenario_id: str) -> str:
    if scenario_id == "S1":
        return "场景一重点检验完整链路是否闭合，因此保留首次稳定交战就绪、首次发射门通过、首次接力制导成功以及规避前后风险区变化等指标。"
    if scenario_id == "S2":
        return "场景二重点检验信息缺口下的持续压制能力，因此强调首次放行时刻、接力制导成功率、规避后恢复时间以及高风险区滞留边界。"
    return "场景三重点检验高压近距条件下的生存优先能力，因此强调首次稳定交战就绪时间、首次放行前移幅度、高风险区滞留时间以及规避后守区能力。"


def figure_text_map(scenario_id: str) -> Dict[str, str]:
    if scenario_id == "S1":
        return {
            "overview": "该图给出了代表样本中敌我总体航迹、双路前出、首次交战以及规避回转后的收束关系。它用于建立整个决策分析的空间背景。",
            "control": "该图以同一坐标展示左右两路最近敌我距离的压缩和回收过程，便于识别何时进入 NLT、MTR、LR、TR、DOR 与 DR 等关键控制距离节点。",
            "chain": "该图按统一时间轴显示稳定交战就绪、发射门通过、接力制导成功和敌机击落的先后关系，可用于区分“具备跟踪条件”“具备放行条件”和“真正形成打击结果”三个不同层次。",
            "safety": "该图将规避与回转阶段的我方最小速度、平均速度和最大下降率放在同一张图中，用于判断规避动作是否在安全包线内完成。",
        }
    if scenario_id == "S2":
        return {
            "overview": "该图展示信息缺口条件下代表样本的总体态势，重点可观察规避后重新压向敌机的路径是否形成二次压制。",
            "control": "该图把左右两路控制距离压缩和回收放在同一张图中，用于识别规避触发后何时回到可继续压制的距离窗口。",
            "chain": "该图给出持续压制过程中稳定交战就绪、发射门通过、接力制导成功与击杀事件的累计演化，可用于判断压制链是否在规避后仍能继续推进。",
            "safety": "该图用于判断长期压制与多次回转过程中速度与下降率是否失控，从而验证回转不是以飞行品质失稳为代价换取压制效果。",
        }
    return {
        "overview": "该图展示高压近距开局下的代表样本总体航迹，重点可观察首轮前移、规避脱离和回转守区的空间关系。",
        "control": "该图把左右两路控制距离节点放在同一张图中，可直接看出高压样本中节点整体前移以及规避后距离回收较慢的事实。",
        "chain": "该图给出高压样本中稳定交战就绪、发射门通过、接力制导成功和击落事件的累计时序，用于观察首轮兑现虽早，但后续交战能力衰减也更快。",
        "safety": "该图用于验证高压规避阶段虽动作激烈，但最小速度和下降率总体仍维持在可解释的包线范围内。",
    }


def timeline_metrics_text(sample: SampleData, at_time: float) -> str:
    row = first_row_at_or_after(sample.timeline, at_time)
    if row is None:
        return "未记录时序统计。"
    parts = []
    items = [
        ("稳定交战就绪目标数", "stable_ready_count", False),
        ("发射门通过累计事件数", "gate_pass_unique_count", False),
        ("接力制导成功累计事件数", "relay_success_unique_count", False),
        ("敌机击落数", "enemy_kill_count", False),
        ("我方损失数", "friendly_loss_count", False),
        ("最近敌我距离", "nearest_enemy_to_friendly_km", True),
    ]
    for label, col, is_km in items:
        if col not in row.index:
            continue
        val = safe_float(row[col])
        if not math.isfinite(val):
            continue
        if is_km:
            parts.append(f"{label}约为 {val:.1f} km")
        else:
            parts.append(f"{label}为 {int(round(val))}")
    return "，".join(parts) + "。" if parts else "未记录时序统计。"


def timeline_row(sample: SampleData, at_time: float) -> Optional[pd.Series]:
    return first_row_at_or_after(sample.timeline, at_time)


def decision_row(sample: SampleData, at_time: float) -> Optional[pd.Series]:
    return first_row_at_or_after(sample.decision_trace, at_time)


def snapshot_for(row: Optional[pd.Series], side: str = "left") -> Dict[str, object]:
    if row is None:
        return {}
    return parse_snapshot(row.get(f"{side}_decision_snapshot_json"))


def int_from_row(row: Optional[pd.Series], key: str) -> Optional[int]:
    if row is None or key not in row.index:
        return None
    value = safe_float(row[key])
    if not math.isfinite(value):
        return None
    return int(round(value))


def float_from_row(row: Optional[pd.Series], key: str) -> float:
    if row is None or key not in row.index:
        return float("nan")
    return safe_float(row[key])


def decision_window_rows(scenario_id: str, sample: SampleData) -> List[List[object]]:
    rows: List[List[object]] = []
    if scenario_id == "S1":
        rows.extend(
            [
                [
                    "95.4-128.4 s",
                    "战术决策层 / 机动决策层",
                    "敌我参考距离由 199.9 km 压缩至 177.8 km，预警图景持续可用；95.4 s 时稳定交战就绪目标数仍为 0，128.4 s 时增至 2，但双方仍处于 NLT 外至 NLT-MELD 压缩段。",
                    "左路由“拖曳射击”切换为“前后夹击”，右路切换为“钳形夹击”，先完成主压与支援分工，不提前释放火力窗口。",
                    "直到 274.8 s 才出现首次发射门通过，说明该阶段模板切换的作用是塑形与站位，而不是抢先发射。",
                ],
                [
                    "193.0-274.8 s",
                    "参数决策层 / 战术决策层",
                    "193.0 s 时左路进入 MELD-MTR 塑形段，稳定交战就绪目标数为 2，最近敌我距离约 137.6 km；动态首次发射窗口仍对应 LR=84.0 km、TR=67.0 km。",
                    "继续保持“前后夹击”与“钳形夹击”模板，等待控制距离压缩和稳定火控数量继续上升后再放行。",
                    "274.8 s 时稳定交战就绪目标数升至 3，并出现首次发射门通过，表明系统没有因为先出现稳定跟踪就过早开火。",
                ],
                [
                    "324.4 s",
                    "阶段决策层 / 机动决策层",
                    "导弹来袭触发强制打断；最近敌我距离已压缩至 59.8 km，累计发射门通过事件数为 6，累计接力制导成功事件数为 15，左路生存边界已收缩到 DOR=52.2 km、DR=48.2 km、MAR=38.9 km。",
                    "系统由交战切换为规避，中断继续压制，优先保留生存余度和后续回转能力。",
                    "规避后到 447.0 s，最近敌我距离重新拉大到 79.7 km，累计敌机击落数升至 2，高风险区累计滞留保持为 0 s，说明规避没有中断已形成的杀伤链，且避免了高风险突入。",
                ],
                [
                    "447.0 s 及以后",
                    "阶段决策层 / 战术决策层",
                    "预警图景恢复，累计接力制导成功事件数已达 26，累计敌机击落数为 2，最近敌我距离恢复到 79.7 km。",
                    "系统结束规避并恢复交战，通过回转重新建立清场态势，而不是在导弹刚来袭后立即折返。",
                    "代表样本最终形成 4:0 战果，且高风险区累计滞留时间为 0 s，证明回转时机既保证了生存，也保留了继续清场的机会。",
                ],
            ]
        )
        return rows
    if scenario_id == "S2":
        rows.extend(
            [
                [
                    "86.2-140.0 s",
                    "战术决策层",
                    "左路已先进入 NLT-MELD 压缩段，右路仍处于 NLT 外；预警图景尚可使用，但敌方高低混合来袭导致双路几何不同步。",
                    "左右两路均采用“并列压制”模板，优先保证横向覆盖和火控连续性，而不是使用穿插幅度更大的模板。",
                    "在预警链尚未完全失配前，系统允许双路不同步推进但不拆散骨架，为后续信息缺口阶段保留连续感知基础。",
                ],
                [
                    "236.2 s",
                    "参数决策层",
                    "左路已进入 LR-TR 首次发射窗口段，稳定交战就绪目标数为 3，最近敌我距离约 81.2 km，但预警图景已不可用。",
                    "系统仍按首次发射窗口放行，不因外层信息缺口而放宽发射门条件。",
                    "首次发射门通过出现在信息缺口已显现之后，说明放行依据仍是稳定火控与控制距离，而不是简单依赖预警链路。",
                ],
                [
                    "279.0 s",
                    "阶段决策层 / 机动决策层",
                    "导弹来袭触发规避；最近敌我距离已压缩至 55.2 km，累计发射门通过事件数为 6，累计接力制导成功事件数为 16，左路已逼近 DOR=49.5 km、DR=45.5 km、MAR=38.0 km 的生存边界。",
                    "系统中断持续压制，由交战切换为规避，优先保护后续再组织能力。",
                    "规避发生时首轮杀伤链已经形成，但系统没有继续贪图近距追加压制，表明生存边界优先级高于继续压近。",
                ],
                [
                    "390.2-400.2 s",
                    "阶段决策层",
                    "规避后最近敌我距离仍在 56 km 至 59 km 区间内，累计敌机击落数为 1，外层图景逐步恢复，但双路尚未完全回到可持续压制状态。",
                    "390.2 s 先进入“截获重组”，400.2 s 再恢复“交战”，即先重建队形和目标分配，再恢复压制。",
                    "这 10 s 过渡说明恢复交战不是瞬时切换；若在 390.2 s 直接恢复交战，则重组不足，若继续拖延到 20 s 以后，则会把本可利用的再压制窗口让掉。",
                ],
                [
                    "416.8 s",
                    "机动决策层 / 阶段决策层",
                    "恢复交战仅 16.6 s 后再次出现导弹来袭，最近敌我距离进一步压缩到 50.2 km，系统尚未新增击落结果。",
                    "系统再次切回规避，说明恢复压制后仍持续监视生存边界，不因刚刚回转就强行维持攻击模板。",
                    "代表样本最终形成 3:0 战果且高风险区累计滞留为 0 s，证明“规避—重组—再压制—再规避”的节奏有效地把持续压制与生存控制结合起来。",
                ],
            ]
        )
        return rows
    rows.extend(
        [
            [
                "5.0-68.8 s",
                "参数决策层 / 战术决策层",
                "开局仅 5.0 s 即形成首次稳定交战就绪；控制距离节点已前移到 MTR-LR 附近，传统远距塑形时间几乎不存在。",
                "左路采用“前后夹击”，右路采用“高低配合”，在最短时间内建立主压与高位支援分工，并保持 5 s 稳定火控门限而不是继续拉长等待。",
                "68.8 s 即出现首次发射门通过，若把稳定门限延长至 10 s 或 20 s，则首轮拦截窗口将继续后移，难以对低空突防形成及时拦截。",
            ],
            [
                "131.8 s",
                "机动决策层",
                "左路已进入 TR-DOR 脱离准备段，最近敌我距离约 57.1 km，目标已处于高风险区，累计发射门通过事件数为 5，累计接力制导成功事件数为 10。",
                "左右两路同时切入“战术规避”模板，先在模板层中断继续攻击的动作序列。",
                "这一步发生在状态层正式规避之前，说明系统并不是等到完全进入规避状态后才响应，而是先在机动层打断攻击链，为生存动作让路。",
            ],
            [
                "136.2 s",
                "阶段决策层",
                "导弹来袭信号明确，最近敌我距离继续压缩至 55.9 km；左路首次发射窗口已收缩到 LR=56.2 km、TR=50.2 km，生存边界进一步收缩到 DOR=45.2 km、DR=41.2 km、MAR=39.7 km。",
                "系统由交战正式切换为规避，执行强制止压。",
                "从 131.8 s 的模板打断到 136.2 s 的状态规避，仅有 4.4 s 过渡，说明高压场景下决策链的关键不是继续争取一次额外发射，而是抢在生存边界被耗尽前止压。",
            ],
            [
                "255.8-265.8 s",
                "阶段决策层 / 战术决策层",
                "规避后累计敌机击落数为 2、我方损失数为 1，最近敌我距离回升到 62.6 km 左右，但右路已难以恢复完整双路结构。",
                "255.8 s 先进入“截获重组”，265.8 s 左路恢复“前后夹击”并回到 LR-TR 窗口附近，尝试以剩余兵力继续守区。",
                "这 10 s 不是机械等待，而是剩余兵力完成最小重组所需的过渡时间；继续等待 20 s 不会恢复右路能力，只会进一步放大高风险区突入。最终样本形成 2:2 结果，说明回转后能力恢复有限，生存优先虽必要，但代价真实存在。",
            ],
        ]
    )
    return rows


def lane_example_table(sample: SampleData, target_times: List[float]) -> List[List[object]]:
    return []


def scenario_decision_windows(scenario_id: str) -> List[float]:
    if scenario_id == "S1":
        return [95.4, 128.4, 193.0, 274.8, 324.4, 447.0]
    if scenario_id == "S2":
        return [86.2, 140.0, 236.2, 279.0, 390.2, 400.2, 416.8]
    return [5.0, 68.8, 131.8, 136.2, 255.8, 265.8]


def decision_analysis_paragraphs(scenario_id: str, sample: SampleData) -> List[str]:
    paragraphs: List[str] = []
    if scenario_id == "S1":
        paragraphs.append(
            "（1）95.4 s 至 128.4 s 的核心任务不是攻击，而是塑形。95.4 s 时左右两路仍处于远距接敌段，稳定交战就绪目标数为 0，说明双机虽已占位，但尚未形成可持续火控。到 128.4 s，稳定交战就绪目标数升至 2，左路切换为“前后夹击”、右路切换为“钳形夹击”，其作用是把两条攻击通道分化为主压与支撑，而不是借着刚出现稳定跟踪就立即开火。直到 274.8 s 才出现首次发射门通过，证明这一阶段的模板切换确实服务于几何塑形，而不是把攻击时机前置。"
        )
        paragraphs.append(
            "（2）193.0 s 至 274.8 s 的分析重点是为什么没有更早放行。193.0 s 时左路已经进入 MELD-MTR 塑形段，最近敌我距离约 137.6 km，稳定交战就绪目标数为 2，说明火控条件已初步具备；但动态首次发射窗口仍要求进入 LR/TR 区间后才能释放火力，因此系统继续保持“前后夹击”和“钳形夹击”，等待距离和站位进一步收敛。到 274.8 s，稳定交战就绪目标数升至 3，首次发射门通过才真正出现。这说明参数层把 5 s 作为稳定火控的必要门限，而不是充分门限；是否允许发射，还要叠加控制距离窗口和友机安全约束。因此，这里的关键不是把门限改成 10 s 或 20 s，而是要证明 5 s 之后系统仍然会继续筛选，直到攻击时机成熟。"
        )
        paragraphs.append(
            "（3）324.4 s 的规避触发回答了为什么此时必须中断攻击。该时刻最近敌我距离已经压缩到 59.8 km，累计发射门通过事件数为 6，累计接力制导成功事件数为 15，说明首轮杀伤链已经启动；与此同时，左路生存边界已收缩到 DOR=52.2 km、DR=48.2 km、MAR=38.9 km。若继续追压，新增收益只可能是争取更近的压制位置，而代价是进一步侵蚀规避余度，因此系统在导弹来袭后立即切入规避。这里的有效性不在于“规避动作发生了”，而在于规避发生时，首轮链条已经得到兑现，说明中断攻击并没有浪费前序决策成果。"
        )
        paragraphs.append(
            "（4）447.0 s 的回转恢复回答了为什么不是更早回转。324.4 s 进入规避后，到 447.0 s 最近敌我距离已恢复到 79.7 km，累计敌机击落数升至 2，累计接力制导成功事件数升至 26，说明规避阶段不仅拉开了安全距离，也让已发射导弹的中制导链继续兑现。如果在导弹刚来袭后就立即回转，则既不能保证前序导弹效能，也会把双机重新送回压缩边界；若继续等待更久，则会丢失后续清场窗口。最终代表样本形成 4:0 战果且高风险区累计滞留为 0 s，证明这一回转时机兼顾了生存与清场。"
        )
        return paragraphs
    if scenario_id == "S2":
        paragraphs.append(
            "（1）86.2 s 至 140.0 s 的关键不是谁先更快进入攻击窗口，而是在信息条件即将恶化前把压制骨架搭稳。左路先进入 NLT-MELD 压缩段而右路仍留在外层，说明系统允许双路不同步推进；此时左右两路都采用“并列压制”，原因在于高低混合来袭条件下，保持横向覆盖比复杂穿插更重要。换言之，场景二首先验证的是在信息缺口到来前，系统能否保持双路模板不散。"
        )
        paragraphs.append(
            "（2）236.2 s 的首次放行说明信息缺口并没有使发射门松动。该时刻预警图景已经不可用，但左路已进入 LR-TR 首次发射窗口段，稳定交战就绪目标数达到 3，最近敌我距离约 81.2 km，系统才允许第一次放行。如果在更早的 140.0 s 放行，则既没有进入首次发射窗口，稳定火控目标数也不足；因此这里验证的是，外层信息缺口不会自动转化为参数层的放松。"
        )
        paragraphs.append(
            "（3）279.0 s 的规避触发解释了为什么持续压制必须在此中断。此时累计发射门通过事件数已达 6，累计接力制导成功事件数达 16，说明首轮压制已经取得实际收益；但最近敌我距离已压缩到 55.2 km，左路同时逼近 DOR=49.5 km、DR=45.5 km、MAR=38.0 km 的生存边界。系统在导弹来袭后立即规避，表明持续压制不是无条件延续，而是以仍保有回转能力为前提。"
        )
        paragraphs.append(
            "（4）390.2 s 至 400.2 s 给出了“为什么是 10 s 而不是 20 s”的实际例证。279.0 s 进入规避后，系统并没有在威胁稍有下降时就立即回转，而是先在 390.2 s 进入“截获重组”，再经过 10 s 于 400.2 s 恢复“交战”。这 10 s 的作用是完成最小重组和重新分配，而不是单纯等待。如果在 390.2 s 直接恢复交战，双路结构尚未稳定；如果再拖延 20 s，则 416.8 s 的第二次导弹来袭会把这段等待全部吞掉，却并不会带来更大的生存收益。"
        )
        paragraphs.append(
            "（5）416.8 s 的第二次规避说明回转并非固定脚本。恢复交战仅 16.6 s 后，最近敌我距离又压缩到 50.2 km，系统尚未新增击落结果，但已再次触及生存边界，于是重新切回规避。代表样本最终形成 3:0 战果且高风险区累计滞留为 0 s，说明场景二真正验证成功的不是“始终压住不动”，而是“在信息缺口和导弹威胁交替出现时，仍能按规避—重组—再压制节奏稳定运转”。"
        )
        return paragraphs
    paragraphs.append(
        "（1）5.0 s 至 68.8 s 回答了为什么 5 s 稳定火控门限在高压场景中是必要的最小值。代表样本在 5.0 s 即形成首次稳定交战就绪，说明控制距离节点已整体前移；到 68.8 s 出现首次发射门通过时，稳定交战就绪目标数已经升至 4。如果把门限延长到 10 s 或 20 s，首轮发射窗口会继续向后推，而低空突防目标并不会因此降低威胁，所以这里的参数选择本质上是在“火控可靠性”和“窗口不被错失”之间取最小可行平衡。"
    )
    paragraphs.append(
        "（2）131.8 s 的模板打断是场景三最关键的决策证据。此时左右两路同时切入“战术规避”模板，最近敌我距离约 57.1 km，目标已经进入高风险区，累计发射门通过事件数为 5，累计接力制导成功事件数为 10。也就是说，首轮攻击利益已经兑现，而生存边界正迅速收缩，因此机动决策层首先中断继续攻击的动作序列，优先为生存动作让路。"
    )
    paragraphs.append(
        "（3）136.2 s 的正式规避进一步说明阶段决策层和机动决策层并不是一层完成全部判断。131.8 s 时只是模板层先打断，136.2 s 才在导弹来袭信号明确后由状态层正式切换为规避。两者间仅 4.4 s 的过渡，说明高压场景下真正要抢的是止压时机，而不是多争取一次近距压制机会。"
    )
    paragraphs.append(
        "（4）255.8 s 至 265.8 s 的回转恢复说明高压场景下“10 s 不是 20 s”的含义与场景二不同。255.8 s 时系统先进入“截获重组”，当时累计敌机击落数为 2、我方损失数为 1，说明虽然首轮拦截成功，但兵力已经受损；265.8 s 左路恢复“前后夹击”并回到 LR-TR 附近，是剩余兵力能够完成的最小恢复。若继续等待 20 s，并不会让右路重新具备完整能力，只会让高风险区持续暴露时间更长。最终样本形成 2:2 结果，说明场景三要验证的不是是否更早开火，而是止压后能否用有限兵力守住责任区。"
    )
    return paragraphs


def scenario_conclusion_text(scenario_id: str, loaded: Dict[str, SampleData]) -> str:
    rows = stability_summary_table(scenario_id, loaded)
    if scenario_id == "S1":
        return (
            "场景一证明，在低风险正面对进条件下，分层战术决策方法可以形成“接敌塑形、稳定跟踪、首轮放行、规避回转、继续清场”的完整闭环。"
            "多数样本保持 4:0 结果且高风险区累计滞留时间为 0 s，说明控制距离推进和规避回转时机总体合理；"
            "但 001313 样本出现 2 架我机损失，也说明即便在低风险场景下，后段执行仍受局部几何波动影响。"
        )
    if scenario_id == "S2":
        return (
            "场景二证明，该方法在外层信息存在持续缺口时，依然能够维持战术骨架、压制方向和接力制导链。"
            "代表样本取得 3:0 战果且无高风险区突入，说明“规避后回转再压制”的链条是成立的；"
            "但 001353 样本出现长时间高风险区滞留，说明中风险持续压制场景对参数层止压与回收时机尤为敏感。"
        )
    return (
        "场景三证明，高压近距条件下系统确实能够把交战链整体前移，并在导弹来袭时及时切入生存优先逻辑。"
        "但代表样本与补充样本均未形成完全清场，且高风险区累计滞留时间明显上升，说明该场景的真正难点不在于是否更早开火，而在于规避后剩余兵力如何继续守区。"
    )


def build_report() -> str:
    loaded = all_samples_loaded()
    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# 第六章仿真验证文档")
    lines.append("")
    lines.append("## 6.3 仿真验证")
    lines.append("")
    lines.append(
        "本节选取指定样本开展仿真验证，并按三个任务场景展开："
        "场景一验证低风险正面对进条件下的完整决策闭环，场景二验证信息缺口条件下的持续压制能力，场景三验证低空突防条件下的生存优先与止压能力。"
    )
    lines.append(
        "各场景均采用“代表样本详析 + 其余样本稳定性支撑”的写法，重点说明控制距离节点如何推动模板切换、导弹来袭何时触发规避、规避后何时回转，以及参数层为何采用当前门限。"
    )
    lines.append("")
    lines.append("### 6.3.1 验证对象与术语说明")
    lines.append("")
    lines.append("表6-1 指定验证样本")
    sample_rows = []
    for scenario_id in ["S1", "S2", "S3"]:
        for spec in scenario_samples(scenario_id):
            sample_rows.append(
                [
                    scenario_id,
                    spec.sample_label,
                    "代表详析样本" if spec.representative else "稳定性支撑样本",
                    spec.note,
                ]
            )
    lines.append(md_table(["场景", "样本编号", "使用方式", "说明"], sample_rows))
    lines.append("")
    lines.append("表6-2 本节术语说明")
    lines.append(md_table(["术语", "含义"], TERM_TABLE))
    lines.append("")
    lines.append("表6-3 决策层输入、输出与验证判据")
    lines.append(md_table(["决策层级", "主要输入", "主要输出", "验证判据"], decision_layer_table()))
    lines.append("")
    lines.append(
        "控制距离节点缩写含义如下：NLT 为不晚于截获距离，MELD 为混战进入距离，MTR 为机动跟踪距离，LR 为首次发射准备距离，TR 为交战转换距离，DOR 为脱离决断距离，DR 为重组决断距离，MAR 为最小规避距离。"
    )
    lines.append("")
    lines.append(
        "其中，稳定交战就绪门限由任务初始化参数直接给出，当前统一设置为 5.0 s，最小跟踪质量门限为 0.45，跟踪宽限时间为 2.5 s。"
        "因此，文中凡提及“达到稳定交战就绪”均表示目标已持续满足该门限，而不是临时出现一次可见即判定为就绪。"
    )
    lines.append("")

    figure_counter = 1
    table_counter = 4

    for scenario_id in ["S1", "S2", "S3"]:
        meta = SCENARIO_META[scenario_id]
        rep = loaded[f"{scenario_id}-{meta['representative_key']}"]
        figure_texts = figure_text_map(scenario_id)
        section_no = {"S1": "6.3.2", "S2": "6.3.3", "S3": "6.3.4"}[scenario_id]
        lines.append(f"### {section_no} {meta['title']}")
        lines.append("")

        lines.append("#### （1）场景目标与初始状态")
        lines.append("")
        lines.append(scenario_intro_text(scenario_id))
        lines.append(initial_state_text(scenario_id))
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}初始状态")
        lines.append(md_table(["飞机编号", "阵营", "机型", "初始航向", "初始高度", "初始速度", "挂载导弹数"], initial_state_table(scenario_id)))
        table_counter += 1
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}场景设定与分析重点")
        lines.append(md_table(["项目", "内容"], scene_setup_table(scenario_id)))
        table_counter += 1
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}样本选取说明")
        lines.append(md_table(["样本编号", "用途", "说明"], sample_selection_table(scenario_id)))
        table_counter += 1
        lines.append("")

        lines.append("#### （2）总体态势与样本稳定性")
        lines.append("")
        lines.append(overview_text(scenario_id, rep))
        lines.append("")

        overview_rel = copy_image(rep.spec.sample_dir / "overview.png", f"{scenario_id}_overview.png")
        lines.append(f"图6-{figure_counter} {meta['title']}代表样本总体态势图")
        lines.append(f"![图6-{figure_counter} {meta['title']}代表样本总体态势图]({overview_rel})")
        lines.append(figure_texts["overview"])
        figure_counter += 1
        lines.append("")

        zone_rel = plot_zone_entry_timeline(rep, f"{scenario_id}_zone_timeline.png")
        lines.append(f"图6-{figure_counter} {meta['title']}敌机风险区进入时序图")
        lines.append(f"![图6-{figure_counter} {meta['title']}敌机风险区进入时序图]({zone_rel})")
        lines.append(
            "该图以敌机编号为纵轴、时间为横轴，直接标示每一架敌机进入低风险区、中风险区和高风险区的持续时段。"
            "它用于回答“敌方在什么时间点进入了我方哪一层风险区”这一问题，也是判断规避是否及时的直接证据。"
        )
        figure_counter += 1
        lines.append("")

        stability_rel = plot_sample_stability_table([loaded[f"{scenario_id}-{spec.sample_tag}"] for spec in scenario_samples(scenario_id)], f"{scenario_id}_sample_stability.png")
        lines.append(f"图6-{figure_counter} {meta['title']}样本稳定性对比图")
        lines.append(f"![图6-{figure_counter} {meta['title']}样本稳定性对比图]({stability_rel})")
        lines.append(
            "该图用于横向比较指定样本的首次稳定交战就绪、首次发射门通过和首次接力制导成功三个关键里程碑时刻。"
            "它服务于判断代表样本的节奏是否具有代表性，以及其他样本是在哪个链条环节出现提前、滞后或断裂。"
        )
        figure_counter += 1
        lines.append("")

        lines.append(f"表6-{table_counter} {meta['title']}样本稳定性摘要")
        lines.append(
            md_table(
                ["样本编号", "首次稳定交战就绪", "首次发射门通过", "首次接力成功", "高风险区累计滞留", "敌机击落 / 我方损失"],
                stability_summary_table(scenario_id, loaded),
            )
        )
        table_counter += 1
        lines.append("")

        lines.append("#### （3）场景重点指标与代表样本关键节点")
        lines.append("")
        lines.append(scenario_focus_intro(scenario_id))
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}重点指标")
        lines.append(md_table(["指标项", "数值", "分析用途"], focus_metrics_table(scenario_id, rep)))
        table_counter += 1
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}代表样本关键时间节点")
        lines.append(md_table(["节点", "时间"], milestone_table(rep)))
        table_counter += 1
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}代表样本控制距离节点")
        lines.append(md_table(["航路", "控制距离段", "首次进入时间", "进入距离", "节点含义"], control_phase_table_cn(rep)))
        table_counter += 1
        lines.append("")

        control_rel = plot_control_distance_single(rep, f"{scenario_id}_control_distance.png")
        lines.append(f"图6-{figure_counter} {meta['title']}控制距离节点推进图")
        lines.append(f"![图6-{figure_counter} {meta['title']}控制距离节点推进图]({control_rel})")
        lines.append(figure_texts["control"])
        figure_counter += 1
        lines.append("")

        chain_rel = plot_engagement_chain_single(rep, f"{scenario_id}_engagement_chain.png")
        lines.append(f"图6-{figure_counter} {meta['title']}交战链展开图")
        lines.append(f"![图6-{figure_counter} {meta['title']}交战链展开图]({chain_rel})")
        lines.append(figure_texts["chain"])
        figure_counter += 1
        lines.append("")

        lines.append("#### （4）代表样本决策方法与有效性分析")
        lines.append("")
        lines.append("本节从决策输入、决策输出、执行时机和结果验证四个方面，对代表样本展开分析。")
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}代表样本关键决策窗口")
        lines.append(
            md_table(
                ["时间窗", "主导决策层级", "决策输入", "决策输出", "有效性验证"],
                decision_window_rows(scenario_id, rep),
            )
        )
        table_counter += 1
        lines.append("")
        for paragraph in decision_analysis_paragraphs(scenario_id, rep):
            lines.append(paragraph)
            lines.append("")

        lines.append("综合上述关键时段可见，稳定交战就绪门限、发射窗口、生存边界和模板动作并不是彼此独立的静态规则，而是在控制距离推进过程中共同起作用。稳定交战就绪门限负责保证火控质量，控制距离窗口负责确定何时允许攻击，生存边界负责决定何时必须中断攻击，模板动作映射负责把“攻击”“规避”“回转”落实为可执行的机动参数。")
        lines.append("")
        lines.append("因此，文中所讨论的“为什么此时规避”“为什么此时回转”“为什么门限取 5 s 或 10 s 而不是更长”都不能脱离具体时间窗单独回答，而必须回到同一套判据：当前是否仍处在可攻击窗口内，是否还满足稳定火控门限，是否已经逼近 DOR、DR、MAR 等生存边界，回转后是否还能重新建立交战几何。只有把这些条件放在同一时段内联合考察，才能判定决策是否有效。")
        lines.append("")

        safety_rel = plot_safety_single(rep, f"{scenario_id}_safety_envelope.png")
        lines.append(f"图6-{figure_counter} {meta['title']}飞行安全包线图")
        lines.append(f"![图6-{figure_counter} {meta['title']}飞行安全包线图]({safety_rel})")
        lines.append(figure_texts["safety"])
        figure_counter += 1
        lines.append("")

        lines.append("#### （5）场景结论")
        lines.append("")
        lines.append(scenario_conclusion_text(scenario_id, loaded))
        lines.append("")

    lines.append("### 6.3.5 三场景横向归纳")
    lines.append("")
    lines.append(f"表6-{table_counter} 三场景横向归纳")
    cross_rows = [
        ["场景一", "完整链路闭环", "验证模板切换、首次放行、规避回转和最终清场是否构成完整闭环。", "多数样本可实现 4:0，且高风险区累计滞留为 0 s。", "低风险场景下仍可能因后段执行波动出现我机损失。"],
        ["场景二", "信息缺口下持续压制", "验证预警链路受扰后，系统是否仍能恢复压制链并长期维持。", "代表样本在预警失配 400 s 条件下形成 3:0 战果。", "中风险场景对止压和回收时机更敏感，极端样本会出现长时间高风险区滞留。"],
        ["场景三", "高压近距止压回收", "验证首轮交战链前移后，系统是否能及时规避并守住生存边界。", "首次稳定交战就绪时间压缩至 5.0 s，首轮放行明显前移。", "高风险样本中回转后剩余兵力的持续守区能力仍偏弱。"],
    ]
    lines.append(md_table(["场景", "验证重点", "主要验证问题", "直接证据", "边界位置"], cross_rows))
    lines.append("")
    lines.append(
        "综合三类场景可见，本章方法的有效性并不体现在单一指标的绝对最优，而体现在不同压力结构下都能给出有因果依据的模板选择、规避触发和参数回收。"
        "低风险场景说明完整闭环成立，中风险场景说明规避后可恢复持续压制，高风险场景说明系统具备生存优先的止压能力。"
        "因此，本章更适合得出的结论是：分层战术决策方法已经具备可解释的实战链条，但高风险残局中的持续守区和后段回收策略仍需继续优化。"
    )
    lines.append("")
    return "\n".join(lines)


FINAL_SAMPLE_TAGS = {
    "S1": "001401",
    "S2": "225438",
    "S3": "001339",
}

FRIENDLY_IDS = ["A0100", "A0200", "A0300", "A0400"]
ENEMY_IDS = ["B0100", "B0200", "B0300", "B0400"]

STATE_COLORS = {
    "INTERCEPT": "#577590",
    "ENGAGE": "#d1495b",
    "EVADE": "#264653",
    "PATROL": "#4d908e",
    "UNKNOWN": "#bdbdbd",
}

TACTIC_COLORS = {
    "DRAG_SHOOT": "#577590",
    "PINCER_ATTACK": "#f8961e",
    "FRONT_BACK": "#d1495b",
    "HIGH_LOW": "#6a994e",
    "SIDE_BY_SIDE": "#277da1",
    "TACTICAL_EVASION": "#264653",
    "UNKNOWN": "#bdbdbd",
}

PHASE_COLORS = {
    "BEYOND_NLT": "#adb5bd",
    "NLT_MELD": "#90be6d",
    "MELD_MTR": "#43aa8b",
    "MTR_LR": "#4d908e",
    "LR_TR": "#277da1",
    "TR_DOR": "#577590",
    "DOR_DR": "#f8961e",
    "DR_MAR": "#d1495b",
    "BEYOND_MAR": "#6c757d",
}

SCENARIO_EVENT_MARKERS: Dict[str, List[Tuple[float, str]]] = {
    "S1": [(99.6, "转入交战"), (266.6, "首次放行"), (322.0, "导弹来袭规避"), (456.6, "恢复交战")],
    "S2": [(50.0, "信息缺口下转入交战"), (236.2, "首次放行"), (279.0, "首次规避"), (400.2, "恢复交战"), (416.8, "再次规避")],
    "S3": [(0.2, "高压开局"), (64.0, "首次放行"), (131.8, "首次规避"), (271.4, "恢复交战"), (1056.2, "后段规避")],
}

_ACMI_TRACK_CACHE: Dict[str, Dict[str, pd.DataFrame]] = {}
_ACMI_INITIAL_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {}


def final_sample_spec(scenario_id: str) -> SampleSpec:
    return find_sample(FINAL_SAMPLE_TAGS[scenario_id], scenario_id)


def final_sample_data(loaded: Dict[str, SampleData], scenario_id: str) -> SampleData:
    return loaded[f"{scenario_id}-{FINAL_SAMPLE_TAGS[scenario_id]}"]


def sample_acmi_path(sample: SampleData) -> Path:
    files = sorted(sample.spec.sample_dir.glob("*.txt.acmi"))
    if not files:
        raise FileNotFoundError(f"{sample.spec.sample_dir} 下未找到 ACMI 文件")
    return files[0]


def parse_acmi_initial_snapshot(sample: SampleData) -> Dict[str, Dict[str, float]]:
    cache_key = str(sample.spec.sample_dir)
    if cache_key in _ACMI_INITIAL_CACHE:
        return _ACMI_INITIAL_CACHE[cache_key]
    initial: Dict[str, Dict[str, float]] = {}
    wanted = set(FRIENDLY_IDS + ENEMY_IDS)
    current_time: Optional[float] = None
    for line in sample_acmi_path(sample).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("#"):
            if current_time is None:
                current_time = safe_float(line[1:])
            elif initial:
                break
            continue
        if not current_time or "," not in line:
            continue
        ident, rest = line.split(",", 1)
        if ident not in wanted or ident in initial:
            continue
        matched = re.search(r"T=([^,]+)", rest)
        if not matched:
            continue
        parts = matched.group(1).split("|")
        if len(parts) < 6:
            continue
        name_matched = re.search(r"Name=([^,]+)", rest)
        initial[ident] = {
            "lon": safe_float(parts[0]),
            "lat": safe_float(parts[1]),
            "alt_m": safe_float(parts[2]),
            "heading_deg": safe_float(parts[5]),
            "name": safe_text(name_matched.group(1) if name_matched else ("SU27" if ident.startswith("A") else "F16")),
        }
        if len(initial) == len(wanted):
            break
    _ACMI_INITIAL_CACHE[cache_key] = initial
    return initial


def parse_acmi_tracks(sample: SampleData) -> Dict[str, pd.DataFrame]:
    cache_key = str(sample.spec.sample_dir)
    if cache_key in _ACMI_TRACK_CACHE:
        return _ACMI_TRACK_CACHE[cache_key]
    records: Dict[str, List[Tuple[float, float, float, float]]] = {ident: [] for ident in FRIENDLY_IDS + ENEMY_IDS}
    current_time = float("nan")
    for raw_line in sample_acmi_path(sample).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_time = safe_float(line[1:])
            continue
        if not math.isfinite(current_time) or "," not in line:
            continue
        ident, rest = line.split(",", 1)
        if ident not in records:
            continue
        matched = re.search(r"T=([^,]+)", rest)
        if not matched:
            continue
        parts = matched.group(1).split("|")
        if len(parts) < 3:
            continue
        records[ident].append((current_time, safe_float(parts[0]), safe_float(parts[1]), safe_float(parts[2])))
    result: Dict[str, pd.DataFrame] = {}
    for ident, rows in records.items():
        result[ident] = pd.DataFrame(rows, columns=["time_s", "lon", "lat", "alt_m"])
    _ACMI_TRACK_CACHE[cache_key] = result
    return result


def project_track_frame(tracks: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    initial_lon = [safe_float(df.iloc[0]["lon"]) for ident, df in tracks.items() if ident in FRIENDLY_IDS and not df.empty]
    initial_lat = [safe_float(df.iloc[0]["lat"]) for ident, df in tracks.items() if ident in FRIENDLY_IDS and not df.empty]
    lon0 = sum(initial_lon) / len(initial_lon)
    lat0 = sum(initial_lat) / len(initial_lat)
    cos_lat = math.cos(math.radians(lat0))
    projected: Dict[str, pd.DataFrame] = {}
    for ident, df in tracks.items():
        if df.empty:
            projected[ident] = df.copy()
            continue
        frame = df.copy()
        frame["x_km"] = (frame["lon"] - lon0) * 111.32 * cos_lat
        frame["y_km"] = (frame["lat"] - lat0) * 111.32
        projected[ident] = frame
    return projected


def track_point_at_time(track: pd.DataFrame, time_s: float) -> Optional[pd.Series]:
    if track.empty:
        return None
    idx = (track["time_s"] - time_s).abs().idxmin()
    return track.loc[idx]


def categorical_segments(df: pd.DataFrame, column: str) -> List[Tuple[float, float, str]]:
    if df.empty:
        return []
    values = [safe_text(value, "UNKNOWN") for value in df[column]]
    times = df["time_s"].tolist()
    segments: List[Tuple[float, float, str]] = []
    start_time = safe_float(times[0])
    current_value = values[0]
    for idx in range(1, len(values)):
        if values[idx] != current_value:
            end_time = safe_float(times[idx])
            segments.append((start_time, end_time, current_value))
            start_time = end_time
            current_value = values[idx]
    segments.append((start_time, safe_float(times[-1]), current_value))
    return segments


def color_for_state(raw: str) -> str:
    return STATE_COLORS.get(raw.upper(), "#bdbdbd")


def color_for_tactic(raw: str) -> str:
    return TACTIC_COLORS.get(raw.upper(), "#bdbdbd")


def color_for_phase(raw: str) -> str:
    return PHASE_COLORS.get(phase_code(raw), "#bdbdbd")


def plot_spatial_trajectory_single(sample: SampleData, scenario_id: str, dst_name: str) -> str:
    tracks = project_track_frame(parse_acmi_tracks(sample))
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.set_facecolor("#fbf8f2")
    friendly_colors = {"A0100": "#b11226", "A0200": "#d1495b", "A0300": "#8b1e3f", "A0400": "#f9844a"}
    enemy_colors = {"B0100": "#1d3557", "B0200": "#457b9d", "B0300": "#2a9d8f", "B0400": "#4cc9f0"}
    for ident in FRIENDLY_IDS:
        df = tracks.get(ident)
        if df is None or df.empty:
            continue
        thin = df.iloc[::8].copy()
        ax.plot(thin["x_km"], thin["y_km"], color=friendly_colors[ident], linewidth=2.2, label=f"{ident} 我方")
        ax.scatter(df.iloc[0]["x_km"], df.iloc[0]["y_km"], color="white", edgecolor=friendly_colors[ident], s=52, zorder=4)
        ax.scatter(df.iloc[-1]["x_km"], df.iloc[-1]["y_km"], color=friendly_colors[ident], s=36, marker="^", zorder=4)
    for ident in ENEMY_IDS:
        df = tracks.get(ident)
        if df is None or df.empty:
            continue
        thin = df.iloc[::8].copy()
        ax.plot(thin["x_km"], thin["y_km"], color=enemy_colors[ident], linewidth=2.0, linestyle="--", label=f"{ident} 敌方")
        ax.scatter(df.iloc[0]["x_km"], df.iloc[0]["y_km"], color="white", edgecolor=enemy_colors[ident], s=44, zorder=4)
        ax.scatter(df.iloc[-1]["x_km"], df.iloc[-1]["y_km"], color=enemy_colors[ident], s=34, marker="s", zorder=4)
    for time_s, label in SCENARIO_EVENT_MARKERS[scenario_id]:
        pts = []
        for ident in FRIENDLY_IDS:
            row = track_point_at_time(tracks.get(ident, pd.DataFrame()), time_s)
            if row is not None:
                pts.append((safe_float(row["x_km"]), safe_float(row["y_km"])))
        if not pts:
            continue
        x = sum(item[0] for item in pts) / len(pts)
        y = sum(item[1] for item in pts) / len(pts)
        ax.scatter([x], [y], color="#1f1f1f", s=28, zorder=5)
        ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9, color="#1f1f1f")
    ax.set_xlabel("东西向相对位移 / km")
    ax.set_ylabel("南北向相对位移 / km")
    ax.set_title("敌我航迹与关键决策节点空间分布")
    ax.grid(color="#d0d0d0", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path = ASSET_DIR / dst_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return f"./{ASSET_DIR.name}/{out_path.name}"


def plot_state_tactic_timeline_single(sample: SampleData, scenario_id: str, dst_name: str) -> str:
    lanes = [
        ("任务状态", "cap_state", state_display, color_for_state),
        ("左路模板", "left_tactic", tactic_display, color_for_tactic),
        ("右路模板", "right_tactic", tactic_display, color_for_tactic),
        ("左路阶段", "left_phase", phase_display, color_for_phase),
        ("右路阶段", "right_phase", phase_display, color_for_phase),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.set_facecolor("#fbf8f2")
    lane_height = 0.76
    y_positions = {label: idx for idx, (label, _, _, _) in enumerate(reversed(lanes))}
    for label, column, display_func, color_func in lanes:
        y = y_positions[label]
        for start, end, raw_value in categorical_segments(sample.timeline, column):
            width = max(0.2, end - start)
            color = color_func(safe_text(raw_value, "UNKNOWN"))
            ax.add_patch(Rectangle((start, y - lane_height / 2), width, lane_height, facecolor=color, edgecolor="white", linewidth=0.9))
            show_text = display_func(raw_value)
            if width >= 24.0:
                ax.text(start + width / 2, y, show_text, ha="center", va="center", fontsize=8, color="white" if color not in {"#adb5bd", "#90be6d", "#f8961e"} else "#1f1f1f")
    for time_s, label in SCENARIO_EVENT_MARKERS[scenario_id]:
        ax.axvline(time_s, color="#6c757d", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.text(time_s, len(lanes) - 0.05, label, rotation=90, fontsize=8, va="top", ha="right", color="#495057")
    ax.set_xlim(0.0, safe_float(sample.timeline["time_s"].iloc[-1]))
    ax.set_ylim(-0.8, len(lanes) - 0.2)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlabel("时间 / s")
    ax.set_title("任务状态、战术模板与控制阶段时序")
    ax.grid(axis="x", color="#d0d0d0", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    out_path = ASSET_DIR / dst_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return f"./{ASSET_DIR.name}/{out_path.name}"


def decision_snapshot_series(sample: SampleData, side: str, field: str) -> pd.DataFrame:
    rows: List[Tuple[float, float]] = []
    column = f"{side}_decision_snapshot_json"
    for _, row in sample.decision_trace.iterrows():
        snapshot = parse_snapshot(row.get(column))
        value = safe_float(snapshot.get(field))
        if math.isfinite(value):
            rows.append((safe_float(row["time_s"]), value))
    return pd.DataFrame(rows, columns=["time_s", field])


def plot_control_distance_actual_single(sample: SampleData, scenario_id: str, dst_name: str) -> str:
    left_series = decision_snapshot_series(sample, "left", "shooter_distance_km")
    right_series = decision_snapshot_series(sample, "right", "shooter_distance_km")
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.set_facecolor("#fbf8f2")
    if not sample.timeline.empty:
        time_series = sample.timeline[["time_s", "nearest_enemy_to_friendly_km"]].dropna()
        ax.plot(time_series["time_s"], time_series["nearest_enemy_to_friendly_km"], color="#6c757d", linewidth=1.4, linestyle="--", label="全局最近敌我距离")
    if not left_series.empty:
        ax.plot(left_series["time_s"], left_series["shooter_distance_km"], color="#b11226", linewidth=2.3, label="左路配对距离")
    if not right_series.empty:
        ax.plot(right_series["time_s"], right_series["shooter_distance_km"], color="#277da1", linewidth=2.3, label="右路配对距离")
    for _, row in sample.control_nodes.iterrows():
        time_s = safe_float(row["first_time_s"])
        dist = safe_float(row["distance_km"])
        if not math.isfinite(time_s) or not math.isfinite(dist):
            continue
        route = safe_text(row["route"])
        color = "#b11226" if route == "left" else "#277da1"
        ax.scatter([time_s], [dist], color=color, s=28, zorder=4)
        ax.text(time_s + 4.0, dist + (3.5 if route == "left" else -4.0), phase_code(row["phase"]).replace("_", "-"), fontsize=8, color=color)
    for time_s, label in SCENARIO_EVENT_MARKERS[scenario_id]:
        ax.axvline(time_s, color="#adb5bd", linestyle=":", linewidth=0.8, alpha=0.9)
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("距离 / km")
    ax.set_title("左右两路实际配对距离与节点触发时刻")
    ax.grid(color="#d0d0d0", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper right")
    fig.tight_layout()
    out_path = ASSET_DIR / dst_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return f"./{ASSET_DIR.name}/{out_path.name}"


def plot_event_chain_single(sample: SampleData, scenario_id: str, dst_name: str) -> str:
    df = sample.timeline.copy()
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.set_facecolor("#fbf8f2")
    series = [
        ("稳定交战就绪目标数", "stable_ready_count", "#2a6f97"),
        ("发射门通过累计事件数", "gate_pass_unique_count", "#f8961e"),
        ("接力制导成功累计事件数", "relay_success_unique_count", "#43aa8b"),
        ("敌机击落数", "enemy_kill_count", "#d1495b"),
        ("我方损失数", "friendly_loss_count", "#6d597a"),
    ]
    for label, column, color in series:
        if column in df.columns:
            ax.step(df["time_s"], df[column], where="post", linewidth=2.0, color=color, label=label)
    for time_s, label in SCENARIO_EVENT_MARKERS[scenario_id]:
        ax.axvline(time_s, color="#adb5bd", linestyle=":", linewidth=0.8, alpha=0.9)
        ax.text(time_s, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0, label, rotation=90, fontsize=8, va="top", ha="right", color="#495057")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("累计数量")
    ax.set_title("交战链关键事件累计时序")
    ax.grid(color="#d0d0d0", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    out_path = ASSET_DIR / dst_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return f"./{ASSET_DIR.name}/{out_path.name}"


def plot_awacs_gap_single(sample: SampleData, dst_name: str) -> str:
    from matplotlib import transforms

    df = sample.timeline.copy()
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.set_facecolor("#fbf8f2")
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for start, end, raw_value in categorical_segments(df, "enemy_script_awacs_state"):
        text = safe_text(raw_value, "UNKNOWN").upper()
        color = "#d8f3dc" if "AVAILABLE" in text else "#ffd6a5"
        label = "预警可用" if "AVAILABLE" in text else "预警失配"
        ax.add_patch(Rectangle((start, 0.90), max(0.2, end - start), 0.08, transform=trans, facecolor=color, edgecolor="white", linewidth=0.8))
        if end - start >= 18.0:
            ax.text((start + end) / 2, 0.94, label, transform=trans, ha="center", va="center", fontsize=8, color="#1f1f1f")
    ax.plot(df["time_s"], df["nearest_enemy_to_friendly_km"], color="#1d3557", linewidth=2.2, label="最近敌我距离")
    ax.step(df["time_s"], df["stable_ready_count"] * 6 + 20, where="post", color="#43aa8b", linewidth=1.8, label="稳定交战就绪目标数（缩放示意）")
    for time_s, label in SCENARIO_EVENT_MARKERS["S2"]:
        row = timeline_row(sample, time_s)
        y = safe_float(row["nearest_enemy_to_friendly_km"]) if row is not None else float("nan")
        if math.isfinite(y):
            ax.scatter([time_s], [y], color="#d1495b", s=30, zorder=4)
            ax.annotate(label, (time_s, y), xytext=(6, 6), textcoords="offset points", fontsize=8, color="#d1495b")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("距离 / km")
    ax.set_title("预警可用性、稳定跟踪与距离压缩关系")
    ax.grid(color="#d0d0d0", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out_path = ASSET_DIR / dst_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return f"./{ASSET_DIR.name}/{out_path.name}"


def plot_high_risk_pressure_single(sample: SampleData, dst_name: str) -> str:
    df = sample.timeline.copy()
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.set_facecolor("#fbf8f2")
    ax.step(df["time_s"], df["truth_high_count"], where="post", color="#d1495b", linewidth=2.2, label="高风险区内敌机数")
    ax.fill_between(df["time_s"], 0, df["truth_high_count"], step="post", color="#d1495b", alpha=0.18)
    ax2 = ax.twinx()
    ax2.plot(df["time_s"], df["nearest_enemy_to_friendly_km"], color="#1d3557", linewidth=2.0, label="最近敌我距离")
    loss_rows = df[df["friendly_loss_count"].diff().fillna(df["friendly_loss_count"]) > 0]
    for _, row in loss_rows.iterrows():
        time_s = safe_float(row["time_s"])
        ax.axvline(time_s, color="#6d597a", linestyle="--", linewidth=1.0)
        ax2.annotate("我机损失", (time_s, safe_float(row["nearest_enemy_to_friendly_km"])), xytext=(6, 6), textcoords="offset points", fontsize=8, color="#6d597a")
    for time_s, label in SCENARIO_EVENT_MARKERS["S3"]:
        ax.axvline(time_s, color="#adb5bd", linestyle=":", linewidth=0.8)
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("高风险区敌机数 / 架")
    ax2.set_ylabel("最近敌我距离 / km")
    ax.set_title("高风险区压力、距离压缩与损失时刻")
    ax.grid(color="#d0d0d0", linestyle="--", linewidth=0.6, alpha=0.7)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out_path = ASSET_DIR / dst_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return f"./{ASSET_DIR.name}/{out_path.name}"


def initial_state_table_detailed(sample: SampleData, scenario_id: str) -> List[List[object]]:
    meta = SCENARIO_META[scenario_id]
    initial = parse_acmi_initial_snapshot(sample)
    rows: List[List[object]] = []
    for ident in FRIENDLY_IDS + ENEMY_IDS:
        info = initial.get(ident, {})
        is_friendly = ident.startswith("A")
        rows.append(
            [
                ident,
                "我方" if is_friendly else "敌方",
                safe_text(info.get("name"), "Su-27SK" if is_friendly else "F-16"),
                f"{safe_float(info.get('lon')):.4f}" if math.isfinite(safe_float(info.get("lon"))) else "未记录",
                f"{safe_float(info.get('lat')):.4f}" if math.isfinite(safe_float(info.get("lat"))) else "未记录",
                f"{safe_float(info.get('alt_m')):.0f}" if math.isfinite(safe_float(info.get("alt_m"))) else "未记录",
                f"{safe_float(info.get('heading_deg')):.1f}" if math.isfinite(safe_float(info.get("heading_deg"))) else "未记录",
                meta["friendly_speed_text"] if is_friendly else f"{safe_float(meta['enemy_speed_mps']):.0f} m/s",
                "4",
            ]
        )
    return rows


def sample_pool_overview_rows() -> List[List[object]]:
    return [
        ["场景一", "225438、225448、001313、001401", "001401", "优先保留零损失、零高风险突入、较大安全余度和较早清场时刻。"],
        ["场景二", "225438、001339、001353", "225438", "优先保留信息缺口下的零高风险突入、零损失和持续压制结果。"],
        ["场景三", "233513、001339", "001339", "场景重点为生存优先，不以击落数最大化为唯一目标，优先保留较晚损失时刻和较小高风险滞留。"],
    ]


def candidate_comparison_table_final(scenario_id: str, loaded: Dict[str, SampleData]) -> Tuple[List[str], List[List[object]]]:
    rows: List[List[object]] = []
    if scenario_id == "S1":
        headers = ["样本", "击落 / 损失", "首次放行", "接力制导成功率", "最小敌我距离", "高风险滞留", "首次敌机击落", "结论"]
        for spec in scenario_samples(scenario_id):
            sample = loaded[f"{scenario_id}-{spec.sample_tag}"]
            rows.append(
                [
                    spec.sample_label,
                    f"{format_count(metric(sample, 'enemy_kill_count'))} / {format_count(metric(sample, 'friendly_loss_count'))}",
                    format_seconds(metric(sample, "first_gate_pass_time_s")),
                    format_ratio(metric(sample, "relay_success_rate_unique")),
                    format_km(metric(sample, "min_enemy_to_friendly_km")),
                    format_seconds(metric(sample, "high_risk_breach_time_s")),
                    format_seconds(metric(sample, "first_enemy_kill_time_s")),
                    "入选" if spec.sample_tag == FINAL_SAMPLE_TAGS[scenario_id] else "未入选",
                ]
            )
        return headers, rows
    if scenario_id == "S2":
        headers = ["样本", "击落 / 损失", "预警失配累计", "首次放行", "接力制导成功率", "航迹连续性", "高风险滞留", "结论"]
        for spec in scenario_samples(scenario_id):
            sample = loaded[f"{scenario_id}-{spec.sample_tag}"]
            rows.append(
                [
                    spec.sample_label,
                    f"{format_count(metric(sample, 'enemy_kill_count'))} / {format_count(metric(sample, 'friendly_loss_count'))}",
                    format_seconds(metric(sample, "enemy_awacs_denied_time_s")),
                    format_seconds(metric(sample, "first_gate_pass_time_s")),
                    format_ratio(metric(sample, "relay_success_rate_unique")),
                    format_ratio(metric(sample, "radar_any_track_continuity_post_activation")),
                    format_seconds(metric(sample, "high_risk_breach_time_s")),
                    "入选" if spec.sample_tag == FINAL_SAMPLE_TAGS[scenario_id] else "未入选",
                ]
            )
        return headers, rows
    headers = ["样本", "击落 / 损失", "首次放行", "首次我机损失", "高风险滞留", "最小敌我距离", "意图识别准确率", "结论"]
    for spec in scenario_samples(scenario_id):
        sample = loaded[f"{scenario_id}-{spec.sample_tag}"]
        rows.append(
            [
                spec.sample_label,
                f"{format_count(metric(sample, 'enemy_kill_count'))} / {format_count(metric(sample, 'friendly_loss_count'))}",
                format_seconds(metric(sample, "first_gate_pass_time_s")),
                format_seconds(metric(sample, "first_friendly_loss_time_s")),
                format_seconds(metric(sample, "high_risk_breach_time_s")),
                format_km(metric(sample, "min_enemy_to_friendly_km")),
                format_ratio(metric(sample, "intent_truth_accuracy")),
                "入选" if spec.sample_tag == FINAL_SAMPLE_TAGS[scenario_id] else "未入选",
            ]
        )
    return headers, rows


def selected_sample_reason_text(scenario_id: str, loaded: Dict[str, SampleData]) -> str:
    if scenario_id == "S1":
        selected = final_sample_data(loaded, "S1")
        s225448 = loaded["S1-225448"]
        s225438 = loaded["S1-225438"]
        return (
            f"场景一的候选样本中，225438、225448 与 001401 均形成 4:0 且高风险区累计滞留时间为 0 s。"
            f"若仅看发射门通过率，225448 更高；但场景一验证的重点不是激进放行，而是低风险条件下的完整闭环与安全回转。"
            f"001401 的最小敌我距离达到 {format_km(metric(selected, 'min_enemy_to_friendly_km'))}，高于 225448 的 {format_km(metric(s225448, 'min_enemy_to_friendly_km'))} 和 225438 的 {format_km(metric(s225438, 'min_enemy_to_friendly_km'))}；"
            f"其首次敌机击落时间为 {format_seconds(metric(selected, 'first_enemy_kill_time_s'))}，也是同类零损失样本中最早。"
            f"因此，001401 更适合作为场景一的最终分析样本。"
        )
    if scenario_id == "S2":
        selected = final_sample_data(loaded, "S2")
        s1339 = loaded["S2-001339"]
        s1353 = loaded["S2-001353"]
        return (
            f"场景二要求在累计 {format_seconds(metric(selected, 'enemy_awacs_denied_time_s'))} 的预警失配条件下维持压制。"
            f"225438 是唯一同时满足 3:0 战果、我方零损失和高风险区累计滞留为 0 s 的候选样本。"
            f"001339 虽然接力制导成功率更高，但最终仍出现 1 架我机损失；001353 的高风险区累计滞留达到 {format_seconds(metric(s1353, 'high_risk_breach_time_s'))}。"
            f"因此，225438 最能体现“信息缺口下保持压制而不过度冒险”的验证目标。"
        )
    selected = final_sample_data(loaded, "S3")
    s233513 = loaded["S3-233513"]
    return (
        f"场景三的验证目标是生存优先，而不是单纯追求更多击落。"
        f"233513 虽然形成 2:2 结果，但高风险区累计滞留为 {format_seconds(metric(s233513, 'high_risk_breach_time_s'))}，首次我机损失时间仅为 {format_seconds(metric(s233513, 'first_friendly_loss_time_s'))}。"
        f"001339 将高风险区累计滞留压缩到 {format_seconds(metric(selected, 'high_risk_breach_time_s'))}，并把首次我机损失推迟到 {format_seconds(metric(selected, 'first_friendly_loss_time_s'))}；"
        f"虽然击落数降低为 1，但更符合低空突防样本中“止压、回转、守区”的验证目的，因此选为场景三的最终分析样本。"
    )


def scene_setup_table_final(scenario_id: str) -> List[List[object]]:
    meta = SCENARIO_META[scenario_id]
    return [
        ["任务目标", meta["task_focus"]],
        ["敌方初始距离", f"{safe_float(meta['initial_distance_km']):.1f} km"],
        ["敌方高度配置", " / ".join(f"{safe_float(item):.1f} km" for item in meta["enemy_altitudes_km"])],
        ["预警条件", safe_text(meta["awacs_mode"])],
        ["场景风险定位", safe_text(meta["expected_zone"])],
        ["正文入选样本", final_sample_spec(scenario_id).sample_label],
        ["本场景分析侧重点", safe_text(meta["scene_goal"])],
    ]


def focus_metrics_table_final(scenario_id: str, sample: SampleData) -> List[List[object]]:
    if scenario_id == "S1":
        return [
            ["机载雷达覆盖率（存活敌机、雷达激活后）", format_ratio(metric(sample, "radar_detection_coverage_ratio_alive_targets_post_activation")), "说明在低风险正面对进场景中，外层感知足以支撑双路模板切换和后续清场。"],
            ["任意航迹连续性（雷达激活后）", format_ratio(metric(sample, "radar_any_track_continuity_post_activation")), "说明控制距离推进期间目标航迹没有频繁断裂。"],
            ["首次稳定交战就绪时间", format_seconds(metric(sample, "first_stable_ready_time_s")), "用于判断协同探测链何时真正转化为稳定火控，而不是瞬时可见。"],
            ["首次发射门通过时间", format_seconds(metric(sample, "first_gate_pass_time_s")), "用于识别稳定跟踪与实际放行之间是否仍存在必要的几何筛选。"],
            ["接力制导成功率（去重）", format_ratio(metric(sample, "relay_success_rate_unique")), "用于验证规避前已发射导弹在后续回转阶段是否仍能保持中制导链。"],
            ["最小敌我距离", format_km(metric(sample, "min_enemy_to_friendly_km")), "用于验证首轮放行与回转没有把双机压入不可恢复近距缠斗。"],
            ["高风险区累计滞留时间", format_seconds(metric(sample, "high_risk_breach_time_s")), "低风险场景应尽量保持为 0 s，用于证明回转时机没有放任敌方突入最高风险区。"],
            ["敌机击落数 / 我方损失数", f"{format_count(metric(sample, 'enemy_kill_count'))} / {format_count(metric(sample, 'friendly_loss_count'))}", "从最终结果检验完整链路是否兑现为有效清场。"],
        ]
    if scenario_id == "S2":
        return [
            ["预警失配累计时间", format_seconds(metric(sample, "enemy_awacs_denied_time_s")), "说明本场景的外层信息缺口强度。"],
            ["机载雷达覆盖率（存活敌机、雷达激活后）", format_ratio(metric(sample, "radar_detection_coverage_ratio_alive_targets_post_activation")), "说明预警失配后机载感知仍能维持对目标群的覆盖。"],
            ["任意航迹连续性（雷达激活后）", format_ratio(metric(sample, "radar_any_track_continuity_post_activation")), "说明持续压制不是依赖偶然发现，而是依赖连续航迹。"],
            ["首次发射门通过时间", format_seconds(metric(sample, "first_gate_pass_time_s")), "用于验证信息缺口不会自动转化为提前放行。"],
            ["接力制导成功率（去重）", format_ratio(metric(sample, "relay_success_rate_unique")), "用于衡量规避与回转之间中制导链的保持能力。"],
            ["首次规避至重新交战总耗时", "121.2 s", "由 279.0 s 首次规避到 400.2 s 恢复交战计算，反映持续压制的恢复效率。"],
            ["截获重组到恢复交战缓冲", "10.0 s", "由 390.2 s 截获重组到 400.2 s 恢复交战计算，用于解释为何不是更早或更晚回转。"],
            ["敌机击落数 / 我方损失数", f"{format_count(metric(sample, 'enemy_kill_count'))} / {format_count(metric(sample, 'friendly_loss_count'))}", "用于从最终结果验证“规避—重组—再压制”链条是否成立。"],
        ]
    return [
        ["首次稳定交战就绪时间", format_seconds(metric(sample, "first_stable_ready_time_s")), "说明低空突防样本中控制距离链已整体前移。"],
        ["首次发射门通过时间", format_seconds(metric(sample, "first_gate_pass_time_s")), "用于判断首轮攻击是否足够前移，避免错过低空突防拦截窗口。"],
        ["首次规避至截获重组耗时", "129.6 s", "由 131.8 s 首次规避到 261.4 s 截获重组计算，反映止压后的回收时长。"],
        ["截获重组到恢复交战缓冲", "10.0 s", "由 261.4 s 截获重组到 271.4 s 恢复交战计算，用于说明为什么取 10 s 而不是更长等待。"],
        ["高风险区累计滞留时间", format_seconds(metric(sample, "high_risk_breach_time_s")), "用于衡量高压样本中系统被迫承受的最高风险持续时间。"],
        ["首次我机损失时间", format_seconds(metric(sample, "first_friendly_loss_time_s")), "用于衡量生存优先策略是否把损失尽量后推。"],
        ["最小敌我距离", format_km(metric(sample, "min_enemy_to_friendly_km")), "用于说明该样本已经逼近近距极限，为强制规避提供几何依据。"],
        ["意图识别准确率", format_ratio(metric(sample, "intent_truth_accuracy")), "用于说明低空突防场景中对敌意图判断本身就更难，必须与生存边界联合分析。"],
    ]


def control_node_table_final(sample: SampleData) -> List[List[object]]:
    rows: List[List[object]] = []
    for _, row in sample.control_nodes.iterrows():
        rows.append(
            [
                "左路" if safe_text(row["route"]) == "left" else "右路",
                phase_display(row["phase"]),
                format_seconds(safe_float(row["first_time_s"])),
                format_km(safe_float(row["distance_km"])),
                safe_text(row["description"]),
            ]
        )
    return rows


def decision_window_rows_final(scenario_id: str) -> List[List[object]]:
    if scenario_id == "S1":
        return [
            [
                "99.6-150.4 s",
                "预警图景可用，最近威胁由 200 km 级进入 160 km 级，阶段层由截获保持转入交战执行，但仍判定为低风险压缩阶段。",
                "左右两路先统一采用“拖曳射击”，随后左路于 148.0 s 切为“钳形夹击”，右路于 150.4 s 切为“前后夹击”，把双路从对称巡逻改为异步塑形。",
                "机动层并未提前抢动作，而是先完成目标占用与左右分工，保证后续放行时不会四机同向压缩。",
                "稳定交战就绪门限已经满足，但动态首次发射窗口仍在 LR/TR 之后，当前配对距离仍为 156 km 至 190 km，远高于发射窗口。",
                "这一时间窗证明场景一的第一步不是抢射，而是先把双路攻击通道搭稳；因此“为什么不是更早发射”的答案来自参数层和阶段层的联合约束，而不是单一门限。",
            ],
            [
                "266.6-295.0 s",
                "阶段层判定敌方已经压入中风险区，允许从塑形进入首轮攻击准备。",
                "左路保持“钳形夹击”，右路保持“前后夹击”，说明模板没有因首次稳定跟踪一出现就反复抖动。",
                "机动层继续维持压制几何，不在首枚导弹离架后立即拉脱，而是先保证前序攻击链成形。",
                "266.6 s 首次发射门通过，295.0 s 首次接力制导成功；这两个时刻之间相差 28.4 s，说明 5 s 稳定门限只是必要条件，真正放行仍依赖控制距离和照射链闭合。",
                "该窗口回答了“为什么是 5 s 而不是 10 s 或 20 s”：若把稳定门限继续拉长，首轮攻击将整体后移；而当前门限下系统仍然继续筛选到 266.6 s 才放行，因此并不存在‘一稳定就立刻发射’的问题。",
            ],
            [
                "322.0 s",
                "阶段层收到明确的导弹来袭告警，任务状态立即由交战切换为规避。",
                "左右两路此时都已经完成首轮攻击模板，未再切换到新的攻击模板，而是保持当前模板骨架等待规避打断。",
                "机动层在这一刻中断继续压近动作，优先释放规避机动空间，而不是再追求额外一轮近距压制。",
                "左路配对距离约为 54.1 km，而左路生存边界 DOR 为 55.6 km、DR 为 51.6 km，说明左路实际上已经跌破 DOR，仅比 DR 多出约 2.5 km；继续压近将直接侵蚀重组余度。",
                "因此，322.0 s 的规避不是因为‘距离看起来差不多了’才发生，而是因为导弹来袭与 DOR 被穿越同时出现，阶段层和参数层在同一时刻给出了强制打断条件。",
            ],
            [
                "446.6-456.6 s",
                "446.6 s 预警图景恢复，阶段层先把状态切回截获重组；456.6 s 敌方再次压入中风险区后，才恢复交战。",
                "恢复阶段没有马上改成新的激进模板，而是让右路在 456.6 s 重新接管 B0400，先把双路攻击职责补齐。",
                "机动层在这 10 s 内完成回转后的最小重组，防止规避后立刻把两路再次送入压缩边界。",
                "参数层没有简单按‘距离一回升就交战’处理，而是在 446.6 s 到 456.6 s 之间保留了 10 s 缓冲。此时累计接力制导成功事件已达 37，累计击落 3 架，距离也回到 97 km 至 100 km 级，已经具备安全回转的几何基础。",
                "这回答了“为什么是 10 s 而不是更早或更晚”：446.6 s 之前尚无恢复的外层图景，不能贸然回转；若在 456.6 s 之后继续等待 20 s，则会直接压缩最后 1 架敌机的清场窗口。",
            ],
        ]
    if scenario_id == "S2":
        return [
            [
                "50.0 s",
                "50.0 s 时阶段层转入交战，同时敌方预警图景状态切为 DENIED，说明本场景从一开始就进入‘信息缺口下执行’而非‘信息完整后再受扰’。",
                "战术层把左右两路统一为“并列压制”，优先保证横向覆盖，而不是一开始就采用穿插幅度更大的模板。",
                "机动层因此首先执行的是骨架维持，而不是速度更快但更容易断链的内切机动。",
                "参数层给出的发射窗口仍在 LR/TR 之后，当前左路和右路配对距离分别约 192.3 km 与 197.1 km，远未达到放行条件。",
                "这一窗口证明场景二的起点不是‘因为没有预警就放宽条件’，而是在预警缺口已经出现时仍先守住感知和模板骨架。",
            ],
            [
                "236.2 s",
                "外层预警仍不可用，但阶段层依然只在敌方进入中风险压缩后允许首轮放行。",
                "战术层仍保持双路“并列压制”，说明信息缺口并未导致模板反复抖动。",
                "机动层此时让左路进入 LR-TR 首次发射窗口，而右路仍处于 MTR-LR 准备段，体现双路不同步压制。",
                "参数层只在稳定交战就绪目标数达到 3、左路配对距离压缩到 78.9 km 后，于 236.2 s 给出首次发射门通过；由于预警图景仍不可用，这一放行完全依赖机载连续跟踪和控制距离，而不是外层信息补偿。",
                "因此，场景二回答了“信息缺口是否会导致放行门限松动”：不会，首轮放行仍然发生在稳定火控和距离窗口同时满足之后。",
            ],
            [
                "279.0 s",
                "阶段层在导弹来袭触发下立即切换为规避。",
                "左路已经由“并列压制”转入“前后夹击”，右路仍保持“并列压制”，说明系统规避前并没有把两路强行合并，而是保留原有分工。",
                "机动层在这一刻中断持续压制，而不是再争取一次短促补射。",
                "参数层给出的生存边界已经非常紧：左路配对距离 55.3 km，对应 DOR 49.5 km；右路配对距离 63.4 km，对应 DOR 55.1 km。左路只剩约 5.8 km 的 DOR 余度，右路只剩约 8.3 km 的余度。",
                "因此，279.0 s 的规避不是‘打完再走’，而是在首轮链条已经形成 6 次发射门通过和 16 次接力制导成功后，主动保留再组织能力的止压动作。",
            ],
            [
                "390.2-400.2 s",
                "390.0 s 预警图景恢复，390.2 s 阶段层先切入截获重组，400.2 s 才恢复交战。",
                "战术层在这 10 s 内没有立刻把右路改为新的攻击模板，而是在 400.4 s 才把右路补切到“前后夹击”，完成双路结构归一。",
                "机动层利用这 10 s 完成回转后的最小重组，而不是直接从规避跳回攻击。",
                "参数层并未把‘距离回升’当成唯一条件：390.2 s 时最近敌我距离仍只有 59.4 km，400.2 s 时为 56.0 km，说明回转不是因为距离已经完全拉开，而是因为外层图景恢复并且模板骨架已经可以重新闭合。",
                "这正是“为什么是 10 s 而不是 20 s”的直接证据：10 s 足以完成重组，继续等 20 s 只会把可用压制窗口白白让掉，而不会显著增加生存余度。",
            ],
            [
                "416.8 s 与 513.0-523.0 s",
                "416.8 s 再次出现导弹来袭，阶段层第二次切回规避；513.0 s 先重组，523.0 s 再恢复交战。",
                "战术层此时已把左右两路都切到“前后夹击”，说明系统在第二次恢复前先把骨架补齐。",
                "机动层的节奏不是‘一恢复就一直追’，而是允许再规避、再回转的循环。",
                "参数层上，416.8 s 右路配对距离仅 43.4 km，而右路 DOR 已收缩到 44.0 km，说明右路实际已经触碰 DOR；直到 513.0 s 至 523.0 s 距离重新扩展到 87 km 至 89 km，系统才允许第二次恢复交战。",
                "这一窗口证明场景二的有效性不在于完全避免规避，而在于规避后仍能恢复压制，并且每一次恢复都仍受同一套生存边界约束。 ",
            ],
        ]
    return [
        [
            "0.2-64.0 s",
            "场景三从 0.2 s 起就因敌方压入中风险区而直接进入交战，阶段层没有传统远距巡逻缓冲。",
            "战术层左路采用“前后夹击”，右路在 0.4 s 即切为“钳形夹击”，说明系统一开始就按非对称模板快速展开。",
            "机动层在 0.8 s 已把左路推进到 MTR-LR，体现低空突防条件下的前移塑形。",
            "参数层坚持 5.0 s 稳定门限，并在 64.0 s 给出首次发射门通过。若把稳定门限机械延长到 10 s 或 20 s，则首轮窗口会继续后移，而低空突防样本最稀缺的恰恰就是前段时间余度。",
            "因此，场景三的第一步验证的不是‘能不能更快开火’，而是‘在时间余度极小的样本里，能否用有限门限把首轮链条前移’。",
        ],
        [
            "131.8 s",
            "阶段层在导弹来袭信号出现后立即切换为规避。",
            "战术层此时左路保持“前后夹击”，右路仍维持“钳形夹击”，说明规避触发前系统并没有先把模板做成保守收缩，而是让模板执行到被迫打断为止。",
            "机动层在这一步执行的是强制止压，而不是继续争取一轮追加攻击。",
            "参数层给出的几何约束十分明确：左路配对距离约 53.4 km，对应 DOR 为 51.5 km、DR 为 47.5 km，仅剩 1.9 km 的 DOR 余度；同时全局最近敌我距离已压到 62.2 km，高风险区压力已经形成。",
            "因此，131.8 s 的规避不是抽象的‘感觉危险了’，而是导弹来袭与 DOR 余度几乎耗尽同时发生后，对生存优先逻辑的强制兑现。",
        ],
        [
            "261.4-271.4 s",
            "261.4 s 预警图景恢复，阶段层先切到截获重组；271.4 s 才恢复交战。",
            "战术层没有简单恢复到开局模板，而是在 271.4 s 把右路目标切到 B0400，说明恢复动作首先服务于重新分配剩余可打目标。",
            "机动层在这 10 s 中完成剩余兵力的回转和重新占位，而不是立即把双路再压回去。",
            "参数层上，261.4 s 时最近敌我距离回升到 76.9 km，累计接力制导成功事件达到 28，已经具备重新组织的空间；271.4 s 的 10 s 缓冲用于完成目标再占用与模板恢复。",
            "这给出了‘为什么是 10 s 而不是 20 s’的第一处直接例证：10 s 足以完成重组，继续拖到 20 s 并不能恢复更多兵力，只会浪费高压样本中极其有限的守区窗口。",
        ],
        [
            "327.4 s 与 421.2-431.2 s",
            "327.4 s 再次出现导弹来袭，阶段层第二次切回规避；421.2 s 先重组，431.2 s 再恢复交战。",
            "战术层直到 431.2 s 才把右路模板重新归一到“前后夹击”，说明系统没有在 421.2 s 一恢复外层图景就直接回打。",
            "机动层在这 10 s 中保留了回转缓冲，避免在高压近距下从规避直接跳回攻击。",
            "参数层上，327.4 s 时最近敌我距离仅 49.3 km，左路 DOR 为 51.5 km，已经再次逼近生存边界；421.2 s 虽已恢复图景，但最近敌我距离只有 42.9 km，因此阶段层只允许进入截获重组，不允许直接交战。直到 431.2 s 模板恢复、目标重分配完成后，才再次进入交战。",
            "这一窗口说明高压样本中的回转不是‘看到目标就回’，而是必须先过阶段层的重组门，再过战术层的重新占位门，最后才是交战门。",
        ],
        [
            "1056.2-1180.0 s",
            "在 990.2 s 已经出现 1 架我机损失的情况下，1056.2 s 再次因导弹来袭进入规避；1170.0 s 截获重组，1180.0 s 恢复交战。",
            "战术层在后段仍能恢复到双路“前后夹击”，说明生存优先不是简单撤离，而是争取以剩余兵力继续守区。",
            "机动层继续维持 10 s 的重组缓冲，没有因为后段压力增大而把规避后的等待时间无限拉长。",
            "参数层上，1056.2 s 时最近敌我距离已压到 43.5 km，左路 DOR 仅 36.5 km、右路配对距离 43.6 km，说明后段余度已经非常薄；1170.0 s 到 1180.0 s 之间距离回升到 72 km 至 73 km，才允许再度进入交战。",
            "这一窗口证明 001339 虽然没有追求更高击落数，但确实把损失推迟到了 990.2 s，并且在损失发生后仍保持了可解释的“规避—重组—守区”节奏，这正是场景三选择该样本的原因。",
        ],
    ]


def decision_analysis_paragraphs_final(scenario_id: str) -> List[str]:
    if scenario_id == "S1":
        return [
            "（1）场景一的第一处关键不在于打出去多少枚导弹，而在于 99.6 s 至 150.4 s 这一段为什么只做塑形、不做放行。99.6 s 转入交战时，左右两路都还在 NLT 外段，稳定交战就绪目标数为 0；到 148.0 s 和 150.4 s，左右两路才完成“左钳形、右前后”的异步模板分化。此时配对距离仍为 156 km 至 190 km，远高于 LR/TR 窗口，因此系统把这一段用于搭建双路攻击骨架，而不是提前放行。",
            "（2）场景一的第二处关键是 266.6 s 首次放行与 295.0 s 首次接力制导成功之间的关系。001401 在 104.4 s 已经形成首次稳定交战就绪，但直到 266.6 s 才给出首次发射门通过，说明“5 s 稳定门限”只是放行前提，并不是一达到就自动发射。放行之后又经过 28.4 s 才出现首次接力制导成功，进一步说明参数层并不是单点判定，而是在距离窗口、照射责任和后续中制导链都闭合后，才把首轮攻击链真正做实。",
            "（3）322.0 s 的规避触发具备非常明确的因果链。阶段层的触发信号是导弹来袭，参数层的触发条件是左路已经从 54.1 km 压到 DOR=55.6 km 以内，只比 DR=51.6 km 多出约 2.5 km。也就是说，即使没有额外口头解释，数据本身已经表明此时继续压近会直接吃掉重组余度。因此，这一规避不是‘保守’，而是对已经发生的生存边界穿越做出的必要响应。",
            "（4）场景一对“为什么回转等待 10 s 而不是更久”给出了完整答案。446.6 s 先恢复到截获重组，说明阶段层先要求外层图景重新可用；456.6 s 才恢复交战，说明战术层还需要 10 s 完成目标再分配。回转之前系统已经累计击落 3 架敌机、形成 37 次接力制导成功事件，因此这 10 s 并非无效等待，而是把前序杀伤链兑现与后续最后清场窗口衔接起来的最小过渡时间。",
        ]
    if scenario_id == "S2":
        return [
            "（1）场景二的分析不能只看最终 3:0 结果，更关键的是 50.0 s 起就进入预警缺口这一事实。系统在同一时刻既把状态切到交战，又把敌方预警图景状态记录为 DENIED，因此这里的“交战”并不是建立在完整外层图景上的。之所以仍选择“并列压制”，是因为信息缺口条件下首先要保住双路横向覆盖，而不是追求更激进但更容易断链的穿插动作。",
            "（2）236.2 s 的首次放行说明信息缺口没有让发射门变得宽松。此时预警图景仍不可用，但机载雷达激活后的航迹连续性仍有 86.0%，稳定交战就绪目标数达到 3，左路真正进入 LR-TR 窗口后才放行。因此，场景二验证的是“在信息不完整时仍按控制距离和稳定火控发射”，而不是“因为看不清就提前打”。",
            "（3）279.0 s 的首次规避和 390.2 s 至 400.2 s 的第一次恢复，构成了场景二最核心的决策验证。规避发生时，左路只剩约 5.8 km 的 DOR 余度；390.2 s 预警图景恢复后，系统没有立即回打，而是先进入截获重组，再用 10 s 完成右路模板归一。这说明 10 s 缓冲的本质不是时间常数，而是完成‘图景恢复—模板补齐—再交战’所需的最小重组时长。",
            "（4）416.8 s 的再次规避进一步证明了场景二不是单次规避后就一路追击到底。第二次恢复交战仅持续 16.6 s，右路即以 43.4 km 的配对距离触碰 DOR=44.0 km，系统再次止压。也就是说，阶段层并不会因为刚刚恢复过交战就放弃边界监视；生存边界一旦再次被逼近，规避仍然具有最高优先级。",
            "（5）513.0 s 至 523.0 s 的第二次恢复说明场景二的方法价值在于可重复恢复，而不是可永久压制。系统在已有 2 架击落、0 架损失的基础上，仍按‘先截获重组、后恢复交战’的同一套逻辑处理后续过程，并最终形成 3:0 结果。这证明该方法在信息缺口场景下的有效性来自一致的决策纪律，而不是偶然一次成功放行。",
        ]
    return [
        "（1）场景三的第一处关键是控制距离链整体前移。0.2 s 即转入交战，0.4 s 右路切入“钳形夹击”，0.8 s 左路已经推进到 MTR-LR，64.0 s 完成首次放行。这个节奏说明低空突防样本中几乎没有完整远距塑形余度，因此‘为什么不是等 10 s 或 20 s 再放行’的答案很直接：在这样压缩的场景里，门限如果人为加长，首轮拦截窗口本身就会被拖没。",
        "（2）131.8 s 的首次规避明确显示出场景三为何必须优先生存。规避触发时，左路配对距离只有 53.4 km，而 DOR 只有 51.5 km，只剩下 1.9 km 的余度；与此同时，阶段层已经收到导弹来袭信号。也就是说，这一规避并不是在安全边界外‘提前保守’，而是在导弹来袭与 DOR 即将耗尽同时出现时，对强制止压条件的立即执行。",
        "（3）261.4 s 至 271.4 s 的第一次回转，给出了最典型的 10 s 缓冲解释。261.4 s 先切入截获重组，271.4 s 才恢复交战，中间 10 s 用来完成目标再分配和模板恢复，而不是机械等待。若在 261.4 s 直接恢复交战，剩余兵力还没有完成重新占位；若继续拉到 20 s，则会在低空高压场景里把可用守区窗口继续压缩。",
        "（4）327.4 s 的第二次规避和 421.2 s 至 431.2 s 的第二次回转进一步证明，场景三中的回转必须先过重组门，再过交战门。421.2 s 虽已恢复图景，但最近敌我距离仍只有 42.9 km，系统因此只允许进入截获重组；直到 431.2 s 右路模板恢复到“前后夹击”，才重新交战。这里不是简单的‘看见目标就打’，而是阶段层、战术层和参数层共同约束下的二次回转。",
        "（5）001339 被选为场景三正文样本，核心原因并不是它击落更多，而是它把首次我机损失推迟到了 990.2 s，并把高风险区累计滞留压缩到了 111.0 s。后段 1056.2 s 至 1180.0 s 的再规避与再回转说明，即使已经发生损失，系统仍能按同一套 10 s 重组缓冲继续守区。这正体现了生存优先策略的验证目标：不是保证零损失，而是在极端压缩场景中把损失尽量后推，并让剩余兵力保持可解释的守区能力。",
    ]


def scenario_conclusion_text_final(scenario_id: str) -> str:
    if scenario_id == "S1":
        return (
            "场景一表明，在低风险正面对进条件下，分层决策方法能够形成“远距塑形—首轮放行—导弹来袭规避—截获重组—回转清场”的完整闭环。"
            "001401 的战果为 4:0，高风险区累计滞留时间为 0 s，且最小敌我距离达到 57.4 km，说明完整闭环并不是以牺牲生存余度换来的。"
        )
    if scenario_id == "S2":
        return (
            "场景二表明，外层信息存在 400 s 级缺口时，决策方法仍能依靠机载连续跟踪维持压制骨架，并通过两次“规避—重组—再交战”恢复持续压制。"
            "225438 最终形成 3:0 且无高风险区突入，说明该方法的有效性来自一致的放行纪律和生存边界控制，而不是单次激进前压。"
        )
    return (
        "场景三表明，低空突防高压样本中的关键不是多打几枚导弹，而是把首轮攻击前移后，能否及时止压、及时回转并尽量延后损失。"
        "001339 虽只形成 1:1 结果，但首次我机损失被推迟到 990.2 s，高风险区累计滞留控制在 111.0 s，说明生存优先逻辑确实在高压样本中发挥了主导作用。"
    )


def build_report() -> str:
    loaded = all_samples_loaded()
    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# 第六章仿真验证文档")
    lines.append("")
    lines.append("## 6.3 仿真验证")
    lines.append("")
    lines.append(
        "本节仅从给定候选样本池中，为每一类任务场景各选取 1 个最终代表样本进入正文分析。"
        "其余样本只用于筛选论证，不再作为并列正文样本展开。"
    )
    lines.append(
        "三类场景分别对应：低风险正面对进条件下的完整链路验证、信息缺口条件下的持续压制验证、低空突防条件下的生存优先验证。"
        "三类场景的指标口径不再强行统一，而是围绕各自验证重点选择不同证据链。"
    )
    lines.append(
        "正式统计口径继续采用去重统计（unique counting）。稳定交战就绪门限统一为 5.0 s；发射门通过、接力制导成功和导弹发射均按事件去重统计，避免对同一事件重复累计。"
    )
    lines.append("")
    lines.append("### 6.3.1 验证对象、样本筛选原则与术语说明")
    lines.append("")
    lines.append("表6-1 样本池与最终入选结果")
    lines.append(md_table(["场景", "候选样本", "最终入选样本", "入选原则"], sample_pool_overview_rows()))
    lines.append("")
    lines.append("表6-2 本节术语说明")
    lines.append(md_table(["术语", "含义"], TERM_TABLE))
    lines.append("")
    lines.append("表6-3 四层决策验证口径")
    lines.append(md_table(["决策层级", "主要输入", "主要输出", "验证判据"], decision_layer_table()))
    lines.append("")
    lines.append(
        "控制距离节点缩写含义如下：NLT 为不晚于截获距离，MELD 为混战进入距离，MTR 为机动跟踪距离，LR 为首次发射准备距离，TR 为交战转换距离，DOR 为脱离决断距离，DR 为重组决断距离，MAR 为最小规避距离。"
    )
    lines.append("")
    lines.append(
        "本节对“为什么在该时刻规避”“为什么在该时刻回转”“为什么保留 10 s 而不是 20 s 的重组缓冲”的回答，均以同一时间窗内的阶段状态、模板切换、机动打断和参数门限为依据，不再单独使用孤立结果图进行说明。"
    )
    lines.append("")

    figure_counter = 1
    table_counter = 4
    section_map = {"S1": "6.3.2", "S2": "6.3.3", "S3": "6.3.4"}

    for scenario_id in ["S1", "S2", "S3"]:
        sample = final_sample_data(loaded, scenario_id)
        meta = SCENARIO_META[scenario_id]
        lines.append(f"### {section_map[scenario_id]} {meta['title']}")
        lines.append("")

        lines.append("#### （1）场景目标与初始状态")
        lines.append("")
        lines.append(
            f"{meta['title']}。该场景的验证目标是：{meta['task_focus']}"
            f"{meta['scene_goal']}"
        )
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}初始状态")
        lines.append(
            md_table(
                ["飞机ID", "阵营", "机型", "初始经度 / deg", "初始纬度 / deg", "初始高度 / m", "初始航向 / deg", "初始速度", "初始导弹数"],
                initial_state_table_detailed(sample, scenario_id),
            )
        )
        table_counter += 1
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}场景设定与分析重点")
        lines.append(md_table(["项目", "内容"], scene_setup_table_final(scenario_id)))
        table_counter += 1
        lines.append("")

        lines.append("#### （2）候选样本筛选与代表样本确定")
        lines.append("")
        headers, rows = candidate_comparison_table_final(scenario_id, loaded)
        lines.append(f"表6-{table_counter} {meta['title']}候选样本筛选依据")
        lines.append(md_table(headers, rows))
        table_counter += 1
        lines.append("")
        lines.append(selected_sample_reason_text(scenario_id, loaded))
        lines.append("")

        lines.append("#### （3）场景重点指标与总体态势")
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}重点指标")
        lines.append(md_table(["指标项", "数值", "本场景中的分析用途"], focus_metrics_table_final(scenario_id, sample)))
        table_counter += 1
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}关键里程碑时刻")
        lines.append(md_table(["节点", "时间"], milestone_table(sample)))
        table_counter += 1
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}控制距离节点首次触发时刻")
        lines.append(md_table(["航路", "控制阶段", "首次进入时间", "进入距离", "节点含义"], control_node_table_final(sample)))
        table_counter += 1
        lines.append("")

        traj_rel = plot_spatial_trajectory_single(sample, scenario_id, f"{scenario_id}_spatial_trajectory.png")
        lines.append(f"图6-{figure_counter} {meta['title']}敌我航迹与关键决策节点空间图")
        lines.append(f"![图6-{figure_counter} {meta['title']}敌我航迹与关键决策节点空间图]({traj_rel})")
        if scenario_id == "S1":
            lines.append("图中可见，场景一在远距阶段保留了足够长的塑形航段，首轮放行发生在双路已经分化之后，规避回转后又重新回到清场方向。这说明正文入选样本不是依赖单次直冲完成战果，而是在空间上体现出完整的“塑形—攻击—规避—回转”闭环。")
        elif scenario_id == "S2":
            lines.append("图中可见，场景二在预警缺口存在时仍保持双路压制骨架，首次规避后没有直接脱离任务，而是在回转后再次恢复压制。总体航迹说明该样本的价值不在于单次接敌，而在于压制能够被恢复。")
        else:
            lines.append("图中可见，场景三的交战窗口显著前移，航迹从开局起就进入高压压缩态势。入选样本虽然多次规避，但每次规避后都尝试以剩余兵力继续守区，这正对应了生存优先场景的验证目标。")
        figure_counter += 1
        lines.append("")

        timeline_rel = plot_state_tactic_timeline_single(sample, scenario_id, f"{scenario_id}_state_tactic_timeline.png")
        lines.append(f"图6-{figure_counter} {meta['title']}任务状态、战术模板与控制阶段时序图")
        lines.append(f"![图6-{figure_counter} {meta['title']}任务状态、战术模板与控制阶段时序图]({timeline_rel})")
        if scenario_id == "S1":
            lines.append("该图把任务状态、左右两路模板和左右两路控制阶段放在同一时间轴上。可以直接看到 99.6 s 转入交战、148.0 s 与 150.4 s 完成双路异步塑形、322.0 s 规避、446.6 s 截获重组、456.6 s 恢复交战的先后顺序。")
        elif scenario_id == "S2":
            lines.append("该图用于观察信息缺口条件下，双路模板是否仍然稳定，以及规避后是否真的先进入截获重组、再恢复交战。场景二的节奏不是单次进攻，而是多次“规避—重组—再压制”的循环。")
        else:
            lines.append("该图用于识别高压场景中的关键切换顺序：何时直接进入交战、何时因导弹来袭强制止压、何时先截获重组、再恢复交战。场景三中 10 s 的重组缓冲在多次回转中都被重复使用。")
        figure_counter += 1
        lines.append("")

        if scenario_id == "S1":
            special_rel = plot_zone_entry_timeline(sample, f"{scenario_id}_zone_timeline.png")
            lines.append(f"图6-{figure_counter} {meta['title']}敌机风险区进入时序图")
            lines.append(f"![图6-{figure_counter} {meta['title']}敌机风险区进入时序图]({special_rel})")
            lines.append("该图按敌机编号逐一标示其进入低风险区、中风险区和高风险区的持续时段。001401 的四个目标全部只经历“低风险区—中风险区”的推进，没有进入高风险区，说明回转时机没有把敌方放任推入最高风险边界。")
        elif scenario_id == "S2":
            special_rel = plot_awacs_gap_single(sample, f"{scenario_id}_awacs_gap.png")
            lines.append(f"图6-{figure_counter} {meta['title']}预警可用性、稳定跟踪与距离压缩关系图")
            lines.append(f"![图6-{figure_counter} {meta['title']}预警可用性、稳定跟踪与距离压缩关系图]({special_rel})")
            lines.append("图中上方色带给出预警图景可用与失配区段，下方曲线给出最近敌我距离和稳定交战就绪目标数的演化。可以直接看到：首次放行发生在预警失配区段内部，而第一次恢复交战发生在图景恢复之后，这正是信息缺口场景的核心证据。")
        else:
            special_rel = plot_high_risk_pressure_single(sample, f"{scenario_id}_high_risk_pressure.png")
            lines.append(f"图6-{figure_counter} {meta['title']}高风险区压力、距离压缩与损失时刻图")
            lines.append(f"![图6-{figure_counter} {meta['title']}高风险区压力、距离压缩与损失时刻图]({special_rel})")
            lines.append("该图把高风险区内敌机数量、最近敌我距离和我方损失时刻放在同一视图中。001339 的高风险区累计滞留虽然仍达到 111.0 s，但首次我机损失被推迟到 990.2 s，说明生存优先策略确实在压缩样本中延后了代价兑现。")
        figure_counter += 1
        lines.append("")

        control_rel = plot_control_distance_actual_single(sample, scenario_id, f"{scenario_id}_control_distance_actual.png")
        lines.append(f"图6-{figure_counter} {meta['title']}左右两路实际配对距离与节点触发图")
        lines.append(f"![图6-{figure_counter} {meta['title']}左右两路实际配对距离与节点触发图]({control_rel})")
        lines.append("该图使用实际时序曲线，而不是把控制节点端点直接线性连线。左右两路的配对距离压缩、回收、再次压缩都以真实时间序列呈现，因此能够直接用于判断首轮放行前的压缩过程、规避时刻与回转后的恢复过程。")
        figure_counter += 1
        lines.append("")

        chain_rel = plot_event_chain_single(sample, scenario_id, f"{scenario_id}_event_chain_single.png")
        lines.append(f"图6-{figure_counter} {meta['title']}交战链关键事件累计时序图")
        lines.append(f"![图6-{figure_counter} {meta['title']}交战链关键事件累计时序图]({chain_rel})")
        if scenario_id == "S1":
            lines.append("图中可以区分“先形成稳定交战就绪，再出现发射门通过，随后形成接力制导成功，最后转化为击落结果”的层次关系。场景一关注的是完整闭环，因此这一图主要用于说明放行、规避和清场之间的先后顺序。")
        elif scenario_id == "S2":
            lines.append("图中可以看到接力制导成功事件在规避和回转之间仍持续增加，说明持续压制并不是完全依赖发射平台保持原位，而是通过照射责任接替把前序链条继续兑现。")
        else:
            lines.append("图中可以看到场景三虽然较早形成首轮放行，但击落数并没有持续快速上升，这说明高压样本中的真正瓶颈不在于首轮是否够早，而在于规避后的持续守区能力是否足够强。")
        figure_counter += 1
        lines.append("")

        lines.append("#### （4）四层决策关键过程与有效性分析")
        lines.append("")
        lines.append(f"表6-{table_counter} {meta['title']}关键决策窗口")
        lines.append(
            md_table(
                ["时间窗", "阶段决策层", "战术决策层", "机动决策层", "参数决策层", "有效性判定"],
                decision_window_rows_final(scenario_id),
            )
        )
        table_counter += 1
        lines.append("")
        for paragraph in decision_analysis_paragraphs_final(scenario_id):
            lines.append(paragraph)
            lines.append("")

        lines.append("#### （5）场景结论")
        lines.append("")
        lines.append(scenario_conclusion_text_final(scenario_id))
        lines.append("")

    lines.append("### 6.3.5 三类场景横向归纳")
    lines.append("")
    lines.append(f"表6-{table_counter} 三类场景入选样本横向比较")
    lines.append(
        md_table(
            ["场景", "入选样本", "验证重点", "直接证据", "最终结果", "暴露问题"],
            [
                ["场景一", "001401（S1）", "完整链路闭环与安全回转", "4:0、0 s 高风险滞留、最小敌我距离 57.4 km、322.0 s 规避后经 10 s 重组恢复交战", "完整闭环成立", "后段清场仍依赖稳定接力制导链，不宜过早回转"],
                ["场景二", "225438（S2）", "信息缺口下持续压制与重复恢复", "400.0 s 预警失配、236.2 s 首放、279.0 s 首规避、390.2-400.2 s 首次恢复、416.8 s 再规避", "3:0、0 s 高风险滞留", "持续压制样本对 DOR 边界和回转时机高度敏感"],
                ["场景三", "001339（S3）", "低空突防场景的生存优先与止压回收", "64.0 s 首放、131.8 s 首规避、261.4-271.4 s 首次回转、111.0 s 高风险滞留、首次损失延后到 990.2 s", "1:1", "后段剩余兵力守区能力仍然偏弱"],
            ],
        )
    )
    table_counter += 1
    lines.append("")
    lines.append(
        "综合三类场景可见，分层决策方法的有效性并不体现为所有场景都使用同一套指标，而体现为：低风险场景强调完整闭环是否成立，信息缺口场景强调压制是否能够被恢复，高压低空场景强调损失是否能够被后推。"
    )
    lines.append(
        "从最终样本看，场景一证明了完整链路与安全回转可以兼得；场景二证明了在预警缺口下仍可通过“规避—重组—再交战”维持压制；场景三证明了在时间余度极小的样本里，生存优先逻辑能够及时止压并延后损失。"
    )
    lines.append(
        "因此，本节的结论不是简单地给出若干高低不一的指标，而是说明四层决策架构已经能够在不同压力结构下给出可解释的阶段切换、模板选择、机动打断和参数回收；其下一步优化重点将集中在高压残余兵力下的持续守区能力。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(REPORT_PATH)
    print(ASSET_DIR)


if __name__ == "__main__":
    main()
