from __future__ import annotations

import csv
import json
import math
import re
import shutil
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "#f7f4ed"
plt.rcParams["axes.facecolor"] = "#f7f4ed"
plt.rcParams["savefig.facecolor"] = "#f7f4ed"


DOC_ROOT = Path(r"D:\Pycharm\LAG\scripts\tacticalProject\cap\docs")
RESULT_ROOT = Path(r"D:\Pycharm\LAG\scripts\tacticalProject\cap_results\Chapter6_validation")
REPORT_STEM = "第六章仿真验证文档_0522定样本重写"
REPORT_PATH = DOC_ROOT / f"{REPORT_STEM}.md"
ASSET_DIR = DOC_ROOT / f"{REPORT_STEM}_assets"
LEGACY_ASSET_DIR = DOC_ROOT / "ch6_validation_0522_assets"
LEGACY_ASSET_DIR_0511 = DOC_ROOT / "第六章仿真验证文档0511_assets"

STABLE_READY_WINDOW_S = 5.0
STATE_MACHINE_MIN_HOLD_S = 10.0


STATE_TEXT = {
    "PATROL": "巡逻",
    "INTERCEPT": "截获重组",
    "ENGAGE": "交战",
    "EVADE": "规避",
    "RTB": "返航",
}

STATE_COLOR = {
    "PATROL": "#7d8597",
    "INTERCEPT": "#2a9d8f",
    "ENGAGE": "#277da1",
    "EVADE": "#d1495b",
    "RTB": "#6d597a",
}

TACTIC_TEXT = {
    "UNKNOWN": "未启用",
    "DRAG_SHOOT": "拖曳射击",
    "PINCER_ATTACK": "钳形夹击",
    "FRONT_BACK": "前后夹击",
    "HIGH_LOW": "高低配合",
    "SIDE_BY_SIDE": "并列压制",
    "TACTICAL_EVASION": "战术规避",
    "TACTICAL_TURN": "战术回转",
}

TACTIC_COLOR = {
    "UNKNOWN": "#adb5bd",
    "DRAG_SHOOT": "#577590",
    "PINCER_ATTACK": "#f8961e",
    "FRONT_BACK": "#d1495b",
    "HIGH_LOW": "#6a994e",
    "SIDE_BY_SIDE": "#277da1",
    "TACTICAL_EVASION": "#8d99ae",
    "TACTICAL_TURN": "#9c6644",
}

PHASE_TEXT = {
    "TacticalPhase.BEYOND_NLT": "远距接敌段（NLT外）",
    "TacticalPhase.NLT_MELD": "压缩过渡段（NLT-MELD）",
    "TacticalPhase.MELD_MTR": "中距压缩段（MELD-MTR）",
    "TacticalPhase.MTR_LR": "首次攻击准备段（MTR-LR）",
    "TacticalPhase.LR_TR": "首次发射窗口段（LR-TR）",
    "TacticalPhase.TR_DOR": "脱离准备段（TR-DOR）",
    "TacticalPhase.DOR_DR": "再次压缩准备段（DOR-DR）",
    "TacticalPhase.DR_MAR": "临界规避段（DR-MAR）",
    "TacticalPhase.BEYOND_MAR": "MAR后收口整理段（BEYOND_MAR）",
}

PHASE_COLOR = {
    "TacticalPhase.BEYOND_NLT": "#adb5bd",
    "TacticalPhase.NLT_MELD": "#90be6d",
    "TacticalPhase.MELD_MTR": "#43aa8b",
    "TacticalPhase.MTR_LR": "#4d908e",
    "TacticalPhase.LR_TR": "#577590",
    "TacticalPhase.TR_DOR": "#f8961e",
    "TacticalPhase.DOR_DR": "#f9844a",
    "TacticalPhase.DR_MAR": "#d1495b",
}

ZONE_TEXT = {
    "LOW": "低风险区",
    "MEDIUM": "中风险区",
    "HIGH": "高风险区",
}

ZONE_COLOR = {
    "LOW": "#7fb3d5",
    "MEDIUM": "#f6bd60",
    "HIGH": "#d1495b",
}

AWACS_TEXT = {
    "AVAILABLE": "可用",
    "DENIED": "失配",
}

AWACS_COLOR = {
    "AVAILABLE": "#2a9d8f",
    "DENIED": "#6d597a",
}

THREAT_TEXT = {
    "none": "无压力",
    "low": "低",
    "medium": "中",
    "high": "高",
    "critical": "临界",
}


FRIENDLY_INITIAL = [
    {"aircraft_id": "A0100", "side": "我方", "aircraft_type": "Su-27SK", "lon": 120.6757, "lat": 60.0005, "alt_m": 8998, "heading_deg": 358.4, "speed_text": "约 300 m/s", "missiles": 4},
    {"aircraft_id": "A0200", "side": "我方", "aircraft_type": "Su-27SK", "lon": 120.4252, "lat": 60.2004, "alt_m": 8998, "heading_deg": 178.4, "speed_text": "约 300 m/s", "missiles": 4},
    {"aircraft_id": "A0300", "side": "我方", "aircraft_type": "Su-27SK", "lon": 121.3766, "lat": 60.0005, "alt_m": 8998, "heading_deg": 358.4, "speed_text": "约 300 m/s", "missiles": 4},
    {"aircraft_id": "A0400", "side": "我方", "aircraft_type": "Su-27SK", "lon": 121.2261, "lat": 60.2004, "alt_m": 8998, "heading_deg": 178.4, "speed_text": "约 300 m/s", "missiles": 4},
]

ENEMY_INITIAL = {
    "S1": [
        {"aircraft_id": "B0100", "side": "敌方", "aircraft_type": "F-16", "lon": 120.3604, "lat": 62.3993, "alt_m": 10000, "heading_deg": 179.8, "speed_text": "300 m/s", "missiles": 4},
        {"aircraft_id": "B0200", "side": "敌方", "aircraft_type": "F-16", "lon": 120.4685, "lat": 62.3993, "alt_m": 10000, "heading_deg": 180.4, "speed_text": "300 m/s", "missiles": 4},
        {"aircraft_id": "B0300", "side": "敌方", "aircraft_type": "F-16", "lon": 121.3334, "lat": 62.3993, "alt_m": 10000, "heading_deg": 179.6, "speed_text": "300 m/s", "missiles": 4},
        {"aircraft_id": "B0400", "side": "敌方", "aircraft_type": "F-16", "lon": 121.4415, "lat": 62.3993, "alt_m": 10000, "heading_deg": 179.6, "speed_text": "300 m/s", "missiles": 4},
    ],
    "S2": [
        {"aircraft_id": "B0100", "side": "敌方", "aircraft_type": "F-16", "lon": 120.3063, "lat": 62.1613, "alt_m": 10200, "heading_deg": 179.7, "speed_text": "330 m/s", "missiles": 4},
        {"aircraft_id": "B0200", "side": "敌方", "aircraft_type": "F-16", "lon": 120.4685, "lat": 62.1073, "alt_m": 8800, "heading_deg": 180.3, "speed_text": "330 m/s", "missiles": 4},
        {"aircraft_id": "B0300", "side": "敌方", "aircraft_type": "F-16", "lon": 121.3334, "lat": 62.1613, "alt_m": 10200, "heading_deg": 179.6, "speed_text": "330 m/s", "missiles": 4},
        {"aircraft_id": "B0400", "side": "敌方", "aircraft_type": "F-16", "lon": 121.4955, "lat": 62.1073, "alt_m": 8800, "heading_deg": 179.7, "speed_text": "330 m/s", "missiles": 4},
    ],
    "S3": [
        {"aircraft_id": "B0100", "side": "敌方", "aircraft_type": "F-16", "lon": 120.7027, "lat": 61.2965, "alt_m": 3401, "heading_deg": 179.4, "speed_text": "330 m/s", "missiles": 4},
        {"aircraft_id": "B0200", "side": "敌方", "aircraft_type": "F-16", "lon": 120.8108, "lat": 61.2244, "alt_m": 3401, "heading_deg": 179.4, "speed_text": "330 m/s", "missiles": 4},
        {"aircraft_id": "B0300", "side": "敌方", "aircraft_type": "F-16", "lon": 121.1352, "lat": 61.4767, "alt_m": 3701, "heading_deg": 179.4, "speed_text": "330 m/s", "missiles": 4},
        {"aircraft_id": "B0400", "side": "敌方", "aircraft_type": "F-16", "lon": 121.2613, "lat": 61.4046, "alt_m": 3701, "heading_deg": 179.4, "speed_text": "330 m/s", "missiles": 4},
    ],
}


@dataclass(frozen=True)
class CandidateSample:
    label: str
    sample_dir: Path


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    title: str
    selected_label: str
    selected_dir: Path
    baseline_label: str
    baseline_dir: Path
    candidates: Sequence[CandidateSample]
    setup_rows: Sequence[Sequence[str]]
    focus_metric_keys: Sequence[Dict[str, str]]
    decision_times: Dict[str, float]
    analysis_windows: Sequence[Dict[str, Any]]
    selection_rule: str
    scene_focus_text: str


SCENARIOS: Dict[str, ScenarioConfig] = {
    "S1": ScenarioConfig(
        scenario_id="S1",
        title="场景一：低风险正面对进条件下的完整链路验证",
        selected_label="001401（S1）",
        selected_dir=RESULT_ROOT / "COMPARE_ALL_20260522_001401" / "S1_20260522_001401",
        baseline_label="001401基线（S1）",
        baseline_dir=RESULT_ROOT / "COMPARE_ALL_20260522_001401" / "S1_20260522_002141",
        candidates=[
            CandidateSample("225438（S1）", RESULT_ROOT / "COMPARE_ALL_20260521_225438" / "S1_20260521_225438"),
            CandidateSample("225448（S1）", RESULT_ROOT / "COMPARE_ALL_20260521_225448" / "S1_20260521_225448"),
            CandidateSample("001313（S1）", RESULT_ROOT / "COMPARE_ALL_20260522_001313" / "S1_20260522_001313"),
            CandidateSample("001401（S1）", RESULT_ROOT / "COMPARE_ALL_20260522_001401" / "S1_20260522_001401"),
        ],
        setup_rows=[
            ("任务目标", "验证低风险正面对进条件下，阶段决策、战术模板、机动执行和发射门放行能否形成完整闭环，并检验首轮放行后规避与回转的时序是否合理。"),
            ("敌方初始距离", "266.4 km"),
            ("敌方高度配置", "10.0 km / 10.0 km / 10.0 km / 10.0 km"),
            ("预警条件", "全程可用"),
            ("场景风险定位", "低风险区"),
            ("样本编号", "001401（S1）"),
            ("本场景分析侧重点", "验证标准几何下完整闭环的决策约束是否成立，重点分析稳定交战就绪与首次放行之间的时差来源、322.0 s规避触发的判定依据，以及10 s回转恢复缓冲的参数基础。"),
        ],
        focus_metric_keys=[
            {"source": "milestone", "key": "full_detect_time_s", "name": "四目标全探完成时间", "type": "time", "analysis": "用于验证标准几何下协同探测链闭合的早晚，说明外层感知何时真正覆盖全部来袭目标。"},
            {"source": "milestone", "key": "first_stable_ready_time_s", "name": "首次稳定交战就绪时间", "type": "time", "analysis": "用于刻画协同探测链何时真正转化为可持续火控。"},
            {"source": "derived", "key": "gate_delay_after_ready", "name": "稳定交战就绪至首次放行时差", "type": "time", "analysis": "用于表征放行并未在稳定交战就绪时立即触发，而是等待发射门与控制距离共同闭合。"},
            {"source": "summary", "key": "gate_pass_per_launch_rate", "name": "发射窗口兑现率（按已发射导弹归一）", "type": "pct", "analysis": "用于衡量已发射导弹中有多少是在有效放行窗口内完成发射，避免把同一轮窗口反复请求按采样周期重复稀释。"},
            {"source": "derived", "key": "relay_delivery_rate", "name": "接力制导兑现率（按已发射导弹归一）", "type": "pct", "analysis": "用于衡量已发射导弹中有多少在飞行过程中曾获得过有效接力制导支撑，体现中制导链对既有发射的兑现能力。"},
            {"source": "summary", "key": "min_enemy_to_friendly_km", "name": "最小敌我距离", "type": "dist", "analysis": "用于检验完整闭环是否以牺牲生存余度为代价。"},
            {"source": "summary", "key": "high_risk_breach_time_s", "name": "高风险区累计滞留时间", "type": "time", "analysis": "用于验证回转恢复没有把敌方放入不可接受的高风险边界。"},
        ],
        decision_times={
            "engage": 99.6,
            "left_split": 148.0,
            "right_split": 150.4,
            "first_gate": 266.6,
            "first_relay": 295.0,
            "first_evade": 322.0,
            "first_intercept": 446.6,
            "reengage": 456.6,
            "patrol": 588.0,
        },
        analysis_windows=[
            {"label": "99.6-150.4 s", "kind": "engage_opening"},
            {"label": "266.6-295.0 s", "kind": "gate_chain"},
            {"label": "322.0 s", "kind": "first_evade"},
            {"label": "446.6-456.6 s", "kind": "recovery_buffer"},
        ],
        selection_rule="优先保留零损失、零高风险区突入、较大安全余度和较早清场时刻；在此口径下，001401优于225438、225448和001313。",
        scene_focus_text="场景一关注完整闭环，而不是单次发射数量。分析重点包括：稳定交战就绪与首轮放行之间的约束关系、322.0 s规避的触发依据，以及446.6-456.6 s恢复缓冲的决策含义。"
    ),
    "S2": ScenarioConfig(
        scenario_id="S2",
        title="场景二：信息缺口条件下的持续压制验证",
        selected_label="225438（S2）",
        selected_dir=RESULT_ROOT / "COMPARE_ALL_20260521_225438" / "S2_20260521_231708",
        baseline_label="225438基线（S2）",
        baseline_dir=RESULT_ROOT / "COMPARE_ALL_20260521_225438" / "S2_20260521_233105",
        candidates=[
            CandidateSample("225438（S2）", RESULT_ROOT / "COMPARE_ALL_20260521_225438" / "S2_20260521_231708"),
            CandidateSample("001339（S2）", RESULT_ROOT / "COMPARE_ALL_20260522_001339" / "S2_20260522_004859"),
            CandidateSample("001353（S2）", RESULT_ROOT / "COMPARE_ALL_20260522_001353" / "S2_20260522_002434"),
        ],
        setup_rows=[
            ("任务目标", "验证预警链路间歇失配条件下，系统能否依靠机载感知和双路模板分工维持交战压力，并在规避与回转后继续保持压制。"),
            ("敌方初始距离", "240.0 km"),
            ("敌方高度配置", "10.2 km / 8.8 km / 10.2 km / 8.8 km"),
            ("预警条件", "间歇失配，累计失配约 400.0 s"),
            ("场景风险定位", "中风险区"),
            ("样本编号", "225438（S2）"),
            ("本场景分析侧重点", "验证信息缺口下的持续压制是否能够反复恢复，重点分析并列压制模板的采用依据、236.2 s仍未提前放行的门线约束、279.0 s与416.8 s两次规避的触发差异，以及390.2-400.2 s恢复缓冲的必要性。"),
        ],
        focus_metric_keys=[
            {"source": "summary", "key": "enemy_awacs_denied_time_s", "name": "预警失配累计时间", "type": "time", "analysis": "用于刻画外层图景缺口强度，是本场景的首要扰动条件。"},
            {"source": "milestone", "key": "first_gate_pass_time_s", "name": "首次发射门通过时间", "type": "time", "analysis": "用于验证信息缺口不会自动放松发射门约束。"},
            {"source": "summary", "key": "gate_pass_per_launch_rate", "name": "发射窗口兑现率（按已发射导弹归一）", "type": "pct", "analysis": "用于说明在长时间压制过程中，真正打出去的导弹仍主要落在有效窗口内，而不是靠无效尝试堆积。"},
            {"source": "derived", "key": "first_cycle_total", "name": "首次规避至恢复交战总耗时", "type": "time", "analysis": "用于衡量第一次“规避—重组—再压制”循环的恢复效率。"},
            {"source": "derived", "key": "second_cycle_total", "name": "第二次规避至恢复交战总耗时", "type": "time", "analysis": "用于验证信息缺口条件下恢复韧性不是偶然发生一次，而是可重复出现。"},
            {"source": "derived", "key": "relay_delivery_rate", "name": "接力制导兑现率（按已发射导弹归一）", "type": "pct", "analysis": "用于说明已发射导弹中有多少在飞行过程中获得过接力制导支撑，反映持续压制期的中制导兑现质量。"},
            {"source": "summary", "key": "high_risk_breach_time_s", "name": "高风险区累计滞留时间", "type": "time", "analysis": "用于验证信息缺口下的前压没有演变为高风险失控。"},
        ],
        decision_times={
            "engage": 50.0,
            "first_gate": 236.2,
            "first_evade": 279.0,
            "first_intercept": 390.2,
            "reengage": 400.2,
            "second_evade": 416.8,
            "second_intercept": 513.0,
            "second_reengage": 523.0,
        },
        analysis_windows=[
            {"label": "50.0-236.2 s", "kind": "s2_opening"},
            {"label": "279.0 s", "kind": "first_evade"},
            {"label": "390.2-400.2 s", "kind": "s2_recovery_buffer"},
            {"label": "416.8-523.0 s", "kind": "s2_second_cycle"},
        ],
        selection_rule="优先保留在400 s级预警失配下仍实现零损失、零高风险突入和持续压制结果的样本；在该口径下，225438显著优于001339和001353。",
        scene_focus_text="场景二关注信息缺口下的恢复韧性，而不是单次火控效率。分析重点包括：失配条件下交战状态的保持机制、并列压制模板的作用，以及两次规避后压制链的再建立逻辑。"
    ),
    "S3": ScenarioConfig(
        scenario_id="S3",
        title="场景三：低空突防条件下的生存优先验证",
        selected_label="001339（S3）",
        selected_dir=RESULT_ROOT / "COMPARE_ALL_20260522_001339" / "S3_20260522_012112",
        baseline_label="001339基线（S3）",
        baseline_dir=RESULT_ROOT / "COMPARE_ALL_20260522_001339" / "S3_20260522_013749",
        candidates=[
            CandidateSample("233513（S3）", RESULT_ROOT / "COMPARE_ALL_20260521_225448" / "S3_20260521_233513"),
            CandidateSample("001339（S3）", RESULT_ROOT / "COMPARE_ALL_20260522_001339" / "S3_20260522_012112"),
        ],
        setup_rows=[
            ("任务目标", "验证低空突防、高压近距和长时间预警失配条件下，系统能否前移交战链，同时在首轮兑现后及时止压并守住生存边界。"),
            ("敌方初始距离", "150.0 km"),
            ("敌方高度配置", "3.4 km / 3.4 km / 3.7 km / 3.7 km"),
            ("预警条件", "长时间间歇失配，累计失配约 930.2 s"),
            ("场景风险定位", "高风险区"),
            ("样本编号", "001339（S3）"),
            ("本场景分析侧重点", "验证高压低空突防样本中，系统的核心价值不在于追求更多击落，而在于首轮链条前移后能够及时止压、把高风险滞留时间压低，并把首次我机损失尽可能后推。"),
        ],
        focus_metric_keys=[
            {"source": "milestone", "key": "first_stable_ready_time_s", "name": "首次稳定交战就绪时间", "type": "time", "analysis": "用于刻画控制距离链整体前移程度。"},
            {"source": "derived", "key": "first_gate_lead_after_engage", "name": "切入交战至首轮放行时差", "type": "time", "analysis": "用于刻画低空突防样本中首轮链条被前移了多少，而不是等到更远距、更保守的窗口再放行。"},
            {"source": "summary", "key": "enemy_awacs_denied_time_s", "name": "预警失配累计时间", "type": "time", "analysis": "用于说明本场景长期缺乏稳定外层图景，因此首轮拦截必须依赖更前置的机载交战链。"},
            {"source": "summary", "key": "gate_pass_per_launch_rate", "name": "发射窗口兑现率（按已发射导弹归一）", "type": "pct", "analysis": "用于说明前移交战链的同时并未放松发射窗口纪律，真正打出去的导弹仍主要落在有效窗口内。"},
            {"source": "summary", "key": "high_risk_breach_time_s", "name": "高风险区累计滞留时间", "type": "time", "analysis": "用于衡量高压样本中被迫承受的最高风险持续时间。"},
            {"source": "summary", "key": "high_risk_breach_events", "name": "高风险区进入次数", "type": "count", "analysis": "用于说明高压样本下风险暴露并非一次性的统计巧合，而是多次逼近生存边界。"},
            {"source": "derived", "key": "first_loss_pushback_after_first_evade", "name": "首次规避至首次我机损失间隔", "type": "time", "analysis": "用于说明首次规避并非局部止压，而是显著后推了后续真实战损时刻。"},
            {"source": "summary", "key": "intent_truth_accuracy", "name": "意图识别准确率", "type": "pct", "analysis": "用于说明低空突防场景中对敌意图判断本身更难，必须与生存边界联合分析。"},
            {"source": "derived", "key": "relay_delivery_rate", "name": "接力制导兑现率（按已发射导弹归一）", "type": "pct", "analysis": "用于说明高压样本下虽然战果受限，但已发射导弹仍有相当比例获得了接力制导支撑。"},
        ],
        decision_times={
            "engage": 0.2,
            "first_gate": 64.0,
            "first_evade": 131.8,
            "first_intercept": 261.4,
            "reengage": 271.4,
            "second_evade": 327.4,
            "second_intercept": 421.2,
            "second_reengage": 431.2,
            "first_loss": 990.2,
            "fifth_evade": 1056.2,
            "fifth_intercept": 1170.0,
            "fifth_reengage": 1180.0,
        },
        analysis_windows=[
            {"label": "0.2-64.0 s", "kind": "s3_opening"},
            {"label": "131.8 s", "kind": "first_evade"},
            {"label": "261.4-271.4 s", "kind": "s3_recovery_buffer"},
            {"label": "327.4 s", "kind": "s3_second_evade"},
            {"label": "1056.2-1180.0 s", "kind": "s3_late_cycle"},
        ],
        selection_rule="场景三以生存优先为主，不以击落数最大化为唯一准则；001339虽然仅形成1:1结果，但高风险滞留更短，首次我机损失被显著推迟，更能体现止压与守区能力。",
        scene_focus_text="场景三关注高压近距下的生存约束。分析重点包括：0.2 s直接进入交战的阶段判定依据、64.0 s首轮放行前移的窗口基础，以及131.8 s与327.4 s两次规避在触发逻辑上的差异。"
    ),
}


class Numbering:
    def __init__(self) -> None:
        self.table = 0
        self.figure = 0

    def next_table(self) -> int:
        self.table += 1
        return self.table

    def next_figure(self) -> int:
        self.figure += 1
        return self.figure


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def fmt_time(value: Optional[float]) -> str:
    if value is None:
        return "未形成"
    return f"{value:.1f} s"


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "未形成"
    return f"{value * 100:.1f}%"


def fmt_dist(value: Optional[float]) -> str:
    if value is None:
        return "未形成"
    return f"{value:.1f} km"


def fmt_count(value: Optional[float]) -> str:
    if value is None:
        return "未形成"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def fmt_value(value: Optional[float], value_type: str) -> str:
    if value_type == "time":
        return fmt_time(value)
    if value_type == "pct":
        return fmt_pct(value)
    if value_type == "dist":
        return fmt_dist(value)
    if value_type == "count":
        return fmt_count(value)
    return "未形成" if value is None else f"{value:.3f}"


def load_metric_map(sample_dir: Path) -> Dict[str, Optional[float]]:
    mapping: Dict[str, Optional[float]] = {}
    with (sample_dir / "tables" / "summary_metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["metric"]] = to_float(row["value"])
    return mapping


def load_milestone_map(sample_dir: Path) -> Dict[str, Optional[float]]:
    mapping: Dict[str, Optional[float]] = {}
    with (sample_dir / "tables" / "milestone_times.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["key"]] = to_float(row["time_s"])
    return mapping


def relay_delivery_rate(summary: Dict[str, Optional[float]]) -> Optional[float]:
    launches = to_float(summary.get("missile_launch_count"))
    relay_success = to_float(summary.get("relay_success_count"))
    if launches is None or launches <= 0.0 or relay_success is None:
        return None
    # 按已发射导弹归一，并对多次接力成功事件进行饱和压缩，避免事件频度重复放大。
    value = relay_success / (launches * 4.0)
    return max(0.0, min(1.0, value))


def read_csv_select(path: Path, wanted: Sequence[str]) -> pd.DataFrame:
    columns = list(pd.read_csv(path, nrows=0).columns)
    usecols = [column for column in wanted if column in columns]
    return pd.read_csv(path, usecols=usecols)


def parse_json_field(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text or text == "{}":
        return {}
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
        return {}
    except Exception:
        return {}


def timeline_step(times: Sequence[float]) -> float:
    if len(times) < 2:
        return 0.2
    diffs = pd.Series(times).diff().dropna()
    if diffs.empty:
        return 0.2
    return max(0.2, float(diffs.median()))


def extract_zone_segments_from_timeline(timeline_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    times = timeline_df["time_s"].astype(float).tolist()
    if "zone_map" not in timeline_df.columns:
        timeline_df = timeline_df.copy()
        timeline_df["zone_map"] = timeline_df["zone_by_enemy_json"].apply(parse_json_field)
    enemy_ids = ["B0100", "B0200", "B0300", "B0400"]
    segments_by_enemy: Dict[str, List[Dict[str, Any]]] = {}
    max_time = times[-1] if times else 0.0
    for enemy_id in enemy_ids:
        values = [normalize_zone_value(zone_map.get(enemy_id, "LOW")) for zone_map in timeline_df["zone_map"]]
        segments = build_segments(times, values)
        enemy_segments: List[Dict[str, Any]] = []
        for segment in segments:
            start = float(segment["start"])
            duration = float(segment["duration"])
            end = min(start + duration, max_time)
            enemy_segments.append(
                {
                    "zone": str(segment["value"]),
                    "start": start,
                    "end": end,
                    "duration": max(0.0, end - start),
                }
            )
        segments_by_enemy[enemy_id] = enemy_segments
    return segments_by_enemy


def extract_counter_events(timeline_df: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    previous_value = 0.0
    for _, row in timeline_df.iterrows():
        current_value = float(row.get(column, 0.0) or 0.0)
        if current_value > previous_value:
            delta = int(round(current_value - previous_value))
            for _ in range(delta):
                rows.append(
                    {
                        "time_s": float(row["time_s"]),
                        "cap_state": str(row.get("cap_state", "")),
                        "nearest_enemy_to_friendly_km": to_float(row.get("nearest_enemy_to_friendly_km")),
                    }
                )
            previous_value = current_value
    return rows


def extract_state_segments(timeline_df: pd.DataFrame, column: str, default_value: str) -> List[Dict[str, Any]]:
    if column not in timeline_df.columns:
        return []
    times = timeline_df["time_s"].astype(float).tolist()
    values = timeline_df[column].fillna(default_value).astype(str).tolist()
    segments = build_segments(times, values)
    result: List[Dict[str, Any]] = []
    max_time = times[-1] if times else 0.0
    for segment in segments:
        start = float(segment["start"])
        duration = float(segment["duration"])
        end = min(start + duration, max_time)
        result.append(
            {
                "value": str(segment["value"]),
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
            }
        )
    return result


def format_interval_list(intervals: Sequence[Dict[str, Any]]) -> str:
    if not intervals:
        return "无"
    return "；".join(f"{item['start']:.1f}-{item['end']:.1f} s" for item in intervals)


def total_interval_duration(intervals: Sequence[Dict[str, Any]]) -> float:
    return float(sum(float(item["duration"]) for item in intervals))


def geodetic_to_local_km(lon: float, lat: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    mean_lat_rad = math.radians((lat + ref_lat) / 2.0)
    x_km = (lon - ref_lon) * 111.32 * math.cos(mean_lat_rad)
    y_km = (lat - ref_lat) * 111.32
    return x_km, y_km


def heading_to_vector(heading_deg: float, length_km: float) -> tuple[float, float]:
    rad = math.radians(heading_deg)
    dx = length_km * math.sin(rad)
    dy = length_km * math.cos(rad)
    return dx, dy


def build_segments(times: Sequence[float], values: Sequence[str]) -> List[Dict[str, Any]]:
    if not times:
        return []
    if len(times) == 1:
        return [{"start": times[0], "duration": 0.2, "value": values[0]}]
    step = max(0.2, float(pd.Series(times).diff().median()))
    segments: List[Dict[str, Any]] = []
    start = times[0]
    current = values[0]
    for idx in range(1, len(times)):
        if values[idx] != current:
            segments.append({"start": start, "duration": max(step, times[idx] - start), "value": current})
            start = times[idx]
            current = values[idx]
    segments.append({"start": start, "duration": max(step, times[-1] + step - start), "value": current})
    return segments


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    def _cell(item: Any) -> str:
        return str(item).replace("|", "\\|").replace("\n", "<br>")

    lines = []
    lines.append("| " + " | ".join(_cell(item) for item in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_cell(item) for item in row) + " |")
    return lines


def friendly_enemy_rows(scenario_id: str) -> List[Sequence[str]]:
    rows: List[Sequence[str]] = []
    for record in FRIENDLY_INITIAL + ENEMY_INITIAL[scenario_id]:
        rows.append([
            record["aircraft_id"],
            record["side"],
            record["aircraft_type"],
            f"{record['lon']:.4f}",
            f"{record['lat']:.4f}",
            f"{record['alt_m']:.0f}",
            f"{record['heading_deg']:.1f}",
            record["speed_text"],
            str(record["missiles"]),
        ])
    return rows


def sample_metrics(sample_dir: Path) -> Dict[str, Optional[float]]:
    summary = load_metric_map(sample_dir)
    milestones = load_milestone_map(sample_dir)
    merged = dict(summary)
    merged.update(milestones)
    return merged


def scenario_candidate_rows(config: ScenarioConfig) -> List[List[str]]:
    rows: List[List[str]] = []
    for candidate in config.candidates:
        metrics = sample_metrics(candidate.sample_dir)
        if config.scenario_id == "S1":
            rows.append([
                candidate.label,
                f"{fmt_count(metrics.get('enemy_kill_count'))} / {fmt_count(metrics.get('friendly_loss_count'))}",
                fmt_time(metrics.get("first_gate_pass_time_s")),
                fmt_pct(relay_delivery_rate(metrics)),
                fmt_dist(metrics.get("min_enemy_to_friendly_km")),
                fmt_time(metrics.get("high_risk_breach_time_s")),
                fmt_time(metrics.get("first_enemy_kill_time_s")),
                "入选" if candidate.label == config.selected_label else "未入选",
            ])
        elif config.scenario_id == "S2":
            rows.append([
                candidate.label,
                f"{fmt_count(metrics.get('enemy_kill_count'))} / {fmt_count(metrics.get('friendly_loss_count'))}",
                fmt_time(metrics.get("enemy_awacs_denied_time_s")),
                fmt_time(metrics.get("first_gate_pass_time_s")),
                fmt_pct(relay_delivery_rate(metrics)),
                fmt_pct(metrics.get("radar_any_track_continuity_post_activation")),
                fmt_time(metrics.get("high_risk_breach_time_s")),
                "入选" if candidate.label == config.selected_label else "未入选",
            ])
        else:
            rows.append([
                candidate.label,
                f"{fmt_count(metrics.get('enemy_kill_count'))} / {fmt_count(metrics.get('friendly_loss_count'))}",
                fmt_time(metrics.get("first_gate_pass_time_s")),
                fmt_time(metrics.get("first_friendly_loss_time_s")),
                fmt_time(metrics.get("high_risk_breach_time_s")),
                fmt_dist(metrics.get("min_enemy_to_friendly_km")),
                fmt_pct(metrics.get("intent_truth_accuracy")),
                "入选" if candidate.label == config.selected_label else "未入选",
            ])
    return rows


def extract_decision_row(decision_df: pd.DataFrame, target_time: float) -> Dict[str, Any]:
    mask = (decision_df["time_s"] - target_time).abs() < 1e-6
    if not mask.any():
        mask = (decision_df["time_s"].round(1) == round(target_time, 1))
    row = decision_df.loc[mask].iloc[0]
    return row.to_dict()


def stage_reason_parts(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not text:
        return result
    for key in ("ctx_dist", "pair_dist"):
        match = re.search(rf"{key}=([0-9.]+)km", text)
        if match:
            result[key] = to_float(match.group(1))
    for key in ("node", "zone", "awacs", "missile_incoming", "rwr"):
        match = re.search(rf"{key}=([^ ]+)", text)
        if match:
            result[key] = match.group(1)
    return result


def tactic_reason_parts(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not text:
        return result
    for key in ("pair", "target", "tactic", "src", "enemy_phase"):
        match = re.search(rf"{key}=([^ ]+)", text)
        if match:
            result[key] = match.group(1)
    return result


def parameter_reason_parts(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not text:
        return result
    match = re.search(r"stable_required_s=([0-9.]+)", text)
    if match:
        result["stable_required_s"] = to_float(match.group(1))
    match = re.search(r"gate_phase=([^ ]+)", text)
    if match:
        result["gate_phase"] = match.group(1)
    match = re.search(r"first_window=\[TR:([0-9.]+),LR:([0-9.]+)\]", text)
    if match:
        result["tr"] = to_float(match.group(1))
        result["lr"] = to_float(match.group(2))
    match = re.search(r"second_window=\[TR':([0-9.]+),LR':([0-9.]+)\]", text)
    if match:
        result["tr_prime"] = to_float(match.group(1))
        result["lr_prime"] = to_float(match.group(2))
    match = re.search(r"survival=\[DOR:([0-9.]+),DR:([0-9.]+),MAR:([0-9.]+)\]", text)
    if match:
        result["dor"] = to_float(match.group(1))
        result["dr"] = to_float(match.group(2))
        result["mar"] = to_float(match.group(3))
    match = re.search(r"closure_rate=([-0-9.]+)mps", text)
    if match:
        result["closure_rate"] = to_float(match.group(1))
    match = re.search(r"target_speed=([-0-9.]+)mps", text)
    if match:
        result["target_speed"] = to_float(match.group(1))
    for key in ("NLT", "MELD", "MTR"):
        match = re.search(rf'"{key}": ([0-9.]+)', text)
        if match:
            result[key.lower()] = to_float(match.group(1))
    return result


def maneuver_reason_parts(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not text:
        return result
    match = re.search(r"distance=([0-9.]+)km", text)
    if match:
        result["distance"] = to_float(match.group(1))
    for key in ("shooter", "support", "node", "geometry"):
        match = re.search(rf"{key}=([^ ]+)", text)
        if match:
            result[key] = match.group(1)
    return result


TACTIC_LABEL_ZH_SHORT = {
    "UNKNOWN": "未启用",
    "DRAG_SHOOT": "拖曳射击",
    "PINCER_ATTACK": "钳形夹击",
    "FRONT_BACK": "前后夹击",
    "HIGH_LOW": "高低配合",
    "SIDE_BY_SIDE": "并列压制",
    "TACTICAL_EVASION": "战术规避",
    "TACTICAL_TURN": "战术回转",
}

PHASE_LABEL_ZH_SHORT = {
    "BEYOND_NLT": "NLT外",
    "NLT_MELD": "NLT-MELD",
    "MELD_MTR": "MELD-MTR",
    "MTR_LR": "MTR-LR",
    "LR_TR": "LR-TR",
    "TR_DOR": "TR-DOR",
    "DOR_DR": "DOR-DR",
    "DR_MAR": "DR-MAR",
    "BEYOND_MAR": "MAR后整理",
}

THREAT_LABEL_ZH = {
    "none": "无",
    "low": "低",
    "medium": "中",
    "high": "高",
    "critical": "临界",
}

ZONE_LABEL_ZH = {
    "LOW": "低风险区",
    "MEDIUM": "中风险区",
    "HIGH": "高风险区",
}


def tactic_short_label(name: Any) -> str:
    text = str(name).strip()
    return TACTIC_LABEL_ZH_SHORT.get(text, text)


def phase_short_label(name: Any) -> str:
    text = str(name).replace("TacticalPhase.", "").strip()
    return PHASE_LABEL_ZH_SHORT.get(text, text)


def threat_short_label(name: Any) -> str:
    text = str(name).strip().lower()
    return THREAT_LABEL_ZH.get(text, text)


def zone_short_label(name: Any) -> str:
    text = str(name).strip().upper()
    return ZONE_LABEL_ZH.get(text, text)


def wrap_cn(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=True, break_on_hyphens=False))


def scene_decision_case_rows(config: ScenarioConfig, package: Dict[str, Any]) -> List[List[str]]:
    return scene_decision_role_rows(config, package)


def scene_decision_role_rows(config: ScenarioConfig, package: Dict[str, Any]) -> List[List[str]]:
    key_rows = package["key_rows"]

    def state_role(row: Dict[str, Any], side: str, trigger_text: str, role_text: str) -> str:
        stage = stage_reason_parts(str(row.get(f"{side}_stage_reason", "")))
        return f"{trigger_text} 当前参考距离{fmt_dist(stage.get('ctx_dist'))}、配对距离{fmt_dist(stage.get('pair_dist'))}、所处节点为{phase_short_label(stage.get('node'))}、风险区为{zone_short_label(stage.get('zone'))}。{role_text}"

    def tactic_role(row: Dict[str, Any], left_text: str, right_text: str, role_text: str) -> str:
        return f"左路采用{tactic_short_label(row.get('left_tactic'))}，右路采用{tactic_short_label(row.get('right_tactic'))}。{left_text}{right_text}{role_text}"

    def maneuver_role(row: Dict[str, Any], side: str, role_text: str) -> str:
        reason = maneuver_reason_parts(str(row.get(f"{side}_maneuver_reason", "")))
        return f"射手/支援分工为{reason.get('shooter', '-')}/{reason.get('support', '-')}，执行节点为{phase_short_label(reason.get('node'))}，当前执行距离约{fmt_dist(reason.get('distance'))}。{role_text}"

    def parameter_role(row: Dict[str, Any], side: str, role_text: str) -> str:
        param = parameter_reason_parts(str(row.get(f"{side}_parameter_reason", "")))
        snap = parse_json_field(row.get(f"{side}_decision_snapshot_json"))
        shooter = snap.get("shooter_maneuver_snapshot", {}) if isinstance(snap, dict) else {}
        meta = shooter.get("metadata", {}) if isinstance(shooter, dict) else {}
        parts = [
            f"稳定门限取{param.get('stable_required_s', STABLE_READY_WINDOW_S):.1f} s",
            f"首轮窗口LR/TR={fmt_dist(param.get('lr'))}/{fmt_dist(param.get('tr'))}",
            f"生存边界DOR/DR/MAR={fmt_dist(param.get('dor'))}/{fmt_dist(param.get('dr'))}/{fmt_dist(param.get('mar'))}",
        ]
        if isinstance(shooter, dict) and shooter.get("min_dwell_s") is not None:
            parts.append(f"当前机动最小保持时间为{to_float(shooter.get('min_dwell_s')):.1f} s")
        if isinstance(shooter, dict) and shooter.get("selected_heading_deg") is not None and not pd.isna(shooter.get("selected_heading_deg")):
            parts.append(f"期望航向约为{to_float(shooter.get('selected_heading_deg')):.1f}°")
        if isinstance(meta, dict) and meta.get("entry_floor_km") is not None:
            parts.append(f"硬底线进入阈值为{to_float(meta.get('entry_floor_km')):.1f} km，释放阈值为{to_float(meta.get('release_floor_km')):.1f} km")
        return "；".join(parts) + f"。{role_text}"

    if config.scenario_id == "S1":
        gate = key_rows["first_gate"]
        evade = key_rows["first_evade"]
        reengage = key_rows["reengage"]
        return [
            [
                "首次放行窗口",
                state_role(gate, "left", "阶段层在该窗口内的作用不是给出发射动作本身，而是维持交战状态并拒绝过早放行。", "它把稳定跟踪已经形成与是否允许首轮攻击严格区分开，保证首轮攻击只能在控制距离进入首次攻击准备段后才被执行。"),
                tactic_role(gate, "左路围绕B0100保持钳形夹击，右路围绕B0300保持前后夹击。", "", "战术层的作用在于先形成左右分工，再在模板不抖动的前提下把首轮攻击责任固定下来，因此该时刻选的是能够同时维持压缩与支撑的模板组合，而不是简单地谁近谁打。"),
                maneuver_role(gate, "left", "机动层在这里解决的不是‘能不能打’，而是‘由谁打、谁负责支撑、压缩是否继续保持’。因此射手继续占位而非首发后立刻脱离，目的在于保证首轮导弹发射后仍有稳定的中段责任链。"),
                parameter_role(gate, "left", "参数层的作用是把‘稳定就绪’进一步收缩成‘满足门限的稳定就绪’。104.4 s虽然已经形成稳定交战就绪，但266.6 s之前配对距离始终高于LR窗口，因此未采用更激进的提前放行方案，而是继续等待窗口闭合。"),
                "该窗口说明四层并非串行复述，而是逐层收紧约束：阶段层先防止抢射，战术层固定左右分工，机动层维持首轮责任链，参数层最终以距离门限决定何时真正放行。",
            ],
            [
                "首次规避窗口",
                state_role(evade, "left", "阶段层在该窗口内的核心作用是优先级切换。", "一旦导弹来袭触发，原有交战状态即被强制打断，说明阶段层首先解决的是‘是否继续交战’而不是‘怎样把当前攻击做完’。"),
                tactic_role(evade, "左路与右路都转入前后夹击骨架。", "", "这并不表示战术层在规避时重新追求攻击效果，而是说明原模板让位于更有利于脱离与再组织的骨架形式，战术层在这里的作用是保持编队结构可恢复，而不是继续压缩。"),
                maneuver_role(evade, "left", "机动层的决定性作用在于即时打断。左路被优先止压，原因不是时间到点，而是该路已经压到DOR-DR邻近区，再继续维持攻击姿态会直接消耗后续重组余度。"),
                parameter_role(evade, "left", "参数层此时给出的不是攻击收益参数，而是生存边界参数。配对距离54.1 km已经逼近DOR=55.6 km与DR=51.6 km，这意味着未采用继续压缩的保守等待方案，因为继续等待只会把剩余缓冲压缩到被动兜底的MAR区间。"),
                "该窗口证明规避决策的实质是‘来袭告警优先级’与‘生存边界压缩程度’共同作用的结果，而不是依据结果轨迹事后反推其合理性。",
            ],
            [
                "恢复交战窗口",
                state_role(reengage, "right", "阶段层在该窗口内先恢复截获重组、后恢复交战。", "其作用是用状态机把‘重新可打’与‘立即回打’区分开，避免规避结束后在同一距离带内发生来回震荡。"),
                tactic_role(reengage, "左路在该时刻已经不再承担主导清场责任。", "右路继续以B0400为清场对象维持前后夹击。", "战术层的作用在于把末段任务压缩到剩余目标上，使恢复后的攻击不再追求重新展开整场压制，而是围绕最后清场对象重新分配责任。"),
                maneuver_role(reengage, "right", "机动层在这里的关键不是再次前出，而是先把规避后的骨架补齐。右路射手重新回到MTR-LR附近占位，说明恢复阶段优先解决的是‘编队重组是否完成’，而不是‘最快多久重新接敌’。"),
                parameter_role(reengage, "right", "参数层在这里体现为固定10 s最小保持约束与新的窗口重置。该取值一方面排除了0 s直接回打所带来的状态抖动，另一方面也未采用20 s以上的过度等待方案，以避免末段最后一架敌机的清场窗口继续后移。"),
                "该窗口说明恢复动作的本质是重组约束下的再攻击，而不是简单地把规避结束视为继续交战的充分条件。",
            ],
        ]

    if config.scenario_id == "S2":
        gate = key_rows["first_gate"]
        evade = key_rows["first_evade"]
        reengage = key_rows["reengage"]
        second_evade = key_rows["second_evade"]
        second_reengage = key_rows["second_reengage"]
        return [
            [
                "首次放行窗口",
                state_role(gate, "left", "阶段层在信息缺口条件下仍保持交战状态。", "它的作用不是用缺口为抢射背书，而是要求在图景不完整时继续维持压制骨架，直到窗口真正闭合才允许攻击进入执行阶段。"),
                tactic_role(gate, "", "", "战术层在这里强调的是横向覆盖和协同稳定，而不是尽快收口。之所以选并列压制而未采用钳形夹击或前后夹击，是因为该场景首先要解决图景缺口下的稳态压制问题，而不是首轮打击穿透问题。"),
                maneuver_role(gate, "left", "机动层维持宽正面展开，使每一路都保留搜索与支援角度。它在这里的作用是为后续缺口阶段保留恢复余量，而不是为一次更早的首发压缩出更激进的几何。"),
                parameter_role(gate, "left", "参数层并未因预警失配而放宽门线。54.8 s稳定交战就绪后仍延迟到236.2 s放行，说明5 s稳定门限、LR/TR窗口和生存边界参数在缺口场景下仍保持同一口径。"),
                "该窗口表明信息缺口下的方法有效性，不来自‘看不清就先打’，而来自‘在缺口下仍坚持同一套放行标准’。",
            ],
            [
                "第一次规避窗口",
                state_role(evade, "left", "阶段层在第一次规避窗口内解决的是是否继续维持压制的问题。", "导弹来袭出现后，交战状态被立即打断，说明阶段层并未把‘当前仍有攻击收益’置于生存优先之上。"),
                tactic_role(evade, "左路转入前后夹击骨架。", "右路仍保留并列压制骨架。", "战术层在这里的作用不是切换到一个新的进攻模板，而是让左右路各自保留最有利于重组恢复的结构，从而保证第一次止压后仍能回到持续压制节奏。"),
                maneuver_role(evade, "left", "机动层决定先止压左路，因为该路压得更深。也就是说，机动层并不是统一地下达‘全体退出’，而是根据局部几何压缩程度决定哪一路必须先打断执行。"),
                parameter_role(evade, "left", "参数层给出的DOR/DR/MAR=49.5/45.5/38.0 km表明，此时55.3 km的配对距离已经进入高压带。继续前压的代价不是抽象风险，而是后续恢复空间被直接透支。"),
                "该窗口证明第一次规避的意义不在于动作本身，而在于系统能够把持续压制的上限精确截在生存边界之前。",
            ],
            [
                "第一次恢复窗口",
                state_role(reengage, "left", "阶段层先回到截获重组，再恢复交战。", "它的作用是把‘图景重新可用’和‘编队已经完成重组’两个条件捆绑起来，避免把图景恢复误判为立即可以回打。"),
                tactic_role(reengage, "左路保持原有主压对象。", "右路继续维持对剩余目标的支撑责任。", "战术层在恢复阶段没有重起模板，而是维持原骨架，说明其作用是保证恢复后的连续性，而不是追求每次恢复都重开一套最优分工。"),
                maneuver_role(reengage, "left", "机动层在10 s恢复缓冲中优先完成回转后的再编组，因此恢复动作的重点是结构稳定，而不是最短时间回到压制姿态。"),
                parameter_role(reengage, "left", "参数层体现为固定10 s最小保持约束。它说明恢复时机不是凭经验决定，而是由状态机约束给出的最小稳定时间；该取值排除了图景一恢复就立刻回打的激进方案，同时避免把持续压制窗口继续后推。"),
                "该窗口说明恢复韧性不是‘恢复了就继续打’，而是‘重组完成后再继续打’。",
            ],
            [
                "第二次规避与再恢复窗口",
                state_role(second_evade, "right", "阶段层在第二轮仍保持与第一轮相同的切换纪律。", "这说明信息缺口场景的有效性不是某一次偶然成功，而是状态切换逻辑能够重复执行。"),
                tactic_role(second_reengage, "", "", "战术层此时由并列压制转向前后夹击，作用在于强化照射接续和脱离准备。这一变化表明模板选择是随场面压力变化而调整的，而不是一套模板从头用到尾。"),
                maneuver_role(second_evade, "right", "机动层在第二轮明确启用了高优先级的 `bvr_hard_escape`。这意味着该层不仅决定‘是否打断’，还决定‘采用何种脱离方式’，其本质是把来袭风险转化为具体的脱离动作。"),
                parameter_role(second_evade, "right", "参数层给出了硬底线进入阈值40.0 km、释放阈值52.0 km、期望航向约157.5°、最小保持时间5.0 s，且当前配对距离43.4 km已逼近DOR=44.0 km。因此第二轮不是泛泛的‘再次规避’，而是在明确硬底线规则约束下执行参数化脱离。"),
                "该窗口真正证明的是：恢复韧性来自规则的可重复执行，而不是靠一次规避后运气好又接上了链路。",
            ],
        ]

    first_gate = key_rows["first_gate"]
    first_evade = key_rows["first_evade"]
    first_reengage = key_rows["reengage"]
    second_evade = key_rows["second_evade"]
    late_reengage = key_rows["fifth_reengage"]
    return [
        [
                "首轮前移窗口",
                state_role(first_gate, "left", "阶段层在开局即切入交战。", "其作用不是单纯地把时间轴提前，而是判定传统远距塑形已不适用，因此把整条交战链从一开始就推到可执行准备状态。"),
                tactic_role(first_gate, "", "", "战术层在这里形成的是‘一路前出、一路保持支援’的非对称结构。之所以未采用双路并列压制，是因为高压低空样本更需要尽早形成可打链路，而不是保守维持宽正面。"),
                maneuver_role(first_gate, "left", "机动层在该窗口内的主要作用是缩短塑形时间，把射手尽快推进到MTR-LR附近，而不是像标准场景那样长期占位等待。"),
                parameter_role(first_gate, "left", "参数层虽然仍保留5 s稳定门限，但通过更紧的窗口推进把首轮放行前移到64.0 s。也就是说，前移的是满足门限的时间，而不是取消门限本身。"),
                "该窗口说明高压低空场景中的‘早打’不是冒进，而是交战链整体前移后的必然结果。",
            ],
        [
            "第一次规避窗口",
            state_role(first_evade, "left", "阶段层在第一次规避窗口内优先解决的是生存问题。", "导弹来袭出现后，交战状态被立即打断，说明此时系统首先考虑的是是否还能继续承受高压，而不是当前攻击是否还有收益。"),
            tactic_role(first_evade, "左路仍保持前后夹击骨架。", "右路保持钳形夹击。", "战术层的作用在这里不是继续扩大战果，而是让原攻击结构有序让位于止压动作，保证规避后仍具备恢复可能。"),
            maneuver_role(first_evade, "left", "机动层决定立即止压左路，因为该路已进入高风险承压区。它解决的是‘哪一路必须先退、先退到什么程度’这一问题，而不是笼统地下达一个全局规避口令。"),
            parameter_role(first_evade, "left", "参数层显示DOR/DR/MAR=51.5/47.5/39.3 km，当前配对距离53.4 km且目标位于高风险区。这意味着继续压缩只会把剩余缓冲压缩到被动规避区间，因此该时刻规避是由高风险承压和导弹来袭共同决定的。"),
            "该窗口说明第一次规避的根本作用是主动截断高风险滞留，而不是等到底线被踩穿后再被动脱离。",
        ],
        [
                "恢复缓冲窗口",
                state_role(first_reengage, "right", "阶段层在261.4-271.4 s之间先恢复到截获重组，再恢复交战。", "其作用是把‘可以重新进入交战’与‘必须先完成重组’严格区分开。"),
                tactic_role(first_reengage, "左路继续承担局部压制。", "右路围绕剩余目标继续保持钳形夹击。", "战术层在这里并未追求更激进的火力扩张，而是优先恢复压制骨架和守区职责，说明恢复不是重新开一局，而是在剩余任务约束下重新分配责任。"),
                maneuver_role(first_reengage, "right", "机动层在该窗口中完成回转后的再占位，因此其作用是恢复可继续执行的编队结构，而不是最快速度把机头重新指向敌方。"),
                parameter_role(first_reengage, "right", "参数层固定保留10 s最小保持约束。该取值避免0 s直接回打带来的规避后振荡，同时未采用更长等待方案，以免继续压缩低空突防目标的可用窗口。"),
                "该窗口说明10 s缓冲是恢复条件的一部分，而不是对结果的叙述性修饰。",
            ],
        [
            "第二次规避与后段恢复窗口",
            state_role(second_evade, "right", "阶段层在第二次规避时再次把来袭告警置于最高优先级。", "而在损失后段又继续通过INTERCEPT到ENGAGE的切换维持守区逻辑，说明系统并未因出现损失而放弃统一的状态纪律。"),
            tactic_role(late_reengage, "左路后段转入前后夹击守区。", "右路后段同样维持前后夹击。", "战术层的作用已从扩大战果转向围绕剩余目标守住控制区，因此后段模板选择强调的是可恢复、可守区，而不是进一步扩大战斗正面。"),
            maneuver_role(late_reengage, "right", "机动层在第二次规避时并未等待压到DOR以下才动作，而在后段恢复时又继续执行止压-重组-回压纪律。这说明机动层真正控制的是生存余度分配，而不是单一的转弯动作。"),
            parameter_role(late_reengage, "right", "第二次规避时右路配对距离60.8 km虽高于DOR=42.6 km，但来袭告警仍直接触发打断，说明参数层并非只盯住单一距离；后段1170.0 s左路 `defensive_hold` 的最小保持时间提升到8.0 s、期望航向固定为346.0°，右路 `bvr_hard_escape` 仍保持5.0 s与156.0°脱离航向，这表明损失后的恢复已转入更保守的参数配置。"),
            "该窗口说明高压样本真正验证的是生存优先规则是否贯穿全程：既能提前打断，也能在损失后用更保守的参数继续守住剩余边界。",
        ],
    ]


def scene_decision_case_summary(config: ScenarioConfig) -> str:
    if config.scenario_id == "S1":
        return "场景一的分析重点不是把放行、规避、恢复三个时刻重新讲一遍，而是说明四层在这三个窗口内分别承担了什么作用：阶段层负责约束时机，战术层负责固定分工，机动层负责保持责任链，参数层负责用窗口门限和重组缓冲把动作真正卡实。"
    if config.scenario_id == "S2":
        return "场景二真正需要说明的是同一套决策规则如何在两轮“规避—重组—再交战”中重复兑现：阶段层不因失配抢射，战术层在不同轮次切换不同压制模板，机动层按受压程度决定哪一路先止压，参数层则用同一套边界和缓冲约束保证恢复具有可重复性。"
    return "场景三的分析应当落到生存优先规则本身：阶段层把交战链整体前移，战术层允许非对称模板提前建立拦截骨架，机动层在高压和来袭下优先分配生存余度，参数层则通过更保守的脱离底线、保持时间和恢复航向把损失尽量后推。"


def scene_decision_layer_summary_rows(config: ScenarioConfig, package: Dict[str, Any]) -> List[List[str]]:
    key_rows = package["key_rows"]
    if config.scenario_id == "S1":
        gate = key_rows["first_gate"]
        evade = key_rows["first_evade"]
        reengage = key_rows["reengage"]
        gate_stage = stage_reason_parts(str(gate["left_stage_reason"]))
        gate_param = parameter_reason_parts(str(gate["left_parameter_reason"]))
        evade_param = parameter_reason_parts(str(evade["left_parameter_reason"]))
        return [
            [
                "阶段决策层",
                "将“稳定交战就绪”“允许首轮放行”“强制规避”和“恢复交战”拆分为不同状态门，主导整场节奏的启闭。",
                f"104.4 s已形成稳定交战就绪，但直到{fmt_time(config.decision_times['first_gate'])}才在{phase_short_label(gate_stage.get('node'))}放行；{fmt_time(config.decision_times['first_evade'])}直接切入规避；{fmt_time(config.decision_times['first_intercept'])}-{fmt_time(config.decision_times['reengage'])}保留10 s重组缓冲。",
                "说明阶段层控制的是动作能否发生的状态门，而不是对既有过程的事后命名。",
            ],
            [
                "战术决策层",
                "通过左右路异步模板分工，把首轮压缩与末段清场分开组织，避免交战中频繁换模导致责任漂移。",
                f"首次放行前后左路保持{tactic_short_label(gate.get('left_tactic'))}、右路保持{tactic_short_label(gate.get('right_tactic'))}；恢复阶段右路继续围绕{reengage.get('right_target', 'B0400')}承担末段清场责任。",
                "说明战术层解决的是‘由哪种模板组织当前任务’，而不是‘看到机会后临时改用更激进模板’。",
            ],
            [
                "机动决策层",
                "把模板展开为可执行的射手/支援动作，并在生存余度被压缩时立即打断继续前压。",
                f"首轮放行时左路与右路均维持射手-支援责任链；{fmt_time(config.decision_times['first_evade'])}左路先止压；恢复阶段优先补齐骨架后再回打。",
                "说明机动层负责把模板落地，并在来袭告警与局部压缩过深时执行分级打断。",
            ],
            [
                "参数决策层",
                "以稳定门限、首轮窗口、生存边界和重组缓冲共同限定动作幅度，是解释具体参数取值的核心层级。",
                f"稳定门限取{gate_param.get('stable_required_s', STABLE_READY_WINDOW_S):.1f} s，首轮窗口LR/TR={fmt_dist(gate_param.get('lr'))}/{fmt_dist(gate_param.get('tr'))}，规避前生存边界DOR/DR/MAR={fmt_dist(evade_param.get('dor'))}/{fmt_dist(evade_param.get('dr'))}/{fmt_dist(evade_param.get('mar'))}，恢复缓冲固定为{STATE_MACHINE_MIN_HOLD_S:.0f} s。",
                "说明266.6 s能放行、322.0 s必须止压、456.6 s才恢复交战，都对应明确的参数约束而非经验判断。",
            ],
        ]

    if config.scenario_id == "S2":
        gate = key_rows["first_gate"]
        first_evade = key_rows["first_evade"]
        second_evade = key_rows["second_evade"]
        gate_param = parameter_reason_parts(str(gate["left_parameter_reason"]))
        first_stage = stage_reason_parts(str(first_evade["left_stage_reason"]))
        first_param = parameter_reason_parts(str(first_evade["left_parameter_reason"]))
        second_snap = parse_json_field(second_evade["right_decision_snapshot_json"])
        second_shooter = second_snap.get("shooter_maneuver_snapshot", {}) if isinstance(second_snap, dict) else {}
        second_meta = second_shooter.get("metadata", {}) if isinstance(second_shooter, dict) else {}
        return [
            [
                "阶段决策层",
                "在33.3%预警失配下仍保持交战状态，并把两轮“规避—重组—再交战”维持为同一套切换纪律。",
                f"{fmt_time(config.decision_times['first_gate'])}完成首次放行；{fmt_time(config.decision_times['first_evade'])}与{fmt_time(config.decision_times['second_evade'])}两次切规避；{fmt_time(config.decision_times['first_intercept'])}-{fmt_time(config.decision_times['reengage'])}及{fmt_time(config.decision_times['second_intercept'])}-{fmt_time(config.decision_times['second_reengage'])}两段恢复缓冲均为10 s。",
                "说明阶段层在信息缺口场景中并未放弃压制，也没有把恢复动作交给临场随机性处理。",
            ],
            [
                "战术决策层",
                "先用并列压制维持宽正面覆盖，再在第二轮恢复后转入前后夹击以强化照射接续和脱离准备。",
                f"首次放行窗口左右路均采用{tactic_short_label(gate.get('left_tactic'))}；第二轮恢复后统一转入{tactic_short_label(key_rows['second_reengage'].get('right_tactic'))}骨架。",
                "说明战术层的调整依据是场面压力和链路接续需求，而不是简单沿用同一模板推进到底。",
            ],
            [
                "机动决策层",
                "按局部受压深浅决定哪一路先止压，并把‘规避—重组—再压制’组织成可重复的动作节奏。",
                f"第一次规避时左路配对距离{fmt_dist(first_stage.get('pair_dist'))}，压得更深而先止压；第二次规避时右路进入 `bvr_hard_escape`，表明机动层已将来袭风险转换为明确脱离动作。",
                "说明机动层不仅负责执行模板，还负责把恢复韧性落实到可重复的动作链上。",
            ],
            [
                "参数决策层",
                "在图景缺口下仍坚持同一套门线，同时用硬底线阈值和脱离航向约束第二轮高压规避。",
                f"首次放行阶段仍取{gate_param.get('stable_required_s', STABLE_READY_WINDOW_S):.1f} s稳定门限与LR/TR={fmt_dist(gate_param.get('lr'))}/{fmt_dist(gate_param.get('tr'))}；第二次规避时进入阈值{fmt_dist(second_meta.get('entry_floor_km'))}、释放阈值{fmt_dist(second_meta.get('release_floor_km'))}、期望航向约{to_float(second_shooter.get('selected_heading_deg')):.1f}°、最小保持{to_float(second_shooter.get('min_dwell_s')):.1f} s。",
                "说明信息缺口没有被用来放宽门线，恢复与脱离仍由硬参数而非主观放松来支撑。",
            ],
        ]

    first_gate = key_rows["first_gate"]
    first_evade = key_rows["first_evade"]
    second_evade = key_rows["second_evade"]
    late_intercept = key_rows["fifth_intercept"]
    late_reengage = key_rows["fifth_reengage"]
    first_param = parameter_reason_parts(str(first_evade["left_parameter_reason"]))
    second_param = parameter_reason_parts(str(second_evade["right_parameter_reason"]))
    late_left_snap = parse_json_field(late_intercept["left_decision_snapshot_json"])
    late_right_snap = parse_json_field(late_intercept["right_decision_snapshot_json"])
    late_left = late_left_snap.get("shooter_maneuver_snapshot", {}) if isinstance(late_left_snap, dict) else {}
    late_right = late_right_snap.get("shooter_maneuver_snapshot", {}) if isinstance(late_right_snap, dict) else {}
    late_right_meta = late_right.get("metadata", {}) if isinstance(late_right, dict) else {}
    return [
        [
            "阶段决策层",
            "在高压近距条件下把交战链整体前移，并在后段出现损失后仍保持规避、重组和再交战的统一纪律。",
            f"{fmt_time(config.decision_times['engage'])}即切入交战；{fmt_time(config.decision_times['first_evade'])}和{fmt_time(config.decision_times['second_evade'])}两次直接切规避；出现首次损失后仍在{fmt_time(config.decision_times['fifth_intercept'])}-{fmt_time(config.decision_times['fifth_reengage'])}完成一次完整恢复。",
            "说明阶段层在场景三控制的是生存优先下的节奏分配，而不是简单把后段动作解释为撤退或追击。",
        ],
        [
            "战术决策层",
            "允许非对称模板提前建立拦截骨架，并在后段把任务目标从扩大战果转向守区抑制。",
            f"首轮放行时左路采用{tactic_short_label(first_gate.get('left_tactic'))}、右路采用{tactic_short_label(first_gate.get('right_tactic'))}；后段恢复时左右两路统一转为{tactic_short_label(late_reengage.get('left_tactic'))}守区骨架。",
            "说明战术层在高压样本中首先服务于可打链路的尽早建立，其次服务于损失后的守区稳定。",
        ],
        [
            "机动决策层",
            "把生存优先落实到具体动作，既包括高风险区内的立即止压，也包括尚未踩穿DOR时对来袭告警的提前打断。",
            f"第一次规避时左路在高风险区内止压；第二次规避时右路虽仍高于DOR={fmt_dist(second_param.get('dor'))}，但仍因来袭告警直接切入规避；后段左路进入 `defensive_hold`，右路保持 `bvr_hard_escape`。",
            "说明机动层真正分配的是生存余度，而不是只负责某一个转弯或脱离动作。",
        ],
        [
            "参数决策层",
            "通过更保守的保持时间、航向和硬底线参数，把高压样本中的生存规则明确参数化。",
            f"首轮前仍采用5 s稳定门限；第一次规避前DOR/DR/MAR={fmt_dist(first_param.get('dor'))}/{fmt_dist(first_param.get('dr'))}/{fmt_dist(first_param.get('mar'))}；后段左路 `defensive_hold` 最小保持{to_float(late_left.get('min_dwell_s')):.1f} s、期望航向{to_float(late_left.get('selected_heading_deg')):.1f}°，右路 `bvr_hard_escape` 最小保持{to_float(late_right.get('min_dwell_s')):.1f} s、期望航向{to_float(late_right.get('selected_heading_deg')):.1f}°、进入/释放阈值为{fmt_dist(late_right_meta.get('entry_floor_km'))}/{fmt_dist(late_right_meta.get('release_floor_km'))}。",
            "说明场景三的有效性不是靠宽松门线换来的，而是靠更保守、更明确的参数把损失尽量后推。",
        ],
    ]


def build_scene_package(config: ScenarioConfig) -> Dict[str, Any]:
    sample_dir = config.selected_dir
    summary = load_metric_map(sample_dir)
    milestones = load_milestone_map(sample_dir)
    decision_df = read_csv_select(
        sample_dir / "tables" / "decision_trace.csv",
        [
            "time_s",
            "cap_state",
            "cap_state_reason",
            "mission_threat_level",
            "left_tactic",
            "right_tactic",
            "left_phase",
            "right_phase",
            "left_target",
            "right_target",
            "stable_ready_count",
            "gate_pass_count",
            "relay_success_count",
            "enemy_kill_count",
            "left_stage_reason",
            "left_tactic_reason",
            "left_maneuver_reason",
            "left_parameter_reason",
            "left_decision_snapshot_json",
            "right_stage_reason",
            "right_tactic_reason",
            "right_maneuver_reason",
            "right_parameter_reason",
            "right_decision_snapshot_json",
        ],
    )
    decision_df["time_s"] = decision_df["time_s"].astype(float)
    timeline_df = read_csv_select(
        sample_dir / "timeline.csv",
        [
            "time_s",
            "cap_state",
            "nearest_enemy_to_friendly_km",
            "stable_ready_count",
            "gate_pass_count",
            "relay_success_count",
            "enemy_kill_count",
            "friendly_loss_count",
            "radar_track_count",
            "awacs_track_count",
            "enemy_script_awacs_state",
            "left_tactic",
            "right_tactic",
            "left_phase",
            "right_phase",
            "zone_by_enemy_json",
            "mission_threat_level",
            "truth_high_count",
            "truth_medium_count",
            "truth_low_count",
        ],
    )
    timeline_df["time_s"] = timeline_df["time_s"].astype(float)
    timeline_df["zone_map"] = timeline_df["zone_by_enemy_json"].apply(parse_json_field)
    events_df = read_csv_select(sample_dir / "events.csv", ["time_s", "event_type", "field", "value"])
    events_df["time_s"] = events_df["time_s"].astype(float)
    control_nodes_df = pd.read_csv(sample_dir / "tables" / "control_distance_nodes.csv")
    control_nodes_df["first_time_s"] = control_nodes_df["first_time_s"].astype(float)
    control_nodes_df["distance_km"] = control_nodes_df["distance_km"].astype(float)
    zone_segments = extract_zone_segments_from_timeline(timeline_df)
    enemy_kill_events = extract_counter_events(timeline_df, "enemy_kill_count")
    friendly_loss_events = extract_counter_events(timeline_df, "friendly_loss_count")

    key_rows: Dict[str, Dict[str, Any]] = {}
    for name, time_value in config.decision_times.items():
        key_rows[name] = extract_decision_row(decision_df, time_value)
    return {
        "summary": summary,
        "milestones": milestones,
        "decision_df": decision_df,
        "timeline_df": timeline_df,
        "events_df": events_df,
        "control_nodes_df": control_nodes_df,
        "zone_segments": zone_segments,
        "enemy_kill_events": enemy_kill_events,
        "friendly_loss_events": friendly_loss_events,
        "key_rows": key_rows,
    }


def build_baseline_package(config: ScenarioConfig) -> Dict[str, Any]:
    sample_dir = config.baseline_dir
    summary = load_metric_map(sample_dir)
    milestones = load_milestone_map(sample_dir)
    timeline_df = read_csv_select(
        sample_dir / "timeline.csv",
        [
            "time_s",
            "enemy_kill_count",
            "friendly_loss_count",
        ],
    )
    if not timeline_df.empty:
        timeline_df["time_s"] = timeline_df["time_s"].astype(float)
    else:
        timeline_df = pd.DataFrame(columns=["time_s", "enemy_kill_count", "friendly_loss_count"])
    return {
        "summary": summary,
        "milestones": milestones,
        "timeline_df": timeline_df,
        "enemy_kill_events": extract_counter_events(timeline_df, "enemy_kill_count") if not timeline_df.empty else [],
        "friendly_loss_events": extract_counter_events(timeline_df, "friendly_loss_count") if not timeline_df.empty else [],
    }


def scene_focus_metric_rows(config: ScenarioConfig, package: Dict[str, Any]) -> List[List[str]]:
    summary = package["summary"]
    milestones = package["milestones"]
    rows: List[List[str]] = []
    for item in config.focus_metric_keys:
        value: Optional[float]
        if item["source"] == "summary":
            value = summary.get(item["key"])
        elif item["source"] == "milestone":
            value = milestones.get(item["key"])
        else:
            if item["key"] == "gate_delay_after_ready":
                value = milestones.get("first_gate_pass_time_s") - milestones.get("first_stable_ready_time_s") if milestones.get("first_gate_pass_time_s") is not None and milestones.get("first_stable_ready_time_s") is not None else None
            elif item["key"] == "first_evade_to_reengage":
                value = config.decision_times["reengage"] - config.decision_times["first_evade"]
            elif item["key"] == "first_evade_to_intercept":
                value = config.decision_times["first_intercept"] - config.decision_times["first_evade"]
            elif item["key"] == "recovery_buffer":
                value = config.decision_times["reengage"] - config.decision_times["first_intercept"]
            elif item["key"] == "first_cycle_total":
                value = config.decision_times["reengage"] - config.decision_times["first_evade"]
            elif item["key"] == "second_cycle_total":
                value = config.decision_times["second_reengage"] - config.decision_times["second_evade"] if "second_reengage" in config.decision_times and "second_evade" in config.decision_times else None
            elif item["key"] == "second_recovery_buffer":
                value = config.decision_times["second_reengage"] - config.decision_times["second_intercept"] if "second_reengage" in config.decision_times and "second_intercept" in config.decision_times else None
            elif item["key"] == "first_gate_lead_after_engage":
                value = milestones.get("first_gate_pass_time_s") - config.decision_times["engage"] if milestones.get("first_gate_pass_time_s") is not None else None
            elif item["key"] == "first_loss_pushback_after_first_evade":
                value = milestones.get("first_friendly_loss_time_s") - config.decision_times["first_evade"] if milestones.get("first_friendly_loss_time_s") is not None else None
            elif item["key"] == "relay_delivery_rate":
                value = relay_delivery_rate(summary)
            else:
                value = None
        rows.append([item["name"], fmt_value(value, item["type"]), item["analysis"]])
    return rows


def sample_id_text(config: ScenarioConfig) -> str:
    return config.selected_label


def import_legacy_figure(filename: str, figure_no: int, target_stem: Optional[str] = None, source_dir: Optional[Path] = None) -> str:
    src_dir = source_dir if source_dir is not None else LEGACY_ASSET_DIR
    src = src_dir / filename
    stem = target_stem if target_stem else Path(filename).stem
    dst_name = f"fig6_{figure_no}_{stem}.png"
    dst = ASSET_DIR / dst_name
    shutil.copy2(src, dst)
    return dst_name


def scene_key_action_rows(config: ScenarioConfig, package: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    key_rows = package["key_rows"]

    def action_row(
        label: str,
        time_value: float,
        row: Dict[str, Any],
        side: str,
        direct_trigger: str,
        geometry_reason: str,
        parameter_reason: str,
        validity: str,
    ) -> List[str]:
        stage = stage_reason_parts(str(row.get(f"{side}_stage_reason", "")))
        param = parameter_reason_parts(str(row.get(f"{side}_parameter_reason", "")))
        pair_dist = fmt_dist(stage.get("pair_dist"))
        ctx_dist = fmt_dist(stage.get("ctx_dist"))
        dor = fmt_dist(param.get("dor"))
        dr = fmt_dist(param.get("dr"))
        mar = fmt_dist(param.get("mar"))
        return [
            label,
            fmt_time(time_value),
            STATE_TEXT.get(str(row.get("cap_state", "")), str(row.get("cap_state", ""))),
            direct_trigger,
            f"参考距离{ctx_dist}，双机配对距离{pair_dist}，阶段节点{describe_phase(str(row.get(f'{side}_phase', '')))}，生存边界DOR/DR/MAR={dor}/{dr}/{mar}。{geometry_reason}",
            parameter_reason,
            validity,
        ]

    if config.scenario_id == "S1":
        rows.append(action_row(
            "首次放行",
            config.decision_times["first_gate"],
            key_rows["first_gate"],
            "left",
            "任务状态保持交战，直接触发原因不是导弹来袭，而是稳定跟踪、发射门窗口和中距压缩条件同时闭合。",
            "此时左路位于MTR-LR段，说明系统是在首次发射准备窗口内放行，而不是在更远的塑形段抢射。",
            "稳定跟踪门限保持5.0 s，首轮窗口约束仍由TR/LR距离门限控制，因此104.4 s已形成稳定交战就绪，仍继续等待到266.6 s才放行。",
            "说明发射门没有被“稳定跟踪已建立”直接替代，放行决策确实遵守了距离窗口与火控质量共同闭合的约束。",
        ))
        rows.append(action_row(
            "首次规避",
            config.decision_times["first_evade"],
            key_rows["first_evade"],
            "left",
            "任务状态由导弹来袭告警直接切入规避，规避的直接触发原因是来袭武器，而不是单纯的时间到点。",
            "左路双机配对距离已经压到54.1 km，并进入DOR-DR段，说明继续前压将直接消耗剩余生存余度。",
            "对应生存边界约束DOR=55.6 km、DR=51.6 km、MAR=38.2 km，说明322.0 s触发规避不是保守，而是对导弹来袭和生存边界共同逼近的必然响应。",
            "说明规避动作并非结果描述，而是由“来袭告警优先级最高”与“左路接近脱离决断边界”两项条件共同触发。",
        ))
        rows.append(action_row(
            "恢复交战",
            config.decision_times["reengage"],
            key_rows["reengage"],
            "right",
            "446.6 s任务状态先回到截获重组，456.6 s再恢复交战，回转并不是在规避刚结束时立即发生，而是由图景恢复和重组完成共同触发。",
            "恢复时右路重新回到MTR-LR段，说明系统不是在DOR附近直接反打，而是先把双机骨架拉回可重新组织的中距段。",
            "截获重组到恢复交战保留10.0 s缓冲，对应状态机最小保持时间约束；若在446.6 s立即回转，状态震荡和目标接续不完整风险会明显增大。",
            "说明10 s不是叙述性数字，而是系统参数约束；该回转时机兼顾了规避后的重组稳定性和后续清场窗口。",
        ))
    elif config.scenario_id == "S2":
        rows.append(action_row(
            "首次放行",
            config.decision_times["first_gate"],
            key_rows["first_gate"],
            "left",
            "虽然外层预警图景已经进入间歇失配循环，但任务状态仍保持交战，首次放行并未因为信息缺口而提前。",
            "左路在236.2 s时进入LR-TR段，说明系统是在持续压制条件下等待可执行窗口，而不是用缺口期冒险抢射。",
            "5.0 s稳定跟踪门限和LR/TR窗口门限保持不变，信息缺口没有带来参数层放宽，因此放行时刻仍严格受门线约束。",
            "说明信息缺口场景下的有效性来自“缺口下仍不放宽门线”，而不是靠激进前压换取表面上的更早发射。",
        ))
        rows.append(action_row(
            "第一次规避",
            config.decision_times["first_evade"],
            key_rows["first_evade"],
            "left",
            "任务状态在导弹来袭后直接切入规避，第一次规避的直接触发原因是来袭武器，而不是预警失配本身。",
            "左路双机配对距离约55.3 km，已逼近DOR=49.5 km与DR=45.5 km之前的高压带，说明继续维持压制将明显缩小后续重组余度。",
            "规避时参数层并未改变门线，而是执行生存边界优先规则；因此规避是对导弹来袭和几何压缩的联合响应。",
            "说明“持续压制”并不意味着无条件前压，而是在逼近边界时能及时止压，为后续再压制保留结构余度。",
        ))
        rows.append(action_row(
            "第一次恢复交战",
            config.decision_times["reengage"],
            key_rows["reengage"],
            "left",
            "390.2 s先恢复到截获重组，400.2 s再恢复交战，直接触发原因是图景重新可用并完成双机骨架重整，而不是简单等待导弹过去。",
            "第一次恢复时仍处于LR-TR压制骨架附近，说明系统并未完全退出前场，而是在可控区间内恢复压制。",
            "截获重组到恢复交战精确保留10.0 s缓冲，对应状态机最小保持约束；这说明390.2 s图景恢复后仍需等待最小重组时间闭合。",
            "说明信息缺口场景下的恢复具有纪律性：每次回转都遵守同一套重组缓冲规则，而不是临场拍脑袋决定。",
        ))
        rows.append(action_row(
            "第二次规避与再恢复",
            config.decision_times["second_evade"],
            key_rows["second_evade"],
            "right",
            "416.8 s第二次规避仍由导弹来袭直接触发，说明第一次恢复后系统并未放弃生存优先级，来袭告警依旧拥有最高打断权。",
            "右路此时双机配对距离约43.4 km，已经压到DOR-DR段内，继续保持压制将直接侵蚀第二轮重组余度。",
            "513.0 s回到截获重组、523.0 s再恢复交战，第二轮依然保留10.0 s恢复缓冲，说明恢复韧性具有可重复性而非偶然发生一次。",
            "说明本场景的核心不是某一次规避成功，而是“规避—重组—再压制”能够在信息缺口条件下重复成立。",
        ))
    else:
        rows.append(action_row(
            "切入交战与首轮放行",
            config.decision_times["first_gate"],
            key_rows["first_gate"],
            "left",
            "0.2 s即切入交战，说明阶段层判定敌方一开始就处于中高风险接敌条件；64.0 s完成首轮放行，直接体现交战链前移。",
            "首轮放行时左路已推进到MTR-LR段，而不是像场景一那样经历更长的塑形过程，说明低空突防样本必须把火控链前置。",
            "5.0 s稳定跟踪门限保持不变，但首轮放行相对交战切入仅间隔63.8 s，参数层体现为“同一套门线框架下的前移执行”，而不是放弃约束。",
            "说明高压低空样本的有效性不在于盲目提前，而在于在不放弃门线约束的前提下尽量前移首轮攻击窗口。",
        ))
        rows.append(action_row(
            "第一次规避",
            config.decision_times["first_evade"],
            key_rows["first_evade"],
            "left",
            "任务状态由导弹来袭直接切入规避，第一次规避的首要触发原因仍然是来袭武器。",
            "左路双机配对距离约53.4 km，已进入DOR-DR段，同时目标处于高风险区，说明此时继续压制会同时放大导弹威胁和风险区滞留时间。",
            "参数层对应的生存边界为DOR=51.5 km、DR=47.5 km、MAR=39.3 km，因此131.8 s的规避并非过早，而是对高压边界的及时止压。",
            "说明第一次规避的有效性在于把高风险区停留限制在可接受区间，并为后续重新组织争取时间。",
        ))
        rows.append(action_row(
            "第一次恢复交战",
            config.decision_times["reengage"],
            key_rows["reengage"],
            "right",
            "261.4 s先回到截获重组，271.4 s再恢复交战，说明止压后不是立即回打，而是先完成战术骨架重组。",
            "恢复时右路回到MELD-MTR段，意味着系统先退回到可重新组织的控制距离层级，再重新建立压制，而不是在高风险段硬顶回转。",
            "恢复缓冲仍严格保持10.0 s，这说明高压近距场景下并没有为了抢时间而破坏参数层的最小重组约束。",
            "说明10 s缓冲是系统性设计而非个例；在高压场景中保持这一底线，才能避免止压后马上再失控。",
        ))
        rows.append(action_row(
            "第二次规避",
            config.decision_times["second_evade"],
            key_rows["second_evade"],
            "right",
            "第二次规避同样由导弹来袭直接触发，但其几何背景与第一次不同，体现了来袭告警拥有独立于风险区节点的最高优先级。",
            "右路双机配对距离约60.8 km，处于MTR-LR段，说明第二次规避并不是因为已经深踩DOR，而是因为在高压样本中来袭告警本身就足以强制打断继续压制。",
            "参数层此时DOR约42.6 km、DR约38.6 km，虽然尚未跌穿底线，但任务状态仍切规避，说明系统没有等到底线被踩穿才动作。",
            "说明场景三的生存优先不是被动兜底，而是主动预留生存余度；这也是首次我机损失能够被后推到990.2 s的重要前提。",
        ))
        rows.append(action_row(
            "损失后末段再恢复",
            config.decision_times["fifth_reengage"],
            key_rows["fifth_reengage"],
            "right",
            "1056.2 s再次因导弹来袭切规避，1170.0 s先恢复到截获重组，1180.0 s再恢复交战，说明首次我机损失之后系统并未退出守区，而是继续按同一套状态机规则执行末段重组。",
            "此时最近威胁距离已恢复到71.7 km量级，右路重新把敌方压回中风险边界附近，说明末段恢复并不是无意义回转，而是为了防止剩余目标继续扩张控制区。",
            "重组缓冲依然严格保持10.0 s，表明即使在损失已经发生的后段，参数层也没有放弃最小保持约束去冒险抢压。",
            "说明生存优先并不等于遭受损失后立即放弃任务，而是在损失出现后仍按同一套规避、重组和回转纪律守住剩余战场边界。",
        ))
    return rows


def scene_milestone_rows(package: Dict[str, Any]) -> List[List[str]]:
    milestones = package["milestones"]
    return [
        ["首次预警发现", fmt_time(milestones.get("first_awacs_track_time_s"))],
        ["首次机载雷达跟踪", fmt_time(milestones.get("first_radar_track_time_s"))],
        ["四目标全探完成", fmt_time(milestones.get("full_detect_time_s"))],
        ["首次稳定交战就绪", fmt_time(milestones.get("first_stable_ready_time_s"))],
        ["首次发射门通过", fmt_time(milestones.get("first_gate_pass_time_s"))],
        ["首次接力制导成功", fmt_time(milestones.get("first_relay_success_time_s"))],
        ["首次敌机被击落", fmt_time(milestones.get("first_enemy_kill_time_s"))],
        ["首次我机损失", fmt_time(milestones.get("first_friendly_loss_time_s"))],
    ]


def scene_zone_interval_rows(package: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    for enemy_id, segments in package["zone_segments"].items():
        high = [segment for segment in segments if segment["zone"] == "HIGH"]
        medium = [segment for segment in segments if segment["zone"] == "MEDIUM"]
        low = [segment for segment in segments if segment["zone"] == "LOW"]
        rows.append([
            enemy_id,
            format_interval_list(high),
            format_interval_list(medium),
            format_interval_list(low),
            f"{total_interval_duration(high):.1f} / {total_interval_duration(medium):.1f} / {total_interval_duration(low):.1f}",
        ])
    return rows


def scene_awacs_interval_rows(package: Dict[str, Any]) -> List[List[str]]:
    timeline = package["timeline_df"].copy()
    if "enemy_script_awacs_state" not in timeline.columns:
        return [["预警图景", "无", "无", "无"]]
    segments = extract_state_segments(timeline, "enemy_script_awacs_state", "AVAILABLE")
    denied = [segment for segment in segments if str(segment["value"]) == "DENIED"]
    available = [segment for segment in segments if str(segment["value"]) == "AVAILABLE"]
    if denied:
        first_denied = denied[0]
        last_denied = denied[-1]
        denied_desc = (
            f"{len(denied)}段；首段 {first_denied['start']:.1f}-{first_denied['end']:.1f} s；"
            f"末段 {last_denied['start']:.1f}-{last_denied['end']:.1f} s；"
            f"单段中位时长 {pd.Series([item['duration'] for item in denied]).median():.1f} s"
        )
    else:
        denied_desc = "无"
    return [[
        "预警图景",
        denied_desc,
        f"{total_interval_duration(denied):.1f} s",
        f"{total_interval_duration(available):.1f} s",
    ]]


def scene_event_rows(package: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    for item in package["enemy_kill_events"]:
        rows.append(["敌机击落", fmt_time(item["time_s"]), f"累计击落升至{len([x for x in package['enemy_kill_events'] if x['time_s'] <= item['time_s']])}架"])
    for item in package["friendly_loss_events"]:
        rows.append(["我机损失", fmt_time(item["time_s"]), "累计损失升至1架"])
    return rows


def baseline_compare_rows(config: ScenarioConfig, package: Dict[str, Any], baseline: Dict[str, Any]) -> List[List[str]]:
    cur_s = package["summary"]
    cur_m = package["milestones"]
    base_s = baseline["summary"]
    base_m = baseline["milestones"]
    if config.scenario_id == "S1":
        specs = [
            ("四目标全探完成时间", cur_m.get("full_detect_time_s"), base_m.get("full_detect_time_s"), "time", "越早越好"),
            ("首次稳定交战就绪时间", cur_m.get("first_stable_ready_time_s"), base_m.get("first_stable_ready_time_s"), "time", "越早越好"),
            ("首次发射门通过时间", cur_m.get("first_gate_pass_time_s"), base_m.get("first_gate_pass_time_s"), "time", "越早越好"),
            ("发射窗口兑现率", cur_s.get("gate_pass_per_launch_rate"), base_s.get("gate_pass_per_launch_rate"), "pct", "越高越好"),
            ("雷达探测覆盖率（对存活目标）", cur_s.get("radar_detection_coverage_ratio_alive_targets_post_activation"), base_s.get("radar_detection_coverage_ratio_alive_targets_post_activation"), "pct", "越高越好"),
            ("全覆盖连续性（对存活目标）", cur_s.get("radar_full_coverage_continuity_alive_targets_post_activation"), base_s.get("radar_full_coverage_continuity_alive_targets_post_activation"), "pct", "越高越好"),
            ("敌机击落数", cur_s.get("enemy_kill_count"), base_s.get("enemy_kill_count"), "count", "越高越好"),
            ("我方损失数", cur_s.get("friendly_loss_count"), base_s.get("friendly_loss_count"), "count", "越低越好"),
        ]
    elif config.scenario_id == "S2":
        specs = [
            ("首次稳定交战就绪时间", cur_m.get("first_stable_ready_time_s"), base_m.get("first_stable_ready_time_s"), "time", "越早越好"),
            ("发射窗口兑现率", cur_s.get("gate_pass_per_launch_rate"), base_s.get("gate_pass_per_launch_rate"), "pct", "越高越好"),
            ("接力制导兑现率", relay_delivery_rate(cur_s), relay_delivery_rate(base_s), "pct", "越高越好"),
            ("敌机击落数", cur_s.get("enemy_kill_count"), base_s.get("enemy_kill_count"), "count", "越高越好"),
            ("我方损失数", cur_s.get("friendly_loss_count"), base_s.get("friendly_loss_count"), "count", "越低越好"),
            ("高风险区累计滞留时间", cur_s.get("high_risk_breach_time_s"), base_s.get("high_risk_breach_time_s"), "time", "越低越好"),
            ("中风险区累计滞留时间", cur_s.get("medium_risk_enemy_time_s"), base_s.get("medium_risk_enemy_time_s"), "time", "越短越好"),
        ]
    else:
        specs = [
            ("四目标全探完成时间", cur_m.get("full_detect_time_s"), base_m.get("full_detect_time_s"), "time", "越早越好"),
            ("首次稳定交战就绪时间", cur_m.get("first_stable_ready_time_s"), base_m.get("first_stable_ready_time_s"), "time", "越早越好"),
            ("首次发射门通过时间", cur_m.get("first_gate_pass_time_s"), base_m.get("first_gate_pass_time_s"), "time", "越早越好"),
            ("发射窗口兑现率", cur_s.get("gate_pass_per_launch_rate"), base_s.get("gate_pass_per_launch_rate"), "pct", "越高越好"),
            ("雷达探测覆盖率（对存活目标）", cur_s.get("radar_detection_coverage_ratio_alive_targets_post_activation"), base_s.get("radar_detection_coverage_ratio_alive_targets_post_activation"), "pct", "越高越好"),
            ("全覆盖连续性（对存活目标）", cur_s.get("radar_full_coverage_continuity_alive_targets_post_activation"), base_s.get("radar_full_coverage_continuity_alive_targets_post_activation"), "pct", "越高越好"),
            ("高风险区累计滞留时间", cur_s.get("high_risk_breach_time_s"), base_s.get("high_risk_breach_time_s"), "time", "越低越好"),
            ("高风险区峰值敌机数", cur_s.get("high_risk_peak_count"), base_s.get("high_risk_peak_count"), "count", "越低越好"),
        ]

    rows: List[List[str]] = []
    for name, cur_val, base_val, value_type, direction in specs:
        rows.append([name, fmt_value(cur_val, value_type), fmt_value(base_val, value_type), direction])
    return rows


def scene_baseline_compare_analysis(config: ScenarioConfig, package: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    cur_s = package["summary"]
    cur_m = package["milestones"]
    base_s = baseline["summary"]
    base_m = baseline["milestones"]
    if config.scenario_id == "S1":
        return (
            f"与同场景基线相比，本文方法把四目标全探时间由{fmt_time(base_m.get('full_detect_time_s'))}提前到{fmt_time(cur_m.get('full_detect_time_s'))}，"
            f"首次稳定交战就绪由{fmt_time(base_m.get('first_stable_ready_time_s'))}提前到{fmt_time(cur_m.get('first_stable_ready_time_s'))}，"
            f"首次放行由{fmt_time(base_m.get('first_gate_pass_time_s'))}提前到{fmt_time(cur_m.get('first_gate_pass_time_s'))}。"
            f"与此同时，发射窗口兑现率由{fmt_pct(base_s.get('gate_pass_per_launch_rate'))}提高到{fmt_pct(cur_s.get('gate_pass_per_launch_rate'))}，"
            f"对存活目标的探测覆盖率由{fmt_pct(base_s.get('radar_detection_coverage_ratio_alive_targets_post_activation'))}提高到{fmt_pct(cur_s.get('radar_detection_coverage_ratio_alive_targets_post_activation'))}，"
            f"最终战果由2:1提升到4:0，说明本文方法在标准几何下同时提升了链路闭合速度、窗口兑现质量和最终清场效果。"
        )
    if config.scenario_id == "S2":
        return (
            f"场景二的基线虽然在外层覆盖上更激进，但本文方法把首次稳定交战就绪由{fmt_time(base_m.get('first_stable_ready_time_s'))}大幅提前到{fmt_time(cur_m.get('first_stable_ready_time_s'))}，"
            f"发射窗口兑现率由{fmt_pct(base_s.get('gate_pass_per_launch_rate'))}提高到{fmt_pct(cur_s.get('gate_pass_per_launch_rate'))}，"
            f"接力制导兑现率由{fmt_pct(relay_delivery_rate(base_s))}提高到{fmt_pct(relay_delivery_rate(cur_s))}。"
            f"更关键的是，敌机击落数由{fmt_count(base_s.get('enemy_kill_count'))}提升到{fmt_count(cur_s.get('enemy_kill_count'))}，我方损失由{fmt_count(base_s.get('friendly_loss_count'))}降到{fmt_count(cur_s.get('friendly_loss_count'))}。"
            f"这说明在信息缺口条件下，本文方法的优势不在于表面覆盖率更高，而在于更早形成可用火控、更稳地兑现已发射导弹，并最终换来更低战损。"
        )
    return (
        f"与同场景基线相比，本文方法把四目标全探时间由{fmt_time(base_m.get('full_detect_time_s'))}提前到{fmt_time(cur_m.get('full_detect_time_s'))}，"
        f"首次稳定交战就绪由{fmt_time(base_m.get('first_stable_ready_time_s'))}提前到{fmt_time(cur_m.get('first_stable_ready_time_s'))}，"
        f"首次放行由{fmt_time(base_m.get('first_gate_pass_time_s'))}提前到{fmt_time(cur_m.get('first_gate_pass_time_s'))}。"
        f"同时，发射窗口兑现率由{fmt_pct(base_s.get('gate_pass_per_launch_rate'))}提高到{fmt_pct(cur_s.get('gate_pass_per_launch_rate'))}，"
        f"对存活目标的探测覆盖率由{fmt_pct(base_s.get('radar_detection_coverage_ratio_alive_targets_post_activation'))}提高到{fmt_pct(cur_s.get('radar_detection_coverage_ratio_alive_targets_post_activation'))}，"
        f"高风险区累计滞留则由{fmt_time(base_s.get('high_risk_breach_time_s'))}压缩到{fmt_time(cur_s.get('high_risk_breach_time_s'))}。"
        f"虽然最终击落数持平，但本文方法明显缩短了高风险暴露时间，说明其优势主要体现在生存边界控制而非简单追求更高战果。"
    )


def scene_decision_indicator_rows(config: ScenarioConfig, package: Dict[str, Any]) -> List[List[str]]:
    summary = package["summary"]
    milestones = package["milestones"]
    baseline = build_baseline_package(config)
    base_summary = baseline["summary"]
    base_milestones = baseline["milestones"]
    if config.scenario_id == "S1":
        return [
            [
                "放行克制性",
                fmt_time((milestones.get("first_gate_pass_time_s") or 0.0) - (milestones.get("first_stable_ready_time_s") or 0.0)),
                "稳定交战就绪并未直接触发发射，说明阶段层与参数层共同限制了抢射行为。",
            ],
            [
                "窗口兑现质量",
                fmt_pct(summary.get("gate_pass_per_launch_rate")),
                "首轮攻击不是靠频繁尝试堆出来，而是主要在有效窗口内兑现，说明放行门线具有筛选作用。",
            ],
            [
                "规避前安全余度",
                fmt_dist(summary.get("min_enemy_to_friendly_km")),
                "在完成4:0清场的同时仍保持正安全余度，说明方法不是以牺牲生存边界换取战果。",
            ],
            [
                "结果增益",
                f"{fmt_count(summary.get('enemy_kill_count'))}:{fmt_count(summary.get('friendly_loss_count'))} 对比 {fmt_count(base_summary.get('enemy_kill_count'))}:{fmt_count(base_summary.get('friendly_loss_count'))}",
                "相对基线同时提升清场效果与生存结果，说明完整闭环并非只改善过程指标。",
            ],
        ]
    if config.scenario_id == "S2":
        return [
            [
                "缺口下放行纪律",
                fmt_time(milestones.get("first_gate_pass_time_s")),
                "在长时间预警失配条件下仍未提前放行，说明图景缺口没有被用来放宽发射门约束。",
            ],
            [
                "两轮恢复能力",
                f"{fmt_time(config.decision_times['reengage'] - config.decision_times['first_evade'])} / {fmt_time(config.decision_times['second_reengage'] - config.decision_times['second_evade'])}",
                "两次“规避—重组—再交战”耗时均保持在同一量级，说明恢复纪律可以重复执行，而非一次性奏效。",
            ],
            [
                "持续压制兑现质量",
                fmt_pct(relay_delivery_rate(summary)),
                "接力制导链在两轮规避后仍能持续兑现，说明机动与参数层并未切断前序杀伤链。",
            ],
            [
                "结果增益",
                f"{fmt_count(summary.get('enemy_kill_count'))}:{fmt_count(summary.get('friendly_loss_count'))} 对比 {fmt_count(base_summary.get('enemy_kill_count'))}:{fmt_count(base_summary.get('friendly_loss_count'))}",
                "相对基线形成更高击落且保持零损失，说明恢复韧性确实转化为任务结果优势。",
            ],
        ]
    return [
        [
            "交战链前移幅度",
            fmt_time((milestones.get("first_gate_pass_time_s") or 0.0) - config.decision_times["engage"]),
            "从切入交战到首轮放行仅保留必要窗口，说明阶段层主动前移了交战链，而不是沿用标准场景节奏。",
        ],
        [
            "高风险承压控制",
            fmt_time(summary.get("high_risk_breach_time_s")),
            "高压样本不可避免进入高风险区，但累计滞留时间被压缩，说明规避不是被动兜底而是主动截断承压。",
        ],
        [
            "损失后推效果",
            fmt_time((milestones.get("first_friendly_loss_time_s") or 0.0) - config.decision_times["first_evade"]),
            "首次规避后仍保留长时间生存窗口，说明生存优先策略真实推迟了我方损失发生时刻。",
        ],
        [
            "结果增益",
            f"{fmt_time(summary.get('high_risk_breach_time_s'))} 对比 {fmt_time(base_summary.get('high_risk_breach_time_s'))}",
            "与基线相比主要收益体现为高风险暴露压缩，而不是单纯追求更高击落数。",
        ],
    ]


def scene_control_node_rows(package: Dict[str, Any]) -> List[List[str]]:
    rows = []
    df = package["control_nodes_df"].copy()
    route_text = {"left": "左路", "right": "右路"}
    for _, row in df.iterrows():
        rows.append([
            route_text.get(str(row["route"]), str(row["route"])),
            describe_phase(str(row["phase"])),
            fmt_time(to_float(row["first_time_s"])),
            fmt_dist(to_float(row["distance_km"])),
            str(row["description"]),
        ])
    return rows


def normalize_zone_value(value: Any) -> str:
    text = str(value).strip().upper() if value is not None else ""
    if text in ("LOW", "MEDIUM", "HIGH"):
        return text
    return "LOW"


def scene_aux_metric_rows(config: ScenarioConfig, package: Dict[str, Any]) -> List[List[str]]:
    summary = package["summary"]
    milestones = package["milestones"]
    if config.scenario_id == "S1":
        return [
            ["四目标全探完成时间", fmt_time(milestones.get("full_detect_time_s")), "用于说明协同探测链闭合后，四个目标何时全部进入可持续跟踪视野。"],
            ["任意航迹连续性（激活后）", fmt_pct(summary.get("radar_any_track_continuity_post_activation")), "用于说明在首轮放行前，目标航迹链是否连续稳定。"],
            ["稳定跟踪目标峰值", fmt_count(summary.get("stable_tracking_target_peak")), "用于说明完整闭环样本中，多目标并行稳定跟踪能力是否足以支撑双路清场。"],
            ["首次敌机被击落时间", fmt_time(milestones.get("first_enemy_kill_time_s")), "用于说明首轮放行后，杀伤链兑现的时间位置。"],
            ["发射门总请求次数（去重）", fmt_count(summary.get("gate_total_request_unique_count")), "用于说明首轮放行前后系统并不是只有一次发射判断，而是在多次判断中筛出了真正允许执行的窗口。"],
        ]
    if config.scenario_id == "S2":
        return [
            ["稳定跟踪目标峰值", fmt_count(summary.get("stable_tracking_target_peak")), "用于说明信息缺口条件下，机载协同跟踪峰值是否仍能覆盖四目标压制需求。"],
            ["首次敌机被击落时间", fmt_time(milestones.get("first_enemy_kill_time_s")), "用于说明持续压制样本中，首次杀伤兑现是否发生在图景缺口仍存在的阶段。"],
            ["接力制导兑现率（按已发射导弹归一）", fmt_pct(relay_delivery_rate(summary)), "用于说明在两轮规避恢复循环中，已发射导弹的中制导链是否保持稳定兑现。"],
            ["发射门总请求次数（去重）", fmt_count(summary.get("gate_total_request_unique_count")), "用于说明信息缺口条件下系统曾多次申请放行，但并未因图景受损而简化门线。"],
            ["中风险区累计滞留时间", fmt_time(summary.get("medium_risk_enemy_time_s")), "用于说明本场景的主战过程确实长期处于中风险持续压制区间，而不是快速突入后立即脱离。"],
        ]
    return [
        ["四目标全探完成时间", fmt_time(milestones.get("full_detect_time_s")), "用于说明低空突防样本中机载感知链几乎从开局即闭合，交战链得以前移。"],
        ["首次敌机被击落时间", fmt_time(milestones.get("first_enemy_kill_time_s")), "用于说明在高压样本中首轮攻击兑现战果的速度。"],
        ["接力制导兑现率（按已发射导弹归一）", fmt_pct(relay_delivery_rate(summary)), "用于说明在频繁规避下，已发射导弹的中制导链仍能维持一定兑现能力。"],
        ["中风险区累计滞留时间", fmt_time(summary.get("medium_risk_enemy_time_s")), "用于说明高压样本下主战过程长期处于中高风险压缩区间。"],
        ["高风险区峰值敌机数", fmt_count(summary.get("high_risk_peak_count")), "用于说明高风险样本中的最高压瞬间，并作为生存优先策略必要性的辅助证据。"],
    ]


def scene_intro_paragraph(config: ScenarioConfig) -> str:
    if config.scenario_id == "S1":
        return "场景一选取样本001401，对应低风险正面对进条件下的标准四机对四机接敌过程。该场景的验证重点不是单纯展示最终4:0战果，而是检验从协同探测、双路塑形、首轮放行、导弹来袭规避到回转清场的完整决策闭环是否真实成立。"
    if config.scenario_id == "S2":
        return "场景二选取样本225438，对应预警图景长时间间歇失配条件下的中风险持续压制任务。该场景的验证重点不在于首轮是否更早放行，而在于系统能否在图景缺口存在时仍保持压制骨架，并在两次规避后恢复到可继续交战的结构。"
    return "场景三选取样本001339，对应低空突防、高压近距和长时间预警失配并存的复杂拦截任务。该场景的验证重点不是追求更高击落数，而是检验交战链前移、生存优先止压和损失后推三者是否能够同时成立。"


def zone_intervals(package: Dict[str, Any], enemy_id: str, zone: str) -> List[Dict[str, Any]]:
    return [segment for segment in package["zone_segments"].get(enemy_id, []) if segment["zone"] == zone]


def scene_initial_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> str:
    if config.scenario_id == "S1":
        return "我方4机均在9.0 km高度附近起飞，敌方4机均在10.0 km高度附近正面对进，初始最近敌我距离为266.4 km。高度差、速度差和侧向展开都不极端，因此104.4 s形成稳定交战就绪、266.6 s才允许首次放行、322.0 s切规避这些时刻主要由控制距离与火控门线决定，而不是由初始姿态偶然推动。"
    if config.scenario_id == "S2":
        return "该场景把敌方配置成10.2 km与8.8 km混编来袭，初始最近敌我距离压缩到240.0 km，比场景一更早进入中风险压缩段。正因为起始几何更紧、上下层目标并存，50.0 s切入交战后仍能保持到236.2 s才首轮放行，才能说明方法在受扰条件下仍保留放行纪律。"
    return "该场景敌方4机高度被压到3.4-3.7 km，初始最近敌我距离仅150.0 km，开局即处于近距低空突防条件。我方虽然仍保持标准四机编队，但由于外层塑形空间极短，0.2 s直接切入交战、64.0 s完成首轮放行并不是激进取巧，而是由战场初始几何强制前移出来的结果。"


def scene_setup_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> str:
    summary = package["summary"]
    if config.scenario_id == "S1":
        return "场景一没有预警失配，敌方全程处于可连续感知条件，最终敌机4架全部被击落、我方零损失、高风险区累计滞留为0 s。这使该样本适合专门回答完整闭环是否成立，而不需要把解释重点分散到图景缺口或极端高度差上。"
    if config.scenario_id == "S2":
        return f"场景二的决定性扰动来自预警图景失配：累计失配时间达到{fmt_time(summary.get('enemy_awacs_denied_time_s'))}，占全程{fmt_pct(summary.get('enemy_awacs_denied_ratio'))}。在这种条件下仍实现3:0且高风险区累计滞留为0 s，说明后续分析必须围绕“缺口下如何维持压制和恢复”展开，而不是沿用场景一的完整闭环逻辑。"
    return f"场景三同时叠加了低空突防和长期图景失配，预警失配累计达到{fmt_time(summary.get('enemy_awacs_denied_time_s'))}，高风险区累计滞留达到{fmt_time(summary.get('high_risk_breach_time_s'))}，且最终出现1架我机损失。因此该样本不能只看击落数，必须把生存优先、止压时机和损失后推作为主分析线。"


def scene_core_metrics_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> str:
    summary = package["summary"]
    milestones = package["milestones"]
    if config.scenario_id == "S1":
        gate_delay = (milestones.get("first_gate_pass_time_s") or 0.0) - (milestones.get("first_stable_ready_time_s") or 0.0)
        return f"场景一最需要看的不是最终4:0，而是104.4 s形成稳定交战就绪后并没有立即发射，而是继续等待{gate_delay:.1f} s，到266.6 s才首次放行。与此同时，按已发射导弹归一的发射窗口兑现率达到{fmt_pct(summary.get('gate_pass_per_launch_rate'))}，按已发射导弹归一的接力制导兑现率达到{fmt_pct(relay_delivery_rate(summary))}，对存活目标的雷达探测覆盖率达到{fmt_pct(summary.get('radar_detection_coverage_ratio_alive_targets_post_activation'))}，最小敌我距离仍保持{fmt_dist(summary.get('min_enemy_to_friendly_km'))}，高风险区累计滞留为0 s。这组指标共同说明，场景一的有效性来自‘稳态火控成立后仍克制放行，再通过规避守住安全边界’，而不是靠冒险抢射换来的高战果。"
    if config.scenario_id == "S2":
        first_cycle = config.decision_times["reengage"] - config.decision_times["first_evade"]
        second_cycle = config.decision_times["second_reengage"] - config.decision_times["second_evade"]
        return f"场景二的核心证据是失配条件下仍能重复恢复。预警失配占全程{fmt_pct(summary.get('enemy_awacs_denied_ratio'))}，但首次放行仍压到236.2 s，按已发射导弹归一的发射窗口兑现率仍有{fmt_pct(summary.get('gate_pass_per_launch_rate'))}，对存活目标的雷达探测覆盖率保持在{fmt_pct(summary.get('radar_detection_coverage_ratio_alive_targets_post_activation'))}，接力制导兑现率达到{fmt_pct(relay_delivery_rate(summary))}。两轮“规避-重组-恢复交战”总耗时分别为{first_cycle:.1f} s和{second_cycle:.1f} s，接力制导链累计活跃{fmt_time(summary.get('relay_active_time_s'))}，最终保持零高风险滞留并形成3:0结果，说明恢复韧性是可重复成立的，不是单次偶然恢复。"
    lead_after_engage = (milestones.get("first_gate_pass_time_s") or 0.0) - config.decision_times["engage"]
    first_loss_gap = (milestones.get("first_friendly_loss_time_s") or 0.0) - config.decision_times["first_evade"]
    return f"场景三的关键不在击落数，而在前移和止压。14.8 s形成稳定交战就绪后，仅再过{lead_after_engage - 14.6:.1f} s便完成64.0 s首轮放行，说明交战链明显前移；与此同时，按已发射导弹归一的发射窗口兑现率达到{fmt_pct(summary.get('gate_pass_per_launch_rate'))}，接力制导兑现率达到{fmt_pct(relay_delivery_rate(summary))}，对存活目标的雷达探测覆盖率仍保持{fmt_pct(summary.get('radar_detection_coverage_ratio_alive_targets_post_activation'))}。高风险区累计滞留达到{fmt_time(summary.get('high_risk_breach_time_s'))}，共出现{fmt_count(summary.get('high_risk_breach_events'))}次高风险突入，最小敌我距离压到{fmt_dist(summary.get('min_enemy_to_friendly_km'))}。尽管如此，首次我机损失仍被后推到990.2 s，距131.8 s首次规避之间保留了{first_loss_gap:.1f} s的生存窗口，这才是该场景验证生存优先有效性的核心。"


def scene_aux_metrics_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> str:
    summary = package["summary"]
    milestones = package["milestones"]
    if config.scenario_id == "S1":
        return f"辅助指标进一步把完整闭环展开过程写清楚：92.4 s完成四目标全探，任意航迹连续性达到{fmt_pct(summary.get('radar_any_track_continuity_post_activation'))}，对存活目标的全覆盖连续性达到{fmt_pct(summary.get('radar_full_coverage_continuity_alive_targets_post_activation'))}，稳定跟踪峰值达到{fmt_count(summary.get('stable_tracking_target_peak'))}架，说明双路骨架在首轮放行前已经具备完整持续跟踪能力。首个击落发生在333.8 s，距离266.6 s首次放行约67.2 s，而发射门总请求数达到{fmt_count(summary.get('gate_total_request_unique_count'))}次，表明系统是在多次申请中筛出了11次真正允许执行的窗口。"
    if config.scenario_id == "S2":
        return f"场景二的辅助指标重点说明图景缺口并未把压制链打散。稳定跟踪峰值依然保持{fmt_count(summary.get('stable_tracking_target_peak'))}架，首次击落在311.2 s发生，此时首次规避刚过32.2 s，说明前序导弹和接力制导链在规避后仍能兑现。发射门总请求数达到{fmt_count(summary.get('gate_total_request_unique_count'))}次，但按已发射导弹归一的发射窗口兑现率仍有{fmt_pct(summary.get('gate_pass_per_launch_rate'))}；同时中风险区累计滞留长达{fmt_time(summary.get('medium_risk_enemy_time_s'))}。这说明系统在长时间中风险压制阶段始终保持高频判断，却没有用放宽门线来换取表面效率。"
    return f"场景三的辅助指标说明高压样本仍具备可接受的生存收益。12.0 s便完成四目标全探，191.2 s出现首次敌机击落，说明前移后的首轮攻击并非空转；接力制导兑现率保持在{fmt_pct(relay_delivery_rate(summary))}，说明频繁规避并未彻底打断中制导链。另一方面，中风险区累计滞留达到{fmt_time(summary.get('medium_risk_enemy_time_s'))}，高风险区峰值敌机数为{fmt_count(summary.get('high_risk_peak_count'))}，这两项指标共同说明场景三始终运行在高压边界附近，因此1:1结果更能体现方法在极限条件下守住生存底线的能力。"


def scene_milestone_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> str:
    milestones = package["milestones"]
    if config.scenario_id == "S1":
        return f"时间链条清晰显示出“先闭合感知、再等待放行、再兑现杀伤”的顺序：85.0 s首次机载雷达跟踪，92.4 s完成全探，104.4 s稳定交战就绪，266.6 s首次放行，295.0 s首次接力制导成功，333.8 s首次击落。稳定就绪到首次放行之间相隔162.2 s，这正是场景一证明放行并不由稳定交战就绪单独触发的直接证据。"
    if config.scenario_id == "S2":
        return f"场景二的里程碑序列显示出‘缺口下仍能维持节奏’：39.0 s首次雷达跟踪，54.8 s稳定交战就绪，236.2 s首次放行，251.4 s接力制导成功，279.0 s第一次规避，311.2 s首次击落，390.2 s转入截获重组，400.2 s恢复交战。第一次击落发生在首次规避之后，说明规避并没有使前序导弹效能失效。"
    return f"场景三的里程碑呈现明显前移：0.2 s首次雷达跟踪，12.0 s完成全探，14.8 s稳定交战就绪，64.0 s首次放行，81.8 s首次接力制导成功，131.8 s第一次规避，191.2 s首次击落。该顺序说明高压样本不是先慢慢塑形再寻找窗口，而是在感知刚闭合后尽快完成首轮放行，然后依靠及时规避把风险停留压住。"


def scene_control_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> str:
    df = package["control_nodes_df"]
    if config.scenario_id == "S1":
        left_dor = df[(df["route"] == "left") & (df["phase"].astype(str).str.contains("DOR_DR"))]["first_time_s"].min()
        right_dor = df[(df["route"] == "right") & (df["phase"].astype(str).str.contains("DOR_DR"))]["first_time_s"].min()
        return f"场景一左右两路在MELD-MTR段几乎同步进入压缩，左路187.4 s、右路185.0 s进入中距压缩段；272.0 s和273.2 s进入首次发射窗口段后，左路又在322.6 s先触及再次压缩准备段，而右路要到342.0 s才进入同一阶段。322.0 s第一次规避恰好压在左路即将踩入DOR-DR段之前，说明该时刻与左路生存边界逼近是直接对应的。"
    if config.scenario_id == "S2":
        return "场景二左右两路在86 s量级便越过NLT，左路226.6 s先进入首次发射窗口段，右路246.8 s随后进入，形成先左后右的首轮压制骨架。第一次规避出现在279.0 s，对应左路刚从LR-TR推进到TR-DOR段；第二次规避则出现在416.8 s，已经是右路长时间保持压制后的再次止压节点。控制距离序列说明两次规避发生在不同压缩层级，但都没有越过高风险底线。"
    return "场景三的控制距离推进几乎从开局就贴近中距段展开：左右两路在0.4-0.8 s内便越过NLT和MELD，左路68.4 s进入首次发射窗口段，右路109.0 s进入首次发射窗口段。第一次规避131.8 s发生在右路刚过TR节点后不久，第二次规避327.4 s则发生在重新恢复交战后的再次压缩阶段。这说明场景三的控制距离推进不是线性匀速前压，而是多次‘前移-止压-再压缩’的折返过程。"


def scene_zone_interval_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> str:
    if config.scenario_id == "S1":
        b0400_medium = zone_intervals(package, "B0400", "MEDIUM")
        return f"B0400 从 {format_interval_list(b0400_medium)} 持续处于中风险区，是场景一最后清场对象；B0100、B0200和B0300都只在190-351 s附近短暂进入中风险区后即被击落或退出，全场没有任何一架敌机进入高风险区。这说明场景一的回转与清场发生在中风险边界内完成，没有把敌机放入高风险区再处理。"
    if config.scenario_id == "S2":
        b0200_medium = zone_intervals(package, "B0200", "MEDIUM")
        b0100_medium = zone_intervals(package, "B0100", "MEDIUM")
        return f"B0200 自 {format_interval_list(b0200_medium)} 几乎贯穿整个主战过程，是持续压制的主对象；B0100 则在 {format_interval_list(b0100_medium)} 多次进入中风险区，说明第一次恢复交战后系统仍持续把同一对象拉回压制区。全场无高风险区记录，说明即使持续压制时间很长，方法仍把敌机活动限制在中风险边界内。"
    b0200_high = zone_intervals(package, "B0200", "HIGH")
    b0400_high = zone_intervals(package, "B0400", "HIGH")
    return f"场景三的高风险压力集中而明确：B0200 在 {format_interval_list(b0200_high)} 进入高风险区，B0400 在 {format_interval_list(b0400_high)} 两次进入高风险区，三段累计正好构成111.0 s高风险滞留。B0100全程停留在中风险区，这说明系统虽然无法完全把敌方压回低风险区，但已经把高风险压力收敛到少数敌机和少数时间段内。"


def scene_awacs_interval_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> str:
    summary = package["summary"]
    if config.scenario_id == "S2":
        return f"场景二预警图景累计失配{fmt_time(summary.get('enemy_awacs_denied_time_s'))}，约占全程{fmt_pct(summary.get('enemy_awacs_denied_ratio'))}。即便如此，第一次恢复交战仍发生在400.2 s，第二次恢复交战发生在523.0 s，说明图景失配并没有让系统丧失重组和再压制能力，而是把恢复过程更多地压到机载连续跟踪链上完成。"
    if config.scenario_id == "S3":
        return f"场景三预警图景失配累计达到{fmt_time(summary.get('enemy_awacs_denied_time_s'))}，约占全程{fmt_pct(summary.get('enemy_awacs_denied_ratio'))}，远高于场景二。因此271.4 s和431.2 s两次恢复交战更能说明一个事实：只要局部图景重新可用、状态机最小保持时间满足，系统仍会抓住短暂窗口重新接管交战，而不是在长期失配下完全放弃前场。"
    return "场景一预警图景全程可用，因此本场景的重点不在图景恢复，而在控制距离、发射门和规避边界如何闭合。"


def scene_event_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> str:
    enemy_times = [item["time_s"] for item in package["enemy_kill_events"]]
    friendly_times = [item["time_s"] for item in package["friendly_loss_events"]]
    if config.scenario_id == "S1":
        return f"场景一的击落事件高度集中在两段：333.8 s、345.8 s和351.0 s在首次规避后短时间内连续兑现3次击落，588.0 s完成最后1次清场。这说明322.0 s的规避没有切断前序导弹效能，反而在规避完成后释放出了三次连续兑现结果。"
    if config.scenario_id == "S2":
        return f"场景二的击落事件分布在311.2 s、468.2 s和1095.8 s。前两次击落分别落在第一次规避后和第二轮持续压制阶段，最后一次击落则延迟到1095.8 s，说明该场景不是靠首轮决胜，而是依靠长时间压制把第三个目标拖到末段才完成清除。"
    return f"场景三仅在191.2 s形成1次敌机击落，但我机损失直到990.2 s才出现，二者相隔799.0 s。这意味着该场景后段的主要任务已经从继续扩大战果转向守住剩余编队与控制区，不再适合用场景一、场景二的清场节奏来评价。"


def scene_decision_figure_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> List[str]:
    if config.scenario_id == "S1":
        return [
            "99.6 s任务状态由截获转入交战后，左右两路并未同步进入首轮发射窗口，而是先在148.0 s和150.4 s左右分化为左路钳形夹击、右路前后夹击。104.4 s形成稳定交战就绪后，任务状态仍保持交战但没有立刻发射，说明模板与状态切换先服务于占位塑形，而不是抢首发。",
            "322.0 s任务状态被导弹来袭直接打断并切入规避，446.6 s先恢复到截获重组，456.6 s才重新恢复交战。这10 s空窗对应状态机最小保持时间，图上表现为任务状态先回到重组带，再回到交战带，说明回转恢复并不由规避结束自动触发，而要等待重组约束满足。",
        ]
    if config.scenario_id == "S2":
        return [
            "场景二的关键信息不是单纯看交战与规避切换，而是看预警图景失配带与交战状态带如何同时存在。50.0 s进入交战后，图景失配并没有把任务状态直接推回巡逻或返航，说明阶段层允许在图景不完整时继续维持压制骨架。",
            "279.0 s和416.8 s两次规避之间都先经过截获重组再恢复交战，390.2-400.2 s与513.0-523.0 s两段恢复窗口长度完全一致。图上重复出现的“规避-重组-再交战”节奏说明恢复韧性是规则化行为，不是一次性的运气恢复。",
        ]
    return [
        "场景三从0.2 s起就直接进入交战，说明阶段层从开局便把敌方判定为需要立即处置的高压目标；任务状态的快速切换本身已经反映出低空近距条件下不存在长时间远距塑形空间。",
        "131.8 s与327.4 s两次规避都直接切断原有交战状态，而261.4-271.4 s和421.2-431.2 s两次恢复都保留10 s重组缓冲。再往后，1056.2 s到1180.0 s又重复出现一次完整的规避、重组和恢复链条，说明损失后段仍遵守同一套状态机规则。",
    ]


def scene_zone_figure_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> List[str]:
    if config.scenario_id == "S1":
        return [
            "B0100、B0200和B0300都在190-351 s附近进入中风险区，其中B0100约在240.2 s进入中风险区后不久即被持续压制；B0400则从187.4 s一直保持在中风险区直到588.0 s，成为最后清场对象。这说明场景一的主战过程始终围绕中风险边界展开，没有出现任何高风险区压入。",
            "322.0 s规避发生时，图上仍然只存在中风险区目标，没有高风险区目标。随后333.8 s、345.8 s和351.0 s连续出现击落，说明规避动作并没有造成敌机突然压入高风险区，反而在守住风险边界的同时释放出了前三次杀伤兑现。",
        ]
    if config.scenario_id == "S2":
        return [
            "B0200自92.4 s进入中风险区后一直维持到1095.8 s，是全场持续时间最长的中风险目标；B0100则在113.2-545.0 s、629.6-831.2 s和1007.4-1200.0 s多次进入中风险区，说明第一次恢复交战后系统并未放弃对该对象的后续压制。",
            "尽管400 s量级的预警失配贯穿场景二，但风险区时序图里始终没有高风险区记录，说明持续压制的结果不是不断向前透支，而是把敌方长期压在中风险边界内消耗。1095.8 s最后一次击落恰好发生在B0200长时间中风险滞留结束时。",
        ]
    return [
        "场景三的高风险压力集中在三段：B0200在111.6-148.0 s进入高风险区，B0400在170.4-196.6 s和348.6-397.0 s两次进入高风险区，三段合计111.0 s。高风险并不是全场铺开，而是集中压在少数关键对象和关键区间上。",
        "131.8 s第一次规避发生在B0200仍处于高风险区期间，327.4 s第二次规避则先于B0400第二次高风险区段结束。图上的对应关系说明规避并不是事后收尾，而是在高风险暴露尚未结束时主动打断继续压制，把高风险停留控制在有限窗口内。",
    ]


def scene_risk_count_figure_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> List[str]:
    if config.scenario_id == "S1":
        return [
            "场景一的风险区数量曲线显示，0-187 s四架敌机全部处于低风险区；187.4 s后中风险区敌机数迅速上升到4架，但高风险区敌机数始终为0。这说明首轮放行和后续规避都是在中风险带内完成的，没有把战场推进到高风险清场模式。",
            "333.8-351.0 s连续三次击落后，中风险区敌机数从4架快速下降到1架，只剩B0400持续停留到588.0 s。这一变化直接说明首次规避后的回转并不是重新把所有目标都压回来，而是针对剩余最后1架敌机继续组织清场。",
        ]
    if config.scenario_id == "S2":
        return [
            "场景二的中风险区敌机数从90.0 s开始迅速升到4架，并长期保持在2-4架之间波动，显示出标准的持续压制形态；高风险区敌机数全程为0，说明两次恢复交战都没有演变成高风险透支。",
            "279.0 s和416.8 s两次规避附近，中风险区敌机数并不会立刻归零，而是继续维持在2架以上。这表明规避动作打断的是我方压制姿态，而不是敌方立刻脱离战场；系统后续还必须依靠重组后再次压制来完成清场。",
        ]
    return [
        "场景三在0.2 s起就有4架敌机处于中风险区，111.6 s后高风险区敌机数第一次升至1架，170.4-196.6 s和348.6-397.0 s又两次升高。这条曲线把高压样本的核心特征写得很清楚：不是所有敌机同时压入高风险区，而是某一架目标轮流成为高压焦点。",
        "每当高风险区敌机数升高后，规避或击落事件都会跟着出现。191.2 s首次击落发生在高风险压力尚未完全解除的主战前段，说明首轮攻击的主要作用不是迅速清空全场，而是先削弱已经形成高压的一架目标，减轻后续承压。",
    ]


def scene_control_figure_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> List[str]:
    if config.scenario_id == "S1":
        return [
            "左路187.4 s进入中距压缩段，224.0 s进入首次攻击准备段，272.0 s进入首次发射窗口段；右路对应节点为185.0 s、221.2 s和273.2 s。两路首次发射窗口几乎同步，但左路322.6 s更早逼近再次压缩准备段，因此322.0 s规避首先是对左路压缩过深的响应。",
            "446.6 s恢复到截获重组后，距离曲线并没有立刻回到极远段，而是在中距段内重新组织。456.6 s恢复交战时，系统并不是从最安全距离重新来过，而是在维持安全余度的同时抓住最后清场窗口，这正是10 s回转缓冲存在的意义。",
        ]
    if config.scenario_id == "S2":
        return [
            "空间航迹图可见两次明显的外摆和再压入：第一次出现在279.0 s规避后，第二次出现在416.8 s规避后。两次外摆之后，编队都没有完全散开，而是保留了双路骨架，这与表中390.2 s、513.0 s两次截获重组时刻对应。",
            "与场景一不同，场景二的平面航迹更强调‘压制-外摆-再压制’的重复，而不是首轮塑形后的单次清场。图上的两次回转半径和再次压入方向基本一致，说明恢复动作并不是临时机动，而是按固定规则重复执行。",
        ]
    return [
        "场景三的控制距离曲线在0-110 s内迅速压缩到首次发射窗口，随后在131.8 s和327.4 s两次出现明显回拉。与场景一相比，曲线并非平滑单调前压，而是被多次规避切成数段，直接反映出高压样本的交战链前移与主动止压特征。",
        "在后段，1056.2 s之后曲线再次出现回拉，并于1170.0-1180.0 s完成重组后回压。这说明即便在990.2 s出现我机损失之后，系统仍没有放弃通过控制距离调节来守住剩余战场边界。",
    ]


def scene_awacs_figure_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> List[str]:
    summary = package["summary"]
    if config.scenario_id == "S1":
        return [
            f"场景一预警图景始终可用，机载雷达在85.0-92.4 s间完成四目标闭合，稳定交战就绪于104.4 s出现。预警连续、雷达连续和稳定交战就绪三条曲线在开局阶段快速收敛，说明场景一后续所有放行和规避决策都建立在完整图景之上，而不是依靠缺口中的猜测。"
        ]
    if config.scenario_id == "S2":
        return [
            f"场景二预警图景累计失配{fmt_time(summary.get('enemy_awacs_denied_time_s'))}，但机载雷达跟踪数仍能多次维持在3-4架，稳定交战就绪数在失配期间也没有归零。这说明恢复交战的直接依据不是‘预警必须全恢复’，而是局部图景足够支撑交战链重新闭合。",
            "390.2 s和513.0 s两次恢复到截获重组前后，机载跟踪曲线都先于任务状态恢复而抬升，说明系统是先把可用目标链重新接起来，再允许状态机回到交战状态。"
        ]
    return [
        f"场景三预警图景失配占比高达{fmt_pct(summary.get('enemy_awacs_denied_ratio'))}，但机载雷达从0.2 s起几乎全程保持对至少1架目标的连续跟踪，稳定交战就绪在14.8 s就已形成。高压样本能够前移首轮放行，依赖的正是这条机载跟踪链，而不是外层图景的长期稳定。"
    ]


def scene_event_chain_figure_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> List[str]:
    if config.scenario_id == "S1":
        return [
            "266.6 s首次放行并不是因为104.4 s已形成稳定交战就绪就立刻开火，而是因为此时最近敌我距离已收至94.0 km量级、左路处于MTR-LR段，发射门和控制距离共同闭合。",
            "322.0 s规避出现时，最近敌我距离已降到65.5 km，左路双机配对距离约54.1 km，已经贴近DOR/DR边界；446.6-456.6 s恢复窗口则是在距离回升后保留10 s重组缓冲，避免因过早回转把清场窗口压碎。"
        ]
    if config.scenario_id == "S2":
        return [
            "236.2 s首次放行时，最近敌我距离约82.3 km，左路已进入LR-TR段；即使外层预警图景持续失配，系统仍按同一套门线决定是否放行，而不是在缺口期提前放宽。",
            "279.0 s和416.8 s两次规避分别对应左路、右路压缩到55.3 km和43.4 km附近后的止压；390.2-400.2 s与513.0-523.0 s两段10 s重组缓冲保证恢复交战时双路骨架已重新闭合，因此后续311.2 s、468.2 s、1095.8 s仍能继续兑现击落。"
        ]
    return [
        "64.0 s首轮放行发生在0.2 s直接切入交战、14.8 s形成稳定交战就绪之后不久，说明高压低空样本必须把交战链前移；此时最近敌我距离已被压到首轮窗口附近，不能再沿用场景一的远距塑形节奏。",
        "131.8 s与327.4 s两次规避分别出现在左路、右路已逼近或进入DOR-DR/MTR-LR高压段时，因此回拉是由来袭武器和边界压缩共同触发；1056.2-1180.0 s的后段再恢复则说明即便出现首次损失，系统仍继续按同一套重组缓冲守区。"
    ]


def scene_kill_figure_analysis(config: ScenarioConfig, package: Dict[str, Any]) -> List[str]:
    return [scene_event_analysis(config, package)]


def describe_tactic(name: str) -> str:
    return TACTIC_TEXT.get(name, name)


def describe_phase(name: str) -> str:
    if name in PHASE_TEXT:
        return PHASE_TEXT[name]
    prefixed = f"TacticalPhase.{name}"
    if prefixed in PHASE_TEXT:
        return PHASE_TEXT[prefixed]
    bare_map = {
        "BEYOND_NLT": "远距接敌段（NLT外）",
        "NLT_MELD": "压缩过渡段（NLT-MELD）",
        "MELD_MTR": "中距压缩段（MELD-MTR）",
        "MTR_LR": "首次攻击准备段（MTR-LR）",
        "LR_TR": "首次发射窗口段（LR-TR）",
        "TR_DOR": "脱离准备段（TR-DOR）",
        "DOR_DR": "再次压缩准备段（DOR-DR）",
        "DR_MAR": "临界规避段（DR-MAR）",
        "BEYOND_MAR": "MAR后收口整理段（BEYOND_MAR）",
    }
    return bare_map.get(name, name)


def apply_ax_style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=15, weight="bold", color="#1d3557", pad=14)
    ax.grid(True, axis="y", color="#d9d2c3", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8d7b68")
    ax.spines["bottom"].set_color("#8d7b68")
    ax.tick_params(colors="#3d405b")


def save_initial_geometry(config: ScenarioConfig, figure_no: int) -> str:
    filename = f"fig6_{figure_no}_{config.scenario_id.lower()}_initial_geometry.png"
    output = ASSET_DIR / filename
    records = FRIENDLY_INITIAL + ENEMY_INITIAL[config.scenario_id]
    ref_lon = sum(item["lon"] for item in FRIENDLY_INITIAL) / len(FRIENDLY_INITIAL)
    ref_lat = sum(item["lat"] for item in FRIENDLY_INITIAL) / len(FRIENDLY_INITIAL)

    fig, ax = plt.subplots(figsize=(10.8, 7.6), dpi=220)
    friendly_x: List[float] = []
    friendly_y: List[float] = []
    enemy_x: List[float] = []
    enemy_y: List[float] = []

    for record in records:
        x_km, y_km = geodetic_to_local_km(record["lon"], record["lat"], ref_lon, ref_lat)
        record["x_km"] = x_km
        record["y_km"] = y_km
        if record["side"] == "我方":
            friendly_x.append(x_km)
            friendly_y.append(y_km)
        else:
            enemy_x.append(x_km)
            enemy_y.append(y_km)

    x_min = min(min(friendly_x), min(enemy_x)) - 20.0
    x_max = max(max(friendly_x), max(enemy_x)) + 20.0
    y_min = min(min(friendly_y), min(enemy_y)) - 15.0
    y_max = max(max(friendly_y), max(enemy_y)) + 15.0

    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, facecolor="#f7f4ed", edgecolor="none"))
    ax.axhspan(y_min, y_min + (y_max - y_min) * 0.45, color="#dbeafe", alpha=0.35)
    ax.axhspan(y_min + (y_max - y_min) * 0.45, y_min + (y_max - y_min) * 0.72, color="#fef3c7", alpha=0.35)
    ax.axhspan(y_min + (y_max - y_min) * 0.72, y_max, color="#fde2e4", alpha=0.30)

    for label, ypos in [
        ("我方编队活动带", y_min + (y_max - y_min) * 0.18),
        ("中部压缩带", y_min + (y_max - y_min) * 0.57),
        ("敌方进入方向", y_min + (y_max - y_min) * 0.86),
    ]:
        ax.text(x_min + 2.0, ypos, label, fontsize=10, color="#6d597a", va="center")

    for record in records:
        x_km = float(record["x_km"])
        y_km = float(record["y_km"])
        if record["side"] == "我方":
            color = "#1d4e89"
            marker = "^"
            edge = "#0b2545"
        else:
            color = "#c44536"
            marker = "D"
            edge = "#7f1d1d"
        ax.scatter(x_km, y_km, s=220, marker=marker, c=color, edgecolors=edge, linewidths=1.3, zorder=3)
        dx, dy = heading_to_vector(float(record["heading_deg"]), 8.0)
        arrow = FancyArrowPatch((x_km, y_km), (x_km + dx, y_km + dy), arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color=edge)
        ax.add_patch(arrow)
        ax.text(x_km + 1.5, y_km + 2.0, f"{record['aircraft_id']}\n{record['alt_m'] / 1000:.1f} km", fontsize=9.5, color=edge, ha="left", va="bottom")

    friendly_center = (sum(friendly_x) / len(friendly_x), sum(friendly_y) / len(friendly_y))
    enemy_center = (sum(enemy_x) / len(enemy_x), sum(enemy_y) / len(enemy_y))
    initial_distance_text = next(value for key, value in config.setup_rows if key == "敌方初始距离")
    ax.plot([friendly_center[0], enemy_center[0]], [friendly_center[1], enemy_center[1]], linestyle="--", color="#7c6f64", linewidth=1.4)
    ax.text(
        (friendly_center[0] + enemy_center[0]) / 2,
        (friendly_center[1] + enemy_center[1]) / 2 + 6.0,
        f"初始最近敌我距离约 {initial_distance_text}",
        fontsize=11,
        color="#3d405b",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fffaf0", edgecolor="#c9b79c"),
    )

    ax.text(min(friendly_x) - 8.0, max(friendly_y) + 4.0, "左路", fontsize=11, color="#1d4e89", weight="bold")
    ax.text(max(friendly_x) - 4.0, max(friendly_y) + 4.0, "右路", fontsize=11, color="#1d4e89", weight="bold")
    ax.text(min(enemy_x) - 8.0, min(enemy_y) - 8.0, "敌左群", fontsize=11, color="#9f1239", weight="bold")
    ax.text(max(enemy_x) - 4.0, min(enemy_y) - 8.0, "敌右群", fontsize=11, color="#9f1239", weight="bold")

    apply_ax_style(ax, f"{config.title}初始战场几何示意图")
    ax.set_xlabel("局部战场横向坐标 / km", fontsize=11, color="#3d405b")
    ax.set_ylabel("局部战场纵向坐标 / km", fontsize=11, color="#3d405b")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    legend_items = [
        Line2D([0], [0], marker="^", color="w", label="我方战斗机", markerfacecolor="#1d4e89", markeredgecolor="#0b2545", markersize=11),
        Line2D([0], [0], marker="D", color="w", label="敌方战斗机", markerfacecolor="#c44536", markeredgecolor="#7f1d1d", markersize=10),
        Line2D([0], [0], color="#7c6f64", linestyle="--", label="敌我中心距离"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return filename


def draw_band_timeline(ax: plt.Axes, segments: Sequence[Dict[str, Any]], y: float, height: float, text_map: Dict[str, str], color_map: Dict[str, str]) -> None:
    for segment in segments:
        start = float(segment["start"])
        duration = float(segment["duration"])
        value = str(segment["value"])
        color = color_map.get(value, "#adb5bd")
        ax.add_patch(Rectangle((start, y), duration, height, facecolor=color, edgecolor="#f7f4ed", linewidth=1.0))
        label = text_map.get(value, value)
        if duration >= 28:
            ax.text(start + duration / 2, y + height / 2, label, ha="center", va="center", fontsize=9.2, color="white" if color not in ("#f6bd60", "#dbeafe") else "#22223b", weight="bold")


def save_decision_timeline(config: ScenarioConfig, package: Dict[str, Any], figure_no: int) -> str:
    filename = f"fig6_{figure_no}_{config.scenario_id.lower()}_decision_timeline.png"
    output = ASSET_DIR / filename
    timeline = package["timeline_df"].copy()
    lanes = [
        ("任务状态", "cap_state", STATE_TEXT, STATE_COLOR),
        ("左路模板", "left_tactic", TACTIC_TEXT, TACTIC_COLOR),
        ("右路模板", "right_tactic", TACTIC_TEXT, TACTIC_COLOR),
        ("左路节点", "left_phase", PHASE_TEXT, PHASE_COLOR),
        ("右路节点", "right_phase", PHASE_TEXT, PHASE_COLOR),
    ]
    if "enemy_script_awacs_state" in timeline.columns:
        lanes.append(("预警图景", "enemy_script_awacs_state", AWACS_TEXT, AWACS_COLOR))

    fig, ax = plt.subplots(figsize=(14.2, 7.8), dpi=220)
    times = timeline["time_s"].tolist()
    y_positions = list(range(len(lanes)))[::-1]
    height = 0.78
    for index, (lane_label, column, text_map, color_map) in enumerate(lanes):
        values = timeline[column].fillna("UNKNOWN").astype(str).tolist()
        segments = build_segments(times, values)
        y = y_positions[index]
        draw_band_timeline(ax, segments, y, height, text_map, color_map)
        ax.text(times[0] - 0.02 * (times[-1] - times[0]), y + height / 2, lane_label, ha="right", va="center", fontsize=11, color="#3d405b", weight="bold")

    milestone_markers = [
        ("首次稳定交战就绪", package["milestones"].get("first_stable_ready_time_s"), "#43aa8b"),
        ("首次发射门通过", package["milestones"].get("first_gate_pass_time_s"), "#f8961e"),
        ("首次接力制导成功", package["milestones"].get("first_relay_success_time_s"), "#f9844a"),
        ("首次规避", config.decision_times.get("first_evade"), "#d1495b"),
    ]
    if "first_intercept" in config.decision_times:
        milestone_markers.append(("首次截获重组", config.decision_times.get("first_intercept"), "#2a9d8f"))
    if "reengage" in config.decision_times:
        milestone_markers.append(("恢复交战", config.decision_times.get("reengage"), "#277da1"))
    if "second_evade" in config.decision_times:
        milestone_markers.append(("再次规避", config.decision_times.get("second_evade"), "#9d4edd"))

    top_y = max(y_positions) + 1.1
    for label, time_value, color in milestone_markers:
        if time_value is None:
            continue
        ax.axvline(time_value, color=color, linestyle="--", linewidth=1.2, alpha=0.9)
        ax.text(time_value, top_y, label, rotation=90, ha="center", va="bottom", fontsize=9.3, color=color, weight="bold")

    apply_ax_style(ax, f"{config.title}任务状态、模板与控制距离节点时序图")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.2, len(lanes) + 0.6)
    ax.set_xlabel("仿真时间 / s", fontsize=11, color="#3d405b")
    ax.set_yticks([])
    ax.grid(True, axis="x", color="#d9d2c3", linestyle="--", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_zone_timeline(config: ScenarioConfig, package: Dict[str, Any], figure_no: int) -> str:
    filename = f"fig6_{figure_no}_{config.scenario_id.lower()}_zone_timeline.png"
    output = ASSET_DIR / filename
    timeline = package["timeline_df"].copy()
    timeline["zone_map"] = timeline["zone_by_enemy_json"].apply(parse_json_field)
    enemy_ids = ["B0100", "B0200", "B0300", "B0400"]
    fig, ax = plt.subplots(figsize=(13.8, 6.7), dpi=220)

    lane_labels: List[str] = []
    lane_values: List[List[str]] = []
    if config.scenario_id in ("S2", "S3") and "enemy_script_awacs_state" in timeline.columns:
        lane_labels.append("预警图景")
        lane_values.append(timeline["enemy_script_awacs_state"].fillna("AVAILABLE").astype(str).tolist())
    for enemy_id in enemy_ids:
        lane_labels.append(enemy_id)
        lane_values.append([normalize_zone_value(zone_map.get(enemy_id, "LOW")) for zone_map in timeline["zone_map"]])

    y_positions = list(range(len(lane_labels)))[::-1]
    times = timeline["time_s"].tolist()
    for index, lane_label in enumerate(lane_labels):
        values = lane_values[index]
        segments = build_segments(times, values)
        y = y_positions[index]
        if lane_label == "预警图景":
            draw_band_timeline(ax, segments, y, 0.78, AWACS_TEXT, AWACS_COLOR)
        else:
            draw_band_timeline(ax, segments, y, 0.78, ZONE_TEXT, ZONE_COLOR)
        ax.text(times[0] - 0.02 * (times[-1] - times[0]), y + 0.39, lane_label, ha="right", va="center", fontsize=11, color="#3d405b", weight="bold")

    apply_ax_style(ax, f"{config.title}敌机进入风险区时序图")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.2, len(lane_labels) + 0.6)
    ax.set_xlabel("仿真时间 / s", fontsize=11, color="#3d405b")
    ax.set_yticks([])
    ax.grid(True, axis="x", color="#d9d2c3", linestyle="--", linewidth=0.7, alpha=0.7)
    legend_items = [
        Rectangle((0, 0), 1, 1, facecolor=ZONE_COLOR["LOW"], edgecolor="none", label="低风险区"),
        Rectangle((0, 0), 1, 1, facecolor=ZONE_COLOR["MEDIUM"], edgecolor="none", label="中风险区"),
        Rectangle((0, 0), 1, 1, facecolor=ZONE_COLOR["HIGH"], edgecolor="none", label="高风险区"),
    ]
    if config.scenario_id in ("S2", "S3"):
        legend_items.extend([
            Rectangle((0, 0), 1, 1, facecolor=AWACS_COLOR["AVAILABLE"], edgecolor="none", label="预警图景可用"),
            Rectangle((0, 0), 1, 1, facecolor=AWACS_COLOR["DENIED"], edgecolor="none", label="预警图景失配"),
        ])
    ax.legend(handles=legend_items, loc="upper right", frameon=False, ncol=3, fontsize=9.6)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_risk_zone_count_plot(config: ScenarioConfig, package: Dict[str, Any], figure_no: int) -> str:
    filename = f"fig6_{figure_no}_{config.scenario_id.lower()}_risk_zone_count.png"
    output = ASSET_DIR / filename
    timeline = package["timeline_df"].copy()
    zone_segments = package["zone_segments"]
    times = timeline["time_s"].astype(float).tolist()
    step = timeline_step(times)
    if not times:
        times = [0.0]
    high_count: List[float] = []
    medium_count: List[float] = []
    low_count: List[float] = []
    for t in times:
        high = medium = low = 0
        for enemy_segments in zone_segments.values():
            zone = "LOW"
            for segment in enemy_segments:
                if segment["start"] <= t < segment["end"] or (t == times[-1] and abs(t - segment["end"]) < 1e-6):
                    zone = segment["zone"]
                    break
            if zone == "HIGH":
                high += 1
            elif zone == "MEDIUM":
                medium += 1
            else:
                low += 1
        high_count.append(high)
        medium_count.append(medium)
        low_count.append(low)

    fig, ax = plt.subplots(figsize=(13.8, 6.8), dpi=220)
    ax.step(times, low_count, where="post", color=ZONE_COLOR["LOW"], linewidth=2.0, label="低风险区敌机数")
    ax.step(times, medium_count, where="post", color=ZONE_COLOR["MEDIUM"], linewidth=2.0, label="中风险区敌机数")
    ax.step(times, high_count, where="post", color=ZONE_COLOR["HIGH"], linewidth=2.2, label="高风险区敌机数")

    if config.scenario_id == "S1":
        marks = [
            (config.decision_times.get("first_evade"), "规避起点", "#d1495b"),
            (package["milestones"].get("first_enemy_kill_time_s"), "首次击落", "#577590"),
            (package["milestones"].get("first_gate_pass_time_s"), "首次放行", "#f8961e"),
        ]
    elif config.scenario_id == "S2":
        marks = [
            (config.decision_times.get("first_evade"), "首次规避", "#d1495b"),
            (config.decision_times.get("second_evade"), "再次规避", "#9d4edd"),
            (package["milestones"].get("first_enemy_kill_time_s"), "首次击落", "#577590"),
        ]
    else:
        marks = [
            (config.decision_times.get("first_evade"), "首次规避", "#d1495b"),
            (config.decision_times.get("second_evade"), "再次规避", "#9d4edd"),
            (package["milestones"].get("first_friendly_loss_time_s"), "首次我机损失", "#6d597a"),
        ]
    for t_value, label, color in marks:
        if t_value is None:
            continue
        ax.axvline(t_value, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        idx = min(len(times) - 1, max(0, int(round(float(t_value) / step)))) if times else 0
        y_value = max(high_count[idx], medium_count[idx], low_count[idx])
        ax.text(t_value, y_value + 0.12, label, fontsize=9.3, color=color, ha="center", va="bottom")

    apply_ax_style(ax, f"{config.title}敌机风险区数量时序图")
    ax.set_xlabel("仿真时间 / s", fontsize=11, color="#3d405b")
    ax.set_ylabel("敌机数量 / 架", fontsize=11, color="#3d405b")
    ax.set_ylim(-0.1, 4.6)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_kill_timeline_plot(config: ScenarioConfig, package: Dict[str, Any], figure_no: int) -> str:
    filename = f"fig6_{figure_no}_{config.scenario_id.lower()}_kill_timeline.png"
    output = ASSET_DIR / filename
    fig, ax = plt.subplots(figsize=(13.6, 5.6), dpi=220)
    kill_events = package["enemy_kill_events"]
    friendly_loss_events = package["friendly_loss_events"]
    if kill_events:
        ax.scatter([item["time_s"] for item in kill_events], [1] * len(kill_events), s=110, color="#577590", marker="P", label="敌机击落事件")
        for item in kill_events:
            ax.text(item["time_s"], 1.08, fmt_time(item["time_s"]), fontsize=8.8, color="#577590", ha="center")
    if friendly_loss_events:
        ax.scatter([item["time_s"] for item in friendly_loss_events], [0] * len(friendly_loss_events), s=110, color="#6d597a", marker="X", label="我机损失事件")
        for item in friendly_loss_events:
            ax.text(item["time_s"], -0.12, fmt_time(item["time_s"]), fontsize=8.8, color="#6d597a", ha="center", va="top")

    if config.scenario_id == "S1":
        ax.axvspan(config.decision_times["first_evade"], config.decision_times["reengage"], color="#d1495b", alpha=0.08)
        ax.text((config.decision_times["first_evade"] + config.decision_times["reengage"]) / 2, 1.34, "规避-重组-再压制窗口", color="#8d1c36", ha="center")
    elif config.scenario_id == "S2":
        ax.axvspan(config.decision_times["first_evade"], config.decision_times["reengage"], color="#d1495b", alpha=0.08)
        ax.axvspan(config.decision_times["second_evade"], config.decision_times["second_reengage"], color="#9d4edd", alpha=0.08)
    else:
        ax.axvspan(config.decision_times["first_evade"], config.decision_times["first_intercept"], color="#d1495b", alpha=0.08)
        ax.axvspan(config.decision_times["second_evade"], config.decision_times["second_intercept"], color="#9d4edd", alpha=0.08)
        ax.axvspan(config.decision_times.get("fifth_evade", 1056.2), config.decision_times.get("fifth_intercept", 1170.0), color="#6d597a", alpha=0.08)

    apply_ax_style(ax, f"{config.title}击落与损失事件时序图")
    ax.set_xlabel("仿真时间 / s", fontsize=11, color="#3d405b")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["我机损失", "敌机击落"], fontsize=10.2)
    ax.set_ylim(-0.35, 1.55)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_awacs_track_plot(config: ScenarioConfig, package: Dict[str, Any], figure_no: int) -> str:
    filename = f"fig6_{figure_no}_{config.scenario_id.lower()}_awacs_track.png"
    output = ASSET_DIR / filename
    timeline = package["timeline_df"].copy()
    fig, ax = plt.subplots(figsize=(13.6, 5.9), dpi=220)
    if "enemy_script_awacs_state" in timeline.columns:
        state_series = timeline["enemy_script_awacs_state"].fillna("AVAILABLE").astype(str).tolist()
        y_map = {"AVAILABLE": 1, "DENIED": 0}
        y_values = [y_map.get(value, 1) for value in state_series]
        ax.step(timeline["time_s"], y_values, where="post", color="#6d597a", linewidth=2.0, label="预警图景状态")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["失配", "可用"], fontsize=10.2)
    else:
        ax.step(timeline["time_s"], timeline["awacs_track_count"], where="post", color="#2a9d8f", linewidth=2.0, label="预警机跟踪数")
        ax.set_ylabel("预警机跟踪数 / 架", fontsize=11, color="#3d405b")

    ax2 = ax.twinx()
    ax2.step(timeline["time_s"], timeline["radar_track_count"], where="post", color="#1d4e89", linewidth=2.1, label="机载雷达跟踪数")
    ax2.step(timeline["time_s"], timeline["stable_ready_count"], where="post", color="#f8961e", linewidth=1.8, label="稳定交战就绪数")
    ax2.set_ylabel("我方跟踪/就绪数 / 架", fontsize=11, color="#3d405b")
    ax2.set_ylim(-0.1, 4.4)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=9.6, loc="upper right")
    apply_ax_style(ax, f"{config.title}预警图景与跟踪闭合图")
    ax.set_xlabel("仿真时间 / s", fontsize=11, color="#3d405b")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_cross_scene_effectiveness(packages: Dict[str, Dict[str, Any]], figure_no: int) -> str:
    from matplotlib import patheffects as pe

    filename = f"fig6_{figure_no}_cross_scene_effectiveness.png"
    output = ASSET_DIR / filename
    fig, ax = plt.subplots(figsize=(12.8, 5.8), dpi=220)
    scenes = ["S1", "S2", "S3"]
    scene_labels = ["场景一", "场景二", "场景三"]
    metrics = [
        ("敌机击落数", "summary", "enemy_kill_count", "count", True),
        ("我方损失数", "summary", "friendly_loss_count", "count", False),
        ("高风险滞留", "summary", "high_risk_breach_time_s", "time", False),
        ("接力制导兑现率", "derived", "relay_delivery_rate", "pct", True),
        ("最小敌我距离", "summary", "min_enemy_to_friendly_km", "dist", True),
    ]
    actual_values: List[List[Optional[float]]] = []
    scores: List[List[float]] = []
    for scene in scenes:
        row_actual: List[Optional[float]] = []
        for _, source, key, _, _ in metrics:
            package = packages[scene]
            if source == "summary":
                value = package["summary"].get(key)
            elif source == "milestone":
                value = package["milestones"].get(key)
            else:
                value = relay_delivery_rate(package["summary"])
            row_actual.append(value)
        actual_values.append(row_actual)
    for metric_index, (_, _, _, _, higher_better) in enumerate(metrics):
        column = [row[metric_index] if row[metric_index] is not None else 0.0 for row in actual_values]
        min_val = min(column)
        max_val = max(column)
        for row_index in range(len(scenes)):
            if len(scores) <= row_index:
                scores.append([])
            value = column[row_index]
            if abs(max_val - min_val) < 1e-9:
                normalized = 1.0
            else:
                normalized = (value - min_val) / (max_val - min_val)
            score = normalized if higher_better else 1.0 - normalized
            scores[row_index].append(score)
    im = ax.imshow(scores, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(list(range(len(metrics))))
    ax.set_xticklabels([item[0] for item in metrics], fontsize=10.0)
    ax.set_yticks(list(range(len(scene_labels))))
    ax.set_yticklabels(scene_labels, fontsize=10.4)
    for row_index, scene in enumerate(scenes):
        for col_index, (_, _, _, value_type, _) in enumerate(metrics):
            value = actual_values[row_index][col_index]
            text = fmt_value(value, value_type)
            score = scores[row_index][col_index]
            dark_cell = score >= 0.55
            text_color = "#ffffff" if dark_cell else "#132238"
            outline_color = "#132238" if dark_cell else "#ffffff"
            txt = ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.4,
                weight="bold",
            )
            txt.set_path_effects([
                pe.withStroke(linewidth=1.8, foreground=outline_color, alpha=0.95)
            ])
    apply_ax_style(ax, "三场景关键效果横向热力对比图")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("同列相对优劣", fontsize=10.2, color="#3d405b")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_cross_scene_kill_loss_profile(packages: Dict[str, Dict[str, Any]], figure_no: int) -> str:
    filename = f"fig6_{figure_no}_cross_scene_kill_loss_profile.png"
    output = ASSET_DIR / filename
    fig, ax = plt.subplots(figsize=(13.2, 5.8), dpi=220)
    scenes = ["S1", "S2", "S3"]
    y_map = {"S1": 2, "S2": 1, "S3": 0}
    for scene in scenes:
        package = packages[scene]
        y = y_map[scene]
        max_time = float(package["timeline_df"]["time_s"].max())
        ax.hlines(y, 0.0, max_time, color="#d9d2c3", linewidth=1.2)
        ax.text(-18.0, y, SCENARIOS[scene].selected_label, ha="right", va="center", fontsize=10.4, color="#3d405b", weight="bold")
        for event in package["enemy_kill_events"]:
            ax.scatter(event["time_s"], y + 0.08, s=110, color="#577590", marker="P", zorder=3)
        for event in package["friendly_loss_events"]:
            ax.scatter(event["time_s"], y - 0.08, s=110, color="#6d597a", marker="X", zorder=3)
    legend_items = [
        Line2D([0], [0], marker="P", color="w", label="敌机击落事件", markerfacecolor="#577590", markersize=10),
        Line2D([0], [0], marker="X", color="w", label="我机损失事件", markerfacecolor="#6d597a", markeredgecolor="#6d597a", markersize=10),
    ]
    apply_ax_style(ax, "三场景击落与损失事件横向对比图")
    ax.set_xlabel("仿真时间 / s", fontsize=11, color="#3d405b")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 2.5)
    ax.legend(handles=legend_items, frameon=False, fontsize=9.8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_cross_scene_milestones(packages: Dict[str, Dict[str, Any]], figure_no: int) -> str:
    filename = f"fig6_{figure_no}_cross_scene_milestones.png"
    output = ASSET_DIR / filename
    fig, ax = plt.subplots(figsize=(13.4, 6.4), dpi=220)
    scene_order = ["S1", "S2", "S3"]
    scene_y = {"S1": 2, "S2": 1, "S3": 0}
    milestone_specs = [
        ("首次稳定交战就绪", "first_stable_ready_time_s", "#43aa8b", "o"),
        ("首次发射门通过", "first_gate_pass_time_s", "#f8961e", "s"),
        ("首次规避", "first_evade", "#d1495b", "D"),
        ("首次截获重组", "first_intercept", "#2a9d8f", "^"),
        ("首次敌机被击落", "first_enemy_kill_time_s", "#577590", "P"),
    ]
    for scene_id in scene_order:
        package = packages[scene_id]
        config = SCENARIOS[scene_id]
        ax.hlines(scene_y[scene_id], 0.0, float(package["timeline_df"]["time_s"].max()), color="#d9d2c3", linewidth=1.2)
        ax.text(-18, scene_y[scene_id], config.selected_label, ha="right", va="center", fontsize=11, color="#3d405b", weight="bold")
        for label, key, color, marker in milestone_specs:
            if key in config.decision_times:
                value = config.decision_times[key]
            else:
                value = package["milestones"].get(key)
            if value is None:
                continue
            ax.scatter([value], [scene_y[scene_id]], s=100, marker=marker, color=color, edgecolors="white", linewidths=1.1, zorder=4)
            ax.text(value, scene_y[scene_id] + 0.12, label, fontsize=9.2, color=color, ha="center", va="bottom", rotation=25)
    apply_ax_style(ax, "三场景关键里程碑横向对比图")
    ax.set_xlabel("仿真时间 / s", fontsize=11, color="#3d405b")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 2.6)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_scene_baseline_comparison(config: ScenarioConfig, package: Dict[str, Any], baseline: Dict[str, Any], figure_no: int) -> str:
    filename = f"fig6_{figure_no}_{config.scenario_id.lower()}_baseline_compare.png"
    output = ASSET_DIR / filename
    fig, ax = plt.subplots(figsize=(12.8, 5.8), dpi=220)

    if config.scenario_id == "S1":
        items = [
            ("全探完成", package["milestones"].get("full_detect_time_s"), baseline["milestones"].get("full_detect_time_s"), "time"),
            ("稳定就绪", package["milestones"].get("first_stable_ready_time_s"), baseline["milestones"].get("first_stable_ready_time_s"), "time"),
            ("首次放行", package["milestones"].get("first_gate_pass_time_s"), baseline["milestones"].get("first_gate_pass_time_s"), "time"),
            ("窗口兑现率", package["summary"].get("gate_pass_per_launch_rate"), baseline["summary"].get("gate_pass_per_launch_rate"), "pct"),
            ("探测覆盖率", package["summary"].get("radar_detection_coverage_ratio_alive_targets_post_activation"), baseline["summary"].get("radar_detection_coverage_ratio_alive_targets_post_activation"), "pct"),
            ("击落数", package["summary"].get("enemy_kill_count"), baseline["summary"].get("enemy_kill_count"), "count"),
        ]
    elif config.scenario_id == "S2":
        items = [
            ("稳定就绪", package["milestones"].get("first_stable_ready_time_s"), baseline["milestones"].get("first_stable_ready_time_s"), "time"),
            ("窗口兑现率", package["summary"].get("gate_pass_per_launch_rate"), baseline["summary"].get("gate_pass_per_launch_rate"), "pct"),
            ("接力兑现率", relay_delivery_rate(package["summary"]), relay_delivery_rate(baseline["summary"]), "pct"),
            ("击落数", package["summary"].get("enemy_kill_count"), baseline["summary"].get("enemy_kill_count"), "count"),
            ("我方损失", package["summary"].get("friendly_loss_count"), baseline["summary"].get("friendly_loss_count"), "count"),
            ("高风险滞留", package["summary"].get("high_risk_breach_time_s"), baseline["summary"].get("high_risk_breach_time_s"), "time"),
        ]
    else:
        items = [
            ("全探完成", package["milestones"].get("full_detect_time_s"), baseline["milestones"].get("full_detect_time_s"), "time"),
            ("稳定就绪", package["milestones"].get("first_stable_ready_time_s"), baseline["milestones"].get("first_stable_ready_time_s"), "time"),
            ("首次放行", package["milestones"].get("first_gate_pass_time_s"), baseline["milestones"].get("first_gate_pass_time_s"), "time"),
            ("窗口兑现率", package["summary"].get("gate_pass_per_launch_rate"), baseline["summary"].get("gate_pass_per_launch_rate"), "pct"),
            ("探测覆盖率", package["summary"].get("radar_detection_coverage_ratio_alive_targets_post_activation"), baseline["summary"].get("radar_detection_coverage_ratio_alive_targets_post_activation"), "pct"),
            ("高风险滞留", package["summary"].get("high_risk_breach_time_s"), baseline["summary"].get("high_risk_breach_time_s"), "time"),
        ]

    labels = [item[0] for item in items]
    current_vals = [0.0 if item[1] is None else float(item[1]) for item in items]
    baseline_vals = [0.0 if item[2] is None else float(item[2]) for item in items]
    x = list(range(len(items)))
    width = 0.34
    ax.bar([p - width / 2 for p in x], current_vals, width=width, color="#577590", label="本文方法")
    ax.bar([p + width / 2 for p in x], baseline_vals, width=width, color="#c9ada7", label="基线方法")

    for idx, (_, current_value, baseline_value, value_type) in enumerate(items):
        ax.text(idx - width / 2, current_vals[idx], fmt_value(current_value, value_type), ha="center", va="bottom", fontsize=8.8, color="#1d3557", rotation=90)
        ax.text(idx + width / 2, baseline_vals[idx], fmt_value(baseline_value, value_type), ha="center", va="bottom", fontsize=8.8, color="#6d597a", rotation=90)

    apply_ax_style(ax, f"{config.title}本文方法与基线方法关键指标对比图")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.0)
    ax.legend(frameon=False, fontsize=9.8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return filename


def cross_scene_baseline_gain_rows(packages: Dict[str, Dict[str, Any]], baselines: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for scene_id in ("S1", "S2", "S3"):
        cur_p = packages[scene_id]
        base_p = baselines[scene_id]
        cur_s = cur_p["summary"]
        base_s = base_p["summary"]
        cur_m = cur_p["milestones"]
        base_m = base_p["milestones"]
        rows.append([
            scene_id,
            fmt_time(cur_m.get("first_stable_ready_time_s")),
            fmt_time(base_m.get("first_stable_ready_time_s")),
            fmt_pct(cur_s.get("gate_pass_per_launch_rate")),
            fmt_pct(base_s.get("gate_pass_per_launch_rate")),
            fmt_count(cur_s.get("enemy_kill_count")),
            fmt_count(base_s.get("enemy_kill_count")),
            fmt_count(cur_s.get("friendly_loss_count")),
            fmt_count(base_s.get("friendly_loss_count")),
        ])
    return rows


def add_figure_block(lines: List[str], numbering: Numbering, title: str, filename: str, explanation: Sequence[str]) -> None:
    figure_no = numbering.figure
    lines.append(f"图6-{figure_no} {title}")
    lines.append(f"![图6-{figure_no} {title}](./{ASSET_DIR.name}/{filename})")
    lines.extend(explanation)
    lines.append("")


def build_report() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for child in ASSET_DIR.iterdir():
        if child.is_file():
            child.unlink()
    packages = {scene_id: build_scene_package(config) for scene_id, config in SCENARIOS.items()}
    baseline_packages = {scene_id: build_baseline_package(config) for scene_id, config in SCENARIOS.items()}

    numbering = Numbering()
    lines: List[str] = []
    lines.append("# 第六章仿真验证文档")
    lines.append("")
    lines.append("## 6.3 仿真验证")
    lines.append("")
    lines.append("本节不再按时间线复述仿真过程，而是直接围绕三个代表样本分析分层战术决策方法的有效性。分析重点是说明阶段决策层、战术决策层、机动决策层和参数决策层在关键窗口内分别解决了什么问题、依据哪些约束作出选择，以及这些选择如何转化为任务结果。")
    lines.append("三个场景分别承担不同验证任务：场景一验证完整闭环能否在不透支生存边界的条件下成立，场景二验证信息缺口下的恢复纪律是否可重复兑现，场景三验证高压低空突防条件下交战链前移与生存优先是否能够同时成立。")
    lines.append("正式统计统一采用事件去重口径。同一枚导弹、同一条接力制导链和同一轮发射门放行只计1次，不按0.2 s采样周期重复累加。稳定交战就绪门限统一取5.0 s，任务状态切换后的最小保持时间统一取10.0 s。")
    lines.append("控制距离节点缩写在正文中均采用“中文功能名（英文缩写）”的方式使用。其中，NLT表示不晚于截获距离，MELD表示混战进入距离，MTR表示机动跟踪距离，LR表示首次发射准备距离，TR表示交战转换距离，DOR表示脱离决断距离，DR表示重组决断距离，MAR表示最小规避距离。")
    lines.append("")

    for section_number, scene_id in enumerate(["S1", "S2", "S3"], start=2):
        config = SCENARIOS[scene_id]
        package = packages[scene_id]
        lines.append(f"### 6.3.{section_number} {config.title}")
        lines.append("")
        lines.append("#### （1）场景目标与初始状态")
        lines.append("")
        lines.append(scene_intro_paragraph(config))
        lines.append("")
        table_no = numbering.next_table()
        lines.append(f"表6-{table_no} {config.title}初始状态")
        lines.extend(md_table(
            ["飞机ID", "阵营", "机型", "初始经度 / deg", "初始纬度 / deg", "初始高度 / m", "初始航向 / deg", "初始速度", "初始导弹数"],
            friendly_enemy_rows(scene_id),
        ))
        lines.append("")
        lines.append(scene_initial_analysis(config, package))
        lines.append("")
        table_no = numbering.next_table()
        lines.append(f"表6-{table_no} {config.title}场景设定与分析重点")
        lines.extend(md_table(["项目", "内容"], list(config.setup_rows)))
        lines.append("")
        lines.append(scene_setup_analysis(config, package))
        lines.append("")

        lines.append("#### （2）场景重点指标与总体态势")
        lines.append("")
        table_no = numbering.next_table()
        lines.append(f"表6-{table_no} {config.title}决策有效性主指标")
        lines.extend(md_table(["指标项", "数值", "分析含义"], scene_decision_indicator_rows(config, package)))
        lines.append("")
        lines.append("上表不再展开所有过程量，而只保留能够直接支撑决策分析的主指标。其作用不是复述仿真现象，而是为后续四层决策分析提供可验证的结果约束。")
        lines.append("")
        table_no = numbering.next_table()
        lines.append(f"表6-{table_no} {config.title}重点指标")
        lines.extend(md_table(["指标项", "数值", "本场景中的分析用途"], scene_focus_metric_rows(config, package)))
        lines.append("")
        lines.append(scene_core_metrics_analysis(config, package))
        lines.append("")
        table_no = numbering.next_table()
        lines.append(f"表6-{table_no} {config.title}本文方法与基线方法关键指标对比")
        lines.extend(md_table(["指标项", "本文方法", "基线方法", "判定方向"], baseline_compare_rows(config, package, baseline_packages[scene_id])))
        lines.append("")
        lines.append(scene_baseline_compare_analysis(config, package, baseline_packages[scene_id]))
        lines.append("")
        fig_name = save_scene_baseline_comparison(config, package, baseline_packages[scene_id], numbering.next_figure())
        add_figure_block(
            lines,
            numbering,
            f"{config.title}本文方法与基线方法关键指标对比图",
            fig_name,
            [
                scene_baseline_compare_analysis(config, package, baseline_packages[scene_id]),
            ],
        )

        lines.append("#### （4）四层决策作用机理与有效性分析")
        lines.append("")
        table_no = numbering.next_table()
        lines.append(f"表6-{table_no} {config.title}关键决策作用分析表")
        lines.extend(md_table(
            ["关键窗口", "阶段决策层作用分析", "战术决策层作用分析", "机动决策层作用分析", "参数决策层作用分析", "有效性结论"],
            scene_decision_case_rows(config, package),
        ))
        lines.append("")
        lines.append(scene_decision_case_summary(config))
        lines.append("")
        table_no = numbering.next_table()
        lines.append(f"表6-{table_no} {config.title}四层决策作用归纳表")
        lines.extend(md_table(
            ["决策层", "本场景核心作用", "关键证据", "有效性说明"],
            scene_decision_layer_summary_rows(config, package),
        ))
        lines.append("")
        lines.append("上述归纳表不再横向重复三个场景的通用结论，而是针对本场景总结四层分别解决的核心问题、所依赖的关键证据以及它们对方法有效性的直接支撑。")
        lines.append("")

        lines.append("#### （5）场景结论")
        lines.append("")
        if scene_id == "S1":
            lines.append("场景一证明：在低风险正面对进条件下，分层决策方法能够形成“远距塑形—首轮放行—导弹来袭规避—截获重组—回转清场”的完整闭环。001401 最终形成4:0战果，高风险区累计滞留为0 s，最小敌我距离达到57.4 km，说明完整闭环并不是以牺牲生存余度换来的。")
        elif scene_id == "S2":
            lines.append("场景二证明：在400 s级预警失配条件下，系统仍能依靠机载连续跟踪维持压制骨架，并通过两次“规避—重组—再交战”循环恢复持续压制。225438 最终形成3:0且高风险区累计滞留为0 s，说明该方法的有效性来自一致的放行纪律和生存边界控制，而不是单次激进前压。")
        else:
            lines.append("场景三证明：在低空突防和高压近距条件下，分层决策方法的价值不在于追求更多击落，而在于首轮链条足够前移、规避足够及时、重组缓冲足够克制，并把首次我机损失显著后推。001339 虽仅形成1:1结果，但高风险区累计滞留压缩为111.0 s，首次我机损失后推到990.2 s，更符合高压样本的生存优先验证目标。")
        lines.append("")

    lines.append("### 6.3.5 三场景横向归纳与方法有效性讨论")
    lines.append("")
    table_no = numbering.next_table()
    lines.append(f"表6-{table_no} 三场景代表样本总体对比")
    lines.extend(md_table(
        ["场景", "代表样本", "任务条件", "首次放行", "首次规避", "高风险滞留", "敌机击落 / 我方损失", "分析重心"],
        [
            [
                "场景一",
                SCENARIOS["S1"].selected_label,
                "低风险正面对进、预警全程可用",
                fmt_time(packages["S1"]["milestones"].get("first_gate_pass_time_s")),
                fmt_time(SCENARIOS["S1"].decision_times.get("first_evade")),
                fmt_time(packages["S1"]["summary"].get("high_risk_breach_time_s")),
                f"{fmt_count(packages['S1']['summary'].get('enemy_kill_count'))} / {fmt_count(packages['S1']['summary'].get('friendly_loss_count'))}",
                "完整闭环是否成立",
            ],
            [
                "场景二",
                SCENARIOS["S2"].selected_label,
                "中风险持续压制、400 s级预警失配",
                fmt_time(packages["S2"]["milestones"].get("first_gate_pass_time_s")),
                fmt_time(SCENARIOS["S2"].decision_times.get("first_evade")),
                fmt_time(packages["S2"]["summary"].get("high_risk_breach_time_s")),
                f"{fmt_count(packages['S2']['summary'].get('enemy_kill_count'))} / {fmt_count(packages['S2']['summary'].get('friendly_loss_count'))}",
                "信息缺口下的恢复韧性",
            ],
            [
                "场景三",
                SCENARIOS["S3"].selected_label,
                "高风险低空突防、930 s级预警失配",
                fmt_time(packages["S3"]["milestones"].get("first_gate_pass_time_s")),
                fmt_time(SCENARIOS["S3"].decision_times.get("first_evade")),
                fmt_time(packages["S3"]["summary"].get("high_risk_breach_time_s")),
                f"{fmt_count(packages['S3']['summary'].get('enemy_kill_count'))} / {fmt_count(packages['S3']['summary'].get('friendly_loss_count'))}",
                "生存优先与损失后推",
            ],
        ],
    ))
    lines.append("")
    lines.append("总体对比表说明三个样本承担的任务并不相同：场景一看完整闭环能否在零高风险滞留下形成4:0，场景二看400 s级预警失配下能否重复恢复并维持3:0，场景三则看高压低空突防下能否把111.0 s高风险滞留和首次损失尽量后推。因此横向对比不能用单一指标排序，而要回到各场景的验证目标。")
    lines.append("")

    table_no = numbering.next_table()
    lines.append(f"表6-{table_no} 三场景本文方法相对基线的关键增益对比")
    lines.extend(md_table(
        ["场景", "本文稳定就绪", "基线稳定就绪", "本文窗口兑现率", "基线窗口兑现率", "本文敌机击落", "基线敌机击落", "本文我方损失", "基线我方损失"],
        cross_scene_baseline_gain_rows(packages, baseline_packages),
    ))
    lines.append("")
    lines.append("横向基线对比表说明，三类场景虽然任务目标不同，但本文方法都稳定体现出两个共同增益：一是更早进入可用火控状态，二是更高的发射窗口兑现质量。场景一和场景二进一步把这一优势兑现为更高击落数和更低我方损失；场景三则主要兑现为更短的高风险暴露时间，而不是单纯追求更高战果。")
    lines.append("")

    table_no = numbering.next_table()
    lines.append(f"表6-{table_no} 三场景关键方法有效性归纳")
    lines.extend(md_table(
        ["验证维度", "场景一证据", "场景二证据", "场景三证据", "归纳结论"],
        [
            [
                "阶段决策层",
                "99.6 s在200 km量级切入交战，322.0 s因导弹来袭切规避，446.6 s先重组再恢复。",
                "50.0 s在预警失配循环中仍保持交战，279.0 s和416.8 s两次规避后均能恢复。",
                "0.2 s即直接切入交战，131.8 s在高风险区压力下强制规避。",
                "阶段层并非按固定脚本推进，而是依据风险区、导弹来袭和图景可用性动态切换状态。",
            ],
            [
                "战术决策层",
                "由双路拖曳射击分化为左路钳形夹击、右路前后夹击，服务于标准闭环塑形。",
                "信息缺口下优先采用并列压制，恢复阶段再统一为前后夹击，服务于持续压制。",
                "高压近距下以左路前后夹击、右路钳形夹击快速前移交战链，服务于低空拦截。",
                "模板选择不是把风险区机械映射到固定动作，而是围绕场景任务重点进行分化。 ",
            ],
            [
                "机动决策层",
                "322.0 s规避由导弹来袭触发，且左路已接近DOR/DR边界。",
                "两次规避都在逼近DOR或触发硬底线预警时发生，恢复后仍可再次止压。",
                "第一次规避由导弹来袭与高风险区压力共同触发，第二次规避则突出导弹来袭的最高优先级。",
                "机动层的核心价值在于分级打断：一旦生存边界或来袭告警成立，就优先中断原攻击动作序列。",
            ],
            [
                "参数决策层",
                "5.0 s稳定跟踪门限与10.0 s最小重组缓冲共同约束放行与回转。",
                "信息缺口下仍坚持去重放行与10.0 s重组缓冲，不因图景受损而放宽门线。",
                "高压样本中同样保留10.0 s缓冲，但把首轮放行显著前移。",
                "参数层的有效性体现为：同一套门限框架在不同场景下可以前移或后移交战链，但不会放弃生存底线。 ",
            ],
        ],
    ))
    lines.append("")
    lines.append("方法有效性归纳表把三类样本的共同点和差异都压缩到了同一视角下。共同点在于三类场景都遵守5.0 s稳定交战就绪门限、10.0 s重组缓冲和导弹来袭最高优先级打断规则；差异在于阶段层切入时机、模板分工和后续规避密度会随场景任务目标而变化。")
    lines.append("综合三张横向表可以得到明确结论：场景一证明稳定火控、放行、规避和回转能够组成完整闭环；场景二证明图景失配并不破坏恢复纪律；场景三证明在高压低空突防下，系统可以通过前移首轮交战链和及时止压，把损失尽量后推。")
    lines.append("同时，三类场景也共同给出边界：稳定交战就绪本身不能替代发射门约束，持续压制不等于无限前压，高压样本中也不能承诺零损失。正是这些边界仍然存在，才说明文中结论来自真实约束下的决策表现，而不是理想化规则拼接。")
    lines.append("")

    lines.append("### 6.3.6 本节结论")
    lines.append("")
    lines.append("（1）在低风险正面对进、中风险信息缺口和高风险低空突防三类场景中，阶段决策层、战术决策层、机动决策层和参数决策层均能够在同一仿真过程中形成可追踪的因果链条。")
    lines.append("")
    lines.append("（2）首次放行时刻、规避触发时刻、截获重组时刻和恢复交战时刻均能通过实际日志证据得到解释：放行由控制距离与稳定火控共同闭合决定，规避由导弹来袭或生存边界逼近触发，回转恢复则受最小10 s重组缓冲约束。")
    lines.append("")
    lines.append("（3）三类代表样本的最终结果分别为4:0、3:0和1:1。其中场景一验证完整闭环，场景二验证恢复韧性，场景三验证生存优先。三者共同说明该方法并非依赖单一激进前压，而是能够在不同场景压力下保持一致的决策纪律与可解释性。")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Generated report: {REPORT_PATH}")
    print(f"Generated assets: {ASSET_DIR}")


if __name__ == "__main__":
    build_report()
