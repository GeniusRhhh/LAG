from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


TIME_RE = re.compile(r"\bt=(\d+(?:\.\d+)?)s\b")
STATE_TIME_RE = re.compile(r"\[状态\]\s+\[(\d+(?:\.\d+)?)s\]")
AWACS_TRACK_RE = re.compile(r"探测航迹数=(\d+)")
TARGET_TIMELINE_RE = re.compile(
    r"^\[\d{2}:\d{2}:\d{2}\]\s+"
    r"(?P<target>B\d{4})\s+awacs=(?P<awacs>\d+)s\s+fcr=(?P<fcr>\d+)s\s+ready=(?P<ready>\d+)s\s+"
    r"zones=(?P<zones>.+)$"
)
SNAPSHOT_RE = re.compile(
    r"\[BATTLE_SNAPSHOT\]\s+t=(?P<time>\d+(?:\.\d+)?)s\s+"
    r"A_alive=(?P<a_alive>\d+)\s+B_alive=(?P<b_alive>\d+)\s+"
    r"A_ms=(?P<a_ms>\d+)\s+B_ms=(?P<b_ms>\d+)\s+"
    r"zones\(H/M/L/O\)=(?P<h>\d+)/(?P<m>\d+)/(?P<l>\d+)/(?P<o>\d+)\s+"
    r"tactic\(L/R\)=(?P<left_tactic>[^/]+)/(?P<right_tactic>[^ ]+)\s+"
    r"tracking=(?P<tracking>[A-Z]+)\s+relay=(?P<relay>[A-Z]+)\s+"
    r"gate=(?P<gate_pass>\d+)/(?P<gate_block>\d+)\s+"
    r"relay_ok=(?P<relay_success>\d+)/(?P<relay_attempt>\d+)"
)
STATUS_RE = re.compile(
    r"\[状态\]\s+\[(?P<time>\d+(?:\.\d+)?)s\]\s+(?P<state>[A-Z]+)"
    r"(?:\s+\|\s+战术:(?P<tactic>.*?))?\s+\|\s+全局最小距离:(?P<distance>\d+(?:\.\d+)?)km"
)
LAUNCH_RECORD_RE = re.compile(
    r"t=(?P<time>\d+)s\s+shooter=(?P<shooter>[AB]\d{4})\s+target=(?P<target>[AB]\d{4})\s+"
    r"missile=(?P<missile>[A-Z0-9]+)\s+model=(?P<model>\S+)\s+guide=(?P<guide>\S+)\s+"
    r"dist=(?P<distance>\d+(?:\.\d+)?)km"
)
MISSILE_END_RE = re.compile(
    r"\[导弹结束\]\s+missile=(?P<missile>[A-Z0-9]+)\s+state=(?P<state>[A-Z]+)\s+"
    r"target=(?P<target>[AB]\d{4})\s+tof=(?P<tof>\d+(?:\.\d+)?)s\s+guide=(?P<guide>[AB]\d{4})"
)
MISSILE_END_DETAIL_RE = re.compile(
    r"\[MISSILE_END_DETAIL\]\s+missile=(?P<missile>[A-Z0-9]+)\s+reason=(?P<reason>[a-z_]+)"
)
FINAL_SUMMARY_RE = re.compile(
    r"\[COOP_VERIFY\]\s+SUMMARY\s+pass=(?P<gate_pass>\d+)\s+block=(?P<gate_block>\d+)\s+"
    r"pass_rate=(?P<pass_rate>\d+(?:\.\d+)?)%\s+"
    r"relay_attempt=(?P<relay_attempt>\d+)\s+relay_success=(?P<relay_success>\d+)\s+"
    r"relay_rate=(?P<relay_rate>\d+(?:\.\d+)?)%\s+midcourse_peak=(?P<midcourse_peak>\d+)"
)
MISSION_RESULT_RE = re.compile(r"我方存活:\s*(\d+)/4,\s*敌方存活:\s*(\d+)/4")


@dataclass
class LineInfo:
    index: int
    text: str
    explicit_time: Optional[float]


def read_lines(log_path: Path) -> List[LineInfo]:
    raw_lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    result: List[LineInfo] = []
    for index, text in enumerate(raw_lines):
        explicit_time = extract_explicit_time(text)
        result.append(LineInfo(index=index, text=text, explicit_time=explicit_time))
    return result


def extract_explicit_time(text: str) -> Optional[float]:
    state_match = STATE_TIME_RE.search(text)
    if state_match:
        return float(state_match.group(1))
    time_match = TIME_RE.search(text)
    if time_match:
        return float(time_match.group(1))
    return None


def build_time_lookup(lines: List[LineInfo]) -> Dict[int, Optional[float]]:
    prev_idx: Optional[int] = None
    prev_time: Optional[float] = None
    prev_map: Dict[int, Tuple[Optional[int], Optional[float]]] = {}
    for item in lines:
        if item.explicit_time is not None:
            prev_idx = item.index
            prev_time = item.explicit_time
        prev_map[item.index] = (prev_idx, prev_time)

    next_idx: Optional[int] = None
    next_time: Optional[float] = None
    next_map: Dict[int, Tuple[Optional[int], Optional[float]]] = {}
    for item in reversed(lines):
        if item.explicit_time is not None:
            next_idx = item.index
            next_time = item.explicit_time
        next_map[item.index] = (next_idx, next_time)

    lookup: Dict[int, Optional[float]] = {}
    for item in lines:
        if item.explicit_time is not None:
            lookup[item.index] = item.explicit_time
            continue
        prev_pair = prev_map[item.index]
        next_pair = next_map[item.index]
        candidates: List[Tuple[int, float]] = []
        if prev_pair[0] is not None and prev_pair[1] is not None:
            candidates.append((abs(item.index - prev_pair[0]), prev_pair[1]))
        if next_pair[0] is not None and next_pair[1] is not None:
            candidates.append((abs(next_pair[0] - item.index), next_pair[1]))
        lookup[item.index] = min(candidates, key=lambda x: x[0])[1] if candidates else None
    return lookup


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    ensure_dir(output_base.parent)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def compress_points(rows: List[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    last_value = object()
    for row in sorted(rows, key=lambda item: float(item["time_s"])):
        value = row[key]
        if value != last_value:
            result.append(row)
            last_value = value
    return result


def parse_awacs_tracks(lines: List[LineInfo], time_lookup: Dict[int, Optional[float]]) -> List[Dict[str, object]]:
    by_time: Dict[float, int] = {}
    for item in lines:
        match = AWACS_TRACK_RE.search(item.text)
        if not match:
            continue
        time_s = time_lookup.get(item.index)
        if time_s is None:
            continue
        tracks = int(match.group(1))
        by_time[time_s] = max(by_time.get(time_s, 0), tracks)
    rows = [{"time_s": t, "awacs_tracks": by_time[t]} for t in sorted(by_time)]
    return compress_points(rows, "awacs_tracks")


def parse_target_timeline(lines: List[LineInfo]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in lines:
        match = TARGET_TIMELINE_RE.search(item.text)
        if not match:
            continue
        rows.append(
            {
                "target": match.group("target"),
                "awacs_s": int(match.group("awacs")),
                "fcr_s": int(match.group("fcr")),
                "ready_s": int(match.group("ready")),
                "zones": match.group("zones"),
            }
        )
    return rows


def parse_snapshots(lines: List[LineInfo]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in lines:
        match = SNAPSHOT_RE.search(item.text)
        if not match:
            continue
        rows.append(
            {
                "time_s": float(match.group("time")),
                "a_alive": int(match.group("a_alive")),
                "b_alive": int(match.group("b_alive")),
                "a_ms": int(match.group("a_ms")),
                "b_ms": int(match.group("b_ms")),
                "zone_high": int(match.group("h")),
                "zone_medium": int(match.group("m")),
                "zone_low": int(match.group("l")),
                "zone_outer": int(match.group("o")),
                "left_tactic": match.group("left_tactic"),
                "right_tactic": match.group("right_tactic"),
                "tracking": match.group("tracking"),
                "relay": match.group("relay"),
                "gate_pass": int(match.group("gate_pass")),
                "gate_block": int(match.group("gate_block")),
                "relay_success": int(match.group("relay_success")),
                "relay_attempt": int(match.group("relay_attempt")),
            }
        )
    return sorted(rows, key=lambda row: row["time_s"])


def parse_states(lines: List[LineInfo]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in lines:
        match = STATUS_RE.search(item.text)
        if not match:
            continue
        rows.append(
            {
                "time_s": float(match.group("time")),
                "state": match.group("state"),
                "tactic_text": (match.group("tactic") or "").strip(),
                "distance_km": float(match.group("distance")),
            }
        )
    return rows


def parse_launch_records(lines: List[LineInfo]) -> Dict[str, Dict[str, object]]:
    launches: Dict[str, Dict[str, object]] = {}
    for item in lines:
        match = LAUNCH_RECORD_RE.search(item.text)
        if not match:
            continue
        launches[match.group("missile")] = {
            "launch_time_s": float(match.group("time")),
            "shooter": match.group("shooter"),
            "target": match.group("target"),
            "distance_km": float(match.group("distance")),
            "guide_mode": match.group("guide"),
        }
    return launches


def parse_missile_events(
    lines: List[LineInfo], launches: Dict[str, Dict[str, object]]
) -> Tuple[List[Dict[str, object]], Counter, Counter]:
    hit_rows: List[Dict[str, object]] = []
    miss_reasons_friendly: Counter = Counter()
    miss_reasons_enemy: Counter = Counter()
    seen_hit: set[str] = set()
    seen_miss_reason: set[Tuple[str, str]] = set()

    for item in lines:
        hit_match = MISSILE_END_RE.search(item.text)
        if hit_match:
            missile = hit_match.group("missile")
            state = hit_match.group("state")
            tof_s = float(hit_match.group("tof"))
            launch = launches.get(missile)
            side = "friendly" if missile.startswith("A") else "enemy"
            hit_time_s = (
                launch["launch_time_s"] + tof_s if launch and state == "HIT" else None
            )
            if state == "HIT" and missile not in seen_hit:
                seen_hit.add(missile)
                hit_rows.append(
                    {
                        "missile": missile,
                        "side": side,
                        "target": hit_match.group("target"),
                        "shooter": launch["shooter"] if launch else hit_match.group("guide"),
                        "hit_time_s": hit_time_s,
                        "tof_s": tof_s,
                    }
                )
            continue

        miss_match = MISSILE_END_DETAIL_RE.search(item.text)
        if not miss_match:
            continue
        missile = miss_match.group("missile")
        reason = miss_match.group("reason")
        key = (missile, reason)
        if key in seen_miss_reason:
            continue
        seen_miss_reason.add(key)
        if missile.startswith("A"):
            miss_reasons_friendly[reason] += 1
        else:
            miss_reasons_enemy[reason] += 1

    hit_rows.sort(key=lambda row: float(row["hit_time_s"]))
    return hit_rows, miss_reasons_friendly, miss_reasons_enemy


def parse_final_summary(lines: List[LineInfo]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for item in lines:
        match = FINAL_SUMMARY_RE.search(item.text)
        if match:
            summary = {
                "gate_pass": int(match.group("gate_pass")),
                "gate_block": int(match.group("gate_block")),
                "pass_rate_pct": float(match.group("pass_rate")),
                "relay_attempt": int(match.group("relay_attempt")),
                "relay_success": int(match.group("relay_success")),
                "relay_rate_pct": float(match.group("relay_rate")),
                "midcourse_peak": int(match.group("midcourse_peak")),
            }
        mission_match = MISSION_RESULT_RE.search(item.text)
        if mission_match:
            summary["friendly_alive_final"] = int(mission_match.group(1))
            summary["enemy_alive_final"] = int(mission_match.group(2))
    return summary


def build_ready_curve(target_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ordered = sorted((int(row["ready_s"]), row["target"]) for row in target_rows if row["ready_s"] >= 0)
    cumulative = 0
    rows = [{"time_s": 0, "ready_targets": 0}]
    seen: set[str] = set()
    for time_s, target in ordered:
        if target in seen:
            continue
        seen.add(target)
        cumulative += 1
        rows.append({"time_s": time_s, "ready_targets": cumulative})
    return rows


def build_segments(rows: List[Dict[str, object]], label_key: str, end_time: float) -> List[Tuple[float, float, str]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: float(row["time_s"]))
    segments: List[Tuple[float, float, str]] = []
    for idx, row in enumerate(ordered):
        start = float(row["time_s"])
        end = float(ordered[idx + 1]["time_s"]) if idx + 1 < len(ordered) else float(end_time)
        label = str(row[label_key] or "-")
        if end <= start:
            continue
        segments.append((start, end, label))
    return segments


def color_map(labels: Iterable[str]) -> Dict[str, str]:
    palette = [
        "#2F5597",
        "#C0504D",
        "#4F81BD",
        "#9BBB59",
        "#8064A2",
        "#F79646",
        "#00B0F0",
        "#7F7F7F",
        "#1F4E78",
        "#943634",
    ]
    mapping: Dict[str, str] = {}
    for idx, label in enumerate(sorted(set(labels))):
        mapping[label] = palette[idx % len(palette)]
    return mapping


def plot_awacs_tracks(rows: List[Dict[str, object]], output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    times = [float(row["time_s"]) for row in rows]
    values = [int(row["awacs_tracks"]) for row in rows]
    ax.step(times, values, where="post", linewidth=2.4, color="#2F5597")
    ax.scatter(times, values, color="#2F5597", s=42)
    ax.set_title("协同探测有效性：AWACS可用航迹数随时间变化")
    ax.set_xlabel("仿真时间 T+s")
    ax.set_ylabel("可用敌方航迹数")
    ax.set_ylim(-0.2, 4.5)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.grid(True, alpha=0.3)
    if times:
        ax.annotate("T+50s 全量感知完成", xy=(times[-1], values[-1]), xytext=(times[-1] + 30, values[-1] - 0.7),
                    arrowprops={"arrowstyle": "->", "color": "#2F5597"}, fontsize=10)
    save_figure(fig, output_base)


def plot_ready_curve(rows: List[Dict[str, object]], output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    times = [float(row["time_s"]) for row in rows]
    values = [int(row["ready_targets"]) for row in rows]
    ax.step(times, values, where="post", linewidth=2.4, color="#C0504D")
    ax.scatter(times, values, color="#C0504D", s=42)
    ax.set_title("协同跟踪有效性：稳定 Ready 目标数累积曲线")
    ax.set_xlabel("仿真时间 T+s")
    ax.set_ylabel("累计 Ready 目标数")
    ax.set_ylim(-0.2, 4.5)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.grid(True, alpha=0.3)
    if len(times) > 1:
        ax.annotate("T+192s 首个 Ready", xy=(times[1], values[1]), xytext=(times[1] + 35, values[1] + 0.35),
                    arrowprops={"arrowstyle": "->", "color": "#C0504D"}, fontsize=10)
        ax.annotate("T+237s 全目标 4/4 Ready", xy=(times[-1], values[-1]), xytext=(times[-1] - 130, values[-1] - 0.8),
                    arrowprops={"arrowstyle": "->", "color": "#C0504D"}, fontsize=10)
    save_figure(fig, output_base)


def plot_state_tactic_timeline(
    state_rows: List[Dict[str, object]],
    snapshot_rows: List[Dict[str, object]],
    output_base: Path,
    end_time: float,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    state_segments = build_segments(state_rows, "state", end_time)
    left_tactic_rows = [{"time_s": row["time_s"], "left_tactic": row["left_tactic"]} for row in snapshot_rows]
    right_tactic_rows = [{"time_s": row["time_s"], "right_tactic": row["right_tactic"]} for row in snapshot_rows]
    left_segments = build_segments(left_tactic_rows, "left_tactic", end_time)
    right_segments = build_segments(right_tactic_rows, "right_tactic", end_time)

    labels = [seg[2] for seg in state_segments + left_segments + right_segments]
    cmap = color_map(labels)
    rows_cfg = [
        ("系统状态", 28, state_segments),
        ("左侧模板", 18, left_segments),
        ("右侧模板", 8, right_segments),
    ]

    for row_label, y, segments in rows_cfg:
        for start, end, label in segments:
            ax.broken_barh([(start, end - start)], (y, 6), facecolors=cmap[label], edgecolors="white")
            if end - start >= 35:
                ax.text((start + end) / 2, y + 3, label, ha="center", va="center", fontsize=8, color="white")

    ax.set_xlim(0, end_time)
    ax.set_ylim(4, 38)
    ax.set_yticks([31, 21, 11])
    ax.set_yticklabels([item[0] for item in rows_cfg])
    ax.set_xlabel("仿真时间 T+s")
    ax.set_title("分层决策有效性：系统状态与主模板演化时间带")
    ax.grid(True, axis="x", alpha=0.25)
    legend_handles = [Patch(facecolor=cmap[label], label=label) for label in sorted(cmap)]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    save_figure(fig, output_base)


def plot_alive_and_missiles(snapshot_rows: List[Dict[str, object]], output_base: Path) -> None:
    times = [float(row["time_s"]) for row in snapshot_rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)

    axes[0].step(times, [row["a_alive"] for row in snapshot_rows], where="post", linewidth=2.3, label="我方存活数", color="#2F5597")
    axes[0].step(times, [row["b_alive"] for row in snapshot_rows], where="post", linewidth=2.3, label="敌方存活数", color="#C0504D")
    axes[0].set_title("打断机制有效性：敌我存活数变化")
    axes[0].set_ylabel("存活数")
    axes[0].set_ylim(-0.2, 4.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].step(times, [row["a_ms"] for row in snapshot_rows], where="post", linewidth=2.3, label="我方剩余导弹", color="#4F81BD")
    axes[1].step(times, [row["b_ms"] for row in snapshot_rows], where="post", linewidth=2.3, label="敌方剩余导弹", color="#F79646")
    axes[1].set_title("持续作战能力：敌我剩余导弹数变化")
    axes[1].set_xlabel("仿真时间 T+s")
    axes[1].set_ylabel("剩余导弹数")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    save_figure(fig, output_base)


def plot_gate_and_relay(snapshot_rows: List[Dict[str, object]], output_base: Path) -> None:
    times = [float(row["time_s"]) for row in snapshot_rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)

    axes[0].step(times, [row["gate_pass"] for row in snapshot_rows], where="post", linewidth=2.3, label="gate_pass", color="#2F5597")
    axes[0].step(times, [row["gate_block"] for row in snapshot_rows], where="post", linewidth=2.3, label="gate_block", color="#C0504D")
    axes[0].set_title("发射门禁有效性：通过/拦截累积曲线")
    axes[0].set_ylabel("累计次数")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].step(times, [row["relay_success"] for row in snapshot_rows], where="post", linewidth=2.3, label="relay_success", color="#9BBB59")
    axes[1].step(times, [row["relay_attempt"] for row in snapshot_rows], where="post", linewidth=2.3, label="relay_attempt", color="#8064A2")
    axes[1].set_title("接力制导有效性：成功/尝试累积曲线")
    axes[1].set_xlabel("仿真时间 T+s")
    axes[1].set_ylabel("累计次数")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    save_figure(fig, output_base)


def plot_risk_zone_counts(snapshot_rows: List[Dict[str, object]], output_base: Path) -> None:
    times = [float(row["time_s"]) for row in snapshot_rows]
    high = [row["zone_high"] for row in snapshot_rows]
    medium = [row["zone_medium"] for row in snapshot_rows]
    low = [row["zone_low"] for row in snapshot_rows]
    outer = [row["zone_outer"] for row in snapshot_rows]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.stackplot(times, high, medium, low, outer, labels=["高风险区", "中风险区", "低风险区", "区外"], colors=["#C0504D", "#F79646", "#9BBB59", "#D9E1F2"])
    ax.set_title("系统效果：敌方目标在风险区中的分布变化")
    ax.set_xlabel("仿真时间 T+s")
    ax.set_ylabel("敌方目标数")
    ax.set_ylim(0, 4.2)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    save_figure(fig, output_base)


def plot_missile_end_reasons(
    miss_reasons_friendly: Counter, miss_reasons_enemy: Counter, output_base: Path
) -> None:
    friendly_labels = list(miss_reasons_friendly.keys())
    enemy_labels = list(miss_reasons_enemy.keys())
    labels = sorted(set(friendly_labels + enemy_labels))
    friendly_values = [miss_reasons_friendly.get(label, 0) for label in labels]
    enemy_values = [miss_reasons_enemy.get(label, 0) for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    axes[0].bar(labels, friendly_values, color="#4F81BD")
    axes[0].set_title("我方未命中导弹失效原因")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, enemy_values, color="#C0504D")
    axes[1].set_title("敌方未命中导弹失效原因")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("导引链质量分析：导弹失效原因统计")
    save_figure(fig, output_base)


def plot_kill_timeline(hit_rows: List[Dict[str, object]], output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    y_map = {"friendly": 1, "enemy": 0}
    color_map_local = {"friendly": "#2F5597", "enemy": "#C0504D"}
    label_map = {"friendly": "我方击落敌机", "enemy": "敌方击落我机"}
    shown = set()
    for row in hit_rows:
        side = str(row["side"])
        y = y_map[side]
        color = color_map_local[side]
        label = label_map[side]
        ax.scatter(float(row["hit_time_s"]), y, color=color, s=70, label=label if label not in shown else None)
        shown.add(label)
        ax.text(float(row["hit_time_s"]) + 8, y + 0.05, f"{row['shooter']}→{row['target']}", fontsize=9)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["敌方击落我机", "我方击落敌机"])
    ax.set_xlabel("仿真时间 T+s")
    ax.set_title("战果形成过程：击杀事件时间图")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="upper right")
    save_figure(fig, output_base)


def write_summary_md(
    output_path: Path,
    summary: Dict[str, object],
    target_rows: List[Dict[str, object]],
    snapshot_rows: List[Dict[str, object]],
    hit_rows: List[Dict[str, object]],
    miss_reasons_friendly: Counter,
    miss_reasons_enemy: Counter,
    figure_dir: Path,
) -> None:
    latest_snapshot = snapshot_rows[-1] if snapshot_rows else {}
    lines: List[str] = []
    lines.append("# 第六章验证素材自动生成摘要")
    lines.append("")
    lines.append(f"- 生成时间：自动脚本输出")
    lines.append(f"- 发射门禁：pass={summary.get('gate_pass')} block={summary.get('gate_block')} pass_rate={summary.get('pass_rate_pct')}%")
    lines.append(f"- 接力制导：attempt={summary.get('relay_attempt')} success={summary.get('relay_success')} relay_rate={summary.get('relay_rate_pct')}%")
    lines.append(f"- 中制导峰值：{summary.get('midcourse_peak')}")
    lines.append(f"- 最终存活：我方 {summary.get('friendly_alive_final')}/4，敌方 {summary.get('enemy_alive_final')}/4")
    lines.append("")
    lines.append("## 目标里程碑")
    for row in target_rows:
        lines.append(
            f"- {row['target']}: awacs={row['awacs_s']}s, fcr={row['fcr_s']}s, ready={row['ready_s']}s, zones={row['zones']}"
        )
    lines.append("")
    lines.append("## 击杀事件")
    for row in hit_rows:
        lines.append(
            f"- T+{row['hit_time_s']:.1f}s: {row['shooter']} -> {row['target']} ({'我方' if row['side']=='friendly' else '敌方'})"
        )
    lines.append("")
    lines.append("## 导弹失效原因")
    lines.append(f"- 我方：{dict(miss_reasons_friendly)}")
    lines.append(f"- 敌方：{dict(miss_reasons_enemy)}")
    lines.append("")
    lines.append("## 生成图表")
    for figure_name in sorted(path.name for path in figure_dir.glob("*.png")):
        lines.append(f"- {figure_name}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_raw_tables(
    csv_dir: Path,
    awacs_rows: List[Dict[str, object]],
    target_rows: List[Dict[str, object]],
    snapshot_rows: List[Dict[str, object]],
    state_rows: List[Dict[str, object]],
    ready_rows: List[Dict[str, object]],
    hit_rows: List[Dict[str, object]],
    miss_reasons_friendly: Counter,
    miss_reasons_enemy: Counter,
) -> None:
    write_csv(csv_dir / "awacs_tracks.csv", awacs_rows)
    write_csv(csv_dir / "target_timeline.csv", target_rows)
    write_csv(csv_dir / "battle_snapshots.csv", snapshot_rows)
    write_csv(csv_dir / "state_timeline.csv", state_rows)
    write_csv(csv_dir / "ready_cumulative.csv", ready_rows)
    write_csv(csv_dir / "kill_timeline.csv", hit_rows)
    reason_rows = []
    for side, counter in (("friendly", miss_reasons_friendly), ("enemy", miss_reasons_enemy)):
        for reason, count in counter.items():
            reason_rows.append({"side": side, "reason": reason, "count": count})
    write_csv(csv_dir / "missile_end_reasons.csv", reason_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 6 validation assets from a CAP log.")
    parser.add_argument("--log", required=True, help="Path to the CAP log file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()

    log_path = Path(args.log).resolve()
    out_dir = Path(args.out).resolve()
    figure_dir = out_dir / "figures"
    csv_dir = out_dir / "csv"
    ensure_dir(figure_dir)
    ensure_dir(csv_dir)

    lines = read_lines(log_path)
    time_lookup = build_time_lookup(lines)

    awacs_rows = parse_awacs_tracks(lines, time_lookup)
    target_rows = parse_target_timeline(lines)
    snapshot_rows = parse_snapshots(lines)
    state_rows = parse_states(lines)
    ready_rows = build_ready_curve(target_rows)
    launches = parse_launch_records(lines)
    hit_rows, miss_reasons_friendly, miss_reasons_enemy = parse_missile_events(lines, launches)
    summary = parse_final_summary(lines)

    export_raw_tables(
        csv_dir=csv_dir,
        awacs_rows=awacs_rows,
        target_rows=target_rows,
        snapshot_rows=snapshot_rows,
        state_rows=state_rows,
        ready_rows=ready_rows,
        hit_rows=hit_rows,
        miss_reasons_friendly=miss_reasons_friendly,
        miss_reasons_enemy=miss_reasons_enemy,
    )

    end_time = max([row["time_s"] for row in snapshot_rows], default=1200.0)
    plot_awacs_tracks(awacs_rows, figure_dir / "fig01_awacs_tracks")
    plot_ready_curve(ready_rows, figure_dir / "fig02_ready_targets")
    plot_state_tactic_timeline(state_rows, snapshot_rows, figure_dir / "fig03_state_tactic_timeline", end_time)
    plot_alive_and_missiles(snapshot_rows, figure_dir / "fig04_alive_and_missiles")
    plot_gate_and_relay(snapshot_rows, figure_dir / "fig05_gate_and_relay")
    plot_risk_zone_counts(snapshot_rows, figure_dir / "fig06_risk_zone_counts")
    plot_missile_end_reasons(miss_reasons_friendly, miss_reasons_enemy, figure_dir / "fig07_missile_end_reasons")
    plot_kill_timeline(hit_rows, figure_dir / "fig08_kill_timeline")

    write_summary_md(
        output_path=out_dir / "README_ch6_assets.md",
        summary=summary,
        target_rows=target_rows,
        snapshot_rows=snapshot_rows,
        hit_rows=hit_rows,
        miss_reasons_friendly=miss_reasons_friendly,
        miss_reasons_enemy=miss_reasons_enemy,
        figure_dir=figure_dir,
    )

    (out_dir / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Generated assets in: {out_dir}")


if __name__ == "__main__":
    main()
