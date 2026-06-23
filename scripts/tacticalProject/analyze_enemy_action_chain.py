#!/usr/bin/env python
import argparse
import json
import re
from collections import Counter
from pathlib import Path


HIGH_LEVEL_RE = re.compile(
    r"\[(?P<time>\d{2}:\d{2}:\d{2})\].*?\[根因链-高层决策\] (?P<agent>B\d{4}) "
    r".*?phase=(?P<phase>[A-Z_]+) mode=(?P<mode>[a-z]+) action=(?P<action>[a-z_]+) "
    r".*?cmd=\((?P<alt_cmd>\d+),(?P<hdg_cmd>\d+),(?P<vel_cmd>\d+)\) "
    r"alt=(?P<alt>-?\d+)m vc=(?P<vc>-?\d+)m/s v_up=(?P<vup>-?\d+(?:\.\d+)?)m/s"
)
RTB_RE = re.compile(
    r"\[(?P<time>\d{2}:\d{2}:\d{2})\].*?\[根因链-RTB判定\] (?P<agent>B\d{4}) .*?e_low=(?P<e_low>[01])"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze enemy action chains from CAP rootcause logs.")
    parser.add_argument("log_files", nargs="+", help="Rootcause log files to analyze.")
    parser.add_argument("--window", type=int, default=10, help="Number of compressed segments before first e_low to keep.")
    parser.add_argument("--output", type=str, default="", help="Optional json output path.")
    return parser.parse_args()


def compress_records(records):
    segments = []
    for record in records:
        key = (
            record["phase"],
            record["mode"],
            record["action"],
            record["alt_cmd"],
            record["hdg_cmd"],
            record["vel_cmd"],
        )
        if segments and segments[-1]["key"] == key:
            segments[-1]["count"] += 1
            segments[-1]["end_time"] = record["time"]
            segments[-1]["last_alt_m"] = record["alt_m"]
            segments[-1]["last_vc_mps"] = record["vc_mps"]
            segments[-1]["last_v_up_mps"] = record["v_up_mps"]
        else:
            segments.append(
                {
                    "key": key,
                    "phase": record["phase"],
                    "mode": record["mode"],
                    "action": record["action"],
                    "cmd": [record["alt_cmd"], record["hdg_cmd"], record["vel_cmd"]],
                    "count": 1,
                    "start_time": record["time"],
                    "end_time": record["time"],
                    "start_alt_m": record["alt_m"],
                    "start_vc_mps": record["vc_mps"],
                    "start_v_up_mps": record["v_up_mps"],
                    "last_alt_m": record["alt_m"],
                    "last_vc_mps": record["vc_mps"],
                    "last_v_up_mps": record["v_up_mps"],
                }
            )
    for segment in segments:
        segment.pop("key", None)
    return segments


def analyze_log(path: Path, window: int):
    records_by_agent = {}
    first_elow = {}

    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        match = HIGH_LEVEL_RE.search(line)
        if match:
            agent = match.group("agent")
            records_by_agent.setdefault(agent, []).append(
                {
                    "line_no": line_no,
                    "time": match.group("time"),
                    "phase": match.group("phase"),
                    "mode": match.group("mode"),
                    "action": match.group("action"),
                    "alt_cmd": int(match.group("alt_cmd")),
                    "hdg_cmd": int(match.group("hdg_cmd")),
                    "vel_cmd": int(match.group("vel_cmd")),
                    "alt_m": int(match.group("alt")),
                    "vc_mps": int(match.group("vc")),
                    "v_up_mps": float(match.group("vup")),
                }
            )
            continue

        match = RTB_RE.search(line)
        if match and match.group("e_low") == "1":
            agent = match.group("agent")
            first_elow.setdefault(
                agent,
                {
                    "time": match.group("time"),
                    "line_no": line_no,
                },
            )

    summary = {
        "log_file": str(path),
        "agents": {},
    }

    for agent, records in sorted(records_by_agent.items()):
        segments = compress_records(records)
        action_counts = Counter(record["action"] for record in records)
        phase_counts = Counter(record["phase"] for record in records)
        mode_counts = Counter(record["mode"] for record in records)

        toggles = 0
        for idx in range(2, len(segments)):
            if segments[idx]["action"] == segments[idx - 2]["action"] and segments[idx]["action"] != segments[idx - 1]["action"]:
                toggles += 1

        pre_elow_segments = []
        first_elow_meta = first_elow.get(agent)
        if first_elow_meta is not None:
            cutoff = first_elow_meta["line_no"]
            eligible = [segment for segment in segments if segment["end_time"] <= first_elow_meta["time"]]
            pre_elow_segments = eligible[-window:]

        summary["agents"][agent] = {
            "record_count": len(records),
            "segment_count": len(segments),
            "action_switches": max(0, len(segments) - 1),
            "short_toggle_count": toggles,
            "first_e_low": first_elow_meta,
            "action_counts": dict(action_counts),
            "phase_counts": dict(phase_counts),
            "mode_counts": dict(mode_counts),
            "segments_head": segments[:12],
            "segments_before_first_e_low": pre_elow_segments,
        }

    return summary


def main():
    args = parse_args()
    payload = {"logs": [analyze_log(Path(log_file), args.window) for log_file in args.log_files]}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
