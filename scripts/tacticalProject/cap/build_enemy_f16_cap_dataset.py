#!/usr/bin/env python
import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


MODEL_FORWARD_TAG = "[_lowlevel_control][model_forward][state]"
ENERGY_L3_RE = re.compile(r"\[(?:HARD_RECOVERY|ENVELOPE)\]\[(B\d{4})\]\[step=(\d+)\].*stage=ENERGY_L3")
AGENT_RE = re.compile(r"\[(B\d{4})\]")
METRIC_RE = re.compile(
    r"model=(?P<model>[^\s]+)\s+in_alt=(?P<alt_idx>\d+)\s+in_hdg=(?P<hdg_idx>\d+)\s+in_vel=(?P<vel_idx>\d+).*?"
    r"alt=(?P<alt_m>-?\d+(?:\.\d+)?)m\s+vc=(?P<vc_mps>-?\d+(?:\.\d+)?)mps.*?"
    r"\bg=(?P<g>-?\d+(?:\.\d+)?)"
)
TIME_RE = re.compile(r"\[T\+(?P<time>\d+\.\d+)s\]")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
SEMANTICS_BY_MODEL_KEY = {
    "f16_legacy": "legacy_3x5x3",
    "f16_native_direct": "cap_native_15x17x7",
    "f16_native_residual": "cap_native_residual_15x17x7",
    "enemy_rule_pid": "legacy_3x5x3",
    "f16_online_override": "legacy_3x5x3",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a healthy enemy-F16 CAP dataset from tacticalProject CAP logs.")
    parser.add_argument(
        "--log-glob",
        action="append",
        default=[],
        help="Glob pattern(s) for CAP logs. Defaults to tacticalProject/cap_results/**/cap_*.log and root cap_*.log.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output .npz dataset path.")
    parser.add_argument(
        "--model-key",
        action="append",
        default=[],
        help="Keep only these low-level model keys, e.g. f16_native_direct, f16_legacy. Defaults to all.",
    )
    parser.add_argument("--agents", type=str, default="B0100,B0200,B0300,B0400")
    parser.add_argument("--exclude-after-energy-l3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-alt-m", type=float, default=6000.0)
    parser.add_argument("--min-vc-mps", type=float, default=170.0)
    parser.add_argument("--max-abs-g", type=float, default=6.0)
    parser.add_argument("--max-per-command", type=int, default=0, help="Optional cap per (alt,hdg,vel) command triple.")
    parser.add_argument("--command-semantics", type=str, default="", help="Force dataset command semantics instead of inferring from model keys.")
    parser.add_argument("--allow-mixed-semantics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-energy-clean-episodes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def default_globs(repo_root: Path):
    cap_root = repo_root / "scripts" / "tacticalProject" / "cap_results"
    return [
        str(cap_root / "cap_*.log"),
        str(cap_root / "**" / "cap_*.log"),
    ]


def collect_log_files(patterns):
    files = set()
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            path = Path(match)
            if path.is_file() and path.suffix.lower() == ".log" and not path.name.endswith("_rootcause.log"):
                files.add(path.resolve())
    return sorted(files)


def parse_numeric_array(text: str, expected_len: int | None = None):
    values = [float(token) for token in NUM_RE.findall(text)]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(values)} from: {text[:120]}")
    return values


def extract_array_after_token(line: str, token: str) -> str:
    token_idx = line.find(token)
    if token_idx < 0:
        raise ValueError(f"Token not found: {token}")
    start = line.find("[", token_idx)
    end = line.find("]", start)
    if start < 0 or end < 0:
        raise ValueError(f"Bracketed array not found after token: {token}")
    return line[start + 1 : end]


def find_first_energy_onset_by_line(log_path: Path, wanted_agents: set[str]):
    onset = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            match = ENERGY_L3_RE.search(line)
            if not match:
                continue
            agent = match.group(1)
            if agent not in wanted_agents or agent in onset:
                continue
            onset[agent] = line_no
    return onset


def infer_episode_semantics(model_keys_seen):
    semantics = {
        SEMANTICS_BY_MODEL_KEY[key]
        for key in model_keys_seen
        if key in SEMANTICS_BY_MODEL_KEY
    }
    if not semantics:
        return "unknown"
    if len(semantics) == 1:
        return next(iter(semantics))
    return "mixed"


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    patterns = args.log_glob or default_globs(repo_root)
    log_files = collect_log_files(patterns)
    if not log_files:
        raise FileNotFoundError("No CAP logs matched the provided patterns.")

    allowed_models = {item.strip() for item in args.model_key if item.strip()}
    wanted_agents = {item.strip() for item in args.agents.split(",") if item.strip()}

    obs_list = []
    action_list = []
    cmd_list = []
    episode_index_list = []
    log_index_list = []
    time_s_list = []
    alt_m_list = []
    vc_mps_list = []
    g_list = []
    model_keys = []
    episode_metadata = []
    command_counter = Counter()
    kept_per_command = Counter()
    next_episode_index = 0

    for log_idx, log_path in enumerate(log_files):
        onset_by_agent = find_first_energy_onset_by_line(log_path, wanted_agents)
        episode_idx_by_agent = {}

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                if MODEL_FORWARD_TAG not in line:
                    continue

                agent_match = AGENT_RE.search(line)
                metric_match = METRIC_RE.search(line)
                if not agent_match or not metric_match:
                    continue

                agent_id = agent_match.group(1)
                if agent_id not in wanted_agents:
                    continue
                if args.require_energy_clean_episodes and agent_id in onset_by_agent:
                    continue
                if args.exclude_after_energy_l3 and line_no >= onset_by_agent.get(agent_id, 10**12):
                    continue

                model_key = metric_match.group("model").strip()
                if allowed_models and model_key not in allowed_models:
                    continue

                alt_m = float(metric_match.group("alt_m"))
                vc_mps = float(metric_match.group("vc_mps"))
                g_load = float(metric_match.group("g"))
                if alt_m < args.min_alt_m or vc_mps < args.min_vc_mps or abs(g_load) > args.max_abs_g:
                    continue

                try:
                    obs = parse_numeric_array(extract_array_after_token(line, "inp="), expected_len=12)
                    action = [int(round(x)) for x in parse_numeric_array(extract_array_after_token(line, "out_raw="), expected_len=4)]
                except Exception:
                    continue

                cmd_key = (
                    int(metric_match.group("alt_idx")),
                    int(metric_match.group("hdg_idx")),
                    int(metric_match.group("vel_idx")),
                )
                if args.max_per_command > 0 and kept_per_command[cmd_key] >= args.max_per_command:
                    continue

                if agent_id not in episode_idx_by_agent:
                    episode_idx_by_agent[agent_id] = next_episode_index
                    next_episode_index += 1
                    episode_metadata.append(
                        {
                            "episode_index": episode_idx_by_agent[agent_id],
                            "log_path": str(log_path),
                            "agent_id": agent_id,
                            "first_energy_l3_line": int(onset_by_agent.get(agent_id, -1)),
                            "model_keys_seen": [],
                        }
                    )

                ep_idx = episode_idx_by_agent[agent_id]
                episode_metadata[ep_idx]["model_keys_seen"].append(model_key)

                time_match = TIME_RE.search(line)
                time_s = float(time_match.group("time")) if time_match else -1.0

                obs_list.append(obs)
                action_list.append(action)
                cmd_list.append(cmd_key)
                episode_index_list.append(ep_idx)
                log_index_list.append(log_idx)
                time_s_list.append(time_s)
                alt_m_list.append(alt_m)
                vc_mps_list.append(vc_mps)
                g_list.append(g_load)
                model_keys.append(model_key)
                command_counter[cmd_key] += 1
                kept_per_command[cmd_key] += 1

    if not obs_list:
        raise RuntimeError("No healthy CAP samples were extracted. Relax filters or use different log patterns.")

    for item in episode_metadata:
        item["model_keys_seen"] = sorted(set(item["model_keys_seen"]))
        item["command_semantics"] = infer_episode_semantics(item["model_keys_seen"])

    model_key_names = sorted(set(model_keys))
    model_key_to_index = {name: idx for idx, name in enumerate(model_key_names)}
    model_key_index_list = np.asarray([model_key_to_index[name] for name in model_keys], dtype=np.int32)

    observed_semantics = sorted(
        {
            item["command_semantics"]
            for item in episode_metadata
            if item.get("command_semantics")
        }
    )
    command_semantics = str(args.command_semantics or "").strip()
    if not command_semantics:
        effective_semantics = [item for item in observed_semantics if item not in ("unknown", "mixed")]
        mixed_present = any(item in ("mixed", "unknown") for item in observed_semantics)
        if mixed_present or len(effective_semantics) != 1:
            if not args.allow_mixed_semantics:
                raise RuntimeError(
                    f"Dataset semantics are not pure: observed={observed_semantics}. "
                    "Use a pure log set or pass --allow-mixed-semantics/--command-semantics intentionally."
                )
            command_semantics = "mixed"
        else:
            command_semantics = effective_semantics[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        obs=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(action_list, dtype=np.int64),
        cmd_indices=np.asarray(cmd_list, dtype=np.int64),
        episode_index=np.asarray(episode_index_list, dtype=np.int32),
        log_index=np.asarray(log_index_list, dtype=np.int32),
        time_s=np.asarray(time_s_list, dtype=np.float32),
        alt_m=np.asarray(alt_m_list, dtype=np.float32),
        vc_mps=np.asarray(vc_mps_list, dtype=np.float32),
        g_load=np.asarray(g_list, dtype=np.float32),
        model_key_index=model_key_index_list,
    )

    metadata = {
        "dataset_path": str(output_path),
        "log_count": len(log_files),
        "sample_count": len(obs_list),
        "episode_count": len(episode_metadata),
        "allowed_models": sorted(allowed_models) if allowed_models else ["all"],
        "filters": {
            "exclude_after_energy_l3": bool(args.exclude_after_energy_l3),
            "require_energy_clean_episodes": bool(args.require_energy_clean_episodes),
            "min_alt_m": float(args.min_alt_m),
            "min_vc_mps": float(args.min_vc_mps),
            "max_abs_g": float(args.max_abs_g),
            "max_per_command": int(args.max_per_command),
        },
        "command_semantics": command_semantics,
        "model_key_names": model_key_names,
        "episodes": episode_metadata,
        "command_counts": [
            {
                "alt_idx": int(key[0]),
                "hdg_idx": int(key[1]),
                "vel_idx": int(key[2]),
                "count": int(count),
            }
            for key, count in command_counter.most_common()
        ],
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
