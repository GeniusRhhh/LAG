#!/usr/bin/env python
import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lqyLAG.algorithms.ppo.ppo_actor import PPOActor
from lqyLAG.config import get_config
from envs.JSBSim.model.baseline_actor import BaselineActor


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior-clone an enemy F16 CAP low-level model from healthy CAP logs.")
    parser.add_argument("--dataset", type=str, required=True, help="Input dataset .npz from build_enemy_f16_cap_dataset.py")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-chunks", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--balance-by-command", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmstart-baseline", type=str, default="")
    parser.add_argument("--warmstart-ppo-actor", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--export-name", type=str, default="f16_enemy_cap_bc_direct.pt")
    return parser.parse_args()


def load_dataset_metadata(dataset_path: Path):
    meta_path = dataset_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def actor_uses_mlp_actlayer(state_dict) -> bool:
    return any(key.startswith("act.mlp.") or key.startswith("mlp.") for key in state_dict.keys())


def build_ppo_args(use_mlp_actlayer: bool):
    parser = get_config()
    args = parser.parse_args([])
    args.hidden_size = "128 128"
    args.act_hidden_size = "128 128" if use_mlp_actlayer else ""
    args.activation_id = 1
    args.use_feature_normalization = False
    args.use_recurrent_policy = True
    args.recurrent_hidden_size = 128
    args.recurrent_hidden_layers = 1
    args.gain = 0.01
    args.use_prior = False
    return args


def copy_matching_state(src_state, dst_module, extra_prefix_map=None) -> int:
    dst_state = dst_module.state_dict()
    matched = 0
    for key, value in dst_state.items():
        src_key = key
        if extra_prefix_map:
            for dst_prefix, src_prefix in extra_prefix_map.items():
                if key.startswith(dst_prefix):
                    src_key = src_prefix + key[len(dst_prefix) :]
                    break
        if src_key in src_state and src_state[src_key].shape == value.shape:
            dst_state[key] = src_state[src_key]
            matched += 1
    dst_module.load_state_dict(dst_state, strict=False)
    return matched


def export_ppo_actor_to_baseline(ppo_actor_path: Path, export_path: Path, use_mlp_actlayer: bool):
    ppo_state = torch.load(ppo_actor_path, map_location="cpu", weights_only=True)
    actor = BaselineActor(input_dim=12, use_mlp_actlayer=use_mlp_actlayer)
    copy_matching_state(ppo_state, actor)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), export_path)


def build_chunks(obs, actions, cmd_indices, episode_index, seq_len: int, stride: int):
    chunks = []
    unique_episodes = sorted(set(int(x) for x in episode_index.tolist()))
    for ep_idx in unique_episodes:
        mask = episode_index == ep_idx
        ep_obs = obs[mask]
        ep_actions = actions[mask]
        ep_cmd = cmd_indices[mask]
        if len(ep_obs) < seq_len:
            continue
        for start in range(0, len(ep_obs) - seq_len + 1, max(1, stride)):
            end = start + seq_len
            chunks.append(
                {
                    "episode": ep_idx,
                    "obs": ep_obs[start:end],
                    "actions": ep_actions[start:end],
                    "cmd_key": tuple(int(x) for x in ep_cmd[start]),
                }
            )
    return chunks


def split_episodes(episode_index, val_ratio: float, seed: int):
    episode_ids = sorted(set(int(x) for x in episode_index.tolist()))
    rng = random.Random(seed)
    rng.shuffle(episode_ids)
    val_count = max(1, int(round(len(episode_ids) * val_ratio))) if len(episode_ids) > 1 else 0
    val_set = set(episode_ids[:val_count])
    train_set = set(episode_ids[val_count:])
    if not train_set and val_set:
        train_set.add(val_set.pop())
    return train_set, val_set


def flatten_chunk_batch(batch):
    obs_bt = np.stack([item["obs"] for item in batch], axis=1)
    act_bt = np.stack([item["actions"] for item in batch], axis=1)
    seq_len, batch_size = obs_bt.shape[0], obs_bt.shape[1]
    masks = np.ones((seq_len, batch_size, 1), dtype=np.float32)
    masks[0, :, 0] = 0.0
    return (
        obs_bt.reshape(seq_len * batch_size, obs_bt.shape[-1]),
        act_bt.reshape(seq_len * batch_size, act_bt.shape[-1]),
        masks.reshape(seq_len * batch_size, 1),
        batch_size,
    )


def iterate_batches(chunks, batch_chunks: int, balance_by_command: bool, seed: int):
    rng = np.random.default_rng(seed)
    if not chunks:
        return
    if balance_by_command:
        counts = Counter(item["cmd_key"] for item in chunks)
        weights = np.asarray([1.0 / math.sqrt(counts[item["cmd_key"]]) for item in chunks], dtype=np.float64)
        weights = weights / weights.sum()
        order = rng.choice(len(chunks), size=len(chunks), replace=True, p=weights)
    else:
        order = rng.permutation(len(chunks))

    for start in range(0, len(order), batch_chunks):
        batch = [chunks[int(idx)] for idx in order[start : start + batch_chunks]]
        if batch:
            yield batch


def compute_validation_metrics(actor, chunks, device: torch.device):
    actor.eval()
    total_steps = 0
    exact_match = 0
    per_head_correct = np.zeros(4, dtype=np.int64)
    total_nll = 0.0

    with torch.no_grad():
        for batch in iterate_batches(chunks, batch_chunks=16, balance_by_command=False, seed=0):
            obs_flat, act_flat, masks_flat, batch_size = flatten_chunk_batch(batch)
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
            act_t = torch.as_tensor(act_flat, dtype=torch.int64, device=device)
            masks_t = torch.as_tensor(masks_flat, dtype=torch.float32, device=device)
            rnn_t = torch.zeros((batch_size, 1, 128), dtype=torch.float32, device=device)

            log_probs, _ = actor.evaluate_actions(obs_t, rnn_t, act_t, masks_t)
            pred_actions, _, _ = actor(obs_t, rnn_t, masks_t, deterministic=True)

            total_nll += float((-log_probs).sum().item())
            total_steps += int(act_t.shape[0])
            matches = pred_actions.eq(act_t)
            per_head_correct += matches.sum(dim=0).detach().cpu().numpy()
            exact_match += int(matches.all(dim=1).sum().item())

    if total_steps == 0:
        return {
            "steps": 0,
            "exact_match_rate": 0.0,
            "per_head_accuracy": [0.0, 0.0, 0.0, 0.0],
            "avg_nll": 0.0,
        }
    return {
        "steps": total_steps,
        "exact_match_rate": float(exact_match) / float(total_steps),
        "per_head_accuracy": [float(x) / float(total_steps) for x in per_head_correct.tolist()],
        "avg_nll": float(total_nll) / float(total_steps),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    payload = np.load(dataset_path)
    dataset_metadata = load_dataset_metadata(dataset_path)
    obs = payload["obs"].astype(np.float32)
    actions = payload["actions"].astype(np.int64)
    cmd_indices = payload["cmd_indices"].astype(np.int64)
    episode_index = payload["episode_index"].astype(np.int32)

    train_episodes, val_episodes = split_episodes(episode_index, args.val_ratio, args.seed)
    train_mask = np.isin(episode_index, list(train_episodes))
    val_mask = np.isin(episode_index, list(val_episodes)) if val_episodes else np.zeros_like(episode_index, dtype=bool)

    train_chunks = build_chunks(obs[train_mask], actions[train_mask], cmd_indices[train_mask], episode_index[train_mask], args.seq_len, args.stride)
    val_chunks = build_chunks(obs[val_mask], actions[val_mask], cmd_indices[val_mask], episode_index[val_mask], args.seq_len, args.stride) if val_episodes else []
    if not train_chunks:
        raise RuntimeError("No train chunks were built. Reduce --seq-len or use a larger dataset.")

    use_mlp_actlayer = False
    warmstart_info = {}
    actor_state = None

    if args.warmstart_ppo_actor:
        actor_path = Path(args.warmstart_ppo_actor)
        actor_state = torch.load(actor_path, map_location="cpu", weights_only=True)
        use_mlp_actlayer = actor_uses_mlp_actlayer(actor_state)
        warmstart_info = {"type": "ppo_actor", "path": str(actor_path)}
    elif args.warmstart_baseline:
        baseline_path = Path(args.warmstart_baseline)
        actor_state = torch.load(baseline_path, map_location="cpu", weights_only=True)
        use_mlp_actlayer = actor_uses_mlp_actlayer(actor_state)
        warmstart_info = {"type": "baseline", "path": str(baseline_path)}

    ppo_args = build_ppo_args(use_mlp_actlayer)
    obs_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(12,), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([41, 41, 41, 30])
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    actor = PPOActor(ppo_args, obs_space, act_space, device=device)
    if actor_state is not None:
        matched = copy_matching_state(actor_state, actor)
        warmstart_info["matched_keys"] = int(matched)

    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_history = []
    best_val = -1.0
    best_actor_path = out_dir / "actor_bc_best.pt"
    latest_actor_path = out_dir / "actor_bc_latest.pt"

    for epoch in range(1, args.epochs + 1):
        actor.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch in iterate_batches(train_chunks, batch_chunks=args.batch_chunks, balance_by_command=args.balance_by_command, seed=args.seed + epoch):
            obs_flat, act_flat, masks_flat, batch_size = flatten_chunk_batch(batch)
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
            act_t = torch.as_tensor(act_flat, dtype=torch.int64, device=device)
            masks_t = torch.as_tensor(masks_flat, dtype=torch.float32, device=device)
            rnn_t = torch.zeros((batch_size, 1, 128), dtype=torch.float32, device=device)

            log_probs, entropy = actor.evaluate_actions(obs_t, rnn_t, act_t, masks_t)
            loss = -log_probs.mean() - args.entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            batch_count += 1

        val_metrics = compute_validation_metrics(actor, val_chunks, device) if val_chunks else {
            "steps": 0,
            "exact_match_rate": 0.0,
            "per_head_accuracy": [0.0, 0.0, 0.0, 0.0],
            "avg_nll": 0.0,
        }
        train_metrics = compute_validation_metrics(actor, train_chunks[: min(len(train_chunks), 256)], device)

        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batch_count, 1),
            "train_probe_exact_match_rate": train_metrics["exact_match_rate"],
            "train_probe_avg_nll": train_metrics["avg_nll"],
            "val_exact_match_rate": val_metrics["exact_match_rate"],
            "val_avg_nll": val_metrics["avg_nll"],
            "val_per_head_accuracy": val_metrics["per_head_accuracy"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        metrics_history.append(record)
        torch.save(actor.state_dict(), latest_actor_path)

        if (val_chunks and val_metrics["exact_match_rate"] >= best_val) or (not val_chunks and epoch == args.epochs):
            best_val = val_metrics["exact_match_rate"]
            torch.save(actor.state_dict(), best_actor_path)

        print(json.dumps(record, ensure_ascii=False))

    if not best_actor_path.exists():
        torch.save(actor.state_dict(), best_actor_path)

    export_path = out_dir / args.export_name
    export_ppo_actor_to_baseline(best_actor_path, export_path, use_mlp_actlayer)
    command_semantics = str(dataset_metadata.get("command_semantics", "legacy_3x5x3") or "legacy_3x5x3")
    metadata = {
        "schema_version": 1,
        "training_mode": "cap_log_behavior_cloning",
        "dataset_path": str(dataset_path),
        "warmstart": warmstart_info,
        "device": str(device),
        "epochs": int(args.epochs),
        "seq_len": int(args.seq_len),
        "stride": int(args.stride),
        "batch_chunks": int(args.batch_chunks),
        "command_semantics": command_semantics,
        "observation_layout": [
            "command_altitude_m_div_1000",
            "command_heading_rad",
            "command_velocity_mps_div_100",
            "altitude_m_div_5000",
            "roll_sin",
            "roll_cos",
            "pitch_sin",
            "pitch_cos",
            "u_mps_div_340",
            "v_mps_div_340",
            "w_mps_div_340",
            "vc_mps_div_340",
        ],
        "action_layout": [
            "aileron_index_0_40",
            "elevator_index_0_40",
            "rudder_index_0_40",
            "throttle_index_0_29",
        ],
        "train_chunk_count": len(train_chunks),
        "val_chunk_count": len(val_chunks),
        "metrics_history": metrics_history,
        "best_actor_path": str(best_actor_path),
        "latest_actor_path": str(latest_actor_path),
        "export_path": str(export_path),
    }
    write_json(out_dir / "bc_metrics.json", metadata)
    write_json(export_path.with_suffix(".json"), metadata)
    write_json(best_actor_path.with_suffix(".json"), metadata)
    write_json(latest_actor_path.with_suffix(".json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
