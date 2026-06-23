#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TACTICAL_ROOT = REPO_ROOT / "scripts" / "tacticalProject"


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    while path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


for _path in (REPO_ROOT, TACTICAL_ROOT):
    _prepend_sys_path(_path)

from lqyLAG.algorithms.ppo.ppo_policy import PPOPolicy
from lqyLAG.algorithms.ppo.ppo_trainer import PPOTrainer
from lqyLAG.algorithms.utils.buffer import ReplayBuffer
from lqyLAG.config import get_config
from envs.JSBSim.model.baseline_actor import BaselineActor

try:
    from .cap_enemy_lowlevel_train_env import CapEnemyLowlevelTrainEnv
except ImportError:
    from cap.cap_enemy_lowlevel_train_env import CapEnemyLowlevelTrainEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Online PPO fine-tuning for an enemy F16 low-level CAP actor.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-agent-id", type=str, default="B0100")
    parser.add_argument("--max-steps", type=int, default=4200)
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epoch", type=int, default=8)
    parser.add_argument("--num-mini-batch", type=int, default=4)
    parser.add_argument("--data-chunk-length", type=int, default=32)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--value-loss-coef", type=float, default=1.0)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--hidden-size", type=str, default="128 128")
    parser.add_argument("--act-hidden-size", type=str, default="128 128")
    parser.add_argument("--warmstart-ppo-actor", type=str, default="")
    parser.add_argument("--warmstart-baseline", type=str, default="")
    parser.add_argument("--enable-safe-teacher-for-other-enemies", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--eval-enable-safe-teacher", action="store_true")
    return parser.parse_args()


def build_ppo_args(args):
    parser = get_config()
    ppo_args = parser.parse_args([])
    effective_chunk_length = max(1, min(int(args.data_chunk_length), int(args.buffer_size)))
    data_chunks = max(1, int(args.buffer_size) // effective_chunk_length)
    effective_num_mini_batch = max(1, min(int(args.num_mini_batch), data_chunks))
    ppo_args.hidden_size = args.hidden_size
    ppo_args.act_hidden_size = args.act_hidden_size
    ppo_args.activation_id = 1
    ppo_args.use_feature_normalization = False
    ppo_args.use_recurrent_policy = True
    ppo_args.recurrent_hidden_size = 128
    ppo_args.recurrent_hidden_layers = 1
    ppo_args.gain = 0.01
    ppo_args.use_prior = False
    ppo_args.lr = args.lr
    ppo_args.gamma = args.gamma
    ppo_args.gae_lambda = args.gae_lambda
    ppo_args.buffer_size = args.buffer_size
    ppo_args.n_rollout_threads = 1
    ppo_args.use_proper_time_limits = False
    ppo_args.use_gae = True
    ppo_args.ppo_epoch = args.ppo_epoch
    ppo_args.clip_param = args.clip_param
    ppo_args.use_clipped_value_loss = False
    ppo_args.num_mini_batch = args.num_mini_batch
    ppo_args.value_loss_coef = args.value_loss_coef
    ppo_args.entropy_coef = args.entropy_coef
    ppo_args.use_max_grad_norm = True
    ppo_args.max_grad_norm = 2.0
    ppo_args.data_chunk_length = effective_chunk_length
    ppo_args.num_mini_batch = effective_num_mini_batch
    return ppo_args


def actor_uses_mlp_actlayer(state_dict) -> bool:
    return any(key.startswith("act.mlp.") or key.startswith("mlp.") for key in state_dict.keys())


def copy_matching_state(src_state, dst_module) -> int:
    dst_state = dst_module.state_dict()
    matched = 0
    for key, value in dst_state.items():
        if key in src_state and src_state[key].shape == value.shape:
            dst_state[key] = src_state[key]
            matched += 1
    dst_module.load_state_dict(dst_state, strict=False)
    return matched


def export_ppo_actor_to_baseline(ppo_actor_path: Path, export_path: Path):
    ppo_state = torch.load(ppo_actor_path, map_location="cpu", weights_only=True)
    actor = BaselineActor(input_dim=12, use_mlp_actlayer=actor_uses_mlp_actlayer(ppo_state))
    copy_matching_state(ppo_state, actor)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), export_path)


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    env = CapEnemyLowlevelTrainEnv(
        train_agent_id=args.train_agent_id,
        max_steps=args.max_steps,
        enable_safe_teacher_for_other_enemies=args.enable_safe_teacher_for_other_enemies,
    )

    ppo_args = build_ppo_args(args)
    obs_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(12,), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([41, 41, 41, 30])
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    policy = PPOPolicy(ppo_args, obs_space, act_space, device=device)
    trainer = PPOTrainer(ppo_args, device=device)
    buffer = ReplayBuffer(ppo_args, num_agents=1, obs_space=obs_space, act_space=act_space)

    warmstart = {}
    if args.warmstart_ppo_actor:
        actor_state = torch.load(args.warmstart_ppo_actor, map_location="cpu", weights_only=True)
        warmstart = {
            "type": "ppo_actor",
            "path": str(args.warmstart_ppo_actor),
            "matched_keys": int(copy_matching_state(actor_state, policy.actor)),
        }
    elif args.warmstart_baseline:
        baseline_state = torch.load(args.warmstart_baseline, map_location="cpu", weights_only=True)
        warmstart = {
            "type": "baseline",
            "path": str(args.warmstart_baseline),
            "matched_keys": int(copy_matching_state(baseline_state, policy.actor)),
        }

    metrics_history = []
    global_step = 0
    best_reward = float("-inf")
    latest_actor_path = output_root / "actor_online_latest.pt"
    best_actor_path = output_root / "actor_online_best.pt"

    obs, info = env.reset(seed=args.seed)
    command_semantics = str(getattr(env.task, "enemy_lowlevel_semantics", "legacy_3x5x3") or "legacy_3x5x3")
    buffer.step = 0
    buffer.obs[0, 0, 0] = obs.copy()
    rnn_states_actor = np.zeros((1, 1, ppo_args.recurrent_hidden_layers, ppo_args.recurrent_hidden_size), dtype=np.float32)
    rnn_states_critic = np.zeros_like(rnn_states_actor)
    masks = np.ones((1, 1, 1), dtype=np.float32)
    episode_reward = 0.0
    episode_count = 0

    try:
        for update in range(1, args.updates + 1):
            buffer.clear()
            buffer.obs[0, 0, 0] = obs.copy()
            buffer.rnn_states_actor[0] = rnn_states_actor.copy()
            buffer.rnn_states_critic[0] = rnn_states_critic.copy()
            buffer.masks[0] = masks.copy()

            update_reward_sum = 0.0
            teacher_event_sum = 0
            done_count = 0

            for step in range(args.buffer_size):
                policy.prep_rollout()
                values, actions, action_log_probs, next_rnn_states_actor, next_rnn_states_critic = policy.get_actions(
                    obs[np.newaxis, :],
                    rnn_states_actor.reshape(1, ppo_args.recurrent_hidden_layers, ppo_args.recurrent_hidden_size),
                    rnn_states_critic.reshape(1, ppo_args.recurrent_hidden_layers, ppo_args.recurrent_hidden_size),
                    masks.reshape(1, 1),
                )

                action_np = actions.detach().cpu().numpy().reshape(4).astype(np.int64)
                value_np = values.detach().cpu().numpy().reshape(1, 1, 1)
                log_prob_np = action_log_probs.detach().cpu().numpy().reshape(1, 1, 1)
                next_rnn_actor_np = next_rnn_states_actor.detach().cpu().numpy().reshape(
                    1, 1, ppo_args.recurrent_hidden_layers, ppo_args.recurrent_hidden_size
                )
                next_rnn_critic_np = next_rnn_states_critic.detach().cpu().numpy().reshape(
                    1, 1, ppo_args.recurrent_hidden_layers, ppo_args.recurrent_hidden_size
                )

                next_obs, reward, terminated, truncated, step_info = env.step(action_np)
                done = bool(terminated or truncated)

                episode_reward += float(reward)
                update_reward_sum += float(reward)
                teacher_event_sum += int(step_info.get("teacher_events_delta", 0))
                global_step += 1

                if done:
                    done_count += 1
                    episode_count += 1
                    reset_obs, _ = env.reset(seed=args.seed + update + step + 1)
                    store_obs = reset_obs
                    masks = np.zeros((1, 1, 1), dtype=np.float32)
                    next_rnn_actor_np[:] = 0.0
                    next_rnn_critic_np[:] = 0.0
                    episode_reward = 0.0
                else:
                    store_obs = next_obs
                    masks = np.ones((1, 1, 1), dtype=np.float32)

                buffer.insert(
                    obs=store_obs.reshape(1, 1, -1),
                    actions=action_np.reshape(1, 1, -1).astype(np.float32),
                    rewards=np.asarray([[[reward]]], dtype=np.float32),
                    masks=masks.copy(),
                    action_log_probs=log_prob_np.astype(np.float32),
                    value_preds=value_np.astype(np.float32),
                    rnn_states_actor=next_rnn_actor_np.astype(np.float32),
                    rnn_states_critic=next_rnn_critic_np.astype(np.float32),
                )

                obs = store_obs
                rnn_states_actor = next_rnn_actor_np
                rnn_states_critic = next_rnn_critic_np

            policy.prep_rollout()
            next_value = policy.get_values(
                obs[np.newaxis, :],
                rnn_states_critic.reshape(1, ppo_args.recurrent_hidden_layers, ppo_args.recurrent_hidden_size),
                masks.reshape(1, 1),
            )
            next_value = next_value.detach().cpu().numpy().reshape(1, 1, 1)
            buffer.compute_returns(next_value)

            policy.prep_training()
            train_info = trainer.train(policy, buffer)
            buffer.after_update()

            mean_update_reward = float(update_reward_sum) / float(args.buffer_size)
            record = {
                "update": int(update),
                "global_step": int(global_step),
                "mean_step_reward": mean_update_reward,
                "teacher_events": int(teacher_event_sum),
                "episodes_completed": int(done_count),
                "train_info": {k: float(v) for k, v in train_info.items()},
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            metrics_history.append(record)
            print(json.dumps(record, ensure_ascii=False))

            torch.save(policy.actor.state_dict(), latest_actor_path)
            if mean_update_reward >= best_reward:
                best_reward = mean_update_reward
                torch.save(policy.actor.state_dict(), best_actor_path)

    finally:
        env.close()

    if not best_actor_path.exists():
        torch.save(policy.actor.state_dict(), best_actor_path)

    export_path = output_root / "f16_enemy_cap_online.pt"
    export_ppo_actor_to_baseline(best_actor_path, export_path)
    metadata = {
        "schema_version": 1,
        "training_mode": "cap_real_online_ppo",
        "train_agent_id": args.train_agent_id,
        "device": str(device),
        "max_steps": int(args.max_steps),
        "buffer_size": int(args.buffer_size),
        "updates": int(args.updates),
        "warmstart": warmstart,
        "metrics_history": metrics_history,
        "latest_actor_path": str(latest_actor_path),
        "best_actor_path": str(best_actor_path),
        "export_path": str(export_path),
        "teacher_enabled_for_other_enemies": bool(args.enable_safe_teacher_for_other_enemies),
        "command_semantics": command_semantics,
    }
    write_json(output_root / "online_training_metrics.json", metadata)
    write_json(export_path.with_suffix(".json"), metadata)
    write_json(best_actor_path.with_suffix(".json"), metadata)
    write_json(latest_actor_path.with_suffix(".json"), metadata)

    if args.eval_steps > 0:
        from subprocess import run

        eval_root = output_root / "eval"
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("eval_enemy_f16_cap_longrun.py")),
            "--ppo-actor-path",
            str(best_actor_path),
            "--output-root",
            str(eval_root),
            "--steps",
            str(args.eval_steps),
            "--command-semantics",
            command_semantics,
        ]
        if args.eval_enable_safe_teacher:
            cmd.append("--enable-safe-teacher")
        run(cmd, cwd=str(REPO_ROOT), check=True)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
