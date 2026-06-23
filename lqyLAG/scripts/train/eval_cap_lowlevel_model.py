#!/usr/bin/env python
import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.envs import SingleControlEnv
from envs.JSBSim.model.baseline_actor import BaselineActor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CAP-native low-level BaselineActor.")
    parser.add_argument("--scenario-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--hardcase-json", type=str, default="")
    parser.add_argument("--hardcase-prob", type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def actor_uses_mlp_actlayer(state_dict) -> bool:
    return any(key.startswith("act.mlp.") for key in state_dict.keys())


def load_hardcase_pool(path: str):
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return [dict(case) for case in payload.get("cases", [payload])]
    if isinstance(payload, list):
        return [dict(case) for case in payload]
    raise ValueError(f"Unsupported hardcase payload type: {type(payload)!r}")


class BaselineController:
    def __init__(self, model_path: str):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        actor = BaselineActor(input_dim=12, use_mlp_actlayer=actor_uses_mlp_actlayer(state_dict))
        actor.load_state_dict(state_dict)
        actor.eval()
        self.actor = actor
        self.reset()

    def reset(self):
        self.rnn_state = np.zeros((1, 1, 128), dtype=np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        action, next_state = self.actor(obs[np.newaxis, :], self.rnn_state)
        self.rnn_state = next_state.detach().cpu().numpy()
        return action.detach().cpu().numpy().reshape(-1).astype(np.int64)


def configure_logging(output_path: Path):
    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(output_path.with_suffix(".eval.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler], force=True)
    logging.info(f"[log] {output_path.with_suffix('.eval.log')}")


def main():
    args = parse_args()
    output_path = Path(args.output) if args.output else Path(args.model_path).with_name("cap_lowlevel_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(output_path)
    set_seed(args.seed)

    env = SingleControlEnv(args.scenario_name)
    env.seed(args.seed)
    hardcase_pool = load_hardcase_pool(args.hardcase_json)
    if hardcase_pool and hasattr(env.task, "set_external_init_pool"):
        env.task.set_external_init_pool(hardcase_pool, args.hardcase_prob)
        logging.info(f"[hardcase] loaded={len(hardcase_pool)} prob={args.hardcase_prob:.2f}")

    controller = BaselineController(args.model_path)
    agent_id = env.ego_ids[0]
    episodes = []

    for episode_id in range(args.episodes):
        obs = env.reset()[0]
        controller.reset()
        profile = getattr(env.task, "get_profile_name", lambda _aid: "unknown")(agent_id)

        min_alt_m = float("inf")
        min_vc_mps = float("inf")
        max_aoa_deg = 0.0
        mean_abs_delta_heading = []
        mean_abs_delta_altitude = []
        mean_abs_delta_speed = []
        total_reward = 0.0

        while True:
            action = controller.act(obs.astype(np.float32))
            obs, rewards, done, info = env.step(np.expand_dims(action, axis=0))
            obs = obs[0]
            total_reward += float(rewards[0][0])

            aircraft = env.agents[agent_id]
            min_alt_m = min(min_alt_m, float(aircraft.get_property_value(c.position_h_sl_m)))
            min_vc_mps = min(min_vc_mps, float(aircraft.get_property_value(c.velocities_vc_mps)))
            max_aoa_deg = max(max_aoa_deg, abs(float(aircraft.get_property_value(c.aero_alpha_deg))))
            mean_abs_delta_heading.append(abs(float(aircraft.get_property_value(c.delta_heading))))
            mean_abs_delta_altitude.append(abs(float(aircraft.get_property_value(c.delta_altitude))))
            mean_abs_delta_speed.append(abs(float(aircraft.get_property_value(c.delta_velocities_u))))

            if bool(done[0][0]):
                break

        termination_reason = str(info.get(f"{agent_id}_termination_reason", "")).strip().lower()
        survived = bool(info.get(f"{agent_id}_termination_success", False) or termination_reason == "timeout" or env.current_step >= env.max_steps)
        episodes.append(
            {
                "episode": episode_id,
                "profile": profile,
                "survived_to_timeout": survived,
                "termination_reason": termination_reason,
                "steps": int(env.current_step),
                "total_reward": total_reward,
                "min_alt_m": min_alt_m,
                "min_vc_mps": min_vc_mps,
                "max_aoa_deg": max_aoa_deg,
                "mean_abs_delta_heading_deg": float(np.mean(mean_abs_delta_heading)) if mean_abs_delta_heading else 0.0,
                "mean_abs_delta_altitude_m": float(np.mean(mean_abs_delta_altitude)) if mean_abs_delta_altitude else 0.0,
                "mean_abs_delta_speed_mps": float(np.mean(mean_abs_delta_speed)) if mean_abs_delta_speed else 0.0,
            }
        )

        if (episode_id + 1) % 20 == 0:
            survive_rate = np.mean([ep["survived_to_timeout"] for ep in episodes])
            logging.info(f"[eval] episode={episode_id + 1}/{args.episodes} survive_rate={survive_rate:.2%}")

    env.close()

    profile_counts = Counter(ep["profile"] for ep in episodes)
    summary = {
        "episodes": len(episodes),
        "survive_rate": float(np.mean([ep["survived_to_timeout"] for ep in episodes])),
        "crash_rate": float(np.mean([not ep["survived_to_timeout"] for ep in episodes])),
        "avg_reward": float(np.mean([ep["total_reward"] for ep in episodes])),
        "avg_steps": float(np.mean([ep["steps"] for ep in episodes])),
        "avg_min_alt_m": float(np.mean([ep["min_alt_m"] for ep in episodes])),
        "avg_min_vc_mps": float(np.mean([ep["min_vc_mps"] for ep in episodes])),
        "avg_max_aoa_deg": float(np.mean([ep["max_aoa_deg"] for ep in episodes])),
        "avg_abs_delta_heading_deg": float(np.mean([ep["mean_abs_delta_heading_deg"] for ep in episodes])),
        "avg_abs_delta_altitude_m": float(np.mean([ep["mean_abs_delta_altitude_m"] for ep in episodes])),
        "avg_abs_delta_speed_mps": float(np.mean([ep["mean_abs_delta_speed_mps"] for ep in episodes])),
        "profile_counts": dict(profile_counts),
        "termination_reason_counts": dict(Counter(ep["termination_reason"] for ep in episodes)),
        "hardcase_pool_size": int(len(hardcase_pool)),
        "hardcase_prob": float(args.hardcase_prob if hardcase_pool else 0.0),
        "profile_summary": {},
    }

    for profile_name in profile_counts.keys():
        group = [ep for ep in episodes if ep["profile"] == profile_name]
        summary["profile_summary"][profile_name] = {
            "episodes": len(group),
            "survive_rate": float(np.mean([ep["survived_to_timeout"] for ep in group])),
            "avg_reward": float(np.mean([ep["total_reward"] for ep in group])),
            "avg_min_alt_m": float(np.mean([ep["min_alt_m"] for ep in group])),
            "avg_min_vc_mps": float(np.mean([ep["min_vc_mps"] for ep in group])),
            "avg_max_aoa_deg": float(np.mean([ep["max_aoa_deg"] for ep in group])),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": episodes}, f, indent=2)

    logging.info(f"[summary] {summary}")
    logging.info(f"[write] {output_path}")


if __name__ == "__main__":
    main()
