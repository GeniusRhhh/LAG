from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv

try:
    from ..acmi_writer import AcmiRecorder
    from ..run_helpers import (
        build_step_actions,
        create_patrol_task,
        detect_patrol_display,
        log_final_document_summary,
        log_final_guidance_summary,
        log_heading_changes,
        log_structured_battle_events,
        maybe_log_guidance_snapshot,
        prepare_patrol_config,
    )
    from ..run_logging import set_sim_log_clock, setup_logging
except ImportError:
    from acmi_writer import AcmiRecorder
    from run_helpers import (
        build_step_actions,
        create_patrol_task,
        detect_patrol_display,
        log_final_document_summary,
        log_final_guidance_summary,
        log_heading_changes,
        log_structured_battle_events,
        maybe_log_guidance_snapshot,
        prepare_patrol_config,
    )
    from run_logging import set_sim_log_clock, setup_logging

try:
    from .bootstrap import RuntimeContext
except ImportError:
    from bootstrap import RuntimeContext


class CAPEnv(MultipleCombatEnv):
    """CAP simulation environment with explicit aircraft model mapping."""

    def __init__(self, config_name: str, my_aircraft_type: str, enemy_aircraft_type: str):
        self._my_aircraft_type = my_aircraft_type
        self._enemy_aircraft_type = enemy_aircraft_type
        super().__init__(config_name)

    def load_simulator(self) -> None:
        for uid, conf in self.config.aircraft_configs.items():
            if uid.startswith("A"):
                conf["model"] = self._my_aircraft_type
            elif uid.startswith("B"):
                conf["model"] = self._enemy_aircraft_type
        super().load_simulator()
        for uid, sim in self._jsbsims.items():
            sim._display_name = uid


def run_cap_simulation_core(
    context: RuntimeContext,
    max_steps: int,
    output_dir: Optional[str],
    allow_patrol_fallback: bool,
    my_aircraft_type: str,
    enemy_aircraft_type: str,
) -> None:
    if output_dir is None:
        output_dir = os.path.join(context.tactical_dir, "cap_results")

    log_file = setup_logging(output_dir)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    acmi_recorder = AcmiRecorder()

    logging.info("=" * 50)
    logging.info("CAP巡逻仿真启动")
    logging.info("=" * 50)

    env = None
    try:
        try:
            config_name = prepare_patrol_config(context.current_dir, context.project_root)
        except FileNotFoundError as exc:
            logging.error(f"配置文件不存在: {exc}")
            return

        env = CAPEnv(config_name, my_aircraft_type=my_aircraft_type, enemy_aircraft_type=enemy_aircraft_type)
        env.max_steps = max_steps

        patrol_task = create_patrol_task(env, allow_patrol_fallback=allow_patrol_fallback)
        env.task = patrol_task

        env.reset()
        env.max_steps = max_steps
        set_sim_log_clock(int(getattr(env, "current_step", 0)), 0.0)

        acmi_path = os.path.join(output_dir, f"{timestamp}_cap_4v4.txt.acmi")
        acmi_recorder.reset()
        acmi_recorder.write_header(acmi_path)

        patrol_display = detect_patrol_display(env)

        logging.info(f"我方: {my_aircraft_type.upper()} x4 (A0100-A0400)")
        logging.info(f"敌方: {enemy_aircraft_type.upper()} x4 (B0100-B0400)")
        logging.info(f"最大步数: {max_steps} ({max_steps * env.time_interval:.0f}秒)")
        logging.info(f"ACMI: {acmi_path}")
        logging.info("-" * 50)
        logging.info(
            f"A0100初始航向: {patrol_display.heading_deg}° | 巡逻模式: {patrol_display.patrol_mode}"
        )

        step = 0
        dt = env.time_interval
        last_states = {}

        while step < max_steps:
            step += 1
            time_s = step * dt
            set_sim_log_clock(step, time_s)

            actions = build_step_actions(env, patrol_task)
            env.step(actions)
            set_sim_log_clock(int(getattr(env, "current_step", step)), time_s)

            maybe_log_guidance_snapshot(patrol_task, step, dt)
            acmi_recorder.write_frame(acmi_path, env, time_s)
            log_heading_changes(env, patrol_task, patrol_display, last_states, time_s)
            log_structured_battle_events(env, patrol_task, last_states, time_s)

            if all(not env.agents[agent_id].is_alive for agent_id in env.agents):
                break

        logging.info("=" * 50)
        logging.info("仿真结束")
        logging.info(f"总时间: {step * dt:.0f}秒 ({step}步)")

        alive_a = sum(1 for aid in env.agents if aid.startswith("A") and env.agents[aid].is_alive)
        alive_b = sum(1 for aid in env.agents if aid.startswith("B") and env.agents[aid].is_alive)
        logging.info(f"我方存活: {alive_a}/4, 敌方存活: {alive_b}/4")

        log_final_guidance_summary(patrol_task)
        log_final_document_summary(patrol_task)

        logging.info(f"ACMI文件: {acmi_path}")
        logging.info(f"日志文件: {log_file}")
        logging.info("=" * 50)

    except Exception as exc:
        logging.error(f"仿真错误: {exc}", exc_info=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                logging.warning(f"环境关闭失败: {close_exc}")
