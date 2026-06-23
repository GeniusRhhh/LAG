from __future__ import annotations

import inspect
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

try:
    from .run_logging import get_sim_log_prefix
except ImportError:
    from cap.run_logging import get_sim_log_prefix


@dataclass(frozen=True)
class PatrolDisplayContext:
    heading_deg: float
    patrol_mode: str
    hot_name: str
    cold_name: str


def _normalize_tactic_name(name: object) -> str:
    text = str(name or "").strip()
    return "" if text.lower() in {"", "unknown", "none", "未决策"} else text


def _normalize_phase_name(name: object) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    if "." in text:
        text = text.split(".")[-1]
    return text


def prepare_patrol_config(current_dir: str, project_root: str, config_name: str = 'patrol_config') -> str:
    local_config = os.path.join(current_dir, 'config', 'patrol_config.yaml')
    if not os.path.exists(local_config):
        raise FileNotFoundError(local_config)

    jsbsim_config_dir = os.path.join(project_root, 'envs', 'JSBSim', 'configs')
    os.makedirs(jsbsim_config_dir, exist_ok=True)
    shutil.copy2(local_config, os.path.join(jsbsim_config_dir, f'{config_name}.yaml'))
    return config_name


def create_patrol_task(env, allow_patrol_fallback: bool = False):
    try:
        from cap.cap_task import CAPTask

        patrol_task = CAPTask(env.config)
        logging.info("[CAPTask] 使用CAPTask（P1-4完整巡逻：含FAOR/靶眼/编队/状态机）")
        try:
            captask_file = inspect.getfile(patrol_task.__class__)
        except Exception:
            captask_file = 'n/a'
        logging.info(f"[CAPTask来源] module={patrol_task.__class__.__module__} file={captask_file}")
        logging.info("[COOP_VERIFY] ENABLED source=CAPTask metrics=gate/relay/midcourse_peak")
    except Exception as exc:
        if not allow_patrol_fallback:
            logging.error(
                "[FATAL] CAPTask初始化失败，已禁止自动回退到SimplePatrolTask。"
                "如需临时回退请加 --allow-patrol-fallback"
            )
            raise RuntimeError(f"CAPTask initialization failed: {exc}") from exc

        logging.warning(f"[WARNING] CAPTask不可用，回退SimplePatrolTask: {exc}")
        from cap.patrol_task import SimplePatrolTask

        patrol_task = SimplePatrolTask(env.config)
        logging.warning("[COOP_VERIFY] DISABLED source=SimplePatrolTask reason=no_guidance_snapshot")

    print(f"[信息] 任务类型: {type(patrol_task)}")
    print(f"[信息] 任务模块: {patrol_task.__class__.__module__}")
    return patrol_task


def detect_patrol_display(env) -> PatrolDisplayContext:
    a0100_heading = 0.0
    if hasattr(env.config, 'aircraft_configs') and 'A0100' in env.config.aircraft_configs:
        init_state = env.config.aircraft_configs['A0100'].get('init_state', {})
        a0100_heading = init_state.get('ic_psi_true_deg', 0.0)

    if abs(a0100_heading) < 45 or abs(a0100_heading - 180) < 45:
        patrol_mode = "南北"
        hot_name, cold_name = "北(热段)", "南(冷段)"
    elif abs(a0100_heading - 90) < 45 or abs(a0100_heading - 270) < 45:
        patrol_mode = "东西"
        hot_name, cold_name = "东(热段)", "西(冷段)"
    else:
        patrol_mode = "未知"
        hot_name, cold_name = "热段", "冷段"

    return PatrolDisplayContext(
        heading_deg=a0100_heading,
        patrol_mode=patrol_mode,
        hot_name=hot_name,
        cold_name=cold_name,
    )


def build_step_actions(env, patrol_task) -> np.ndarray:
    agent_ids = list(getattr(env, 'ego_ids', [])) + list(getattr(env, 'enm_ids', []))
    if not agent_ids:
        agent_ids = list(env.agents.keys())

    act_dim = 4
    try:
        if hasattr(env.task, 'action_space') and hasattr(env.task.action_space, 'nvec'):
            act_dim = int(len(env.task.action_space.nvec))
    except Exception:
        act_dim = 4

    actions = np.zeros((1, len(agent_ids), act_dim), dtype=np.float32)
    for index, agent_id in enumerate(agent_ids):
        try:
            alt_cmd, hdg_cmd, vel_cmd = patrol_task.get_action(env, agent_id)
        except Exception as exc:
            step = int(getattr(env, 'current_step', 0))
            time_s = float(step) * float(getattr(env, 'time_interval', 0.2))
            logging.exception(
                "%s [GET_ACTION_EXCEPTION] agent=%s task=%s fallback=(7,8,3) error=%s",
                get_sim_log_prefix(step=step, time_s=time_s),
                agent_id,
                patrol_task.__class__.__name__,
                exc,
            )
            alt_cmd, hdg_cmd, vel_cmd = (7, 8, 3)

        actions[0, index, 0] = float(alt_cmd)
        if act_dim > 1:
            actions[0, index, 1] = float(hdg_cmd)
        if act_dim > 2:
            actions[0, index, 2] = float(vel_cmd)
    return actions


def maybe_log_guidance_snapshot(patrol_task, step: int, dt: float) -> None:
    if step % int(max(1, round(60.0 / dt))) != 0:
        return
    if not hasattr(patrol_task, 'get_guidance_verification_snapshot'):
        return

    try:
        snapshot = patrol_task.get_guidance_verification_snapshot() or {}
        gate_pass_raw = int(snapshot.get('prelaunch_gate_pass_count', 0))
        gate_block_raw = int(snapshot.get('prelaunch_gate_block_count', 0))
        gate_pass_unique = int(snapshot.get('prelaunch_gate_unique_pass_count', gate_pass_raw))
        gate_block_unique = int(snapshot.get('prelaunch_gate_unique_block_count', gate_block_raw))
        logging.info(
            "[协同验收快照] "
            f"gate_pass_raw={gate_pass_raw} "
            f"gate_block_raw={gate_block_raw} "
            f"gate_pass_unique={gate_pass_unique} "
            f"gate_block_unique={gate_block_unique} "
            f"relay_attempt={int(snapshot.get('relay_attempt_count', 0))} "
            f"relay_success={int(snapshot.get('relay_success_count', 0))} "
            f"active_guided_peak={int(snapshot.get('active_guided_missile_peak', 0))}"
        )
        logging.info(
            "[COOP_VERIFY] SNAPSHOT "
            f"pass_raw={gate_pass_raw} "
            f"block_raw={gate_block_raw} "
            f"pass_unique={gate_pass_unique} "
            f"block_unique={gate_block_unique} "
            f"pass_rate_raw={float(snapshot.get('prelaunch_gate_raw_pass_rate', 0.0)):.2%} "
            f"pass_rate_unique={float(snapshot.get('prelaunch_gate_unique_pass_rate', 0.0)):.2%} "
            f"relay_attempt={int(snapshot.get('relay_attempt_count', 0))} "
            f"relay_success={int(snapshot.get('relay_success_count', 0))} "
            f"relay_attempt_unique={int(snapshot.get('relay_unique_attempt_count', 0))} "
            f"relay_success_unique={int(snapshot.get('relay_unique_success_count', 0))} "
            f"relay_rate_raw={float(snapshot.get('relay_raw_success_rate', 0.0)):.2%} "
            f"relay_rate_unique={float(snapshot.get('relay_unique_success_rate', 0.0)):.2%} "
            f"midcourse_peak={int(snapshot.get('active_guided_missile_peak', 0))}"
        )
    except Exception as exc:
        logging.warning(f"[协同验收快照] 读取失败: {exc}")


def log_heading_changes(
    env,
    patrol_task,
    display: PatrolDisplayContext,
    last_states: Dict[str, Dict[str, float]],
    time_s: float,
) -> None:
    for agent_id in ['A0100', 'A0200', 'A0300', 'A0400']:
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            continue

        target_heading = None
        if hasattr(patrol_task, 'patrol_machines') and agent_id in getattr(patrol_task, 'patrol_machines', {}):
            patrol_machine = patrol_task.patrol_machines[agent_id]
            try:
                target_heading = float(patrol_machine.STATE_HEADINGS.get(patrol_machine.state, 0))
            except Exception:
                target_heading = None

        if target_heading is None and hasattr(patrol_task, 'aircraft_states') and agent_id in getattr(
            patrol_task, 'aircraft_states', {}
        ):
            state = patrol_task.aircraft_states[agent_id]
            target_heading = float(state.get('target_heading', 0))

        if target_heading is None:
            continue

        last_heading = last_states.get(agent_id, {}).get('target_heading', target_heading)
        if abs(((target_heading - last_heading) + 180) % 360 - 180) > 1e-6:
            if display.patrol_mode == "南北":
                if abs((target_heading - 0) % 360) < 1e-6:
                    heading_name = display.hot_name
                elif abs((target_heading - 180) % 360) < 1e-6:
                    heading_name = display.cold_name
                elif abs((target_heading - 270) % 360) < 1e-6:
                    heading_name = "西(过渡)"
                elif abs((target_heading - 90) % 360) < 1e-6:
                    heading_name = "东(过渡)"
                else:
                    heading_name = f"航向{target_heading:.0f}°"
            else:
                heading_name = f"航向{target_heading:.0f}°"
            logging.info(f"[{time_s:.0f}s] 🔧 {agent_id} 转向 -> {heading_name}")

        last_states[agent_id] = {'target_heading': target_heading}


def log_structured_battle_events(
    env,
    patrol_task,
    last_states: Dict[str, object],
    time_s: float,
) -> None:
    cap_state = str(getattr(getattr(patrol_task, 'cap_state_machine', None), 'state', '') or '')
    detect_mode = ''
    coop_detection = getattr(patrol_task, 'coop_detection', None)
    if coop_detection is not None:
        detect_mode = str(getattr(getattr(coop_detection, 'mode', None), 'value', '') or '')

    phase_items = []
    bridge = getattr(patrol_task, 'tactical_bridge', None)
    assignments = getattr(patrol_task, '_tactic_assignments_by_agent', {}) or {}
    for lead_id, label in (('A0100', 'LEFT'), ('A0300', 'RIGHT')):
        tactic_name = ''
        phase_name = ''
        if bridge and getattr(bridge, 'initialized', False):
            try:
                info = bridge.get_agent_tactic_info(lead_id) or {}
            except Exception:
                info = {}
            tactic_name = _normalize_tactic_name(info.get('tactic'))
            phase_name = _normalize_phase_name(info.get('phase'))
        assign = assignments.get(lead_id)
        target_id = str(getattr(assign, 'target_id', '') or '')
        if assign is not None and getattr(assign, 'tactic', None) is not None and not tactic_name:
            tactic_name = _normalize_tactic_name(getattr(assign.tactic, 'value', assign.tactic))
        phase_items.append(f"{label}:{tactic_name or '-'}|{phase_name or '-'}|{target_id or '-'}")

    event_key = (
        cap_state,
        detect_mode,
        tuple(phase_items),
    )
    if last_states.get('battle_event_key') != event_key:
        logging.info(
            "[BATTLE_EVENT] t=%.1fs cap_state=%s detect_mode=%s %s",
            float(time_s),
            cap_state or '-',
            detect_mode or '-',
            " ".join(phase_items),
        )
        last_states['battle_event_key'] = event_key

    alive_key = tuple(
        f"{aid}:{'ALIVE' if getattr(agent, 'is_alive', False) else 'DEAD'}"
        for aid, agent in sorted(getattr(env, 'agents', {}).items(), key=lambda item: item[0])
        if str(aid).startswith(('A', 'B'))
    )
    if last_states.get('alive_event_key') != alive_key:
        logging.info("[ALIVE_EVENT] t=%.1fs %s", float(time_s), " ".join(alive_key))
        last_states['alive_event_key'] = alive_key

def log_final_document_summary(patrol_task) -> None:
    try:
        def _fmt_time(v):
            if v is None:
                return "-"
            try:
                return f"{float(v):.0f}s"
            except Exception:
                return str(v)

        awacs_times = getattr(patrol_task, '_target_first_awacs_time', {}) or {}
        fcr_times = getattr(patrol_task, '_target_first_fcr_time', {}) or {}
        ready_times = getattr(patrol_task, '_target_first_ready_time', {}) or {}
        zone_history = getattr(patrol_task, '_zone_transition_history', {}) or {}
        target_ids = sorted(set(awacs_times.keys()) | set(fcr_times.keys()) | set(ready_times.keys()) | set(zone_history.keys()))
        if target_ids:
            logging.info("[TARGET_TIMELINE]")
            for tid in target_ids:
                zones = zone_history.get(tid, []) or []
                zone_text = " -> ".join(f"{zone}@{float(ts):.0f}s" for ts, zone in zones) if zones else "-"
                logging.info(
                    "  %s awacs=%s fcr=%s ready=%s zones=%s",
                    tid,
                    _fmt_time(awacs_times.get(tid)),
                    _fmt_time(fcr_times.get(tid)),
                    _fmt_time(ready_times.get(tid)),
                    zone_text,
                )
                logging.info(
                    "[TARGET_EVENT] target=%s awacs=%s fcr=%s ready=%s zones=%s",
                    tid,
                    _fmt_time(awacs_times.get(tid)),
                    _fmt_time(fcr_times.get(tid)),
                    _fmt_time(ready_times.get(tid)),
                    zone_text,
                )

        missile_adapter = getattr(patrol_task, 'missile_adapter', None)
        if missile_adapter is not None and hasattr(missile_adapter, 'get_summary_snapshot'):
            missile_summary = missile_adapter.get_summary_snapshot() or {}
            raw_state_counts = missile_summary.get('state_counts', {}) or {}
            launch_events = list(getattr(missile_adapter, '_launch_events', []) or [])
            outcome_events = list(getattr(missile_adapter, '_outcome_events', []) or [])

            def _build_side_state_counts(side_prefix: str):
                counts = {
                    "ready": 0,
                    "launched": 0,
                    "guiding": 0,
                    "terminal": 0,
                    "hit": 0,
                    "miss": 0,
                }
                try:
                    missile_statuses = getattr(missile_adapter, '_missiles', {}) or {}
                    for status in missile_statuses.values():
                        guide_id = str(getattr(status, 'guide_agent_id', '') or '')
                        missile_id = str(getattr(status, 'missile_id', '') or '')
                        if not (guide_id.startswith(side_prefix) or missile_id.startswith(side_prefix)):
                            continue
                        state_name = str(getattr(getattr(status, 'state', None), 'value', '') or '').lower()
                        if state_name in counts:
                            counts[state_name] = int(counts.get(state_name, 0)) + 1
                    return counts
                except Exception:
                    return dict(raw_state_counts)

            def _emit_side_missile_report(side_prefix: str) -> None:
                by_shooter = {
                    shooter: bucket
                    for shooter, bucket in (missile_summary.get('by_shooter', {}) or {}).items()
                    if str(shooter).startswith(side_prefix)
                }
                launch_count = sum(int(bucket.get('launches', 0)) for bucket in by_shooter.values())
                state_counts = _build_side_state_counts(side_prefix)

                logging.info(f"[MISSILE_SUMMARY_{side_prefix}]")
                logging.info(
                    "  total_launch=%d hit=%d miss=%d guiding=%d terminal=%d",
                    int(launch_count),
                    int(state_counts.get('hit', 0)),
                    int(state_counts.get('miss', 0)),
                    int(state_counts.get('guiding', 0)),
                    int(state_counts.get('terminal', 0)),
                )
                for shooter in sorted(by_shooter.keys()):
                    bucket = by_shooter[shooter]
                    logging.info(
                        "  %s launches=%d avg_dist=%.1fkm targets=%s",
                        shooter,
                        int(bucket.get('launches', 0)),
                        float(bucket.get('avg_distance_km', 0.0)),
                        bucket.get('targets', {}),
                    )

                logging.info(f"[MISSILE_LAUNCH_RECORD_{side_prefix}]")
                side_events = [event for event in launch_events if str(event.get('shooter', '')).startswith(side_prefix)]
                if not side_events:
                    logging.info("  -")
                else:
                    for event in side_events:
                        logging.info(
                            "  t=%.0fs shooter=%s target=%s missile=%s model=%s guide=%s dist=%.1fkm left=%d",
                            float(event.get('time', 0.0)),
                            str(event.get('shooter', '-') or '-'),
                            str(event.get('target', '-') or '-'),
                            str(event.get('missile_id', '-') or '-'),
                            str(event.get('missile_model', '-') or '-'),
                            str(event.get('guidance_mode', '-') or '-'),
                            float(event.get('distance_km', 0.0)),
                            int(event.get('missiles_left', 0)),
                        )
                logging.info(f"[MISSILE_OUTCOME_RECORD_{side_prefix}]")
                side_outcomes = [event for event in outcome_events if str(event.get('missile_id', '')).startswith(side_prefix)]
                if not side_outcomes:
                    logging.info("  -")
                    return
                for event in side_outcomes:
                    logging.info(
                        "  t=%.0fs missile=%s state=%s target=%s shooter=%s tof=%.1fs dist=%.1fkm reason=%s",
                        float(event.get('time', 0.0)),
                        str(event.get('missile_id', '-') or '-'),
                        str(event.get('state', '-') or '-'),
                        str(event.get('target_id', '-') or '-'),
                        str(event.get('shooter_id', '-') or '-'),
                        float(event.get('time_of_flight', 0.0) or 0.0),
                        float(event.get('distance_to_target', 0.0) or 0.0),
                        str(event.get('reason', '-') or '-'),
                    )

            _emit_side_missile_report('A')
            _emit_side_missile_report('B')
    except Exception as exc:
        logging.warning(f"[FINAL_DOC_SUMMARY] 杈撳嚭澶辫触: {exc}")


def log_final_guidance_summary(patrol_task) -> None:
    if not hasattr(patrol_task, 'get_guidance_verification_snapshot'):
        return

    try:
        snapshot = patrol_task.get_guidance_verification_snapshot() or {}
        gate_pass_raw = int(snapshot.get('prelaunch_gate_pass_count', 0))
        gate_block_raw = int(snapshot.get('prelaunch_gate_block_count', 0))
        gate_pass_unique = int(snapshot.get('prelaunch_gate_unique_pass_count', gate_pass_raw))
        gate_block_unique = int(snapshot.get('prelaunch_gate_unique_block_count', gate_block_raw))
        logging.info("[协同跟踪/接力制导验收]")
        logging.info(
            f"  发射前门禁: raw={gate_pass_raw}/{gate_block_raw} "
            f"unique={gate_pass_unique}/{gate_block_unique} "
            f"pass_rate_raw={float(snapshot.get('prelaunch_gate_raw_pass_rate', 0.0)):.2%} "
            f"pass_rate_unique={float(snapshot.get('prelaunch_gate_unique_pass_rate', 0.0)):.2%}"
        )
        logging.info(
            f"  接力制导: attempts={int(snapshot.get('relay_attempt_count', 0))} "
            f"success={int(snapshot.get('relay_success_count', 0))} "
            f"attempts_unique={int(snapshot.get('relay_unique_attempt_count', 0))} "
            f"success_unique={int(snapshot.get('relay_unique_success_count', 0))} "
            f"success_rate_raw={float(snapshot.get('relay_raw_success_rate', 0.0)):.2%} "
            f"success_rate_unique={float(snapshot.get('relay_unique_success_rate', 0.0)):.2%} "
            f"by_reason={snapshot.get('relay_success_by_reason', {})}"
        )
        logging.info(f"  中制导活跃峰值导弹数: {int(snapshot.get('active_guided_missile_peak', 0))}")
        logging.info(
            "[COOP_VERIFY] SUMMARY "
            f"pass_raw={gate_pass_raw} "
            f"block_raw={gate_block_raw} "
            f"pass_unique={gate_pass_unique} "
            f"block_unique={gate_block_unique} "
            f"pass_rate_raw={float(snapshot.get('prelaunch_gate_raw_pass_rate', 0.0)):.2%} "
            f"pass_rate_unique={float(snapshot.get('prelaunch_gate_unique_pass_rate', 0.0)):.2%} "
            f"relay_attempt={int(snapshot.get('relay_attempt_count', 0))} "
            f"relay_success={int(snapshot.get('relay_success_count', 0))} "
            f"relay_attempt_unique={int(snapshot.get('relay_unique_attempt_count', 0))} "
            f"relay_success_unique={int(snapshot.get('relay_unique_success_count', 0))} "
            f"relay_rate_raw={float(snapshot.get('relay_raw_success_rate', 0.0)):.2%} "
            f"relay_rate_unique={float(snapshot.get('relay_unique_success_rate', 0.0)):.2%} "
            f"midcourse_peak={int(snapshot.get('active_guided_missile_peak', 0))}"
        )
    except Exception as exc:
        logging.warning(f"[协同跟踪/接力制导验收] 输出失败: {exc}")
