"""Logging helpers for the tactical executor."""

import logging


def log_phase_transition(agent_id, old_phase, new_phase, current_time, distance=None, altitude=None):
    """Record a tactical phase transition."""
    if not agent_id.startswith('A'):
        return
    try:
        transition_info = f"🗷 [阶段切换] {agent_id}: {old_phase} -> {new_phase} (时间: {current_time:.1f}s)"
        if distance is not None:
            transition_info += f" | 距离: {distance:.1f}km"
        if altitude is not None:
            transition_info += f" | 高度: {altitude:.0f}m"
        logging.info(transition_info)
    except Exception as e:
        logging.error(f"阶段切换日志记录失败: {e}")


def log_maneuver_execution(agent_id, maneuver_type, current_heading, target_heading, current_time, additional_info=""):
    """Record maneuver execution details."""
    try:
        maneuver_info = (
            f"🛈 [机动执行] {agent_id}: {maneuver_type} | "
            f"{current_heading:.1f}° -> {target_heading:.1f}° (时间: {current_time:.1f}s)"
        )
        if additional_info:
            maneuver_info += f" | {additional_info}"
        logging.info(maneuver_info)
    except Exception as e:
        logging.error(f"机动执行日志记录失败: {e}")


def log_missile_launch_check(agent_id, distance_ok, heading_ok, distance_val, heading_val, current_time):
    """Record missile launch checks."""
    try:
        status = "✓ 满足" if (distance_ok and heading_ok) else "✗ 不满足"
        distance_status = "✓" if distance_ok else "✗"
        heading_status = "✓" if heading_ok else "✗"

        logging.info(f"🎆 [发射条件检查] {agent_id} {status} (时间: {current_time:.1f}s)")
        logging.info(f"    距离: {distance_status} {distance_val:.1f}km | 朝向: {heading_status} {heading_val:.1f}°")
    except Exception as e:
        logging.error(f"导弹发射条件日志记录失败: {e}")


def log_enemy_behavior(agent_id, behavior_type, details, current_time):
    """Record enemy behavior."""
    try:
        logging.info(f"👻 [敌方行为] {agent_id}: {behavior_type} | {details} (时间: {current_time:.1f}s)")
    except Exception as e:
        logging.error(f"敌方行为日志记录失败: {e}")
