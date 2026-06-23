"""
敌方AI行为选择过程的详细调试日志增强
用于追踪为什么选择某个行为以及最终导致下降
"""

import logging
import os

DESCENT_DEBUG_LOG = os.path.join(
    os.path.dirname(__file__),
    'cap_results',
    'descent_trace_detailed.log'
)

# 创建输出目录
os.makedirs(os.path.dirname(DESCENT_DEBUG_LOG), exist_ok=True)

# 配置详细日志
descent_logger = logging.getLogger('DescentTrace')
descent_logger.setLevel(logging.DEBUG)

handler = logging.FileHandler(DESCENT_DEBUG_LOG, mode='w', encoding='utf-8')
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
handler.setFormatter(formatter)
descent_logger.addHandler(handler)


def log_action_selection_trace(
    agent_id: str,
    current_time: float,
    current_altitude: float,
    tactical_phase: str,
    tactical_mode: str,
    available_actions: dict,
    selected_action: str,
    selection_scores: dict = None
):
    """记录动作选择的完整过程"""
    
    descent_logger.info(f"")
    descent_logger.info(f"{'='*100}")
    descent_logger.info(
        f"[ACTION_SELECTION] T={current_time:.1f}s | {agent_id} @{current_altitude:.0f}m | "
        f"Phase={tactical_phase} | Mode={tactical_mode}"
    )
    descent_logger.info(f"{'='*100}")
    
    # 记录可用动作和权重
    if available_actions:
        descent_logger.info(f"【可用动作权重】")
        sorted_actions = sorted(available_actions.items(), key=lambda x: x[1], reverse=True)
        for action, weight in sorted_actions[:5]:
            descent_logger.info(f"  {action:40s} weight={weight:8.4f}")
        
        if len(sorted_actions) > 5:
            descent_logger.info(f"  ... 还有 {len(sorted_actions)-5} 个动作")
    
    # 记录选择的动作
    descent_logger.warning(f"【选定动作】 {selected_action}")
    
    if selection_scores:
        descent_logger.info(f"【选择评分明细】")
        for key, value in selection_scores.items():
            descent_logger.info(f"  {key}: {value}")


def log_tactical_decision_process(
    agent_id: str,
    current_time: float,
    current_altitude: float,
    threat_level: str,
    threat_distance: float,
    threat_assessment_details: dict = None
):
    """记录战术决策过程"""
    
    descent_logger.info(f"")
    descent_logger.info(f"[TACTICAL_DECISION] {agent_id} | Threat={threat_level} @{threat_distance:.0f}m")
    
    if threat_assessment_details:
        descent_logger.info(f"【威胁评估细节】")
        for key, value in threat_assessment_details.items():
            descent_logger.info(f"  {key:40s}: {value}")


def log_altitude_command_chain(
    agent_id: str,
    current_time: float,
    current_altitude: float,
    action_type: str,
    step_description: str,
    cmd_index: int = None,
    delta_altitude: float = None,
    reason: str = ""
):
    """记录高度命令的转换链"""
    
    # 颜色标记：如果是下降则标记为WARNING
    if delta_altitude is not None and delta_altitude < -100:
        descent_logger.warning(
            f"[ALTITUDE_CMD] {agent_id} @{current_altitude:.0f}m | {action_type:30s} | "
            f"{step_description:40s} | delta={delta_altitude:8.0f}m | {reason}"
        )
    else:
        descent_logger.info(
            f"[ALTITUDE_CMD] {agent_id} @{current_altitude:.0f}m | {action_type:30s} | "
            f"{step_description:40s} | delta={delta_altitude:8.0f}m if delta_altitude is not None else 'N/A' | {reason}"
        )


def log_function_execution_trace(
    agent_id: str,
    function_name: str,
    altitude_before: float,
    altitude_after: float,
    parameters: dict = None,
    reason_for_call: str = ""
):
    """记录函数执行及其高度影响"""
    
    alt_change = altitude_after - altitude_before
    
    if alt_change < -100:
        descent_logger.warning(
            f"[FUNC_EXEC] {function_name:40s} | {agent_id} | "
            f"alt: {altitude_before:.0f}m -> {altitude_after:.0f}m (Δ={alt_change:8.0f}m) | {reason_for_call}"
        )
        if parameters:
            descent_logger.warning(f"  Parameters: {parameters}")
    else:
        descent_logger.info(
            f"[FUNC_EXEC] {function_name:40s} | {agent_id} | "
            f"alt: {altitude_before:.0f}m -> {altitude_after:.0f}m (Δ={alt_change:8.0f}m)"
        )


def log_state_machine_transition(
    agent_id: str,
    old_phase: str,
    new_phase: str,
    current_altitude: float,
    trigger_reason: str
):
    """记录状态机转换"""
    
    descent_logger.info(
        f"[STATE_TRANSITION] {agent_id} @{current_altitude:.0f}m | "
        f"{old_phase:20s} -> {new_phase:20s} | trigger={trigger_reason}"
    )


def log_safety_override(
    agent_id: str,
    current_altitude: float,
    original_action: str,
    override_action: str,
    reason: str
):
    """记录安全保护的动作覆盖"""
    
    descent_logger.warning(
        f"[SAFETY_OVERRIDE] {agent_id} @{current_altitude:.0f}m | "
        f"原始={original_action:40s} -> 覆盖={override_action:40s} | {reason}"
    )


def log_command_normalization(
    agent_id: str,
    altitude_idx_raw: int,
    altitude_idx_normalized: int,
    altitude_delta_raw: float,
    altitude_delta_normalized: float,
    heading_idx: int,
    velocity_idx: int
):
    """记录命令标准化过程"""
    
    if altitude_delta_raw != altitude_delta_normalized:
        descent_logger.warning(
            f"[CMD_NORM] {agent_id} | Alt clipped: "
            f"idx {altitude_idx_raw}->{altitude_idx_normalized} | "
            f"delta {altitude_delta_raw:.0f}->{altitude_delta_normalized:.0f}m"
        )
    else:
        descent_logger.debug(
            f"[CMD_NORM] {agent_id} | Alt: idx={altitude_idx_normalized} delta={altitude_delta_normalized:.0f}m | "
            f"Hdg: idx={heading_idx} | Vel: idx={velocity_idx}"
        )


def print_descent_trace_header(simulation_id: str):
    """打印日志头"""
    
    descent_logger.info("")
    descent_logger.info("="*100)
    descent_logger.info(f"【敌方下降异常追踪 - 仿真ID:{simulation_id}】")
    descent_logger.info(f"追踪对象: B0100, B0200, B0300, B0400")
    descent_logger.info(f"记录内容: 完整的决策链、动作选择、命令转换、高度变化")
    descent_logger.info("="*100)
    descent_logger.info("")


def print_descent_trace_footer():
    """打印日志尾*/
    
    descent_logger.info("")
    descent_logger.info("="*100)
    descent_logger.info("【追踪结束】")
    descent_logger.info("="*100)
