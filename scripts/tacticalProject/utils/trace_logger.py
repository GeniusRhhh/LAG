import logging
import os
import threading
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

from cap.run_logging import get_sim_log_prefix


_TRACE_LOGGER: Optional[logging.Logger] = None
_TRACE_LOG_PATH: Optional[str] = None

# 全局会话/任务标识（贯穿全流程）
_SESSION_ID: str = os.getenv("TACTICAL_SESSION_ID", "") or uuid.uuid4().hex
_TASK_ID: str = os.getenv("TACTICAL_TASK_ID", "") or os.getenv("TASK_ID", "") or ""

# 运行期节流/去重缓存（仅内存，不影响控制台）
_TRACE_STATE: Dict[str, Dict[str, Any]] = {
    "last_by_key": {},
    "last_step_by_key": {},
}

# 日志展示风格：
# - compact: 前缀只保留 仿真时间/对象/阶段/战术；尽量单行汇总
# - verbose: 输出完整字段（UTC/SID/PID/TID/MOD/TYPE/STATUS/SIM/DUR...）
_TRACE_STYLE: str = (os.getenv("TACTICAL_TRACE_STYLE", "") or "compact").strip().lower()


def set_trace_style(style: str) -> None:
    """运行期切换日志展示风格。"""
    global _TRACE_STYLE
    s = (style or "").strip().lower()
    _TRACE_STYLE = s if s in ("compact", "verbose") else "compact"


def get_trace_style() -> str:
    return _TRACE_STYLE


_TYPE_ZH = {
    "EVENT": "事件",
    "DECISION": "决策",
    "ACTION": "动作",
    "STATE": "状态",
    "SENSOR": "传感器",
    "GATE": "门限",
    "CALL": "调用",
    "SPAN": "耗时",
    "EXCEPTION": "异常",
    "TACTIC": "战术",
    "INTERRUPT": "打断",
}

_STATUS_ZH = {
    "INPUT": "输入",
    "CHANGED": "变化",
    "PERIODIC": "周期",
    "ENTER": "进入",
    "RETURN": "返回",
    "ALLOW": "允许",
    "DENY": "拒绝",
    "START": "开始",
    "END": "结束",
    "OK": "正常",
    "FAIL": "失败",
    "FALLBACK": "兜底",
    "EXCEPTION": "异常",
    "FAILSAFE": "兜底窗口",
    "FORCE_EVASION": "强制规避",
    "FORCE_RTB": "强制返航",
    "PARAM_TUNE": "参数调整",
    "MISSILE_EVASION": "导弹规避",
}


def _to_zh(v: Any, mapping: Dict[str, str]) -> Any:
    if v is None:
        return None
    s = str(v)
    return mapping.get(s, s)


def _get_env_time(env: Any) -> Tuple[Optional[int], Optional[float]]:
    step = None
    sim_time = None
    try:
        if env is not None:
            step = getattr(env, "current_step", None)
            ti = getattr(env, "time_interval", None)
            if step is not None and ti is not None:
                sim_time = float(step) * float(ti)
    except Exception:
        pass
    return step, sim_time


def get_session_id() -> str:
    return _SESSION_ID


def get_task_id() -> str:
    return _TASK_ID


def set_task_id(task_id: str) -> None:
    global _TASK_ID
    _TASK_ID = str(task_id or "")


def _utc_ts_ms() -> str:
    # UTC，毫秒
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _fmt_val(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        # 尽量保持“打印风格”可读：小数不过度堆叠
        return f"{v:.3f}".rstrip('0').rstrip('.')
    return str(v)


def _fmt_kv(data: Dict[str, Any]) -> str:
    parts = []
    for k, v in data.items():
        if v is None:
            continue
        # 列表/元组：尽量压缩为逗号分隔
        if isinstance(v, (list, tuple)):
            try:
                vv = ",".join(str(x) for x in v)
            except Exception:
                vv = str(v)
            parts.append(f"{k}={vv}")
        else:
            parts.append(f"{k}={_fmt_val(v)}")
    return " ".join(parts)


def _format_sim_time_mmss(sim_t: Optional[float]) -> str:
    try:
        return get_sim_log_prefix(step=0, time_s=float(sim_t or 0.0))
    except Exception:
        return get_sim_log_prefix(step=0, time_s=0.0)


def _format_prefix(
    *,
    ts: Optional[str] = None,
    sim_t: Optional[float] = None,
    step: Optional[int] = None,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    module: Optional[str] = None,
    event_type: Optional[str] = None,
    status: Optional[str] = None,
    duration_ms: Optional[float] = None,
    me: Optional[str] = None,
    enemy: Optional[str] = None,
    phase: Optional[str] = None,
    tactic: Optional[str] = None,
) -> str:
    # compact：只保留仿真时间/对象/阶段/战术（便于阅读）
    # verbose：保留完整字段（用于深度调试）

    if _TRACE_STYLE != "verbose":
        parts = []
        if sim_t is not None:
            parts.append(get_sim_log_prefix(step=step, time_s=float(sim_t)))
        if me:
            parts.append(f"[我机={me}]")
        if enemy:
            parts.append(f"[敌机={enemy}]")
        if phase:
            parts.append(f"[阶段={phase}]")
        if tactic:
            parts.append(f"[战术={tactic}]")
        return "".join(parts) + " "

    sid = session_id or _SESSION_ID
    tid = task_id if task_id is not None else _TASK_ID
    pid = os.getpid()
    th = threading.get_ident()
    parts = [
        f"[UTC={ts or _utc_ts_ms()}]",
        f"[SID={sid}]",
    ]
    if tid:
        parts.append(f"[TASK={tid}]")
    parts.append(f"[PID={pid}]")
    parts.append(f"[TID={th}]")
    parts.append(f"[MOD={module or 'unknown'}]")
    parts.append(f"[TYPE={event_type or 'EVENT'}]")
    if status:
        parts.append(f"[STATUS={status}]")
    if sim_t is not None:
        try:
            parts.append(f"[SIM={float(sim_t):.3f}s]")
        except Exception:
            parts.append(f"[SIM={sim_t}s]")
    if duration_ms is not None:
        try:
            parts.append(f"[DUR={float(duration_ms):.1f}ms]")
        except Exception:
            parts.append(f"[DUR={duration_ms}ms]")

    if me:
        parts.append(f"[ME={me}]")
    if enemy:
        parts.append(f"[EN={enemy}]")
    if phase:
        parts.append(f"[PHASE={phase}]")
    if tactic:
        parts.append(f"[TACTIC={tactic}]")

    return "".join(parts) + " "


def _format_trace_line(payload: Dict[str, Any]) -> str:
    # 单行：适合直接作为“全流程复盘报告”的原始材料
    ts = payload.get("时间戳")
    sim_t = payload.get("仿真时间")
    step = payload.get("步")
    sid = payload.get("会话")
    task_id = payload.get("任务")
    module = payload.get("模块")
    event_type = _to_zh(payload.get("类型"), _TYPE_ZH)
    status = _to_zh(payload.get("状态"), _STATUS_ZH)
    duration_ms = payload.get("耗时_ms")
    me = payload.get("我机")
    enemy = payload.get("敌机")
    phase = payload.get("阶段")
    tactic = payload.get("战术")
    event = payload.get("事件")
    note = payload.get("说明")
    data = payload.get("数据") or {}

    prefix = _format_prefix(
        ts=ts,
        sim_t=sim_t,
        step=step,
        session_id=sid,
        task_id=task_id,
        module=module,
        event_type=event_type,
        status=status,
        duration_ms=duration_ms,
        me=me,
        enemy=enemy,
        phase=phase,
        tactic=tactic,
    )
    # 内容尽量中文、尽量一行
    event_str = f"{event}" if event else ""
    note_str = f" | {note}" if note else ""
    data_str = f" | {_fmt_kv(data)}" if isinstance(data, dict) and data else ""

    # compact：把 类型/状态 放入内容，避免前缀过长但又丢信息
    if _TRACE_STYLE != "verbose":
        meta = []
        if event_type:
            meta.append(f"类型={event_type}")
        if status:
            meta.append(f"状态={status}")
        meta_str = (" | " + " ".join(meta)) if meta else ""
        return f"{prefix}{event_str}{meta_str}{note_str}{data_str}".rstrip()

    event_str2 = f" {event_str}" if event_str else ""
    return f"{prefix}{event_str2}{note_str}{data_str}".rstrip()


def _default_log_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"tactical_trace_{ts}.log")


def get_trace_logger(log_path: Optional[str] = None) -> logging.Logger:
    """获取仅写文件的追踪日志器（不向控制台输出）。"""
    global _TRACE_LOGGER, _TRACE_LOG_PATH

    if _TRACE_LOGGER is not None:
        return _TRACE_LOGGER

    _TRACE_LOG_PATH = log_path or os.getenv("TACTICAL_TRACE_LOG", "") or _default_log_path()
    os.makedirs(os.path.dirname(_TRACE_LOG_PATH), exist_ok=True)

    logger = logging.getLogger("tactical_trace")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(_TRACE_LOG_PATH, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    _TRACE_LOGGER = logger
    return logger


def get_trace_log_path() -> Optional[str]:
    return _TRACE_LOG_PATH


def trace_event(
    *,
    事件: str,
    env: Any = None,
    模块: Optional[str] = None,
    类型: Optional[str] = None,
    状态: Optional[str] = None,
    耗时_ms: Optional[float] = None,
    我机: Optional[str] = None,
    敌机: Optional[str] = None,
    阶段: Optional[str] = None,
    战术: Optional[str] = None,
    说明: Optional[str] = None,
    数据: Optional[Dict[str, Any]] = None,

) -> None:
    """写一条中文事件日志到文件（纯文本单行）。

    设计目标：像控制台 logging.info 那样可读，但只写文件、不污染控制台。
    """
    logger = get_trace_logger()

    step, sim_time = _get_env_time(env)

    payload: Dict[str, Any] = {
        "时间戳": _utc_ts_ms(),
        "仿真时间": sim_time,
        "步": step,
        "会话": _SESSION_ID,
        "任务": _TASK_ID,
        "模块": 模块,
        "类型": 类型,
        "状态": 状态,
        "耗时_ms": 耗时_ms,
        "我机": 我机,
        "敌机": 敌机,
        "阶段": 阶段,
        "战术": 战术,
        "事件": 事件,
        "说明": 说明,
        "数据": 数据 or {},
    }

    try:
        logger.info(_format_trace_line(payload))
    except Exception:
        # 最后兜底：至少写入事件名（不输出现实时间/step）
        try:
            step, sim_time = _get_env_time(env)
            pfx = _format_prefix(
                ts=_utc_ts_ms(),
                sim_t=sim_time,
                session_id=_SESSION_ID,
                task_id=_TASK_ID,
                module=模块,
                event_type=类型,
                status=状态,
                duration_ms=耗时_ms,
                me=我机,
                enemy=敌机,
                phase=阶段,
                tactic=战术,
            )
            logger.info(f"{pfx}{事件}")
        except Exception:
            logger.info(f"事件={事件}")


def trace_block(
    *,
    标题: str,
    env: Any = None,
    模块: Optional[str] = None,
    类型: Optional[str] = None,
    状态: Optional[str] = None,
    耗时_ms: Optional[float] = None,
    我机: Optional[str] = None,
    敌机: Optional[str] = None,
    阶段: Optional[str] = None,
    战术: Optional[str] = None,
    说明: Optional[str] = None,
    要点: Optional[Iterable[str]] = None,
    数据: Optional[Dict[str, Any]] = None,
) -> None:
    """写一段“控制台打印风格”的多行日志块。

    典型用法：在关键节点（态势/威胁评估、模板选择、机动/参数决策、雷达/导弹事件）输出原因链。
    """
    logger = get_trace_logger()
    step, sim_time = _get_env_time(env)
    ts = _utc_ts_ms()
    prefix = _format_prefix(
        ts=ts,
        sim_t=sim_time,
        step=step,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        module=模块,
        event_type=类型,
        status=状态,
        duration_ms=耗时_ms,
        me=我机,
        enemy=敌机,
        phase=阶段,
        tactic=战术,
    )

    header = f"{prefix} {标题}"
    if 说明:
        header += f" | {说明}"

    def _flatten(obj: Any, parent: str = "") -> Iterable[Tuple[str, Any]]:
        """将嵌套dict/list展开为(点路径, 值)序列，便于逐行打印。"""
        if obj is None:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if v is None:
                    continue
                key = f"{parent}.{k}" if parent else str(k)
                if isinstance(v, dict):
                    yield from _flatten(v, key)
                elif isinstance(v, (list, tuple)):
                    # list/tuple：若元素是dict，继续展开；否则保留为紧凑字符串
                    if any(isinstance(x, dict) for x in v):
                        for i, x in enumerate(v):
                            yield from _flatten(x, f"{key}[{i}]")
                    else:
                        yield (key, v)
                else:
                    yield (key, v)
        elif isinstance(obj, (list, tuple)):
            if any(isinstance(x, dict) for x in obj):
                for i, x in enumerate(obj):
                    yield from _flatten(x, f"{parent}[{i}]")
            else:
                yield (parent or "value", obj)
        else:
            yield (parent or "value", obj)

    try:
        # compact：尽量一行汇总（要点/数据扁平化合并）
        if _TRACE_STYLE != "verbose":
            merged: Dict[str, Any] = {}
            if 数据:
                try:
                    for k, v in _flatten(数据):
                        if v is None:
                            continue
                        merged[str(k)] = v
                except Exception:
                    merged.update(数据)
            if 要点:
                try:
                    pts = [str(x).strip() for x in 要点 if x is not None and str(x).strip()]
                    if pts:
                        merged["要点"] = " ; ".join(pts)
                except Exception:
                    pass

            line = header
            if merged:
                line = f"{line} | {_fmt_kv(merged)}"
            logger.info(line)
            return

        # verbose：保留多行材料
        logger.info(header)
        if 要点:
            for line in 要点:
                if line is None:
                    continue
                s = str(line).rstrip()
                if not s:
                    continue
                logger.info(f"{prefix}  - {s}")
        if 数据:
            for k, v in _flatten(数据):
                if v is None:
                    continue
                logger.info(f"{prefix}  - {k}: {_fmt_val(v)}")
    except Exception:
        trace_event(事件=标题, env=env, 模块=模块, 类型=类型, 状态=状态, 耗时_ms=耗时_ms, 我机=我机, 敌机=敌机, 阶段=阶段, 战术=战术, 说明=说明, 数据=数据)


def trace_if_changed(
    *,
    key: str,
    value: Any,
    标题: str,
    env: Any = None,
    模块: Optional[str] = None,
    类型: Optional[str] = None,
    状态: Optional[str] = None,
    耗时_ms: Optional[float] = None,
    我机: Optional[str] = None,
    敌机: Optional[str] = None,
    阶段: Optional[str] = None,
    战术: Optional[str] = None,
    说明: Optional[str] = None,
    要点: Optional[Iterable[str]] = None,
    数据: Optional[Dict[str, Any]] = None,
) -> None:
    """仅在 value 变化时写入一段日志块（避免每步刷屏）。"""
    last = _TRACE_STATE["last_by_key"].get(key, object())
    if last == value:
        return
    _TRACE_STATE["last_by_key"][key] = value
    trace_block(
        标题=标题,
        env=env,
        模块=模块,
        类型=类型,
        状态=状态,
        耗时_ms=耗时_ms,
        我机=我机,
        敌机=敌机,
        阶段=阶段,
        战术=战术,
        说明=说明,
        要点=要点,
        数据=数据,
    )


@contextmanager
def trace_span(
    *,
    标题: str,
    env: Any = None,
    模块: Optional[str] = None,
    类型: Optional[str] = None,
    我机: Optional[str] = None,
    敌机: Optional[str] = None,
    阶段: Optional[str] = None,
    战术: Optional[str] = None,
    说明: Optional[str] = None,
    数据: Optional[Dict[str, Any]] = None,
    log_start: bool = False,
    slow_ms: Optional[float] = None,
):
    """记录一个可测量耗时的事件区间（用于节点切换/雷达检查/算法推理等）。

    为避免刷屏：默认不打印 START；END 仅在耗时超过 slow_ms 或 verbose 模式时打印。
    异常一定打印。
    """
    t0 = datetime.now(timezone.utc)
    if log_start or _TRACE_STYLE == "verbose":
        trace_event(
            事件=f"{标题}:开始",
            env=env,
            模块=模块,
            类型=类型 or "SPAN",
            状态="START",
            我机=我机,
            敌机=敌机,
            阶段=阶段,
            战术=战术,
            说明=说明,
            数据=数据,
        )
    ok = False
    try:
        yield
        ok = True
    except Exception as e:
        dt = (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0
        trace_exception(
            事件=f"{标题}:异常",
            env=env,
            模块=模块,
            类型=类型 or "SPAN",
            我机=我机,
            敌机=敌机,
            阶段=阶段,
            战术=战术,
            异常=e,
            耗时_ms=dt,
        )
        raise
    finally:
        dt = (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0
        threshold = slow_ms
        if threshold is None:
            threshold = 50.0 if _TRACE_STYLE != "verbose" else 0.0

        if _TRACE_STYLE == "verbose" or dt >= float(threshold):
            trace_event(
                事件=f"{标题}:结束",
                env=env,
                模块=模块,
                类型=类型 or "SPAN",
                状态="OK" if ok else "FAIL",
                耗时_ms=dt,
                我机=我机,
                敌机=敌机,
                阶段=阶段,
                战术=战术,
            )


def trace_exception(
    *,
    事件: str,
    env: Any = None,
    模块: Optional[str] = None,
    类型: Optional[str] = None,
    我机: Optional[str] = None,
    敌机: Optional[str] = None,
    阶段: Optional[str] = None,
    战术: Optional[str] = None,
    异常: Optional[BaseException] = None,
    耗时_ms: Optional[float] = None,
    说明: Optional[str] = None,
):
    tb = None
    try:
        tb = traceback.format_exc()
    except Exception:
        tb = None
    data = {
        "exception_type": type(异常).__name__ if 异常 is not None else None,
        "exception": str(异常) if 异常 is not None else None,
        "traceback": tb,
    }
    trace_block(
        标题=事件,
        env=env,
        模块=模块,
        类型=类型 or "EXCEPTION",
        状态="EXCEPTION",
        耗时_ms=耗时_ms,
        我机=我机,
        敌机=敌机,
        阶段=阶段,
        战术=战术,
        说明=说明,
        数据=data,
    )


def trace_throttle(
    *,
    key: str,
    min_steps: int,
    标题: str,
    env: Any = None,
    模块: Optional[str] = None,
    类型: Optional[str] = None,
    状态: Optional[str] = None,
    耗时_ms: Optional[float] = None,
    我机: Optional[str] = None,
    敌机: Optional[str] = None,
    阶段: Optional[str] = None,
    战术: Optional[str] = None,
    说明: Optional[str] = None,
    要点: Optional[Iterable[str]] = None,
    数据: Optional[Dict[str, Any]] = None,
) -> None:
    """按步数节流写入日志块（例如每30步/60步一次摘要）。"""
    step, _ = _get_env_time(env)
    if step is None:
        # 无step时直接写
        trace_block(标题=标题, env=env, 模块=模块, 类型=类型, 状态=状态, 耗时_ms=耗时_ms, 我机=我机, 敌机=敌机, 阶段=阶段, 战术=战术, 说明=说明, 要点=要点, 数据=数据)
        return

    last_step = _TRACE_STATE["last_step_by_key"].get(key)
    # 处理“仿真重置/新一局”：step 计数器回到0，但节流缓存仍保留旧 last_step。
    # 若不处理，会导致新一局在 step 超过旧 last_step 前都不再输出该 key 的周期日志。
    try:
        if last_step is not None and int(step) < int(last_step):
            last_step = None
    except Exception:
        pass

    if last_step is not None and (int(step) - int(last_step)) < int(min_steps):
        return
    _TRACE_STATE["last_step_by_key"][key] = int(step)
    trace_block(标题=标题, env=env, 模块=模块, 类型=类型, 状态=状态, 耗时_ms=耗时_ms, 我机=我机, 敌机=敌机, 阶段=阶段, 战术=战术, 说明=说明, 要点=要点, 数据=数据)
