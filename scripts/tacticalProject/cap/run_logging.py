from __future__ import annotations

import logging
import os
import threading
from datetime import datetime


_SIM_CLOCK_LOCK = threading.Lock()
_SIM_CLOCK = {
    "step": 0,
    "time_s": 0.0,
}


def set_sim_log_clock(step: int | None, time_s: float | None) -> None:
    with _SIM_CLOCK_LOCK:
        _SIM_CLOCK["step"] = int(step or 0)
        _SIM_CLOCK["time_s"] = float(time_s or 0.0)


def get_sim_log_clock() -> tuple[int, float]:
    with _SIM_CLOCK_LOCK:
        return int(_SIM_CLOCK["step"]), float(_SIM_CLOCK["time_s"])


def get_sim_log_prefix(step: int | None = None, time_s: float | None = None) -> str:
    if step is None or time_s is None:
        step, time_s = get_sim_log_clock()
    return f"[秒数：{float(time_s):.1f} 仿真步数：{int(step)}]"


def _strip_legacy_sim_prefix(message: str) -> str:
    text = str(message or "")
    if text.startswith("[秒数："):
        return text
    legacy_prefixes = (
        "【仿真时间：",
        "銆愪豢鐪熸椂闂达細",
    )
    for prefix in legacy_prefixes:
        if text.startswith(prefix):
            end = text.find("】")
            if end != -1:
                return text[end + 1 :].lstrip()
    return text


class _SimTimestampPrefixFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            message = record.getMessage()
        except Exception:
            return True
        normalized = _strip_legacy_sim_prefix(message)
        if normalized.startswith("[秒数："):
            record.msg = normalized
            record.args = ()
            return True
        step, time_s = get_sim_log_clock()
        record.msg = f"{get_sim_log_prefix(step, time_s)} {normalized}"
        record.args = ()
        return True


class _SanitizeLogTextFilter(logging.Filter):
    _REPLACEMENTS = (
        ("\u2713 ", ""),
        ("\u2708\ufe0f  ", ""),
        ("\U0001f4e1 ", ""),
        ("\U0001f50d ", ""),
        ("\U0001f3af ", ""),
        ("\u26a0\ufe0f ", ""),
        ("\u26a0 ", ""),
    )

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            message = record.getMessage()
        except Exception:
            return True
        sanitized = message
        for old, new in self._REPLACEMENTS:
            sanitized = sanitized.replace(old, new)
        if sanitized != message:
            record.msg = sanitized
            record.args = ()
        return True


class _NoRootCauseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return '根因链-' not in msg and '友机根因记录' not in msg


class _RootCauseOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            msg = record.getMessage()
        except Exception:
            return False
        return '根因链-' in msg or '友机根因记录' in msg


def setup_logging(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"cap_{timestamp}.log")
    rootcause_file = os.path.join(output_dir, f"cap_{timestamp}_rootcause.log")

    set_sim_log_clock(0, 0.0)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8-sig")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(_SimTimestampPrefixFilter())
    file_handler.addFilter(_SanitizeLogTextFilter())
    file_handler.addFilter(_NoRootCauseFilter())

    rootcause_handler = logging.FileHandler(rootcause_file, encoding="utf-8-sig")
    rootcause_handler.setLevel(logging.INFO)
    rootcause_handler.setFormatter(fmt)
    rootcause_handler.addFilter(_SimTimestampPrefixFilter())
    rootcause_handler.addFilter(_SanitizeLogTextFilter())
    rootcause_handler.addFilter(_RootCauseOnlyFilter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    console_handler.addFilter(_SimTimestampPrefixFilter())
    console_handler.addFilter(_SanitizeLogTextFilter())
    console_handler.addFilter(_NoRootCauseFilter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, rootcause_handler, console_handler],
        force=True,
    )
    logging.info("根因链日志已单独输出: %s", rootcause_file)
    return log_file
