from __future__ import annotations

import logging
import os
from datetime import datetime


class _SanitizeLogTextFilter(logging.Filter):
    _REPLACEMENTS = (
        ("\\u2713 ", ""),
        ("\\u2708\\ufe0f  ", ""),
        ("\\U0001f4e1 ", ""),
        ("\\U0001f50d ", ""),
        ("\\U0001f3af ", ""),
        ("\\u26a0\\ufe0f ", ""),
        ("\\u26a0 ", ""),
        ("✓ ", ""),
        ("✈️  ", ""),
        ("📡 ", ""),
        ("🔍 ", ""),
        ("🎯 ", ""),
        ("🔥 ", ""),
        ("🚀 ", ""),
        ("🛡️ ", ""),
        ("🏠 ", ""),
        ("🔄 ", ""),
        ("⚠️ ", ""),
        ("⚠ ", ""),
        ("❌ ", ""),
        ("✅ ", ""),
        ("⛔ ", ""),
        ("🔗 ", ""),
        ("🔓 ", ""),
        ("📍 ", ""),
        ("🧭 ", ""),
        ("🔁 ", ""),
        ("馃摗 ", ""),
        ("馃攳 ", ""),
        ("馃摫 ", ""),
        ("馃幆 ", ""),
        ("馃敟 ", ""),
        ("馃殌 ", ""),
        ("棣冩畬 ", ""),
        ("棣冩懌 ", ""),
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
        return "根因链" not in msg and "友机根因记录" not in msg


def setup_logging(output_dir: str) -> str:
    """Configure console/file logging for CAP runs."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"cap_{timestamp}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8-sig")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(_SanitizeLogTextFilter())
    file_handler.addFilter(_NoRootCauseFilter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    console_handler.addFilter(_SanitizeLogTextFilter())
    console_handler.addFilter(_NoRootCauseFilter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True,
    )
    return log_file
