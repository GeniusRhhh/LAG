"""Compatibility wrapper for the refactored CAP task."""

from .cap_task_impl import CAPTask, _CAP_CONTROL_DEBUG, _CAP_DEBUG_PRINT, _CAP_VERIFY_TABLE

__all__ = ["CAPTask", "_CAP_CONTROL_DEBUG", "_CAP_DEBUG_PRINT", "_CAP_VERIFY_TABLE"]
