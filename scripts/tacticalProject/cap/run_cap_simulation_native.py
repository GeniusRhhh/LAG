"""
Run CAP simulation with enemy F16 CAP-native low-level explicitly enabled.

Usage:
    C:\\Users\\ZRF\\.conda\\envs\\lag_gpu\\python.exe run_cap_simulation_native.py
    C:\\Users\\ZRF\\.conda\\envs\\lag_gpu\\python.exe run_cap_simulation_native.py --steps 3000
"""

import importlib.util
import os
import sys
from pathlib import Path


def _load_run_module():
    script_path = Path(__file__).with_name("run_cap_simulation.py")
    module_name = "cap_run_cap_simulation_native_entry"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


os.environ["CAP_ENEMY_F16_NATIVE_ENABLED"] = "1"


if __name__ == "__main__":
    _load_run_module().main()
