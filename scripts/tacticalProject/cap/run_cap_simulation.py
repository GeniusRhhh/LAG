"""
CAP巡逻仿真运行脚本。

用法:
    C:\\Users\\ZRF\\.conda\\envs\\lag_gpu\\python.exe run_cap_simulation.py
    C:\\Users\\ZRF\\.conda\\envs\\lag_gpu\\python.exe run_cap_simulation.py --steps 3000
"""

import argparse

# 飞机模型配置
MY_AIRCRAFT_TYPE = 'su27sk'
ENEMY_AIRCRAFT_TYPE = 'f16'

try:
    from .runtime.bootstrap import bootstrap_runtime
except ImportError:
    from runtime.bootstrap import bootstrap_runtime


_RUNTIME_CONTEXT = bootstrap_runtime(
    entry_file=__file__,
    my_aircraft_type=MY_AIRCRAFT_TYPE,
    enemy_aircraft_type=ENEMY_AIRCRAFT_TYPE,
)

try:
    from .runtime.engine import run_cap_simulation_core
except ImportError:
    from runtime.engine import run_cap_simulation_core


def run_cap_simulation(max_steps: int = 6000, output_dir: str = None, allow_patrol_fallback: bool = False):
    """运行 CAP 巡逻仿真。"""
    run_cap_simulation_core(
        context=_RUNTIME_CONTEXT,
        max_steps=max_steps,
        output_dir=output_dir,
        allow_patrol_fallback=allow_patrol_fallback,
        my_aircraft_type=MY_AIRCRAFT_TYPE,
        enemy_aircraft_type=ENEMY_AIRCRAFT_TYPE,
    )


def main():
    parser = argparse.ArgumentParser(description='CAP巡逻仿真')
    parser.add_argument('--steps', type=int, default=6000, help='最大步数(默认6000=20分钟)')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument(
        '--allow-patrol-fallback',
        action='store_true',
        help='当CAPTask初始化失败时，允许回退到SimplePatrolTask（默认关闭）',
    )
    args = parser.parse_args()
    run_cap_simulation(
        max_steps=args.steps,
        output_dir=args.output,
        allow_patrol_fallback=args.allow_patrol_fallback,
    )


if __name__ == "__main__":
    main()
