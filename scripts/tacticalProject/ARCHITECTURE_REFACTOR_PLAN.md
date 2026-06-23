# TacticalProject Refactor Plan (No Logic Change)

## Goals
- Keep tactical behavior and core algorithms unchanged.
- Reduce oversized files and duplicated definitions.
- Make module boundaries explicit: domain types, decision flow, maneuver execution, simulation adapters.

## Current Large-File Baseline
- `cap/cap_task.py`: 4759 lines
- `tactical_task.py`: 3470 lines
- `unified_enemy_tactical_ai.py`: 3347 lines
- `simulation/radar_manager.py`: 2533 lines

## Refactor Rules
- Refactor only structure, naming clarity, and file layout.
- No tactical thresholds/decision math changes in refactor commits.
- Every step must pass `py_compile` on touched files.

## Completed In This Wave
- Split `tactical_task.py` into a compatibility wrapper and `tactical_task_impl.py`.
- Verified the wrapper split with `py_compile` and `run_cap_simulation.py --steps 1`.
- Split three remaining large tactical entry modules into compatibility wrappers:
- `simulation/radar_manager.py` now re-exports from `simulation/radar_manager_impl.py`
- `tactical_executor.py` now re-exports from `tactical_executor_impl.py`
- `unified_enemy_tactical_ai.py` now re-exports from `unified_enemy_tactical_ai_impl.py`
- Verified the wrapper split with `py_compile` and `run_cap_simulation.py --steps 1`.
- Removed duplicated method definitions overridden later in class bodies.
- Extracted shared enemy AI enums/dataclasses to `enemy_ai_types.py`.
- Split `cap/run_cap_simulation.py` orchestration from 417 lines to 151 lines.
- Extracted CAP run-time concerns into dedicated modules:
- `cap/run_logging.py`: root-cause aware logging setup
- `cap/acmi_writer.py`: ACMI/Tacview frame serialization
- `cap/run_helpers.py`: task bootstrap, patrol display, action building, verification reporting
- Split CAPTask low-level implementation out of the main class:
- `cap/cap_task_refactor_helpers.py`: `normalize_action` and `_lowlevel_control`
- `cap/cap_task.py` reduced from 4417 lines to 3984 lines without changing the method interface
- Split `tactical_task.py` through helper delegation:
- `tactical_task_refactor_helpers.py`: phase update, node decision, low-level policy, missile launch flow
- `tactical_task.py` reduced from 3342 lines to 1642 lines
- Split `unified_enemy_tactical_ai.py` through shared types + helper delegation:
- `enemy_ai_types.py`: shared enums/dataclasses
- `enemy_ai_refactor_helpers.py`: situation/threat/action selection/execution helpers
- `unified_enemy_tactical_ai.py` reduced from 3209 lines to 1798 lines
- Verified the refactor with `py_compile` and `run_cap_simulation.py --steps 1`.
- Split radar shared types and helper-only logic:
- `simulation/radar_types.py`: shared radar enums/dataclasses/models
- `simulation/radar_refactor_helpers.py`: radar geometry, signal math, jamming/reporting helpers
- `simulation/radar_manager.py` reduced from 2079 lines to 1711 lines while preserving the public API
- `simulation/__init__.py` now imports shared radar types directly
- Verified the radar refactor with `py_compile`, `run_cap_simulation.py --steps 120`, and `run_cap_simulation.py --steps 4200`.
- Split CAP guidance/relay verification logic:
- `cap/cap_guidance_helpers.py`: stable dual-track window, prelaunch gate, relay guidance helpers
- `cap/cap_task.py` reduced from 3984 lines to 3768 lines with wrapper delegation only
- Rewrote `cap/cap_task_refactor_helpers.py::_lowlevel_control` to remove unreachable duplicated code and normalize rootcause low-level log labels to clean Chinese
- Verified the CAP refactor with `py_compile`, `run_cap_simulation.py --steps 200`, and `run_cap_simulation.py --steps 4200` under `CAP_ROOTCAUSE_TRACE=1`.
- Split CAP situational-awareness orchestration:
- `cap/cap_picture_helpers.py`: AWACS update, fused picture update, CAP context builders
- `cap/cap_task.py` further reduced from 3768 lines to 3477 lines
- Restored `simulation/radar_manager.py` global singleton declaration after wrapper extraction regression
- Re-verified the larger refactor with `py_compile`, `run_cap_simulation.py --steps 200`, and `run_cap_simulation.py --steps 4200` under `CAP_ROOTCAUSE_TRACE=1`.
- Split CAP tactic-selection and situation-log chain:
- `cap/cap_tactic_helpers.py`: node compression logging, pair target stickiness, pair-node status, mission evaluation, tactic selection
- `cap/cap_task.py` further reduced from 3477 lines to 2965 lines
- Re-verified the larger tactic refactor with `py_compile`, `run_cap_simulation.py --steps 200`, and `run_cap_simulation.py --steps 4200` under `CAP_ROOTCAUSE_TRACE=1`.
- Split enemy maneuver execution and command-shaping helpers:
- `enemy_ai_maneuver_helpers.py`: action-parameter generation, core maneuver executors, heading-hold command shaping
- `unified_enemy_tactical_ai.py` further reduced from 1798 lines to 1352 lines
- Re-verified this wave with `py_compile` and `run_cap_simulation.py --steps 5`.
- Split enemy RTB/mission-completion helper chain:
- `enemy_ai_rtb_helpers.py`: defensive split, unified RTB init/execute, altitude-change execution, RTB decision, mission-completion checks
- `unified_enemy_tactical_ai.py` further reduced from 1352 lines to 1155 lines
- Re-verified this wave with `py_compile` and `run_cap_simulation.py --steps 5`.
- Split TacticalTask second-attack / DR-decision cluster:
- `tactical_task_reengage_helpers.py`: formation-reset startup/execution, second-attack tactic selection/readiness, DR reengage/retreat unified decisions, threat counters
- `tactical_task.py` further reduced from 1697 lines to 1388 lines
- Re-verified this wave with `py_compile` and `cap/run_cap_simulation.py --steps 5`.
- Split TacticalTask main decision entry:
- `tactical_task_action_mixin.py`: `get_action` orchestration, sensor-driven tactical interruption, missile-evade priority handling, and route dispatch
- `tactical_task_impl.py` reduced from 1388 lines to 872 lines while keeping the public class name and behavior stable
- Re-verified this wave with `py_compile` and `cap/run_cap_simulation.py --steps 1`.
- Split CAP runtime tail:
- `cap/cap_task_runtime_mixin.py`: step entry plus CAP runtime / verification / guidance / picture-state delegation
- `cap/cap_task_impl.py` reduced from 2196 lines to 2106 lines while keeping the public class name and behavior stable
- Re-verified this wave with `py_compile` and `cap/run_cap_simulation.py --steps 1`.
- Split CAP tracking/intent/missile tail cluster:
- `cap/cap_tracking_helpers.py`: intent recognition, cooperative tracking updates, missile state update, cooperative tracking startup
- `cap/cap_task.py` now delegates these methods for clearer boundary between CAP state machine and tracking/missile maintenance logic
- Re-verified this wave with `py_compile` and `cap/run_cap_simulation.py --steps 5`.
- Split CAP root-cause safety checks:
- `cap/cap_rootcause_logging.py`: incoming-missile detection and fuel-critical RTB checks
- `cap/cap_task.py` now delegates these diagnostics so the main task file stays focused on orchestration
- `cap/cap_task.py` reduced to 2738 lines after the new delegation
- Re-verified this wave with `py_compile` and `run_cap_simulation.py --steps 1`.
- Split CAP lifecycle orchestration:
- `cap/cap_task_init_helpers.py`: CAPTask initialization wiring, models, adapters, and shared state setup
- `cap/cap_task_step_helpers.py`: the full CAP step loop and patrol/intercept coordination
- `cap/cap_task.py` reduced further to 2412 lines and now acts as a thin delegate shell
- Re-verified this wave with `py_compile` and `run_cap_simulation.py --steps 1`.

## Next Waves
1. `tactical_task_impl.py` split:
- `tactical_task_state.py`: state and caches
- `tactical_task_missile.py`: launch policy and logging
- `tactical_task_reengage.py`: second-attack and reset flow

2. `simulation/radar_manager_impl.py` split:
- `radar_model.py`: model and equations
- `radar_tracking.py`: lock/track update logic
- `radar_events.py`: launch/alert event bridge

3. `unified_enemy_tactical_ai_impl.py` split:
- `enemy_ai_analysis.py`: situation analysis + threat assessment
- `enemy_ai_selection.py`: tactical mode + action selection
- `enemy_ai_execution.py`: command execution + maneuver mapping

4. `cap/cap_task.py` split:
- `cap/cap_lowlevel_control.py`: normalize/low-level output
- `cap/cap_patrol_state.py`: patrol machine and phase transitions
- `cap/cap_state_init_helpers.py`: further break init wiring into smaller state-specific builders
- `cap/cap_patrol_step_helpers.py`: split the remaining radar/track bookkeeping out of the step helper

## Acceptance Criteria
- No behavior regressions in CAP and tactical runs.
- Reduced average file size for top 4 modules.
- No duplicated method names in same class (except `@property` getter/setter pairs).
