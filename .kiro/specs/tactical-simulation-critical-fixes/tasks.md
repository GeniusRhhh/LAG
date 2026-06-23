# Implementation Plan: Tactical Simulation Critical Fixes

## Overview

This implementation plan addresses three critical issues in priority order:
1. Issue 1 (Radar Type Detection) - Quick fix
2. Issue 2 (Output File Management) - Quick fix  
3. Issue 3 (Missile Evasion Crashes) - Deep analysis and fix

The approach is to fix simple issues first, then conduct thorough investigation and fix for the complex crash issue.

## Tasks

- [ ] 1. Fix radar type detection bug (Issue 1)
  - [ ] 1.1 Create helper method for radar type determination
    - Add `_get_radar_type()` method to `UnifiedDataRecorder` class
    - Method should check `env.task.friend_lowlevel_type` for friendly aircraft
    - Method should check `env.task.enemy_lowlevel_type` for enemy aircraft
    - Return "N001VE" for SU27, "APG-68" for F16
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ]* 1.2 Write property test for radar type correctness
    - **Property 1: Friendly Aircraft Radar Type Correctness**
    - **Validates: Requirements 1.1**

  - [ ]* 1.3 Write property test for enemy radar type correctness
    - **Property 2: Enemy Aircraft Radar Type Correctness**
    - **Validates: Requirements 1.2**

  - [ ] 1.4 Update trajectory recording to use new helper method
    - Replace radar type detection logic in `record_aircraft_trajectory()` method
    - Update both friendly aircraft section (lines ~209-217)
    - Update enemy aircraft section (lines ~130-139)
    - Remove all hardcoded "APG-68" defaults
    - _Requirements: 1.1, 1.2, 1.5_

  - [ ]* 1.5 Write unit tests for radar type edge cases
    - Test missing `friend_lowlevel_type` attribute
    - Test missing `enemy_lowlevel_type` attribute
    - Test invalid aircraft types
    - Test exception handling
    - _Requirements: 1.3, 1.4_

  - [ ] 1.6 Verify fix with batch simulation
    - Run `batch_run_simulation.py` with SU27 vs F16 configuration
    - Check generated CSV files for correct radar types
    - Verify A-prefix aircraft show "N001VE"
    - Verify B-prefix aircraft show "APG-68"
    - _Requirements: 1.1, 1.2, 1.5_

- [ ] 2. Remove unnecessary output files (Issue 2)
  - [ ] 2.1 Modify save_csv_files() method
    - Remove radar data saving logic
    - Remove missile data saving logic
    - Remove missile analysis generation
    - Remove missile summary generation
    - Keep only trajectory data saving
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ]* 2.2 Write integration test for file generation
    - Run simulation and check output directory
    - Verify ACMI files exist
    - Verify trajectory CSV files exist
    - Verify radar status CSV files do NOT exist
    - Verify missile trajectory CSV files do NOT exist
    - Verify missile analysis CSV files do NOT exist
    - Verify missile summary TXT files do NOT exist
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ] 2.3 Update batch_run_simulation.py cleanup logic
    - Remove file deletion logic (no longer needed)
    - Verify only trajectory files are generated
    - _Requirements: 2.3_

- [ ] 3. Checkpoint - Verify Issues 1 and 2 are fixed
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Investigate missile evasion crash root cause (Issue 3)
  - [ ] 4.1 Analyze crash logs and identify patterns
    - Review A0200 crash scenario (6380m → -2.5m in 12s)
    - Document altitude loss rates for different command combinations
    - Identify which commands lead to excessive descent
    - Document speed loss during high-G turns
    - _Requirements: 3.7, 3.8, 3.9, 4.1, 4.2, 4.3_

  - [ ] 4.2 Analyze command mapping and control authority
    - Review altitude command indices and their effects
    - Review heading command indices and turn rates
    - Review velocity command indices and acceleration
    - Identify control saturation issues
    - Document energy loss patterns
    - _Requirements: 3.6, 4.4, 4.5, 4.6_

  - [ ] 4.3 Document root cause findings
    - Create analysis document with evidence
    - Identify specific command combinations that cause crashes
    - Explain why compensation mechanisms don't work
    - Provide recommendations for fixes
    - _Requirements: 3.10, 4.7_

- [ ] 5. Implement safe evasion command generation
  - [ ] 5.1 Create safe evasion command generator
    - Add `_safe_evasion_command()` method to tactical task classes
    - Implement Rule 1: Never descend during evasion (alt_cmd >= 7)
    - Implement Rule 2: Always accelerate during evasion (vel_cmd >= 4)
    - Implement Rule 3: Reduce turn rate if speed is low
    - Implement Rule 4: Prioritize speed recovery if critically low
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 5.2 Write property test for safe altitude
    - **Property 4: Safe Altitude During Evasion**
    - **Validates: Requirements 3.1**

  - [ ]* 5.3 Write property test for airspeed above stall
    - **Property 5: Airspeed Above Stall During Turns**
    - **Validates: Requirements 3.2**

  - [ ]* 5.4 Write property test for descent rate limits
    - **Property 6: Descent Rate Limits**
    - **Validates: Requirements 3.3**

  - [ ]* 5.5 Write property test for energy conservation
    - **Property 7: Energy Conservation**
    - **Validates: Requirements 3.4**

  - [ ]* 5.6 Write property test for control input bounds
    - **Property 8: Control Input Bounds**
    - **Validates: Requirements 3.6**

- [ ] 6. Integrate safe evasion into tactical task system
  - [ ] 6.1 Update friendly aircraft evasion logic
    - Modify `tactical_task.py` to use `_safe_evasion_command()`
    - Replace existing evasion command generation
    - Add safety checks before executing commands
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ] 6.2 Update enemy aircraft evasion logic
    - Modify `unified_enemy_tactical_ai.py` to use safe evasion
    - Apply same safety rules as friendly aircraft
    - Ensure consistent behavior across both sides
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 6.3 Write integration tests for evasion scenarios
    - Test rapid descent scenario (A0200 crash case)
    - Test high-G turn at various speeds
    - Test combined turn + descent maneuvers
    - Test speed recovery from low-energy states
    - Test long duration simulations (11 minutes)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 7. Add safety monitoring and logging
  - [ ] 7.1 Implement safety violation detection
    - Add checks for altitude below safe threshold
    - Add checks for speed below stall speed
    - Add checks for excessive descent rate
    - Add checks for control saturation
    - _Requirements: 3.1, 3.2, 3.3, 3.6_

  - [ ] 7.2 Add detailed logging for evasion maneuvers
    - Log command generation decisions
    - Log safety rule applications
    - Log state changes during evasion
    - Log any safety violations detected
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 7.3 Write unit tests for safety monitoring
    - Test altitude violation detection
    - Test speed violation detection
    - Test descent rate violation detection
    - Test control saturation detection
    - _Requirements: 3.1, 3.2, 3.3, 3.6_

- [ ] 8. Final validation and testing
  - [ ] 8.1 Run comprehensive test suite
    - Run all unit tests
    - Run all property tests
    - Run all integration tests
    - Verify no crashes occur
    - _Requirements: All_

  - [ ] 8.2 Run extended simulation tests
    - Run multiple 11-minute simulations
    - Test with different aircraft configurations
    - Test with different tactical scenarios
    - Verify no crashes in any scenario
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ] 8.3 Validate output data quality
    - Check trajectory CSV files for correct radar types
    - Verify only required files are generated
    - Check for any data quality issues
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3_

- [ ] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Issues 1 and 2 are quick fixes that should be completed first
- Issue 3 requires careful investigation before implementing fixes
- The crash issue is MOST CRITICAL according to user requirements
- All fixes should be validated with the actual crash scenarios from logs
- Property tests should run minimum 100 iterations each
- Integration tests should use realistic simulation scenarios
