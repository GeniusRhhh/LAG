# Implementation Plan: V03 Altitude Control Fix

## Overview

This implementation plan fixes the V03 scenario altitude control bug by modifying the POPUP_TURN maneuver logic in `scenario_generator.py`. The fix changes the condition for issuing altitude_rate commands from `distance < 230` to `170 <= distance < 230`, ensuring continuous command execution throughout the climb phase. The stage transition is moved to occur only when `distance < 170`, after the climb is complete.

## Tasks

- [ ] 1. Fix POPUP_TURN maneuver logic in scenario_generator.py
  - Modify the condition in lines 415-445 to use `170 <= distance < 230` for stage 0
  - Move stage transition to stage 1 into the `distance < 170` condition block
  - Ensure stage initialization check exists at the start of POPUP_TURN logic
  - Add comments explaining the fix and why continuous execution is needed
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 3.3, 3.4_

- [ ]* 2. Write property test for continuous altitude commands
  - **Property 1: Continuous Altitude Rate Commands**
  - **Validates: Requirements 1.1, 1.2, 5.3**
  - Use hypothesis to generate random distances in range [170.0, 229.9]
  - Verify all returned commands contain altitude_rate = 0.10 km/s
  - Run minimum 100 iterations
  - _Requirements: 1.1, 1.2, 5.3_

- [ ]* 3. Write property test for stage monotonicity
  - **Property 4: Stage Monotonicity**
  - **Validates: Requirements 3.2, 3.4**
  - Generate sequence of decreasing distances from 250km to 150km
  - Verify _popup_stage remains 0 until distance < 170km
  - Verify _popup_stage transitions to 1 and never reverts
  - Run minimum 100 iterations with different distance sequences
  - _Requirements: 3.2, 3.4_

- [ ]* 4. Write property test for altitude integration correctness
  - **Property 5: Correct Altitude Integration**
  - **Validates: Requirements 4.1, 4.3**
  - Generate random time steps and verify altitude change = rate × dt × 3280.84
  - Verify instantaneous rate never exceeds 0.10 km/s
  - Test with various dt values (0.1s to 1.0s)
  - Run minimum 100 iterations
  - _Requirements: 4.1, 4.3_

- [ ] 5. Write unit tests for boundary conditions
  - [ ] 5.1 Test stage initialization to 0
    - Verify _popup_stage initializes to 0 on first call
    - _Requirements: 3.1_
  
  - [ ] 5.2 Test climb starts at 230km boundary
    - Verify no climb at distance = 230.1km
    - Verify climb starts at distance = 229.9km
    - _Requirements: 1.1_
  
  - [ ] 5.3 Test climb stops at 170km boundary
    - Verify climb continues at distance = 170.1km
    - Verify climb stops at distance = 169.9km
    - Verify stage transitions to 1
    - _Requirements: 1.3, 1.4_
  
  - [ ] 5.4 Test edge case: simulation starts below 170km
    - Verify no climb commands issued
    - Verify stage immediately transitions to 1
    - _Requirements: 1.3, 1.4_

- [ ]* 6. Write integration test for end-to-end altitude increase
  - **Property 2: End-to-End Altitude Increase**
  - **Validates: Requirements 2.1, 5.2**
  - Run complete V03 scenario simulation (max 3000 steps)
  - Verify initial altitude is 16404ft ± 100ft
  - Verify final altitude is at least 24606ft
  - Verify altitude increase occurs between 230km and 170km
  - Record altitude history for analysis
  - _Requirements: 2.1, 5.2, 5.3_

- [ ]* 7. Write integration test for altitude stabilization
  - **Property 3: Altitude Stabilization After Climb**
  - **Validates: Requirements 2.3**
  - Run V03 scenario past the 170km mark
  - Verify altitude remains constant (±50ft) after distance < 170km
  - Check at least 100 steps after climb completion
  - _Requirements: 2.3_

- [ ] 8. Create V7 test file and test runner
  - Create `test_v03_fix_v7.py` in `scripts/tacticalProject/cap/tests/`
  - Import all test functions from tasks 2-7
  - Add test configuration and setup
  - Add main block to run tests with pytest
  - Document expected results and success criteria
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 9. Checkpoint - Run all tests and verify fix
  - Run pytest on test_v03_fix_v7.py with verbose output
  - Verify all property tests pass with 100+ iterations
  - Verify all unit tests pass
  - Verify integration tests show correct altitude increase
  - Check for any test failures or warnings
  - If any tests fail, debug and fix before proceeding

- [ ] 10. Update documentation
  - Update V03_V6_Complete_Diagnosis.md with V7 fix details
  - Document the root cause and solution
  - Add before/after code comparison
  - Document test results and validation
  - _Requirements: All_

- [ ] 11. Final checkpoint - Ensure all tests pass
  - Re-run complete test suite
  - Verify no regressions in other scenarios
  - Confirm Tacview visualization shows smooth climb (manual check)
  - Ask user if any questions or issues arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The core fix is in task 1 - all other tasks are validation
- Property tests use hypothesis library for property-based testing
- Integration tests require running full simulation (may take several minutes)
- Each property test references its design document property number
- Minimum 100 iterations per property test ensures thorough coverage
