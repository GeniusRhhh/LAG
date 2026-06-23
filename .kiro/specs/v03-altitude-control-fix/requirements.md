# Requirements Document

## Introduction

This document specifies the requirements for fixing the V03 scenario altitude control issue in the CAP tactical system. The V03 scenario (POPUP_TURN maneuver) has a critical bug where enemy aircraft don't climb as expected during the low-altitude penetration and pop-up maneuver. The issue has been diagnosed through multiple iterations (V4, V5, V6) and requires a V7 fix to correct the altitude control logic.

## Glossary

- **POPUP_TURN**: A tactical maneuver where aircraft fly at low altitude (3.5km) to avoid radar detection, then rapidly climb to engagement altitude (9km) while turning
- **Scenario_Generator**: The Python module responsible for generating enemy aircraft behavior scenarios
- **altitude_rate**: The rate of altitude change in km/s, used to control aircraft climb/descent
- **_popup_stage**: A state variable tracking which phase of the POPUP_TURN maneuver is currently active
- **Integration_Step**: A single simulation time step where altitude_rate is applied to update aircraft altitude
- **V03_Scenario**: The specific test scenario that validates low-altitude penetration with pop-up maneuver

## Requirements

### Requirement 1: Continuous Altitude Rate Execution

**User Story:** As a simulation developer, I want the altitude_rate command to execute continuously throughout the climb phase, so that enemy aircraft reach the target altitude as designed.

#### Acceptance Criteria

1. WHEN the distance is between 170km and 230km AND _popup_stage is 0, THE Scenario_Generator SHALL set altitude_rate to 0.10 km/s
2. WHEN the distance is between 170km and 230km, THE Scenario_Generator SHALL continue issuing altitude_rate commands on every simulation step
3. WHEN the distance becomes less than 170km, THE Scenario_Generator SHALL set altitude_rate to 0.0 to stop climbing
4. WHEN the distance becomes less than 170km, THE Scenario_Generator SHALL set _popup_stage to 1 to mark completion

### Requirement 2: Altitude Change Validation

**User Story:** As a test engineer, I want to verify that enemy aircraft altitude increases correctly during the climb phase, so that I can confirm the fix works as intended.

#### Acceptance Criteria

1. WHEN the V03 scenario executes, THE System SHALL increase enemy aircraft altitude from 16404ft to at least 24606ft
2. WHEN the climb phase is active (170km < distance < 230km), THE System SHALL show continuous altitude increase in simulation logs
3. WHEN the climb phase completes (distance < 170km), THE System SHALL maintain the final altitude without further changes
4. WHEN altitude changes occur, THE System SHALL record altitude values in the test output for verification

### Requirement 3: Stage Transition Logic

**User Story:** As a simulation developer, I want the stage transition to occur only after the climb is complete, so that the maneuver executes in the correct sequence.

#### Acceptance Criteria

1. WHEN the simulation starts, THE Scenario_Generator SHALL initialize _popup_stage to 0
2. WHILE distance is greater than or equal to 170km, THE Scenario_Generator SHALL maintain _popup_stage at 0
3. WHEN distance becomes less than 170km, THE Scenario_Generator SHALL transition _popup_stage from 0 to 1
4. AFTER _popup_stage transitions to 1, THE Scenario_Generator SHALL not revert to stage 0

### Requirement 4: Smooth Climb Trajectory

**User Story:** As a simulation analyst, I want the altitude increase to be smooth and continuous, so that the aircraft behavior is realistic and predictable.

#### Acceptance Criteria

1. WHEN altitude_rate is active, THE System SHALL apply the rate on every integration step without interruption
2. WHEN visualized in Tacview, THE System SHALL show a smooth climb trajectory without sudden jumps or oscillations
3. WHEN altitude changes occur, THE System SHALL not exceed the specified climb rate of 0.10 km/s
4. IF altitude oscillation or sudden jumps occur, THEN THE System SHALL log an error indicating trajectory instability

### Requirement 5: Test Validation

**User Story:** As a quality assurance engineer, I want automated tests to verify the altitude control fix, so that regressions can be detected quickly.

#### Acceptance Criteria

1. THE Test_Suite SHALL include a test case specifically for V03 altitude control
2. WHEN the test executes, THE Test_Suite SHALL verify altitude increases from 16404ft to 24606ft
3. WHEN the test executes, THE Test_Suite SHALL verify altitude change occurs between 230km and 170km distance
4. WHEN the test completes, THE Test_Suite SHALL report pass/fail status with detailed altitude measurements
5. IF the altitude increase is less than 8000ft, THEN THE Test_Suite SHALL fail with a descriptive error message
