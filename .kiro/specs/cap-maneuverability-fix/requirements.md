# Requirements Document

## Introduction

The CAP (Combat Air Patrol) tactical system currently suffers from two critical maneuverability issues that prevent effective threat interception. Aircraft respond extremely slowly to heading changes due to discrete action space quantization errors, and fly too slowly during intercept phases due to inadequate speed coordination. These issues manifest when engaging enemy formations at distances of 230-338km, where aircraft fail to adjust course quickly enough to intercept threats.

This specification addresses both root causes while maintaining the existing 7-layer cooperative detection algorithm architecture.

## Glossary

- **CAP_System**: The Combat Air Patrol tactical control system managing 4-aircraft formations
- **Heading_Controller**: The component responsible for converting target headings into discrete heading commands
- **Speed_Coordinator**: The VelocityCoordination algorithm (Algorithm 2.7) managing formation speed synchronization
- **Action_Space**: The discrete set of control commands (altitude, heading, velocity) available to aircraft
- **Quantization_Error**: The error introduced when mapping continuous values to discrete action indices
- **INTERCEPT_Phase**: The mission phase where aircraft move toward detected threats (400-200km range)
- **ENGAGE_Phase**: The mission phase where aircraft engage threats (<200km range)
- **PATROL_Phase**: The mission phase where aircraft maintain patrol patterns (>400km from threats)
- **norm_hdg**: The discrete heading change array in radians: [-π, -2π/3, ..., 0, ..., π]
- **norm_vel**: The discrete velocity change array normalized by 100 m/s: [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
- **Dead_Zone**: The heading difference threshold (currently 3°) below which no heading change is commanded

## Requirements

### Requirement 1: Heading Control Responsiveness

**User Story:** As a tactical system operator, I want aircraft to respond accurately to small heading changes (5-15°), so that they can effectively intercept threats at long ranges (230-338km).

#### Acceptance Criteria

1. WHEN the target heading differs from current heading by 5-15°, THE Heading_Controller SHALL generate a non-zero heading command that produces a proportional heading change
2. WHEN the target heading differs from current heading by less than 3°, THE Heading_Controller SHALL maintain current heading to prevent oscillation
3. WHEN the target heading differs from current heading by more than 15°, THE Heading_Controller SHALL generate the maximum appropriate heading command within the discrete action space
4. WHEN converting heading differences to discrete commands, THE Heading_Controller SHALL minimize quantization error by selecting the closest available discrete action
5. THE Heading_Controller SHALL handle heading wraparound correctly for differences crossing 0°/360° boundary

### Requirement 2: Heading Command Quantization

**User Story:** As a system designer, I want the discrete heading action space to support fine-grained heading adjustments, so that small course corrections are not lost to quantization.

#### Acceptance Criteria

1. THE Action_Space SHALL provide heading change increments of 15° or smaller to support tactical maneuvering
2. WHEN the current norm_hdg array has minimum spacing of π/12 (15°), THE System SHALL either refine the array or implement interpolation to support smaller changes
3. THE Heading_Controller SHALL map continuous heading differences to discrete commands using nearest-neighbor selection
4. WHEN multiple discrete commands are equally close to the target, THE Heading_Controller SHALL select the command that minimizes overshoot
5. THE System SHALL document the relationship between heading difference (degrees), heading difference (radians), and discrete command index

### Requirement 3: Speed Control During Intercept

**User Story:** As a tactical pilot, I want aircraft to maintain 250-300 m/s during INTERCEPT phase, so that I can close distance with threats effectively.

#### Acceptance Criteria

1. WHEN the CAP_System is in INTERCEPT_Phase, THE Speed_Coordinator SHALL compute target speeds in the range 250-300 m/s
2. WHEN the CAP_System is in ENGAGE_Phase, THE Speed_Coordinator SHALL compute target speeds in the range 250-330 m/s to support weapon employment
3. WHEN the CAP_System is in PATROL_Phase, THE Speed_Coordinator SHALL compute target speeds in the range 200-250 m/s for fuel efficiency
4. THE Speed_Coordinator SHALL prevent speed from decreasing below 250 m/s during INTERCEPT_Phase unless fuel constraints require it
5. WHEN formation synchronization conflicts with phase-based speed requirements, THE Speed_Coordinator SHALL prioritize phase-based minimums while maintaining synchronization within ±15 seconds

### Requirement 4: Phase-Aware Speed Coordination

**User Story:** As a system architect, I want speed coordination to consider mission phase, so that aircraft fly at tactically appropriate speeds for each phase.

#### Acceptance Criteria

1. THE Speed_Coordinator SHALL accept mission phase (PATROL, INTERCEPT, ENGAGE) as input to the speed computation algorithm
2. WHEN computing synchronized speeds, THE Speed_Coordinator SHALL apply phase-specific speed bounds before selecting candidate speeds
3. THE Speed_Coordinator SHALL modify the cost function J = |ΔT| + λ·Σ(v - v_nominal)² to use phase-specific v_nominal values
4. WHEN in INTERCEPT_Phase, THE Speed_Coordinator SHALL use v_nominal = 275 m/s (midpoint of 250-300 range)
5. WHEN in ENGAGE_Phase, THE Speed_Coordinator SHALL use v_nominal = 290 m/s (midpoint of 250-330 range)
6. WHEN in PATROL_Phase, THE Speed_Coordinator SHALL use v_nominal = 225 m/s (midpoint of 200-250 range)

### Requirement 5: Speed Command Execution

**User Story:** As a control system engineer, I want speed commands to be converted accurately to discrete velocity actions, so that commanded speeds are achieved in practice.

#### Acceptance Criteria

1. WHEN the Speed_Coordinator outputs a target speed, THE System SHALL convert it to a discrete velocity command using the norm_vel array
2. THE System SHALL compute speed difference as (target_speed - current_speed) and normalize by 100 m/s before discrete mapping
3. WHEN the normalized speed difference exceeds norm_vel bounds (±1.5), THE System SHALL clip to the maximum available command
4. THE System SHALL select the discrete velocity command that minimizes absolute difference from the normalized target
5. WHEN current speed is within ±10 m/s of target speed, THE System SHALL issue a maintain-speed command (norm_vel[3] = 0) to prevent oscillation

### Requirement 6: Heading Control Integration

**User Story:** As a system integrator, I want heading control fixes to integrate seamlessly with the existing 7-layer cooperative detection algorithm, so that tactical behavior is preserved.

#### Acceptance Criteria

1. WHEN the 7-layer algorithm (Layers 1-7) computes a target heading, THE Heading_Controller SHALL convert it to a discrete command without modifying the algorithm's output
2. THE Heading_Controller SHALL operate in Layer 7 (Execution Layer) after dynamic path adjustment smoothing
3. WHEN dynamic path adjustment applies heading smoothing (alpha=0.7), THE Heading_Controller SHALL use the smoothed heading as input
4. THE System SHALL preserve the existing dead zone logic (3° threshold) to prevent oscillation
5. THE Heading_Controller SHALL maintain compatibility with the existing normalize_action() method signature

### Requirement 7: Speed Coordination Integration

**User Story:** As a system integrator, I want speed coordination enhancements to integrate with the existing VelocityCoordination algorithm (Algorithm 2.7), so that formation synchronization is preserved.

#### Acceptance Criteria

1. THE Speed_Coordinator SHALL extend the existing VelocityCoordination class without breaking its public interface
2. WHEN compute_coordinated_speeds() is called, THE Speed_Coordinator SHALL accept an optional mission_phase parameter
3. THE Speed_Coordinator SHALL preserve the existing synchronization logic (ΔT_sync = 15s tolerance)
4. THE Speed_Coordinator SHALL preserve the existing cost function structure while allowing phase-specific v_nominal
5. WHEN mission_phase is not provided, THE Speed_Coordinator SHALL default to PATROL_Phase behavior for backward compatibility

### Requirement 8: Verification and Testing

**User Story:** As a quality assurance engineer, I want comprehensive tests to verify that heading and speed fixes work correctly across various scenarios, so that regressions are prevented.

#### Acceptance Criteria

1. THE System SHALL provide test scenarios covering heading changes from 3° to 180° in 5° increments
2. THE System SHALL provide test scenarios covering all three mission phases (PATROL, INTERCEPT, ENGAGE)
3. THE System SHALL provide test scenarios covering formation synchronization with phase-based speed constraints
4. WHEN testing heading control, THE System SHALL verify that commanded heading changes are proportional to target differences
5. WHEN testing speed control, THE System SHALL verify that aircraft maintain phase-appropriate speeds throughout mission phases
6. THE System SHALL provide test scenarios matching the user's reported cases: enemy square formation (230km) and line formation (338km)

### Requirement 9: Performance Monitoring

**User Story:** As a system operator, I want diagnostic logging to monitor heading and speed command execution, so that I can verify correct behavior during missions.

#### Acceptance Criteria

1. WHEN CAP_MANEUVER_DEBUG environment variable is set to '1', THE System SHALL log heading command details every 30 seconds
2. THE System SHALL log: current heading, target heading, heading difference (degrees and radians), selected command index, and actual command value
3. WHEN CAP_MANEUVER_DEBUG is set, THE System SHALL log speed command details every 30 seconds
4. THE System SHALL log: current speed, target speed, mission phase, speed difference, normalized difference, and selected command index
5. THE System SHALL log heading and speed diagnostics only for agent A0100 to minimize log volume

### Requirement 10: Backward Compatibility

**User Story:** As a system maintainer, I want the fixes to maintain backward compatibility with existing code, so that other system components are not affected.

#### Acceptance Criteria

1. THE System SHALL preserve the existing _get_intercept_action() method signature
2. THE System SHALL preserve the existing VelocityCoordination class constructor signature
3. THE System SHALL preserve the existing norm_hdg and norm_vel array structures
4. WHEN existing code calls VelocityCoordination without mission_phase parameter, THE System SHALL function correctly with default behavior
5. THE System SHALL not modify the 7-layer cooperative detection algorithm structure (Layers 1-6)
