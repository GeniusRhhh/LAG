# Requirements Document

## Introduction

This specification addresses three critical issues in the tactical air combat simulation system that affect data accuracy, system performance, and aircraft survivability. These fixes are essential for ensuring reliable simulation results and preventing aircraft crashes during combat maneuvers.

## Glossary

- **System**: The tactical air combat simulation system
- **Radar_Type**: The radar model identifier (APG-68 for F-16, N001VE for SU-27)
- **CSV_File**: Comma-separated values data file
- **ACMI_File**: Air Combat Maneuvering Instrumentation file format
- **Trajectory_Data**: Aircraft position, velocity, and state information over time
- **Evasion_Maneuver**: Defensive aircraft maneuver to avoid incoming missiles
- **Control_Command**: Discrete altitude, heading, and velocity command indices
- **Baseline_Model**: Low-level flight control neural network model

## Requirements

### Requirement 1: Radar Type Detection Accuracy

**User Story:** As a simulation analyst, I want accurate radar type identification in trajectory data, so that I can correctly analyze aircraft sensor capabilities and tactical behavior.

#### Acceptance Criteria

1. WHEN recording friendly aircraft (A-prefix, SU-27) trajectory data, THE System SHALL set own_radar_type to "N001VE"
2. WHEN recording enemy aircraft (B-prefix, F-16) trajectory data, THE System SHALL set own_radar_type to "APG-68"
3. WHEN determining radar type, THE System SHALL check env.task.friend_lowlevel_type for friendly aircraft
4. WHEN determining radar type, THE System SHALL check env.task.enemy_lowlevel_type for enemy aircraft
5. THE System SHALL NOT default to "APG-68" for all aircraft regardless of type

### Requirement 2: Output File Management

**User Story:** As a simulation operator, I want to keep only essential output files, so that I can reduce storage usage and focus on relevant data.

#### Acceptance Criteria

1. WHEN saving simulation results, THE System SHALL generate ACMI files (.acmi)
2. WHEN saving simulation results, THE System SHALL generate log files (.log)
3. WHEN saving simulation results, THE System SHALL generate trajectory CSV files (_trajectory_*.csv)
4. WHEN saving simulation results, THE System SHALL NOT generate radar status CSV files (_radar_status_*.csv)
5. WHEN saving simulation results, THE System SHALL NOT generate missile trajectory CSV files (_missile_trajectory_*.csv)
6. WHEN saving simulation results, THE System SHALL NOT generate missile analysis CSV files (_missile_analysis_*.csv)
7. WHEN saving simulation results, THE System SHALL NOT generate missile summary TXT files (_missile_summary_*.txt)

### Requirement 3: Missile Evasion Crash Prevention

**User Story:** As a simulation developer, I want to identify and fix the root cause of aircraft crashes during missile evasion, so that aircraft can survive defensive maneuvers.

#### Acceptance Criteria

1. WHEN an aircraft performs missile evasion maneuvers, THE System SHALL maintain safe altitude above ground level
2. WHEN an aircraft performs high-G turns during evasion, THE System SHALL maintain sufficient airspeed to prevent stall
3. WHEN altitude commands are issued during evasion, THE System SHALL prevent excessive descent rates
4. WHEN combining turn and descent commands, THE System SHALL account for energy loss
5. WHEN speed falls below safe thresholds during evasion, THE System SHALL prioritize speed recovery over other maneuvers
6. THE System SHALL map tactical commands to JSBSim control inputs without causing control saturation
7. WHEN analyzing crash causes, THE System SHALL examine altitude command values (alt=6, 9, 10)
8. WHEN analyzing crash causes, THE System SHALL examine heading command values (hdg=13 for aggressive turns)
9. WHEN analyzing crash causes, THE System SHALL examine velocity command values and energy management
10. THE System SHALL NOT rely solely on compensation mechanisms that have been proven ineffective

### Requirement 4: Root Cause Analysis Documentation

**User Story:** As a simulation developer, I want comprehensive root cause analysis of the crash issue, so that I can implement effective fixes rather than band-aid solutions.

#### Acceptance Criteria

1. THE System SHALL document the relationship between control commands and aircraft state changes
2. THE System SHALL document energy loss patterns during high-G maneuvers
3. THE System SHALL document altitude loss rates for different command combinations
4. THE System SHALL identify which command combinations lead to stall conditions
5. THE System SHALL analyze control authority limitations during evasion
6. THE System SHALL examine the mapping between tactical-level commands and low-level JSBSim controls
7. THE System SHALL provide evidence-based recommendations for fixes
