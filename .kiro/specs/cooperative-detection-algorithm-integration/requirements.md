# Cooperative Detection Algorithm Integration - Requirements

## Overview

Integrate 5 unconnected cooperative detection algorithms from the algorithm report into the CAP task main execution loop, following the 7-layer architecture defined in "Algorithm 0".

## Background

The cooperative detection algorithm report (协同探测算法报告_上交版.md) defines 9 algorithms for multi-aircraft cooperative detection. Currently, only 4 are integrated into `cap_task.py`:

**Connected (4):**
- ✅ Algorithm 2.6 (Formation Guidance)
- ✅ Algorithm 2.7 (Velocity Coordination)
- ✅ 2.2.1 (Enemy Center Estimation)
- ✅ 3.1-3.2 (Radar Scan Allocation)

**Not Connected (5):**
- ❌ Algorithm 1.1 (IMM-EKF) - `update_target_state_imm()` never called
- ❌ Algorithm 2.5 (sigma_man propagation) - `propagate_sigma_man()` never called
- ❌ Formula 2.4 (Distribution range) - `compute_distribution_range()` never called
- ❌ Algorithm 2.10 (Dynamic path adjustment) - `dynamic_path_adjustment()` never called
- ❌ Formula 3.3 (Dynamic scan range) - `compute_dynamic_scan_range()` never called

## Problem Statement

The current R_target (confidence radius) calculation is oversimplified. It doesn't follow the complete chain defined in the report:

**Current (Simplified):**
```
estimate_enemy_center() → simple formula → R_target
```

**Required (Full Chain):**
```
IMM-EKF → sigma_man propagation → sigma_enemy synthesis → R_target = k_σ·σ_enemy
```

## User Stories

### 1. IMM-EKF State Estimation Integration

**As a** cooperative detection system  
**I want** to use IMM-EKF for target state estimation  
**So that** I can accurately track maneuvering targets with multiple motion models

**Acceptance Criteria:**
- 1.1 `update_target_state_imm()` is called for each detected target
- 1.2 IMM maintains 4 motion models (CV/CT/CA/Singer)
- 1.3 State covariance matrix P is updated and stored
- 1.4 Model probabilities are tracked and logged

### 2. Maneuver Uncertainty Propagation

**As a** cooperative detection system  
**I want** to propagate maneuver uncertainty during no-measurement periods  
**So that** confidence regions grow appropriately when targets maneuver

**Acceptance Criteria:**
- 2.1 `propagate_sigma_man()` is called each decision cycle
- 2.2 Covariance is propagated forward using IMM predict
- 2.3 Confidence ellipse is computed with chi-square threshold
- 2.4 sigma_man = max(r_j) across all targets
- 2.5 sigma_man increases when measurements are lost
- 2.6 sigma_man decreases when measurements resume

### 3. Distribution Range Calculation

**As a** cooperative detection system  
**I want** to compute enemy distribution range using the full formula  
**So that** R_target accurately reflects spatial distribution + AWACS error + maneuver uncertainty

**Acceptance Criteria:**
- 3.1 `compute_distribution_range()` is called to compute sigma_enemy
- 3.2 sigma_spatial is computed from target positions
- 3.3 sigma_enemy = max(sigma_spatial, sigma_awacs) + sigma_man
- 3.4 R_target = k_sigma * sigma_enemy (k_sigma = 3.0)
- 3.5 R_target is used by formation guidance (Algorithm 2.6)

### 4. Dynamic Path Adjustment

**As a** cooperative detection system  
**I want** to adjust guidance paths dynamically based on events  
**So that** the formation responds appropriately to AWACS updates, maneuvers, and target loss

**Acceptance Criteria:**
- 4.1 `dynamic_path_adjustment()` is called each cycle
- 4.2 Four event types are detected: AWACS_UPDATE, MANEUVER_DETECTED, AWACS_LOST, TARGET_LOST
- 4.3 Heading and speed commands are smoothed with alpha coefficient
- 4.4 Event-triggered actions are logged
- 4.5 Smooth transitions prevent command jumps

### 5. Dynamic Scan Range Calculation

**As a** cooperative detection system  
**I want** to compute radar scan range dynamically based on confidence ellipse  
**So that** scan width adapts to target uncertainty and distance

**Acceptance Criteria:**
- 5.1 `compute_dynamic_scan_range()` is called for radar allocation
- 5.2 Scan angle = 2 * arctan(r / distance)
- 5.3 Scan range is clamped to [10°, 60°] (radar limits)
- 5.4 Scan range increases when uncertainty grows
- 5.5 Scan range decreases when targets are close

### 6. 7-Layer Architecture Integration

**As a** cooperative detection system  
**I want** to follow the 7-layer flow from Algorithm 0  
**So that** all algorithms execute in the correct order each decision cycle

**Acceptance Criteria:**
- 6.1 Layer 1 (Information): Q_awacs, target_lost computed
- 6.2 Layer 2 (Estimation): IMM-EKF updates target states
- 6.3 Layer 3 (Prediction): sigma_man propagated
- 6.4 Layer 4 (Region): sigma_enemy and R_target computed
- 6.5 Layer 5 (Planning): Formation guidance uses R_target
- 6.6 Layer 6 (Coordination): Velocity coordination applied
- 6.7 Layer 7 (Execution): Commands converted to control inputs
- 6.8 All layers execute in sequence each cycle

## Constraints

1. **Compatibility**: Must not break existing functionality
2. **Performance**: Decision cycle must remain ≤ 0.2s
3. **Logging**: Add debug logging for each algorithm (throttled)
4. **Fallback**: Gracefully handle missing IMM module
5. **Testing**: Verify with existing test scenarios

## Success Metrics

1. All 5 algorithms are called in `_get_intercept_action()`
2. R_target follows the complete chain: IMM → sigma_man → sigma_enemy → R_target
3. Confidence regions grow/shrink appropriately with measurement updates
4. Formation guidance adapts to dynamic R_target
5. No performance degradation (cycle time ≤ 0.2s)
6. Existing tests pass

## Out of Scope

- Modifying the algorithm implementations themselves
- Adding new algorithms not in the report
- Changing the report specifications
- Performance optimization beyond basic integration

## References

- Algorithm Report: `scripts/tacticalProject/cap/docs/协同探测算法报告_上交版.md`
- Main Task: `scripts/tacticalProject/cap/cap_task.py`
- Detection Module: `scripts/tacticalProject/cap/tactics/cooperative_detection.py`
