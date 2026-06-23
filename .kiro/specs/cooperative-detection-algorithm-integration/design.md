# Cooperative Detection Algorithm Integration - Design

## Architecture Overview

This design integrates 5 unconnected algorithms into the `_get_intercept_action()` method following the 7-layer architecture from Algorithm 0 in the report.

### Current State

```
_get_intercept_action():
  1. Get enemy center (simple)
  2. Call formation_guidance.compute_guidance()
  3. Call velocity_coordination.compute_coordinated_speeds()
  4. Return commands
```

### Target State

```
_get_intercept_action():
  # Layer 1: Information
  Q_awacs = compute_awacs_quality()
  target_lost = check_target_lost()
  
  # Layer 2: Estimation (IMM-EKF)
  for target in targets:
      pos, P = coop_detection.update_target_state_imm(target, obs, time)
  
  # Layer 3: Prediction (sigma_man)
  sigma_man, r_targets = coop_detection.propagate_sigma_man(time, gamma=0.99)
  
  # Layer 4: Region (sigma_enemy, R_target)
  sigma_enemy, R_target = coop_detection.compute_distribution_range(
      target_positions, time, sigma_awacs=2.5, k_sigma=3.0
  )
  
  # Layer 5: Planning (Formation Guidance)
  guidance = formation_guidance.compute_guidance(
      agents, enemy_center, R_target, formation_pairs
  )
  
  # Layer 6: Coordination (Velocity)
  speeds = velocity_coordination.compute_coordinated_speeds(
      positions, target_positions, current_speeds
  )
  
  # Layer 7: Execution + Dynamic Adjustment
  headings, speeds, action = coop_detection.dynamic_path_adjustment(
      current_headings, current_speeds,
      target_headings, target_speeds,
      event_awacs_update, event_maneuver, event_lost
  )
  
  return commands
```

## Component Design

### 1. IMM-EKF Integration (Algorithm 1.1)

**Location:** `cooperative_detection.py::update_target_state_imm()`

**Integration Point:** Layer 2 (Estimation) in `_get_intercept_action()`

**Data Flow:**
```
Input: target_id, position (x,y), current_time
Process: IMM filter with 4 models (CV/CT/CA/Singer)
Output: (estimated_position, covariance_matrix P)
Storage: self._target_covariances[target_id] = P
```

**Implementation:**
```python
# In _get_intercept_action()
for target_id, obs_pos in detected_targets.items():
    est_pos, P = self.coop_detection.update_target_state_imm(
        target_id, obs_pos, current_time
    )
    # P is stored internally for sigma_man calculation
```

### 2. Maneuver Uncertainty Propagation (Algorithm 2.5)

**Location:** `cooperative_detection.py::propagate_sigma_man()`

**Integration Point:** Layer 3 (Prediction) in `_get_intercept_action()`

**Data Flow:**
```
Input: current_time, gamma (confidence level)
Process:
  1. For each target, predict covariance forward (no measurement)
  2. Extract position covariance P_pos (2x2)
  3. Compute eigenvalues, r_j = sqrt(chi2 * lambda_max)
  4. sigma_man = max(r_j) across all targets
Output: (sigma_man, {target_id: r_j})
```

**Implementation:**
```python
# In _get_intercept_action()
sigma_man, r_targets = self.coop_detection.propagate_sigma_man(
    current_time, gamma=0.99
)
# sigma_man used in Layer 4
```

### 3. Distribution Range Calculation (Formula 2.4)

**Location:** `cooperative_detection.py::compute_distribution_range()`

**Integration Point:** Layer 4 (Region) in `_get_intercept_action()`

**Formula:**
```
sigma_spatial = sqrt(mean((pos - center)^2))
sigma_enemy = max(sigma_spatial, sigma_awacs) + sigma_man
R_target = k_sigma * sigma_enemy
```

**Implementation:**
```python
# In _get_intercept_action()
sigma_enemy, R_target = self.coop_detection.compute_distribution_range(
    target_positions=detected_targets,
    current_time=current_time,
    sigma_awacs=2.5,  # AWACS error (km)
    k_sigma=3.0       # 3-sigma coverage
)
# R_target passed to formation_guidance
```

### 4. Dynamic Path Adjustment (Algorithm 2.10)

**Location:** `cooperative_detection.py::dynamic_path_adjustment()`

**Integration Point:** Layer 7 (Execution) in `_get_intercept_action()`

**Event Detection:**
```python
event_awacs_update = (current_time - last_awacs_time) < 1.0
event_maneuver = detect_maneuver(target_velocities)
event_awacs_lost = (current_time - last_awacs_time) > 10.0
event_target_lost = target_lost_flag
```

**Smoothing:**
```
theta_new = alpha * theta_current + (1 - alpha) * theta_target
v_new = alpha * v_current + (1 - alpha) * v_target
```

**Implementation:**
```python
# In _get_intercept_action()
new_headings, new_speeds, action_taken = self.coop_detection.dynamic_path_adjustment(
    current_headings={aid: self._get_heading(env, aid) for aid in agents},
    current_speeds={aid: get_speed(env, aid) for aid in agents},
    target_headings=guidance_headings,
    target_speeds=coordinated_speeds,
    event_awacs_update=awacs_updated,
    event_maneuver_detected=maneuver_detected,
    event_awacs_lost=awacs_lost,
    event_target_lost=target_lost,
    alpha=0.7  # Smoothing coefficient
)
```

### 5. Dynamic Scan Range (Formula 3.3)

**Location:** `cooperative_detection.py::compute_dynamic_scan_range()`

**Integration Point:** Radar scan allocation in `_assign_directed_scan()`

**Formula:**
```
theta_scan = 2 * arctan(r / distance)
theta_scan = clamp(theta_scan, 10, 60)  # Radar limits
```

**Implementation:**
```python
# In _assign_directed_scan()
for aid in agents:
    distance = compute_distance(agent_pos, target_pos)
    scan_range = self.coop_detection.compute_dynamic_scan_range(
        target_id, distance, gamma=0.99
    )
    # Use scan_range for radar mode selection
```

## Integration Flow

### Modified `_get_intercept_action()` Pseudocode

```python
def _get_intercept_action(self, env, agent_id: str) -> Tuple[int, int, int]:
    """INTERCEPT state action with full 7-layer algorithm integration"""
    
    # ========== Layer 1: Information ==========
    current_time = env.current_step * 0.2
    awacs_tracks = self.picture.get_awacs_tracks()
    Q_awacs = self._compute_awacs_quality(awacs_tracks)
    target_lost = self._check_target_lost(awacs_tracks, current_time)
    
    # ========== Layer 2: Estimation (IMM-EKF) ==========
    detected_targets = {}
    for track in awacs_tracks:
        tid = track.track_id
        obs_pos = (track.position[0], track.position[1])
        est_pos, P = self.coop_detection.update_target_state_imm(
            tid, obs_pos, current_time
        )
        detected_targets[tid] = est_pos
    
    # ========== Layer 3: Prediction (sigma_man) ==========
    sigma_man, r_targets = self.coop_detection.propagate_sigma_man(
        current_time, gamma=0.99
    )
    
    # ========== Layer 4: Region (sigma_enemy, R_target) ==========
    sigma_enemy, R_target = self.coop_detection.compute_distribution_range(
        target_positions=detected_targets,
        current_time=current_time,
        sigma_awacs=2.5,
        k_sigma=3.0
    )
    
    enemy_center, _ = self.coop_detection.estimate_enemy_center(detected_targets)
    
    # ========== Layer 5: Planning (Formation Guidance) ==========
    agents = self._get_alive_agents(env)
    agent_positions = {aid: self._get_battlefield_pos(env, aid) for aid in agents}
    
    guidance = self.formation_guidance.compute_guidance(
        agent_positions=agent_positions,
        enemy_center=enemy_center,
        R_target=R_target,  # ← Now uses full chain result
        formation_pairs=[('A0100', 'A0200'), ('A0300', 'A0400')]
    )
    
    # ========== Layer 6: Coordination (Velocity) ==========
    current_speeds = {aid: self._get_speed(env, aid) for aid in agents}
    coordinated_speeds = self.velocity_coordination.compute_coordinated_speeds(
        agent_positions=agent_positions,
        target_positions={aid: guidance[aid]['target_pos'] for aid in agents},
        current_speeds=current_speeds,
        v_nominal=250.0,
        v_min=200.0,
        v_max=300.0,
        delta_T_sync=5.0
    )
    
    # ========== Layer 7: Execution + Dynamic Adjustment ==========
    current_headings = {aid: self._get_heading(env, aid) for aid in agents}
    target_headings = {aid: guidance[aid]['target_heading'] for aid in agents}
    
    # Event detection
    awacs_updated = self._detect_awacs_update(current_time)
    maneuver_detected = self._detect_maneuver(detected_targets, current_time)
    awacs_lost = self._detect_awacs_lost(current_time)
    
    new_headings, new_speeds, action_taken = self.coop_detection.dynamic_path_adjustment(
        current_headings=current_headings,
        current_speeds=current_speeds,
        target_headings=target_headings,
        target_speeds=coordinated_speeds,
        event_awacs_update=awacs_updated,
        event_maneuver_detected=maneuver_detected,
        event_awacs_lost=awacs_lost,
        event_target_lost=target_lost,
        alpha=0.7
    )
    
    # Convert to action commands
    my_heading = new_headings[agent_id]
    my_speed = new_speeds[agent_id]
    alt_cmd, hdg_cmd, spd_cmd = self._convert_to_commands(
        env, agent_id, my_heading, my_speed
    )
    
    # Logging (throttled)
    if current_time % 10 < 0.3:
        log.info(f"[INTERCEPT] {agent_id} | sigma_man={sigma_man:.2f} "
                 f"R_target={R_target:.2f} action={action_taken}")
    
    return alt_cmd, hdg_cmd, spd_cmd
```

## Data Structures

### Target State Storage

```python
# In CooperativeDetection class
self._imm_estimators: Dict[str, IMMFilter] = {}
self._target_covariances: Dict[str, np.ndarray] = {}  # P matrices
self._last_track_time: Dict[str, float] = {}
```

### Event Tracking

```python
# In CAPTask class
self._last_awacs_update_time: float = 0.0
self._last_target_velocities: Dict[str, Tuple[float, float]] = {}
```

## Error Handling

### Missing IMM Module

```python
try:
    from .imm_estimator import IMMFilter
except ImportError:
    log.warning("IMM module not found, using fallback estimation")
    # Fallback: use simple Kalman filter or direct observation
```

### Invalid Covariance

```python
if P is None or P.shape[0] < 2:
    # Use default covariance
    P = np.eye(4) * 10.0
```

### Numerical Stability

```python
# Clamp eigenvalues to prevent sqrt of negative
eigenvalues = np.maximum(eigenvalues, 1e-6)
```

## Performance Considerations

1. **IMM Update Frequency**: Only update when new measurements arrive
2. **Covariance Propagation**: Cache results for 0.2s decision cycle
3. **Logging Throttling**: Log every 10s to avoid spam
4. **Fallback Paths**: Graceful degradation if algorithms fail

## Testing Strategy

1. **Unit Tests**: Test each algorithm in isolation
2. **Integration Tests**: Test 7-layer flow with mock data
3. **Scenario Tests**: Run with existing CAP scenarios
4. **Performance Tests**: Verify cycle time ≤ 0.2s
5. **Regression Tests**: Ensure existing functionality unchanged

## Correctness Properties

**Property 1.1**: IMM-EKF Convergence
- **Validates**: Requirements 1.1-1.4
- **Property**: After N measurements, state estimate error < threshold
- **Test**: Track known trajectory, measure RMSE

**Property 2.1**: Confidence Region Growth
- **Validates**: Requirements 2.1-2.6
- **Property**: sigma_man increases monotonically during no-measurement period
- **Test**: Simulate measurement loss, verify sigma_man(t+dt) >= sigma_man(t)

**Property 3.1**: R_target Composition
- **Validates**: Requirements 3.1-3.5
- **Property**: R_target = k_sigma * (max(sigma_spatial, sigma_awacs) + sigma_man)
- **Test**: Verify formula with known inputs

**Property 4.1**: Command Smoothness
- **Validates**: Requirements 4.1-4.5
- **Property**: |heading(t+1) - heading(t)| < max_rate * dt
- **Test**: Verify no command jumps exceed physical limits

**Property 5.1**: Scan Range Bounds
- **Validates**: Requirements 5.1-5.5
- **Property**: 10° ≤ scan_range ≤ 60° (radar limits)
- **Test**: Verify clamping for all distances

**Property 6.1**: Layer Execution Order
- **Validates**: Requirements 6.1-6.8
- **Property**: Layers execute in sequence 1→2→3→4→5→6→7
- **Test**: Instrument code to verify call order

## Migration Path

### Phase 1: Add Logging (No Behavior Change)
- Add calls to all 5 algorithms
- Log results but don't use them
- Verify no performance impact

### Phase 2: Enable IMM-EKF
- Use IMM results for state estimation
- Keep R_target calculation unchanged
- Verify tracking accuracy

### Phase 3: Enable Full Chain
- Use sigma_man → sigma_enemy → R_target
- Verify formation guidance adapts
- Monitor R_target values

### Phase 4: Enable Dynamic Adjustment
- Enable event-triggered path adjustment
- Verify smooth transitions
- Monitor command smoothness

### Phase 5: Enable Dynamic Scan Range
- Use dynamic scan range for radar
- Verify scan mode selection
- Monitor detection performance

## Rollback Plan

If integration causes issues:
1. Disable dynamic adjustment (alpha=1.0 → no smoothing)
2. Revert to simple R_target calculation
3. Disable IMM-EKF, use direct observations
4. Full rollback to current implementation

## Success Criteria

- [ ] All 5 algorithms called in correct order
- [ ] R_target follows full chain: IMM → sigma_man → sigma_enemy → R_target
- [ ] Confidence regions grow/shrink appropriately
- [ ] Formation guidance uses dynamic R_target
- [ ] Commands are smoothed (no jumps)
- [ ] Cycle time ≤ 0.2s
- [ ] Existing tests pass
- [ ] New property tests pass
