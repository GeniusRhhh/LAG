# Design Document

## Overview

This design addresses three critical issues in the tactical air combat simulation system. The fixes are prioritized as follows:
1. **Issue 1 (Radar Type Detection)**: Quick fix - correct the radar type detection logic
2. **Issue 2 (Output File Management)**: Quick fix - remove unnecessary file generation
3. **Issue 3 (Missile Evasion Crashes)**: Deep analysis and fix - requires investigation of root causes

The design follows a systematic approach: fix the simple issues first, then conduct thorough root cause analysis for the complex crash issue.

## Architecture

### System Components

The fixes involve three main components:

1. **Data Recorder** (`unified_data_recorder.py`)
   - Records aircraft trajectory data including radar types
   - Manages CSV file generation
   - Located in `scripts/tacticalTemplateProject/`

2. **Simulation Runner** (`run_simulation.py`, `batch_run_simulation.py`)
   - Orchestrates simulation execution
   - Manages output file generation
   - Located in `scripts/tacticalProject/`

3. **Tactical Task System** (`tactical_task.py`, `unified_enemy_tactical_ai.py`)
   - Generates tactical commands for aircraft
   - Handles missile evasion logic
   - Maps high-level tactical commands to low-level control inputs
   - Located in `scripts/tacticalProject/`

### Data Flow

```
Simulation Environment
  ↓
Tactical Task (generates commands)
  ↓
Baseline Model (executes commands)
  ↓
JSBSim (physics simulation)
  ↓
Data Recorder (records state)
  ↓
CSV Files (output)
```

## Components and Interfaces

### Issue 1: Radar Type Detection Fix

#### Current Implementation Problem

In `unified_data_recorder.py` lines 130-139 and 209-217, the radar type detection logic defaults to 'APG-68' for both friendly and enemy aircraft:

```python
# Current (buggy) code
enemy_radar_type = 'Unknown'
try:
    if hasattr(env.task, 'enemy_lowlevel_type'):
        if env.task.enemy_lowlevel_type == 'SU27':
            enemy_radar_type = 'N001VE'
        else:
            enemy_radar_type = 'APG-68'  # Bug: defaults to APG-68
    else:
        enemy_radar_type = 'APG-68'  # Bug: defaults to APG-68
except:
    enemy_radar_type = 'APG-68'  # Bug: defaults to APG-68
```

#### Fixed Implementation

The fix correctly maps aircraft types to radar types:
- Friendly aircraft (A-prefix, SU-27) → N001VE radar
- Enemy aircraft (B-prefix, F-16) → APG-68 radar

```python
# Fixed code
def _get_radar_type(self, env, agent_id: str) -> str:
    """Determine radar type based on aircraft type"""
    is_friendly = agent_id.startswith('A')
    
    if is_friendly:
        aircraft_type = getattr(env.task, 'friend_lowlevel_type', 'F16')
    else:
        aircraft_type = getattr(env.task, 'enemy_lowlevel_type', 'F16')
    
    return 'N001VE' if aircraft_type == 'SU27' else 'APG-68'
```

### Issue 2: Output File Management

#### Current Implementation

`unified_data_recorder.py` saves multiple file types:
- Trajectory CSV (keep)
- Radar status CSV (remove)
- Missile trajectory CSV (remove)
- Missile analysis CSV (remove)
- Missile summary TXT (remove)

#### Fixed Implementation

Modify `save_csv_files()` method to only save trajectory data:

```python
def save_csv_files(self, output_dir: str, timestamp: str = None) -> Dict[str, str]:
    """Save only trajectory CSV files"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    
    # Save only trajectory data
    if self.trajectory_data:
        trajectory_df = pd.DataFrame(self.trajectory_data)
        trajectory_file = os.path.join(output_dir, f"{self.project_name}_trajectory_{timestamp}.csv")
        trajectory_df.to_csv(trajectory_file, index=False, encoding='utf-8-sig')
        saved_files['trajectory'] = trajectory_file
        print(f"飞机轨迹数据已保存: {trajectory_file}")
    
    return saved_files
```

### Issue 3: Missile Evasion Crash Analysis

#### Root Cause Investigation

Based on the crash analysis document and log evidence, the crashes occur due to:

1. **Excessive Altitude Loss**: Altitude commands (alt=6, 9, 10) cause rapid descent
   - A0200 drops from 6380m → 5486m → 3841m → 2810m → crashes at -2.5m in ~12 seconds
   - Descent rate: ~500-800 m/s (far exceeds safe limits)

2. **Energy Loss During High-G Turns**: Combining aggressive turns (hdg=13) with descent causes energy depletion
   - High-G turns bleed energy rapidly
   - Simultaneous descent compounds the problem
   - Speed drops below stall speed

3. **Ineffective Speed Compensation**: Current compensation mechanisms don't work
   - User explicitly stated: "补偿机制是没用的" (compensation mechanisms don't work)
   - Speed compensation (vel_cmd=6) is applied too late
   - Baseline model cannot recover from severe energy deficit

#### Command Mapping Analysis

The tactical command system uses discrete indices:

**Altitude Commands** (15 levels):
```python
norm_delta_altitude = [-1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500] / 1000.0
# Index 6: -50m (slight descent)
# Index 9: +150m (climb)
# Index 10: +300m (climb)
```

**Heading Commands** (17 levels):
```python
norm_delta_heading = [-π, -2π/3, -π/2, -5π/12, -π/3, -π/4, -π/6, -π/12, 0, π/12, π/6, π/4, π/3, 5π/12, π/2, 2π/3, π]
# Index 13: +5π/12 ≈ +75° (aggressive turn)
```

**Velocity Commands** (7 levels):
```python
norm_delta_velocity = [-150, -100, -50, 0, 50, 100, 150] / 100.0
# Index 3: 0 (maintain speed)
# Index 6: +150m/s (accelerate)
```

#### Proposed Solution

The fix involves three strategies:

1. **Limit Descent Rate During Evasion**
   - Cap altitude commands to prevent excessive descent
   - Never use alt < 7 (index 7 = 0m, maintain altitude) during evasion
   - Prefer alt = 8-10 (climb) during evasion

2. **Prioritize Speed Recovery**
   - Always use vel_cmd >= 4 (accelerate) during evasion
   - Increase speed threshold for triggering recovery from 200 to 250 m/s
   - Apply speed recovery before executing other maneuvers

3. **Decouple Turn and Descent**
   - Avoid combining aggressive turns (hdg >= 12) with descent (alt < 7)
   - Execute maneuvers sequentially: turn first, then adjust altitude
   - Reduce turn rate when speed is low

```python
def _safe_evasion_command(self, current_state, threat_level):
    """Generate safe evasion commands that prevent crashes"""
    alt_cmd, hdg_cmd, vel_cmd = self._base_evasion_logic(threat_level)
    
    # Rule 1: Never descend during evasion
    if alt_cmd < 7:
        alt_cmd = 8  # Slight climb instead
    
    # Rule 2: Always accelerate during evasion
    if vel_cmd < 4:
        vel_cmd = 4  # Minimum: maintain/slight acceleration
    
    # Rule 3: Reduce turn rate if speed is low
    if current_state.speed < 250 and hdg_cmd >= 12:
        hdg_cmd = 10  # Reduce to moderate turn
        vel_cmd = 5   # Increase acceleration
    
    # Rule 4: Prioritize speed recovery if critically low
    if current_state.speed < 220:
        alt_cmd = 7   # Level flight
        hdg_cmd = 8   # Straight ahead
        vel_cmd = 6   # Maximum acceleration
    
    return alt_cmd, hdg_cmd, vel_cmd
```

## Data Models

### Trajectory Record

```python
@dataclass
class TrajectoryRecord:
    Time_s: float
    Agent_ID: str
    X_m: float
    Y_m: float
    Z_m: float
    Velocity_m_s: float
    Heading_deg: float
    Pitch_deg: float
    Roll_deg: float
    Own_Radar_Type: str  # "APG-68" or "N001VE"
    Own_Radar_Status: str
    Own_Radar_Target_ID: str
    Action_Type: str  # For enemy aircraft
    Action_Intent: str  # For enemy aircraft
```

### Aircraft State

```python
@dataclass
class AircraftState:
    position: Tuple[float, float, float]  # (x, y, altitude)
    velocity: float  # m/s
    heading: float  # radians
    pitch: float  # radians
    roll: float  # radians
    is_alive: bool
    aircraft_type: str  # "F16" or "SU27"
    radar_type: str  # "APG-68" or "N001VE"
```

### Evasion Command

```python
@dataclass
class EvasionCommand:
    alt_cmd: int  # 0-14 (altitude command index)
    hdg_cmd: int  # 0-16 (heading command index)
    vel_cmd: int  # 0-6 (velocity command index)
    is_safe: bool  # Whether command passes safety checks
    safety_violations: List[str]  # List of violated safety rules
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Friendly Aircraft Radar Type Correctness
*For any* friendly aircraft with SU-27 type, the recorded radar type should be "N001VE"
**Validates: Requirements 1.1**

### Property 2: Enemy Aircraft Radar Type Correctness
*For any* enemy aircraft with F-16 type, the recorded radar type should be "APG-68"
**Validates: Requirements 1.2**

### Property 3: Radar Type Consistency
*For any* aircraft, if the aircraft type changes, the radar type should change accordingly
**Validates: Requirements 1.5**

### Property 4: Safe Altitude During Evasion
*For any* evasion maneuver, the aircraft altitude should never fall below 1500m above ground level
**Validates: Requirements 3.1**

### Property 5: Airspeed Above Stall During Turns
*For any* high-G turn maneuver, the aircraft airspeed should remain above 180 m/s (stall speed + margin)
**Validates: Requirements 3.2**

### Property 6: Descent Rate Limits
*For any* altitude command during evasion, the resulting descent rate should not exceed 100 m/s
**Validates: Requirements 3.3**

### Property 7: Energy Conservation
*For any* combined turn and descent maneuver, the total energy loss should not exceed safe thresholds
**Validates: Requirements 3.4**

### Property 8: Control Input Bounds
*For any* tactical command, the mapped JSBSim control inputs should remain within valid ranges [-1, 1]
**Validates: Requirements 3.6**

## Error Handling

### Radar Type Detection Errors

1. **Missing Task Attribute**: If `env.task` doesn't have `friend_lowlevel_type` or `enemy_lowlevel_type`, default to F16/APG-68
2. **Invalid Aircraft Type**: If aircraft type is neither F16 nor SU27, log warning and default to APG-68
3. **Exception During Detection**: Catch all exceptions, log error, and use safe default (APG-68)

### File Generation Errors

1. **Directory Creation Failure**: If output directory cannot be created, log error and skip file generation
2. **CSV Write Failure**: If CSV write fails, log error but continue simulation
3. **Disk Space Issues**: Check available disk space before writing large files

### Evasion Command Safety Errors

1. **Invalid Command Index**: If command index is out of bounds, clamp to valid range
2. **Unsafe Command Detected**: If safety check fails, override with safe default command
3. **Critical Speed Loss**: If speed drops below 180 m/s, immediately switch to emergency recovery mode
4. **Altitude Warning**: If altitude drops below 2000m, trigger altitude recovery regardless of threat

## Testing Strategy

### Dual Testing Approach

This feature requires both unit tests and property-based tests:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs
- Both are complementary and necessary for comprehensive coverage

### Unit Testing

Unit tests should focus on:
- Specific radar type mappings (F16→APG-68, SU27→N001VE)
- File generation with specific configurations
- Known crash scenarios from logs
- Edge cases (missing attributes, invalid types)
- Error handling paths

### Property-Based Testing

Property tests should:
- Run minimum 100 iterations per test
- Use Python's `hypothesis` library for property-based testing
- Tag each test with format: **Feature: tactical-simulation-critical-fixes, Property {number}: {property_text}**
- Generate random aircraft states and verify safety properties hold
- Test command mapping bounds for all possible inputs

### Integration Testing

Integration tests should:
- Run full simulations with different aircraft configurations
- Verify no crashes occur during extended evasion scenarios
- Check that all output files are generated correctly
- Validate trajectory data contains correct radar types

### Test Configuration

```python
# Property test configuration
from hypothesis import given, strategies as st, settings

@settings(max_examples=100)
@given(
    aircraft_type=st.sampled_from(['F16', 'SU27']),
    agent_id=st.sampled_from(['A0100', 'A0200', 'B0100', 'B0200'])
)
def test_radar_type_property(aircraft_type, agent_id):
    """Feature: tactical-simulation-critical-fixes, Property 1: Radar type correctness"""
    # Test implementation
    pass
```

### Crash Scenario Testing

Specific test cases for Issue 3:
1. **Rapid Descent Test**: Simulate the A0200 crash scenario (6380m → crash in 12s)
2. **High-G Turn Test**: Test aggressive turns at various speeds
3. **Combined Maneuver Test**: Test turn + descent combinations
4. **Speed Recovery Test**: Test speed recovery from low-energy states
5. **Long Duration Test**: Run 11-minute simulations to catch cumulative effects
