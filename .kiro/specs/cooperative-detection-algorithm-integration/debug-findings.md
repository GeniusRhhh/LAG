# 调试发现与问题分析

## 问题1：雷达调试输出已注释

✅ **已完成**：已将以下雷达扫描相关的调试输出注释掉：

1. `cooperative_detection.py`:
   - `[扇区分配] 目标数:...` (第220行)
   - `[扇区分配] ✅ 模式:定向扫描...` (第266-273行)
   - `[扫描] DIRECTED | ...` (第131行)

2. `cap_task.py`:
   - `[扇区传递] ...` (第2111行)

现在雷达相关的调试输出已被注释，不会干扰协同探测决策的日志。

## 问题2：为什么没有看到7层架构输出

### 根本原因

**7层架构输出只在INTERCEPT状态下才会触发**，但当前配置下可能没有进入INTERCEPT状态。

### 状态转换条件分析

根据`cap_state_machine.py`的逻辑：

```python
def _should_intercept(self, ctx: StateContext) -> bool:
    """是否应拦截（编队前出）
    
    V5条件：
    - 进入：预警机有信息且距离 < 400km
    - 退出：距离 > 400km + 滞后 或 预警机无信息
    """
    if self._should_engage(ctx):
        return self.state == CAPState.INTERCEPT
    
    # 无预警信息则不进入拦截
    if not ctx.has_awacs_info:
        return False
    
    # 滞后逻辑
    if self.state == CAPState.INTERCEPT:
        exit_threshold = AWACS_DETECTION_RANGE + self.HYSTERESIS  # 430km
        return ctx.min_threat_distance < exit_threshold
    else:
        enter_threshold = AWACS_DETECTION_RANGE  # 400km
        return ctx.min_threat_distance < enter_threshold
```

**进入INTERCEPT状态的条件**：
1. 有预警机信息 (`has_awacs_info = True`)
2. 最近威胁距离 < 400km
3. 不满足ENGAGE条件（距离 >= 200km）

### 当前配置的初始距离

根据`patrol_config.yaml`：

- **我方位置**：
  - A0100: (120.6757°, 60.0°) → 战场坐标约 (75km, 0km)
  - A0200: (120.4252°, 60.2009°) → 战场坐标约 (25km, 100km)
  - A0300: (121.3766°, 60.0°) → 战场坐标约 (125km, 0km)
  - A0400: (121.1261°, 60.2009°) → 战场坐标约 (175km, 100km)

- **敌方位置**：
  - B0100-B0400: 纬度 62.4° → 战场坐标约 Y=250km

**初始距离计算**：
- A0100 (75, 0) 到 B0100 (约40, 250)：
  - 距离 ≈ √[(75-40)² + (0-250)²] ≈ √[1225 + 62500] ≈ **252km**

- A0200 (25, 100) 到 B0100 (约40, 250)：
  - 距离 ≈ √[(25-40)² + (100-250)²] ≈ √[225 + 22500] ≈ **151km**

### 问题诊断

**初始距离约250km，在200-400km之间**，应该会触发INTERCEPT状态，但可能存在以下问题：

1. **预警机信息可能未正确初始化**
   - `has_awacs_info`可能为False
   - 需要检查`Picture`和`MockAwacsDataSource`是否正常工作

2. **StateContext可能未正确构建**
   - `min_threat_distance`可能未正确计算
   - 需要检查`step()`方法中的状态机更新逻辑

3. **状态机可能卡在PATROL状态**
   - 由于某些条件不满足，无法转换到INTERCEPT

## 问题3：调试输出已添加

✅ **已完成**：添加了以下调试输出来诊断状态机问题：

### 1. 状态机调试输出（每6秒）

在`cap_task.py::step()`方法中添加（约第1541行）：

```python
# 🔥 状态机调试输出（每30步=6秒输出一次）
if self.step_count % 30 == 0:
    log.info(f"[状态机] 当前状态={cap_state.value} | "
             f"最近威胁距离={ctx.min_threat_distance:.1f}km | "
             f"预警信息={'有' if ctx.has_awacs_info else '无'} | "
             f"时间={current_time:.1f}s")
```

### 2. AWACS数据源调试输出（每6秒）

在`cap_task.py::_update_awacs_data()`方法中添加（约第1755行）：

```python
# 🔥 AWACS数据源调试输出（每30步=6秒输出一次）
if self.step_count % 30 == 0:
    tracks = self.awacs.get_tracks()
    log.info(f"[AWACS数据] 目标数={len(targets)} | 探测航迹数={len(tracks)} | "
             f"我方位置数={len(friendly_positions)}")
    if tracks:
        for track_id, track in list(tracks.items())[:2]:  # 只显示前2个
            pos = track.position if hasattr(track, 'position') else None
            if pos:
                log.info(f"  {track_id}: 位置=({pos[0]:.1f}, {pos[1]:.1f})km")
```

### 3. 状态上下文调试输出（每6秒）

在`cap_task.py::_calculate_cap_context()`方法中添加（约第1785行）：

```python
# 🔥 AWACS信息调试（每30步输出一次）
if self.step_count % 30 == 0:
    log.info(f"[状态上下文] AWACS航迹数={len(awacs_tracks)} | "
             f"has_awacs_info={has_awacs_info} | "
             f"最近距离={min_distance:.1f}km")
```

## 预期调试输出

运行仿真后，应该每6秒看到类似输出：

```
[AWACS数据] 目标数=4 | 探测航迹数=4 | 我方位置数=4
  B0100: 位置=(40.5, 250.3)km
  B0200: 位置=(80.2, 248.7)km
[状态上下文] AWACS航迹数=4 | has_awacs_info=True | 最近距离=252.3km
[状态机] 当前状态=INTERCEPT | 最近威胁距离=252.3km | 预警信息=有 | 时间=6.0s
```

或者如果状态机卡在PATROL：

```
[AWACS数据] 目标数=4 | 探测航迹数=0 | 我方位置数=4
[状态上下文] AWACS航迹数=0 | has_awacs_info=False | 最近距离=252.3km
[状态机] 当前状态=PATROL | 最近威胁距离=252.3km | 预警信息=无 | 时间=6.0s
```

## 下一步行动

1. ✅ 雷达调试输出已注释（已完成）
2. ✅ 状态机调试输出已添加（已完成）
3. ✅ AWACS数据调试已添加（已完成）
4. ✅ 状态上下文调试已添加（已完成）

5. 🔄 **运行仿真验证**
   ```bash
   cd scripts/tacticalProject
   python cap/run_cap_simulation.py
   ```

6. 🔄 **根据调试输出诊断问题**
   - 如果`has_awacs_info=False`：检查AWACS数据源初始化
   - 如果`has_awacs_info=True`但状态仍为PATROL：检查距离阈值
   - 如果状态为INTERCEPT：应该能看到7层架构输出

7. 🔄 **可能的修复方案**
   - 如果AWACS数据源未初始化：检查`reset()`方法
   - 如果距离计算错误：检查坐标转换逻辑
   - 如果状态转换逻辑错误：调整阈值或滞后参数

## 初始距离总结

根据配置文件：
- **敌我初始距离**：约 **250km**（编队中心到敌方）
- **最近距离**：约 **151km**（A0200到B0100）
- **理论状态**：
  - 如果编队中心距离252km且has_awacs_info=True → **INTERCEPT**
  - 如果最近距离151km < 200km → **ENGAGE**（跳过INTERCEPT）
- **实际状态**：需要通过日志确认

**关键发现**：如果最近距离151km < 200km，则会直接进入ENGAGE状态，跳过INTERCEPT！
这可能是为什么没有看到7层架构输出的原因。

## 建议的配置调整

如果希望确保进入INTERCEPT状态以验证7层架构，可以调整敌方初始位置：

在`patrol_config.yaml`中，将敌方Y坐标从250km调整为300km：

```yaml
enemy:
  spawn_y: 300.0  # 从250km调整为300km
```

这样初始距离约为300km，确保在200-400km范围内，会触发INTERCEPT状态。
