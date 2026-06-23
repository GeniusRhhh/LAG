# 调试输出添加完成报告

## 执行摘要

已成功添加状态机和AWACS数据源的调试输出，用于诊断为什么没有看到7层架构输出。所有调试输出每6秒（30步）输出一次，避免日志过载。

## 添加的调试输出

### 1. 状态机调试输出

**位置**：`cap_task.py::step()` 方法，约第1541行

**代码**：
```python
# 🔥 状态机调试输出（每30步=6秒输出一次）
if self.step_count % 30 == 0:
    log.info(f"[状态机] 当前状态={cap_state.value} | "
             f"最近威胁距离={ctx.min_threat_distance:.1f}km | "
             f"预警信息={'有' if ctx.has_awacs_info else '无'} | "
             f"时间={current_time:.1f}s")
```

**输出内容**：
- 当前CAP状态（PATROL/INTERCEPT/ENGAGE/EVADE/RTB）
- 最近威胁距离（km）
- 预警机信息状态（有/无）
- 当前仿真时间（秒）

**目的**：确认状态机是否正确转换到INTERCEPT状态

### 2. AWACS数据源调试输出

**位置**：`cap_task.py::_update_awacs_data()` 方法，约第1755行

**代码**：
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
elif self.step_count % 30 == 0:
    log.warning(f"⚠️ [AWACS数据] 无敌机目标")
```

**输出内容**：
- 真实敌机目标数
- AWACS探测到的航迹数
- 我方飞机位置数
- 前2个航迹的位置信息

**目的**：确认AWACS数据源是否正常工作，是否生成航迹

### 3. 状态上下文调试输出

**位置**：`cap_task.py::_calculate_cap_context()` 方法，约第1785行

**代码**：
```python
# 🔥 AWACS信息调试（每30步输出一次）
if self.step_count % 30 == 0:
    log.info(f"[状态上下文] AWACS航迹数={len(awacs_tracks)} | "
             f"has_awacs_info={has_awacs_info} | "
             f"最近距离={min_distance:.1f}km")
```

**输出内容**：
- AWACS航迹数量
- has_awacs_info标志状态
- 计算出的最近威胁距离

**目的**：确认StateContext是否正确构建，has_awacs_info是否正确设置

## 预期输出示例

### 场景1：正常进入INTERCEPT状态

```
[AWACS数据] 目标数=4 | 探测航迹数=4 | 我方位置数=4
  B0100: 位置=(40.5, 250.3)km
  B0200: 位置=(80.2, 248.7)km
[状态上下文] AWACS航迹数=4 | has_awacs_info=True | 最近距离=252.3km
[状态机] 当前状态=INTERCEPT | 最近威胁距离=252.3km | 预警信息=有 | 时间=6.0s
================================================================================
[算法0] 协同探测7层架构 | 时间=6.0s 步数=30
[层1-信息] 预警信息=有 | 目标数=4 | 目标丢失=否
[层2-估计] IMM-EKF已更新 4 个目标状态
...
================================================================================
```

### 场景2：状态机卡在PATROL（AWACS无数据）

```
[AWACS数据] 目标数=4 | 探测航迹数=0 | 我方位置数=4
[状态上下文] AWACS航迹数=0 | has_awacs_info=False | 最近距离=252.3km
[状态机] 当前状态=PATROL | 最近威胁距离=252.3km | 预警信息=无 | 时间=6.0s
```

**诊断**：AWACS数据源未生成航迹，需要检查`MockAwacsDataSource`初始化

### 场景3：直接进入ENGAGE状态（跳过INTERCEPT）

```
[AWACS数据] 目标数=4 | 探测航迹数=4 | 我方位置数=4
  B0100: 位置=(40.5, 150.3)km
  B0200: 位置=(80.2, 148.7)km
[状态上下文] AWACS航迹数=4 | has_awacs_info=True | 最近距离=151.2km
[状态机] 当前状态=ENGAGE | 最近威胁距离=151.2km | 预警信息=有 | 时间=6.0s
```

**诊断**：初始距离<200km，直接进入ENGAGE状态，跳过INTERCEPT。需要调整敌方初始位置。

## 诊断流程

### 步骤1：运行仿真

```bash
cd scripts/tacticalProject
python cap/run_cap_simulation.py
```

### 步骤2：检查日志输出

观察前30秒（150步）的日志，重点关注：

1. **AWACS数据**：
   - 目标数是否=4？
   - 探测航迹数是否>0？
   - 如果航迹数=0，AWACS数据源有问题

2. **状态上下文**：
   - has_awacs_info是否=True？
   - 最近距离是否在200-400km之间？
   - 如果has_awacs_info=False，检查AWACS数据源

3. **状态机**：
   - 当前状态是什么？
   - 如果是PATROL且has_awacs_info=True，状态转换逻辑有问题
   - 如果是ENGAGE，初始距离<200km，需要调整配置
   - 如果是INTERCEPT，应该能看到7层架构输出

### 步骤3：根据诊断结果采取行动

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| 探测航迹数=0 | AWACS数据源未初始化 | 检查`reset()`方法，确认`self.awacs`初始化 |
| has_awacs_info=False | StateContext构建错误 | 检查`_calculate_cap_context()`逻辑 |
| 状态=PATROL且has_awacs_info=True | 状态转换逻辑错误 | 检查`_should_intercept()`条件 |
| 状态=ENGAGE | 初始距离<200km | 调整`patrol_config.yaml`中敌方spawn_y |
| 状态=INTERCEPT但无7层输出 | `_get_intercept_action()`未调用 | 检查`get_action()`路由逻辑 |

## 配置调整建议

如果希望确保进入INTERCEPT状态以验证7层架构，建议调整敌方初始位置：

**文件**：`scripts/tacticalProject/cap/config/patrol_config.yaml`

**修改**：
```yaml
enemy:
  spawn_y: 300.0  # 从250km调整为300km，确保初始距离在200-400km之间
```

**效果**：
- 初始距离约300km
- 确保在INTERCEPT状态范围内（200-400km）
- 避免直接进入ENGAGE状态

## 验证清单

- [x] 状态机调试输出已添加
- [x] AWACS数据调试输出已添加
- [x] 状态上下文调试输出已添加
- [x] 调试输出频率设置为每6秒（避免日志过载）
- [x] debug-findings.md已更新
- [ ] 运行仿真验证
- [ ] 根据日志诊断问题
- [ ] 必要时调整配置或修复代码

## 下一步行动

1. **立即运行仿真**，观察调试输出
2. **根据日志诊断**状态机和AWACS数据源状态
3. **如果状态=INTERCEPT**，应该能看到7层架构输出
4. **如果状态≠INTERCEPT**，根据诊断流程修复问题
5. **如果需要**，调整敌方初始位置确保进入INTERCEPT状态

## 总结

✅ 所有调试输出已添加完成  
✅ 诊断流程已明确  
✅ 解决方案已准备  

现在可以运行仿真，通过调试日志快速定位问题根源。
