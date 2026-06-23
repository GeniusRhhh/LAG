# 增强调试输出 - 诊断拦截动作问题

## 问题描述

根据用户日志，系统正确进入INTERCEPT状态，AWACS有4个航迹，但飞机表现为绕圈巡逻而非前出拦截。

## 症状分析

从日志可以看到：
```
[21:43:34] [状态机] 当前状态=INTERCEPT | 最近威胁距离=253.7km | 预警信息=有 | 时间=6.0s
[21:43:47]   A0100: 纬度=60.1459° 航向=285° (x=67.1,y=16.2) [北(热段)]
[21:43:47]   A0200: 纬度=60.0517° 航向=103° (x=68.7,y=5.7) [南(冷段)]
```

**问题**：
- 状态 = INTERCEPT ✅
- AWACS有数据 ✅
- 但飞机航向 = 285°和103°（应该是~0°向北）❌
- 飞机位置标签显示"[北(热段)]"和"[南(冷段)]"，这是巡逻状态的标签 ❌

## 根本原因假设

`_get_intercept_action()`可能没有被调用，或者被调用后因为`target_positions_raw`为空而回退到巡逻动作。

### 可能的原因

1. **Picture中没有威胁航迹**
   - AWACS有航迹，但Picture.get_all_threats()返回空列表
   - 原因：AWACS航迹可能没有正确融合到Picture中

2. **航迹融合时机问题**
   - `_update_picture_with_fusion()`可能在`get_action()`之后才调用
   - 导致`get_action()`时Picture还是空的

3. **航迹属性问题**
   - FusedTrack对象可能缺少`x`或`y`属性
   - 导致`target_positions_raw`为空

## 已添加的增强调试

### 修改1：每步输出调试信息（仅A0100）

**位置**：`cap_task.py::_get_intercept_action()` 第800-850行

**修改前**：
```python
if self.step_count % 30 == 0 and agent_id == 'A0100':
    log.info(f"[DEBUG] _get_intercept_action被调用...")
```

**修改后**：
```python
if agent_id == 'A0100':
    log.info(f"[DEBUG] _get_intercept_action被调用 | agent={agent_id} | step={self.step_count}")
```

**效果**：现在每一步都会输出，而不是每30步。

### 修改2：详细输出Picture威胁信息

**新增代码**：
```python
if agent_id == 'A0100':
    log.info(f"[DEBUG] AWACS航迹数={len(awacs_tracks)} | "
             f"Picture威胁数={len(all_threats)} | "
             f"target_positions_raw数={len(target_positions_raw)}")
    if len(all_threats) > 0:
        log.info(f"[DEBUG] Picture第一个威胁: track_id={all_threats[0].track_id} "
                 f"x={all_threats[0].x:.1f} y={all_threats[0].y:.1f} "
                 f"has_x={hasattr(all_threats[0], 'x')} has_y={hasattr(all_threats[0], 'y')}")
```

**效果**：
- 显示AWACS航迹数
- 显示Picture中的威胁数
- 显示提取到的目标位置数
- 如果有威胁，显示第一个威胁的详细信息

### 修改3：每步输出返回动作

**修改后**：
```python
if agent_id == 'A0100':
    log.info(f"[DEBUG] _get_intercept_action返回 | 当前航向={current_hdg:.1f}° | "
             f"目标航向={target_hdg:.1f}° | 航向差={diff:.1f}° | "
             f"返回动作=(7, {hdg_cmd}, {spd_cmd})")
```

**效果**：每步都显示返回的动作命令。

## 预期调试输出

### 场景1：正常情况（Picture有威胁）

```
[DEBUG] _get_intercept_action被调用 | agent=A0100 | step=30
[DEBUG] AWACS航迹数=4 | Picture威胁数=4 | target_positions_raw数=4
[DEBUG] Picture第一个威胁: track_id=B0100 x=56.1 y=268.0 has_x=True has_y=True
[层1-信息] 预警信息=有 | 目标数=4 | 目标丢失=否
[层2-估计] IMM-EKF已更新 4 个目标状态
...
[DEBUG] _get_intercept_action返回 | 当前航向=285.0° | 目标航向=5.2° | 航向差=-279.8° | 返回动作=(7, 2, 3)
```

### 场景2：Picture为空（问题所在）

```
[DEBUG] _get_intercept_action被调用 | agent=A0100 | step=30
[DEBUG] AWACS航迹数=4 | Picture威胁数=0 | target_positions_raw数=0
[DEBUG] 无目标信息，回退到巡逻动作
```

**诊断**：AWACS有4个航迹，但Picture威胁数=0，说明航迹融合有问题。

### 场景3：Picture有威胁但无x/y属性

```
[DEBUG] _get_intercept_action被调用 | agent=A0100 | step=30
[DEBUG] AWACS航迹数=4 | Picture威胁数=4 | target_positions_raw数=0
[DEBUG] Picture第一个威胁: track_id=B0100 x=56.1 y=268.0 has_x=False has_y=False
[DEBUG] 无目标信息，回退到巡逻动作
```

**诊断**：Picture有威胁，但FusedTrack对象缺少x/y属性。

## 运行测试

```bash
cd scripts/tacticalProject
python cap/run_cap_simulation.py
```

## 根据输出诊断

### 如果看到"_get_intercept_action被调用"

说明方法被正确调用，继续检查：

1. **Picture威胁数=0**
   - 问题：航迹融合未执行或失败
   - 解决：检查`_update_picture_with_fusion()`调用时机
   - 检查：`step()`方法中是否在`get_action()`之前调用融合

2. **Picture威胁数>0但target_positions_raw数=0**
   - 问题：FusedTrack对象缺少x/y属性
   - 解决：检查TrackFusion.fuse()返回的对象结构
   - 检查：FusedTrack是否正确继承Track的x/y属性

3. **target_positions_raw数>0**
   - 说明数据流正常，继续检查航向计算
   - 查看"目标航向"是否合理（应该指向敌方）

### 如果没有看到"_get_intercept_action被调用"

说明方法根本没被调用，检查：

1. **get_action()路由逻辑**
   - 确认`cap_state == CAPState.INTERCEPT`
   - 确认没有其他条件阻止调用

2. **状态机更新时机**
   - 确认状态机在`get_action()`之前更新

## 下一步行动

1. ✅ 增强调试输出已添加
2. 🔄 运行仿真，观察新的调试输出
3. 🔄 根据输出确定根本原因：
   - Picture为空 → 修复航迹融合
   - 属性缺失 → 修复FusedTrack结构
   - 方法未调用 → 修复路由逻辑
4. 🔄 实施针对性修复

## 总结

通过增强的调试输出，我们现在可以精确定位问题：
- 方法是否被调用
- Picture中是否有威胁
- 威胁对象是否有正确的属性
- 返回的动作命令是什么

这将帮助我们快速找到为什么飞机在INTERCEPT状态下仍然执行巡逻动作的根本原因。
