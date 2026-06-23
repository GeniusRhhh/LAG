# TacticalProject 完整项目流程文档

## 目录
1. [仿真启动流程](#1-仿真启动流程)
2. [战术决策节点详细设计](#2-战术决策节点详细设计)
3. [飞机机动动作系统](#3-飞机机动动作系统)
4. [导弹系统设计](#4-导弹系统设计)
5. [雷达系统设计](#5-雷达系统设计)
6. [敌方AI系统](#6-敌方ai系统)
7. [编队重整系统](#7-编队重整系统)
8. [二次进攻系统](#8-二次进攻系统)
9. [问题分析](#9-问题分析)

---

## 1. 仿真启动流程

### 1.1 初始化阶段（run_simulation.py）

**步骤1：环境配置**
- 读取配置文件：`configs/tactical_bvr.yaml`
- 设置飞机类型：`MY_AIRCRAFT_TYPE = 'su27sk'`, `ENEMY_AIRCRAFT_TYPE = 'f16'`
- 设置环境变量：`FRIEND_BASELINE_MODEL = "SU27"`, `ENEMY_BASELINE_MODEL = "F16"`

**步骤2：创建环境**
- 创建 `ConfigurableMultipleCombatEnv` 实例
- 加载JSBSim模拟器
- 初始化4架飞机：A0100(长机), A0200(僚机), B0100(敌长机), B0200(敌僚机)
- 初始状态：
  - 我方：航向360°(北向)，高度6096m
  - 敌方：航向180°(南向)，高度6096m
  - 初始距离：约120km

**步骤3：创建战术系统**
- 创建 `TacticalDecisionManager`（根据tactic_type选择具体管理器）
- 创建 `TacticalTask` 实例
- 初始化所有子系统：
  - `TacticalStateManager`（单例模式）
  - `MissileManager`
  - `CompleteTacticalSystem`
  - `TacticalDecisionMaker`
  - `TacticalExecutor`
  - `ManeuverLibrary`
  - `NodeDecisionMaker`
  - `AdvancedFormationResetManager`
  - `UnifiedEnemyTacticalAI`
  - `UnifiedRadarManager`

**步骤4：环境重置**
- 调用 `env.reset()`
- 调用 `tactical_task.reset(env)`
- 初始化每架飞机的状态：
  - 初始航向、高度记录
  - RNN状态初始化
  - 导弹数量：4枚
  - 阶段：`TacticalPhase.NLT_MELD`

### 1.2 主循环（run_simulation.py:966-1099）

**每步执行流程：**
1. `env.step(dummy_actions)` - 环境步进
2. `env.render(mode="txt")` - 记录ACMI文件
3. 每60秒输出关键状态信息
4. 记录轨迹数据

**关键调用链：**
```
env.step() 
  → env.task.step() (tactical_task.py:980)
    → tactical_task.get_action(env, agent_id) (tactical_task.py:402)
      → tactical_task.normalize_action(env, agent_id, action) (tactical_task.py:1131)
        → tactical_task._use_lowlevel_policy() (tactical_task.py:1163)
```

---

## 2. 战术决策节点详细设计

### 2.1 节点距离定义

| 节点 | 距离(km) | 阶段枚举 | 说明 |
|------|---------|---------|------|
| NLT | 120 | NLT_MELD | 一级控制距离，策略与战术初始选择 |
| MELD | 100 | MELD_MTR | 二级控制距离，战术切换与编队调整 |
| MTR | 80 | MTR_LR | 三级节点，前往攻击占位点 |
| LR | 82 | LR_TR | 四级节点，完成中制导，发射导弹 |
| TR | 75 | TR_DOR | 中制导结束，规避敌方攻击 |
| DOR | 70 | DOR_DR | 根据威胁值分析下一进攻战术 |
| DR | 65 | DR_MAR | Beam机动转侧对，20s内决策 |
| MAR | 40 | BEYOND_MAR | 脱离机动，规避敌方攻击 |

**注意**：LR(82km) > MTR(80km)，这是为了创建缓冲区，防止阶段震荡。

### 2.2 阶段更新逻辑（tactical_task.py:507-594）

**函数**：`_update_phase(env, agent_id, current_time)`

**流程**：
1. 获取目标飞机ID（`get_target_with_fallback`）
2. 计算到目标距离
3. 检查返航状态（`_is_in_rtb_mode`）
4. **僚机滞后处理**：
   - 拖曳射击/前后攻击/并排射击：僚机距离+4km
   - 并排射击在TR-DOR和DOR-DR阶段：僚机距离+8km
5. 根据调整后的距离判断阶段（从大到小判断）
6. 更新阶段（`state_manager.set_agent_phase`）
7. 长机阶段作为全局阶段（`set_global_phase`）
8. 阶段切换时触发节点决策（`_node_decision_at_node`）

**问题点**：
- 僚机滞后可能导致僚机阶段与长机不一致
- 阶段切换可能频繁触发，导致决策重复执行

### 2.3 各节点详细设计

#### 2.3.1 NLT节点（120km）

**触发时机**：`TacticalPhase.NLT_MELD` 阶段

**决策内容**：
- **战术选择**：调用 `complete_tactical_system.select_tactic('NLT', ...)`
  - 输入：我方意图、态势评估、威胁评估
  - 输出：战术类型（DRAG_SHOOT/PINCER_ATTACK/HIGH_LOW_ATTACK/FRONT_BACK/SIDE_BY_SIDE）
  - 算法：决策表查询 + 适应度计算 + 加权随机选择
- **角色分配**：根据战术类型分配长机/僚机角色
- **编队形成**：并排射击队形（双机同高度并排）

**执行动作**：
- 长机/僚机：保持航向0°（北向），形成并排射击队形
- 雷达：开启搜索模式

**代码位置**：
- 决策：`tactical_task.py:583` → `_node_decision_at_node('NLT')`
- 执行：`tactical_executor.py` 各战术的 `execute_*` 方法

#### 2.3.2 MELD节点（100km）

**触发时机**：`TacticalPhase.MELD_MTR` 阶段

**决策内容**：
- **战术确认/切换**：调用 `complete_tactical_system.select_tactic('MELD', ...)`
  - 战术连续性检查：如果当前战术适应度仍高，保持战术
  - 否则切换到新战术
- **编队调整**：根据战术类型调整编队
  - 拖曳射击：长机前出，僚机保持
  - 钳形攻势：双机左右分离
  - 高低攻击：长机高空，僚机低空
  - 前后攻击：僚机建立后排队形（5.5km）
  - 并排射击：保持并排

**执行动作**：
- 根据战术类型执行编队调整机动
- 雷达：切换到跟踪模式

**代码位置**：
- 决策：`tactical_task.py:580` → `_node_decision_at_node('MELD')`
- 执行：`tactical_executor.py` 各战术的MELD_MTR阶段处理

#### 2.3.3 MTR节点（80km）

**触发时机**：`TacticalPhase.MTR_LR` 阶段

**决策内容**：
- **占位点计算**：根据战术类型计算攻击占位点
  - 拖曳射击：长机前出占位点P1，僚机保持后方
  - 钳形攻势：长机左翼占位点，僚机右翼占位点
  - 高低攻击：长机高空占位点，僚机低空占位点
  - 前后攻击：长机前出，僚机保持后排队形
  - 并排射击：双机并排占位点
- **威胁评估**：识别敌方意图，计算威胁值
- **机动决策**：前往占位点的机动方式

**执行动作**：
- 前往攻击占位点
- 保持队形（根据战术类型）

**代码位置**：
- 决策：`tactical_task.py:574` → `_node_decision_at_node('MTR')`
- 执行：`tactical_executor.py` 各战术的MTR_LR阶段处理

#### 2.3.4 LR节点（82km）

**触发时机**：`TacticalPhase.LR_TR` 阶段

**决策内容**：
- **Crank决策**：`_decide_at_lr(env, agent_id)` (tactical_task.py:707)
  - 计算敌机方位角
  - 计算航向差（`heading_diff = abs(enemy_bearing - current_heading)`）
  - 如果航向差 > 15°：选择Crank机动
  - 否则：选择平飞
  - **特殊处理**：HIGH_LOW_ATTACK僚机不执行Crank，保持直飞
- **导弹发射标记**：`missile_launched[agent_id] = True`
  - **问题点**：这里设置标记，但实际发射在`_handle_missile_launches`中检查
  - **问题**：如果此时航向不对，标记已设置，后续会尝试发射

**执行动作**：
- 发射导弹（如果满足条件）
- 执行Crank机动或平飞
- 保持雷达照射（中制导）

**代码位置**：
- 决策：`tactical_task.py:568` → `_decide_at_lr()` + `_node_decision_at_node('LR')`
- 执行：`tactical_executor.py` 各战术的LR_TR阶段处理
- 导弹发射：`tactical_task.py:1305` → `_handle_missile_launches()`

**关键问题**：
- **僚机背对发射问题**：在FRONT_BACK战术中，僚机在LR_TR阶段设置`missile_launched=True`，但此时僚机可能还在建立后排队形，航向不对（夹角121-173度）
- **解决方案**：应该在设置标记前检查航向，或者延迟到航向正确后再设置

#### 2.3.5 TR节点（75km）

**触发时机**：`TacticalPhase.TR_DOR` 阶段

**决策内容**：
- **威胁评估**：识别敌方意图，计算威胁值
- **撤退/规避决策**：
  - 如果威胁值高且敌方意图为攻击：执行撤退
  - 否则：执行规避机动
- **中制导结束**：终止雷达照射

**执行动作**：
- 执行Short Skate机动（规避）
- 或执行脱离机动（撤退）

**代码位置**：
- 决策：`tactical_task.py:591` → `_node_decision_at_node('TR')`
- 执行：`tactical_executor.py` 各战术的TR_DOR阶段处理

#### 2.3.6 DOR节点（70km）

**触发时机**：`TacticalPhase.DOR_DR` 阶段

**决策内容**：
- **威胁分析**：计算敌方威胁值
- **下一战术选择**：根据威胁值和态势选择下一阶段战术
- **编队调整**：有助于下一战术队形形成的机动

**执行动作**：
- 执行规避机动
- 同时进行编队调整

**代码位置**：
- 决策：`tactical_task.py:585` → `_node_decision_at_node('DOR')`
- 执行：`tactical_executor.py` 各战术的DOR_DR阶段处理

#### 2.3.7 DR节点（65km）

**触发时机**：`TacticalPhase.DR_MAR` 阶段

**决策内容**：
- **Beam机动**：转侧对敌方（三九线）
- **20秒决策窗口**：在20秒内根据态势计算下一阶段战术
  - 威胁评估
  - 敌方意图识别
  - 敌机存活数量
  - 我方导弹剩余
- **重新进攻/返航决策**：
  - 如果威胁值低且敌机数量少：选择重新进攻
  - 否则：选择返航

**执行动作**：
- 前20秒：执行Beam机动（侧对敌方）
- 20秒后：
  - 如果重新进攻：启动编队重整（`FORMATION_RESET`）
  - 如果返航：执行180°回转返航

**代码位置**：
- 决策：`tactical_task.py:587` → `_node_decision_at_node('DR')`
- 执行：`tactical_task.py:1548` → `_handle_dr_reengage_unified()` 或 `_handle_dr_retreat_unified()`
- Beam机动：`nodes/dr_node.py:64` → `_beam_maneuver()`

**关键问题**：
- DR决策窗口是20秒，但阶段可能在20秒内切换，导致决策未完成就进入MAR阶段

#### 2.3.8 MAR节点（40km）

**触发时机**：`TacticalPhase.BEYOND_MAR` 阶段

**决策内容**：
- **强制防御**：切换到防御战术（`TACTICAL_EVASION`）
- **脱离机动**：执行脱离机动，规避敌方攻击

**执行动作**：
- 执行脱离机动
- 如果已发射导弹：在LR'或TR'节点执行脱离

**代码位置**：
- 决策：`tactical_task.py:589` → `_node_decision_at_node('MAR')`
- 执行：`tactical_executor.py:execute_tactical_evasion()`

---

## 3. 飞机机动动作系统

### 3.1 低层控制模型

**我方飞机（SU27）**：
- 模型：`BaselineActor` (input_dim=12, use_mlp_actlayer=True)
- 模型文件：`su27_baseline_v1/run40/actor_990.pt`
- 输入：12维
  - [0-2]：高度/航向/速度指令（归一化）
  - [3-11]：飞机状态（高度、滚转、俯仰、速度等）

**敌方飞机（F16）**：
- 模型：`BaselineActor` (input_dim=12, use_mlp_actlayer=False)
- 模型文件：`baseline_model.pt`
- 输入：12维（同我方）

**输出**：4维动作
- [0]：副翼（aileron）：-1.0 ~ 1.0
- [1]：升降舵（elevator）：-1.0 ~ 1.0
- [2]：方向舵（rudder）：-1.0 ~ 1.0
- [3]：油门（throttle）：0.0 ~ 1.0

### 3.2 战术指令到低层控制转换

**函数**：`tactical_task.py:1163` → `_use_lowlevel_policy()`

**转换流程**：
1. 获取战术指令索引：
   - `altitude_cmd_id`：高度指令索引（0-14）
   - `heading_cmd_id`：航向指令索引（0-16）
   - `velocity_cmd_id`：速度指令索引（0-6）

2. 归一化指令值：
   - `norm_delta_altitude[altitude_cmd_id]`：-1.5km ~ +1.5km
   - `norm_delta_heading[heading_cmd_id]`：-π ~ +π
   - `norm_delta_velocity[velocity_cmd_id]`：-150m/s ~ +150m/s

3. 构造12维输入：
   - [0]：高度指令归一化值
   - [1]：航向指令归一化值
   - [2]：速度指令归一化值
   - [3-11]：飞机当前状态

4. 模型推理：
   - SU27：使用`su27_baseline_actor`
   - F16：使用`baseline_model`

5. 归一化输出：
   - `norm_act[0] = action_output[0] / 20 - 1.0`（副翼）
   - `norm_act[1] = action_output[1] / 20 - 1.0`（升降舵）
   - `norm_act[2] = action_output[2] / 20 - 1.0`（方向舵）
   - `norm_act[3] = action_output[3] / 58 + 0.4`（油门）

6. 高度保护（`tactical_task.py:1262`）：
   - 高度 < 1000m：全油门 + 全力拉升 + 改平（如果滚转角>60°）
   - 高度 < 1500m：强制爬升 + 全油门
   - 高度 < 3000m：优先爬升 + 提高油门

### 3.3 战术机动动作库（maneuver_library.py）

**主要机动**：

1. **保持航向**：`maintain_heading_precise(env, agent_id, target_heading)`
   - 计算航向差
   - 如果差值 < 容差：保持航向（指令8）
   - 否则：转向目标航向

2. **Crank机动**：`execute_tactical_crank(env, agent_id, direction, climb)`
   - 方向：'left' 或 'right'
   - 角度：±15°或±30°
   - 可选爬升

3. **Short Skate机动**：`execute_short_skate_precise(env, agent_id, current_time, direction)`
   - 方向：'left' 或 'right'
   - 角度：±30°
   - 持续时间：约10秒

4. **建立后排队形**：`establish_rear_formation(env, agent_id, current_time)`
   - 阶段1：横向对齐（dy < 200m）
   - 阶段2：航向回正（hdg_diff < 10°）
   - 阶段3：纵向调整（dx = -5.5km，速度匹配）
   - 阶段4：稳定区域（dx = -5.5km ± 500m，dy < 200m）

5. **保持后排队形**：`maintain_rear_formation(env, agent_id)`
   - 检查队形偏差
   - 如果偏差大：重新对齐
   - 否则：微调保持

### 3.4 各战术的机动动作

#### 3.4.1 拖曳射击（DRAG_SHOOT）

**长机**：
- NLT_MELD/MELD_MTR：保持航向0°
- MTR_LR：前往占位点P1
- LR_TR：保持直飞诱敌（不执行Crank）
- TR_DOR：Short Skate（左）

**僚机**：
- NLT_MELD/MELD_MTR：保持航向0°
- MTR_LR：前往占位点P2（后方）
- LR_TR：Crank机动（如果航向差>15°）
- TR_DOR：Short Skate（右）

#### 3.4.2 钳形攻势（PINCER_ATTACK）

**长机**：
- MELD_MTR：左转15°分离
- MTR_LR：回调到0°
- LR_TR：Crank机动
- TR_DOR：Short Skate（左）

**僚机**：
- MELD_MTR：右转15°分离
- MTR_LR：回调到0°
- LR_TR：Crank机动
- TR_DOR：Short Skate（右）

#### 3.4.3 高低攻击（HIGH_LOW_ATTACK）

**长机（高空）**：
- MELD_MTR：爬升到8000m
- MTR_LR：保持高空
- LR_TR：Crank机动
- TR_DOR：Short Skate

**僚机（低空）**：
- MELD_MTR：下降到5000m
- MTR_LR：保持低空
- LR_TR：**不执行Crank，保持直飞**（战术特定）
- TR_DOR：Short Skate

#### 3.4.4 前后攻击（FRONT_BACK）

**长机**：
- MELD_MTR：保持航向0°
- MTR_LR：前往占位点
- LR_TR：保持航向20°（不执行Crank）
- TR_DOR：Short Skate（左）

**僚机**：
- MELD_MTR：建立后排队形（5.5km）
  - **问题点**：建立队形需要时间，可能导致阶段滞后
- MTR_LR：保持后排队形
- LR_TR：**保持后排队形**（不执行Crank）
  - **问题点**：此时可能还在建立队形，航向不对，但设置了`missile_launched=True`
- TR_DOR：Short Skate（右）

**关键问题**：
- 僚机在LR_TR阶段设置`missile_launched=True`，但可能还在建立后排队形，航向不对（夹角121-173度）
- 应该延迟导弹发射标记，直到队形建立完成且航向正确

#### 3.4.5 并排射击（SIDE_BY_SIDE）

**长机/僚机**：
- NLT_MELD/MELD_MTR：保持并排
- MTR_LR：保持并排
- LR_TR：同时发射导弹
- TR_DOR：Short Skate（长机左，僚机右）

---

## 4. 导弹系统设计

### 4.1 导弹发射流程

**步骤1：设置发射标记**（tactical_executor.py）
- LR节点：`missile_launched[agent_id] = True`
- 位置：各战术的LR_TR阶段处理

**步骤2：发射检查**（tactical_task.py:1305 → `_handle_missile_launches()`）

**检查条件**：
1. **冷却时间**：`current_time - last_launch_time >= missile_cooldown`（默认5秒）
2. **导弹数量**：`aircraft.num_missiles > 0`
3. **目标存在**：目标飞机存在且存活
4. **航向检查**（tactical_task.py:1385）：
   - 计算目标方位：`TacticalUtils.calculate_bearing(my_aircraft, target)`
   - 计算航向差：`heading_error = abs(normalize_angle_diff(target_bearing - current_heading))`
   - **航向限制**：
     - 雷达锁定：60度
     - LR_TR/TR_DOR阶段（BEAM机动）：90度
     - 二次进攻：45度
     - 其他情况：60度
   - **问题**：如果航向差 > 限制，阻止发射，但标记仍为True，会持续尝试
5. **距离窗口**（missile_manager.py:108-145）：
   - LR阶段：60-85km（首次），55-90km（二次进攻）
   - MTR阶段：40-80km（首次），40-90km（二次进攻）
   - TR阶段：40-75km
   - DOR阶段：40-65km
   - MAR后：40-60km（需雷达锁定）
6. **雷达锁定检查**（missile_manager.py:96）：
   - 调用 `check_missile_launch_conditions(env, agent_id, target_id)`
   - 检查雷达模式、锁定状态、跟踪质量等

**步骤3：执行发射**（tactical_task.py:1443 → `_launch_missile()`）
1. 生成导弹ID：`A10001`, `A10002`, `A10003`, `A10004`
2. 创建导弹：`MissileSimulator.create(parent, target, uid)`
3. 添加到环境：`env.add_temp_simulator(missile)`
4. 更新状态：
   - `last_missile_launch_time[agent_id] = current_time`
   - `missiles_fired[agent_id] += 1`
   - `aircraft.num_missiles -= 1`
   - `missile_launched[agent_id] = False`

### 4.2 导弹限制

**每架飞机最多4枚导弹**：
- 检查：`aircraft_missile_counts[agent_id] < 4`
- 位置：`missile_manager.py:42-60`, `tactical_task.py:1465-1469`

### 4.3 导弹发射问题分析

**问题1：僚机背对发射**
- **现象**：A0200在夹角121-173度时尝试发射
- **原因**：
  1. 在LR_TR阶段设置`missile_launched=True`（tactical_executor.py:966）
  2. 但此时僚机可能还在建立后排队形，航向不对
  3. 发射检查时发现航向不对，阻止发射，但标记仍为True
  4. 每步都尝试发射，产生大量警告日志
- **解决方案**：
  1. 延迟设置发射标记，直到队形建立完成且航向正确
  2. 或者在发射检查失败后清除标记

**问题2：发射标记未清除**
- **现象**：发射被阻止后，标记仍为True，持续尝试
- **解决方案**：在发射检查失败后，如果航向严重不对（>120度），清除标记

---

## 5. 雷达系统设计

### 5.1 雷达模型

**我方雷达（APG-68(V)9）**：
- 最大探测距离：105km（战斗机），165km（大型目标）
- 最大跟踪距离：85km
- 最大锁定距离：70km
- 扫描周期：2秒
- 跟踪更新率：0.3秒（TWS模式）
- 锁定更新率：0.04秒（STT模式）

**敌方雷达（N001VE）**：
- 最大探测距离：90km
- 最大跟踪距离：70km
- 最大锁定距离：45km
- 扫描周期：3.5秒（机械扫描）
- 跟踪更新率：0.5秒（TWS模式）
- 锁定更新率：0.05秒（STT模式）

### 5.2 雷达状态管理（simulation/radar_manager.py）

**雷达状态枚举**：
- `SEARCH`：搜索模式
- `TRACK`：跟踪模式
- `LOCK`：锁定模式
- `STANDBY`：待机模式

**状态转换**：
- NLT节点：SEARCH模式
- MELD节点：切换到TRACK模式
- MTR节点：保持TRACK模式
- LR节点：切换到LOCK模式（发射导弹前）
- TR节点：保持LOCK模式（中制导）
- DOR/DR节点：根据情况切换

**更新函数**：`update_all_radars(env, current_time)`
- 每步调用一次
- 更新所有飞机的雷达状态
- 计算探测概率、跟踪质量等

### 5.3 RWR系统

**威胁等级**：
- 0：无威胁
- 1：雷达搜索
- 2：雷达跟踪
- 3：雷达锁定
- 4：导弹发射
- 5：导弹制导

**ECM系统**：
- 类型：噪声干扰、欺骗干扰、箔条、红外诱饵等
- 激活条件：检测到敌方雷达搜索（威胁等级1）
- 效果：降低被探测/锁定概率

---

## 6. 敌方AI系统

### 6.1 敌方AI流程（unified_enemy_tactical_ai.py）

**主函数**：`get_enemy_command_indices(env, agent_id, current_time)`

**流程**：
1. **意图识别**：识别我方意图（攻击/防御/中立）
2. **威胁评估**：计算威胁值
3. **战术模式选择**：
   - `AGGRESSIVE`：攻击模式
   - `DEFENSIVE`：防御模式
   - `NEUTRAL`：中性模式
4. **阶段更新**：`_update_tactical_phase()`
   - 根据距离判断阶段（5阶段系统）
5. **动作选择**：`_select_action()`
   - 根据阶段和模式选择动作
6. **动作执行**：`_execute_action()`
7. **全局安全检查**：`_apply_global_safety_check()`

### 6.2 敌方高度控制问题

**问题现象**：
- 敌方在每次仿真后缓慢下降高度直至坠毁
- 机头一直往上拉升，但高度仍在下降

**可能原因**：

1. **高度保护阈值设置**（unified_enemy_tactical_ai.py:1320）：
   - 紧急阈值：3000m
   - 警告阈值：6000m
   - 限制阈值：8000m
   - **问题**：如果高度在6000-8000m之间，只限制下降，但不强制爬升

2. **滚转角过大**（unified_enemy_tactical_ai.py:1334）：
   - 如果滚转角 > 60°，优先改平
   - 但改平过程中可能失去升力，导致下降

3. **失速问题**（unified_enemy_tactical_ai.py:1376）：
   - 如果速度 < 失速临界速度，需要压低机头恢复速度
   - 但压低机头会导致高度下降

4. **低层控制模型输出异常**：
   - F16 Baseline模型可能输出异常的高度指令
   - 或者高度指令被其他逻辑覆盖

5. **高度补偿机制未生效**：
   - 日志显示"限制下降"，但可能补偿不够
   - 或者补偿被后续逻辑覆盖

**解决方案**：
1. 提高高度保护阈值
2. 增强滚转角保护
3. 检查低层控制模型输出
4. 增强高度补偿机制

---

## 7. 编队重整系统

### 7.1 编队重整流程（advanced_formation_reset_manager.py）

**触发时机**：DR节点决策重新进攻

**流程**：
1. **启动**：`start_formation_reset(env, current_time)`
   - 设置`active = True`
   - 设置`phase = 'turning'`
   - 目标航向：0°（北向）

2. **执行**：`execute_formation_reset(env, agent_id)`
   - **长机**：左转（逆时针）到0°
     - 计算逆时针距离：`counterclockwise_diff = current_heading`
     - 如果 < 15°：保持航向
     - 否则：左转指令（5或6）
   - **僚机**：右转（顺时针）到0°
     - 计算顺时针距离：`clockwise_diff = (360 - current_heading) % 360`
     - 如果 < 15°：保持航向
     - 否则：右转指令（10或11）

3. **完成检查**：`is_formation_reset_complete(env, agents)`
   - 检查所有飞机航向是否在0°±15°范围内
   - 如果完成：设置`phase = 'complete'`，`active = False`

4. **切换到二次进攻**：
   - 完成编队重整后，切换到`UNIFIED_SECOND_ATTACK`战术

### 7.2 编队重整问题

**问题1：未回到0度航向**
- **现象**：编队重整后未真正回到0度
- **原因**：
  1. 完成条件检查不严格（容差30度）
  2. 或者完成检查后立即切换战术，未等待航向稳定
- **解决方案**：
  1. 严格完成条件（容差15度）
  2. 确保所有飞机都到达0度后再切换

**问题2：朝向目标动态调整**
- **需求**：编队重整后应该先回到0度，之后再动态调整朝向目标
- **当前实现**：编队重整完成后立即切换到二次进攻，可能直接朝向目标
- **解决方案**：
  1. 编队重整完成后，先保持0度航向一段时间
  2. 然后再动态调整朝向目标

---

## 8. 二次进攻系统

### 8.1 二次进攻流程

**步骤1：DR节点决策**（tactical_task.py:1548）
- 决策是否重新进攻
- 如果重新进攻：调用`_handle_dr_reengage_unified()`
  - 设置`is_second_attack = True`
  - 启动编队重整：`formation_reset_manager.start_formation_reset()`
  - 切换到`FORMATION_RESET`战术

**步骤2：编队重整**（advanced_formation_reset_manager.py）
- 执行编队重整机动
- 目标：回到0度航向

**步骤3：切换到二次进攻**（tactical_task.py:1590）
- 编队重整完成后，切换到`UNIFIED_SECOND_ATTACK`战术
- 选择二次进攻战术：`_select_second_attack_tactic()`
  - 根据敌方意图和数量选择
  - 可选：DRAG_SHOOT, PINCER_ATTACK, HIGH_LOW_ATTACK

**步骤4：执行二次进攻**（tactical_executor.py:1114）
- 执行`execute_unified_second_attack()`
- 根据选择的战术执行相应动作
- 在MTR'/LR'节点发射导弹

### 8.2 二次进攻节点

**MTR'节点**（mtr_prime_node.py）：
- 触发时机：二次进攻的MTR_LR阶段
- 核心任务：前往攻击占位点
- 执行动作：根据二次进攻战术执行

**LR'节点**：
- 触发时机：二次进攻的LR_TR阶段
- 核心任务：发射导弹 + 中制导
- 执行动作：发射导弹，执行中制导

**TR'节点**：
- 触发时机：二次进攻的TR_DOR阶段
- 核心任务：中制导结束，规避
- 执行动作：执行规避机动

### 8.3 二次进攻问题

**问题1：编队重整未完成就切换**
- **现象**：编队重整未真正完成就切换到二次进攻
- **原因**：完成条件检查不严格
- **解决方案**：严格完成条件，确保航向到达0度

**问题2：动态调整时机不对**
- **需求**：先回到0度，之后再动态调整朝向目标
- **当前实现**：可能直接朝向目标
- **解决方案**：编队重整完成后，先保持0度一段时间，再动态调整

---

## 9. 问题分析

### 9.1 从日志发现的问题

#### 问题1：僚机背对发射导弹（none_simulation_20251229_152849.log）

**时间点**：约120秒（2分钟）

**现象**：
- A0200在夹角121-173度时尝试发射导弹
- 大量警告日志："⚠️ [A0200] 背对敌机发射被阻止: 夹角XXX° >90°"

**分析**：
1. A0200在LR_TR阶段（约120秒）设置`missile_launched=True`
2. 但此时A0200还在建立后排队形（FRONT_BACK战术）
3. 航向不对（夹角121-173度）
4. 发射检查阻止发射，但标记仍为True
5. 每步都尝试发射，产生大量警告

**根本原因**：
- **FRONT_BACK战术的僚机在LR_TR阶段设置发射标记，但此时可能还在建立后排队形，航向不对**
- **应该延迟设置发射标记，直到队形建立完成且航向正确**

**解决方案**：
1. 在设置发射标记前检查队形状态
2. 如果队形未建立完成，延迟设置
3. 或者在发射检查失败后清除标记

#### 问题2：僚机严重脱离队形

**现象**：
- "⚠️ [A0200]严重脱离队形 dx=-5.90km dy=-578m hdg=-32.3° → 重新对齐"
- "⚠️ [A0200]严重脱离队形 dx=-7.90km dy=772m hdg=32.0° → 重新对齐"

**分析**：
1. 后排队形目标：dx = -5.5km，dy < 200m
2. 实际：dx = -5.90km ~ -7.90km，dy = -578m ~ 772m
3. 说明队形建立逻辑有问题，或者被其他逻辑干扰

**可能原因**：
1. 阶段切换导致队形建立中断
2. 其他机动动作覆盖了队形建立
3. 队形建立逻辑本身有问题

#### 问题3：敌方大坡度改平

**现象**：
- "🛡️ [B0200] 大坡度改平: 滚转-87.3° 高度6458m"
- "🛡️ [B0200] 大坡度改平: 滚转-67.4° 高度6717m"

**分析**：
1. 滚转角过大（-87.3°, -67.4°）
2. 说明敌方AI的滚转角控制有问题
3. 可能导致升力损失，高度下降

**可能原因**：
1. 低层控制模型输出异常
2. 或者高度保护逻辑与滚转角控制冲突

#### 问题4：敌方限制下降但仍在下降

**现象**：
- "🛡️ [B0200] 限制下降: 高度6486m < 8000m，原始指令=1(变化-1000m)→修改为=5"
- 但高度仍在下降

**分析**：
1. 高度保护机制检测到下降指令，修改为小幅下降（指令5 = -100m）
2. 但可能：
   - 修改不够（-100m仍然下降）
   - 或者被后续逻辑覆盖
   - 或者低层控制模型输出异常

#### 问题5：阶段不一致

**现象**：
- 长机在DR_MAR阶段
- 僚机还在MTR_LR阶段
- 导致决策不同步

**分析**：
1. 僚机滞后机制导致阶段不一致
2. 但滞后可能过大，导致决策不同步

### 9.2 优先级冲突问题

#### 冲突1：FORMATION_RESET vs 其他战术

**问题**：编队重整期间可能被其他战术逻辑干扰

**当前实现**：
- `tactical_task.py:463`：FORMATION_RESET优先级最高
- 但可能在某些情况下被覆盖

**解决方案**：
1. 确保FORMATION_RESET优先级最高
2. 在FORMATION_RESET期间，禁止其他战术切换

#### 冲突2：导弹发射 vs 队形建立

**问题**：导弹发射标记在队形建立期间设置

**当前实现**：
- LR_TR阶段设置发射标记
- 但FRONT_BACK僚机可能还在建立队形

**解决方案**：
1. 延迟设置发射标记
2. 或者检查队形状态后再设置

#### 冲突3：高度保护 vs 战术指令

**问题**：高度保护可能被战术指令覆盖

**当前实现**：
- 高度保护在`_use_lowlevel_policy`中
- 但可能在战术执行器中被覆盖

**解决方案**：
1. 确保高度保护优先级最高
2. 在最终输出前再次检查

---

## 10. 需要修改的地方（待用户确认）

### 10.1 导弹发射逻辑

**问题**：僚机背对发射

**修改建议**：
1. 在设置发射标记前检查航向
2. 延迟设置发射标记，直到航向正确
3. 或者在发射检查失败后清除标记

### 10.2 编队重整逻辑

**问题**：未真正回到0度航向

**修改建议**：
1. 严格完成条件（容差15度）
2. 确保所有飞机都到达0度后再切换
3. 编队重整完成后，先保持0度一段时间，再动态调整朝向目标

### 10.3 敌方高度控制

**问题**：失速下降

**修改建议**：
1. 提高高度保护阈值
2. 增强滚转角保护
3. 检查低层控制模型输出
4. 增强高度补偿机制

### 10.4 队形建立逻辑

**问题**：僚机严重脱离队形

**修改建议**：
1. 检查队形建立逻辑
2. 确保队形建立不被其他逻辑干扰
3. 增加队形保持机制

### 10.5 阶段同步

**问题**：长机/僚机阶段不一致

**修改建议**：
1. 调整僚机滞后机制
2. 确保关键决策节点同步

---

## 11. 代码关键位置索引

### 11.1 主流程
- 仿真启动：`run_simulation.py:815`
- 主循环：`run_simulation.py:966`
- 任务步进：`tactical_task.py:980`
- 动作获取：`tactical_task.py:402`

### 11.2 阶段更新
- 阶段更新：`tactical_task.py:507`
- 阶段切换决策：`tactical_task.py:566`

### 11.3 节点决策
- NLT节点：`tactical_task.py:583`
- MELD节点：`tactical_task.py:580`
- MTR节点：`tactical_task.py:574`
- LR节点：`tactical_task.py:568` + `tactical_task.py:707`
- TR节点：`tactical_task.py:591`
- DOR节点：`tactical_task.py:585`
- DR节点：`tactical_task.py:587` + `tactical_task.py:1548`
- MAR节点：`tactical_task.py:589`

### 11.4 导弹系统
- 发射检查：`tactical_task.py:1305`
- 执行发射：`tactical_task.py:1443`
- 发射条件：`missile_manager.py:26`

### 11.5 战术执行
- 拖曳射击：`tactical_executor.py:111`
- 钳形攻势：`tactical_executor.py:200`
- 高低攻击：`tactical_executor.py:725`
- 前后攻击：`tactical_executor.py:871`
- 并排射击：`tactical_executor.py:995`

### 11.6 编队重整
- 启动：`advanced_formation_reset_manager.py:29`
- 执行：`advanced_formation_reset_manager.py:42`
- 完成检查：`advanced_formation_reset_manager.py:119`

### 11.7 敌方AI
- 主函数：`unified_enemy_tactical_ai.py:2466`
- 高度保护：`unified_enemy_tactical_ai.py:1298`

---

## 12. 下一步工作

1. **用户对照文档检查设计**
2. **用户指出需要修改的地方**
3. **我根据用户反馈进行修改**

---

**文档版本**：v1.0
**创建时间**：2025-01-09
**最后更新**：2025-01-09










