# ACTIONS_CATALOG（项目内“动作/机动/指令”清单）

> 目的：把项目里“已经定义/设计/实现”的动作入口全部列出来，写清楚**参数是什么**、**最终会输出什么三元组指令**、以及**会导致什么行为**，便于逐条审计。

## 0. 动作输出统一格式

项目里绝大多数动作最终都会落到同一种高层离散动作三元组：

- `action = (altitude_cmd_id, heading_cmd_id, velocity_cmd_id)`
- 默认平稳飞行：`(7, 8, 3)`（保持高度/航向/速度）

这些离散索引通过 `scripts/tacticalProject/tactical_utils.py` 的映射规则解释为“应该爬升/下降多少、左/右转多少、加/减速多少”的**离散意图**，然后由下层控制器/环境执行。

## 1. 离散指令空间（索引→语义）

**来源文件**：`scripts/tacticalProject/tactical_utils.py`

### 1.1 航向指令 `heading_cmd_id`（0–16）

函数：`TacticalUtils.convert_heading_to_index(heading_diff_rad)`

- 输入参数：
  - `heading_diff_rad`：目标航向差（弧度），内部会转换为度 `heading_diff_deg`
- 输出：`heading_cmd_id`（0–16），按“航向差区间”离散

| heading_cmd_id | 对应的 heading_diff_deg 区间（度） | 语义（大致） |
|---:|---|---|
| 0 | `< -120` | 大幅左转 |
| 1 | `[-120, -90)` | 大幅左转 |
| 2 | `[-90, -60)` | 大幅左转 |
| 3 | `[-60, -45)` | 中/大左转 |
| 4 | `[-45, -30)` | 中左转 |
| 5 | `[-30, -20)` | 小左转 |
| 6 | `[-20, -10)` | 小左转 |
| 7 | `[-10, -5)` | 微左转 |
| 8 | `[-5, 5)` | 保持航向 |
| 9 | `[5, 10)` | 微右转 |
| 10 | `[10, 20)` | 小右转 |
| 11 | `[20, 30)` | 小右转 |
| 12 | `[30, 45)` | 中右转 |
| 13 | `[45, 60)` | 中/大右转 |
| 14 | `[60, 90)` | 大右转 |
| 15 | `[90, 120)` | 大右转 |
| 16 | `>= 120` | 大幅右转 |

### 1.2 高度指令 `altitude_cmd_id`（0–12）

函数：`TacticalUtils.convert_altitude_to_index(altitude_cmd_value_m)`

- 输入参数：
  - `altitude_cmd_value_m`：期望高度变化（米，正=爬升，负=下降）
- 输出：`altitude_cmd_id`（0–12），按“高度变化区间”离散

| altitude_cmd_id | altitude_cmd_value_m 区间（米） | 语义（大致） |
|---:|---|---|
| 0 | `< -1000` | 大幅下降 |
| 2 | `[-1000, -500)` | 大幅下降 |
| 3 | `[-500, -300)` | 中幅下降 |
| 4 | `[-300, -150)` | 中幅下降 |
| 5 | `[-150, -100)` | 小幅下降 |
| 6 | `[-100, -50)` | 小幅下降 |
| 7 | `[-50, 50)` | 保持高度 |
| 8 | `[50, 100)` | 小幅爬升 |
| 9 | `[100, 150)` | 小幅爬升 |
| 10 | `[150, 300)` | 小/中爬升 |
| 11 | `[300, 500)` | 中幅爬升 |
| 12 | `>= 500` | 大幅爬升 |

> 注：`altitude_cmd_id=1` 在该函数里不会返回（历史遗留/兼容空间）。

### 1.3 速度指令 `velocity_cmd_id`（0–6）

函数：`TacticalUtils.convert_velocity_to_index(velocity_cmd_value_mps)`

- 输入参数：
  - `velocity_cmd_value_mps`：期望速度变化（m/s，正=加速，负=减速）
- 输出：`velocity_cmd_id`（0–6）

| velocity_cmd_id | velocity_cmd_value_mps 区间（m/s） | 语义（大致） |
|---:|---|---|
| 0 | `< -100` | 大幅减速 |
| 1 | `[-100, -50)` | 大幅减速 |
| 2 | `[-50, -20)` | 小幅减速 |
| 3 | `[-20, 20)` | 保持速度 |
| 4 | `[20, 50)` | 小幅加速 |
| 5 | `[50, 100)` | 大幅加速 |
| 6 | `>= 100` | 大幅加速 |

## 2. TacticalTask 提供的“动作委托/封装”

**来源文件**：`scripts/tacticalProject/tactical_task.py`

这些方法本质是把 TacticalExecutor / ManeuverLibrary 的动作统一为：返回离散三元组。

- `_maintain_heading_precise(env, agent_id, target_heading_deg, duration=10.0)`
  - 委托到 `scripts/tacticalProject/maneuver_library.py::ManeuverLibrary.maintain_heading_precise`
  - 行为：计算当前航向与目标航向差，若差值>3°则用 `convert_heading_to_index()` 给出修正航向索引

- `_execute_short_skate_precise(env, agent_id, current_time, skate_direction=None)`
  - 委托到 `ManeuverLibrary.execute_short_skate_precise`

- `_execute_tactical_crank(env, agent_id, direction, climb=False)`
  - 委托到 `ManeuverLibrary.execute_tactical_crank`

- `_establish_rear_formation(env, agent_id, current_time)` / `_maintain_rear_formation(env, agent_id)`
  - 委托到 `ManeuverLibrary.establish_rear_formation/maintain_rear_formation`

- `_turn_to_heading(env, agent_id, target_heading, speed_cmd=3, altitude_cmd=7)`
  - 委托到 `TacticalUtils.turn_to_heading`（用于把“目标航向”转成 `(alt_cmd, hdg_cmd, vel_cmd)`）

## 3. 机动库（我方）——`scripts/tacticalProject/maneuver_library.py`

类：`ManeuverLibrary(tactical_task)`

### 3.1 `maintain_heading_precise(env, agent_id, target_heading_deg, duration=10.0)`

- 参数：
  - `target_heading_deg`：目标航向（度）
  - `duration`：目前主要用于调用侧语义（函数内部并未用它做状态机）
- 输出：
  - 默认 `alt=7, vel=3`
  - 若航向误差 `> 3°`，则 `heading_cmd_id = convert_heading_to_index(heading_diff_rad)`
- 行为：朝目标航向“逐步”修正；误差在 3° 内返回保持航向 `8`

### 3.2 `execute_short_skate_precise(env, agent_id, current_time, direction='auto')`

- 参数：
  - `direction`：`'auto' | 'left' | 'right'`
    - `'auto'` 会参考 `task.selected_tactic` 与是否长/僚机，决定 crank 与 turn-cold 的方向
- 输出：三段式状态机（三元组随阶段变化）
  - 阶段1 `crank`：小角度 crank（用 `6/10/8` 做温和转向），`alt=7, vel=3`
  - 阶段2 `turn_cold`：大角度转向（同样用 `6/10/8`），`alt=7, vel=3`
  - 阶段3 `escape`：先保持航向并 `vel=5` 加速一段时间，再收敛到返航基准航向（我方默认 180°）
- 关键内部参数：
  - 对 `A0200`（僚机）与其他飞机，三段持续时间不同：`crank_duration / turn_cold_duration / escape_duration`

### 3.3 `execute_beam_maneuver(env, agent_id)`

- 参数：无（内部用 `task._get_enemy_bearing` 取敌机方位）
- 行为：
  - 若本机刚发射导弹且 `(<15s)`：进入“中制导保护”，直接返回 `(7,8,3)`
  - 否则计算 3/9 线目标航向（敌方方位 ±90°），并做“单次 5 秒转向 + 10 秒冷却”的循环节奏
  - 额外限制：单次目标转向被限制在 ±30° 以内（避免过激转向）

### 3.4 `execute_tactical_crank(env, agent_id, direction='left', climb=True)`

- 参数：
  - `direction`：`left|right`
  - `climb`：`True` 表示 crank+爬升（当前实现用较温和的爬升索引），`False` 表示 crank+（更温和的）下降/平飞
- 行为：两阶段状态机
  - `crank_climb`（约5秒）：输出 `(alt_cmd, hdg_cmd, vel_cmd=4)`
  - `level_off`（约2秒）：输出 `(7,8,3)` 并清理状态

### 3.5 `execute_tactical_climb(env, agent_id, direction='left')`

- 行为：三阶段（初始crank → 爬升+反向crank → 平飞回正），目标爬升约 `+1000m`

### 3.6 `execute_tactical_descent(env, agent_id, direction='left')`

- 行为：三阶段（初始crank → 下降+反向crank → 平飞回正），目标下降到 `max(current_alt-500, 3000)`

### 3.7 `execute_notch_back(env, agent_id, direction='left')`

- 行为：若在 `task.returning_agents` 返航集合里则跳过 Notch Back。
- 否则执行一段“beam+下降”的持续机动（当前实现持续 15 秒），结束后回到平飞。

### 3.8 `establish_rear_formation(env, agent_id, current_time)` / `maintain_rear_formation(env, agent_id)`

- 用途：前后攻击（FRONT_BACK）中，僚机建立并保持长机后方队形。
- 关键参数（内部阈值/状态机）：
  - 状态：`LATERAL_ALIGN → HEADING_CORRECT → LONGITUDINAL_ADJUST → FORMATION_HOLD`
  - 目标：
    - 横向误差容差：`TARGET_DY_TOLERANCE=200m`
    - 纵向距离区间（僚机在长机后方）：`TARGET_DX_MIN=-12000m` 到 `TARGET_DX_MAX=-7000m`
    - 航向容差：`TARGET_HEADING_TOLERANCE=10°`
  - 输出：通过调 `heading_cmd_id` 与 `velocity_cmd_id` 微调（高度也会按阈值修正）

## 4. 编队重置动作（AdvancedFormationResetManager）

**来源文件**：`scripts/tacticalProject/advanced_formation_reset_manager.py`

- `start_formation_reset(env, current_time)`
  - 行为：启动重置流程（phase=`turning`）

- `execute_formation_reset(env, agent_id) -> (alt_cmd, hdg_cmd, vel_cmd)`
  - 行为要点：从常见的“南向(180°)”重置到“北向(0°)”
    - 长机强制左转（逆时针），僚机强制右转（顺时针），让两机“向内侧靠拢”
  - 完成条件：双机都在 `heading_tolerance=5°` 内对准 0°，然后保持 0° 航向 `8s`
  - 高度：趋向 `target_alt=8000m`（用离散高度索引做温和修正）

- `is_formation_reset_complete(env, agents)`
  - 行为：分阶段检查（turning / holding_zero / complete）

## 5. 导弹动作（发射 gating + 执行）

**来源文件**：`scripts/tacticalProject/missile_manager.py`

### 5.1 `should_launch_missile(env, agent_id, current_phase, target_id=None, salvo_mode='single', is_second_attack=False) -> bool`

- 关键参数：
  - `current_phase`：`TacticalPhase`（不同阶段窗口不同）
  - `salvo_mode`：`single|double|salvo`
  - `is_second_attack`：二次进攻会改变发射窗口（更宽或更严格，取决于阶段/朝向策略）
- 关键约束（会直接拒绝发射）：
  - 单机发射上限：`MAX_MISSILES_PER_AIRCRAFT=4`
  - MAR 保护：距离 `<= 40km` 禁止新发射
  - 雷达/跟踪质量综合检查：`simulation/radar_manager.py::check_missile_launch_conditions`（失败则倾向拒绝或回退）
  - 朝向限制：根据雷达是否锁定、阶段、二次进攻等动态设置 `max_heading_error`（禁止“背对发射”）
- 阶段窗口（示例，最终以代码为准）：
  - `LR_TR`：首次 60–85km；二次进攻扩展 55–90km
  - `MTR_LR`：首次 40–80km；二次进攻扩展 40–90km
  - `TR_DOR`：40–75km
  - `DOR_DR`：40–65km

### 5.2 `execute_missile_launch(env, agent_id, target_id=None, salvo_mode='single') -> bool`

- 行为：真正扣减弹药并记录发射（注意：调用环境接口处当前是注释占位）
- 导弹ID：按发射序列生成（`A0100 -> A10001..A10004` 这种格式）
- 发射后写入：
  - `state_manager.record_missile_launch(...)`
  - `missile_tracks[missile_uid] = {launcher,target,launch_time,status}`

## 6. 战术模板（我方“组合动作”）入口

**来源文件**：`scripts/tacticalProject/tactical_executor.py`

这些不是“单一机动”，而是按阶段输出离散三元组的**组合策略**：

- `execute_drag_shoot(env, agent_id)`
- `execute_pincer_attack(env, agent_id)`
- `execute_high_low_attack(env, agent_id)`
- `execute_front_back(env, agent_id)`
- `execute_side_by_side(env, agent_id)`
- 防御/规避类：`execute_tactical_evasion`, `execute_tactical_turn`
- 二次进攻统一入口：`execute_unified_second_attack`

每个模板在不同阶段（NLT/MELD/MTR/LR/TR/DOR/DR/MAR）会输出不同机动/发射标记；详见：`scripts/tacticalProject/FIVE_TACTICS_IMPLEMENTATION_AUDIT.md`。

## 7. 敌方动作（UnifiedEnemyTacticalAI）

**来源文件**：`scripts/tacticalProject/unified_enemy_tactical_ai.py`

敌方同样输出 `(alt_cmd, hdg_cmd, vel_cmd)`，并定义了动作枚举 `ActionType`：

- 基础：`MAINTAIN_HEADING / TURN_LEFT / TURN_RIGHT / CLIMB / DESCEND / ACCELERATE / DECELERATE`
- 战术：`CRANK_LEFT / CRANK_RIGHT / NOTCH_MANEUVER / BEAM_MANEUVER`
- 规避：`DIVE_ESCAPE / CHAFF_FLARE_MANEUVER / SPIRAL_DIVE`
- 组合：`SHORT_SKATE / DEFENSIVE_SPLIT / AGGRESSIVE_APPROACH / RETURN_TO_BASE`

动作执行主入口：

- `get_enemy_action(env, agent_id, current_time) -> (alt_cmd, hdg_cmd, vel_cmd)`
- 内部：`_execute_action(..., action_type, ...)` 分派到 `_execute_*` / `_maintain_*` 系列实现；参数由 `ActionParameters` 生成。

---

如果你希望我把这个清单再“收敛成可打勾的表格”（每个动作一行：入口函数/参数/状态依赖/输出范围/触发条件/日志关键字），我可以继续把这份文档升级成严格的审计表格式。