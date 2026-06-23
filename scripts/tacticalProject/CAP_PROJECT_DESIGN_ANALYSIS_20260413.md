# CAP Project Design And Analysis

## 1. 文档目的

这份文档基于当前项目代码和近期日志，描述当前 CAP 仿真项目的实际设计、执行链路、敌我双方控制逻辑、与任务要求的匹配关系，以及已经确认存在的问题。

这不是需求文档，而是“当前实现说明 + 问题定位说明”。重点是回答下面几个问题：

- 当前项目实际上是怎么运行的。
- 哪些逻辑在 CAP 层，哪些逻辑在旧战术引擎层。
- 协同探测、协同跟踪、战术模板、接力制导、敌方波次 AI 分别怎么实现。
- 现有实现与任务场景要求相比，哪些已实现，哪些只是有框架，哪些存在偏差。
- 日志里出现的异常轨迹，根本问题到底落在哪一层。

分析依据主要来自以下代码路径：

- `scripts/tacticalProject/cap/`
- `scripts/tacticalProject/tactical_task_impl.py`
- `scripts/tacticalProject/tactical_task_action_mixin.py`
- `scripts/tacticalProject/tactical_executor_support_mixin.py`
- `scripts/tacticalProject/cap/tactics/tactical_bridge.py`
- `scripts/tacticalProject/cap/adapters/enemy_ai_adapter.py`
- `scripts/tacticalProject/cap/adapters/missile_adapter.py`

重点参考日志：

- `scripts/tacticalProject/cap_results/cap_20260413_105822.log`
- `scripts/tacticalProject/cap_results/cap_20260413_111035.log`


## 2. 任务要求基线

按你给出的任务场景，系统目标可归纳为：

- 红方 4 机保护红方高价值空中单元。
- 责任区按高风险、中风险、低风险划分。
- 20 分钟内不让敌方进入高风险区。
- 严格控制敌方进入中风险区数量。
- 红方损失不能超过 75%。
- 对远方目标一般不主动前出攻击。
- 需要在预警误差存在、甚至目标丢失的条件下，完成：
  - 协同探测
  - 协同跟踪
  - 意图识别
  - 协同攻击
  - 发射后持续制导
  - 接力制导
  - 规避和回转

任务中明确要求的战术模板包括：

- 低风险区：`Drag Shoot`、`Pincer Attack`、`High-Low Attack`、`Side-by-Side`
- 中风险区：`Front-Back`、`Tactical Evasion`、`Tactical Turn`
- 高风险区：`Tactical Evasion`、`Tactical Turn`


## 3. 当前项目的总体架构

当前项目不是一个单层控制器，而是“CAP 外层任务系统 + 图景/雷达/协同模块 + 旧 2v2 战术引擎桥接 + 敌方独立波次 AI + 导弹与仿真补偿器”的组合系统。

可以把它理解成 6 层：

1. 环境与飞机层
- JSBSim 环境负责 8 架飞机、低层飞行动力学、导弹临时模拟器、机体状态。

2. CAP 外层任务层
- `CAPTask` 是当前 4v4 顶层任务对象。
- 它维护 CAP 状态机、责任区、图景、预警、协同探测、协同交战、任务评估、日志等。

3. 图景与感知层
- 包含预警机输入、机载火控雷达扫描、航迹融合、风险区判定、最近威胁距离计算。

4. 我方战术层
- CAP 层会先做对子级目标和战术分配。
- 但真正的机动动作，很多时候不是 CAP 自己直接算，而是通过 `TacticalBridge` 把 4v4 问题拆成两个 2v2 子问题，交给旧的 `TacticalTask` 执行。

5. 敌方战术层
- 敌方主要由 `EnemyAIAdapter` 管理，按左右双机对子、波次状态、最近我方距离、编队恢复和 turn/release/hard line 逻辑机动。

6. 导弹与评估层
- `MissileAdapter` 负责导弹库存、发射、状态、统计。
- `MissionEvaluator` 负责按高风险区突破、我方损失等条件给出任务结果。


## 4. CAPTask 的初始化结构

`cap/cap_task_init_helpers.py` 中，`initialize_cap_task()` 会初始化以下核心组件：

- 坐标系统、FAOR、bullseye、编队参数、巡逻状态机
- `CooperativeDetection`
- `CooperativeEngagement`
- `MissionEvaluator`
- `TacticSelector`
- `CAPStateMachine`
- `Picture`
- `MockAwacsDataSource`
- `TrackFusion`
- `FormationGuidance`
- `VelocityCoordination`
- `CAPRadarManager`
- `EnemyAIAdapter`
- `MissileAdapter`
- `IntentAdapter`
- `TacticExecutorAdapter`
- `TacticalBridge`

同时初始化大量运行期缓存：

- `_tactic_assignments_by_agent`
- `_last_tactic_assignments_by_pair`
- `_pair_target_memory`
- `_stable_tracking_elapsed`
- `_guidance_verify`
- `_target_first_awacs_time`
- `_target_first_fcr_time`
- `_target_first_ready_time`
- `_zone_transition_history`

这说明当前系统不是一个轻量战术脚本，而是一个较重的状态缓存型系统。很多行为并非单步决策，而依赖“上一轮已锁定目标/节点/窗口”的持续状态。


## 5. 主循环执行顺序

`cap/cap_task_step_helpers.py` 的 `run_cap_step()` 是每个 step 的主流程。按顺序：

1. 更新 step 和当前时间。
2. 如果当前 CAP 状态在 `PATROL/INTERCEPT`，更新预警机数据。
3. 更新 `Picture`。
4. 计算 CAP 状态上下文 `StateContext`。
5. 运行 `CAPStateMachine.update()`，得到 `PATROL / INTERCEPT / ENGAGE / EVADE / RTB`。
6. 若在 `PATROL/INTERCEPT`，运行协同探测；否则只维护雷达跟踪。
7. 更新协同交战。
8. 若在 `INTERCEPT` 且进入 NLT 窗口，运行意图识别。
9. 若在 `ENGAGE` 且距离落在 `LR~MTR` 间，运行协同跟踪。
10. 更新导弹。
11. 更新任务评估。
12. 更新战术选择。
13. 周期性输出状态、battle snapshot、验证日志。

这条顺序很重要，因为它说明：

- 图景是先更新，再驱动状态机。
- 战术选择排在任务评估后。
- 真正的动作输出发生在 `get_action()` 中，而 `get_action()` 又依赖前面步骤已经写入的状态和分配。


## 6. CAP 状态机设计

`cap/cap_state_machine.py` 定义了五个外层状态：

- `PATROL`
- `INTERCEPT`
- `ENGAGE`
- `EVADE`
- `RTB`

状态转换核心依据：

- 最近威胁距离 `ctx.min_threat_distance`
- 是否有预警信息
- 高风险区/中风险区是否有敌机
- 是否有来袭导弹
- 燃油是否临界
- 任务时间是否超过 20 分钟

当前逻辑特点：

- `EVADE` 触发条件是“来袭导弹”或“最近威胁距离 < MAR”。
- `ENGAGE` 的条件不仅取决于距离，还取决于高风险/中风险区是否已有敌机。
- `RTB` 只有在敌机已无或任务结束且燃油/时间条件满足时才进入。

当前实现里，`CAPState.EVADE` 最终并不会直接走全局规避动作，而是仍然复用 `_get_engage_action()`。这件事是目前稳定基线的一部分。


## 7. 责任区、风险区与任务评估

### 7.1 风险区定义

`cap/picture/picture.py` 里 `Picture.get_risk_zone()` 按 FAOR 纵深把区域划成：

- `HIGH`
- `MEDIUM`
- `LOW`
- `OUTSIDE`

当前默认 `faor_length = 300km`，因此逻辑上是：

- 0-100 km：高风险
- 100-200 km：中风险
- 200-300 km：低风险

### 7.2 MissionEvaluator

`cap/mission_evaluator.py` 用下面的方式评价任务：

- 若我方损失比例 >= 0.75，则失败。
- 若高风险区持续突破时间 > 60s，则失败。
- 若任务时间达到 20 分钟：
  - 高风险区无敌机
  - 中风险区敌机数不超过容忍值
  - 我方损失比例 < 0.5
  - 则判成功
  - 否则判部分成功

这里与任务要求有两点差异：

- 任务原文强调“20 分钟内不让敌方进入高风险区”，而不是“允许突破但持续不超过 60s”。当前评估器是较宽松实现。
- 成功条件里把我方损失 < 50% 作为成功阈值，而任务要求上限是 75%。当前评估器把“成功”和“失败”之间又拆出一个内部的 partial 区间。


## 8. 图景层、预警机、雷达与航迹融合

### 8.1 AWACS

`cap/cap_picture_helpers.py::_update_awacs_data()` 会把敌机真值位置喂给 `MockAwacsDataSource`，再生成带误差的预警航迹。

预警信息只在 `PATROL/INTERCEPT` 阶段主动刷新。进入其他阶段时，AWACS 更新停止，但图景中已有的航迹不会因此自动清空。

### 8.2 机载雷达

`cap/cap_radar.py` 实现了 CAP 专用火控雷达管理器：

- 最大探测距离 200km
- 最大高精跟踪目标数 6
- 扫描模式支持 `SEARCH / TRACK / LOCK / DIRECTED / SEARCH_ALL`
- 扫描时间表与任务要求一致：
  - 方位 10/30/60
  - 俯仰 1/2/4 bars
  - 例如 30 度、2 bars 为 10s

并且有一个关键限制：

- 前向视场硬限制为 `±60°`

也就是说，雷达并不是 360 度全向感知；目标不在前向视场内时，轨迹质量会很快下降。

### 8.3 航迹融合

`cap/picture/track_fusion.py` 将 AWACS 和雷达航迹融合：

- 优先按 `track_id` 直接关联
- 再对剩余航迹做最近邻空间匹配
- 支持简单加权融合，也预留滤波/状态估计器接口

融合后生成 `FusedTrack`，包含：

- `timestamp`
- `awacs_track`
- `radar_track`
- `lost_since`

### 8.4 当前图景层最关键的问题

`Picture.get_all_threats()` 和 `Picture.get_threats_in_zone()` 只是把所有 hostile track 原样返回，不做时效过滤。

而 `TrackFusion.fuse()` 的处理是：

- 如果旧航迹 5 秒没有更新，就标记 `lost_since = current_time`
- 但只要该 track 没被移除，它仍然保留在 `_fused_tracks` 中
- 然后 `Picture.update_track()` 继续把它放进 `Picture.tracks`

结果就是：

- 航迹“丢失”只是打了 `lost_since` 标记
- 但很多上层逻辑仍然把它当作正常敌机目标使用

这是当前项目最深的一条问题链。


## 9. 协同探测设计

`cap/tactics/cooperative_detection.py` 实现了 3 种协同探测模式：

- `SWEEP`
- `DIRECTED`
- `SEARCH`

对应任务要求三种情形：

1. 无预警信息：推磨扫描
2. 有预警信息：定向扫描
3. 有预警信息但目标丢失：全面搜索

### 9.1 模式切换

`update()` 的逻辑是：

- `target_lost=True`：立刻进入 `SEARCH`
- `has_awacs_info=True`：进入 `DIRECTED`
- 否则：进入 `SWEEP`

### 9.2 扫描分配

- `SWEEP`：把当前可用热机按左右分布切扇区，覆盖总计约 120 度前向区域
- `DIRECTED`：以 AWACS 给出的目标方位为中心，多机略微分散扫描
- `SEARCH`：在目标丢失后扩大覆盖范围重新找回

当前结论：

- 这部分框架是存在的，且模式切换逻辑是成立的。
- 但它产出的图景和后续战术链路之间，仍然受“陈旧航迹未过滤”问题影响。


## 10. 协同跟踪与协同交战设计

`cap/tactics/cooperative_engagement.py` 负责：

- 目标分配
- 发射前双机协同跟踪
- 发射后接力制导候选

### 10.1 目标分配

当前分配原则：

- 先给每个目标分一个主射手
- 若进入 cooperative 跟踪窗口，再给目标补第二架备份跟踪机

### 10.2 发射前协同跟踪

规则大体是：

- 当距离发射窗口足够近时，`need_cooperative = distance_to_launch < 20km`
- 此时目标会得到主跟踪机和备份跟踪机
- `TrackingStatus` 记录 primary/secondary tracker 和 cooperative 标志

### 10.3 接力制导

`start_guidance()` 会在发射时预选 relay candidate。

`request_relay()` 在原制导机需要脱离时把制导交给备份。

因此“发射后接力制导”的结构在系统里是有的。

### 10.4 当前问题

这部分本身不是空白，但依赖：

- 目标分配是否稳定
- 跟踪质量是否持续
- 导弹统计是否正确

一旦图景失真或局部控制器互相抢方向，协同跟踪就会很快掉到 `q=0.00 / lock=N`。


## 11. 我方战术选择与执行结构

当前我方战术不是单一来源，而是三层叠加：

1. CAP 层的战术选择器
2. 对子级战术分配逻辑
3. 旧 `TacticalTask` 战术引擎通过 `TacticalBridge` 输出的真实机动

### 11.1 TacticSelector

`cap/tactic_selector.py` 里定义了战术类型：

- `T_DS`
- `T_PA`
- `T_HL`
- `T_SBS`
- `T_FB`
- `T_TE`
- `T_TT`

按风险区定义候选集合：

- 低风险：`DS / PA / HL / SBS`
- 中风险：`FB / TE / TT`
- 高风险：`TE / TT`

但是当前代码里存在一个非常明显的调试残留：

- `select_tactic()` 开头直接把 `self._force_tactic = TacticType.T_FB`

这意味着如果这个选择器独立决定战术，它会被硬拉到 `FRONT_BACK`。

不过这不是最终全部行为来源，因为 CAP 后面还会从 `TacticalBridge` 读取旧战术引擎的“真实战术”覆盖对子战术。

### 11.2 CAP 层对子目标分配

`cap/cap_tactic_helpers.py::_update_tactic_selection()` 会先把 4v4 拆成左右两个对子：

- 左对：`A0100/A0200` 对 `B0100/B0200`
- 右对：`A0300/A0400` 对 `B0300/B0400`

然后：

- 从 `Picture` 中挑出每个对子自己的敌方候选
- 用 `_pick_pair_target_with_stickiness()` 做目标粘滞
- 再尝试读取 `TacticalBridge` 当前对子实际战术

所以 CAP 层对子分配本质上是：

- 用 CAP 图景先分对子目标
- 再允许桥接来的旧战术引擎覆盖具体战术类型

### 11.3 TacticalBridge

`cap/tactics/tactical_bridge.py` 是当前项目的关键桥：

- 把 4v4 环境映射成两个 2v2 虚拟环境
- 左对子映成 `A0100/A0200 vs B0100/B0200`
- 右对子映成 `A0100/A0200 vs B0100/B0200` 这种虚拟编号，但实际上包着真实 A0300/A0400、B0300/B0400

这个桥的作用不是只读，而是直接调用旧版 `TacticalTask` 产生命令动作。

因此：

- CAP 层负责外层状态、图景、责任区、对子目标
- 具体近距战术机动，大量还是旧 2v2 引擎在做

### 11.4 CAP get_action 与桥接执行

`cap/cap_task_impl.py::_get_engage_action()` 的逻辑大致是：

1. 先取对子战术分配 `assign`
2. 根据 `assign.target_id` 在 `Picture` 里找目标航迹
3. 若没找到，再取对子最近可见威胁
4. 若分配的战术是 `DS/PA/HL/SBS/FB/TE` 等，并且 `tactical_bridge` 可用：
   - 直接调用 `self.tactical_bridge.get_action()`
5. 否则才走 CAP 自己的战术执行函数

这说明一个事实：

- 当前我方大部分战术外观，不是 CAP 自己算出来的，而是“CAP 的目标/战术分配 + 旧 TacticalTask 的具体机动”共同决定的。


## 12. 旧 TacticalTask 的关键机动逻辑

旧引擎的实际机动主要集中在 `tactical_task_impl.py` 与 `tactical_task_action_mixin.py`。

### 12.1 核心阶段节点

当前旧战术引擎内部使用一套 BVR 节点链：

- `NLT`
- `MELD`
- `MTR`
- `LR`
- `TR`
- `DOR`
- `DR`
- `MAR`

以及各种阶段名：

- `BEYOND_NLT`
- `NLT_MELD`
- `MELD_MTR`
- `MTR_LR`
- `LR_TR`
- `TR_DOR`
- `DOR_DR`
- `DR_MAR`

### 12.2 ADAPTIVE_ATTACK

`_execute_adaptive_attack()` 是当前最频繁被落到的“兜底机动”。

它内部会根据：

- 当前目标是否仍活着
- 当前距离
- 阶段名
- 我机是否处于二次进攻
- 对子内存活数
- 编队间距
- 是否有来袭导弹

来决定以下行为：

- 近距规避
- 进入 `BVR_STANDOFF`
- 触发 `FORMATION_RECOVER`
- 继续分叉攻击 heading

### 12.3 BVR_STANDOFF

`_build_bvr_standoff_command()` 的本质是：

- 以目标方位为基准
- 加 180 度反向
- 再给左右机一个 `±26°` 的 split offset

这就是你在日志里看到 A0100/A0200 一左一右大偏置的直接来源。

### 12.4 FORMATION_RECOVER

`_build_dynamic_regroup_command()` 会按编队槽位重建目标点：

- 先算对子 rear north、anchor east
- 给长机/僚机分左右槽位
- 若 slot error 足够大，就强行把飞机往该槽位拉

这里的问题是：

- 这个槽位是几何重建槽位，不是威胁朝向槽位
- 在近距对抗里也可能触发
- 一旦 slot_error 大，就会表现成大幅横向回拉

这正是你看到“偏置后又回正，回正后又偏置”的第二来源。

### 12.5 TACTICAL_TURN 的返航拦截

`tactical_executor_support_mixin.py::execute_tactical_turn()` 中，若敌机仍活着且当前 env 里还能看到敌机，则：

- 不允许真正执行返航/回转
- 直接把战术改回 `ADAPTIVE_ATTACK`

这意味着：

- 系统里虽然有 `TACTICAL_TURN`
- 但很多时候它并不会真的变成“回转离场”
- 而是在敌机仍活着时被硬拦回接敌逻辑


## 13. 敌方 AI 系统设计

敌方当前主要由 `cap/adapters/enemy_ai_adapter.py` 驱动。

### 13.1 敌方基本结构

敌方默认也按左右两个对子：

- `PAIR_LEFT = B0100/B0200`
- `PAIR_RIGHT = B0300/B0400`

每个对子有一套波次状态机，`EnemyWaveState` 记录：

- `phase`
- `wave_index`
- `turn_entry_distance_km`
- `resume_distance_km`
- `regroup_y_km`
- `hold_until`

### 13.2 敌方主要相位

实际日志中常见的相位：

- `APPROACH`
- `TURN_SOUTH`
- `TURN_NORTH`
- `REGROUP_NORTH`

这些相位不是按“进入我方高风险区立刻返航”控制，而是按波次控制几何阈值控制，比如：

- `turn`
- `release`
- `hard standoff`
- `regroup timeout`

因此敌方前出很深这件事，不能简单判为 bug。它更像是当前敌方控制器按自己设计在运行。

### 13.3 敌方距离口径

敌方波次控制器计算最近我方距离时，用的是当前 env 中真实飞机位置。

这也是为什么敌方日志中的 `min_dist=38~45km` 往往比我方 CAP 顶层日志更可信。


## 14. 导弹与制导设计

`cap/adapters/missile_adapter.py` 负责：

- 库存
- 发射事件
- 导弹状态迁移
- 命中/脱靶统计

当前实现说明里仍写着：

- 我方使用 `R-27ER`
- 敌方使用 `AIM-120C`

这与任务要求“双方都按 120D，150km”存在配置偏差。

这会影响：

- 发射距离窗口
- 中制导依赖
- 双方脱离逻辑
- 命中统计解释

因此这是当前系统与任务设定的一个明确不一致点。


## 15. 当前实现与任务要求的匹配情况

### 15.1 已有实现框架

以下内容当前代码里是明确存在的：

- FAOR / bullseye / 风险区划分
- 热冷巡逻与 CAP 状态机
- 预警机输入
- 火控雷达扫描
- 航迹融合
- 协同探测三模式：`SWEEP / DIRECTED / SEARCH`
- 协同交战分配
- 发射前双机协同跟踪
- 发射后接力制导
- 战术模板枚举：`DS / PA / HL / SBS / FB / TE / TT`
- 敌方左右对子波次 AI
- 任务评估与 battle snapshot 日志

### 15.2 已有框架但未形成稳定闭环

以下内容虽然“有模块”，但目前没有稳定实现到任务要求：

- 协同探测到协同跟踪的稳定闭环
- 协同跟踪到稳定发射许可的闭环
- 协同攻击模板的稳定执行闭环
- 接力制导在制导机受压条件下的鲁棒性
- 规避与回转动作在近距窗口下的稳定性

### 15.3 明确偏差

当前实现与任务要求存在这些明确偏差：

- 任务要求中高风险区是严格禁止进入的控制目标，当前评估器允许“突破不超过 60 秒”。
- 任务要求用 120D/150km，当前导弹实现说明仍是 `R-27ER / AIM-120C`。
- CAP 顶层战术选择器有 `T_FB` 强制覆盖调试残留。
- 当前真实机动很大程度由旧 `TacticalTask` 驱动，而不是纯 CAP 模块自洽闭环。


## 16. 已经确认存在的问题

以下问题是当前已经能确认存在的，不是猜测。

### 16.1 图景层陈旧航迹未过滤

这是当前最深层、最基础的问题。

问题链：

- `TrackFusion` 会把过期航迹标记为 `lost_since`
- `Picture.get_all_threats()` 不过滤 `lost_since`
- `CAP` 状态上下文、风险区统计、对子状态日志都继续拿这些 track 做计算

直接后果：

- 顶层状态机看到的“最近威胁距离”可以严重滞后
- 风险区统计会滞后
- 对子日志里显示的目标节点和距离会滞后

### 16.2 顶层 CAP 与战术执行层使用不同几何来源

当前系统至少有两种“敌我距离”来源：

- CAP 顶层：大量使用 `Picture`
- 战术引擎/敌方 AI：大量使用当前 env 真实位置

后果：

- 上层认为敌机还在 100km 外
- 下层其实已经在 40km 窗口机动
- 状态判断、战术选择、射击门禁、机动方向会互相打架

### 16.3 ADAPTIVE_ATTACK / BVR_STANDOFF / FORMATION_RECOVER 存在控制仲裁冲突

这是当前轨迹异常的第二条主问题链。

在近距窗口里：

- 模板战术可能被 bypass 到 `ADAPTIVE_ATTACK`
- `ADAPTIVE_ATTACK` 又可能内部切到 `BVR_STANDOFF`
- 同时若对子间距/槽位条件满足，又会触发 `FORMATION_RECOVER`

因此实际动作可能在同一时段表现为：

- 先分叉偏置
- 再被拉回槽位
- 再次根据近距条件偏置

这不是一个健康的单战术动作，而是控制器冲突。

### 16.4 FORMATION_RECOVER 在近距仍可能下达大尺度槽位回拉

`FORMATION_RECOVER` 当前没有足够强的“近距抑制”。

当敌机已经在 40km 左右时，它仍可能因为 `slot_error` 大于阈值而要求几十公里级回拉。这会制造肉眼可见的大幅展开和回正。

### 16.5 TACTICAL_TURN 会被敌机存活条件拦回 ADAPTIVE_ATTACK

当前“回转/返航”不是只由当前战术模板决定，还被 `execute_tactical_turn()` 的“敌机仍活着则禁止回转”硬拦截。

结果是：

- 你以为系统在执行回转
- 实际上它经常被打回接敌逻辑

### 16.6 频繁 sim_recreate 会放大机动外观突兀性

日志里 `friendly_sim_recreate` 和 `enemy_sim_recreate` 很频繁。

这类重建会重置：

- 速度
- 俯仰
- 垂直速度

它不是最根本问题，但会把本来就不稳定的偏置/回拉表现得更突兀、更像抖动。

### 16.7 TacticSelector 中存在强制 `T_FB` 的调试残留

这不是日志异常的唯一根因，但它是明确代码问题。

若某些路径真的依赖 `TacticSelector.select_tactic()` 的返回，则实际战术会被强制到 `FRONT_BACK`，与任务区分战术设计不一致。

### 16.8 导弹统计和任务口径仍需谨慎

当前导弹统计曾出现过命中归属混淆。虽然你之前让我修过一版，但日志分析仍不能只看尾部统计，必须结合 `MISSILE_END` 事件。


## 17. 对 `cap_20260413_105822.log` 的异常分析

你问的是 7 分 22 秒开始到 8 分 34 秒左右，A0100/A0200 为什么左右迅速偏置、循环往复。

### 17.1 这一段实际在执行什么

这段不是单一战术模板在稳定执行。

它的控制链路是：

1. 左对子原始战术标签还是 `SIDE_BY_SIDE`
2. 但因为距离已经过近，日志明确写出：
   - `DR决策-A0100/A0200 距离过近，暂不进入模板重整，切换到 ADAPTIVE_ATTACK`
3. 进入 `ADAPTIVE_ATTACK` 后，近距又触发 `BVR_STANDOFF`
4. `BVR_STANDOFF` 给 A0100/A0200 发出一左一右的 split heading

所以你看到的左右偏置，直接对应的是：

- A0100：`BVR_STANDOFF` 约 200 度左右
- A0200：`BVR_STANDOFF` 约 140-155 度左右

### 17.2 为什么会循环往复

因为这段时间并不是单独只有 `BVR_STANDOFF`，还叠加了：

- `EVADE` 顶层状态长期存在
- `FORMATION_RECOVER`
- `friendly_sim_recreate`

因此表现上是：

- 分叉
- 调整
- 再分叉
- 再修正

### 17.3 结论

这一段不是正常的、健康的战术机动。

更准确地说：

- 它不是“某个模板本来就该这么飞”
- 而是“模板标签、近距自适应攻击、standoff 偏置、编队回拉、状态机失真”共同叠加出来的异常行为


## 18. 对 `cap_20260413_111035.log` 的异常分析

### 18.1 10 分 14 秒开始 A0100/A0200 分别朝东南和西南展开，为什么

这段展开的第一推力来自 `ADAPTIVE_ATTACK` / `BVR_STANDOFF` 的左右分叉航向。

在 600s 附近：

- A0100 处于 `DOR_DR / ADAPTIVE_ATTACK`
- A0200 也处于 `DOR_DR / ADAPTIVE_ATTACK`
- 两机 heading 已经一左一右分开

这部分“展开”本身可以理解成近距双机偏置攻击的几何结果，不完全是 bug。

### 18.2 为什么后面越飞越展开，而且持续很久

问题不在“最初展开”，而在“展开后没有收住”。

这是两条问题链叠加：

1. 图景层距离越来越假
- 雷达关闭后，CAP 顶层还在认为最近威胁 90km、100km、110km、120km
- 但敌方波次控制器此时看到的真实距离已经在 38-45km

2. 控制器持续互相抢动作
- 一边 `BVR_STANDOFF` 继续要求偏置
- 一边 `FORMATION_RECOVER` 又把飞机拉回槽位
- 再叠加 sim_recreate，轨迹就会很像“展开 - 回正 - 再展开”

### 18.3 12 分 22 秒、13 分 05 秒、13 分 38 秒附近 A0100 的偏置是否正常

这些时刻附近，A0100 的主要动作都可以对上 `BVR_STANDOFF`：

- heading 连续被推到 204、209、216、218、222、226、230 度一带
- 这不是随机抖动，而是近距 standoff 反向偏置

所以：

- “发生偏置”本身可解释
- “持续这么久，而且在上层还以为敌机很远时依然一直偏置”不正常

### 18.4 14 分 12 秒开始 A0200 循环偏置回正是否正常

这段的特征是：

- A0200 仍不断收到 `BVR_STANDOFF`
- 同时 A0100 或另一个对子成员会触发 `FORMATION_RECOVER`
- 槽位目标点又很远

因此 A0200 的“偏置后回正、再偏置”不是正常单动作，而是：

- standoff 偏置
- 编队槽位回拉
- 上层图景误判

叠加后的冲突结果。

### 18.5 A0300/A0400 类似现象为什么也会发生

原因相同，只是右对子更容易同时受到：

- `PINCER_ATTACK` 被 bypass 到 `ADAPTIVE_ATTACK`
- `BVR_STANDOFF`
- `FORMATION_RECOVER`
- 右侧局部跟踪质量塌陷

所以右对子经常比左对子还更像“被多个控制器抢航向”。


## 19. 当前最核心的根本问题判断

综合代码和日志，当前最根本的问题不是单个战术模板错，而是下面这两个问题叠加：

### 根本问题 1：图景层陈旧航迹继续参与决策

这是当前最基础的问题。

因为它会直接污染：

- CAP 状态机
- 风险区统计
- 对子节点日志
- 战术目标距离判断

只要这件事不修，系统上层对敌情的认知就可能长期比真实几何慢几十公里。

### 根本问题 2：近距窗口缺少统一的动作仲裁器

当前近距阶段至少有四个动作来源：

- 模板战术
- `ADAPTIVE_ATTACK`
- `BVR_STANDOFF`
- `FORMATION_RECOVER`

再加上：

- `TACTICAL_TURN` 会被强行拦回接敌
- `sim_recreate` 会改变即时飞行状态

所以行为层会表现出明显的冲突和抖动。


## 20. 优先排查与修复顺序建议

如果要逐个击破，建议按这个顺序：

1. 先修 `Picture` 的时效过滤
- 让 `get_all_threats()`、`get_threats_in_zone()` 至少能过滤掉长期 `lost_since` 的旧航迹。
- 这一条不修，后面很多行为分析都会被误导。

2. 统一“用于决策的最近威胁距离”口径
- 顶层状态机、对子状态、风险区统计、战术层要明确哪些用实时 env，哪些用图景，不能混用。

3. 给近距阶段建立动作仲裁优先级
- 明确近距窗口里：
  - 是不是还允许 `FORMATION_RECOVER`
  - 模板战术被 bypass 后，谁优先
  - `BVR_STANDOFF` 和 `FORMATION_RECOVER` 不能同时抢控制

4. 单独检查 `TACTICAL_TURN` 的拦截条件
- 现在“敌机仍活着”这个条件过强，会让很多本应回转的情况被打回接敌。

5. 评估是否降低 `friendly_sim_recreate` 对行为层的扰动
- 它不是首因，但会让观察到的轨迹更难分析。

6. 最后再回头修命中率和导弹统计
- 现在命中率低不只是导弹问题，而是整个感知-机动-发射链路不稳定的结果。


## 21. 当前结论摘要

当前项目不是“没实现”，而是“实现了很多模块，但这些模块没有形成稳定一致的作战闭环”。

最核心的确认结论有四条：

1. CAP 图景层会把陈旧丢失航迹继续当成有效威胁使用。
2. 顶层 CAP 与底层战术/敌方 AI 使用的几何来源不一致。
3. 近距窗口下 `ADAPTIVE_ATTACK / BVR_STANDOFF / FORMATION_RECOVER` 缺少稳定仲裁。
4. 你看到的多次左右偏置、展开、回正，并不是正常单战术动作，而是多个控制逻辑冲突后的外观结果。

从问题优先级上看，最该先修的不是某个单独机动模板，而是：

- 图景时效
- 距离口径统一
- 近距动作仲裁

这三条不解决，后续继续在模板层细修，大概率会继续出现“改一处、坏一处、每次飞得都不一样”的现象。
