# CAP战术系统重构任务清单

## 第一阶段：感知层 (P0)

### 任务1.1：创建Picture数据结构
- [x] 创建 `cap/picture/__init__.py`
- [x] 创建 `cap/picture/picture.py`
  - [x] 定义 `Track` 数据类
  - [x] 定义 `FusedTrack` 数据类
  - [x] 定义 `Picture` 类
  - [x] 实现 `get_threats_in_zone()` 方法
  - [x] 实现 `get_nearest_threat()` 方法
  - [x] 实现 `is_target_lost()` 方法

### 任务1.2：实现预警机数据源
- [x] 创建 `cap/picture/awacs_source.py`
  - [x] 实现 `MockAwacsDataSource` 类
  - [x] 实现探测范围检查 (400km)
  - [x] 实现位置误差添加 (2-3km)
  - [x] 实现高度误差添加 (500m)
  - [x] 实现航向误差添加 (10°)
  - [x] 实现目标丢失模拟 (最长20秒)
  - [x] 实现 `update()` 方法

### 任务1.3：实现数据融合
- [x] 创建 `cap/picture/track_fusion.py`
  - [x] 实现 `TrackFusion` 类
  - [x] 实现预警机数据与雷达数据融合
  - [x] 实现融合置信度计算
  - [x] 实现丢失计时管理

### 任务1.4：感知层单元测试
- [x] 创建 `tests/test_picture.py`
  - [x] 测试Track数据结构
  - [x] 测试MockAwacsDataSource误差范围
  - [x] 测试目标丢失场景
  - [x] 测试数据融合逻辑

---

## 第二阶段：CAP状态机 (P0)

### 任务2.1：实现CAP状态机
- [x] 创建 `cap/cap_state_machine.py`
  - [x] 定义 `CAPState` 枚举 (PATROL/INTERCEPT/ENGAGE/EVADE/RTB)
  - [x] 实现 `CAPStateMachine` 类
  - [x] 实现 `_check_patrol_transition()` - PR距离触发
  - [x] 实现 `_check_intercept_transition()` - NLT确认触发
  - [x] 实现 `_check_engage_transition()` - 威胁/完成触发
  - [x] 实现 `_check_evade_transition()` - 规避完成触发

### 任务2.2：定义战术控制距离
- [x] 创建 `cap/control_ranges.py`
  - [x] 定义 `ControlRanges` 数据类
  - [x] 添加PR距离 (280km)
  - [x] 实现从配置文件加载

### 任务2.3：集成状态机到CAPTask
- [x] 修改 `cap/cap_task.py`
  - [x] 添加 `CAPStateMachine` 实例
  - [x] 在 `step()` 中调用状态机更新
  - [x] 根据状态调用不同执行逻辑

### 任务2.4：状态机单元测试
- [x] 创建 `tests/test_cap_state_machine.py`
  - [x] 测试PATROL → INTERCEPT转换
  - [x] 测试INTERCEPT → ENGAGE转换
  - [x] 测试ENGAGE → EVADE转换
  - [x] 测试EVADE → INTERCEPT转换

---

## 第三阶段：协同探测 (P0)

### 任务3.1：实现协同探测模块
- [x] 创建 `cap/tactics/__init__.py`
- [x] 创建 `cap/tactics/cooperative_detection.py`
  - [x] 定义 `DetectionMode` 枚举 (SWEEP/DIRECTED/SEARCH)
  - [x] 实现 `CooperativeDetection` 类
  - [x] 实现 `_sweep_scan_assignment()` - 推磨扫描
  - [x] 实现 `_directed_scan_assignment()` - 定向扫描
  - [x] 实现 `_search_scan_assignment()` - 全域扫描

### 任务3.2：实现雷达扫描参数
- [x] 创建 `cap/tactics/radar_scan.py`
  - [x] 定义 `RadarScanParams` 数据类
  - [x] 实现扫描时间计算
  - [x] 实现扫描模式配置 (±10°/±30°/±60°)

### 任务3.3：更新配置文件
- [x] 修改 `cap/config/cap_config.yaml`
  - [x] 添加 `control_ranges` 配置
  - [x] 添加 `awacs` 配置
  - [x] 添加 `radar` 配置

### 任务3.4：协同探测集成测试
- [x] 创建 `tests/test_cooperative_detection.py`
  - [x] 测试无预警推磨扫描
  - [x] 测试有预警定向扫描
  - [x] 测试目标丢失全域扫描
  - [x] 测试模式切换逻辑

---

## 第四阶段：协同跟踪和制导 (P1)

### 任务4.1：实现协同跟踪模块
- [x] 创建 `cap/tactics/cooperative_engagement.py` (合并实现)
  - [x] 实现 `CooperativeEngagement` 类
  - [x] 实现 `should_start_tracking()` - 发射前10秒判断 (distance_to_launch < 20km)
  - [x] 实现 `assign_trackers()` - 分配跟踪任务 (update方法)
  - [x] 实现 `_evaluate_tracker_suitability()` - 评估跟踪适合度 (_select_best_shooter)
  - [x] 实现 `get_tracking_quality()` - 获取跟踪质量 (通过cap_radar)
  - [x] 实现 `TrackingStatus` 数据类
  - [x] 实现跟踪质量告警 (阈值0.6)
  - [x] 实现分配粘性 (5秒)

### 任务4.2：实现接力制导模块
- [x] 创建 `cap/tactics/cooperative_engagement.py` (合并实现)
  - [x] 实现 `start_guidance()` - 开始制导并预选接力候选
  - [x] 实现 `request_relay()` - 请求接力制导
  - [x] 实现 `_evaluate_relay_suitability()` - 评估接力适合度 (选择最近友机)
  - [x] 实现 `GuidanceStatus` 数据类

### 任务4.3：协同跟踪和制导测试
- [x] 创建 `tests/test_cooperative_tracking.py`
  - [x] 测试发射前10秒触发
  - [x] 测试双机跟踪分配
  - [x] 测试跟踪质量计算
- [x] 创建 `tests/test_relay_guidance.py`
  - [x] 测试接力触发条件
  - [x] 测试接力机选择
  - [x] 测试接力执行

---

## 第五阶段：战术决策增强 (P1)

### 任务5.1：实现任务评估模块
- [x] 创建 `cap/mission_evaluator.py`
  - [x] 实现 `MissionEvaluator` 类
  - [x] 实现高风险区敌机统计
  - [x] 实现中风险区敌机统计
  - [x] 实现我方损失统计
  - [x] 实现任务成功判定

### 任务5.2：实现基于风险区的战术选择
- [x] 创建 `cap/tactic_selector.py`
  - [x] 实现 `TacticSelector` 类
  - [x] 实现 `_select_tactic_by_zone()` 方法
  - [x] 低风险区：T_DS, T_PA, T_HL, T_SBS
  - [x] 中风险区：T_FB, T_PA
  - [x] 高风险区：T_TE, T_TT
  - [x] 实现最优攻击者选择算法

### 任务5.3：集成现有战术模块
- [x] 修改 `cap/cap_task.py`
  - [x] 导入 `MissionEvaluator`
  - [x] 导入 `TacticSelector`
  - [x] 添加 `_update_mission_evaluation()` 方法
  - [x] 添加 `_update_tactic_selection()` 方法

### 任务5.4：全流程集成测试
- [x] 创建 `tests/test_cap_integration.py`
  - [x] 测试完整CAP任务流程
  - [x] 测试敌机从低风险区逼近场景
  - [x] 测试任务评估期间交战
  - [x] 测试高风险区突破跟踪

---

## 第六阶段：系统集成与完善 (P0 - 新增)

> **背景**: 当前CAP系统框架已完成，但存在以下问题需要解决：
> 1. 敌方行为使用简单周期性偏移，导致轨迹摆动
> 2. 导弹发射功能未实现
> 3. 协同跟踪和接力制导未完整集成
> 4. 未集成现有的UnifiedEnemyTacticalAI等系统

### 任务6.1：创建CAP专用适配器模块
- **Status**: completed
- **Priority**: HIGH
- [x] 创建 `cap/adapters/__init__.py`
- [x] 创建 `cap/adapters/enemy_ai_adapter.py` - 敌方AI适配器
- [x] 创建 `cap/adapters/missile_adapter.py` - 导弹系统适配器
- [x] 创建 `cap/adapters/intent_adapter.py` - 意图识别适配器
- [x] 创建 `cap/adapters/tactic_executor_adapter.py` - 战术执行适配器
- **验收标准**:
  - [x] 适配器模块独立于原有代码
  - [x] 提供统一的接口供cap_task.py调用
  - [x] 处理导入失败的情况（优雅降级）

### 任务6.2：敌方AI适配器实现
- **Status**: completed
- **Priority**: HIGH
- **依赖**: 任务6.1
- [x] 封装 `UnifiedEnemyTacticalAI`
- [x] 实现 `get_enemy_action(env, agent_id, current_time, threat_info)` 方法
- [x] 实现 `update_enemy_phase(distance_km)` 方法
- [x] 支持5阶段行为（NLT_MELD, MELD_MTR, MTR_TR, TR_DOR, DOR_DR）
- **验收标准**:
  - [x] 敌方不再摆动，使用智能行为
  - [x] 敌方轨迹平滑

### 任务6.3：导弹系统适配器实现
- **Status**: completed
- **Priority**: HIGH
- **依赖**: 任务6.1
- [x] 封装 `MissileManager` 和 `TacticalStateManager`
- [x] 实现 `should_launch(env, agent_id, target_id, distance_km)` 方法
- [x] 实现 `execute_launch(env, agent_id, target_id)` 方法
- [x] 实现 `get_missile_status(agent_id)` 方法
- **验收标准**:
  - [x] 在LR节点(100km)正确触发导弹发射
  - [x] 支持单发和齐射模式
  - [x] 正确跟踪导弹状态

### 任务6.4：意图识别适配器实现
- **Status**: completed
- **Priority**: MEDIUM
- **依赖**: 任务6.1
- [x] 封装 `EnemyIntentRecognizer`
- [x] 实现 `analyze_intent(enemy_tracks)` 方法
- [x] 实现 `get_threat_level(intent_type)` 方法
- **验收标准**:
  - [x] 正确识别敌方意图（攻击/侦察/突防）
  - [x] 在NLT节点(180km)触发意图识别

### 任务6.5：战术执行适配器实现
- **Status**: completed
- **Priority**: MEDIUM
- **依赖**: 任务6.1
- [x] 封装 `TacticalExecutor`
- [x] 实现 `execute_tactic(tactic_type, env, agent_id, target_info)` 方法
- [x] 实现 `get_available_tactics(distance_km)` 方法
- **验收标准**:
  - [x] 支持7种战术模板执行
  - [x] 根据距离自动选择合适战术

### 任务6.6：更新cap_task.py集成适配器
- **Status**: completed
- **Priority**: HIGH
- **依赖**: 任务6.2, 6.3, 6.4, 6.5
- [x] 导入适配器模块
- [x] 在__init__中初始化适配器
- [x] 修改 `_get_enemy_action()` 使用EnemyAIAdapter
- [x] 修改 `_get_engage_action()` 使用MissileAdapter
- [x] 在NLT节点添加意图识别调用
- [x] 在战术执行中使用TacticExecutorAdapter
- **验收标准**:
  - [x] 敌方使用智能AI行为
  - [x] 导弹在LR节点正确发射
  - [x] 意图识别在NLT节点触发
  - [x] 战术执行使用现有模板

### 任务6.7：实现协同跟踪功能
- **Status**: completed
- **Priority**: HIGH
- **依赖**: 任务6.6
- [x] 在MTR节点(120km)启动协同跟踪
- [x] 实现 `_start_cooperative_tracking(env, target_id)` 方法
- [x] 选择至少2架飞机进行跟踪
- [x] 实现跟踪数据融合
- **验收标准**:
  - [x] MTR节点触发协同跟踪
  - [x] 至少2机同时跟踪同一目标
  - [x] 跟踪数据正确融合

### 任务6.8：实现接力制导功能
- **Status**: completed
- **Priority**: MEDIUM
- **依赖**: 任务6.3, 6.7
- [x] 在TR节点(80km)支持接力制导
- [x] 实现 `_check_relay_guidance(env, missile_id, current_guide_id)` 方法
- [x] 检查当前制导机状态
- [x] 如果需要脱离，选择友机接替
- [x] 转移制导任务
- **验收标准**:
  - [x] TR节点支持接力制导
  - [x] 制导机脱离时自动转移任务
  - [x] 导弹制导不中断

### 任务6.9：完善规避机动
- **Status**: completed
- **Priority**: MEDIUM
- **依赖**: 任务6.6
- [x] 实现 `_execute_beam_maneuver(env, agent_id, threat_direction)` - 三九Beam机动
- [x] 实现 `_execute_notch_maneuver(env, agent_id, threat_direction)` - Notch机动
- [x] 实现 `_execute_dive_escape(env, agent_id)` - 俯冲脱离
- **验收标准**:
  - [x] 检测到来袭导弹时触发规避
  - [x] 支持三种规避机动
  - [x] MAR节点(35km)强制规避

### 任务6.10：集成测试与验证
- **Status**: completed
- **Priority**: HIGH
- **依赖**: 任务6.6, 6.7, 6.8, 6.9
- [x] 创建 `tests/test_full_integration.py`
- [x] 敌方AI行为测试（不再摆动）
- [x] 导弹发射测试（LR节点触发）
- [x] 协同跟踪测试（MTR节点）
- [x] 接力制导测试（TR节点）
- [x] 规避机动测试（MAR节点）
- [x] 全流程仿真测试
- **验收标准**:
  - [x] 所有测试用例通过
  - [x] 全流程仿真无异常
  - [x] 敌方轨迹平滑（无摆动）
  - [x] 导弹正确发射

---

## 第七阶段：配置和文档 (P2)

### 任务7.1：完善配置管理
- [ ] 修改 `cap/config_manager.py`
  - [ ] 添加战术控制距离配置加载
  - [ ] 添加预警机配置加载
  - [ ] 添加雷达配置加载
  - [ ] 添加任务评估配置加载

### 任务7.2：更新运行脚本
- [ ] 修改 `cap/run_cap_simulation.py`
  - [ ] 支持新的CAP状态机
  - [ ] 添加状态转换日志
  - [ ] 添加任务评估输出

### 任务7.3：编写使用文档
- [ ] 创建 `cap/README.md`
  - [ ] 模块结构说明
  - [ ] 配置参数说明
  - [ ] 使用示例

---

## 依赖关系

```
第一阶段 ──▶ 第二阶段 ──▶ 第三阶段 ──▶ 第四阶段 ──▶ 第五阶段
                                                      │
                                                      ▼
                                               第六阶段（系统集成）
                                                      │
    ┌─────────────────────────────────────────────────┼─────────────────────────────────────────────────┐
    │                                                 │                                                 │
    ▼                                                 ▼                                                 ▼
任务6.1 ──┬──▶ 任务6.2 ──┐                      任务6.3 ──┐                                      任务6.4, 6.5
(适配器)  │              │                      (导弹)    │                                      (意图/战术)
          │              ▼                                ▼                                            │
          └──────▶ 任务6.6 ◀──────────────────────────────┴────────────────────────────────────────────┘
                  (集成)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     任务6.7     任务6.8     任务6.9
    (协同跟踪)  (接力制导)  (规避机动)
        │           │           │
        └───────────┼───────────┘
                    ▼
                任务6.10
              (集成测试)
                    │
                    ▼
               第七阶段
              (配置文档)
```

---

## CAP任务流程图（使用更新后的距离节点）

```
距离(km)  300    280    180    150    120    100    80     60     50     45  42  38  35
          │      │      │      │      │      │      │      │      │      │   │   │   │
节点:     │     PR     NLT    MELD   MTR    LR     TR    DOR    DR    MTR' LR' TR' MAR
          │      │      │      │      │      │      │      │      │      │   │   │   │
状态:   PATROL  │◀────INTERCEPT────▶│◀──────────ENGAGE──────────▶│◀─二次进攻─▶│◀EVADE
          │      │      │      │      │      │      │      │      │      │   │   │   │
功能:     │   协同探测  意图识别  雷达融合  协同跟踪  导弹发射  接力制导  防御机动  决断点  二次跟踪/发射/制导  紧急规避
```

---

## 进度跟踪

| 阶段 | 任务数 | 完成数 | 进度 |
|-----|-------|-------|------|
| 第一阶段：感知层 | 4 | 4 | 100% |
| 第二阶段：状态机 | 4 | 4 | 100% |
| 第三阶段：协同探测 | 4 | 4 | 100% |
| 第四阶段：协同跟踪 | 3 | 3 | 100% |
| 第五阶段：战术决策 | 4 | 4 | 100% |
| 第六阶段：系统集成 | 10 | 10 | 100% |
| 第七阶段：配置文档 | 3 | 0 | 0% |
| **总计** | **32** | **29** | **91%** |

---

## 需要集成的现有系统

| 系统 | 文件位置 | 功能 |
|------|----------|------|
| UnifiedEnemyTacticalAI | `tacticalProject/unified_enemy_tactical_ai.py` | 敌方智能行为（5阶段） |
| MissileManager | `tacticalProject/missile_manager.py` | 导弹发射/跟踪 |
| TacticalStateManager | `tacticalProject/tactical_state_manager.py` | 战术状态管理 |
| EnemyIntentRecognizer | `tacticalProject/intent/enemy_intent.py` | 敌方意图识别 |
| TacticalExecutor | `tacticalProject/tactical_executor.py` | 战术模板执行 |

---

## 距离节点定义（已更新）

| 节点 | 距离 | 说明 |
|------|------|------|
| PR | 280km | 态势距离 - 触发INTERCEPT |
| NLT | 180km | 初始距离 - 意图识别 |
| MELD | 150km | 目标跟踪距离 - 雷达融合 |
| MTR | 120km | 最小跟踪距离 - 协同跟踪 |
| LR | 100km | 发射距离 - 导弹发射 |
| TR | 80km | 转换距离 - 接力制导 |
| DOR | 60km | 期望脱离距离 - 防御机动 |
| DR | 50km | 决断距离 - 决断点 |
| **二次进攻区域 (DR-MAR)** | | |
| MTR' | 45km | 第二轮跟踪距离 |
| LR' | 42km | 第二轮发射距离 |
| TR' | 38km | 第二轮转换距离 |
| MAR | 35km | 最小脱离距离 - 紧急规避 |
