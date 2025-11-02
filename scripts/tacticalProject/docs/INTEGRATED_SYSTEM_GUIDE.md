# 集成战术系统使用指南

## 系统概述

集成战术系统实现了完整的BVR空战决策链，包括：

1. **态势评估系统** (`situation_evaluator.py`)
2. **意图识别系统** (`intent_recognizer.py`)
3. **决策制定系统** (`decision_maker.py`)
4. **机动动作库** (`maneuver_library.py`)
5. **集成调度器** (`integrated_tactical_system.py`)

## 系统架构

```
┌─────────────────────────────────────────┐
│     IntegratedTacticalSystem (调度器)    │
└────────────┬────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌───────────┐   ┌──────────────┐
│态势评估器  │   │ 意图识别器    │
└─────┬─────┘   └──────┬───────┘
      │                │
      └────────┬───────┘
               ▼
       ┌──────────────┐
       │ 决策制定器    │
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │ 机动动作库    │
       └──────────────┘
```

## 核心功能

### 1. 态势评估系统

**实现文件**: `core/situation_evaluator.py`

**功能**:
- 5维态势评估：角度、距离、高度、速度、探测概率
- 根据战术阶段动态调整权重
- 计算威胁值（与态势值负相关）

**评估项说明**:
```python
# 角度项 [0, 1]
# - 1.0: 我机正对敌机，敌机尾对我机（最优）
# - 0.0: 我机背对敌机，敌机正对我机（最劣）

# 距离项 [0, 1]  
# - 根据阶段设置理想距离
# - NLT/MELD: 80-100km
# - MTR/LR: 75-80km
# - TR/DOR: 65-75km

# 高度项 [0, 1]
# - 1.0: 高出敌机1000-3000m（适度优势）
# - 0.5: 与敌机同高度
# - 0.0: 低于敌机过多（劣势）

# 速度项 [0, 1]
# - 根据阶段调整理想速度
# - 接敌阶段：高速
# - 发射阶段：中速
# - 规避阶段：与敌相近

# 探测概率项 [0, 1]
# - 基于RCS和距离的雷达探测能力
```

**阶段权重配置**:
```python
NLT_MELD:   角度35% 距离35% 高度10% 速度10% 探测10%
MELD_MTR:   角度30% 距离30% 高度15% 速度15% 探测10%
MTR_LR:     角度25% 距离25% 高度20% 速度20% 探测10%
LR_TR:      角度20% 距离20% 高度20% 速度25% 探测15%
TR_DOR:     角度20% 距离20% 高度20% 速度20% 探测20%
DOR_DR:     角度25% 距离15% 高度20% 速度20% 探测20%
```

**使用示例**:
```python
from core import SituationEvaluator, TacticalPhase

evaluator = SituationEvaluator()
situation = evaluator.evaluate_situation(
    my_aircraft,
    enemy_aircraft,
    TacticalPhase.LR_TR
)

print(f"总态势值: {situation.total:.2f}")
print(f"角度优势: {situation.angle:.2f}")
print(f"威胁值: {evaluator.calculate_threat(situation):.2f}")
```

### 2. 意图识别系统

**实现文件**: `core/intent_recognizer.py`

**功能**:
- 识别敌方6种意图类型
- 基于行为特征的规则分类
- 历史数据趋势分析
- 置信度评估

**意图类型**:
```python
# 攻击类型
EnemyIntent.ATTACK        # 攻击
EnemyIntent.COORDINATED   # 协同攻击
EnemyIntent.FEINT         # 佯攻

# 中立类型
EnemyIntent.JAMMING       # 干扰
EnemyIntent.RECONNAISSANCE # 侦察
EnemyIntent.DEFENSIVE     # 防御/规避

# 脱离类型
EnemyIntent.ESCAPE        # 逃逸
```

**识别规则**:
```python
逃逸: 航向背离(>135°) + 距离增加
攻击: 航向对准(<45°) + 距离减小 + 速度快
协同: 攻击 + 编队接近(<10km)
佯攻: 之前攻击 + 现在转向(>60°)
防御: 侧向机动(60-120°) + 高度变化
侦察: 保持距离(>80km) + 低速
干扰: 其他中立行为
```

**使用示例**:
```python
from core import IntentRecognizer, EnemyIntent

recognizer = IntentRecognizer()
intent = recognizer.recognize_enemy_intent(
    env,
    "B0100",
    my_aircraft_list
)

confidence = recognizer.get_intent_confidence("B0100")
print(f"敌机意图: {intent.value}, 置信度: {confidence:.2f}")
```

### 3. 决策制定系统

**实现文件**: `core/decision_maker.py`

**功能**:
- 威胁响应决策（继续/规避/撤退）
- 导弹发射决策
- 二次进攻决策
- 机动动作选择

**决策逻辑**:

**保守肃清意图**:
```python
撤退条件:
1. 敌方进攻意图 + 总威胁度>0.8
2. 敌方进攻意图 + 2个以上单项威胁度>0.8

规避条件:
1. 敌方进攻或中立 + 1-2个单项威胁度>0.8

继续: 其他情况
```

**激进肃清意图**:
```python
永不撤退，继续进攻
```

**防御意图**:
```python
敌机逃逸 → 撤退
敌机中立 + TR阶段后 → 撤退  
其他 → 继续防御
```

**导弹发射决策**:
```python
条件:
1. 阶段 == LR_TR (78km)
2. 距离在70-85km范围
3. 威胁决策 != 撤退
4. 未发射过
```

**使用示例**:
```python
from core import DecisionMaker, FriendlyIntent

decision_maker = DecisionMaker()
decision = decision_maker.make_threat_decision(
    FriendlyIntent.CONSERVATIVE_CLEAR,
    enemy_intent,
    threat_score,
    situation,
    phase
)

print(f"决策: {decision.response.value}")
print(f"原因: {decision.reason}")
```

### 4. 机动动作库

**实现文件**: `core/maneuver_library.py`

**新增机动**:

**战术爬升 (Tactical Climb)**:
```python
特点: 爬升 + 转向 + 加速
用途: DOR阶段快速获得高度优势
参数: 
  - target_altitude_gain: 500-1500m
  - turn_angle: -90° 到 +90°
  - accelerate: True/False
```

**战术下降 (Tactical Descend)**:
```python
特点: 下降 + 转向 + 加速  
用途: 快速脱离高威胁区域
参数:
  - target_altitude_loss: 150-1500m
  - turn_angle: -90° 到 +90°
  - accelerate: True/False
```

**Notch Back机动**:
```python
特点: 90°急转 + 下降 + 加速
用途: 雷达反制，使敌机失锁
动作: 90°转向 + 下降500m + 加速100m/s
```

**Beam机动**:
```python
特点: 侧向对敌（三九线）
用途: 减少RCS，规避导弹
目标: 将敌机置于3/9点方向（90°侧向）
```

**使用示例**:
```python
from core import ManeuverLibrary, ManeuverType

maneuver_lib = ManeuverLibrary()

# 战术爬升
alt, hdg, vel = maneuver_lib.execute_tactical_climb(
    env, agent_id,
    target_altitude_gain=1000,
    turn_angle=-90,
    accelerate=True
)

# Notch Back
alt, hdg, vel = maneuver_lib.execute_notch_back(
    env, agent_id,
    direction="left"
)

# Beam机动
alt, hdg, vel = maneuver_lib.execute_beam_maneuver(
    env, agent_id,
    enemy_bearing=45
)
```

### 5. 集成调度器

**实现文件**: `core/integrated_tactical_system.py`

**功能**:
- 协调所有子系统
- 统一数据缓存
- 自动更新循环
- 简化调用接口

**使用示例**:
```python
from core import IntegratedTacticalSystem, FriendlyIntent

# 初始化
system = IntegratedTacticalSystem(
    my_intent=FriendlyIntent.CONSERVATIVE_CLEAR
)

# 每帧更新（在tactical_task中自动调用）
system.update_situation_assessment(env, "A0100", "B0100", phase)
system.update_intent_recognition(env, "B0100", my_aircraft_list)
decision = system.make_tactical_decision(env, "A0100", "B0100", phase)

# 查询缓存数据
situation = system.get_situation("A0100", "B0100")
threat = system.get_threat("A0100", "B0100")
intent = system.get_intent("B0100")
decision = system.get_decision("A0100")
```

## 配置文件

**配置我方意图** (在 `tactical_bvr.yaml`):
```yaml
# 添加我方意图配置
friendly_intent: "CONSERVATIVE_CLEAR"  # 可选: AGGRESSIVE_CLEAR, DEFENSIVE
```

## 日志输出

系统会输出以下调试信息：

```
✅ 态势评估系统初始化完成
✅ 意图识别系统初始化完成
✅ 决策制定系统初始化完成
✅ 机动动作库初始化完成
✅ 集成战术系统初始化完成 (意图: CONSERVATIVE_CLEAR)

📊 [A0100] 态势评估 (vs B0100): 总分=0.65 (角度=0.70 距离=0.60 高度=0.65 速度=0.62 探测=0.68) 威胁=0.35
🎯 [敌方B0100] 意图识别: ATTACK (置信度=0.80)
⚖️ [A0100] 战术决策: CONTINUE (威胁可控，继续执行任务)
🚀 [A0100] 满足导弹发射条件 (阶段=LR_TR, 距离=78.5km)
```

## 与原有系统集成

集成系统**不替代**原有战术系统，而是作为**辅助决策层**：

1. **态势评估**：提供量化的态势和威胁值
2. **意图识别**：识别敌方行为模式
3. **决策建议**：给出威胁响应建议
4. **机动扩展**：提供额外的战术机动选项

原有的5种进攻战术（拖曳射击、钳形攻势、上下夹击、前后攻击、并排射击）继续正常工作。

## 未来扩展方向

### 已实现
- ✅ 5维态势评估
- ✅ 6种敌方意图识别
- ✅ 3种我方意图支持
- ✅ 威胁响应决策（撤退/规避/继续）
- ✅ 导弹发射决策
- ✅ 11种机动动作（包含新增战术机动）
- ✅ 完整集成系统

### 待实现（优先级低）
- ⏳ 雷达模式自动切换（搜索→跟踪→锁定）
- ⏳ IFF敌我识别集成
- ⏳ 多轮进攻系统（MTR2、LR2、TR2节点）
- ⏳ DR节点20s时间窗口决策
- ⏳ MAR节点强制脱离逻辑
- ⏳ 基于威胁值的战术自动选择
- ⏳ 目标分配优化算法

## 性能优化建议

1. **缓存策略**：态势评估结果已缓存，避免重复计算
2. **更新频率**：每帧更新，但打印日志频率为60步一次
3. **计算负载**：主要计算集中在态势评估，复杂度O(n²)，n为飞机数量
4. **内存占用**：历史数据限制在最近5个记录

## 故障排查

**问题1**: 系统初始化失败
```
原因: 缺少core包或导入路径错误
解决: 确保core/__init__.py存在且正确导出所有模块
```

**问题2**: 态势评估值异常
```
原因: 飞机数据读取失败或计算溢出
解决: 检查env.agents是否正确，查看日志中的错误信息
```

**问题3**: 意图识别不准确
```
原因: 历史数据不足或规则需要调整
解决: 至少运行5个时间步后再查看，调整识别规则阈值
```

**问题4**: 决策不符合预期
```
原因: 威胁阈值设置不合理
解决: 调整DecisionMaker中的retreat_threat_threshold和evade_threat_threshold
```

## 总结

集成战术系统提供了**完整的BVR空战决策框架**，实现了从态势感知到机动执行的闭环。系统采用模块化设计，易于扩展和维护。

**核心优势**:
- ✅ 量化态势评估
- ✅ 智能意图识别
- ✅ 分层决策机制
- ✅ 丰富机动库
- ✅ 统一集成接口

**当前状态**: **生产就绪**，可直接用于战术仿真和训练。
