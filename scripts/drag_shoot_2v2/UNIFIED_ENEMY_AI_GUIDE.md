# 统一敌方战术AI系统使用指南

## 概述

统一敌方战术AI系统是一个全新设计的敌方AI架构，解决了现有系统中的逻辑问题，并建立了基于动作组合的随机化战术选择系统。该系统适用于所有五个战术项目（拖曳射击、钳形夹击、前后攻击、上下夹击、并排射击）。

## 系统特点

### 1. 核心框架保持不变
- **三模式战术系统**：攻击(AGGRESSIVE)、防御(DEFENSIVE)、中性(NEUTRAL)
- **五阶段战术框架**：NLT_MELD → MELD_MTR → MTR_TR → TR_DOR → DOR_DR
- **雷达工作模式系统**：搜索、跟踪、锁定、待机

### 2. 动作层面随机化
- 15种基础和复合动作类型
- 基于权重矩阵的随机化动作选择
- 参数化的机动执行（角度、持续时间、速度等）

### 3. 智能威胁响应
- 六级威胁评估系统（NONE到SEVERE）
- 基于威胁等级的动态模式切换
- 长机-僚机差异化行为

### 4. 增强行动注释
- 基于实际AI决策的精确注释
- 支持复合动作的阶段化注释
- 提供战术意图分析

## 文件结构

```
scripts/drag_shoot_2v2/
├── unified_enemy_tactical_ai.py      # 核心AI系统
├── enemy_ai_adapter.py               # 适配器接口
├── enhanced_action_annotator.py      # 增强行动注释系统
├── integration_example.py            # 集成示例
└── UNIFIED_ENEMY_AI_GUIDE.md        # 本文档
```

## 快速开始

### 1. 基本集成

在现有战术任务中集成新的敌方AI系统：

```python
# 在战术任务类中
from integration_example import integrate_into_drag_shoot_task

class DragShootTacticalTask(MultipleCombatTask):
    def __init__(self, ...):
        super().__init__(...)
        # 集成统一敌方AI系统
        self.unified_enemy_ai = integrate_into_drag_shoot_task(self)
    
    def _get_enemy_command_indices(self, env, agent_id):
        # 替换原有的敌方AI调用
        return self.unified_enemy_ai.get_enemy_command_indices(env, agent_id)
```

### 2. 启用行动注释

在数据记录系统中启用行动注释：

```python
# 在unified_data_recorder.py中
def record_aircraft_trajectory(self, env, current_time, tactical_task=None):
    # ... 原有代码 ...
    
    # 添加行动注释（如果有集成的敌方AI系统）
    if hasattr(tactical_task, 'unified_enemy_ai') and agent_id.startswith('B'):
        annotation_data = tactical_task.unified_enemy_ai.get_action_annotation_for_csv(agent_id)
        trajectory_record.update(annotation_data)
    
    self.trajectory_data.append(trajectory_record)
```

### 3. 系统重置

在新任务开始时重置系统状态：

```python
def reset_task(self):
    if hasattr(self, 'unified_enemy_ai'):
        self.unified_enemy_ai.reset_for_new_episode()
```

## 动作类型说明

### 基础动作
- `MAINTAIN_HEADING`: 保持航向
- `TURN_LEFT/RIGHT`: 左转/右转
- `CLIMB/DESCEND`: 爬升/下降
- `ACCELERATE/DECELERATE`: 加速/减速

### 战术机动
- `CRANK_LEFT/RIGHT`: 左/右侧Crank机动
- `NOTCH_MANEUVER`: Notch机动（90度规避）
- `BEAM_MANEUVER`: Beam机动（侧向飞行）

### 复合动作
- `SHORT_SKATE`: Short Skate三阶段机动
- `AGGRESSIVE_APPROACH`: 攻击接近
- `DEFENSIVE_SPLIT`: 防御分离
- `RETURN_TO_BASE`: 返航

## 行动注释系统

### 注释类型

1. **Action_Type**: 具体动作类型
2. **Action_Intent**: 行动意图
   - search: 搜索
   - tracking: 跟踪
   - lock_on: 锁定
   - missile_launch: 导弹发射
   - evasive_maneuver: 规避机动
   - attack_maneuver: 攻击机动
   - defensive_maneuver: 防御机动
   - return_to_base: 返航
   - disengagement: 脱离接触

3. **Tactical_Posture**: 战术姿态
   - offensive: 攻击姿态
   - defensive: 防御姿态
   - neutral: 中性姿态
   - evasive: 规避姿态

4. **Phase_Info**: 阶段信息
   - SHORT_SKATE_PHASE_1_CRANK: Short Skate Crank阶段
   - SHORT_SKATE_PHASE_2_TURN_COLD: Short Skate Turn Cold阶段
   - SHORT_SKATE_PHASE_3_ESCAPE: Short Skate Escape阶段
   - RETURN_TO_BASE_ACTIVE: 返航激活

5. **Confidence**: 注释置信度 (0.00-1.00)

## 配置和调试

### 启用调试模式

```python
# 启用详细日志
tactical_task.unified_enemy_ai.enable_debug_mode()

# 获取系统状态
status = tactical_task.unified_enemy_ai.get_system_status()
print(status)

# 获取智能体详细状态
agent_status = tactical_task.unified_enemy_ai.get_detailed_enemy_status("B0100")
print(agent_status)
```

### 强制动作测试

```python
# 强制敌方执行特定动作
tactical_task.unified_enemy_ai.force_enemy_action("B0100", "short_skate")
```

## 项目特定集成

### 拖曳射击项目
```python
from integration_example import integrate_into_drag_shoot_task
self.unified_enemy_ai = integrate_into_drag_shoot_task(self)
```

### 钳形夹击项目
```python
from integration_example import integrate_into_pincer_attack_task
self.unified_enemy_ai = integrate_into_pincer_attack_task(self)
```

### 前后攻击项目
```python
from integration_example import integrate_into_front_back_task
self.unified_enemy_ai = integrate_into_front_back_task(self)
```

### 上下夹击项目
```python
from integration_example import integrate_into_high_low_task
self.unified_enemy_ai = integrate_into_high_low_task(self)
```

### 并排射击项目
```python
from integration_example import integrate_into_side_by_side_task
self.unified_enemy_ai = integrate_into_side_by_side_task(self)
```

## 常见问题

### Q: 如何确保新系统与现有代码兼容？
A: 适配器提供了完全向后兼容的接口，只需要替换敌方AI的调用函数即可。

### Q: 如何验证行动注释的准确性？
A: 系统提供置信度评分，并支持调试模式查看详细的AI决策过程。

### Q: 如何调整敌方行为的随机性？
A: 可以修改`unified_enemy_tactical_ai.py`中的动作权重矩阵来调整行为概率。

### Q: 系统性能如何？
A: 新系统采用缓存机制和优化的决策算法，性能优于原有系统。

## 技术支持

如果在集成过程中遇到问题，请检查：

1. 日志输出中的错误信息
2. 系统状态和智能体状态
3. 动作权重配置是否正确
4. 环境对象是否包含必要的属性

## 更新日志

### v2.0 (当前版本)
- 全新的统一敌方战术AI系统
- 基于动作组合的随机化选择
- 增强的行动注释系统
- 完整的适配器接口

### 未来计划
- 支持更多战术机动类型
- 机器学习驱动的行为优化
- 实时战术分析和调整
