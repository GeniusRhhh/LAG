# 增强敌方AI机动逻辑 - 技术文档

## 概述

本文档描述了为`drag_shoot_2v2`项目实现的增强敌方AI机动逻辑。新的AI系统提供了智能的威胁评估、动态机动选择和战术响应能力，显著提升了敌方飞机的战术表现。

## 主要增强功能

### 1. 智能威胁评估系统

#### 多维度威胁分析
- **导弹威胁**: 基于距离、速度和类型的动态评估
- **敌机威胁**: 考虑距离、接近速度、航向威胁
- **雷达威胁**: 锁定和跟踪状态评估
- **战术态势**: 数量优劣势、包围状态分析
- **能量状态**: 高度和速度优劣势评估

#### 威胁等级分类
- `CRITICAL`: 严重威胁，立即规避
- `HIGH`: 高威胁，防御机动
- `MEDIUM`: 中等威胁，平衡攻防
- `LOW`: 低威胁，主动攻击
- `NONE`: 无威胁，正常巡逻

### 2. 动态机动选择

#### 智能机动类型
- **CAP巡逻**: 智能巡逻，保持战斗准备
- **攻击性接敌**: 根据距离和威胁调整接敌策略
- **防御转弯**: 基于威胁方向的智能规避
- **规避机动**: 高机动性S型机动
- **Notch机动**: 90度雷达规避
- **Split-S机动**: 快速俯冲转弯
- **桶滚机动**: 复杂三维机动
- **攻击定位**: 智能定位到最佳攻击位置

#### 战术响应逻辑
```
严重威胁 → 立即规避 (Split-S/Notch/桶滚)
高威胁   → 防御机动 (防御转弯/规避机动)
中等威胁 → 平衡攻防 (攻击定位/防御转弯)
低威胁   → 主动攻击 (攻击性接敌/攻击定位)
无威胁   → 智能巡逻 (CAP巡逻/主动接敌)
```

### 3. 增强机动指令生成

#### 动作空间兼容性 [15,17,7]
- **高度指令**: 15个选项 (-1500m 到 +1500m)
- **航向指令**: 17个选项 (-180° 到 +180°)
- **速度指令**: 7个选项 (-150m/s 到 +150m/s)

#### 智能指令映射
- 根据目标方位计算最佳拦截航向
- 基于威胁方向选择规避方向
- 动态调整攻击角度和距离

### 4. 战术态势感知

#### 态势分析要素
- 敌我数量对比
- 最近敌机距离
- 能量优劣势状态
- 包围/被包围状态
- 导弹威胁数量

#### 智能决策支持
- 数量劣势时优先防御
- 能量优势时考虑反击
- 被包围时选择最佳突围方向

## 技术实现

### 核心类结构

```python
class EnemyTacticalAI:
    - evaluate_threat_level()      # 威胁评估
    - select_maneuver()           # 机动选择
    - execute_maneuver()          # 机动执行
    - _analyze_tactical_situation() # 态势分析
    - _generate_maneuver_commands() # 指令生成
```

### 关键算法

#### 威胁评估算法
1. 导弹威胁: 距离 + 速度 + 类型
2. 敌机威胁: 距离 + 接近速度 + 航向
3. 战术威胁: 数量 + 包围 + 能量状态
4. 综合评估: 最高威胁等级

#### 机动选择算法
1. 战场边界检查
2. 当前机动状态检查
3. 威胁等级评估
4. 战术态势分析
5. 最优机动选择

## 配置参数

### 威胁评估参数
```python
threat_ranges = {
    "missile_critical": 15000,   # 15km内导弹严重威胁
    "missile_high": 30000,       # 30km内导弹高威胁
    "missile_medium": 50000,     # 50km内导弹中等威胁
    "radar_lock": 45000,         # 45km内雷达锁定
    "enemy_close": 35000,        # 35km内敌机接近
    "enemy_medium": 60000        # 60km内敌机中等威胁
}
```

### 机动参数
```python
maneuver_params = {
    EnemyManeuverType.CAP_PATROL: {"duration": 30.0},
    EnemyManeuverType.AGGRESSIVE_APPROACH: {"duration": 20.0},
    EnemyManeuverType.DEFENSIVE_TURN: {"duration": 15.0},
    EnemyManeuverType.EVASIVE_MANEUVER: {"duration": 12.0},
    # ... 其他机动类型
}
```

## 使用方法

### 基本调用
```python
from enemy_tactical_ai import get_enemy_tactical_command

# 获取敌方战术指令
commands = get_enemy_tactical_command(env, agent_id, current_time)
altitude_cmd, heading_cmd, velocity_cmd = commands
```

### 集成到仿真
增强的敌方AI已自动集成到`drag_shoot_tactical_task.py`中，通过`_get_enemy_action()`方法调用。

## 测试验证

### 功能测试
```bash
python test_enhanced_enemy_ai.py
```

### 仿真测试
```bash
python test_enhanced_ai_simulation.py
```

### 完整仿真
```bash
python run_drag_shoot_simulation.py
```

## 性能指标

### 预期改进
- ✅ 敌方AI保持战斗参与，不离开战场
- ✅ 智能威胁响应，提高生存能力
- ✅ 动态战术调整，增加挑战性
- ✅ 兼容现有动作空间[15,17,7]
- ✅ 保持80%+编队间距合规率
- ✅ 支持1500步/300秒仿真持续时间

### 战术行为验证
- 导弹威胁时执行规避机动
- 数量劣势时采用防御策略
- 能量优势时主动攻击
- 被包围时选择最佳突围

## 注意事项

1. **兼容性**: 完全兼容现有的drag_shoot框架
2. **性能**: 优化的算法确保实时响应
3. **可调性**: 参数化设计便于调整
4. **稳定性**: 包含错误处理和边界检查
5. **可扩展性**: 模块化设计便于功能扩展

## 后续优化建议

1. 根据仿真结果调整威胁评估参数
2. 优化机动持续时间和强度
3. 增加更多高级战术机动
4. 实现协同作战逻辑
5. 添加学习和适应能力

---

**版本**: 1.0  
**更新日期**: 2025-08-08  
**作者**: Augment Agent  
**状态**: 已测试，准备部署
