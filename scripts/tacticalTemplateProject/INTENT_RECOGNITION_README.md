# 敌方意图识别数据集生成 - 实现说明

## 📋 修改概述

为生成敌方意图识别训练数据集，对以下文件进行了修改：

1. **scripts/tacticalProject/unified_enemy_tactical_ai.py** - 添加意图映射逻辑
2. **scripts/tacticalTemplateProject/unified_data_recorder.py** - 修改数据记录逻辑

## 🎯 核心设计

### 1. 函数级意图稳定性
- **原理**: 执行哪个函数 = 该意图标签
- **稳定性**: 函数执行时间(15-50秒)提供自然稳定性
- **动态切换**: 允许意图随战术需要切换
  - 攻击中检测到导弹 → 切换到规避函数 → 意图变为DEFENSE
  - 规避完成 → 返回攻击函数 → 意图恢复为ATTACK

### 2. 17动作类型 → 4意图标签映射

| 动作类型 | 意图标签 | 说明 |
|---------|---------|------|
| MAINTAIN_HEADING | RECONNAISSANCE | 保持航向=侦察 |
| TURN_LEFT | DEFENSE | 防御性转弯 |
| TURN_RIGHT | DEFENSE | 防御性转弯 |
| CLIMB | DEFENSE | 爬升规避 |
| DESCEND | DEFENSE | 下降规避 |
| ACCELERATE | ATTACK | 加速接近 |
| DECELERATE | DEFENSE | 减速拉开 |
| CRANK_LEFT | ATTACK | 战术机动 |
| CRANK_RIGHT | ATTACK | 战术机动 |
| NOTCH_MANEUVER | DEFENSE | 90°规避 |
| BEAM_MANEUVER | DEFENSE | 侧向规避 |
| DIVE_ESCAPE | DEFENSE | 俯冲脱离 |
| CHAFF_FLARE_MANEUVER | DEFENSE | 干扰弹规避 |
| SPIRAL_DIVE | DEFENSE | 螺旋俯冲 |
| SHORT_SKATE | DEFENSE | 3阶段防御脱离 |
| DEFENSIVE_SPLIT | DEFENSE | 防御分离 |
| AGGRESSIVE_APPROACH | ATTACK | 攻击接近 |
| RETURN_TO_BASE | RETREAT | 返航 |

**统计**:
- ATTACK: 3个动作
- DEFENSE: 13个动作
- RETREAT: 1个动作
- RECONNAISSANCE: 1个动作

## 📝 代码修改详情

### 修改1: unified_enemy_tactical_ai.py

#### 1.1 添加意图映射表 (第250-269行)

```python
# 🎯 意图识别映射表 - 17动作类型到4意图标签
self.action_to_intent_mapping = {
    ActionType.MAINTAIN_HEADING: "RECONNAISSANCE",
    ActionType.TURN_LEFT: "DEFENSE",
    ActionType.TURN_RIGHT: "DEFENSE",
    # ... 其他映射
}
```

#### 1.2 添加获取意图方法 (第783-791行)

```python
def get_current_intent(self, agent_id: str) -> str:
    """获取当前意图标签 - 基于当前执行的动作"""
    current_action = self.current_action.get(agent_id, ActionType.MAINTAIN_HEADING)
    return self.action_to_intent_mapping.get(current_action, "RECONNAISSANCE")

def get_current_action_type(self, agent_id: str) -> str:
    """获取当前动作类型名称"""
    current_action = self.current_action.get(agent_id, ActionType.MAINTAIN_HEADING)
    return current_action.value
```

### 修改2: unified_data_recorder.py

#### 2.1 修改轨迹记录方法 (第72-148行)

**关键修改**:

1. **为敌方飞机(B开头)添加意图数据**:
```python
if agent_id.startswith('B'):  # 敌方飞机
    # 获取敌方当前动作类型和意图标签
    action_type = tactical_task.unified_enemy_ai.get_current_action_type(agent_id)
    action_intent = tactical_task.unified_enemy_ai.get_current_intent(agent_id)
    
    trajectory_record['Action_Type'] = action_type
    trajectory_record['Action_Intent'] = action_intent
```

2. **添加我方雷达对敌方的探测状态**:
```python
# 遍历我方飞机(A开头)的雷达状态
for friendly_id in env._jsbsims.keys():
    if friendly_id.startswith('A'):
        radar_state = radar_manager.get_radar_state(friendly_id)
        if radar_state and radar_state.get('target_id') == agent_id:
            # 找到正在跟踪/锁定该敌机的我方雷达
            friendly_radar_type = radar_state.get('radar_type', 'Unknown')
            friendly_radar_status = radar_state.get('status', 'SEARCH')
            friendly_target_id = friendly_id
            break

trajectory_record['Friendly_Radar_Type'] = friendly_radar_type
trajectory_record['Friendly_Radar_Status'] = friendly_radar_status
trajectory_record['Friendly_Aircraft_ID'] = friendly_target_id
```

## 📊 输出数据格式

### CSV文件列定义

**敌方飞机(B开头)数据行**:
```
Time_s, Agent_ID, X_m, Y_m, Z_m, Velocity_m_s, Heading_deg, Pitch_deg, Roll_deg,
Action_Type, Action_Intent, Friendly_Radar_Type, Friendly_Radar_Status, Friendly_Aircraft_ID
```

**友方飞机(A开头)数据行**:
```
Time_s, Agent_ID, X_m, Y_m, Z_m, Velocity_m_s, Heading_deg, Pitch_deg, Roll_deg,
Action_Type, Action_Intent, Friendly_Radar_Type, Friendly_Radar_Status, Friendly_Aircraft_ID
(后5列为空字符串)
```

### 示例数据

**敌方飞机数据**:
```csv
10.5, B0100, 25000, 15000, 8000, 250, 180, 5, 10, 
crank_left, ATTACK, N001VE, TRACK, A0100

15.2, B0100, 24500, 14800, 8100, 245, 175, 3, 8, 
notch_maneuver, DEFENSE, N001VE, SEARCH, 
```

**说明**:
- `Action_Type`: 当前执行的动作类型(如crank_left, notch_maneuver)
- `Action_Intent`: 意图标签(ATTACK/DEFENSE/RETREAT/RECONNAISSANCE)
- `Friendly_Radar_Type`: 我方雷达型号(N001VE/APG-68)
- `Friendly_Radar_Status`: 我方雷达状态(SEARCH/TRACK/LOCK)
- `Friendly_Aircraft_ID`: 正在跟踪该敌机的我方飞机ID

## ⚠️ 重要说明

### 1. 雷达数据来源
- **雷达状态**: 来自我方雷达对敌方的探测
- **用途**: 用于训练意图识别模型(我方通过雷达观测敌方状态来识别意图)
- **不是**: 敌方自己的雷达状态

### 2. 意图切换机制
- 意图标签会随着执行函数的切换而动态变化
- 这是正确的行为,反映了真实的战术意图变化
- 例如: ATTACK → 检测到导弹 → DEFENSE → 规避完成 → ATTACK

### 3. 数据完整性
- 只有敌方飞机(B开头)有意图标签和雷达状态
- 友方飞机(A开头)这些列为空字符串
- 如果无法获取数据,使用默认值:
  - Action_Type: 'maintain_heading'
  - Action_Intent: 'RECONNAISSANCE'
  - Friendly_Radar_Type: 'Unknown'
  - Friendly_Radar_Status: 'SEARCH'

## 🧪 测试验证

运行测试脚本验证修改:
```bash
python scripts/tacticalTemplateProject/test_intent_recognition_data.py
```

测试内容:
1. 验证17动作类型到4意图标签的映射是否正确
2. 验证get_current_intent()方法是否正常工作
3. 验证get_current_action_type()方法是否正常工作

## 📚 使用方法

### 1. 运行仿真生成数据

```python
from unified_data_recorder import UnifiedDataRecorder

# 创建数据记录器
recorder = UnifiedDataRecorder(project_name="intent_recognition")

# 在仿真循环中记录数据
for step in range(max_steps):
    # ... 仿真步骤 ...
    recorder.record_all_data(env, current_time, tactical_task)

# 保存CSV文件
recorder.save_csv_files(output_dir="./intent_data")
```

### 2. 读取生成的数据

```python
import pandas as pd

# 读取轨迹数据
df = pd.read_csv("intent_recognition_trajectory_20260128_120000.csv")

# 筛选敌方数据
enemy_data = df[df['Agent_ID'].str.startswith('B')]

# 查看意图分布
print(enemy_data['Action_Intent'].value_counts())

# 查看动作类型分布
print(enemy_data['Action_Type'].value_counts())
```

## 🔍 数据质量检查

生成数据后,建议进行以下检查:

1. **意图标签分布**: 确保4类意图都有足够的样本
2. **意图切换频率**: 检查意图切换是否合理(不应该每帧都切换)
3. **雷达状态覆盖**: 确保包含SEARCH/TRACK/LOCK三种状态
4. **数据完整性**: 检查是否有缺失值或异常值

## 📞 问题反馈

如有问题,请检查:
1. unified_enemy_tactical_ai.py中的意图映射表是否完整
2. unified_data_recorder.py中的数据记录逻辑是否正确
3. 雷达管理器是否正常工作
4. 仿真环境是否正确初始化

---

**修改完成时间**: 2026-01-28
**修改人**: AI Assistant
**版本**: v1.0
