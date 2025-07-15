# 独立机动函数系统使用指南

## 概述

本系统实现了完全独立、参数化的战术机动函数，符合老师的要求：
- ✅ **独立封装**：机动函数不依赖实时威胁状态
- ✅ **参数化**：通过输入参数控制机动
- ✅ **数学模型清晰**：每个机动有明确的数学模型
- ✅ **易于调整**：参数容易修改
- ✅ **轨迹记录**：能记录机动轨迹

## 系统架构

### 1. 核心组件

```
RefinedTacticalManeuvers.py
├── ManeuverParameters (数据类) - 参数封装
├── ManeuverTrajectory (轨迹类) - 轨迹记录
└── RefinedTacticalManeuvers (主类) - 机动函数
```

### 2. 机动分类

#### 独立机动（可单独测试）
- **Crank**：30-60度偏航维持雷达锁定
- **Beam**：90度横向机动消耗导弹动能
- **Notch**：地面杂波遮蔽机动
- **F-Pole**：保持锁定最大化F-pole

#### 需要配合的机动（需要敌方配合）
- **Skate**：发射-机动-转冷-再攻击序列
- **Short_Skate**：发射后快速脱离

## 使用方法

### 1. 基本使用

```python
from envs.JSBSim.tasks.RefinedTacticalManeuvers import RefinedTacticalManeuvers, ManeuverParameters

# 创建机动系统
maneuvers = RefinedTacticalManeuvers()

# 设置参数
params = ManeuverParameters(
    initial_heading=0.0,      # 初始航向角(弧度)
    initial_altitude=5000.0,  # 初始高度(米)
    initial_velocity=600.0,   # 初始速度(节)
    maneuver_duration=120.0,  # 机动持续时间(秒)
    crank_angle=np.radians(45)  # Crank偏置角(弧度)
)

# 执行机动
action = maneuvers.execute_crank_maneuver(params, current_time=60.0)
print(f"航向指令: {action['heading_cmd']}")
print(f"速度指令: {action['velocity_cmd']}")
print(f"机动阶段: {action['crank_phase']}")
```

### 2. 参数化控制

```python
# 调整Crank角度
params.crank_angle = np.radians(30)  # 30度
params.crank_angle = np.radians(60)  # 60度

# 调整速度
params.initial_velocity = 700.0  # 700节

# 调整高度
params.initial_altitude = 8000.0  # 8000米

# 调整持续时间
params.maneuver_duration = 180.0  # 180秒
```

### 3. 轨迹记录

```python
# 执行完整机动并记录轨迹
for time_step in range(0, 121, 10):  # 每10秒记录一次
    maneuvers.execute_crank_maneuver(params, time_step)

# 获取轨迹数据
trajectory_data = maneuvers.get_trajectory_data()
print(f"总点数: {trajectory_data['total_points']}")
print(f"持续时间: {trajectory_data['duration']}秒")
print(f"总飞行距离: {trajectory_data['distance_traveled']}米")

# 绘制轨迹图
maneuvers.plot_trajectory("trajectory.png")
```

## 数学模型

### 1. Crank机动

**转向模型**: θ(t) = θ₀ + ω·t·sign(θ_target - θ₀)
- θ₀：初始航向角
- ω：转向速率
- θ_target：目标航向角

**速度模型**: v(t) = v₀ + a_v·t
- v₀：初始速度
- a_v：加速度

**高度模型**: h(t) = h₀ (保持高度)

**位置模型**: 
- x(t) = x₀ + ∫v(t)·cos(θ(t))dt
- y(t) = y₀ + ∫v(t)·sin(θ(t))dt

### 2. Beam机动

**横向机动**: θ(t) = θ₀ + ω·t (目标90度)
**多普勒最小化**: 保持垂直角度
**位置模型**: 横向位移最大化

### 3. Notch机动

**高度下降**: h(t) = h₀ - a_h·t
**横向机动**: 结合Beam
**杂波效果**: C(h) = max(0, (h_clutter - h) / h_clutter)
**位置模型**: 下降轨迹

### 4. F-Pole机动

**距离优化**: d(t) = d₀ + v·t·cos(θ)
**角度保持**: θ(t) = θ_optimal
**射击窗口**: W(d) = 1 if d_min < d < d_max else 0
**位置模型**: 保持最佳射击距离

## 测试方法

### 1. 独立机动测试

```bash
# 测试Crank机动
python scripts/train/maneuver_demo.py --maneuver crank --duration 120 --angle 45

# 测试Beam机动
python scripts/train/maneuver_demo.py --maneuver beam --duration 150 --angle 90

# 测试Notch机动
python scripts/train/maneuver_demo.py --maneuver notch --duration 180

# 测试F-Pole机动
python scripts/train/maneuver_demo.py --maneuver fpole --duration 180
```

### 2. 需要配合的机动测试

对于Skate和Short_Skate，需要在实际环境中测试：

```bash
# 使用run_tactical_template.py测试
python scripts/train/run_tactical_template.py --template_id 4 --config test_tactical_1v1
python scripts/train/run_tactical_template.py --template_id 5 --config test_tactical_1v1
```

### 3. 参数调整测试

```bash
# 测试不同角度
python scripts/train/maneuver_demo.py --maneuver crank --angle 30
python scripts/train/maneuver_demo.py --maneuver crank --angle 45
python scripts/train/maneuver_demo.py --maneuver crank --angle 60

# 测试不同速度
python scripts/train/maneuver_demo.py --maneuver crank --velocity 500
python scripts/train/maneuver_demo.py --maneuver crank --velocity 600
python scripts/train/maneuver_demo.py --maneuver crank --velocity 700
```

## 集成到现有系统

### 1. 在TacticalTemplate中使用

```python
# 在TacticalTemplate.py中
from .RefinedTacticalManeuvers import RefinedTacticalManeuvers, ManeuverParameters

class EnhancedTacticalTemplate:
    def __init__(self):
        self.maneuvers = RefinedTacticalManeuvers()
    
    def get_tactical_action(self, template_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        # 创建参数
        params = self._create_maneuver_parameters(state, template_id)
        
        # 获取当前时间
        current_time = getattr(self.env, 'current_step', 0) * 0.1
        
        # 使用独立机动函数
        return self.maneuvers.get_maneuver_action(template_id, params, current_time)
```

### 2. 在测试任务中使用

```python
# 在tactical_template_test_task.py中
def _process_my_tactical_action(self, env, agent_id, action):
    template_id = self.test_template_id
    
    # 使用独立机动函数
    if agent_id in self.tactical_templates:
        state_dict = self.get_basic_state_dict(env, agent_id)
        tactical_action = self.tactical_templates[agent_id].get_tactical_action(template_id, state_dict)
        
        # 记录轨迹数据
        self._record_trajectory_data(env, agent_id, tactical_action)
```

## 输出文件

### 1. ACMI文件（TacView演示）
- 通过`run_tactical_template.py`生成
- 可在TacView中查看动态演示

### 2. 轨迹图
- PNG格式的轨迹图
- 包含2D轨迹、高度变化、速度变化、航向变化

### 3. 数据文件
- JSON格式的详细数据
- 包含轨迹点、机动参数、测试指标

## 常见问题

### Q1: 如何调整机动参数？
A1: 修改`ManeuverParameters`中的相应参数，如`crank_angle`、`initial_velocity`等。

### Q2: 独立机动和需要配合的机动有什么区别？
A2: 
- 独立机动：只涉及自身运动，可以单独测试
- 需要配合的机动：涉及发射导弹、敌方反应等，需要完整环境

### Q3: 如何查看机动效果？
A3: 
- 使用`maneuver_demo.py`查看参数化效果
- 使用`run_tactical_template.py`生成ACMI文件在TacView中查看
- 使用`plot_trajectory()`生成轨迹图

### Q4: 数学模型在哪里？
A4: 每个机动函数的文档字符串中都详细说明了数学模型。

## 总结

这个系统完全符合老师的要求：
1. ✅ **独立封装**：每个机动函数都是独立的，不依赖实时状态
2. ✅ **参数化**：通过`ManeuverParameters`数据类控制所有参数
3. ✅ **数学模型清晰**：每个机动都有明确的数学公式
4. ✅ **易于调整**：参数可以轻松修改，立即生效
5. ✅ **轨迹记录**：完整的轨迹记录和可视化功能

使用这个系统，您可以：
- 独立测试各种机动
- 轻松调整参数
- 查看清晰的数学模型
- 记录完整的轨迹数据
- 在TacView中查看动态演示 