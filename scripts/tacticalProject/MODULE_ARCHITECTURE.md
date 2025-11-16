# 战术项目重构模块关系说明

## 模块结构

```
tacticalProject/
├── tactical_task.py           # 主文件（重构后~500行）
├── tactical_executor.py       # 战术执行器
├── maneuver_library.py        # 机动库
├── tactical_utils.py          # 工具函数
├── tactical_state_manager.py # 状态管理器
├── missile_manager.py         # 导弹管理器
└── tactical_decision_maker.py # 决策制定器
```

## 模块依赖关系

```
tactical_task.py (主控制器)
    ├── TacticalStateManager (状态管理)
    ├── MissileManager (导弹管理)
    ├── TacticalDecisionMaker (决策制定)
    ├── TacticalExecutor (战术执行)
    ├── ManeuverLibrary (机动库)
    └── TacticalUtils (工具函数)

tactical_executor.py
    └── 依赖 tactical_task (通过传入task实例)

maneuver_library.py
    └── 依赖 tactical_task (通过传入task实例)

tactical_utils.py
    └── 无依赖（静态工具类）

tactical_state_manager.py
    └── 依赖 tactical_task.TacticalPhase (枚举类型)

missile_manager.py
    ├── 依赖 tactical_state_manager
    ├── 依赖 tactical_utils
    └── 依赖 tactical_task (用于类型和辅助函数)

tactical_decision_maker.py
    ├── 依赖 tactical_state_manager
    ├── 依赖 tactical_utils
    └── 依赖 tactical_task.TacticalPhase
```

## 重构后的tactical_task.py结构

```python
# 1. 导入模块（约50行）
from tactical_state_manager import TacticalStateManager
from missile_manager import MissileManager
from tactical_decision_maker import TacticalDecisionMaker
from tactical_executor import TacticalExecutor
from maneuver_library import ManeuverLibrary
from tactical_utils import TacticalUtils

# 2. 辅助函数和枚举（约100行）
class TacticalPhase(Enum):
    ...

def get_target_with_fallback(agent_id, env):
    ...

# 3. TacticalTask类（约350行）
class TacticalTask:
    def __init__(self, ...):
        # 初始化各模块
        self.state_manager = TacticalStateManager()
        self.missile_manager = MissileManager(self.state_manager)
        self.decision_maker = TacticalDecisionMaker(...)
        self.executor = TacticalExecutor(self)
        self.maneuver_lib = ManeuverLibrary(self)
    
    def get_action(self, env, agent_id):
        # 主决策流程
        ...
    
    def _select_tactic_at_phase(self, env, agent_id):
        # 阶段决策
        ...
    
    # 其他必要的方法
    ...
```

## 模块功能说明

### 1. tactical_state_manager.py
**职责**: 管理所有战术状态
- 飞机阶段（NLT/MELD/MTR等）
- 机动状态（Short Skate, Beam, Crank等）
- 导弹状态（发射时间、剩余数量）
- 队形状态
- 决策结果缓存

**主要方法**:
- `set_agent_phase()` - 设置飞机阶段
- `get_agent_phase()` - 获取飞机阶段
- `record_missile_launch()` - 记录导弹发射
- `can_launch_missile()` - 检查能否发射
- `is_in_guidance_protection()` - 检查制导保护期
- `clear_maneuver_state()` - 清除机动状态
- `get_status_summary()` - 获取状态摘要

### 2. missile_manager.py
**职责**: 导弹相关操作
- 判断是否应该发射导弹
- 执行导弹发射
- 更新导弹状态
- 检查导弹威胁

**主要方法**:
- `should_launch_missile()` - 判断发射条件
- `execute_missile_launch()` - 执行发射
- `update_missile_status()` - 更新状态
- `get_active_missiles_count()` - 获取激活导弹数
- `check_missile_threat()` - 检查威胁

### 3. tactical_decision_maker.py
**职责**: 战术决策制定
- LR阶段决策（Crank或直飞）
- DOR阶段决策（规避机动选择）
- DR阶段决策（重新交战或返航）
- 目标分配

**主要方法**:
- `make_lr_decision()` - LR阶段决策
- `make_dor_decision()` - DOR阶段决策
- `make_dr_decision()` - DR阶段决策
- `decide_target_assignment()` - 目标分配
- `should_perform_evasion()` - 是否规避

### 4. tactical_executor.py
**职责**: 执行7种战术
- 拖曳射击（Drag Shoot）
- 钳形攻势（Pincer Attack）
- 上下夹击（High-Low Attack）
- 前后攻击（Front-Back Attack）
- 并排攻击（Side-by-Side）
- 战术规避（Tactical Evasion）

**主要方法**:
- `execute_drag_shoot()`
- `execute_pincer_attack()`
- `execute_high_low_attack()`
- `execute_front_back()`
- `execute_side_by_side()`
- `execute_tactical_evasion()`

### 5. maneuver_library.py
**职责**: 提供各种机动动作
- Short Skate（三阶段机动）
- Beam机动（三九机动）
- 战术Crank
- 战术爬升/下降
- Notch Back
- 队形保持

**主要方法**:
- `maintain_heading_precise()` - 精确保持航向
- `execute_short_skate_precise()` - Short Skate机动
- `execute_beam_maneuver()` - Beam机动
- `execute_tactical_crank()` - 战术Crank
- `execute_tactical_climb()` - 战术爬升
- `execute_tactical_descent()` - 战术下降
- `execute_notch_back()` - Notch Back机动
- `establish_rear_formation()` - 建立后方队形

### 6. tactical_utils.py
**职责**: 提供工具函数（静态方法）
- 距离计算
- 角度归一化
- 坐标转换
- CAP边界检查
- 态势参数计算

**主要方法**:
- `calculate_distance_between()` - 计算距离
- `normalize_angle_diff()` - 角度归一化
- `convert_heading_to_index()` - 航向转指令
- `convert_altitude_to_index()` - 高度转指令
- `convert_velocity_to_index()` - 速度转指令
- `get_enemy_bearing()` - 获取敌机方位
- `check_cap_boundary()` - 检查CAP边界
- `turn_to_heading()` - 转向到目标航向
- `calculate_aspect_angle()` - 计算aspect angle
- `calculate_antenna_train_angle()` - 计算ATA
- `calculate_closure_rate()` - 计算接近速率

## 使用示例

### 在tactical_task.py中使用模块

```python
# 初始化
class TacticalTask:
    def __init__(self, ...):
        self.state_manager = TacticalStateManager()
        self.missile_manager = MissileManager(self.state_manager)
        self.decision_maker = TacticalDecisionMaker(
            self.state_manager, 
            self.threat_evaluator, 
            self.situation_evaluator
        )
        self.executor = TacticalExecutor(self)
        self.maneuver_lib = ManeuverLibrary(self)

# 使用状态管理器
def get_action(self, env, agent_id):
    current_phase = self.state_manager.get_agent_phase(agent_id)
    
    # 检查导弹制导保护
    if self.state_manager.is_in_guidance_protection(agent_id, current_time):
        return 7, 8, 3  # 保持姿态
    
    # 执行战术
    if self.selected_tactic == 'DRAG_SHOOT':
        return self.executor.execute_drag_shoot(env, agent_id)

# 使用决策制定器
def _make_lr_decision(self, env, agent_id):
    maneuver = self.decision_maker.make_lr_decision(env, agent_id)
    self.state_manager.lr_maneuver[agent_id] = maneuver

# 使用机动库
def _perform_maneuver(self, env, agent_id):
    return self.maneuver_lib.execute_short_skate(env, agent_id)

# 使用工具函数
def _calculate_distance(self, aircraft1, aircraft2):
    return TacticalUtils.calculate_distance_between(aircraft1, aircraft2)
```

## 注意事项

1. **循环依赖**: tactical_executor和maneuver_library需要访问task实例，通过构造函数传入
2. **静态工具**: TacticalUtils全部是静态方法，不需要实例化
3. **状态管理**: 所有状态应该通过state_manager统一管理，避免在task中直接修改
4. **错误处理**: 每个模块都应该有适当的错误处理和日志记录
5. **测试**: 每个模块创建后都应该进行单元测试

## 重构优势

1. **可维护性**: 代码从4706行拆分为7个模块，每个模块职责清晰
2. **可测试性**: 每个模块可以独立测试
3. **可扩展性**: 新增战术或机动只需修改对应模块
4. **可读性**: 代码结构清晰，易于理解
5. **复用性**: 工具函数可以在其他项目中复用
