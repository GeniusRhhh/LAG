# Tactical Project 代码重构方案

## 现状分析

### 问题描述
- `tactical_task.py` 文件达到 **4706 行**，代码过于庞大
- 包含多个功能模块混杂：战术决策、机动执行、导弹管理、雷达系统等
- 难以维护和理解
- 不利于团队协作和代码复用

### 文件结构分析
```
tactical_task.py (4706行)
├── TacticalTask类 (主类)
├── 战术执行函数 (~1500行)
│   ├── _execute_drag_shoot
│   ├── _execute_pincer_attack
│   ├── _execute_high_low_attack
│   ├── _execute_front_back
│   ├── _execute_side_by_side
│   └── _execute_tactical_evasion
├── 机动库 (~800行)
│   ├── _execute_short_skate
│   ├── _execute_beam_maneuver
│   ├── _execute_tactical_crank
│   ├── _execute_tactical_climb
│   └── _execute_tactical_descent
├── 状态管理 (~600行)
│   ├── 阶段管理
│   ├── 导弹状态追踪
│   └── 机动状态管理
├── 决策系统 (~800行)
│   ├── _decide_at_nlt
│   ├── _decide_at_meld
│   ├── _decide_at_lr
│   ├── _decide_at_tr
│   └── _decide_at_dr
└── 工具函数 (~1000行)
    ├── 距离计算
    ├── 坐标转换
    ├── CAP边界检查
    └── 雷达/导弹管理
```

## 重构方案

### 模块化设计

#### 1. tactical_executor.py (✅已创建)
**职责**: 战术执行器
```python
class TacticalExecutor:
    - execute_drag_shoot()
    - execute_pincer_attack()
    - execute_high_low_attack()
    - execute_front_back()
    - execute_side_by_side()
    - execute_tactical_evasion()
    - execute_tactical_turn()
```

#### 2. maneuver_library.py (待创建)
**职责**: 机动动作库
```python
class ManeuverLibrary:
    - execute_short_skate()
    - execute_beam_maneuver()
    - execute_tactical_crank()
    - execute_tactical_climb()
    - execute_tactical_descent()
    - execute_notch_back()
    - maintain_heading_precise()
    - establish_rear_formation()
```

#### 3. tactical_state_manager.py (待创建)
**职责**: 状态管理
```python
class TacticalStateManager:
    - update_tactical_phase()
    - track_missile_state()
    - track_maneuver_state()
    - manage_agent_phases()
    - check_launch_conditions()
```

#### 4. tactical_decision_maker.py (待创建)
**职责**: 决策逻辑
```python
class TacticalDecisionMaker:
    - decide_at_nlt()
    - decide_at_meld()
    - decide_at_mtr()
    - decide_at_lr()
    - decide_at_tr()
    - decide_at_dor()
    - decide_at_dr()
```

#### 5. tactical_utils.py (待创建)
**职责**: 工具函数
```python
class TacticalUtils:
    - calculate_distance()
    - calculate_distance_between()
    - convert_heading_to_index()
    - convert_altitude_to_index()
    - convert_velocity_to_index()
    - check_cap_boundary()
    - get_enemy_bearing()
    - normalize_angle_diff()
```

#### 6. missile_manager.py (待创建)
**职责**: 导弹管理
```python
class MissileManager:
    - update_missiles()
    - handle_missile_launches()
    - launch_missile()
    - find_best_target()
```

#### 7. tactical_task.py (重构后)
**职责**: 核心任务类，协调各模块
```python
class TacticalTask(MultipleCombatTask):
    def __init__(self):
        self.executor = TacticalExecutor(self)
        self.maneuver = ManeuverLibrary(self)
        self.state_mgr = TacticalStateManager(self)
        self.decision_mgr = TacticalDecisionMaker(self)
        self.utils = TacticalUtils(self)
        self.missile_mgr = MissileManager(self)
    
    def step(self, env):
        # 简化的主循环
        pass
    
    def normalize_action(self, env, agent_id, action):
        # 协调各模块生成动作
        pass
```

### 重构步骤

#### 阶段1: 准备阶段 (✅已完成)
- [x] 分析代码结构
- [x] 设计模块化方案
- [x] 创建 tactical_executor.py
- [x] 测试原始代码运行正常

#### 阶段2: 模块提取 (进行中)
- [ ] 创建 maneuver_library.py
- [ ] 创建 tactical_state_manager.py
- [ ] 创建 tactical_decision_maker.py
- [ ] 创建 tactical_utils.py
- [ ] 创建 missile_manager.py

#### 阶段3: 集成测试
- [ ] 重构 tactical_task.py 主文件
- [ ] 导入并集成所有模块
- [ ] 运行测试，确保功能正常
- [ ] 修复集成问题

#### 阶段4: 优化完善
- [ ] 代码审查和优化
- [ ] 添加文档注释
- [ ] 性能测试
- [ ] 最终验收

## 实施建议

### 渐进式重构策略
1. **保持向后兼容**: 原 tactical_task.py 保留备份
2. **逐个模块迁移**: 每完成一个模块就测试
3. **增量测试**: 确保每一步都不破坏现有功能
4. **版本控制**: 使用git进行版本管理

### 优先级
1. **P0**: tactical_executor.py (✅已完成)
2. **P1**: maneuver_library.py + tactical_utils.py
3. **P2**: tactical_state_manager.py
4. **P3**: tactical_decision_maker.py + missile_manager.py

## 预期收益

### 代码质量
- ✅ 单个文件行数从 4706 减少到 ~500 行
- ✅ 模块职责清晰，符合单一职责原则
- ✅ 便于单元测试和维护

### 可维护性
- ✅ 新功能开发只需修改对应模块
- ✅ Bug修复范围缩小
- ✅ 代码复用性提高

### 团队协作
- ✅ 多人可并行开发不同模块
- ✅ 代码审查更加高效
- ✅ 新成员上手更快

## 使用示例

### 重构前
```python
# tactical_task.py (4706行)
class TacticalTask:
    def _execute_drag_shoot(self, env, agent_id):
        # 150行代码...
        pass
    
    def _execute_short_skate(self, env, agent_id):
        # 100行代码...
        pass
```

### 重构后
```python
# tactical_task.py (核心类，~500行)
from tactical_executor import TacticalExecutor
from maneuver_library import ManeuverLibrary

class TacticalTask:
    def __init__(self):
        self.executor = TacticalExecutor(self)
        self.maneuver = ManeuverLibrary(self)
    
    def normalize_action(self, env, agent_id, action):
        # 调用战术执行器
        if self.selected_tactic == 'DRAG_SHOOT':
            return self.executor.execute_drag_shoot(env, agent_id)
```

## 测试验证

### 功能测试
```bash
# 运行仿真测试
python scripts/tacticalProject/run_simulation.py
```

### 预期结果
- ✅ 仿真正常运行
- ✅ 战术执行正常
- ✅ 无报错或异常

## 当前状态

### 已完成 ✅
- tactical_executor.py 创建完成
- 原始代码测试运行成功（330秒仿真，平局结束）
- 重构方案设计完成

### 下一步
1. 创建 maneuver_library.py
2. 创建 tactical_utils.py
3. 逐步集成测试

## 注意事项

⚠️ **重要提示**:
1. 重构过程中保持原文件备份
2. 每次修改后立即测试
3. 遇到问题及时回滚
4. 保持代码风格一致
5. 添加必要的注释和文档

## 联系方式

如有问题，请联系项目负责人。

---
最后更新: 2025-11-13
状态: 进行中 (阶段1已完成)
