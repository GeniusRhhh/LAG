# 双机钳形夹击战术系统架构设计

## 1. 总体架构

### 1.1 设计原则
- **模块化设计**: 每种战术独立封装，便于维护和扩展
- **组件复用**: 最大化复用现有基础设施
- **策略模式**: 支持动态切换不同战术类型
- **扩展性**: 为未来添加更多战术类型预留接口

### 1.2 核心架构图
```
TacticalFramework
├── BaseTacticalTask (抽象基类)
│   ├── DragShootTacticalTask (拖曳射击)
│   ├── PincerAttackTacticalTask (钳形夹击) [新增]
│   └── [Future Tactical Tasks] (未来战术)
├── SharedComponents (共享组件)
│   ├── PhaseManager (阶段管理)
│   ├── ManeuverExecutor (机动执行)
│   ├── RadarManager (雷达管理)
│   ├── FormationCoordinator (编队协调)
│   └── MissileController (导弹控制)
└── TacticalFactory (战术工厂)
```

## 2. 可复用组件分析

### 2.1 完全可复用组件
1. **距离阶段管理系统**
   - `TacticalPhase` 枚举 (NLT_MELD, MELD_MTR, MTR_TR, TR_DOR, DOR_DR)
   - `_update_tactical_phase()` 方法
   - `_get_phase_by_distance()` 方法
   - 距离计算逻辑

2. **基础机动模块**
   - `BasicManeuvers` 类 (Crank, Short Skate, Turn等)
   - `CompositeManeuverExecutor` 类
   - `_maintain_heading_precise()` 方法
   - 航向/高度/速度控制逻辑

3. **雷达管理系统**
   - `RadarManager` 类
   - 雷达状态更新逻辑
   - 目标检测和跟踪

4. **导弹发射控制**
   - `_handle_missile_launch()` 基础框架
   - 导弹冷却时间管理
   - 发射条件检查

5. **底层飞行控制**
   - `_use_lowlevel_policy()` 方法
   - BaselineActor 模型
   - 动作空间转换逻辑

### 2.2 需要适配的组件
1. **编队协调机制**
   - 基础框架可复用
   - 需要针对钳形战术调整协调逻辑

2. **战术决策逻辑**
   - 阶段转换条件需要重新定义
   - 机动选择策略需要重新设计

## 3. 钳形夹击战术特有组件

### 3.1 钳形机动控制器
```python
class PincerManeuverController:
    """钳形机动控制器"""
    def execute_pincer_spread(self, env, leader_id, wingman_id)
    def execute_pincer_converge(self, env, leader_id, wingman_id)
    def calculate_optimal_crank_angles(self, enemy_position, formation_center)
```

### 3.2 钳形编队协调器
```python
class PincerFormationCoordinator:
    """钳形编队协调器"""
    def maintain_pincer_formation(self, env, current_time)
    def coordinate_pincer_timing(self, env, leader_id, wingman_id)
    def monitor_pincer_geometry(self, env)
```

### 3.3 钳形发射决策器
```python
class PincerLaunchDecisionMaker:
    """钳形发射决策器"""
    def evaluate_launch_window(self, env, agent_id, target_id)
    def calculate_off_boresight_angle(self, launcher_pos, target_pos, launcher_heading)
    def decide_launch_timing(self, pincer_phase, geometry_status)
```

## 4. 实现计划

### 阶段1: 基础架构重构 (1-2天)
1. 创建 `BaseTacticalTask` 抽象基类
2. 重构现有 `DragShootTacticalTask` 继承新基类
3. 提取共享组件到独立模块

### 阶段2: 钳形战术核心实现 (2-3天)
1. 实现 `PincerAttackTacticalTask` 类
2. 开发钳形特有的机动控制逻辑
3. 实现钳形编队协调机制

### 阶段3: 集成测试和优化 (1-2天)
1. 集成测试钳形战术
2. 性能优化和参数调优
3. 文档完善

### 阶段4: 扩展性验证 (1天)
1. 验证架构的扩展性
2. 为未来战术类型预留接口
3. 代码重构和清理
