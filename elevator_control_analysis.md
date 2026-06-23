# 升降舵控制路径分析 (Elevator Control Path Analysis)

## 执行摘要
**重点发现：升降舵命令 (action[1]) 的完整路径中存在潜在的符号问题**

---

## 1. 完整的控制流路径

### 第一步：环境接收 action
**文件**: [d:\Pycharm\LAG\envs\JSBSim\envs\env_base.py](envs/JSBSim/envs/env_base.py#L130)

```python
def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """第 130 行"""
    self.current_step += 1
    info = {"current_step": self.current_step}
    
    # 第 133 行: 首先解包 action
    action = self._unpack(action)
    
    # 第 134-136 行: 为每个 agent 规范化并应用 action
    for agent_id in self.agents.keys():
        a_action = self.task.normalize_action(self, agent_id, action[agent_id])
        self.agents[agent_id].set_property_values(self.task.action_var, a_action)
```

**关键信息**:
- `action` 是多个 agent 的聚合数据
- 通过 `_unpack()` 分离为单个 agent 的 action
- 通过 `normalize_action()` 将 action 从离散转换为连续值
- 通过 `set_property_values()` 应用到 JSBSim

---

### 第二步：Action 变量定义
**文件**: [d:\Pycharm\LAG\envs\JSBSim\tasks\task_base.py](envs/JSBSim/tasks/task_base.py#L37-L44)

```python
def load_variables(self):
    """第 37-44 行"""
    self.action_var = [
        c.fcs_aileron_cmd_norm,     # action[0]: 副翼 [-1, 1]
        c.fcs_elevator_cmd_norm,    # action[1]: 升降舵 [-1, 1]
        c.fcs_rudder_cmd_norm,      # action[2]: 方向舵 [-1, 1]
        c.fcs_throttle_cmd_norm,    # action[3]: 油门 [0.4, 0.9]
    ]
```

**关键信息**:
- 升降舵命令对应 `action[1]`
- 它联系到 `c.fcs_elevator_cmd_norm` 属性

---

### 第三步：Catalog 中的属性定义
**文件**: [d:\Pycharm\LAG\envs\JSBSim\core\catalog.py](envs/JSBSim/core/catalog.py#L193)

```python
fcs_elevator_cmd_norm = Property("fcs/elevator-cmd-norm", 
                                  "elevator commanded position, normalised", 
                                  -1.0,    # min value
                                  1.0)     # max value
```

**关键信息**:
- JSBSim 属性名: `"fcs/elevator-cmd-norm"`
- 值的范围: `-1.0 到 1.0`
- `-1.0` = 完全向下 (dive/pitch down)
- `+1.0` = 完全向上 (climb/pitch up)

---

### 第四步：normalize_action 方法 - HeadingTask
**文件**: [d:\Pycharm\LAG\envs\JSBSim\tasks\heading_task.py](envs/JSBSim/tasks/heading_task.py#L154-L160)

```python
def normalize_action(self, env, agent_id, action):
    """Convert discrete action index into continuous value."""
    norm_act = np.zeros(4)
    norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
    norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.  # 升降舵
    norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
    norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
    return norm_act
```

**action_space 定义** (第 94 行):
```python
self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])
```

**对于 elevator (action[1]) 的计算**:
```
nvec[1] = 41 (0-40 离散值)
norm_act[1] = action[1] * 2. / (41 - 1.) - 1.
            = action[1] * 2. / 40. - 1.
            = action[1] * 0.05 - 1.

// 离散值范围对应:
// action[1] = 0  →  norm_act[1] = 0 * 0.05 - 1 = -1.0  (完全向下)
// action[1] = 20 →  norm_act[1] = 20 * 0.05 - 1 = 0.0  (中立)
// action[1] = 40 →  norm_act[1] = 40 * 0.05 - 1 = 1.0  (完全向上)
```

✅ **HeadingTask 中：没有符号反转**

---

### 第五步：normalize_action 方法 - SingleCombatTask
**文件**: [d:\Pycharm\LAG\envs\JSBSim\tasks\singlecombat_task.py](envs/JSBSim/tasks/singlecombat_task.py#L175-L212)

```python
def normalize_action(self, env, agent_id, action):
    """Convert discrete action index into continuous value."""
    # ... 确保 action 是 ValueError 处理代码 ...
    
    norm_act = np.zeros(4)
    try:
        norm_act[0] = float(action[0]) / 20 - 1.           # 副翼
        norm_act[1] = float(action[1]) / 20 - 1.           # 升降舵 ⚠️
        norm_act[2] = float(action[2]) / 20 - 1.           # 方向舵
        norm_act[3] = float(action[3]) / 58 + 0.4          # 油门
    except (ValueError, TypeError) as e:
        logging.error(f"Error normalizing action {action}: {e}")
        norm_act = np.array([0.0, 0.0, 0.0, 0.7])
    
    # 限制范围
    norm_act[0] = np.clip(norm_act[0], -1.0, 1.0)  # aileron
    norm_act[1] = np.clip(norm_act[1], -1.0, 1.0)  # elevator ✅
    norm_act[2] = np.clip(norm_act[2], -1.0, 1.0)  # rudder
    norm_act[3] = np.clip(norm_act[3], 0.4, 0.9)   # throttle
    
    return norm_act
```

**对于 elevator 的计算**:
```
norm_act[1] = action[1] / 20 - 1

// 离散值范围对应:
// action[1] = 0  →  norm_act[1] = 0/20 - 1 = -1.0  (完全向下)
// action[1] = 20 →  norm_act[1] = 20/20 - 1 = 0.0  (中立)
// action[1] = 40 →  norm_act[1] = 40/20 - 1 = 1.0  (完全向上)
```

✅ **SingleCombatTask 中：也没有符号反转**

---

### 第六步：set_property_values 方法
**文件**: [d:\Pycharm\LAG\envs\JSBSim\core\simulatior.py](envs/JSBSim/core/simulatior.py#L761-L788)

```python
def set_property_values(self, props, values):
    """第 761 行"""
    if len(props) != len(values):
        logging.error(f"Property-value mismatch: props={len(props)}, values={len(values)}")
        raise ValueError("mismatch between properties and values size")
    for prop, value in zip(props, values):
        self.set_property_value(prop, value)

def set_property_value(self, prop, value):
    """第 778 行"""
    if isinstance(prop, Property):
        # ⚠️ 关键行：值被 clipped 到 min 到 max 之间
        value = np.clip(value, prop.min, prop.max)
        self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)
        if "W" in prop.access and prop.update:
            prop.update(self)
    elif isinstance(prop, str):
        self.jsbsim_exec.set_property_value(prop, value)
    else:
        logging.error(f"Invalid prop type: {type(prop)}")
        raise ValueError(f"prop type unhandled: {type(prop)}")
```

**对于 fcs_elevator_cmd_norm 的处理**:
```python
# prop = c.fcs_elevator_cmd_norm
# prop.min = -1.0
# prop.max = 1.0
# value 从 normalize_action 得到，范围 -1.0 到 1.0

value = np.clip(value, -1.0, 1.0)  # 无变化（已在范围内）
self.jsbsim_exec.set_property_value("fcs/elevator-cmd-norm", value)
```

✅ **set_property_value 中：没有符号反转**

---

## 2. 完整的数据流追踪示例

### 场景 1：发送向下俯冲命令
```
Neural Network 输出: action[1] = 0
    ↓
normalize_action(): norm_act[1] = 0/20 - 1 = -1.0  ✅ 向下
    ↓
set_property_value(): np.clip(-1.0, -1.0, 1.0) = -1.0  ✅ 向下
    ↓
JSBSim: fcs/elevator-cmd-norm = -1.0  ✅ 向下俯冲
```

### 场景 2：发送向上爬升命令
```
Neural Network 输出: action[1] = 40
    ↓
normalize_action(): norm_act[1] = 40/20 - 1 = 1.0  ✅ 向上
    ↓
set_property_value(): np.clip(1.0, -1.0, 1.0) = 1.0  ✅ 向上
    ↓
JSBSim: fcs/elevator-cmd-norm = 1.0  ✅ 向上爬升
```

### 场景 3：发送中立命令
```
Neural Network 输出: action[1] = 20
    ↓
normalize_action(): norm_act[1] = 20/20 - 1 = 0.0  ✅ 中立
    ↓
set_property_value(): np.clip(0.0, -1.0, 1.0) = 0.0  ✅ 中立
    ↓
JSBSim: fcs/elevator-cmd-norm = 0.0  ✅ 水平飞行
```

---

## 3. 特殊情况：HierarchicalMultipleCombatTask
**文件**: [d:\Pycharm\LAG\envs\JSBSim\tasks\multiplecombat_task.py](envs/JSBSim/tasks/multiplecombat_task.py#L459-L492)

```python
def normalize_action(self, env, agent_id, action):
    """归一化分层动作，使用低级策略生成控制命令。第 459 行"""
    if agent_id not in env.agents or not env.agents[agent_id].is_alive:
        return np.array([0.0, 0.0, 0.0, 0.7])
    
    # ... 建立观测 ...
    
    # 第 480-486 行: 调用低级策略网络后的处理
    norm_act = np.zeros(4)
    norm_act[0] = action_output[0] / 20 - 1.   # Aileron: [-1, 1]
    norm_act[1] = action_output[1] / 20 - 1.   # Elevator: [-1, 1]  ✅
    norm_act[2] = action_output[2] / 20 - 1.   # Rudder: [-1, 1]
    norm_act[3] = action_output[3] / 58 + 0.4  # Throttle: [0.4, 0.9]
    
    # 第 489-496 行: 极端情况处理
    current_alt = env.agents[agent_id].get_position()[2]
    if current_alt < 500:  # 只在极低高度介入
        norm_act[1] = max(norm_act[1], 0.0)    # ⚠️ 限制为非负（不允许向下）
        norm_act[3] = max(norm_act[3], 0.8)    # 增加推力
        logging.warning(f"Agent {agent_id} emergency altitude: {current_alt:.1f}m")
    
    return norm_act
```

**注意**: 在极低高度 (<500m) 时，elevator 被强制限制为 >= 0.0，防止继续下俯。这是一个安全特性。

---

## 4. 可能的问题区域

### 问题 1：HeadingTask vs SingleCombatTask 的不一致
- **HeadingTask**: 使用 `action[i] * 2. / (nvec[i] - 1.) - 1.` 的标准公式
- **SingleCombatTask**: 使用 `action[i] / 20 - 1.` 的简化公式
- 当 `nvec[i] = 41` 时，这两个公式是等价的，但代码风格不一致

### 问题 2：是否存在其他地方的符号反转？

让我检查 simulator 中是否有任何直接修改 elevator 命令的地方...
