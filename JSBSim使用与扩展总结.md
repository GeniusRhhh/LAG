# JSBSim在本项目中的使用与扩展总结

## 一、JSBSim底层简介

### 1.1 JSBSim是什么？
**JSBSim** (Flight Dynamics Model) 是一个开源的**六自由度飞行动力学仿真引擎**，由NASA和JSBSim团队开发。它是一个通用的、基于物理的飞行仿真库，能够精确模拟飞机在三维空间中的运动。

### 1.2 JSBSim核心功能
- **六自由度动力学**：模拟飞机的三个平移自由度（x, y, z）和三个旋转自由度（roll, pitch, yaw）
- **真实气动模型**：基于气动系数表和风洞数据
- **完整推进系统**：支持多种发动机类型（活塞、涡喷、涡扇等）
- **飞行控制系统**：副翼、升降舵、方向舵、油门控制
- **物理环境模拟**：大气模型、重力、科里奥利力
- **丰富的飞机模型库**：F-16、F-22、C172等几十种飞机模型

### 1.3 JSBSim的数据结构
```
JSBSim核心架构：
├── FGFDMExec (Flight Dynamics Model Executive) - 主仿真引擎
├── 物理模型
│   ├── 气动模型 (Aerodynamics)
│   ├── 推进模型 (Propulsion)
│   ├── 质量惯性 (Mass & Inertia)
│   └── 地面反作用力 (Ground Reactions)
├── 控制系统
│   ├── 飞行控制系统 (FCS)
│   └── 自动驾驶仪 (Autopilot)
└── 环境模型
    ├── 大气模型 (Atmosphere)
    └── 地球模型 (Earth)
```

---

## 二、本项目如何使用JSBSim

### 2.1 JSBSim在项目中的层次结构

```
项目架构（从上到下）：
┌──────────────────────────────────────┐
│  强化学习算法层 (PPO/MAPPO/SAC)     │  ← 决策AI
├──────────────────────────────────────┤
│  战术模板层 (TacticalTemplate)      │  ← 战术规则
├──────────────────────────────────────┤
│  任务层 (Task)                       │  ← 任务定义
├──────────────────────────────────────┤
│  环境包装层 (Env Wrapper)           │  ← Gym接口
├──────────────────────────────────────┤
│  仿真器层 (AircraftSimulator)       │  ← 飞机封装
├──────────────────────────────────────┤
│  JSBSim底层 (FGFDMExec)             │  ← 物理引擎
└──────────────────────────────────────┘
```

### 2.2 JSBSim初始化流程

#### （1）创建JSBSim实例（在`AircraftSimulator.reload()`中）
```python
# 位置：envs/JSBSim/core/simulatior.py，第278行
self.jsbsim_exec = jsbsim.FGFDMExec(os.path.join(get_root_dir(), 'data'))
self.jsbsim_exec.set_debug_level(0)
self.jsbsim_exec.load_model(self.model)  # 加载飞机模型（如'f16'）
```

**关键点**：
- `FGFDMExec` 是JSBSim的主执行对象
- 加载飞机数据目录：`envs/JSBSim/data/`（包含飞机XML配置）
- 支持的飞机模型：F-16, F-22, C172, Su-27等（见`data/aircraft/`目录）

#### （2）属性系统注册（Property Catalog）
```python
# 位置：envs/JSBSim/core/simulatior.py，第283-294行
jsbsim_props = self.jsbsim_exec.query_property_catalog("")
processed_props = []
for prop in jsbsim_props:
    if " " not in prop:
        prop = f"{prop} R"  # 默认添加只读权限
    processed_props.append(prop)
Catalog.add_jsbsim_props(processed_props)
```

**属性系统的作用**：
- JSBSim通过**属性树（Property Tree）**管理所有状态变量
- `Catalog`类封装了属性访问，提供统一接口
- 属性分为只读(R)和可写(W)两种权限

#### （3）设置初始条件
```python
# 位置：envs/JSBSim/core/simulatior.py，第303-308行
for key, value in self.init_state.items():
    self.set_property_value(Catalog[key], value)
success = self.jsbsim_exec.run_ic()  # 运行初始条件
if not success:
    raise RuntimeError("JSBSim failed to init simulation conditions.")
```

**初始条件示例**：
```python
init_state = {
    'ic_long_gc_deg': 120.0,       # 经度
    'ic_lat_geod_deg': 60.0,       # 纬度
    'ic_h_sl_ft': 20000,           # 高度（英尺）
    'ic_psi_true_deg': 0.0,        # 航向角
    'ic_u_fps': 1200.0,            # 前向速度（英尺/秒）
}
```

#### （4）启动发动机
```python
# 位置：envs/JSBSim/core/simulatior.py，第310-314行
propulsion = self.jsbsim_exec.get_propulsion()
n = propulsion.get_num_engines()
for j in range(n):
    propulsion.get_engine(j).init_running()  # 启动所有发动机
propulsion.get_steady_state()  # 达到稳态
```

### 2.3 仿真循环（每个时间步）

#### （1）设置控制指令
```python
# 位置：envs/JSBSim/core/simulatior.py，第496-504行
def set_property_value(self, prop, value):
    if isinstance(prop, Property):
        value = np.clip(value, prop.min, prop.max)  # 限幅
        self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)
```

**控制变量（4个舵面）**：
```python
# 位置：envs/JSBSim/tasks/task_base.py，第41-45行
action_var = [
    c.fcs_aileron_cmd_norm,   # 副翼 [-1, 1]
    c.fcs_elevator_cmd_norm,  # 升降舵 [-1, 1]
    c.fcs_rudder_cmd_norm,    # 方向舵 [-1, 1]
    c.fcs_throttle_cmd_norm,  # 油门 [0, 1]
]
```

#### （2）执行仿真步进
```python
# 位置：envs/JSBSim/core/simulatior.py，第359行
result = self.jsbsim_exec.run()  # JSBSim运行一个时间步（dt）
```

**核心物理计算**（JSBSim内部）：
1. 根据控制输入更新舵面偏转
2. 计算气动力和力矩（基于气动系数表）
3. 求解六自由度运动方程（龙格-库塔积分）
4. 更新飞机状态（位置、速度、姿态）

#### （3）读取状态更新
```python
# 位置：envs/JSBSim/core/simulatior.py，第456-473行
def _update_properties(self):
    # 读取地理坐标
    self._geodetic[:] = self.get_property_values([
        Catalog.position_long_gc_deg,    # 经度
        Catalog.position_lat_geod_deg,   # 纬度
        Catalog.position_h_sl_m          # 高度
    ])
    # 转换为北东地坐标
    self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)
    # 读取姿态角
    self._posture[:] = self.get_property_values([
        Catalog.attitude_roll_rad,       # 滚转角
        Catalog.attitude_pitch_rad,      # 俯仰角
        Catalog.attitude_heading_true_rad # 航向角
    ])
    # 读取速度
    self._velocity[:] = self.get_property_values([
        Catalog.velocities_v_north_mps,  # 北向速度
        Catalog.velocities_v_east_mps,   # 东向速度
        Catalog.velocities_v_down_mps    # 下向速度
    ])
```

### 2.4 JSBSim在项目中的控制流程

```
用户输入动作 → Task层归一化 → 设置JSBSim属性 → JSBSim运行 → 读取状态 → 计算奖励 → 返回给RL
    ↓               ↓                ↓               ↓           ↓           ↓
[高层指令]    [舵面控制]      [物理仿真]      [状态更新]    [任务评估]   [策略学习]
```

---

## 三、项目在JSBSim基础上的扩展

### 3.1 核心扩展内容

#### 扩展1：导弹动力学模拟器（**完全自主实现**）
**位置**：`envs/JSBSim/core/simulatior.py`，第545-1081行

**JSBSim不包含导弹**，项目完全自主实现了：

```python
class MissileSimulator(BaseSimulator):
    """AIM-120C7三段制导导弹模拟器（完全自主实现）"""
```

**实现细节**：
1. **物理模型**：
   - 三自由度质点模型（不依赖JSBSim）
   - 真实的推力模型（单脉冲/双脉冲）
   - 大气密度模型（对流层/平流层）
   - 空气阻力计算

2. **三段制导律**（基于经典导弹理论）：
   ```python
   def _guidance(self):
       if self._phase == BOOST_PHASE:
           return self._boost_guidance()      # 助推段：初始指向
       elif self._phase == MIDCOURSE_PHASE:
           return self._midcourse_guidance()  # 中段：预测拦截
       else:
           return self._terminal_guidance()   # 末段：比例导引
   ```

3. **比例导引律实现**（位置：第916-957行）：
   ```python
   def _terminal_guidance(self):
       # 计算视线角速率
       dbeta = (rel_dy * rel_x - rel_dx * rel_y) / Rxy ** 2
       deps = (rel_dz * Rxy ** 2 - rel_z * (rel_x * rel_dx + rel_y * rel_dy)) / (Rxyz ** 2 * Rxy)
       
       # 比例导引律
       ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
       nz = self.K * v_m / self._g * deps + np.cos(theta_m)
   ```

**导弹模型参数**（真实AIM-120C7数据）：
- 初始质量：161.5 kg
- 燃料质量：40.0 kg
- 推力：14000 N（助推段8秒）
- 最大过载：40 G
- 爆炸半径：40 m

#### 扩展2：机载雷达探测模型
**位置**：`envs/JSBSim/utils/RadarModel.py`

**实现功能**：
1. **脉冲多普勒雷达模拟**：
   - N001VE雷达（Su-27/Su-30）
   - AN/APG-68(V)9雷达（F-16C）

2. **探测概率计算**（基于信噪比）：
   ```python
   SNR = SNR_base + SNR_distance + SNR_angle + SNR_atm
   P_detect = f(SNR, RCS, distance, angle, clutter, ECM)
   ```

3. **多普勒盲区效应**：
   ```python
   v_r = np.dot(target_vel - radar_vel, los_direction)
   if abs(v_r) < v_notch:
       P_detect *= 0.08  # 进入盲区，探测概率急剧下降
   ```

4. **电子对抗（ECM/ECCM）**：
   - 噪声干扰
   - 欺骗干扰
   - 箔条干扰

#### 扩展3：战术规则模板系统
**位置**：`envs/JSBSim/tasks/TacticalTemplate.py`

**14种经典BVR战术模板**：
```python
tactical_templates = {
    1: "Crank",              # 战术偏置
    2: "Beam",               # 横向规避
    3: "Notch",              # 利用地面杂波
    4: "Skate",              # 拖曳射击
    5: "Short_Skate",        # 快速脱离
    6: "Banzai",             # 发射决策合并
    7: "Simple_F_Pole",      # 简单F极
    8: "Advanced_F_Pole",    # 高级F极
    9: "Pincer",             # 钳形夹击
    10: "Defensive_Split",   # 防御性分离
    11: "High_Low",          # 高低搭配
    12: "Engaging_Trail",    # 前后攻击
    13: "Loose_Deuce",       # 松散双机
    14: "Defensive_Sequence" # 防御序列
}
```

**战术模板执行流程**：
```
态势感知 → 威胁评估 → 模板选择 → 机动生成 → JSBSim执行
```

#### 扩展4：精细化战术机动库
**位置**：`envs/JSBSim/tasks/RefinedTacticalManeuvers.py`

**基础机动动作**（7种）：
1. 平飞 (MaintainHeading)
2. 加速 (Accelerate)
3. 减速 (Decelerate)
4. 转弯 (Turn)
5. 拉起 (Climb)
6. 俯冲 (Dive)
7. 斜向飞行 (TacticalTurn)

**复合机动**（3种）：
1. Crank（战术偏置转向）
2. Tactical Crank（高度耦合偏置）
3. Short Skate（三段式机动）

**机动参数化设计**：
```python
def crank(self, angle_deg: float, duration: float, velocity: float, altitude_delta: float):
    # 平滑曲线函数（三次多项式）
    f_smooth(t) = 3t² - 2t³  # 保证起止点导数为零
    
    # 航向变化
    ψ(t) = ψ₀ + Δψ × f_smooth(t/T)
    
    # 横滚角包络
    φ(t) = φ_max × sin(πt/T)
```

#### 扩展5：层次化控制架构

**三层控制结构**：

```
┌─────────────────────────────────────────┐
│  高层：战术决策（TacticalTemplate）    │
│  输出：目标状态 (Δh, Δψ, ΔV)          │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  中层：机动原语（RefinedManeuvers）    │
│  输出：机动轨迹参数                    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  低层：飞行控制（JSBSimController）    │
│  输出：舵面控制 (δa, δe, δr, T)       │
└─────────────────┬───────────────────────┘
                  │
                  ▼
            [JSBSim物理引擎]
```

**控制转换示例**：
```python
# 高层决策
tactical_action = {"heading_delta": 45, "altitude_delta": 500, "velocity_delta": 50}

# 中层机动生成
maneuver = RefinedManeuvers.tactical_turn(
    heading_change=45, 
    altitude_change=500,
    velocity_change=50
)

# 低层控制执行
controller = JSBSimController(jsbsim_exec)
control_cmd = controller.control(maneuver.get_action())

# JSBSim执行
jsbsim_exec.set_property_value(c.fcs_aileron_cmd_norm, control_cmd[0])
jsbsim_exec.set_property_value(c.fcs_elevator_cmd_norm, control_cmd[1])
jsbsim_exec.set_property_value(c.fcs_rudder_cmd_norm, control_cmd[2])
jsbsim_exec.set_property_value(c.fcs_throttle_cmd_norm, control_cmd[3])
jsbsim_exec.run()
```

#### 扩展6：Gym环境包装
**位置**：`envs/JSBSim/envs/`

**三种环境类型**：
1. **SingleControl**：单机航向控制
   - 训练底层飞行控制策略
   - 作为基线或低层策略

2. **SingleCombat**：1v1空战
   - 无武器姿态对抗
   - 带导弹射击任务

3. **MultipleCombat**：2v2协同空战
   - 多智能体协同
   - 战术配合

**Gym接口实现**：
```python
class BaseEnv(gym.Env):
    def reset(self) -> np.ndarray:
        # 重置JSBSim仿真器
        self.reset_simulators()
        return self.get_obs()
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        # 应用动作到JSBSim
        self.agents[agent_id].set_property_values(self.task.action_var, action)
        
        # 运行JSBSim步进
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()  # 调用JSBSim
        
        # 计算奖励和终止条件
        obs = self.get_obs()
        reward, info = self.task.get_reward(self, agent_id)
        done, info = self.task.get_termination(self, agent_id, info)
        
        return obs, reward, done, info
```

#### 扩展7：丰富的奖励函数系统
**位置**：`envs/JSBSim/reward_functions/`

**实现的奖励函数**（17种）：
1. `altitude_reward.py` - 高度保持奖励
2. `energy_advantage_reward.py` - 能量优势奖励
3. `event_driven_reward.py` - 事件驱动奖励
4. `flight_stability_reward.py` - 飞行稳定性奖励
5. `heading_reward.py` - 航向跟踪奖励
6. `missile_posture_reward.py` - 导弹姿态奖励
7. `MissileHitReward.py` - 导弹命中奖励
8. `posture_reward.py` - 姿态优势奖励
9. `RadarLockReward.py` - 雷达锁定奖励
10. `relative_altitude_reward.py` - 相对高度奖励
11. `shoot_penalty_reward.py` - 射击惩罚
12. `TacticalReward.py` - 战术奖励
13. ...

**奖励函数组合**：
```python
class MultipleCombatTask(BaseTask):
    def __init__(self, config):
        self.reward_functions = [
            PostureReward(config),
            MissilePostureReward(config),
            EventDrivenReward(config),
            AltitudeReward(config),
            MissileHitReward(config),
            RadarLockReward(config),
            TacticalReward(config)
        ]
```

#### 扩展8：Tacview渲染系统
**位置**：`envs/JSBSim/utils/TacviewRenderer.py`

**功能**：
- 将JSBSim仿真数据转换为Tacview格式（.acmi文件）
- 支持3D可视化回放
- 显示导弹轨迹、雷达锁定、爆炸效果

**渲染流程**：
```
JSBSim状态 → 日志记录 → ACMI格式 → Tacview播放器
```

---

## 四、仿真流程总结

### 4.1 完整仿真循环

```
1. 初始化阶段：
   ├─ 创建JSBSim实例 (FGFDMExec)
   ├─ 加载飞机模型 (F-16/Su-27)
   ├─ 注册属性目录 (Catalog)
   ├─ 设置初始条件 (位置/速度/姿态)
   └─ 启动发动机

2. 环境重置 (env.reset()):
   ├─ 随机化初始状态
   ├─ 重置JSBSim仿真器
   ├─ 重置任务状态
   └─ 返回初始观测

3. 仿真步进 (env.step(action)):
   ├─ 【RL决策】
   │   ├─ 策略网络输出动作
   │   ├─ 或战术模板生成动作
   │   └─ 动作归一化
   │
   ├─ 【控制执行】
   │   ├─ 设置JSBSim舵面控制
   │   ├─ jsbsim_exec.set_property_value()
   │   └─ jsbsim_exec.run()  ← JSBSim物理引擎运行
   │
   ├─ 【状态更新】
   │   ├─ 读取JSBSim状态
   │   ├─ 更新飞机位置/速度/姿态
   │   ├─ 更新导弹轨迹（自主计算）
   │   └─ 雷达探测计算（自主实现）
   │
   ├─ 【战术评估】
   │   ├─ 检查终止条件
   │   ├─ 计算奖励函数
   │   └─ 记录战术数据
   │
   └─ 返回 (obs, reward, done, info)

4. 可视化渲染 (可选):
   └─ TacviewRenderer生成ACMI文件
```

### 4.2 关键数据流

```
【强化学习训练流程】：
RL策略 → 高层指令 → 机动原语 → 舵面控制 → JSBSim → 飞机状态 → 观测 → RL策略
   ↑                                                                          ↓
   └──────────────────────── 奖励信号 ←──────────────────────────────────────┘

【战术模板流程】：
态势感知 → 威胁评估 → 模板选择 → 机动生成 → JSBSim → 效果评估 → 态势更新
```

---

## 五、项目的创新点

### 5.1 相比纯JSBSim的优势

| 方面 | 纯JSBSim | 本项目扩展 |
|------|----------|------------|
| 飞机动力学 | ✅ 完整实现 | ✅ 继承使用 |
| 导弹模拟 | ❌ 不支持 | ✅ 自主实现（三段制导） |
| 雷达系统 | ❌ 不支持 | ✅ 自主实现（多普勒盲区/ECM） |
| 战术决策 | ❌ 需手动编程 | ✅ 模板化+RL |
| 协同作战 | ❌ 单机 | ✅ 多智能体协同 |
| 强化学习 | ❌ 不支持 | ✅ Gym接口封装 |
| 可视化 | ⚠️ 基础 | ✅ Tacview专业渲染 |

### 5.2 技术创新点

1. **导弹-飞机耦合仿真**：
   - JSBSim负责飞机六自由度
   - 自主实现导弹三自由度
   - 实时交互（比例导引追踪JSBSim飞机）

2. **层次化控制架构**：
   ```
   战术层（模板/RL） → 机动层（原语） → 控制层（PID/JSBSim）
   ```

3. **真实战术知识融合**：
   - 14种经典BVR战术模板
   - 基于真实雷达性能参数
   - 符合实际作战流程

4. **多智能体协同框架**：
   - 支持2v2对抗
   - 长机-僚机协同
   - 战术角色分工

---

## 六、总结

### 6.1 JSBSim在项目中的作用
JSBSim作为**底层物理引擎**，为项目提供：
- ✅ 高保真的飞机六自由度动力学
- ✅ 真实的气动力学模型
- ✅ 完整的飞行控制系统
- ✅ 丰富的飞机模型库

### 6.2 项目的扩展贡献
在JSBSim基础上，项目自主实现了：
- ✅ **导弹动力学与三段制导律**（完全原创）
- ✅ **机载雷达探测与跟踪模型**（包含多普勒盲区、ECM）
- ✅ **14种战术规则模板系统**
- ✅ **层次化控制架构**（战术-机动-控制三层）
- ✅ **Gym强化学习接口**
- ✅ **多智能体协同框架**
- ✅ **Tacview专业可视化**

### 6.3 项目定位
本项目是一个**空战仿真+强化学习**的完整框架：
```
JSBSim物理引擎 + 导弹/雷达扩展 + 战术模板 + 强化学习 = 完整的BVR空战仿真平台
```

**适用场景**：
- 空战AI算法研发
- 战术策略验证
- 多智能体协同研究
- 强化学习算法测试

---

## 参考文献

1. JSBSim官方文档：https://jsbsim-team.github.io/jsbsim/
2. JSBSim GitHub仓库：https://github.com/JSBSim-Team/jsbsim
3. 导弹制导原理：Zarchan, P. (2012). Tactical and Strategic Missile Guidance
4. 空战战术：Shaw, R. L. (1985). Fighter Combat: Tactics and Maneuvering

---

**文档生成时间**：2025年1月
**项目地址**：https://github.com/GeniusRhhh/LAG



