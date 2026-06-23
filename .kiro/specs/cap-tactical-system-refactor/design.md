# CAP战术系统重构设计文档

## 一、待确认问题回答

基于代码分析，对需求文档中的待确认问题给出以下回答：

### 问题1：巡逻路径形状
**当前实现**：四边形跑马圈（有横向段 TRANS_WEST/TRANS_EAST）
**建议**：保持当前四边形实现，原因：
- 原地转弯（两条平行线+半圆）需要更复杂的编队协调
- 四边形跑马圈已经能满足冷热段交替需求
- 如需改为原地转弯，可作为后续优化

### 问题2：4机 vs 2机
**当前实现**：已支持4机（2×2编队），见 `formation_manager.py`
**结论**：无需修改，当前已满足需求

### 问题3：距离参数
**当前实现**：需求文档中的距离参数已与代码对齐
**需要新增**：PR（态势距离）250-300km 节点

### 问题4：原地转弯细节
**当前实现**：使用四边形跑马圈，长僚机Y偏移100km
**建议**：暂不修改，保持当前实现

---

## 二、架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        CAPTask (主任务类)                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    任务管理层                            │   │
│  │  - FAOR管理 (FAORManager)                               │   │
│  │  - 编队管理 (FormationManager)                          │   │
│  │  - 任务指标判定 (MissionEvaluator)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    状态机层                              │   │
│  │  CAPStateMachine: PATROL → INTERCEPT → ENGAGE → RTB     │   │
│  │                              ↓                           │   │
│  │                           EVADE                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    感知层 (Picture)                      │   │
│  │  - MockAwacsDataSource (预警机数据)                     │   │
│  │  - RadarTrackManager (雷达跟踪)                         │   │
│  │  - TrackFusion (数据融合)                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    战术执行层                            │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │   │
│  │  │ 协同探测    │ │ 协同跟踪    │ │ 接力制导    │       │   │
│  │  │ Detection   │ │ Tracking    │ │ Guidance    │       │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘       │   │
│  │  ┌─────────────────────────────────────────────┐       │   │
│  │  │        TacticalTask (现有战术模板)           │       │   │
│  │  │  T_DS, T_PA, T_HL, T_FB, T_SBS, T_TE, T_TT  │       │   │
│  │  └─────────────────────────────────────────────┘       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
cap/
├── cap_task.py              # 主任务类 (修改)
├── cap_state_machine.py     # CAP状态机 (新增)
├── picture/                 # 感知层 (新增)
│   ├── __init__.py
│   ├── picture.py           # 统一态势图接口
│   ├── awacs_source.py      # 预警机数据源
│   ├── radar_track.py       # 雷达跟踪管理
│   └── track_fusion.py      # 数据融合
├── tactics/                 # 协同战术 (新增)
│   ├── __init__.py
│   ├── cooperative_detection.py   # 协同探测
│   ├── cooperative_tracking.py    # 协同跟踪
│   └── relay_guidance.py          # 接力制导
├── mission_evaluator.py     # 任务评估 (新增)
├── faor_manager.py          # FAOR管理 (修改)
├── formation_manager.py     # 编队管理 (修改)
├── patrol_state_machine.py  # 巡逻状态机 (保留)
└── coordinate_system.py     # 坐标系统 (保留)
```

---

## 三、核心类设计

### 3.1 感知层接口 (Picture)

```python
@dataclass
class Track:
    """目标航迹"""
    track_id: int
    position: Tuple[float, float, float]  # (x, y, alt) km
    velocity: Tuple[float, float, float]  # (vx, vy, vz) m/s
    heading: float                         # 航向 (度)
    timestamp: float                       # 时间戳
    source: str                            # 数据源: 'awacs' | 'radar'
    confidence: float                      # 置信度 0-1
    
@dataclass  
class FusedTrack(Track):
    """融合后的航迹"""
    awacs_track: Optional[Track]
    radar_tracks: List[Track]
    fusion_quality: float
    lost_timer: float                      # 丢失计时 (秒)
    group_label: str                       # 编队标注

class Picture:
    """统一态势图接口"""
    awacs_tracks: Dict[int, Track]         # 预警机粗略目标
    radar_tracks: Dict[str, Dict[int, Track]]  # 各机雷达测量
    fused_tracks: Dict[int, FusedTrack]    # 融合后的目标
    
    def get_threats_in_zone(self, zone: RiskZone) -> List[FusedTrack]
    def get_nearest_threat(self, agent_id: str) -> Optional[FusedTrack]
    def is_target_lost(self, track_id: int) -> bool
    def get_track_quality(self, track_id: int) -> float
```

### 3.2 预警机数据源 (MockAwacsDataSource)

```python
class MockAwacsDataSource:
    """模拟预警机数据源"""
    
    def __init__(self, 
                 detection_range: float = 400.0,  # km
                 position_error: float = 2.5,     # km
                 altitude_error: float = 500.0,   # m
                 heading_error: float = 10.0,     # 度
                 update_interval: float = 5.0,    # 秒
                 max_lost_time: float = 20.0):    # 秒
        ...
    
    def update(self, env, timestamp: float) -> Dict[int, Track]:
        """更新预警机探测数据"""
        # 1. 获取所有敌机真实位置
        # 2. 检查是否在探测范围内
        # 3. 添加位置/高度/航向误差
        # 4. 模拟目标丢失场景
        ...
    
    def _add_noise(self, true_pos, true_alt, true_hdg) -> Track:
        """添加测量误差"""
        ...
    
    def _simulate_lost(self, track_id: int) -> bool:
        """模拟目标丢失"""
        ...
```

### 3.3 CAP状态机 (CAPStateMachine)

```python
class CAPState(Enum):
    PATROL = "PATROL"         # 巡逻
    INTERCEPT = "INTERCEPT"   # 拦截
    ENGAGE = "ENGAGE"         # 交战
    EVADE = "EVADE"           # 规避
    RTB = "RTB"               # 返航

class CAPStateMachine:
    """CAP任务状态机"""
    
    def __init__(self, control_ranges: ControlRanges):
        self.state = CAPState.PATROL
        self.ranges = control_ranges
        self.engage_start_time = None
        
    def update(self, picture: Picture, own_state: Dict) -> CAPState:
        """更新状态"""
        if self.state == CAPState.PATROL:
            return self._check_patrol_transition(picture)
        elif self.state == CAPState.INTERCEPT:
            return self._check_intercept_transition(picture, own_state)
        elif self.state == CAPState.ENGAGE:
            return self._check_engage_transition(picture, own_state)
        elif self.state == CAPState.EVADE:
            return self._check_evade_transition(picture, own_state)
        return self.state
    
    def _check_patrol_transition(self, picture: Picture) -> CAPState:
        """检查巡逻状态转换"""
        # PR距离内发现目标 -> INTERCEPT
        threats = picture.get_threats_in_zone(RiskZone.LOW)
        if threats:
            nearest = min(threats, key=lambda t: t.distance)
            if nearest.distance <= self.ranges.PR:
                return CAPState.INTERCEPT
        return CAPState.PATROL
```

### 3.4 协同探测 (CooperativeDetection)

```python
class DetectionMode(Enum):
    SWEEP = "SWEEP"           # 无预警推磨扫描
    DIRECTED = "DIRECTED"     # 有预警定向扫描
    SEARCH = "SEARCH"         # 目标丢失全域扫描

class CooperativeDetection:
    """协同探测战术模板"""
    
    def __init__(self, formation: FormationManager):
        self.formation = formation
        self.mode = DetectionMode.SWEEP
        self.radar_assignments: Dict[str, RadarScanParams] = {}
        
    def update(self, picture: Picture, awacs_available: bool) -> Dict[str, RadarScanParams]:
        """更新探测模式和雷达分配"""
        # 1. 确定探测模式
        if not awacs_available:
            self.mode = DetectionMode.SWEEP
        elif picture.any_target_lost():
            self.mode = DetectionMode.SEARCH
        else:
            self.mode = DetectionMode.DIRECTED
            
        # 2. 分配雷达扫描任务
        return self._assign_radar_tasks(picture)
    
    def _assign_radar_tasks(self, picture: Picture) -> Dict[str, RadarScanParams]:
        """分配雷达扫描任务"""
        if self.mode == DetectionMode.SWEEP:
            return self._sweep_scan_assignment()
        elif self.mode == DetectionMode.DIRECTED:
            return self._directed_scan_assignment(picture)
        else:
            return self._search_scan_assignment()
```

### 3.5 协同跟踪 (CooperativeTracking)

```python
class CooperativeTracking:
    """协同跟踪战术模板 - 发射前10秒双机协同跟踪"""
    
    def __init__(self, formation: FormationManager):
        self.formation = formation
        self.tracking_assignments: Dict[int, List[str]] = {}  # track_id -> [agent_ids]
        self.tracking_start_time: Dict[int, float] = {}
        
    def should_start_tracking(self, picture: Picture, 
                               target_id: int, 
                               time_to_launch: float) -> bool:
        """判断是否应该开始协同跟踪"""
        return time_to_launch <= 10.0  # 发射前10秒
    
    def assign_trackers(self, picture: Picture, 
                        target_id: int,
                        available_agents: List[str]) -> List[str]:
        """分配跟踪任务，确保至少2机跟踪"""
        # 1. 评估各机跟踪适合度
        scores = {}
        for agent_id in available_agents:
            scores[agent_id] = self._evaluate_tracker_suitability(
                picture, target_id, agent_id)
        
        # 2. 选择最佳2机
        sorted_agents = sorted(scores.keys(), key=lambda a: scores[a], reverse=True)
        return sorted_agents[:2]
    
    def get_tracking_quality(self, picture: Picture, target_id: int) -> float:
        """获取协同跟踪质量"""
        trackers = self.tracking_assignments.get(target_id, [])
        if len(trackers) < 2:
            return 0.0
        # 多传感器融合提升精度
        return min(1.0, 0.7 + 0.15 * len(trackers))
```

### 3.6 接力制导 (RelayGuidance)

```python
class RelayGuidance:
    """接力制导战术模板 - 制导机切换"""
    
    def __init__(self, formation: FormationManager):
        self.formation = formation
        self.guidance_assignments: Dict[int, str] = {}  # missile_id -> guiding_agent
        
    def should_relay(self, guiding_agent: str, 
                     picture: Picture,
                     threat_level: float) -> bool:
        """判断是否需要接力制导"""
        # 制导机需要机动（规避来袭导弹、转弯等）
        return threat_level > 0.7 or self._needs_evasion(guiding_agent, picture)
    
    def select_relay_agent(self, missile_id: int,
                           current_guider: str,
                           picture: Picture,
                           available_agents: List[str]) -> Optional[str]:
        """选择接力制导机"""
        candidates = [a for a in available_agents if a != current_guider]
        if not candidates:
            return None
            
        # 评估适合度：雷达锁定状态、相对位置、威胁等级
        scores = {}
        for agent_id in candidates:
            scores[agent_id] = self._evaluate_relay_suitability(
                missile_id, agent_id, picture)
        
        best = max(scores.keys(), key=lambda a: scores[a])
        return best if scores[best] > 0.5 else None
    
    def execute_relay(self, missile_id: int, 
                      from_agent: str, 
                      to_agent: str) -> bool:
        """执行接力制导"""
        # 1. 确保新制导机已锁定目标
        # 2. 切换制导权
        # 3. 更新分配表
        self.guidance_assignments[missile_id] = to_agent
        return True
```

### 3.7 战术控制距离 (ControlRanges)

```python
@dataclass
class ControlRanges:
    """战术控制距离配置"""
    PR: float = 280.0    # 态势距离 (250-300km)
    NLT: float = 180.0   # 初始距离
    MELD: float = 150.0  # 目标跟踪距离
    MTR: float = 120.0   # 最小跟踪距离
    LR: float = 100.0    # 发射距离
    TR: float = 60.0     # 转换距离
    DOR: float = 45.0    # 期望脱离距离
    DR: float = 35.0     # 决断距离
    MAR: float = 25.0    # 最小脱离距离
    
    @classmethod
    def from_config(cls, config_path: str) -> 'ControlRanges':
        """从配置文件加载"""
        ...
```

---

## 四、状态转换流程

### 4.1 CAP任务状态转换图

```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    ▼                                     │
┌─────────┐    PR距离内    ┌───────────┐    NLT确认    ┌─────────┐
│ PATROL  │ ──发现目标──▶ │ INTERCEPT │ ──跟踪完成──▶ │ ENGAGE  │
└─────────┘               └───────────┘               └─────────┘
     ▲                         │                           │
     │                         │                           │
     │                    目标丢失                     威胁来袭
     │                         │                           │
     │                         ▼                           ▼
     │                    ┌─────────┐               ┌─────────┐
     │                    │ SEARCH  │               │  EVADE  │
     │                    └─────────┘               └─────────┘
     │                         │                           │
     │                    重新发现                     规避完成
     │                         │                           │
     │                         ▼                           │
     │                    ┌───────────┐                    │
     └────────────────────│ INTERCEPT │◀───────────────────┘
                          └───────────┘
                               │
                          任务完成/燃油低
                               │
                               ▼
                          ┌─────────┐
                          │   RTB   │
                          └─────────┘
```

### 4.2 协同探测模式切换

```
                    ┌─────────────────────────────────────┐
                    │         CooperativeDetection        │
                    └─────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌───────────┐    ┌───────────┐    ┌───────────┐
            │   SWEEP   │    │ DIRECTED  │    │  SEARCH   │
            │ 推磨扫描  │    │ 定向扫描  │    │ 全域扫描  │
            └───────────┘    └───────────┘    └───────────┘
                 │                │                │
                 │                │                │
            无预警信息       有预警信息       目标丢失
            4机交替扫描     根据预警定向     4机同时扫描
```

### 4.3 距离节点触发流程

```
距离(km)  300    280    180    150    120    100    60     45     35     25
          │      │      │      │      │      │      │      │      │      │
          │      │      │      │      │      │      │      │      │      │
          │     PR     NLT    MELD   MTR    LR     TR    DOR    DR    MAR
          │      │      │      │      │      │      │      │      │      │
          │      │      │      │      │      │      │      │      │      │
动作:     │  协同探测  预设战术 雷达融合 协同跟踪 发射导弹 转主动  防御机动 战术决策 紧急规避
          │      │      │      │      │      │      │      │      │      │
阶段:     巡逻   │◀─────拦截阶段─────▶│◀─────────交战阶段─────────▶│
```

---

## 五、接口设计

### 5.1 CAPTask 主接口

```python
class CAPTask(MultipleCombatTask):
    """CAP战术任务 - 重构后"""
    
    def __init__(self, config):
        super().__init__(config)
        # 现有模块
        self.coord_sys = CoordinateSystem(...)
        self.faor = FAORManager(...)
        self.formation = FormationManager(...)
        self.patrol_machines = {...}
        
        # 新增模块
        self.picture = Picture()
        self.awacs = MockAwacsDataSource(...)
        self.cap_state_machine = CAPStateMachine(...)
        self.coop_detection = CooperativeDetection(...)
        self.coop_tracking = CooperativeTracking(...)
        self.relay_guidance = RelayGuidance(...)
        self.mission_evaluator = MissionEvaluator(...)
        
        # 复用现有模块
        self.intent_recognizer = EnemyIntentRecognizer()  # 从intent/导入
        self.tactical_executor = TacticalExecutor(...)    # 从tactical_executor导入
    
    def step(self, env):
        """执行一步仿真"""
        # 1. 更新感知层
        self._update_picture(env)
        
        # 2. 更新CAP状态机
        self._update_cap_state(env)
        
        # 3. 根据状态执行相应逻辑
        if self.cap_state_machine.state == CAPState.PATROL:
            self._execute_patrol(env)
        elif self.cap_state_machine.state == CAPState.INTERCEPT:
            self._execute_intercept(env)
        elif self.cap_state_machine.state == CAPState.ENGAGE:
            self._execute_engage(env)
        elif self.cap_state_machine.state == CAPState.EVADE:
            self._execute_evade(env)
        elif self.cap_state_machine.state == CAPState.RTB:
            self._execute_rtb(env)
        
        # 4. 更新任务评估
        self.mission_evaluator.update(env, self.picture)
        
        return self._build_step_result(env)
```

### 5.2 与现有模块的集成

```python
# 复用意图识别模块
from scripts.tacticalProject.intent.enemy_intent import EnemyIntentRecognizer

# 复用战术执行模块
from scripts.tacticalProject.tactical_executor import TacticalExecutor

# 复用战术类型定义
from scripts.tacticalProject.tactical_types import (
    TacticalType, ControlRanges, TacticalState
)
```

---

## 六、数据流设计

### 6.1 感知数据流

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   敌机实体   │────▶│ AWACS模拟   │────▶│ awacs_tracks│
│ (env.agents)│     │ (粗略位置)  │     │ (2-3km误差) │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       │                                       ▼
       │            ┌─────────────┐     ┌─────────────┐
       └───────────▶│ 雷达模拟    │────▶│radar_tracks │
                    │ (精确位置)  │     │ (高精度)    │
                    └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ TrackFusion │
                                        │ (数据融合)  │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │fused_tracks │
                                        │ (融合航迹)  │
                                        └─────────────┘
```

### 6.2 决策数据流

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Picture   │────▶│ CAP状态机   │────▶│  当前状态   │
│ (态势图)    │     │ (状态转换)  │     │ (PATROL等) │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       │                                       ▼
       │            ┌─────────────┐     ┌─────────────┐
       └───────────▶│ 意图识别    │────▶│ 敌方意图    │
                    │ (贝叶斯)    │     │ (攻击/规避) │
                    └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ 战术选择    │
                                        │ (基于风险区)│
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │TacticalTask │
                                        │ (战术执行)  │
                                        └─────────────┘
```

---

## 七、配置设计

### 7.1 配置文件结构

```yaml
# cap_config.yaml 扩展

# ============ 战术控制距离 ============
control_ranges:
  PR: 280.0      # 态势距离 (km)
  NLT: 180.0     # 初始距离
  MELD: 150.0    # 目标跟踪距离
  MTR: 120.0     # 最小跟踪距离
  LR: 100.0      # 发射距离
  TR: 60.0       # 转换距离
  DOR: 45.0      # 期望脱离距离
  DR: 35.0       # 决断距离
  MAR: 25.0      # 最小脱离距离

# ============ 预警机配置 ============
awacs:
  detection_range: 400.0    # km
  position_error: 2.5       # km
  altitude_error: 500.0     # m
  heading_error: 10.0       # 度
  update_interval: 5.0      # 秒
  max_lost_time: 20.0       # 秒

# ============ 雷达配置 ============
radar:
  detection_range: 200.0    # km (5m² RCS)
  max_tracks: 6             # 最大跟踪数
  scan_modes:
    narrow:
      azimuth: 10           # ±10°
      elevation_bars: [1, 2, 4]
      scan_times: [2, 4, 8]  # 秒
    medium:
      azimuth: 30           # ±30°
      elevation_bars: [1, 2, 4]
      scan_times: [5, 10, 20]
    wide:
      azimuth: 60           # ±60°
      elevation_bars: [1, 2, 4]
      scan_times: [10, 20, 40]

# ============ 协同跟踪配置 ============
cooperative_tracking:
  pre_launch_time: 10.0     # 发射前跟踪时间 (秒)
  min_trackers: 2           # 最少跟踪机数
  quality_threshold: 0.7    # 跟踪质量阈值

# ============ 任务评估配置 ============
mission:
  duration: 1200.0          # 任务时长 (秒) = 20分钟
  max_loss_ratio: 0.75      # 最大损失比例
  high_risk_tolerance: 0    # 高风险区允许敌机数
  mid_risk_tolerance: 2     # 中风险区允许敌机数
```

---

## 八、实现计划

### 第一阶段：感知层 (P0) - 预计3天

| 任务 | 文件 | 说明 |
|-----|------|------|
| 1.1 | `picture/picture.py` | 定义Track、FusedTrack、Picture类 |
| 1.2 | `picture/awacs_source.py` | 实现MockAwacsDataSource |
| 1.3 | `picture/track_fusion.py` | 实现数据融合逻辑 |
| 1.4 | 单元测试 | 测试感知层功能 |

### 第二阶段：状态机 (P0) - 预计2天

| 任务 | 文件 | 说明 |
|-----|------|------|
| 2.1 | `cap_state_machine.py` | 实现CAPStateMachine |
| 2.2 | `cap_task.py` | 集成状态机到CAPTask |
| 2.3 | 单元测试 | 测试状态转换 |

### 第三阶段：协同探测 (P0) - 预计3天

| 任务 | 文件 | 说明 |
|-----|------|------|
| 3.1 | `tactics/cooperative_detection.py` | 实现三种探测模式 |
| 3.2 | 雷达扫描参数 | 配置扫描模式和时间 |
| 3.3 | 集成测试 | 测试协同探测流程 |

### 第四阶段：协同跟踪和制导 (P1) - 预计3天

| 任务 | 文件 | 说明 |
|-----|------|------|
| 4.1 | `tactics/cooperative_tracking.py` | 实现发射前协同跟踪 |
| 4.2 | `tactics/relay_guidance.py` | 实现接力制导 |
| 4.3 | 集成测试 | 测试协同跟踪和制导 |

### 第五阶段：战术决策增强 (P1) - 预计2天

| 任务 | 文件 | 说明 |
|-----|------|------|
| 5.1 | `mission_evaluator.py` | 实现任务评估 |
| 5.2 | 战术选择逻辑 | 基于风险区的战术选择 |
| 5.3 | 全流程测试 | 端到端测试 |

---

## 九、测试策略

### 9.1 单元测试

- Picture类数据结构测试
- MockAwacsDataSource误差模拟测试
- CAPStateMachine状态转换测试
- CooperativeDetection模式切换测试

### 9.2 集成测试

- 感知层 → 状态机 → 战术执行 全流程测试
- 协同探测 → 协同跟踪 → 接力制导 流程测试
- 任务评估指标验证

### 9.3 场景测试

- 场景1：敌机从低风险区逐步逼近
- 场景2：预警机目标丢失后恢复
- 场景3：制导机需要规避时的接力制导
- 场景4：20分钟任务完成评估
