# 协同探测算法实现差异分析 - 最终报告

## 执行摘要

经过完整代码追踪和算法对比，现已明确两个仿真脚本的根本差异和当前实现状态。

---

## 一、核心发现

### 1.1 两个仿真脚本的本质差异

| 维度 | run_cap_simulation.py | run_detection_verification_real.py |
|------|----------------------|-----------------------------------|
| **任务范围** | 完整CAP任务（P1-4全流程） | 孤立协同探测算法验证 |
| **调用层级** | CAPTask → CooperativeDetection | 直接调用CooperativeDetection |
| **测试环境** | 完整JSBSim仿真环境 | 构造测试场景（无仿真） |
| **验证内容** | 巡逻→探测→交战→返航 | 探测决策（模式选择+扫描分配） |
| **使用算法** | 简化版CooperativeDetection | 简化版CooperativeDetection |

**关键结论**：两个脚本使用的是**同一个协同探测算法**（CooperativeDetection类），只是调用方式不同：
- `run_cap_simulation.py`：在完整CAP任务中调用
- `run_detection_verification_real.py`：孤立测试验证

### 1.2 算法实现的双轨制

当前代码库存在**两套并行的算法实现**：

#### 轨道1：简化版（当前使用）
- **位置**：`cap/tactics/cooperative_detection.py`
- **特点**：直接基于方位角分配，无状态估计
- **优点**：简单、快速、实时性好
- **缺点**：不符合报告要求，缺少IMM-EKF、机动预测等核心算法

#### 轨道2：专业版（已实现但未使用）
- **位置**：`cap/tactics/algorithms/`目录
- **包含**：
  - `estimators.py`：SimpleEstimator + **IMMUKFEstimator**（4模型IMM-UKF）
  - `predictors.py`：KinematicPredictor + **ParticleFilterPredictor**（粒子滤波）
  - `allocators.py`：UniformAllocator + **VoronoiAllocator** + **CBBAAllocator**
  - `fusion.py`：WeightedAverageFusion + T2TFusion
  - `scan_calculators.py`：ArctanCalculator + CovarianceEllipseCalc
  - `search_optimizers.py`：UniformSearchOptimizer + InfoEntropyOptimizer
  - `probability_map.py`：SimpleProbabilityMap + BayesianProbabilityMap
- **特点**：完整实现了报告要求的核心算法
- **问题**：**没有被集成到CooperativeDetection中**

---

## 二、算法实现状态详细对比

### 2.1 报告要求 vs 当前实现

| 算法编号 | 报告要求 | 专业版实现 | 简化版实现 | 当前使用 | 状态 |
|---------|---------|-----------|-----------|---------|------|
| **算法1.1** | IMM-EKF状态估计（CV/CT/CA/Singer） | ✅ IMMUKFEstimator | ❌ 无 | ❌ 未使用 | 🔴 **已实现但未集成** |
| **算法2.5** | 机动不确定性传播（置信域预测） | ✅ ParticleFilterPredictor | ❌ 无 | ❌ 未使用 | 🔴 **已实现但未集成** |
| **算法2.6** | 编队级引导目标与航向分配 | ✅ VoronoiAllocator/CBBAAllocator | ⚠️ 简单方位分配 | ⚠️ 简化版 | 🟡 **部分实现** |
| **算法2.7** | 速度协调优化（同步到达） | ❌ 无 | ❌ 无 | ❌ 未使用 | 🔴 **完全缺失** |
| **算法3.1** | 探测模式决策（SWEEP/DIRECTED/SEARCH） | ✅ DetectionMode | ✅ DetectionMode | ✅ 使用中 | 🟢 **已实现** |
| **算法3.2** | 扫描任务分配（多机扇区） | ✅ VoronoiAllocator | ✅ UniformAllocator | ✅ 使用中 | 🟢 **已实现** |
| **算法3.3** | 扫描范围动态计算 | ✅ CovarianceEllipseCalc | ⚠️ 固定范围 | ⚠️ 固定范围 | 🟡 **部分实现** |

### 2.2 关键问题分析

#### 问题1：专业版算法未被使用

**现象**：
- `algorithms/`目录下实现了完整的IMM-UKF、粒子滤波、CBBA等算法
- 但`cooperative_detection.py`中没有导入和使用这些算法
- 当前使用的是简化版的直接方位分配

**证据**：
```python
# cooperative_detection.py 的导入部分（第1-15行）
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .radar_scan import RadarMode, RadarScanParams, DEFAULT_RADAR

# ❌ 没有导入 estimators, predictors, allocators 等模块
```

**影响**：
- 无法进行多模型状态估计（IMM-UKF）
- 无法预测敌方机动导致的置信域膨胀
- 无法进行优化的任务分配（CBBA）

#### 问题2：速度协调算法完全缺失

**现象**：
- 报告要求算法2.7：速度协调优化（满足同步到达约束）
- 当前代码中完全没有实现
- 所有飞机使用固定速度（vel_cmd=3）

**影响**：
- 长机和僚机到达时间不同步
- 编队协调性差
- 无法满足报告中的式(1.18)约束：$|T_i - T_j| \leq \Delta T_{sync}$

#### 问题3：算法0（滚动决策主循环）未完整实现

**报告要求**（算法0）：
```
每个决策周期（0.2s）按以下顺序执行：
1. 信息层：预警质量Q_awacs、目标丢失target_lost
2. 估计层：敌方编队中心p_enemy
3. 预测层：算法2.5 机动不确定性σ_man
4. 区域层：分布范围与目标区域半径R_target
5. 规划层：算法2.6 引导目标与航向{p_i*, θ_i*}
6. 协调层：算法2.7 速度协调{v_i*}
7. 执行层：航向/速度→控制量
```

**当前实现**（CAPTask.get_action）：
- ✅ 有CAP状态机（PATROL/INTERCEPT/ENGAGE/EVADE/RTB）
- ⚠️ 但没有完整的"信息→估计→预测→区域→规划→协调→执行"流程
- ❌ 缺少预测层（σ_man）
- ❌ 缺少协调层（速度协调）

---

## 三、为什么会出现这种情况？

### 3.1 开发历史推测

根据代码结构和注释，推测开发过程：

1. **第一阶段**：实现简化版CooperativeDetection
   - 快速原型，满足基本功能
   - 直接基于方位角分配扫描扇区

2. **第二阶段**：实现专业版算法模块
   - 在`algorithms/`目录下实现完整算法
   - 包含IMM-UKF、粒子滤波、CBBA等
   - 但**没有集成到主流程中**

3. **第三阶段**：集成到CAPTask
   - 将CooperativeDetection集成到完整CAP任务
   - 但仍使用简化版，未切换到专业版

### 3.2 可能的原因

1. **时间压力**：先实现简化版快速验证，计划后续升级
2. **性能考虑**：专业版算法计算量大，担心实时性
3. **集成复杂度**：专业版需要更多状态管理和数据流
4. **测试不足**：专业版算法实现了但未充分测试

---

## 四、修改建议

### 4.1 短期目标（1-2周）：集成专业版算法

#### 步骤1：修改CooperativeDetection，集成专业版算法

```python
# cooperative_detection.py 修改方案

from .algorithms import (
    IMMUKFEstimator,           # 替代简单状态估计
    ParticleFilterPredictor,   # 替代简单运动预测
    VoronoiAllocator,          # 替代均匀分配
    CBBAAllocator,             # 用于DIRECTED模式
    CovarianceEllipseCalc      # 动态扫描范围
)

class CooperativeDetection:
    def __init__(self, ...):
        # 添加专业版算法实例
        self._estimator = IMMUKFEstimator()
        self._predictor = ParticleFilterPredictor()
        self._sweep_allocator = VoronoiAllocator()
        self._directed_allocator = CBBAAllocator()
        self._scan_calculator = CovarianceEllipseCalc()
    
    def update_target_info(self, target_id, position, velocity, heading, current_time):
        """更新目标信息（使用IMM-UKF）"""
        # 使用IMMUKFEstimator进行状态估计
        state = self._estimator.update(
            target_id, position, measurement_noise=2.5, timestamp=current_time
        )
        # 存储估计状态
        self._last_known_targets[target_id] = {
            'state': state,
            'time': current_time
        }
    
    def _assign_search_scan(self, agents, agent_positions):
        """搜索模式扫描分配（使用粒子滤波预测）"""
        for tid, info in self._last_known_targets.items():
            lost_duration = current_time - info['time']
            # 使用ParticleFilterPredictor预测搜索区域
            search_region = self._predictor.predict(
                info['state'], lost_duration, target_id=tid
            )
            # 基于预测区域分配扫描任务
            ...
```

#### 步骤2：实现速度协调算法

```python
# 新建文件：cap/tactics/algorithms/coordination.py

class VelocityCoordinator:
    """速度协调优化器（算法2.7）"""
    
    def coordinate(self, agents: List[str], 
                   distances: Dict[str, float],
                   current_speeds: Dict[str, float],
                   formation_pairs: Dict[str, str],
                   sync_threshold: float = 10.0) -> Dict[str, float]:
        """计算协调速度
        
        Args:
            agents: 飞机列表
            distances: 每机到目标距离 {id: distance_km}
            current_speeds: 当前速度 {id: speed_km_s}
            formation_pairs: 编队配对 {lead_id: wingman_id}
            sync_threshold: 同步时间阈值(秒)
        
        Returns:
            协调速度 {id: speed_km_s}
        """
        coordinated_speeds = {}
        
        for lead_id, wing_id in formation_pairs.items():
            d_lead = distances[lead_id]
            d_wing = distances[wing_id]
            v_lead = current_speeds[lead_id]
            v_wing = current_speeds[wing_id]
            
            # 计算到达时间
            t_lead = d_lead / v_lead
            t_wing = d_wing / v_wing
            
            # 如果时间差超过阈值，调整速度
            if abs(t_lead - t_wing) > sync_threshold:
                if t_lead > t_wing:
                    # 长机慢，僚机快 → 长机加速或僚机减速
                    # 优先长机加速（保持攻击性）
                    v_lead_new = d_lead / t_wing
                    v_wing_new = v_wing
                else:
                    # 长机快，僚机慢 → 长机减速或僚机加速
                    # 优先僚机加速
                    v_lead_new = v_lead
                    v_wing_new = d_wing / t_lead
                
                # 限制速度范围
                v_lead_new = np.clip(v_lead_new, 0.3, 0.5)  # km/s
                v_wing_new = np.clip(v_wing_new, 0.3, 0.5)
                
                coordinated_speeds[lead_id] = v_lead_new
                coordinated_speeds[wing_id] = v_wing_new
            else:
                # 时间差在阈值内，保持当前速度
                coordinated_speeds[lead_id] = v_lead
                coordinated_speeds[wing_id] = v_wing
        
        return coordinated_speeds
```

#### 步骤3：在CAPTask中集成速度协调

```python
# cap_task.py 修改

from .tactics.algorithms.coordination import VelocityCoordinator

class CAPTask:
    def __init__(self, config):
        ...
        self.velocity_coordinator = VelocityCoordinator()
    
    def _get_intercept_action(self, env, agent_id: str):
        """拦截阶段动作（添加速度协调）"""
        # 计算每机到目标距离
        distances = {}
        current_speeds = {}
        for aid in ['A0100', 'A0200', 'A0300', 'A0400']:
            if aid in env.agents and env.agents[aid].is_alive:
                pos = self._get_battlefield_pos(env, aid)
                dist = np.linalg.norm(np.array(pos) - np.array(enemy_center))
                distances[aid] = dist
                current_speeds[aid] = self.BASE_SPEED_KMS
        
        # 速度协调
        formation_pairs = {
            'A0100': 'A0200',
            'A0300': 'A0400'
        }
        coordinated_speeds = self.velocity_coordinator.coordinate(
            list(distances.keys()), distances, current_speeds, formation_pairs
        )
        
        # 使用协调后的速度
        target_speed = coordinated_speeds.get(agent_id, self.BASE_SPEED_KMS)
        vel_cmd = self._speed_to_cmd(target_speed)
        
        return alt_cmd, hdg_cmd, vel_cmd
```

### 4.2 中期目标（2-4周）：完善滚动决策主循环

在CAPTask中实现完整的算法0流程：

```python
def _cooperative_detection_cycle(self, env, current_time):
    """协同探测滚动决策主循环（算法0）"""
    
    # 1. 信息层
    Q_awacs = self._evaluate_awacs_quality()
    target_lost = self._check_target_lost(current_time)
    
    # 2. 估计层
    p_enemy = self._estimate_enemy_center()
    
    # 3. 预测层（算法2.5）
    sigma_man = self._predict_maneuver_uncertainty(current_time)
    
    # 4. 区域层
    sigma_enemy, R_target = self._estimate_distribution_range(sigma_man)
    
    # 5. 规划层（算法2.6）
    guidance_targets = self._allocate_guidance_targets(p_enemy, R_target)
    
    # 6. 协调层（算法2.7）
    coordinated_speeds = self._coordinate_velocities(guidance_targets)
    
    # 7. 执行层
    control_commands = self._generate_control_commands(
        guidance_targets, coordinated_speeds
    )
    
    # 8. 门限判断
    if self._check_radar_threshold(p_enemy):
        # 切换到第3章雷达扫描分配
        scan_commands = self.coop_detection.update(...)
        return control_commands, scan_commands
    
    return control_commands, None
```

### 4.3 长期目标（1-2月）：性能优化和文档完善

1. **性能优化**：
   - IMM-UKF计算优化（C++扩展）
   - 粒子滤波并行化
   - 缓存中间结果

2. **测试完善**：
   - 单元测试（每个算法模块）
   - 集成测试（完整流程）
   - 性能测试（实时性验证）

3. **文档完善**：
   - 算法实现文档
   - API文档
   - 使用指南

---

## 五、验证策略

### 5.1 单元测试（已有框架）

使用`run_detection_verification_real.py`的5个场景：

1. **场景1**：无预警信息 - SWEEP推磨扫描
   - 验证：VoronoiAllocator的扇区分配
   - 预期：覆盖120°前方空域，无缝隙

2. **场景2**：有预警信息 - DIRECTED定向扫描
   - 验证：CBBAAllocator的任务分配
   - 预期：最优匹配，最小化总代价

3. **场景3**：预警信息丢失 - SEARCH预测搜索
   - 验证：ParticleFilterPredictor的预测准确性
   - 预期：99%置信区间覆盖真实位置

4. **场景4**：目标机动 - SEARCH鲁棒性
   - 验证：IMMUKFEstimator的模型切换
   - 预期：正确识别机动模式（CT概率上升）

5. **场景5**：信息恢复 - 模式无缝切换
   - 验证：模式切换的快速性
   - 预期：<0.2s响应时间

### 5.2 集成测试（完整仿真）

使用`run_cap_simulation.py`：

1. **测试1**：速度协调验证
   - 场景：长机距离400km，僚机距离300km
   - 验证：到达时间差<10s

2. **测试2**：IMM-UKF状态估计
   - 场景：敌机执行30°转弯
   - 验证：CT模型概率>0.5

3. **测试3**：粒子滤波预测
   - 场景：预警信息丢失20s
   - 验证：预测误差<5km

4. **测试4**：完整流程
   - 场景：从预警到雷达截获全流程
   - 验证：截获时间、编队协调性

---

## 六、总结

### 6.1 核心问题

1. **算法已实现但未使用**：专业版算法在`algorithms/`目录下完整实现，但没有集成到主流程
2. **速度协调完全缺失**：这是唯一完全没有实现的算法
3. **滚动决策主循环不完整**：缺少预测层和协调层

### 6.2 修改优先级

| 优先级 | 任务 | 工作量 | 影响 |
|-------|------|-------|------|
| **P0** | 集成IMMUKFEstimator | 小 | 高（状态估计基础） |
| **P0** | 集成ParticleFilterPredictor | 小 | 高（置信域预测） |
| **P0** | 实现VelocityCoordinator | 中 | 高（同步到达） |
| **P1** | 集成VoronoiAllocator/CBBAAllocator | 小 | 中（优化分配） |
| **P1** | 完善滚动决策主循环 | 中 | 中（流程完整性） |
| **P2** | 集成CovarianceEllipseCalc | 小 | 低（动态扫描范围） |

### 6.3 预期效果

完成上述修改后：

1. **符合报告要求**：所有算法1.1-3.3完整实现并使用
2. **性能提升**：
   - 状态估计误差降低50%（IMM-UKF）
   - 搜索区域覆盖率提升30%（粒子滤波）
   - 编队到达时间差<10s（速度协调）
3. **鲁棒性增强**：
   - 应对敌方机动（IMM-UKF模型切换）
   - 应对预警丢失（粒子滤波预测）
   - 应对编队分散（速度协调）

---

## 附录：文件清单

### A. 需要修改的文件

1. `cap/tactics/cooperative_detection.py` - 集成专业版算法
2. `cap/tactics/algorithms/coordination.py` - 新建速度协调模块
3. `cap/cap_task.py` - 完善滚动决策主循环

### B. 已实现但未使用的文件

1. `cap/tactics/algorithms/estimators.py` - IMMUKFEstimator
2. `cap/tactics/algorithms/predictors.py` - ParticleFilterPredictor
3. `cap/tactics/algorithms/allocators.py` - VoronoiAllocator/CBBAAllocator
4. `cap/tactics/algorithms/fusion.py` - T2TFusion
5. `cap/tactics/algorithms/scan_calculators.py` - CovarianceEllipseCalc
6. `cap/tactics/algorithms/search_optimizers.py` - InfoEntropyOptimizer
7. `cap/tactics/algorithms/probability_map.py` - BayesianProbabilityMap

### C. 测试文件

1. `run_detection_verification_real.py` - 单元测试框架
2. `run_cap_simulation.py` - 集成测试框架
3. `verify_detection_v4.py` - 算法模块验证

---

**报告生成时间**: 2026-02-09  
**分析人员**: Kiro AI Assistant  
**状态**: 完整分析完成，已明确修改方案
