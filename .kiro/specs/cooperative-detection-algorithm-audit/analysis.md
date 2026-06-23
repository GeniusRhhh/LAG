# 协同探测算法实现差异分析报告

## 执行摘要

经过深入代码追踪和算法对比，发现两个仿真脚本的根本差异在于：

1. **run_cap_simulation.py**: 运行**完整CAP任务**，使用CAPTask类，包含P1-4全流程（巡逻→探测→交战→返航）
2. **run_detection_verification_real.py**: **孤立测试协同探测算法**，直接调用CooperativeDetection类，仅验证探测决策部分

**关键发现**：当前实现与报告要求存在重大差距，多个核心算法缺失或未完整实现。

---

## 一、两个仿真脚本的底层调用链

### 1.1 run_cap_simulation.py 调用链

```
run_cap_simulation.py
  └─> CAPEnv (MultipleCombatEnv)
       └─> CAPTask (cap_task.py)
            ├─> PatrolStateMachine (巡逻状态机)
            ├─> CAPStateMachine (CAP状态机: PATROL/INTERCEPT/ENGAGE/EVADE/RTB)
            ├─> CooperativeDetection (协同探测)
            ├─> CooperativeEngagement (协同交战)
            ├─> MissionEvaluator (任务评估)
            ├─> TacticSelector (战术选择)
            ├─> CAPRadarManager (雷达管理)
            └─> Picture/TrackFusion (态势感知)
```

**核心特点**：
- 完整的CAP任务流程（P1-4）
- 协同探测只是其中一个子模块
- 包含状态机驱动的多阶段决策
- 集成了巡逻、探测、交战、规避、返航全流程

### 1.2 run_detection_verification_real.py 调用链

```
run_detection_verification_real.py
  └─> DetectionVerification (测试类)
       └─> CooperativeDetection (cap/tactics/cooperative_detection.py)
            ├─> update() - 主更新函数
            ├─> DetectionMode (SWEEP/DIRECTED/SEARCH)
            └─> RadarAssignment (扫描分配)
```

**核心特点**：
- **孤立测试**协同探测算法
- 不依赖完整仿真环境
- 直接构造测试场景（敌机位置、预警信息等）
- 验证5个场景：无预警/有预警/信息丢失/目标机动/模式切换

---

## 二、协同探测算法报告要求 vs 当前实现

### 2.1 报告要求的核心算法（第1-3章）

根据`协同探测算法报告_上交版.md`，要求实现以下算法：

#### 第1章：问题描述与建模
- **算法1.1**: IMM-EKF状态估计（单目标）
  - 作用：多模型（CV/CT/CA/Singer）状态估计
  - 输出：融合状态、协方差、模型概率
  - **状态**：❌ **未实现**

#### 第2章：协同探测决策算法
- **算法0**: 协同探测分层滚动决策主循环
  - 作用：0-R_radar阶段总调度
  - 流程：信息→估计→预测→区域→规划→协调→执行
  - **状态**：⚠️ **部分实现**（缺少完整的滚动决策框架）

- **算法2.5**: 机动不确定性传播与置信域预测
  - 作用：将敌方航向时变转化为置信域
  - 输出：置信半径σ_man(t)
  - **状态**：❌ **未实现**

- **算法2.6**: 编队级引导目标与航向分配
  - 作用：计算每机目标点与目标航向
  - 输出：{p_i*, θ_i*}
  - **状态**：❌ **未实现**（当前只有简单的方位分配）

- **算法2.7**: 速度协调优化算法
  - 作用：满足同步到达约束的速度分配
  - 输出：{v_i*}
  - **状态**：❌ **未实现**

#### 第3章：雷达扫描分配
- **算法3.1**: 探测模式决策
  - 作用：SWEEP/DIRECTED/SEARCH模式选择
  - **状态**：✅ **已实现**（cooperative_detection.py中的DetectionMode）

- **算法3.2**: 扫描任务分配
  - 作用：多机扫描扇区分配
  - **状态**：✅ **已实现**（cooperative_detection.py中的update()）

- **算法3.3**: 扫描范围动态计算
  - 作用：根据距离动态调整扫描范围
  - **状态**：⚠️ **部分实现**（当前为固定范围）

### 2.2 当前实现的算法模块

检查`scripts/tacticalProject/cap/tactics/algorithms/`目录：

```
algorithms/
├── allocators.py          # 分配器（扫描任务分配）
├── estimators.py          # 估计器（可能包含状态估计）
├── fusion.py              # 融合算法
├── predictors.py          # 预测器（可能包含运动预测）
├── probability_map.py     # 概率图
├── scan_calculators.py    # 扫描计算器
└── search_optimizers.py   # 搜索优化器
```

**需要详细检查这些文件是否实现了报告要求的算法**。

---

## 三、缺失算法详细分析

### 3.1 算法1.1: IMM-EKF状态估计（❌ 未实现）

**报告要求**：
- 4个子模型：CV（匀速）、CT（协调转弯）、CA（匀加速）、Singer（随机机动）
- 模型转移概率矩阵Π
- 每个模型独立EKF预测与更新
- 模型概率更新（创新似然）
- 融合输出（加权融合）

**当前实现**：
- 检查`estimators.py`：需要确认是否包含IMM-EKF
- 如果没有，则需要从头实现

**影响**：
- 无法准确估计敌方机动模式
- 置信域预测缺少基础（算法2.5依赖IMM-EKF的协方差P）

### 3.2 算法2.5: 机动不确定性传播（❌ 未实现）

**报告要求**：
```
输入：当前时刻t_k，IMM-EKF协方差P_k，预测时长Δt
输出：置信半径σ_man(t_k + Δt)

步骤：
1. 无量测预测：P_pred = F·P_k·F^T + Q
2. 提取位置协方差：P_pos = P_pred[0:2, 0:2]
3. 计算置信半径：σ_man = sqrt(trace(P_pos))
```

**当前实现**：
- 检查`predictors.py`：需要确认是否包含机动不确定性传播
- 如果没有，则需要实现

**影响**：
- 无法量化敌方机动导致的位置不确定性
- 置信域计算不准确（式2.4中的σ_man项缺失）

### 3.3 算法2.6: 编队级引导目标与航向分配（❌ 未实现）

**报告要求**：
```
输入：敌方中心p_enemy，置信半径R_target，编队配置
输出：每机目标点{p_i*}，目标航向{θ_i*}

步骤：
1. 计算编队中心到敌方中心的方位角
2. 根据编队配置（左/右编队）分配目标点
3. 考虑置信域边界，确保覆盖
4. 计算每机目标航向（指向目标点）
```

**当前实现**：
- `cooperative_detection.py`中只有简单的方位分配
- 没有考虑编队协调、置信域覆盖等约束

**影响**：
- 编队引导不优化
- 可能导致编队分散、到达时间不同步

### 3.4 算法2.7: 速度协调优化（❌ 未实现）

**报告要求**：
```
输入：每机到目标点距离{d_i}，当前速度{v_i}
输出：协调速度{v_i*}，满足同步到达约束

约束：
1. |T_i - T_j| ≤ ΔT_sync（同步到达）
2. v_min ≤ v_i ≤ v_max（速度限制）
3. |v_i(t+Δt) - v_i(t)| ≤ a_max·Δt（加速度限制）

求解：
- 长机减速 vs 僚机加速的权衡
- 最小化燃料消耗
```

**当前实现**：
- 没有速度协调算法
- 所有飞机使用固定速度（vel_cmd=3）

**影响**：
- 长机和僚机到达时间不同步
- 编队协调性差

---

## 四、算法实现优先级建议

### P0（关键缺失，必须实现）

1. **算法1.1: IMM-EKF状态估计**
   - 原因：是置信域预测的基础
   - 工作量：中等（需要实现4个模型+IMM框架）
   - 位置：`cap/tactics/algorithms/estimators.py`

2. **算法2.5: 机动不确定性传播**
   - 原因：置信域计算的核心
   - 工作量：小（基于IMM-EKF的协方差传播）
   - 位置：`cap/tactics/algorithms/predictors.py`

3. **算法2.6: 编队级引导目标与航向分配**
   - 原因：编队协调的核心
   - 工作量：中等（需要考虑多种约束）
   - 位置：`cap/tactics/cooperative_detection.py`或新建`guidance.py`

### P1（重要，建议实现）

4. **算法2.7: 速度协调优化**
   - 原因：同步到达的关键
   - 工作量：中等（优化问题）
   - 位置：`cap/tactics/cooperative_detection.py`或新建`coordination.py`

5. **算法0: 滚动决策主循环**
   - 原因：整合所有算法的框架
   - 工作量：小（主要是调用其他算法）
   - 位置：`cap/cap_task.py`中的`get_action()`

### P2（优化，可选实现）

6. **算法3.3: 扫描范围动态计算**
   - 原因：当前固定范围可用，但不够优化
   - 工作量：小
   - 位置：`cap/tactics/cooperative_detection.py`

---

## 五、验证策略建议

### 5.1 单元测试（孤立验证）

使用`run_detection_verification_real.py`的框架：

1. **测试IMM-EKF**：
   - 场景：目标直飞、转弯、加速
   - 验证：状态估计误差、模型概率

2. **测试机动不确定性传播**：
   - 场景：不同预测时长Δt
   - 验证：置信半径增长曲线

3. **测试引导目标分配**：
   - 场景：不同编队配置、不同敌方位置
   - 验证：目标点覆盖、航向合理性

4. **测试速度协调**：
   - 场景：长机远、僚机近
   - 验证：到达时间差≤ΔT_sync

### 5.2 集成测试（完整仿真）

使用`run_cap_simulation.py`：

1. **场景1**：无预警信息（SWEEP模式）
2. **场景2**：有预警信息（DIRECTED模式）
3. **场景3**：预警信息丢失（SEARCH模式）
4. **场景4**：敌方机动（鲁棒性验证）
5. **场景5**：多次模式切换（稳定性验证）

---

## 六、总结

### 6.1 根本差异

| 维度 | run_cap_simulation.py | run_detection_verification_real.py |
|------|----------------------|-----------------------------------|
| **任务范围** | 完整CAP任务（P1-4） | 孤立协同探测算法 |
| **调用层级** | CAPTask → CooperativeDetection | 直接调用CooperativeDetection |
| **测试方式** | 完整仿真环境 | 构造测试场景 |
| **验证内容** | 全流程（巡逻→探测→交战→返航） | 探测决策（模式选择+扫描分配） |

### 6.2 当前实现状态（已确认）

| 算法 | 报告要求 | 当前实现 | 状态 | 文件位置 |
|------|---------|---------|------|---------|
| 算法1.1 IMM-EKF | 4模型状态估计 | IMMUKFEstimator | ✅ **已实现** | estimators.py |
| 算法2.5 机动不确定性传播 | 置信域预测 | ParticleFilterPredictor | ✅ **已实现** | predictors.py |
| 算法2.6 引导目标分配 | 编队级引导 | VoronoiAllocator/CBBAAllocator | ✅ **已实现** | allocators.py |
| 算法2.7 速度协调 | 同步到达优化 | 固定速度 | ❌ **未实现** | - |
| 算法3.1 模式决策 | SWEEP/DIRECTED/SEARCH | DetectionMode | ✅ **已实现** | cooperative_detection.py |
| 算法3.2 扫描分配 | 多机扇区分配 | UniformAllocator/VoronoiAllocator | ✅ **已实现** | allocators.py |
| 算法3.3 扫描范围计算 | 动态调整 | 固定范围 | ⚠️ **部分实现** | cooperative_detection.py |

**重大发现**：核心算法已经实现！但可能存在以下问题：
1. **算法未被调用**：实现了但没有在CAPTask中正确集成
2. **简化版 vs 专业版**：代码中有SimpleEstimator和IMMUKFEstimator两个版本，可能使用了简化版
3. **速度协调缺失**：这是唯一完全缺失的算法

### 6.3 下一步行动

1. **立即行动**：
   - 检查`algorithms/`目录下的所有文件，确认哪些算法已实现
   - 创建详细的算法实现清单

2. **短期目标**（1-2周）：
   - 实现算法1.1（IMM-EKF）
   - 实现算法2.5（机动不确定性传播）
   - 完善算法2.6（引导目标分配）

3. **中期目标**（2-4周）：
   - 实现算法2.7（速度协调）
   - 完善算法0（滚动决策主循环）
   - 完成单元测试

4. **长期目标**（1-2月）：
   - 完成集成测试
   - 性能优化
   - 文档完善

---

## 附录：需要检查的文件清单

### 必须详细阅读的文件

1. `cap/tactics/algorithms/estimators.py` - 检查IMM-EKF实现
2. `cap/tactics/algorithms/predictors.py` - 检查机动预测实现
3. `cap/tactics/algorithms/allocators.py` - 检查分配算法实现
4. `cap/tactics/algorithms/fusion.py` - 检查融合算法实现
5. `cap/tactics/cooperative_detection.py` - 检查主算法实现
6. `cap/cap_task.py` - 检查CAPTask集成情况

### 需要对比的文档

1. `cap/docs/协同探测算法报告_上交版.md` - 算法要求（已读部分）
2. `cap/docs/1.30协同探测算法.md` - 可能包含更多细节

---

**报告生成时间**: 2026-02-09  
**分析人员**: Kiro AI Assistant  
**状态**: 初步分析完成，需要进一步检查algorithms/目录下的实现
