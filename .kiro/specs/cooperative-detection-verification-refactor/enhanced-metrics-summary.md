# 协同探测验证指标增强总结

## 修改时间
2026-02-13

## 修改文件
- `scripts/tacticalProject/cap/tests/run_detection_verification_real.py`

## 新增指标

### 1. 目标分配均衡度（基尼系数）
- **指标名称**: `target_allocation_gini`
- **计算方法**: 使用基尼系数衡量4架飞机对4个目标的分配均衡性
- **取值范围**: 0-1，越小越均衡（0表示完全均衡，1表示完全不均衡）
- **意义**: 体现协同探测算法是否能够均匀分配目标，避免某些飞机探测过多目标而其他飞机闲置

### 2. 探测覆盖率
- **指标名称**: `detection_coverage_ratio`
- **计算方法**: 每个时刻被至少1架飞机探测的目标数量 / 总目标数量，取平均值
- **取值范围**: 0-1，越大越好
- **意义**: 体现协同探测算法是否能够保持对所有目标的持续覆盖

### 3. 重复探测率
- **指标名称**: `redundant_detection_ratio`
- **计算方法**: 多架飞机同时探测同一目标的次数 / 总探测次数
- **取值范围**: 0-1，越小越好
- **意义**: 体现协同探测算法是否能够避免资源浪费（多架飞机探测同一目标）

### 4. 首探时间标准差
- **指标名称**: `first_detect_time_std`
- **计算方法**: 4个目标首次被探测时间的标准差
- **取值范围**: 时间（秒），越小越均衡
- **意义**: 体现协同探测算法是否能够同步探测所有目标，避免某些目标被延迟探测

### 5. 目标分配热力图数据
- **指标名称**: `allocation_heatmap`
- **数据结构**: `{aid: {tid: {'count': int, 'ratio': float}}}`
- **意义**: 记录每架飞机探测每个目标的时间分布，便于可视化分析

## CSV输出更新

### 英文CSV表头新增字段
- `proposed_target_allocation_gini`
- `baseline_target_allocation_gini`
- `delta_target_allocation_gini`
- `proposed_detection_coverage_ratio`
- `baseline_detection_coverage_ratio`
- `delta_detection_coverage_ratio`
- `proposed_redundant_detection_ratio`
- `baseline_redundant_detection_ratio`
- `delta_redundant_detection_ratio`
- `proposed_first_detect_time_std`
- `baseline_first_detect_time_std`
- `delta_first_detect_time_std`

### 中文CSV表头新增字段
- `我方算法_目标分配均衡度`
- `无算法_目标分配均衡度`
- `目标分配均衡度差值(无-有)`
- `我方算法_探测覆盖率`
- `无算法_探测覆盖率`
- `探测覆盖率差值(有-无)`
- `我方算法_重复探测率`
- `无算法_重复探测率`
- `重复探测率差值(无-有)`
- `我方算法_首探时间标准差(s)`
- `无算法_首探时间标准差(s)`
- `首探时间标准差差值(s)(无-有)`

## 文本报告更新

在每个场景的PROPOSED/BASELINE结果块中，新增以下输出：
```
target_allocation_gini=0.123 (越小越均衡)
detection_coverage_ratio=0.856 (越大越好)
redundant_detection_ratio=0.234 (越小越好)
first_detect_time_std=12.3 s (越小越均衡)
```

在DELTA对比块中，新增以下输出：
```
target_allocation_gini: -0.050 (负值表示proposed更均衡)
detection_coverage_ratio: +0.120 (正值表示baseline覆盖更好)
redundant_detection_ratio: -0.080 (负值表示proposed重复更少)
first_detect_time_std: -5.2 s (负值表示proposed更均衡)
```

## 控制台日志更新

在验证指标汇总部分，新增"协同探测效能指标"章节：
```
[协同探测效能指标]
  目标分配均衡度(基尼系数): 0.123 (越小越均衡)
  平均探测覆盖率: 0.856 (85.6%)
  重复探测率: 0.234 (23.4%，越低越好)
  首探时间标准差: 12.3s (越小越均衡)
  目标分配热力图:
    A0100: B0100:45.2% | B0200:12.3% | B0300:8.9% | B0400:5.6%
    A0200: B0100:15.6% | B0200:48.9% | B0300:10.2% | B0400:7.8%
    A0300: B0100:8.9% | B0200:10.5% | B0300:52.3% | B0400:12.1%
    A0400: B0100:6.7% | B0200:9.2% | B0300:11.8% | B0400:56.7%
```

## 实现细节

### 数据收集
在主仿真循环中，每一步都收集：
1. `detection_timeline`: 记录每架飞机探测到的目标列表
2. `coverage_timeline`: 记录被探测的目标数量

### 指标计算
在仿真结束后，基于收集的数据计算所有新增指标：
1. 统计每架飞机探测每个目标的总次数
2. 计算基尼系数（使用标准基尼系数公式）
3. 计算平均覆盖率
4. 计算重复探测率
5. 计算首探时间标准差
6. 生成目标分配热力图数据

## 预期效果

这些新增指标能够更全面地体现协同探测算法的优势：

1. **目标分配均衡度**: proposed应该显著低于baseline（更均衡）
2. **探测覆盖率**: proposed应该高于baseline（更好的覆盖）
3. **重复探测率**: proposed应该低于baseline（更少的资源浪费）
4. **首探时间标准差**: proposed应该低于baseline（更同步的探测）

这些指标与现有的"首次锁定时间"、"全探延迟"等指标互补，能够从多个维度证明协同探测算法的有效性。

## 向后兼容性

- 所有现有指标保持不变
- 新增指标不影响现有代码逻辑
- CSV文件向后兼容（新增列在末尾）
- 如果某些指标无法计算（如数据不足），会返回None而不会报错
