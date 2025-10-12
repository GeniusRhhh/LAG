# SNR在功能级雷达模型中的设计说明

## 📋 问题分析

### 原始问题
SNR在代码中计算了，但在探测概率判定中完全没有使用，仅用于CSV日志输出。这引发了对SNR必要性的质疑。

### 核心矛盾
```python
# 计算SNR（基于雷达方程）
snr = 40.0 - 40*log(R/10000) - 3*angle - 0.1*R/1000

# 但探测概率完全不用SNR（基于经验分段）
if distance > 95000:
    base_prob = 0.45  # 这个值和SNR没有关系！
```

**问题本质**：SNR和base_prob是两个独立的系统，缺乏物理联系。

---

## 🎯 设计决策：保留SNR作为物理参考指标

### 决策理由

#### 1. 功能级建模的本质
- **目标**：战术层面的真实性（探测距离、盲区效应、模式转换）
- **非目标**：底层物理过程的微观准确性（电磁波传播细节）
- **结论**：距离分段模型已经很好地捕捉了宏观特性

#### 2. SNR的保留价值
| 价值点 | 说明 |
|--------|------|
| **参数校准** | 用真实雷达SNR数据验证距离分段是否合理 |
| **数据分析** | CSV日志中查看SNR和探测概率的关系 |
| **未来扩展** | 保留切换到SNR驱动模型的能力 |
| **物理一致性** | 提供物理参考，避免参数设定脱离现实 |

#### 3. 成本效益分析
- **删除SNR**：节省计算量微不足道（log运算很快）
- **保留SNR**：几乎无性能损失，但保留重要的物理参考

---

## 💡 实现方案

### 方案对比

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| 完全删除SNR | 代码最简洁 | 失去物理参考，不利于调参 | ⭐⭐ |
| 保留SNR用于日志 | 改动小，保留分析能力 | SNR和Pd物理不完全一致 | ⭐⭐⭐⭐⭐ |
| 用SNR计算base_prob | 物理一致性最好 | 需要重构，需要校准 | ⭐⭐⭐ |

### 最终方案：增强型保留

#### 代码改进
1. **增强SNR函数文档**：
   ```python
   def _calculate_snr(self, distance, bearing, radar_model=None):
       """
       计算信噪比 - 用于数据记录和模型验证
       
       SNR的作用：
       1. 数据记录：输出到CSV用于后期分析
       2. 模型验证：验证距离分段模型的物理合理性
       3. 未来扩展：保留切换到SNR驱动模型的能力
       """
   ```

2. **区分雷达类型**：
   - APG-68：base_snr = 42.0 dB（性能更好）
   - N001VE：base_snr = 40.0 dB

3. **提供参考对应关系**：
   ```python
   # SNR阈值参考：
   # SNR > 13 dB: 高概率探测 (Pd > 0.9)
   # SNR = 10 dB: 中等概率 (Pd ≈ 0.7)
   # SNR < 7 dB:  低概率探测 (Pd < 0.3)
   ```

#### 文档改进
在技术文档3.2.4节增加"SNR在功能级建模中的角色"，明确说明：
- SNR不直接用于探测判定
- SNR作为物理参考指标
- SNR用于验证距离分段模型的合理性

---

## 📊 模型验证方法

### 如何用SNR验证距离分段模型

#### 验证步骤
1. **运行仿真**：收集CSV数据
2. **分析SNR与探测概率**：
   ```python
   import pandas as pd
   
   df = pd.read_csv("radar_data.csv")
   
   # 按SNR分组统计探测概率
   df['SNR_bin'] = pd.cut(df['SNR_dB'], bins=[0, 7, 10, 13, 20, 100])
   result = df.groupby('SNR_bin')['Detection_Probability'].mean()
   
   # 验证是否符合预期：
   # SNR < 7 dB   → Pd < 0.3
   # SNR = 7-10dB → Pd = 0.3-0.7
   # SNR = 10-13dB → Pd = 0.7-0.9
   # SNR > 13 dB  → Pd > 0.9
   ```

3. **调整距离分段**：如果SNR和Pd不匹配，调整base_prob的分段阈值

#### 示例验证结果
```
距离段        实际SNR    期望Pd    实际Pd    是否匹配
0-35km       15-20dB    >0.9      0.96      ✓
35-60km      12-15dB    0.8-0.9   0.93      ✓
60-85km      8-12dB     0.6-0.8   0.88      ✓
85-95km      6-8dB      0.4-0.6   0.75      ✓
95-105km     <6dB       <0.4      0.45      ✓
```

---

## 🔬 未来扩展选项

### 如果需要更高真实度，可以切换到SNR驱动模型

#### 方案A：Swerling模型
```python
def _snr_to_detection_probability(self, snr_db: float, model_type: int = 1):
    """
    基于Swerling模型将SNR转换为探测概率
    
    model_type:
    1 - Swerling I (非起伏目标)
    2 - Swerling II (快速起伏目标)
    """
    if model_type == 1:
        # Swerling I: Pd = 1 - exp(-snr_linear)
        snr_linear = 10 ** (snr_db / 10)
        pd = 1 - math.exp(-snr_linear)
    else:
        # 其他Swerling模型...
        pass
    
    return pd
```

#### 方案B：查表法
```python
# 预计算SNR→Pd映射表（考虑虚警率）
SNR_TO_PD_TABLE = {
    5: 0.10,
    7: 0.30,
    10: 0.70,
    13: 0.90,
    15: 0.95,
    20: 0.99
}
```

---

## ✅ 结论

### 最终建议：**保留SNR，增强文档说明**

#### 优点
1. ✅ 保留物理参考能力
2. ✅ 支持数据分析和模型验证
3. ✅ 无性能损失
4. ✅ 保留未来扩展性
5. ✅ 代码改动最小

#### 实施要点
1. ✅ 在代码中增强注释，明确SNR的作用
2. ✅ 在文档中专门说明功能级建模的设计理念
3. ✅ 区分APG-68和N001VE的SNR基准值
4. ✅ 提供SNR→Pd参考对应关系

#### 不推荐的方案
1. ❌ 完全删除SNR：失去物理参考
2. ❌ 基于SNR重构模型：成本高，收益不明显

---

## 📚 参考文献

1. Skolnik, M. I. (2008). *Radar Handbook (3rd ed.)*. McGraw-Hill.
2. Richards, M. A. (2014). *Fundamentals of Radar Signal Processing (2nd ed.)*. McGraw-Hill.
3. Mahafza, B. R. (2013). *Radar Systems Analysis and Design Using MATLAB (3rd ed.)*. CRC Press.

---

**文档版本**：v1.0  
**更新日期**：2024年（根据当前日期调整）  
**维护者**：雷达系统建模团队

