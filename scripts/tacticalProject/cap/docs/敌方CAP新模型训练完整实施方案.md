# 敌方 CAP 新模型训练完整实施方案

## 1. 目标

本方案的目标不是继续在现有 `baseline_model.pt` 外面修补，而是训练一个真正适配 `run_cap_simulation.py` 的敌方 F16 低层模型。

目标模型需要满足：

- 能直接接入当前 CAP 项目敌方链路
- 能承受 4v4、长时长、真实 CAP 状态机闭环
- 不在前出、交战、撤离、返航过程中异常高度下降
- 4200 step / 840s 长程验证下不早早进入 `ENERGY_L3`

当前已确认的项目事实：

- 当前敌方默认高层已经恢复为 `raw UnifiedEnemyTacticalAI`
- 当前敌方桥接已关闭
- 当前敌方 `wave mode` 已关闭
- 当前敌方低层仍然是 `baseline_model.pt`
- 当前低层输入仍是高层 `15/17/7`，喂模型前压成 legacy 小命令集

因此，真正要替换的是“敌方低层控制能力”，不是再改一层敌方高层包装。

## 2. 根因复述

之前的问题不是单个 if/else，而是：

- 旧低层模型训练场景偏单机命令跟踪，不是真实 4v4 CAP 闭环
- 真实 CAP 部署里存在长期状态累积、阶段切换、返航恢复、导弹威胁等分布
- 旧模型在这些分布上没有稳定能力
- 所以会出现低能量后下沉、或者过早被推到返航恢复段

直接拿旧失败日志做 BC 训练，只会复制失败老师，不能得到真正新能力。

## 3. 训练总路线

必须按下面 4 阶段走：

1. 建立稳定老师
2. 采集真实 CAP 成功轨迹
3. 行为克隆预训练
4. 真实 CAP 闭环在线微调

如果没有“稳定老师”，BC 只会学会坏行为。

## 4. 当前仓库里已经有的脚本

目前已经有一套第一阶段训练管线骨架：

- [build_enemy_f16_cap_dataset.py](D:/Pycharm/LAG/scripts/tacticalProject/cap/build_enemy_f16_cap_dataset.py)
- [collect_enemy_f16_cap_traces.py](D:/Pycharm/LAG/scripts/tacticalProject/cap/collect_enemy_f16_cap_traces.py)
- [train_enemy_f16_cap_bc.py](D:/Pycharm/LAG/scripts/tacticalProject/cap/train_enemy_f16_cap_bc.py)
- [eval_enemy_f16_cap_longrun.py](D:/Pycharm/LAG/scripts/tacticalProject/cap/eval_enemy_f16_cap_longrun.py)
- [train_enemy_f16_cap_pipeline.py](D:/Pycharm/LAG/scripts/tacticalProject/cap/train_enemy_f16_cap_pipeline.py)

这套脚本目前能做：

- 从日志构建数据集
- 训练 BC 模型
- 跑长程验证

这套脚本目前不能解决根因的原因是：

- 数据源主要仍来自旧敌方行为
- 训练目标仍偏“模仿已有行为”
- 没有真实在线 CAP 闭环强化学习
- 没有一个稳定老师作为成功数据来源

所以它是过渡管线，不是最终方案。

## 5. 最终应实现的训练系统结构

最终训练系统应拆成 6 个模块。

### 5.1 稳定老师模块

新增目标文件：

- `scripts/tacticalProject/cap/enemy_safe_teacher.py`

职责：

- 在敌方低层之前做安全接管
- 平时允许跟踪高层命令
- 一旦进入危险区，切换为规则恢复

建议三状态：

- `MODEL`
- `SAFE_HOLD`
- `ENERGY_RECOVER`

建议触发变量：

- `alt_m`
- `vc_mps`
- `v_up_mps`
- `pitch_deg`
- `roll_deg`
- `aoa_deg`
- 当前高层命令
- 是否处于 RTB / DOR_DR / DOR / DR

老师控制逻辑要点：

- 高空低速禁止继续拉升
- 大滚转时禁止继续转弯加深
- 油门优先
- 必要时浅降换速
- 先保包线，再保战术

### 5.2 真实 CAP 数据采集模块

新增目标文件：

- `scripts/tacticalProject/cap/collect_enemy_f16_cap_teacher_dataset.py`

与现有 `collect_enemy_f16_cap_traces.py` 的区别：

- 不是只采普通 trace
- 而是采“老师接管后的成功轨迹”

每条样本至少要记录：

- 时间步
- agent_id
- 高层命令 15/17/7
- 送入低层前的 12 维输入
- 老师实际输出动作
- 当前飞机状态
- 战术阶段
- 是否处于恢复模式
- 是否存在导弹威胁
- 是否最终成功存活到回合结束

建议输出：

- `.npz`
- `.json` metadata

## 5.3 行为克隆预训练模块

继续使用并增强：

- [train_enemy_f16_cap_bc.py](D:/Pycharm/LAG/scripts/tacticalProject/cap/train_enemy_f16_cap_bc.py)

需要补强的点：

- 支持仅用“成功轨迹”训练
- 对危险恢复段加样本权重
- 输出更完整指标

核心训练契约：

- 输入契约必须和在线部署完全一致
- 输出动作仍保持当前低层模型接口一致

如果在线最终仍喂 `baseline` 风格输入，就训练同样输入。
如果未来决定改成 native 直连，就训练时也必须同步改契约。

## 5.4 真实 CAP 在线训练环境

新增目标文件：

- `scripts/tacticalProject/cap/cap_enemy_lowlevel_train_env.py`

作用：

- 不再用 `SingleControl`
- 直接包装真实 `run_cap_simulation` 逻辑
- 只把敌方某架或某对子低层交给待训练模型

基本原则：

- 高层敌方战术逻辑仍用当前真实 unified
- 我方仍用现有战术系统
- 导弹、态势、阶段切换全部保留
- 训练对象只替换敌方低层控制器

这一步是最终方案的关键。

## 5.5 在线强化学习训练模块

新增目标文件：

- `scripts/tacticalProject/cap/train_enemy_f16_cap_online.py`

推荐算法：

- PPO 起步即可
- 如果后面做多机联合，再扩展 MAPPO

奖励建议：

- `r_track`
  - 跟踪高层意图
- `r_survive`
  - 存活奖励
- `r_energy`
  - 速度、垂速、姿态包线奖励
- `r_recover`
  - 从低能量恢复的额外奖励
- `r_fail`
  - `ENERGY_L3`、失速、地撞、持续异常下沉惩罚

关键原则：

- 能量安全优先级必须高于严格命令跟踪
- 模型不能为了跟踪高度/航向目标把自己飞死

### 5.6 长程验证模块

继续增强：

- [eval_enemy_f16_cap_longrun.py](D:/Pycharm/LAG/scripts/tacticalProject/cap/eval_enemy_f16_cap_longrun.py)

最终验收标准不是短测，而是：

- 4200 step / 840s
- 四机不异常下沉
- 不在前中段无意义早返航
- 不进入不可恢复低能量区

## 6. 分阶段实施顺序

### 阶段 A：稳定老师落地

目标：

- 先让当前项目敌方稳定飞

实施步骤：

1. 在敌方低层调用点前增加 `enemy_safe_teacher`
2. 初版只做保守恢复，不求漂亮
3. 在真实 CAP 下跑 5 到 10 轮长程仿真
4. 确认不再早期异常返航、不再异常下沉

验收：

- 敌方能稳定跑完整程
- 返航触发点合理
- 不在 0~120s 开局进入异常恢复态

### 阶段 B：成功数据集采集

目标：

- 采集真实成功轨迹

实施步骤：

1. 用老师控制敌方低层
2. 跑批量 `run_cap_simulation`
3. 过滤失败回合
4. 生成训练集

验收：

- 数据集中大部分 episode 可完整存活
- 包含前出、交战、规避、返航等阶段

### 阶段 C：BC 预训练

目标：

- 得到一个能基本模仿老师的初始化模型

实施步骤：

1. 用成功数据训练 BC
2. 输出模型和验证指标
3. 先跑长程验证，不求完美，只看是否能接近老师

验收：

- 长程表现明显优于旧 baseline
- 早期返航和异常下沉显著减少

### 阶段 D：在线微调

目标：

- 让模型超越规则老师，并真正适配闭环分布

实施步骤：

1. 真实 CAP 环境中加载 BC 初值
2. PPO 在线训练
3. 周期性跑长程验证
4. 保存 best checkpoint

验收：

- 长程稳定性不低于老师
- 战术性优于老师

## 7. 最小可执行落地版本

如果先做最务实版本，不一次把所有东西全做完，建议按下面顺序推进。

### 第一步

实现 `enemy_safe_teacher.py`

这是当前最关键的缺口。

### 第二步

把 `collect_enemy_f16_cap_traces.py` 扩成“老师成功轨迹采集器”

### 第三步

增强 `train_enemy_f16_cap_bc.py`

### 第四步

最后再做 `cap_enemy_lowlevel_train_env.py` 和 `train_enemy_f16_cap_online.py`

## 8. 建议的文件落地清单

建议最终新增/增强这些文件：

- `scripts/tacticalProject/cap/enemy_safe_teacher.py`
- `scripts/tacticalProject/cap/collect_enemy_f16_cap_teacher_dataset.py`
- `scripts/tacticalProject/cap/cap_enemy_lowlevel_train_env.py`
- `scripts/tacticalProject/cap/train_enemy_f16_cap_online.py`
- `scripts/tacticalProject/cap/docs/敌方CAP新模型训练完整实施方案.md`

增强已有文件：

- [train_enemy_f16_cap_bc.py](D:/Pycharm/LAG/scripts/tacticalProject/cap/train_enemy_f16_cap_bc.py)
- [eval_enemy_f16_cap_longrun.py](D:/Pycharm/LAG/scripts/tacticalProject/cap/eval_enemy_f16_cap_longrun.py)
- [train_enemy_f16_cap_pipeline.py](D:/Pycharm/LAG/scripts/tacticalProject/cap/train_enemy_f16_cap_pipeline.py)

## 9. 风险与取舍

### 风险 1

规则老师会比较保守。

影响：

- 战术上不一定漂亮
- 但稳定性会大幅提升

结论：

- 这是值得的
- 因为没有稳定老师就没有可学数据

### 风险 2

真实在线 PPO 成本高。

影响：

- 训练时间长
- 调参成本大

结论：

- 这是最终解决分布问题必须付出的成本

### 风险 3

继续沿用 `baseline_model.pt` 输入契约会限制上限。

结论：

- 短期可以沿用，便于快速替换
- 长期建议考虑 native 直连版本

## 10. 当前建议执行顺序

从当前项目状态出发，建议马上做：

1. 冻结当前敌方 unified 修复版本
2. 开始实现 `enemy_safe_teacher.py`
3. 用老师跑通真实 CAP 长程
4. 采成功数据
5. 用成功数据重新做 BC
6. 再做在线 PPO

## 11. 当前里程碑建议

### 里程碑 1

敌方老师稳定飞完 4200 step

### 里程碑 2

BC 新模型在 4200 step 下不再早期返航

### 里程碑 3

在线 PPO 模型在稳定性上不弱于老师

---

当前与该方案配套的敌方 unified 恢复提交为：

- `7402833` `Restore raw unified enemy AI default path`
