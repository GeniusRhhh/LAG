# F16 Energy 15x17x7 Training Pack

这次按你的要求改了：

- 保留 `lqyLAG` 原生 PPO 训练框架
- 不再使用课程学习
- 不再构建 tactical curriculum
- 不再依赖 rootcause 日志做课程样本
- 直接在 `cap_lowlevel_f16_tactical_energy` 场景上做单阶段 plain PPO 训练

## 训练方式是什么

就是模仿 `lqyLAG` 原本那套 PPO 训练方式。

具体来说：

- 训练入口还是 `train_jsbsim.py`
- 算法还是 `ppo`
- 环境还是 `SingleControl`
- 任务还是 `cap_lowlevel`
- 模型结构还是 `BaselineActor` 对应的 12 维输入 + GRU + 4 路离散动作输出

也就是说，这不是我新造了一套训练器，而是沿用你仓里原本的 PPO 训练基础设施，只把训练任务封装成你要的 F16 `15/17/7` 原生命令模型。

`zykLAG` 里没有你现在这条更完整的 F16 CAP-native 低层训练链，所以这次仍然是以 `lqyLAG` 为主。

## 当前版本不做什么

当前版本明确不做这些事：

- 不做 tactical curriculum
- 不做日志课程回放
- 不做 hardcase curriculum 注入
- 不做 foundation -> tactical 两阶段课程切换

当前训练模式就是：

- plain PPO
- 单场景
- 单阶段
- 直接训练

## 一键训练

```cmd
D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\run_train_f16_energy_15x17x7.cmd
```

如果 `python` 别名不可用，直接用这个：

```cmd
C:\Users\ZRF\.conda\envs\qwen_poetry\python.exe D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\train_f16_energy_15x17x7.py --device cuda --use-eval
```

## 默认训练参数

- 场景：`1/cap_lowlevel_f16_tactical_energy`
- 算法：`ppo`
- 总步数：`10000000`
- rollout 线程：`16`
- eval 线程：`4`
- warmstart：默认从 `baseline_model.pt`

## 额外优化

- 新增 `EnergyManagementReward`，显式奖励速度裕度、比能变化趋势、AoA 控制和抑制 climb-drain
- 在 `CapLowLevelTask` 中增加高空预防性保能量门
- 默认训练场景改为 `cap_lowlevel_f16_tactical_energy`

## 输出位置

- `D:\Pycharm\LAG\scripts\tacticalProject\models\f16_energy_15x17x7_direct.pt`
- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\exports\f16_energy_15x17x7_direct.pt`

训练状态会放在：

- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\status.json`

训练过程文件也会统一导出到：

- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\train_launcher_output.log`
- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\artifacts\summaries\training_log.txt`
- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\artifacts\summaries\training_metrics.json`
- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\artifacts\summaries\training_progress.json`
- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\artifacts\checkpoints\actor_latest.pt`
- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\artifacts\checkpoints\critic_latest.pt`
- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\artifacts\checkpoints\actor_*.pt`
- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\artifacts\checkpoints\critic_*.pt`
- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\export_manifest.json`

如果开启最终评估，还会额外导出：

- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\final_eval_plainppo.json`
- `D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\runs\时间戳\final_eval_output.log`

## 训练后验证

```cmd
python D:\Pycharm\LAG\lqyLAG\f16_energy_15x17x7\validate_f16_energy_15x17x7.py --model-path D:\Pycharm\LAG\scripts\tacticalProject\models\f16_energy_15x17x7_direct.pt --scenario-name 1/cap_lowlevel_f16_tactical_energy
```

## 仍然要强调的一点

即使你现在改成 plain PPO 重新训练，线上如果还继续让敌机走：

- [cap_lowlevel_helpers.py](/D:/Pycharm/LAG/scripts/tacticalProject/cap/cap_lowlevel_helpers.py)
- [cap_task_refactor_helpers.py](/D:/Pycharm/LAG/scripts/tacticalProject/cap/cap_task_refactor_helpers.py)

里的 `f16_legacy` remap，那么新模型依然不算被正确使用。

训练脚本这边我已经按你的要求去掉课程学习了；后面如果你要真正上线验证，还得把线上接入切成原生 `15/17/7` 语义。
