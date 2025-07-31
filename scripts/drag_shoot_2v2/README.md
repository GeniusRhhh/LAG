# 2v2拖曳射击战术仿真系统

## 概述

基于现有MultipleCombatEnv架构实现的2v2拖曳射击战术仿真系统。

## 核心特性

1. **正确的架构集成**: 继承MultipleCombatEnv和MultipleCombatTask
2. **正确的坐标系统**: 使用经纬度设置90km初始距离
3. **直接飞机控制**: 通过JSBSim FCS直接控制飞机
4. **完整的战术实现**: 5个阶段的拖曳射击战术

## 文件结构

```
scripts/drag_shoot_2v2/
├── configs/
│   └── drag_shoot_tactical.yaml       # 配置文件
├── tasks/
│   └── drag_shoot_tactical_task.py    # 任务类
├── envs/
│   └── drag_shoot_env.py              # 环境类
├── run_drag_shoot_simulation.py       # 运行脚本
└── README.md                          # 说明文档
```

## 使用方法

```bash
cd scripts/drag_shoot_2v2
python run_drag_shoot_simulation.py
```

## 战术阶段

1. **NLT-MELD (90-81km)**: 僚机右侧crank 30°
2. **MELD-MTR (81-45km)**: 僚机左侧crank -30°
3. **MTR-TR (45-41km)**: 长机45km、僚机41km发射导弹
4. **TR-DOR (41-19.6km)**: 长机short_skate返航
5. **DOR-DR (19.6-14.5km)**: 僚机short_skate返航

## 输出文件

- `trajectory_*.csv`: 飞机轨迹数据
- `radar_status_*.csv`: 雷达状态数据
- `missile_status_*.csv`: 导弹状态数据
- `mission_timeline_*.yaml`: 任务时间线
- `summary_report_*.txt`: 总结报告

## 技术实现

- **配置加载**: 使用parse_config()加载YAML
- **飞机控制**: 直接设置JSBSim FCS属性
- **机动执行**: 调用BasicManeuvers方法
- **数据记录**: 完整的轨迹和状态记录

---

**版本**: 3.0 (完整修复版本)
**日期**: 2024-07-30
