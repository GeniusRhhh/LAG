# 2v2空战全流程仿真系统 (基于现有架构)

## 概述

本系统基于现有的MultipleCombatEnv和相关任务类，实现了完整的2v2拖曳射击战术仿真。系统复用了现有的AircraftSimulator、MissileSimulator和TacviewRenderer，确保与现有代码架构的完全兼容。

## 架构说明

### 基于现有组件
- **环境类**: MultipleCombatEnv (复用现有)
- **任务类**: DragShootTacticalTask (新增，继承MultipleCombatTask)
- **飞机模拟**: AircraftSimulator (复用现有)
- **导弹模拟**: MissileSimulator (复用现有)
- **ACMI渲染**: TacviewRenderer (复用现有)
- **配置系统**: YAML配置文件 (复用现有格式)

## 系统特性

### 飞机配置
- **己方**: F-16 A0100(长机), A0101(僚机)
- **敌方**: F-16 B0100(长机), B0101(僚机)
- **初始高度**: 20,000英尺 (6096米)
- **初始速度**: 800英尺/秒 (243.84 m/s)
- **初始距离**: 90km

### 武器系统
- **雷达**: AN/APG-68(V)9
  - 探测距离: 100km
  - 跟踪目标: 10个
  - 锁定目标: 2个
- **导弹**: AIM-120C-7
  - 每机携带: 2枚
  - 最大射程: 70-150km
  - 不可逃逸区: 25-45km

### 战术阶段

#### 1. NLT-MELD (90-81km)
- **己方**: 长机平稳飞行，僚机右侧crank
- **敌方**: 平稳飞行
- **雷达**: 搜索模式

#### 2. MELD-MTR (81-45km)
- **己方**: 长机平稳飞行，僚机左侧crank调整平行
- **敌方**: 平稳飞行
- **雷达**: 跟踪→锁定

#### 3. MTR-TR (45-41km)
- **己方**: 长机45km发射，僚机41km发射
- **敌方**: 长机45km发射，僚机41km发射
- **雷达**: 锁定状态

#### 4. TR-DOR (41-19.6km)
- **己方**: 长机左侧short_skate后返航，僚机平稳飞行
- **敌方**: 平稳飞行
- **导弹**: 中段制导

#### 5. DOR-DR (19.6-14.5km)
- **己方**: 长机返航，僚机左侧short_skate后返航
- **敌方**: 19.6km发射第二轮导弹
- **导弹**: 末段制导

## 文件结构

```
envs/JSBSim/
├── configs/2v2/DragShoot/
│   └── DragShootTactical.yaml      # 拖曳射击配置文件
├── tasks/
│   └── drag_shoot_task.py          # 拖曳射击任务类
└── envs/
    └── multiplecombat_env.py       # 环境类(已修改支持新任务)

scripts/
├── run_2v2_drag_shoot.py           # 运行脚本
├── analyze_drag_shoot_data.py      # 数据分析脚本
└── README_2v2_Combat.md           # 本文档

air_combat_results/                 # 输出目录
├── trajectory_YYYYMMDD_HHMMSS.csv  # 轨迹数据
├── radar_status_YYYYMMDD_HHMMSS.csv # 雷达数据
├── missile_status_YYYYMMDD_HHMMSS.csv # 导弹数据
├── mission_timeline_YYYYMMDD_HHMMSS.yaml # 任务时间线
├── simulation_log_YYYYMMDD_HHMMSS.log # 仿真日志
└── summary_report_YYYYMMDD_HHMMSS.txt # 总结报告

analysis_results/                   # 分析结果
├── tactical_trajectory_YYYYMMDD_HHMMSS.png
├── distance_phases_YYYYMMDD_HHMMSS.png
├── radar_timeline_YYYYMMDD_HHMMSS.png
├── missile_analysis_YYYYMMDD_HHMMSS.png
└── tactical_report_YYYYMMDD_HHMMSS.txt
```

## 使用方法

### 1. 运行仿真

```bash
cd scripts
python run_2v2_drag_shoot.py
```

### 2. 分析数据

```bash
python analyze_drag_shoot_data.py
```

### 3. 查看ACMI文件

ACMI文件由现有的TacviewRenderer自动生成，可使用TacView打开进行3D可视化。

### 4. 配置修改

修改 `envs/JSBSim/configs/2v2/DragShoot/DragShootTactical.yaml` 来调整：
- 飞机初始位置和状态
- 战术参数
- 控制距离
- 雷达和导弹参数

## 数据格式

### 轨迹数据 (trajectory.csv)
```csv
Time_s,Agent_ID,Type,X_m,Y_m,Z_m,Heading_deg,Altitude_m,Velocity_m_s
0.0,A0100,F-16,0,0,6096,180.0,6096,243.84
```

### 雷达数据 (radar_status.csv)
```csv
Time_s,Agent_ID,Radar_Type,Status,Target_ID,Target_Distance_km
0.0,A0100,AN/APG-68(V)9,SEARCH,B0100,90.0
```

### 导弹数据 (missile_status.csv)
```csv
Time_s,Missile_ID,Launcher_ID,Type,Status,X_m,Y_m,Z_m,Velocity_m_s,Target_ID
92.2,M001,A0100,AIM-120C-7,LAUNCHED,-5000,0,6096,1360.0,B0100
```

## 技术参数

- **时间步长**: 0.2秒
- **最大仿真时间**: 300秒
- **数据记录频率**: 每0.2秒
- **坐标系**: 笛卡尔坐标系 (米)
- **ACMI兼容**: TacView 2.1格式

## 终止条件

1. 仿真时间达到300秒
2. 任一飞机被击落
3. 飞机高度低于3000米
4. 双方距离超过100km

## 扩展功能

### 自定义参数
可以修改`AirCombat2v2Simulation`类中的参数：
- 控制距离
- 导弹参数
- 雷达参数
- 战术行为

### 添加新战术
在`execute_tactical_behavior`方法中添加新的战术逻辑。

### 数据分析
使用`analyze_combat_data.py`生成：
- 2D轨迹图
- 距离-时间图
- 雷达状态图
- 导弹轨迹图
- 统计报告

## 注意事项

1. 确保安装必要的Python包：pandas, matplotlib, numpy
2. 仿真使用简化的运动学模型，适用于战术分析
3. ACMI文件可用TacView等工具查看
4. 数据文件按时间戳命名，避免覆盖

## 故障排除

### 常见问题
1. **导入错误**: 检查Python路径设置
2. **数据文件未生成**: 检查输出目录权限
3. **ACMI文件无法打开**: 确认TacView版本兼容性

### 调试模式
设置日志级别为DEBUG获取详细信息：
```python
logging.basicConfig(level=logging.DEBUG)
```
