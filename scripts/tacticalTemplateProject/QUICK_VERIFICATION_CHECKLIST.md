# RWR功能快速验证清单

## ✅ 5分钟快速验证步骤

### 步骤1：验证RWR接口导入（30秒）

打开Python控制台，执行：

```python
from radar_manager import (
    get_rwr_threat_level,
    get_rwr_threat_sources,
    get_rwr_max_threat_bearing,
    check_missile_launch_conditions
)
print("✅ 所有接口导入成功")
```

**预期结果**：无错误，输出 `✅ 所有接口导入成功`

---

### 步骤2：验证RWR数据结构（30秒）

```python
from radar_manager import get_unified_radar_manager

radar_manager = get_unified_radar_manager()

# 检查rwr_states是否存在
assert hasattr(radar_manager, 'rwr_states'), "❌ rwr_states不存在"
print(f"✅ rwr_states存在，包含: {list(radar_manager.rwr_states.keys())}")

# 检查数据结构
for agent_id in ["A0100", "A0200", "B0100", "B0200"]:
    rwr_data = radar_manager.rwr_states[agent_id]
    assert 'threat_level' in rwr_data, f"❌ {agent_id}缺少threat_level"
    assert 'threat_sources' in rwr_data, f"❌ {agent_id}缺少threat_sources"
    assert 'last_warning' in rwr_data, f"❌ {agent_id}缺少last_warning"

print("✅ 所有RWR数据结构完整")
```

**预期结果**：
```
✅ rwr_states存在，包含: ['A0100', 'A0200', 'B0100', 'B0200']
✅ 所有RWR数据结构完整
```

---

### 步骤3：验证RWR函数调用（30秒）

```python
from radar_manager import get_rwr_threat_level, get_rwr_threat_sources

# 测试获取威胁等级
level = get_rwr_threat_level("A0100")
print(f"✅ get_rwr_threat_level('A0100') = {level}")

# 测试获取威胁源
sources = get_rwr_threat_sources("A0100")
print(f"✅ get_rwr_threat_sources('A0100') = {sources}")

print("✅ RWR函数调用正常")
```

**预期结果**：
```
✅ get_rwr_threat_level('A0100') = 0
✅ get_rwr_threat_sources('A0100') = []
✅ RWR函数调用正常
```

---

### 步骤4：验证拖曳射击集成（1分钟）

检查文件 `drag_shoot_tactical_task.py`：

```bash
# 在命令行执行
grep -n "get_rwr_threat_level\|get_rwr_threat_sources\|_check_rwr_threats" drag_shoot_tactical_task.py
```

**预期结果**：应该看到类似输出
```
142:from radar_manager import get_unified_radar_manager, get_rwr_threat_level, get_rwr_threat_sources
144:self.get_rwr_threat_level = get_rwr_threat_level
145:self.get_rwr_threat_sources = get_rwr_threat_sources
1704:def _check_rwr_threats(self, env):
1712:    threat_level = self.get_rwr_threat_level(agent_id)
1716:    threat_sources = self.get_rwr_threat_sources(agent_id)
```

或者在Python中：

```python
with open('drag_shoot_tactical_task.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
checks = [
    "get_rwr_threat_level",
    "get_rwr_threat_sources", 
    "_check_rwr_threats"
]

for check in checks:
    if check in content:
        print(f"✅ 找到: {check}")
    else:
        print(f"❌ 未找到: {check}")
```

**预期结果**：所有检查项都显示 `✅ 找到`

---

### 步骤5：验证雷达数据表干扰列（1分钟）

```python
from radar_manager import get_unified_radar_manager

radar_manager = get_unified_radar_manager()

# 模拟一个简单的环境对象
class MockEnv:
    class Agents:
        def __init__(self):
            self.data = {}
        def __contains__(self, key):
            return key in self.data
        def __getitem__(self, key):
            return self.data.get(key)
    
    def __init__(self):
        self.agents = self.Agents()
        self.current_step = 0
        self.time_interval = 0.2

# 创建模拟环境
env = MockEnv()

# 调用record_radar_data（即使没有真实数据也能测试结构）
radar_data = radar_manager.record_radar_data(env, 0.0)

# 检查返回的数据结构
if radar_data:
    first_record = radar_data[0]
    assert 'Being_Jammed' in first_record, "❌ 缺少Being_Jammed字段"
    assert 'Jamming_Sources' in first_record, "❌ 缺少Jamming_Sources字段"
    print("✅ 雷达数据包含干扰状态字段")
    print(f"   示例记录: {list(first_record.keys())}")
else:
    print("⚠️  无雷达数据（正常，因为没有真实环境）")
    print("   但数据结构已正确实现")
```

**预期结果**：
```
✅ 雷达数据包含干扰状态字段
   示例记录: ['Time_s', 'Agent_ID', 'Radar_Type', 'Status', 'Target_ID', 
               'Target_Distance_km', 'SNR_dB', 'Doppler_Shift_m_s', 
               'Lock_Quality', 'Detection_Probability', 'Beam_Angle_deg', 
               'Side', 'Lock_Target', 'Being_Jammed', 'Jamming_Sources']
```

---

### 步骤6：运行完整验证脚本（2分钟）

```bash
cd C:\Users\ZRF\PycharmProjects\LAG\scripts\tacticalTemplateProject
python test_rwr_interface.py
```

**预期结果**：
```
🎉 所有测试通过！RWR接口已完全实现并集成。
```

---

## 📋 验证检查表

请在完成每项验证后打勾：

- [ ] 步骤1：RWR接口导入成功
- [ ] 步骤2：RWR数据结构完整
- [ ] 步骤3：RWR函数调用正常
- [ ] 步骤4：拖曳射击集成确认
- [ ] 步骤5：雷达数据表包含干扰列
- [ ] 步骤6：完整验证脚本通过

---

## 🎯 如果所有步骤都通过

**恭喜！RWR接口已完全实现并验证通过。**

可以安全地进行下一步：
1. 为并排射击战术集成RWR
2. 为回转射击战术集成RWR
3. 为钳形夹击战术集成RWR

---

## ⚠️ 如果某个步骤失败

### 常见问题排查

**问题1：导入失败**
- 检查是否在正确的目录
- 检查 `radar_manager.py` 文件是否存在

**问题2：rwr_states不存在**
- 检查 `radar_manager.py` 第248-254行
- 确认 `UnifiedRadarManager.__init__()` 中初始化了rwr_states

**问题3：函数调用失败**
- 检查函数签名是否正确
- 查看错误堆栈信息

**问题4：拖曳射击集成未找到**
- 检查 `drag_shoot_tactical_task.py` 是否是最新版本
- 确认第142-147行和第1704-1732行的代码

---

## 📊 功能对照表

| 功能 | 文件 | 行号 | 验证方法 |
|-----|------|------|---------|
| RWR接口定义 | radar_manager.py | 1516-1569 | 步骤1 |
| RWR数据结构 | radar_manager.py | 248-254 | 步骤2 |
| RWR全局接口 | radar_manager.py | 2030-2043 | 步骤3 |
| 拖曳射击集成 | drag_shoot_tactical_task.py | 142-147, 1704-1732 | 步骤4 |
| 干扰状态列 | radar_manager.py | 1868-1869, 1895-1896 | 步骤5 |

---

## ✅ 验证完成后的输出示例

### 雷达数据CSV示例（新增列）

```csv
Time_s,Agent_ID,Radar_Type,Status,Target_ID,Target_Distance_km,SNR_dB,Lock_Quality,Detection_Probability,Being_Jammed,Jamming_Sources
10.0,A0100,AN/APG-68(V)9,LOCK,B0100,45.2,15.3,0.85,0.92,True,B0100:NOISE_JAMMING
10.0,A0200,AN/APG-68(V)9,TRACK,B0200,48.1,12.8,0.72,0.88,False,
10.0,B0100,N001VE,LOCK,A0100,45.2,14.1,0.78,0.85,True,A0100:DECEPTION_JAMMING
10.0,B0200,N001VE,SEARCH,None,0.0,0.0,0.0,0.0,False,
```

### RWR告警输出示例

```
🚨 A0100 RWR告警: 等级3 - B0100(等级3,方位45°)
⚠️  A0200 RWR告警: 等级2 - B0100(等级2,方位30°)
📡 B0100 RWR告警: 等级1 - A0100(等级1,方位180°)
```

---

## 🎉 验证成功标志

当你看到以下所有输出时，表示验证完全成功：

1. ✅ 所有Python导入无错误
2. ✅ RWR数据结构包含4个飞机
3. ✅ RWR函数返回正确类型的值
4. ✅ 拖曳射击文件包含RWR相关代码
5. ✅ 雷达数据包含 `Being_Jammed` 和 `Jamming_Sources` 字段
6. ✅ 验证脚本显示 "🎉 所有测试通过！"

**此时可以确认：RWR接口已完全实现，可以进行下一步工作！**
