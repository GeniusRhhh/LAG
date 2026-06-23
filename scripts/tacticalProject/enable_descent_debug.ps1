# 敌方AI下降异常诊断启动脚本 (PowerShell)

Write-Host "========== 敌方AI下降异常诊断系统 ==========" -ForegroundColor Green
Write-Host ""

# 1. 设置环境变量
Write-Host "[1/4] 设置环境变量..." -ForegroundColor Cyan
$env:ENEMY_DESCENT_DEBUG = '1'
$env:CAP_ROOTCAUSE_TRACE = '1'
$env:CAP_DEBUG_PRINT = '1'
$env:CAP_CONTROL_DEBUG = '1'

Write-Host "  ✓ ENEMY_DESCENT_DEBUG = 1" -ForegroundColor Green
Write-Host "  ✓ CAP_ROOTCAUSE_TRACE = 1" -ForegroundColor Green
Write-Host "  ✓ CAP_DEBUG_PRINT = 1" -ForegroundColor Green
Write-Host "  ✓ CAP_CONTROL_DEBUG = 1" -ForegroundColor Green
Write-Host ""

# 2. 准备输出目录
Write-Host "[2/4] 准备输出目录..." -ForegroundColor Cyan
$resultsDir = ".\cap_results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
    Write-Host "  ✓ 创建目录: $resultsDir" -ForegroundColor Green
} else {
    Write-Host "  ✓ 目录已存在: $resultsDir" -ForegroundColor Green
}
Write-Host ""

# 3. 创建调试指南
Write-Host "[3/4] 生成调试指南..." -ForegroundColor Cyan
$guideContent = @'
# 敌方AI下降异常调试指南

## 快速查看日志

运行仿真后，查看以下日志文件：

### 1. 主诊断日志
```
.\cap_results\descent_diagnosis.log
```
显示系统初始化和配置信息。

### 2. 下降追踪日志
```
.\cap_results\descent_debug.log  
```
包含完整的决策链：
- 战术阶段（INTERCEPT, ENGAGE, EVADE, RETURN）
- 威胁评估
- 动作选择
- 命令生成

### 3. 详细追踪日志
```
.\cap_results\descent_trace_detailed.log
```
包含：
- 动作权重详情
- 状态转换
- 命令规范化过程

## 日志搜索技巧

### 查找B0200的所有下降
```powershell
Select-String "DESCENT_DETECTED.*B0200" .\cap_results\descent_debug.log
```

### 查找异常下降（差值大于500m）
```powershell
Select-String "delta.*-[5-9]{3,}|delta.*-[0-9]{4,}" .\cap_results\descent_debug.log
```

### 查找选择DESCEND动作的记录
```powershell
Select-String "DESCEND_ACTION" .\cap_results\descent_debug.log
```

### 查找STATE_TRANSITION（状态转换）
```powershell
Select-String "STATE_TRANSITION" .\cap_results\descent_debug.log
```

## 追踪流程图

```
[战术决策]
    ↓
[威胁评估] → ThreatLevel = SEVERE/MODERATE/NONE
    ↓
[动作选择] → Available: {DIVE_ESCAPE(0.85), BEAM(0.12), ...}
    ↓
    选中: DIVE_ESCAPE
    ↓
[命令生成] → altitude_cmd_idx = 0, delta = -300m
    ↓
[命令标准化] → 检查边界
    ↓
[执行命令] → 调用 _execute_dive_escape()
    ↓
[物理模型]
    ↓
[观测结果] → 高度从14458m → 14158m (下降300m)
```

## 关键追踪点

1. **决策选择** - 为什么选择DIVE_ESCAPE?
   查看: ACTION_SELECTION 中的权重

2. **命令生成** - 命令参数是什么?
   查看: ALTITUDE_CMD 中的 delta 值

3. **函数执行** - 实际执行了什么?
   查看: FUNC_EXEC 中的函数名和参数

4. **物理结果** - 最终高度变化
   查看: DESCENT_DETECTED 中的 Δ 值

## 异常模式

| 模式 | 症状 | 原因 | 处理 |
|------|------|------|------|
| 持续下降 | 连续多帧下降 | 选择了下降动作 | 检查ACTION_SELECTION |
| 陡峭下降 | 单帧下降>500m | 命令delta过大 | 检查ALTITUDE_CMD |
| 不响应 | 低高度仍继续下降 | 安全保护失效 | 检查SAFETY_OVERRIDE |
| 早期下降 | 高度充足时就开始下降 | 状态机问题 | 检查STATE_TRANSITION |

'@

$guideContent | Out-File -FilePath ".\cap_results\DEBUGGING_GUIDE.txt" -Encoding UTF8 -Force
Write-Host "  ✓ 调试指南已生成: .\cap_results\DEBUGGING_GUIDE.txt" -ForegroundColor Green
Write-Host ""

# 4. 显示下一步
Write-Host "[4/4] 准备就绪！" -ForegroundColor Cyan
Write-Host ""
Write-Host "========== 后续步骤 ==========" -ForegroundColor Yellow
Write-Host ""
Write-Host "1️⃣  运行仿真:" -ForegroundColor Cyan
Write-Host '   python .\run_cap_simulation_native.py' -ForegroundColor White
Write-Host ""
Write-Host "2️⃣  查看日志 (实时):" -ForegroundColor Cyan
Write-Host '   Get-Content -Path ".\cap_results\descent_debug.log" -Wait -Tail 50' -ForegroundColor White
Write-Host ""
Write-Host "3️⃣  分析特定智能体 (B0200):" -ForegroundColor Cyan
Write-Host '   Select-String "B0200" .\cap_results\descent_debug.log | Select-String "DESCENT_DETECTED"' -ForegroundColor White
Write-Host ""
Write-Host "4 查看高度变化:" -ForegroundColor Cyan
Write-Host "   Select-String 'delta' .\cap_results\descent_debug.log | head -20" -ForegroundColor White
Write-Host ""
Write-Host "========== Log File Locations ==========" -ForegroundColor Yellow
Write-Host "Diagnosis Log:    .\cap_results\descent_diagnosis.log" -ForegroundColor Gray
Write-Host "Descent Trace:    .\cap_results\descent_debug.log" -ForegroundColor Gray  
Write-Host "Detail Trace:     .\cap_results\descent_trace_detailed.log" -ForegroundColor Gray
Write-Host "Debug Guide:      .\cap_results\DEBUGGING_GUIDE.txt" -ForegroundColor Gray
Write-Host ""
