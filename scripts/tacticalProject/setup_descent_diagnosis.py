#!/usr/bin/env python
"""
敌方AI下降异常诊断启动脚本
在运行仿真之前执行此脚本来启用全面的调试和日志记录
"""

import os
import sys
import logging

# 确保输出目录存在
cap_results_dir = os.path.join(os.path.dirname(__file__), 'cap_results')
os.makedirs(cap_results_dir, exist_ok=True)

# 创建主诊断日志文件
diag_log_file = os.path.join(cap_results_dir, 'descent_diagnosis.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(diag_log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('DescentDiagnosis')

def setup_environment():
    """配置环境变量启用调试"""
    
    logger.info("="*80)
    logger.info("【敌方AI下降异常诊断系统 - 初始化】")
    logger.info("="*80)
    
    # 设置调试开关
    debug_settings = {
        'ENEMY_DESCENT_DEBUG': '1',           # 启用下降调试
        'CAP_DEBUG_PRINT': '1',               # 启用CAP调试输出
        'CAP_ROOTCAUSE_TRACE': '1',           # 启用根因追踪
        'CAP_CONTROL_DEBUG': '1',             # 启用控制调试
        'ENEMY_RTB_ENABLED': '1',             # 启用返航（用于验证）
    }
    
    logger.info("\n【设置环境变量】")
    for key, value in debug_settings.items():
        os.environ[key] = value
        logger.info(f"  {key:30s} = {value}")
    
    logger.info("\n✓ 环境变量已设置")
    
    return debug_settings


def create_instrumentation_guide():
    """创建仪器化指南"""
    
    guide_file = os.path.join(os.path.dirname(__file__), 'cap_results', 'DEBUGGING_GUIDE.md')
    
    guide_content = """# 敌方AI下降异常调试指南

## 快速启动

### 1. 启用调试
```powershell
cd D:\\Pycharm\\LAG\\scripts\\tacticalProject
$env:ENEMY_DESCENT_DEBUG='1'
$env:CAP_ROOTCAUSE_TRACE='1'
```

### 2. 运行仿真
```powershell
python run_cap_simulation_native.py
```

### 3. 查看日志

调试日志将被记录在以下位置：

#### 主诊断日志
- **文件**: `cap_results/descent_diagnosis.log`
- **内容**: 系统初始化、环境配置、总体流程跟踪
- **用途**: 诊断系统是否正确配置

#### 下降追踪日志
- **文件**: `cap_results/descent_debug.log`
- **内容**: 每帧的完整决策链（战术阶段、威胁评估、动作选择、命令转换）
- **格式**: 结构化日志，包含时间戳、智能体ID、物理状态、决策过程

#### 详细追踪日志
- **文件**: `cap_results/descent_trace_detailed.log`
- **内容**: 动作选择权重、状态转换、命令规范化的详细过程
- **用途**: 深入分析为什么选择特定动作

## 日志解读

### 异常下降的识别

查找以下关键词：
```
[DESCENT_DETECTED]      - 检测到下降
[ALTITUDE_CMD]          - 高度命令生成
DESCEND_ACTION          - 选择下降动作
DIVE_ESCAPE             - 俯冲脱离
SPIRAL_DIVE             - 螺旋俯冲
delta_alt<-100          - 显著下降（>100m）
```

### 追踪链路

阅读顺序（最早到最晚）：
1. `[TACTICAL_DECISION]` - 战术决策（威胁评估）
2. `[ACTION_SELECTION]` - 动作选择（权重计算）
3. `[ALTITUDE_CMD]` - 高度命令生成（命令链）
4. `[FUNC_EXEC]` - 函数执行（参数和结果）
5. `[DESCENT_DETECTED]` - 下降发生（物理结果）

### 示例：追踪B0200的异常下降

```
T=476.0s | B0200 @14458m | Phase=INTERCEPT | Mode=NEUTRAL
  [ACTION_SELECTION] 权重: DIVE_ESCAPE=0.8500, BEAM=0.1200, ...
  【选定动作】 DIVE_ESCAPE
  
  [ALTITUDE_CMD] DIVE_ESCAPE | 生成命令 | delta=-300m
  [FUNC_EXEC] _execute_dive_escape | alt: 14458 -> 14158m (Δ=-300m)
  
后续帧：
  [DESCENT_DETECTED] alt: 14158 -> 14058m (Δ=-100m)
  [DESCENT_DETECTED] alt: 14058 -> 13958m (Δ=-100m)
  ...持续下降...
```

## 关键节点

### 战术阶段 (Tactical Phase)
- `NLT_MELD` - 初始集结
- `INTERCEPT` - 截击
- `ENGAGE` - 交战
- `EVADE` - 规避
- `RETURN` - 返航

### 战术模式 (Tactical Mode)  
- `NEUTRAL` - 中立
- `AGGRESSIVE` - 激进
- `DEFENSIVE` - 防御
- `RTB` - 返航

### 异常指标

1. **持续下降** - 连续多帧alt_delta为负
2. **异常加速** - delta_alt > 500m（特别是下降）
3. **不匹配的动作** - MAINTAIN_HEADING却输出DESCEND命令
4. **安全覆盖失效** - altitude < safety_threshold 仍在下降

## 常见异常及排查

| 现象 | 可能原因 | 排查方法 |
|------|--------|--------|
| 无缘无故下降 | 1. 选择了DESCEND/DIVE_ESCAPE<br>2. 返航下降过快<br>3. 安全层失效 | 检查ACTION_SELECTION和FUNC_EXEC日志 |
| 下降过早 | 状态机提前进入RTB | 检查STATE_TRANSITION日志 |
| 下降过快 | 高度命令delta值过大 | 检查ALTITUDE_CMD和命令规范化 |
| 低高度持续下降 | 安全保护未生效 | 检查SAFETY_OVERRIDE日志 |

## 性能提示

调试会产生大量日志文件（单次仿真可能100MB+）。如果需要减少输出：

```python
# 在cap_task.py开始处
import os
os.environ['ENEMY_DESCENT_DEBUG'] = '0'  # 禁用详细调试
os.environ['CAP_DEBUG_PRINT'] = '0'      # 禁用控制台输出
```

## 导出和分析

日志导出后可用脚本分析：

```powershell
# 分析特定智能体的所有下降事件
select-string "\\[DESCENT_DETECTED\\].*B0200" cap_results/descent_debug.log

# 统计下降总数
@(select-string "\\[DESCENT_DETECTED\\]" cap_results/descent_debug.log).Count

# 找出最大下降
select-string "delta.*-[0-9]{4,}" cap_results/descent_debug.log | select-object -first 5
```

"""
    
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    logger.info(f"\n✓ 调试指南已生成: {guide_file}")


def create_analysis_script():
    """创建日志分析脚本"""
    
    analysis_script = os.path.join(
        os.path.dirname(__file__),
        'cap_results',
        'analyze_descent_logs.ps1'
    )
    
    script_content = r"""
# 敌方AI下降日志分析脚本

param(
    [string]$LogFile = "descent_debug.log",
    [string]$Agent = "B0200"
)

Write-Host "=== 敌方AI下降异常分析 ===" -ForegroundColor Green
Write-Host "日志文件: $LogFile"
Write-Host "目标智能体: $Agent"
Write-Host ""

if (-not (Test-Path $LogFile)) {
    Write-Host "❌ 日志文件未找到: $LogFile" -ForegroundColor Red
    exit 1
}

# 1. 统计下降事件总数
$descent_events = @(Select-String "DESCENT_DETECTED.*$Agent" $LogFile)
Write-Host "【统计】 $Agent 下降事件总数: $($descent_events.Count)" -ForegroundColor Yellow

# 2. 提取动作选择序列
Write-Host "`n【动作序列】（仅显示前20个）" -ForegroundColor Yellow
$actions = @(Select-String "ACTION_SELECTION.*$Agent" $LogFile | Select-Object -First 20)
$actions | ForEach-Object {
    if ($_ -match "选定动作\]\s+(.+)$") {
        Write-Host "  - $($matches[1])"
    }
}

# 3. 找出最大下降帧
Write-Host "`n【最大下降事件】" -ForegroundColor Yellow
$max_descent = Select-String "delta.*=.*-[0-9]+.*$Agent" $LogFile | 
    Sort-Object { [int]($_ -replace ".*delta.*=\s*(-[0-9]+).*", '$1') } | 
    Select-Object -First 1

if ($max_descent) {
    Write-Host $max_descent.Line -ForegroundColor Red
} else {
    Write-Host "  无显著下降信息" -ForegroundColor Gray
}

# 4. 分析状态转换
Write-Host "`n【状态转换】" -ForegroundColor Yellow
$states = @(Select-String "STATE_TRANSITION.*$Agent" $LogFile | Select-Object -First 10)
$states | ForEach-Object { Write-Host $_.Line }

Write-Host "`n✓ 分析完成" -ForegroundColor Green

"""
    
    with open(analysis_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    logger.info(f"✓ 分析脚本已生成: {analysis_script}")


def main():
    """主函数"""
    
    try:
        # 1. 配置环境
        debug_settings = setup_environment()
        
        # 2. 生成指南
        create_instrumentation_guide()
        
        # 3. 生成分析脚本
        create_analysis_script()
        
        logger.info("\n" + "="*80)
        logger.info("✅ 诊断系统初始化完成！")
        logger.info("="*80)
        
        logger.info("\n【后续步骤】")
        logger.info("1. 运行仿真: python run_cap_simulation_native.py")
        logger.info("2. 查看日志: cat cap_results/descent_debug.log")
        logger.info("3. 分析日志: .\\cap_results\\analyze_descent_logs.ps1 -Agent B0200")
        
        logger.info("\n【关键输出文件】")
        logger.info(f"- 诊断日志: cap_results/descent_diagnosis.log")
        logger.info(f"- 下降追踪: cap_results/descent_debug.log")
        logger.info(f"- 详细追踪: cap_results/descent_trace_detailed.log")
        logger.info(f"- 调试指南: cap_results/DEBUGGING_GUIDE.md")
        
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
