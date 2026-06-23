@REM 一键运行 - 敌方AI下降异常诊断仿真
@echo off
chcp 65001 >nul

echo.
echo ========================================
echo   一键运行 - 敌方下降异常诊断
echo ========================================
echo.

REM 设置环境变量
echo [1/3] 启用调试...
set ENEMY_DESCENT_DEBUG=1
set CAP_ROOTCAUSE_TRACE=1
set CAP_DEBUG_PRINT=1
set CAP_CONTROL_DEBUG=1

echo   已设置调试环境变量
echo.

REM 创建输出目录
echo [2/3] 准备输出目录...
if not exist cap_results mkdir cap_results
echo   已创建 cap_results/
echo.

REM 运行仿真
echo [3/3] 运行仿真...
echo.
echo ========================================
echo.

python .\run_cap_simulation_native.py

echo.
echo ========================================
echo   仿真完成！
echo ========================================
echo.
echo 日志位置:
echo   - cap_results\descent_debug.log        (★ 看这个文件)
echo   - cap_results\descent_diagnosis.log
echo.
echo 查看日志:
echo   type cap_results\descent_debug.log
echo.
echo 搜索异常下降:
echo   findstr "is_descent" cap_results\descent_debug.log
echo   findstr "B0200" cap_results\descent_debug.log
echo.
pause
