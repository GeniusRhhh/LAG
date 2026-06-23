@echo off
setlocal

set "PYTHON_EXE=C:\Users\ZRF\.conda\envs\qwen_poetry\python.exe"
set "REPO_ROOT=D:\Pycharm\LAG"
set "PIPELINE=%REPO_ROOT%\scripts\tacticalProject\cap\train_enemy_f16_cap_pipeline.py"
set "OUTPUT_ROOT=%REPO_ROOT%\scripts\tacticalProject\models\enemy_f16_cap_full_pipeline"

"%PYTHON_EXE%" "%PIPELINE%" --output-root "%OUTPUT_ROOT%" --collect-episodes 8 --collect-steps 4200 --teacher-mode legacy --enable-safe-teacher --epochs 30 --steps 4200 --online-updates 20 --online-buffer-size 256 --online-max-steps 4200 --online-enable-safe-teacher-for-other-enemies --online-eval-steps 4200

endlocal
