@echo off
setlocal
set "PY_EXE=C:\Users\ZRF\.conda\envs\qwen_poetry\python.exe"
if exist "%PY_EXE%" (
  "%PY_EXE%" "%~dp0train_f16_energy_15x17x7.py" --device cuda --use-eval %*
) else (
  python "%~dp0train_f16_energy_15x17x7.py" --device cuda --use-eval %*
)
endlocal
