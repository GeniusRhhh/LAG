@echo off
echo ================================================================================
echo SU-27 Improved Training
echo ================================================================================
echo.
echo Current Issues:
echo   - Stable flight: Failed
echo   - Heading control: Failed
echo   - Pass rate: 40%% (2/5 tests passed)
echo.
echo Improvements:
echo   1. Training steps: 10M -^> 20M
echo   2. Learning rate: 3e-4 -^> 5e-4
echo   3. Parallel envs: 8 -^> 16
echo   4. Save interval: Every 50 episodes
echo   5. Support resume training
echo.
echo ================================================================================
echo.

set ENV_NAME=SingleControl
set SCENARIO_NAME=1/heading_su27
set ALGORITHM=ppo
set EXPERIMENT_NAME=su27_improved_v1

REM 针对 i7-11700K (8核16线程) 优化的配置
set N_ROLLOUT_THREADS=8
set N_TRAINING_THREADS=4
set NUM_ENV_STEPS=20000000
set EPISODE_LENGTH=1000

set LR=0.0003
set GAMMA=0.99
set GAE_LAMBDA=0.95
set CLIP_PARAM=0.2
set NUM_MINI_BATCH=8
set PPO_EPOCH=10
set ENTROPY_COEF=0.01

set SAVE_INTERVAL=50
set LOG_INTERVAL=5
set USE_EVAL=True
set EVAL_INTERVAL=25
set N_EVAL_ROLLOUT_THREADS=1

set CUDA=True
set SEED=1

echo Configuration:
echo   Environment: %ENV_NAME%
echo   Scenario: %SCENARIO_NAME%
echo   Algorithm: %ALGORITHM%
echo   Experiment: %EXPERIMENT_NAME%
echo   Parallel Envs: %N_ROLLOUT_THREADS%
echo   Total Steps: %NUM_ENV_STEPS%
echo   Learning Rate: %LR%
echo   PPO Epochs: %PPO_EPOCH%
echo   Save Interval: Every %SAVE_INTERVAL% episodes
echo ================================================================================
echo.

set RESUME_FROM=

REM Check for existing models in order of preference
set MODEL_FOUND=0

REM First check SU27_Improved
if exist "..\results\SU27_Improved\1\heading_su27\ppo\su27_improved_v1\run1" (
    echo.
    echo Found existing SU27_Improved training checkpoint
    echo.
    set /p continue_train="Continue training from this checkpoint? (y/n): "
    if /i "%continue_train%"=="y" (
        set RESUME_FROM=--resume_from ..\results\SU27_Improved\1\heading_su27\ppo\su27_improved_v1\run1\actor_latest.pt
        echo Will continue training from SU27_Improved checkpoint
        set MODEL_FOUND=1
    )
)

REM If not continuing, check SU27_Baseline
if %MODEL_FOUND%==0 (
    if exist "..\results\SU27_Baseline\1\heading_su27\ppo\check\run2\actor_5583.pt" (
        echo.
        echo Found SU27_Baseline model: actor_5583.pt
        echo.
        set /p continue_train="Start from this baseline model? (y/n): "
        if /i "%continue_train%"=="y" (
            set RESUME_FROM=--resume_from ..\results\SU27_Baseline\1\heading_su27\ppo\check\run2\actor_5583.pt
            echo Will start from baseline model
            set MODEL_FOUND=1
        )
    )
)

if %MODEL_FOUND%==0 (
    echo.
    echo No existing model found, starting training from scratch
    echo.
)

echo Starting training...
echo.

cd /d "%~dp0"
C:\Users\ZRF\.conda\envs\lag_gpu\python.exe train_su27_improved.py --env_name %ENV_NAME% --scenario_name %SCENARIO_NAME% --algorithm_name %ALGORITHM% --experiment_name %EXPERIMENT_NAME% --n_rollout_threads %N_ROLLOUT_THREADS% --n_training_threads %N_TRAINING_THREADS% --num_env_steps %NUM_ENV_STEPS% --episode_length %EPISODE_LENGTH% --lr %LR% --gamma %GAMMA% --gae_lambda %GAE_LAMBDA% --clip_param %CLIP_PARAM% --num_mini_batch %NUM_MINI_BATCH% --ppo_epoch %PPO_EPOCH% --entropy_coef %ENTROPY_COEF% --save_interval %SAVE_INTERVAL% --log_interval %LOG_INTERVAL% --use_eval %USE_EVAL% --eval_interval %EVAL_INTERVAL% --n_eval_rollout_threads %N_EVAL_ROLLOUT_THREADS% --cuda %CUDA% --seed %SEED% --use_wandb False %RESUME_FROM%

echo.
echo ================================================================================
echo Training Complete!
echo.
echo Next Steps:
echo 1. Test model: python ceshi_su27_flight.py
echo 2. Check logs: training_logs\su27_improved_*.log
echo 3. Continue training if needed (will auto-detect existing model)
echo ================================================================================
pause
