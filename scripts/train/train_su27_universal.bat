@echo off
echo ================================================================================
echo SU-27 Universal Maneuver Training (SAC Algorithm)
echo ================================================================================
echo.
echo This will train a universal SU-27 model like F-16 baseline_model.pt
echo.
echo Training Configuration:
echo - Algorithm: SAC (Soft Actor-Critic)
echo - Task: Heading Control
echo - Total Steps: 10,000,000
echo - Parallel Envs: 8
echo - Estimated Time: 2-4 days
echo.
echo Press Ctrl+C to stop training at any time.
echo.
pause

call conda activate lag_gpu
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    pause
    exit /b 1
)

cd /d C:\Users\ZRF\PycharmProjects\LAG\scripts\train

echo.
echo ================================================================================
echo Starting Training...
echo ================================================================================
echo.

python train_su27_universal_sac.py ^
    --env-name SingleControl ^
    --algorithm-name sac ^
    --scenario-name 1/heading_su27 ^
    --experiment-name universal_v1 ^
    --seed 5 ^
    --n-training-threads 1 ^
    --n-rollout-threads 8 ^
    --cuda ^
    --log-interval 10 ^
    --save-interval 100 ^
    --num-mini-batch 5 ^
    --buffer-size 50000 ^
    --num-env-steps 10000000 ^
    --lr 3e-4 ^
    --gamma 0.99 ^
    --hidden-size "128 128" ^
    --act-hidden-size "128 128" ^
    --recurrent-hidden-size 128 ^
    --recurrent-hidden-layers 1 ^
    --data-chunk-length 8 ^
    --use-recurrent-policy False ^
    --actor-lr 3e-4 ^
    --critic-lr 3e-4 ^
    --alpha-lr 3e-5 ^
    --init-alpha 0.5 ^
    --target-entropy -4.0 ^
    --update-frequency 50 ^
    --tau 0.005 ^
    --batch-size 512 ^
    --warmup-steps 10000 ^
    --reward-norm ^
    --activation-id 0 ^
    --use-feature-normalization ^
    --auto-entropy-tuning ^
    --reward-scale 0.1

echo.
echo ================================================================================
echo Training Completed!
echo ================================================================================
echo.
echo Next Steps:
echo 1. Evaluate the model: python evaluate_su27_model.py --auto
echo 2. Test flight capabilities: python ceshi_su27_flight.py
echo 3. Copy best model to: envs/JSBSim/model/su27_universal_model.pt
echo.
pause
