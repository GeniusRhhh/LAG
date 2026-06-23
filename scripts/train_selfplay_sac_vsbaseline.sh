#!/bin/sh
env="SingleCombat"
scenario="1v1/NoWeapon/vsBaseline"
algo="sac"
exp="v1"
seed=1

echo "Running env=${env}, scenario=${scenario}, algo=${algo}, exp=${exp}, seed=${seed}"


CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} \
    --algorithm-name ${algo} \
    --scenario-name ${scenario} \
    --experiment-name ${exp} \
    --seed ${seed} \
    --n-training-threads 1 \
    --n-rollout_threads 4 \
    --cuda \
    --log-interval 1000 \
    --save-interval 10000 \
    --num-env-steps 2e4 \
    --actor-lr 3e-4 \
    --critic-lr 1e-4 \
    --alpha-lr 3e-4 \
    --batch-size 64 \
    --buffer-size 2000 \
    --target_entropy -1.0 \
    --init-alpha 1.0 \
    --normalize-reward \
    --max_steps 10000 \
    --use_baseline True \
    --baseline_type pursue

