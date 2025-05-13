#!/bin/bash

env="SingleControl"         # 对应 SingleControlEnv
scenario="1/heading"          # 对应 HeadingTask
algo="sac"                  # 算法名称
exp="v0131"  # 你想给此次实验的名字
seed=5
#env="SingleCombat"         # 对应 SingleControlEnv
#scenario="1v1/NoWeapon/Selfplay"          # 对应 HeadingTask
#algo="sac"                  # 算法名称
#exp="Selfpla0213"  # 你想给此次实验的名字
#seed=20
echo "Running env=${env}, scenario=${scenario}, algo=${algo}, exp=${exp}, seed=${seed}"


CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim_sac.py \
    --env-name ${env} \
    --algorithm-name ${algo} \
    --scenario-name ${scenario} \
    --experiment-name ${exp} \
    --seed ${seed} \
    --n-training-threads 1 \
    --n-rollout-threads 8 \
    --cuda \
    --log-interval 1000 \
    --save-interval 10000 \
    --num-env-steps 2e5 \
    --actor-lr 3e-4 \
    --critic-lr 3e-4 \
    --alpha-lr 3e-4 \
    --batch-size 256 \
    --update-per-step 1 \
    --hidden-size "128 128" --act-hidden-size "128 128" \
    --buffer-size 10000 \
    --target_entropy -3 \
    --init-alpha 1.0 \
    --max-episodes 1000
