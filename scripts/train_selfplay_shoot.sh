#!/bin/sh

env="MultipleCombat"
scenario="2v2/ShootMissile/HierarchySelfplay"
algo="mappo"
exp="v1"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
--env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 6 --cuda --log-interval 10 --save-interval 10 \
    --use-selfplay false --selfplay-algorithm "none" --n-choose-opponents 0 \
    --use-eval false --n-eval-rollout-threads 1 --eval-interval 100 --eval-episodes 1 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1e8 \
    --lr 0.0003 --gamma 0.99 --ppo-epoch 10 --clip-params 0.2 --max-grad-norm 0.5 --entropy-coef 0.01 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 10 \
    --use-prior false
#    --user-name "zrf" --use-wandb --wandb-name "thu_jsbsim" \
