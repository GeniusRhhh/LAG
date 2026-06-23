#!/bin/sh
env="SingleControl"
scenario="1/heading"
algo="sac"
exp="v1"
seed=5

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 1 --cuda \
    --log-interval 1 --save-interval 1 \
    --num-mini-batch 5 --buffer-size 50000 --num-env-steps 1e7 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 0.5 \
    --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --use-recurrent-policy False --actor-lr 3e-4 --critic-lr 3e-4 --alpha-lr 3e-5 \
    --init-alpha 0.5 --target-entropy -4.0 --update-frequency 50 --tau 0.005 \
    --batch-size 512 --warmup-steps 10000 --reward-norm --activation-id 0 --use-feature-normalization --auto-entropy-tuning \
    --reward-scale 0.1