#!/bin/sh
env="SingleCombat"
scenario="1v1/NoWeapon/vsBaseline"
algo="sac"
exp="v1"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 8 --cuda \
    --log-interval 1 --save-interval 1 \
    --num-mini-batch 5 --buffer-size 1000 --num-env-steps 5e4 \
    --use-eval --n-eval-rollout-threads 1 --eval-interval 1 --eval-episodes 1 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --use-recurrent-policy True
#    --user-name "jyh" --wandb-name "thu_jsbsim" --use-wandb