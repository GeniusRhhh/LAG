#!/bin/sh
env="SingleCombat"
scenario="1v1/NoWeapon/HierarchySelfplay"
algo="ppo"
exp="v1"
seed=1
#resume_episode=48
#existing_dir="../scripts/results/${env}/${scenario}/${algo}/${exp}/wandb/run-20241121_105853-qs6pagpp/files"


echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 8 --cuda --log-interval 1 --save-interval 1 \
    --use-selfplay --selfplay-algorithm "psro" --n-choose-opponents 1 \
    --use-eval --n-eval-rollout-threads 1 --eval-interval 1 --eval-episodes 1 \
    --num-mini-batch 5 --buffer-size 1000 --num-env-steps 5e6 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 \
    --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --render-mode none\
#    --user-name "zrf" --use-wandb --wandb-name "zrf_jsbsim" \
#    --psro-eval-episodes-per-opponent 5 --psro-meta-update-interval 3

#    --resume-episode ${resume_episode} \
#    --use-existing-dir \
#    --existing-dir ${existing_dir}

