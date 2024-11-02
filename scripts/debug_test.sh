#!/bin/bash

conda_env=url_navi
seed=1

python scripts/train_robust_agent.py \
    --log_dir ./results_seed_"$seed" \
    --seed 777 \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name debug_test \
    --load_config tuned_configs/happo_10p_sp_rvs_circlecross.json \
    --cuda_device cuda:1