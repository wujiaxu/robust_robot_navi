#!/bin/bash


#!/bin/bash

# Define the root directory
root_dir="/home/dl/wu_ws/HARL/crowd_navi_bench/results_vis_aware/crowd_env_vis/crowd_navi/robot_crowd_happo"


human_num=5
result_dir="results_vis_aware"

# python test_runner/ad_hoc_crowd_model_test_runner.py \
#     --log_dir $result_dir \
#     --exp_name sfm_trained_vs_1d_env \
#     --model_dir /home/dl/wu_ws/HARL/crowd_navi_bench/results/crowd_env/crowd_navi/robot_crowd_happo/train_on_sfm_crowd/seed-00001-2024-09-28-20-00-11 \
#     --env crowd_env_vis \
#     --scenario room_361 \
#     --sfm_v0 5 \
#     --sfm_sigma 1.5 \
#     --human_num "$human_num"
    
# python test_runner/ai_crowd_model_test_runner.py \
#     --log_dir $result_dir \
#     --exp_name 1d_with_vis_trained_vs_1d_ai_env \
#     --model_dir $root_dir/train_on_vis_ai_090_4p_3c_rvs_room361/seed-00001-2024-10-08-23-52-05 \
#     --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361/seed-00001-2024-09-27-10-55-57 \
#     --env crowd_env_vis \
#     --scenario room_361 \
#     --sfm_v0 5 \
#     --sfm_sigma 1.5 \
#     --human_num "$human_num"

python test_runner/ai_crowd_model_test_runner.py \
    --log_dir $result_dir \
    --exp_name train_on_vis_2point5_1m_ai_090_4p_3c_rvs_room361_vs_1d_ai_env \
    --model_dir $root_dir/train_on_vis_2point5_1m_ai_090_4p_3c_rvs_room361/seed-00001-2024-10-10-15-51-17 \
    --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361/seed-00001-2024-09-27-10-55-57 \
    --env crowd_env_vis \
    --scenario room_361 \
    --sfm_v0 5 \
    --sfm_sigma 1.5 \
    --human_num "$human_num"

python test_runner/ai_crowd_model_test_runner.py \
    --log_dir $result_dir \
    --exp_name train_on_vis_1point5_1m_ai_090_4p_3c_rvs_room361_vs_1d_ai_env \
    --model_dir $root_dir/train_on_vis_1point5_1m_ai_090_4p_3c_rvs_room361/seed-00001-2024-10-10-15-51-12 \
    --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361/seed-00001-2024-09-27-10-55-57 \
    --env crowd_env_vis \
    --scenario room_361 \
    --sfm_v0 5 \
    --sfm_sigma 1.5 \
    --human_num "$human_num"

    

# python test_runner/ad_hoc_crowd_model_test_runner.py \
#     --log_dir $result_dir \
#     --exp_name 1d_trained_vs_0d_env \
#     --model_dir $root_dir/dis_1p/seed-00001-2024-10-02-19-54-44 \
#     --env crowd_env_vis_ablation_2 \
#     --scenario room_361 \
#     --sfm_v0 5 \
#     --sfm_sigma 1.5 \
#     --human_num "$human_num"



