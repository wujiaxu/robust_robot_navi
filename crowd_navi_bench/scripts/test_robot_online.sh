#!/bin/bash


#!/bin/bash

# Define the root directory
root_dir="/home/dl/wu_ws/HARL/crowd_navi_bench/results_vis_aware/crowd_env_vis/crowd_navi/robot_crowd_happo"


human_num=4
result_dir="single_life_results"

# python test_runner/ai_crowd_model_online_navi.py \
#     --log_dir $result_dir \
#     --logc 200000 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_r_train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online      
    
# python test_runner/ai_crowd_model_online_navi_ablade.py \
#     --log_dir $result_dir \
#     --logc 90 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_90_train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online_ablade

# python test_runner/ai_crowd_model_online_navi_ablade.py \
#     --log_dir $result_dir \
#     --logc 100 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_100_train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online_ablade


# python test_runner/ai_crowd_model_online_navi_ablade.py \
#     --log_dir $result_dir \
#     --logc 75 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_75_train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online_ablade

# python test_runner/ad_hoc_crowd_model_sfm_test_runner.py \
#     --log_dir $result_dir \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --scenario circle_cross \
#     --human_policy ORCA \
#     --exp_name SFM_vs_ORCA_crowd_20p_circlecross  \
#     --human_num 20 

# python test_runner/ad_hoc_crowd_model_sfm_test_runner.py \
#     --log_dir $result_dir \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --scenario circle_cross \
#     --human_policy ORCA \
#     --exp_name SFM_vs_ORCA_crowd_15p_circlecross  \
#     --human_num 15

# python test_runner/ad_hoc_crowd_model_sfm_test_runner.py \
#     --log_dir $result_dir \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 1.0 \
#     --test_episode 100 \
#     --scenario circle_cross \
#     --exp_name SFM_vs_ORCA_crowd_12p_circlecross  \
#     --human_num 12

# python test_runner/ad_hoc_crowd_model_sfm_test_runner.py \
#     --log_dir $result_dir \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 1.0 \
#     --test_episode 100 \
#     --scenario circle_cross \
#     --exp_name SFM_vs_ORCA_crowd_5p_circlecross  \
#     --human_num 4

python test_runner/ad_hoc_crowd_model_sfm_test_runner.py \
    --log_dir $result_dir \
    --sfm_v0 10 \
    --sfm_sigma 0.3 \
    --sfm_initial_speed 1.0 \
    --test_episode 100 \
    --scenario room_256 \
    --exp_name SFM_vs_ORCA_crowd_3p_room256  \
    --human_num 3 

# python test_runner/ad_hoc_crowd_model_sfm_test_runner.py \
#     --log_dir $result_dir \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 1.0 \
#     --test_episode 100 \
#     --scenario room_256 \
#     --exp_name SFM_vs_ORCA_crowd_5p_room256  \
#     --human_num 5 

# python test_runner/ad_hoc_crowd_model_sfm_test_runner.py \
#     --log_dir $result_dir \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 1.0 \
#     --test_episode 100 \
#     --scenario room_256 \
#     --exp_name SFM_vs_ORCA_crowd_6p_room256  \
#     --human_num 6 

# python test_runner/ad_hoc_crowd_model_sfm_test_runner.py \
#     --log_dir $result_dir \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 1.0 \
#     --test_episode 100 \
#     --scenario room_256 \
#     --exp_name SFM_vs_ORCA_crowd_7p_room256  \
#     --human_num 7 

# python test_runner/ad_hoc_crowd_model_sfm_test_runner.py \
#     --log_dir $result_dir \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 1.0 \
#     --test_episode 100 \
#     --scenario room_256 \
#     --exp_name SFM_vs_ORCA_crowd_9p_room256  \
#     --human_num 9 

# python test_runner/ai_crowd_model_online_navi.py \
#     --log_dir $result_dir \
#     --logc 0 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_0_train_on_ai_090_4p_3c_rvs_circlecross_vs_happo_5p_3c_rvs_circlecross_online 

# python test_runner/ai_crowd_model_online_navi_ablade.py \
#     --log_dir $result_dir \
#     --logc 80 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_80_train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online_ablade  

# python test_runner/ai_crowd_model_online_navi.py \
#     --log_dir $result_dir \
#     --logc 200000 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_r_train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online    

# python test_runner/ai_crowd_model_online_navi.py \
#     --log_dir $result_dir \
#     --logc 70 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_70_train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online

# python test_runner/ai_crowd_model_online_navi.py \
#     --log_dir $result_dir \
#     --logc 65 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_65_train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online

# python test_runner/ai_crowd_model_online_navi.py \
#     --log_dir $result_dir \
#     --logc 60 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_60_train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online


# python test_runner/ai_crowd_model_online_navi.py \
#     --log_dir $result_dir \
#     --logc 0 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_0_train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online    

# python test_runner/ai_crowd_model_online_navi.py \
#     --log_dir $result_dir \
#     --logc 80 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --sfm_initial_speed 0.5 \
#     --test_episode 100 \
#     --exp_name logc_80_train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online  





