

#!/bin/bash

# python test_runner/ped_sim_mocap_data_test_runner.py \
#         --log_dir ped_sim \
#         --exp_name ai_crowdsim_3p_rvs_ccp_room256 \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env_ccp/crowd_navi/robot_crowd_ppo/ppo_3p_ccp_rvs_room256 \
#         --cuda_device cuda:1

# python test_runner/ped_sim_test_runner.py \
#         --log_dir ped_sim \
#         --scenario circle_cross \
#         --human_num 10 \
#         --test_episode 10 \
#         --cuda_device cuda:2 \
#         --exp_name ai_crowdsim_10p_rvs_3c_circlecross \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_10p_3c_rvs_circlecross 

# python test_runner/ped_sim_test_runner.py \
#         --log_dir ped_sim \
#         --scenario circle_cross \
#         --human_num 7 \
#         --test_episode 10 \
#         --cuda_device cuda:2 \
#         --exp_name ai_crowdsim_7p_rvs_3c_circlecross \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross 

# python test_runner/ped_sim_test_runner.py \
#         --log_dir ped_sim \
#         --scenario circle_cross \
#         --human_num 5 \
#         --test_episode 10 \
#         --cuda_device cuda:2 \
#         --exp_name ai_crowdsim_5p_rvs_3c_circlecross \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross 

python test_runner/ped_sim_test_runner.py \
        --log_dir ped_sim \
        --scenario room_361 \
        --human_num 5 \
        --test_episode 10 \
        --cuda_device cuda:2 \
        --exp_name ai_crowdsim_5p_rvs_3c_room361 \
        --model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361 

# python test_runner/ped_sim_test_runner.py \
#         --log_dir ped_sim \
#         --scenario room_256 \
#         --human_num 2 \
#         --test_episode 10 \
#         --cuda_device cuda:2 \
#         --exp_name ai_crowdsim_2p_rvs_6c_room256 \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_2p_6c_rvs_room256

# python test_runner/ped_sim_test_runner.py \
#         --log_dir ped_sim \
#         --scenario room_256 \
#         --human_num 4 \
#         --test_episode 10 \
#         --cuda_device cuda:2 \
#         --exp_name ai_crowdsim_4p_rvs_6c_room256 \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_4p_6c_rvs_room256

# python test_runner/ped_sim_test_runner.py \
#         --log_dir ped_sim \
#         --scenario room_256 \
#         --human_num 6 \
#         --test_episode 10 \
#         --cuda_device cuda:2 \
#         --exp_name ai_crowdsim_6p_rvs_6c_room256 \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/results_ver_2_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_CNN_1D_6-6_6c_rvs_room256

# conda_env=url_navi
# declare -a sessions_and_scripts=(
#     "s1: test_runner/ped_sim_mocap_data_test_runner.py \
#         --log_dir ped_sim \
#         --exp_name ai_crowdsim_2p_rvs_6c_room256 \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_2p_6c_rvs_room256 \
#         --cuda_device cuda:0"

#     "s2: test_runner/ped_sim_mocap_data_test_runner.py \
#         --log_dir ped_sim \
#         --exp_name ai_crowdsim_3p_rvs_6c_room256 \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_3p_6c_rvs_room256 \
#         --cuda_device cuda:0"

#     "s3: test_runner/ped_sim_mocap_data_test_runner.py \
#         --log_dir ped_sim \
#         --exp_name ai_crowdsim_2p_rvs_ccp_room256 \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env_ccp/crowd_navi/robot_crowd_ppo/ppo_2p_ccp_rvs_room256 \
#         --cuda_device cuda:1"

#     "s4: test_runner/ped_sim_mocap_data_test_runner.py \
#         --log_dir ped_sim \
#         --exp_name ai_crowdsim_3p_rvs_ccp_room256 \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env_ccp/crowd_navi/robot_crowd_ppo/ppo_3p_ccp_rvs_room256 \
#         --cuda_device cuda:1"

#     "s5: test_runner/ad_hoc_ped_sim_mocap_data_test_runner.py \
#         --log_dir ped_sim \
#         --exp_name ad_hoc_crowdsim_2p_rvs_pt_room256 \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_2p_6c_rvs_room256 \
#         --cuda_device cuda:2"

#     "s6: test_runner/ad_hoc_ped_sim_mocap_data_test_runner.py \
#         --log_dir ped_sim \
#         --exp_name ad_hoc_crowdsim_3p_rvs_pt_room256 \
#         --model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_3p_6c_rvs_room256 \
#         --cuda_device cuda:2"
# )

# for item in "${sessions_and_scripts[@]}"
# do
#     # Split the session, environment, and script information
#     IFS=':' read -r session_name script_path <<< "$item"
    
#     # Start a new Screen session with the specified name
#     # Activate the conda environment and execute the Python script within the session
#     screen -dmS "$session_name" bash -c "source activate $conda_env && python $script_path; exec bash"
    
#     # Optional: Provide some feedback about the session creation
#     echo "Started Screen session '$session_name' running script '$script_path' in conda environment '$conda_env'"

#     sleep 5
# done