#!/bin/bash
# python scripts/train_robust_agent.py \
#     --algo robot_crowd_happo \
#     --env crowd_env \
#     --exp_name ped_sim \
#     --load_config tuned_configs/crowd_navi/ped_sim/happo_5p.json \
#     --cuda_device cuda:0

# python scripts/train_robust_agent.py \
#     --algo robot_crowd_happo \
#     --env crowd_env \
#     --exp_name ped_sim \
#     --load_config tuned_configs/crowd_navi/ped_sim/happo_10p.json \
#     --cuda_device cuda:1


conda_env=url_navi
declare -a sessions_and_scripts=(
    # "1ai_4p_6crvs_room361:training_runner/on_policy_single_agent_runner.py \
    # --load_config configs/room_361/happo_4p_3c_rvs_room361.json \
    # --exp_name train_on_ai_090_4p_3c_rvs_room361\
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361 \
    # --cuda_device cuda:0"
    # "2ai_4p_6crvs_room361:training_runner/on_policy_single_agent_runner.py \
    # --load_config configs/room_361/happo_4p_3c_rvs_room361.json \
    # --exp_name train_on_ai_085_4p_3c_rvs_room361\
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.85_happo_5p_3c_rvs_room361 \
    # --cuda_device cuda:1"
    # "3ai_4p_6crvs_room361:training_runner/on_policy_single_agent_runner.py \
    # --load_config configs/room_361/happo_4p_3c_rvs_room361.json \
    # --exp_name train_on_ai_095_4p_3c_rvs_room361\
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.95_happo_5p_3c_rvs_room361 \
    # --cuda_device cuda:2"
    # "4ai_4p_6crvs_room361:training_runner/on_policy_single_agent_runner.py \
    # --load_config configs/room_361/happo_4p_6c_rvs_room361.json \
    # --exp_name train_on_ai_085_4p_6c_rvs_room361\
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.85_happo_5p_6c_rvs_room361 \
    # --cuda_device cuda:0"
    # "5ai_4p_6crvs_room361:training_runner/on_policy_single_agent_runner.py \
    # --load_config configs/room_361/happo_4p_6c_rvs_room361.json \
    # --exp_name train_on_ai_095_4p_6c_rvs_room361\
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.95_happo_5p_6c_rvs_room361 \
    # --cuda_device cuda:1"
    # "sfm_4p_room361:training_runner/on_policy_single_agent_sfm_runner.py \
    # --load_config configs/room_361/happo_4p_6c_nvs_room361.json \
    # --cuda_device cuda:1"
    # "ai_4p_sp_rvs_room361:training_runner/on_policy_single_agent_runner.py \
    # --exp_name train_on_ai_4p_sp_rvs_room361\
    # --load_config configs/room_361/happo_4p_sp_rvs_room361.json \
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
    # --cuda_device cuda:2"

    "s1:training_runner/on_policy_single_agent_runner.py \
    --log_dir sim2real/256 \
    --exp_name train_on_ai_090_1p_6c_rvs_room256\
    --load_config configs/room_256/happo_1p_6c_rvs_room256.json \
    --human_model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_2p_6c_rvs_room256 \
    --cuda_device cuda:1"
    "s2:training_runner/on_policy_single_agent_runner.py \
    --log_dir sim2real/256 \
    --exp_name train_on_ai_090_2p_6c_rvs_room256\
    --load_config configs/room_256/happo_2p_6c_rvs_room256.json \
    --human_model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_3p_6c_rvs_room256 \
    --cuda_device cuda:2"
    "s3:training_runner/on_policy_single_agent_runner.py \
    --log_dir sim2real/256 \
    --exp_name train_on_ai_1p_sp_rvs_room256\
    --load_config configs/room_256/happo_1p_sp_rvs_room256.json \
    --human_model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_2p_sp_rvs_room256 \
    --cuda_device cuda:1"
    "s4:training_runner/on_policy_single_agent_runner.py \
    --log_dir sim2real/256 \
    --exp_name train_on_ai_2p_sp_rvs_room256\
    --load_config configs/room_256/happo_2p_sp_rvs_room256.json \
    --human_model_dir /home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_3p_sp_rvs_room256 \
    --cuda_device cuda:2"
    # "sfm_4p_cc:training_runner/on_policy_single_agent_sfm_runner.py \
    # --load_config configs/circle_cross_4P/happo_4p_sp_rvs_circlecross.json \
    # --cuda_device cuda:1"
    # "ai_4p_sp_rvs_cc:training_runner/on_policy_single_agent_runner.py \
    # --exp_name train_on_ai_4p_sp_rvs_circlecross\
    # --load_config configs/circle_cross_4P/happo_4p_sp_rvs_circlecross.json \
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_circlecross \
    # --cuda_device cuda:2"


)


for item in "${sessions_and_scripts[@]}"
do
    # Split the session, environment, and script information
    IFS=':' read -r session_name script_path <<< "$item"
    
    # Start a new Screen session with the specified name
    # Activate the conda environment and execute the Python script within the session
    screen -dmS "$session_name" bash -c "source activate $conda_env && python $script_path; exec bash"
    
    # Optional: Provide some feedback about the session creation
    echo "Started Screen session '$session_name' running script '$script_path' in conda environment '$conda_env'"

    sleep 5
done