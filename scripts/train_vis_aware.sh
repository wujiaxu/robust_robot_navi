#!/bin/bash
result_dir="results_vis_aware"
seed=1

conda_env=url_navi
declare -a sessions_and_scripts=(
    "1p: scripts/train_robust_agent.py \
        --log_dir $result_dir \
        --seed $seed \
        --algo robot_crowd_happo \
        --env crowd_env_vis \
        --exp_name dis_1p \
        --load_config tuned_configs/happo_1r4p1dis_sp_rvs_room361.json \
        --distracted_human_num 1 \
        --cuda_device cuda:0"
    "2p: scripts/train_robust_agent.py \
        --log_dir $result_dir \
        --seed $seed \
        --algo robot_crowd_happo \
        --env crowd_env_vis \
        --exp_name dis_2p \
        --load_config tuned_configs/happo_1r4p1dis_sp_rvs_room361.json \
        --distracted_human_num 2 \
        --cuda_device cuda:1"
    "3p: scripts/train_robust_agent.py \
        --log_dir $result_dir \
        --seed $seed \
        --algo robot_crowd_happo \
        --env crowd_env_vis \
        --exp_name dis_3p \
        --load_config tuned_configs/happo_1r4p1dis_sp_rvs_room361.json \
        --distracted_human_num 3 \
        --cuda_device cuda:2"
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