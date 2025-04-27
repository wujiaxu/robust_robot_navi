#!/bin/bash
conda_env=url_navi
declare -a sessions_and_scripts=(

"ss1:drl_vo_train_adhoc.py \
    --algo robot_crowd_ppo \
    --cuda_device cuda:1"

# "ss1:drl_vo_train.py \
#     --exp_name proposed_4p_room361_2 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361 \
#     --algo robot_crowd_ppo \
#     --cuda_device cuda:1"


# "ss_ccp:drl_vo_train.py \
#     --exp_name ccp_4p_room361_2 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_ver_2_seed_1/crowd_env_ccp/crowd_navi/robot_crowd_ppo/ppo_CNN_1D_5-5_ccp_rvs_room361 \
#     --algo robot_crowd_ppo \
#     --cuda_device cuda:2"

# "ss_sp:drl_vo_train.py \
#     --exp_name sp_4p_room361_2 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:0"

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