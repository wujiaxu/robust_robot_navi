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


    "recovery:training_runner/offline_single_agent_cmdp_runner.py"


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