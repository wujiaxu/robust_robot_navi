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
    "ped_sim_5P: scripts/train_robust_agent_cmdp.py \
    --seed 777 \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c_ped_sim \
    --optimality 0.95 \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/crowd_navi/ped_sim/happo_5p_6crvs.json \
    --base_model_dir /home/dl/wu_ws/HARL/results/crowd_env/crowd_navi/robot_crowd_happo/ped_sim_base/happo_5p_rvs_sp \
    --cuda_device cuda:0"
    "ped_sim_5P_room361: scripts/train_robust_agent_cmdp.py \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c_ped_sim \
    --optimality 0.95 \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/crowd_navi/ped_sim/happo_5p_6crvs_room361.json \
    --base_model_dir /home/dl/wu_ws/HARL/results/crowd_env/crowd_navi/robot_crowd_happo/ped_sim_base/happo_5p_rvs_sp_room361 \
    --cuda_device cuda:1"
    "ped_sim_room361_5P: scripts/train_robust_agent.py \
    --seed 777 \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name debug_test \
    --load_config tuned_configs/happo_10p_sp_rvs_circlecross.json \
    --cuda_device cuda:1"
    # "ped_sim_circle_5P: scripts/train_robust_agent.py \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name ped_sim_base \
    # --load_config tuned_configs/crowd_navi/ped_sim/happo_5p_rvs_sp.json \
    # --cuda_device cuda:0"
    # "ped_sim_circle_10P: scripts/train_robust_agent.py \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name ped_sim_base \
    # --load_config tuned_configs/crowd_navi/ped_sim/happo_10p_rvs_sp.json \
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