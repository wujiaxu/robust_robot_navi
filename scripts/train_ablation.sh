#!/bin/bash

conda_env=url_navi
seed=1
optimality=0.90
optimalitystr=$(printf "%.2f" "$optimality")
# configs
group1=room361_5p_nvs
happo_5p_sp_nvs_room361=happo_5p_sp_nvs_room361
ppo_5p_3c_nvs_room361=ppo_5p_3c_nvs_room361
ppo_5p_6c_nvs_room361=ppo_5p_6c_nvs_room361

group2=room361_5p_rvs
happo_5p_sp_rvs_room361=happo_5p_sp_rvs_room361
ppo_5p_3c_rvs_room361=ppo_5p_3c_rvs_room361
ppo_5p_6c_rvs_room361=ppo_5p_6c_rvs_room361

group3=circle_cross_5p_nvs
happo_5p_sp_nvs_circlecross=happo_5p_sp_nvs_circlecross
ppo_5p_3c_nvs_circlecross=ppo_5p_3c_nvs_circlecross
ppo_5p_6c_nvs_circlecross=ppo_5p_6c_nvs_circlecross

group4=circle_cross_5p_rvs
happo_5p_sp_rvs_circlecross=happo_5p_sp_rvs_circlecross
ppo_5p_3c_rvs_circlecross=ppo_5p_3c_rvs_circlecross
ppo_5p_6c_rvs_circlecross=ppo_5p_6c_rvs_circlecross

# group5=circle_cross_5-10p_rvs
# happo_5-10p_sp_rvs_circlecross=happo_5-10p_sp_rvs_circlecross
# ppo_5-10p_3c_rvs_circlecross=ppo_5-10p_3c_rvs_circlecross
# ppo_5-10p_6c_rvs_circlecross=ppo_5-10p_6c_rvs_circlecross

group6=circle_cross_10p_rvs
happo_10p_sp_rvs_circlecross=happo_10p_sp_rvs_circlecross
ppo_10p_3c_rvs_circlecross=ppo_10p_3c_rvs_circlecross

result_dir=results_seed_"$seed" 

# python scripts/train_robust_agent_cmdp.py \
#     --log_dir $result_dir \
#     --seed $seed \
#     --algo robot_crowd_happo \
#     --env crowd_env \
#     --exp_name "$optimalitystr"_$ppo_5p_3c_rvs_circlecross \
#     --optimality $optimality \
#     --lagrangian_k_p 1.0 \
#     --lagrangian_k_i 0.003 \
#     --load_config tuned_configs/ha"$ppo_5p_3c_rvs_circlecross".json \
#     --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_5p_sp_rvs_circlecross \
#     --cuda_device cuda:0

conda_env=url_navi
declare -a sessions_and_scripts=(
#    "ablation_1: scripts/train_robust_agent_cmdp.py \
#     --log_dir $result_dir \
#     --seed $seed \
#     --algo robot_crowd_happo \
#     --env crowd_env \
#     --exp_name c"$optimalitystr"_$ppo_5p_3c_rvs_circlecross \
#     --optimality $optimality \
#     --lagrangian_k_p 1.0 \
#     --lagrangian_k_i 0.003 \
#     --load_config tuned_configs/ha"$ppo_5p_3c_rvs_circlecross".json \
#     --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_5p_sp_rvs_circlecross \
#     --cuda_device cuda:0"
    
    "ablation_2: scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$ppo_10p_3c_rvs_circlecross \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/ha"$ppo_10p_3c_rvs_circlecross".json \
    --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_10p_sp_rvs_circlecross \
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