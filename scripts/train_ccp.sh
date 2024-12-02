#!/bin/bash

conda_env=url_navi
seed=1
version=1
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
ppo_5p_ccp_rvs_circlecross=ppo_5p_ccp_rvs_circlecross

group5=circle_cross_10p_rvs
happo_10p_sp_rvs_circlecross=happo_10p_sp_rvs_circlecross
ppo_10p_3c_rvs_circlecross=ppo_10p_3c_rvs_circlecross

ppo_4p_ccp_rvs_room256=ppo_4p_ccp_rvs_room256
ppo_2p_ccp_rvs_room256=ppo_2p_ccp_rvs_room256
ppo_3p_ccp_rvs_room256=ppo_3p_ccp_rvs_room256
result_dir=room256_results_ver_"$version"_seed_"$seed"  

# python scripts/train_robust_agent.py \
#     --log_dir $result_dir \
#     --seed $seed \
#     --algo robot_crowd_ppo \
#     --env crowd_env_ccp \
#     --exp_name $ppo_5p_ccp_rvs_circlecross \
#     --load_config tuned_configs/$ppo_5p_ccp_rvs_circlecross.json \
#     --cuda_device cuda:0
conda_env=url_navi
declare -a sessions_and_scripts=(
    # "ccp1: scripts/train_robust_agent.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_ppo \
    # --env crowd_env_ccp \
    # --exp_name $ppo_5p_ccp_rvs_circlecross \
    # --load_config tuned_configs/$ppo_5p_ccp_rvs_circlecross.json \
    # --cuda_device cuda:1"

    "ccp1: scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_ppo \
    --env crowd_env_ccp \
    --exp_name $ppo_3p_ccp_rvs_room256 \
    --load_config tuned_configs/$ppo_3p_ccp_rvs_room256.json \
    --cuda_device cuda:1"
    "ccp2: scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_ppo \
    --env crowd_env_ccp \
    --exp_name $ppo_2p_ccp_rvs_room256 \
    --load_config tuned_configs/$ppo_2p_ccp_rvs_room256.json \
    --cuda_device cuda:2"


    # "wo_d: scripts/train_robust_agent.py --algo robot_crowd_happo_woD --env crowd_env --exp_name robust_navi_ablation_woD --cuda_device cuda:1"
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