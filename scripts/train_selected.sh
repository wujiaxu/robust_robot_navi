#!/bin/bash

conda_env=url_navi
version=1
seed=1
optimality=0.9
#0.90
optimalitystr=$(printf "%.2f" "$optimality")
# configs
group1=room361_5p_nvs
happo_5p_sp_nvs_room361=happo_5p_sp_nvs_room361
happo_5p_3c_nvs_room361=happo_5p_3c_nvs_room361
happo_5p_6c_nvs_room361=happo_5p_6c_nvs_room361

group2=room361_5p_rvs
happo_5p_sp_rvs_room361=happo_5p_sp_rvs_room361
happo_5p_3c_rvs_room361=happo_5p_3c_rvs_room361
happo_5p_6c_rvs_room361=happo_5p_6c_rvs_room361

group3=circle_cross_5p_nvs
happo_5p_sp_nvs_circlecross=happo_5p_sp_nvs_circlecross
happo_5p_3c_nvs_circlecross=happo_5p_3c_nvs_circlecross
happo_5p_6c_nvs_circlecross=happo_5p_6c_nvs_circlecross

group4=circle_cross_5p_rvs
happo_5p_sp_rvs_circlecross=happo_5p_sp_rvs_circlecross
happo_5p_3c_rvs_circlecross=happo_5p_3c_rvs_circlecross
happo_5p_6c_rvs_circlecross=happo_5p_6c_rvs_circlecross

group5=circle_cross_10p_rvs
happo_10p_sp_rvs_circlecross=happo_10p_sp_rvs_circlecross
happo_10p_3c_rvs_circlecross=happo_10p_3c_rvs_circlecross
happo_10p_3c_rvs_circlecross_long=happo_10p_3c_rvs_circlecross_long

group6=ucy_students_40p_rvs
happo_12p_sp_rvs_UCYstudents=happo_12p_sp_rvs_UCYstudents
happo_12p_3c_rvs_UCYstudents=happo_12p_3c_rvs_UCYstudents

group7=room256_4p_rvs
happo_4p_sp_rvs_room256=happo_4p_sp_rvs_room256
happo_4p_3c_rvs_room256=happo_4p_3c_rvs_room256
happo_4p_6c_rvs_room256=happo_4p_6c_rvs_room256
ppo_4p_ccp_rvs_room256=ppo_4p_ccp_rvs_room256

group8=room256_2p_rvs
happo_2p_sp_rvs_room256=happo_2p_sp_rvs_room256
happo_2p_3c_rvs_room256=happo_2p_3c_rvs_room256
happo_2p_6c_rvs_room256=happo_2p_6c_rvs_room256
ppo_3p_ccp_rvs_room256=ppo_3p_ccp_rvs_room256

group9=room256_3p_rvs
happo_3p_sp_rvs_room256=happo_3p_sp_rvs_room256
happo_3p_3c_rvs_room256=happo_3p_3c_rvs_room256
happo_3p_6c_rvs_room256=happo_3p_6c_rvs_room256
ppo_3p_ccp_rvs_room256=ppo_3p_ccp_rvs_room256

group10=ucy_3p_rvs

# result_dir=room256_results_ver_"$version"_seed_"$seed" 
result_dir=results_ver_2_seed_"$seed"

# python scripts/train_robust_agent_cmdp.py \
#     --log_dir $result_dir \
#     --seed $seed \
#     --algo robot_crowd_happo \
#     --env crowd_env \
#     --exp_name c"$optimalitystr"_happo_CNN1D_5-5_6c_rvs_room361 \
#     --optimality $optimality \
#     --lagrangian_k_p 1.0 \
#     --lagrangian_k_i 0.003 \
#     --lagrangian_lower_bound 1 \
#     --load_config train_configs/happo_CNN_1D_5-5_6c_rvs_room361.json \
#     --base_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
#     --cuda_device cuda:0

# python scripts/train_robust_agent_cmdp.py \
#     --log_dir $result_dir \
#     --seed $seed \
#     --algo robot_crowd_happo \
#     --env crowd_env \
#     --exp_name c"$optimalitystr"_happo_CNN1D_5-5_12c_rvs_room361 \
#     --optimality $optimality \
#     --lagrangian_k_p 1.0 \
#     --lagrangian_k_i 0.003 \
#     --lagrangian_lower_bound 5 \
#     --load_config train_configs/happo_CNN_1D_5-5_12c_rvs_room361.json \
#     --base_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
#     --cuda_device cuda:2

# python scripts/train_robust_agent.py \
#     --log_dir $result_dir \
#     --seed $seed \
#     --algo robot_crowd_happo \
#     --env crowd_env \
#     --exp_name happo_CNN1D_5-5_sp_rvs_room361 \
#     --load_config train_configs/happo_CNN_1D_5-5_sp_rvs_room361.json \
#     --cuda_device cuda:0


conda_env=url_navi
# Declare arrays with session names and their respective scripts
declare -a sessions_and_scripts=(
    
    # "$group8:scripts/train_robust_agent.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name $happo_2p_sp_rvs_room256 \
    # --load_config tuned_configs/$happo_2p_sp_rvs_room256.json \
    # --cuda_device cuda:0"

    # # - with constraint
    # "$group8:scripts/train_robust_agent_cha.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_$happo_2p_3c_rvs_room256 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --load_config tuned_configs/$happo_2p_3c_rvs_room256.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_2p_sp_rvs_room256 \
    # --cuda_device cuda:0"
    # "$group8:scripts/train_robust_agent_cha.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_$happo_2p_6c_rvs_room256 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --load_config tuned_configs/$happo_2p_6c_rvs_room256.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_2p_sp_rvs_room256 \
    # --cuda_device cuda:0"

    
    # "$group9:scripts/train_robust_agent.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name $happo_3p_sp_rvs_room256 \
    # --load_config tuned_configs/$happo_3p_sp_rvs_room256.json \
    # --cuda_device cuda:1"

    # # - with constraint
    # "$group9:scripts/train_robust_agent_cha.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_$happo_3p_3c_rvs_room256 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --load_config tuned_configs/$happo_3p_3c_rvs_room256.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_3p_sp_rvs_room256 \
    # --cuda_device cuda:1"
    # "$group9:scripts/train_robust_agent_cha.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_$happo_3p_6c_rvs_room256 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --load_config tuned_configs/$happo_3p_6c_rvs_room256.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_3p_sp_rvs_room256 \
    # --cuda_device cuda:1"

    # "$group10:scripts/train_robust_agent.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name happo_EGCL_6-12_sp_rvs_ucystudents \
    # --load_config train_configs/happo_EGCL_6-12_sp_rvs_ucystudents.json \
    # --cuda_device cuda:0"

    # "$group10:scripts/train_robust_agent_cmdp.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_happo_EGCL_6-12_3c_rvs_ucystudents \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --lagrangian_lower_bound 5 \
    # --load_config train_configs/happo_EGCL_6-12_3c_rvs_ucystudents.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/happo_EGCL_6-12_sp_rvs_ucystudents \
    # --cuda_device cuda:0"

    # "r256_6:scripts/train_robust_agent.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name happo_CNN_1D_6-6_sp_rvs_room256 \
    # --load_config train_configs/happo_CNN_1D_6-6_sp_rvs_room256.json \
    # --cuda_device cuda:0"

    # "r256_6:scripts/train_robust_agent_cmdp.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_happo_CNN_1D_6-6_6c_rvs_room256 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --lagrangian_lower_bound 1 \
    # --load_config train_configs/happo_CNN_1D_6-6_6c_rvs_room256.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/happo_CNN_1D_6-6_sp_rvs_room256 \
    # --cuda_device cuda:0"

    # "r256_6:scripts/train_robust_agent_cmdp.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_happo_CNN_1D_6-6_3c_rvs_room256 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --lagrangian_lower_bound 1 \
    # --load_config train_configs/happo_CNN_1D_6-6_3c_rvs_room256.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/happo_CNN_1D_6-6_sp_rvs_room256 \
    # --cuda_device cuda:0"

    # "r256_8:scripts/train_robust_agent.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name happo_CNN_1D_8-8_sp_rvs_room256 \
    # --load_config train_configs/happo_CNN_1D_8-8_sp_rvs_room256.json \
    # --cuda_device cuda:1"

    # "r256_8:scripts/train_robust_agent_cmdp.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_happo_CNN_1D_8-8_6c_rvs_room256 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --lagrangian_lower_bound 1 \
    # --load_config train_configs/happo_CNN_1D_8-8_6c_rvs_room256.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/happo_CNN_1D_8-8_sp_rvs_room256 \
    # --cuda_device cuda:1"

    # "r256_8:scripts/train_robust_agent_cmdp.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_happo_CNN_1D_8-8_3c_rvs_room256 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --lagrangian_lower_bound 1 \
    # --load_config train_configs/happo_CNN_1D_8-8_3c_rvs_room256.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/happo_CNN_1D_8-8_sp_rvs_room256 \
    # --cuda_device cuda:1"


    # "$r256_6:scripts/train_robust_agent.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name happo_CNN_1D_6-6_sp_rvs_room256 \
    # --load_config train_configs/happo_CNN_1D_6-6_sp_rvs_room256.json \
    # --cuda_device cuda:0"

    # "$r256_6:scripts/train_robust_agent_cmdp.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_happo_CNN_1D_6-6_6c_rvs_room256 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --lagrangian_lower_bound 1 \
    # --load_config train_configs/happo_CNN_1D_6-6_6c_rvs_room256.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/happo_CNN_1D_6-6_sp_rvs_room256 \
    # --cuda_device cuda:0"


    "group5:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name happo_CNN_1D_7-7_sp_rvs_circlecross \
    --load_config train_configs/happo_CNN_1D_7-7_sp_rvs_circlecross.json \
    --cuda_device cuda:1"

    "group5:scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_happo_CNN_1D_7-7_3c_rvs_circlecross \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --lagrangian_lower_bound 5 \
    --load_config train_configs/happo_CNN_1D_7-7_3c_rvs_circlecross.json \
    --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/happo_CNN_1D_7-7_sp_rvs_circlecross \
    --cuda_device cuda:1"

    # "room361_3c:scripts/train_robust_agent_cha.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_happo_CNN1D_5-5_3c_rvs_room361 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --lagrangian_lower_bound 1 \
    # --load_config train_configs/happo_CNN_1D_5-5_3c_rvs_room361.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
    # --cuda_device cuda:0"

    # "room361_24c:scripts/train_robust_agent_cmdp.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_happo_CNN1D_5-5_24c_rvs_room361 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --lagrangian_lower_bound 1 \
    # --load_config train_configs/happo_CNN_1D_5-5_24c_rvs_room361.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
    # --cuda_device cuda:1"

    # "room361_12c:scripts/train_robust_agent_cmdp.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_happo_CNN1D_5-5_12c_rvs_room361 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --lagrangian_lower_bound 1 \
    # --load_config train_configs/happo_CNN_1D_5-5_12c_rvs_room361.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
    # --cuda_device cuda:2"

)

# Loop through the sessions and scripts
declare -A session_scripts_map

# Organize scripts under corresponding session names
for item in "${sessions_and_scripts[@]}"; do
    IFS=':' read -r session_name script_cmd <<< "$item"
    session_scripts_map["$session_name"]+="python $script_cmd && "
done

# # Execute all scripts for each session in a Screen session
for session_name in "${!session_scripts_map[@]}"; do
    script_chain="${session_scripts_map[$session_name]}"

    # Remove trailing ' && ' from the final command
    script_chain=${script_chain::-4}

    # Start a new Screen session with the specified name
    screen -dmS "$session_name" bash -c "source activate $conda_env && $script_chain; exec bash"

    # Optional: Provide feedback about the session creation
    echo "Started Screen session '$session_name' running scripts: $script_chain in conda environment '$conda_env'"

    sleep 5
done
