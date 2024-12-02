#!/bin/bash

conda_env=url_navi
version=1
seed=1
optimality=0.90
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

result_dir=room256_results_ver_"$version"_seed_"$seed" 

conda_env=url_navi
# Declare arrays with session names and their respective scripts
declare -a sessions_and_scripts=(
# group 1: room 361 5p nvs
    # - base model
    # "$group7:scripts/train_robust_agent.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name $happo_4p_sp_rvs_room256 \
    # --load_config tuned_configs/$happo_4p_sp_rvs_room256.json \
    # --cuda_device cuda:0"

    # # - with constraint
    # "$group7:scripts/train_robust_agent_cha.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_$happo_4p_3c_rvs_room256 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --load_config tuned_configs/$happo_4p_3c_rvs_room256.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_4p_sp_rvs_room256 \
    # --cuda_device cuda:0"
    # "$group7:scripts/train_robust_agent_cha.py \
    # --log_dir $result_dir \
    # --seed $seed \
    # --algo robot_crowd_happo \
    # --env crowd_env \
    # --exp_name c"$optimalitystr"_$happo_4p_6c_rvs_room256 \
    # --optimality $optimality \
    # --lagrangian_k_p 1.0 \
    # --lagrangian_k_i 0.003 \
    # --load_config tuned_configs/$happo_4p_6c_rvs_room256.json \
    # --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_4p_sp_rvs_room256 \
    # --cuda_device cuda:0"


    "$group8:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_2p_sp_rvs_room256 \
    --load_config tuned_configs/$happo_2p_sp_rvs_room256.json \
    --cuda_device cuda:0"

    # - with constraint
    "$group8:scripts/train_robust_agent_cha.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_2p_3c_rvs_room256 \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_2p_3c_rvs_room256.json \
    --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_2p_sp_rvs_room256 \
    --cuda_device cuda:0"
    "$group8:scripts/train_robust_agent_cha.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_2p_6c_rvs_room256 \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_2p_6c_rvs_room256.json \
    --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_2p_sp_rvs_room256 \
    --cuda_device cuda:0"

    
    "$group9:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_3p_sp_rvs_room256 \
    --load_config tuned_configs/$happo_3p_sp_rvs_room256.json \
    --cuda_device cuda:1"

    # - with constraint
    "$group9:scripts/train_robust_agent_cha.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_3p_3c_rvs_room256 \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_3p_3c_rvs_room256.json \
    --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_3p_sp_rvs_room256 \
    --cuda_device cuda:1"
    "$group9:scripts/train_robust_agent_cha.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_3p_6c_rvs_room256 \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_3p_6c_rvs_room256.json \
    --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_3p_sp_rvs_room256 \
    --cuda_device cuda:1"

)

# Loop through the sessions and scripts
declare -A session_scripts_map

# Organize scripts under corresponding session names
for item in "${sessions_and_scripts[@]}"; do
    IFS=':' read -r session_name script_cmd <<< "$item"
    session_scripts_map["$session_name"]+="python $script_cmd && "
done

# Execute all scripts for each session in a Screen session
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

# declare -a sessions_and_scripts=(
#    "select_1: scripts/train_robust_agent_cha.py \
#     --log_dir $result_dir \
#     --seed $seed \
#     --algo robot_crowd_happo \
#     --env crowd_env \
#     --exp_name c"$optimalitystr"_$happo_5p_3c_rvs_circlecross \
#     --optimality $optimality \
#     --lagrangian_k_p 1.0 \
#     --lagrangian_k_i 0.003 \
#     --load_config tuned_configs/"$happo_5p_3c_rvs_circlecross".json \
#     --base_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_"$seed"/crowd_env/crowd_navi/robot_crowd_happo/$happo_5p_sp_rvs_circlecross \
#     --cuda_device cuda:0"
    
#     "select_2: scripts/train_robust_agent_cha.py \
#     --log_dir $result_dir \
#     --seed $seed \
#     --algo robot_crowd_happo \
#     --env crowd_env \
#     --exp_name c"$optimalitystr"_$happo_10p_3c_rvs_circlecross \
#     --optimality $optimality \
#     --lagrangian_k_p 1.0 \
#     --lagrangian_k_i 0.003 \
#     --load_config tuned_configs/"$happo_10p_3c_rvs_circlecross".json \
#     --base_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_"$seed"/crowd_env/crowd_navi/robot_crowd_happo/$happo_10p_sp_rvs_circlecross \
#     --cuda_device cuda:2"

    # "select_3:scripts/train_robust_agent.py \
    #     --log_dir $result_dir \
    #     --seed $seed \
    #     --algo robot_crowd_happo \
    #     --env crowd_env \
    #     --exp_name $happo_12p_sp_rvs_UCYstudents \
    #     --load_config tuned_configs/$happo_12p_sp_rvs_UCYstudents.json \
    #     --cuda_device cuda:2"
    # "select_4:scripts/train_robust_agent.py \
    #     --log_dir $result_dir \
    #     --seed $seed \
    #     --algo robot_crowd_happo \
    #     --env crowd_env \
    #     --exp_name $happo_10p_sp_rvs_circlecross \
    #     --load_config tuned_configs/$happo_10p_sp_rvs_circlecross.json \
    #     --cuda_device cuda:1"
        # "select_5: scripts/train_robust_agent_cha.py \
        #     --log_dir $result_dir \
        #     --seed $seed \
        #     --algo robot_crowd_happo \
        #     --env crowd_env \
        #     --exp_name c"$optimalitystr"_$happo_10p_3c_rvs_circlecross \
        #     --optimality $optimality \
        #     --lagrangian_k_p 1.0 \
        #     --lagrangian_k_i 0.003 \
        #     --load_config tuned_configs/"$happo_10p_3c_rvs_circlecross".json \
        #     --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_10p_sp_rvs_circlecross \
        #     --cuda_device cuda:1"
        # "select_7: scripts/train_robust_agent_cha.py \
        #     --log_dir $result_dir \
        #     --seed $seed \
        #     --algo robot_crowd_happo \
        #     --env crowd_env \
        #     --exp_name c"$optimalitystr"_$happo_10p_3c_rvs_circlecross_long \
        #     --optimality $optimality \
        #     --lagrangian_k_p 1.0 \
        #     --lagrangian_k_i 0.003 \
        #     --load_config tuned_configs/"$happo_10p_3c_rvs_circlecross_long".json \
        #     --base_model_dir /home/dl/wu_ws/robust_robot_navi/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_10p_sp_rvs_circlecross \
        #     --cuda_device cuda:0"
    

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