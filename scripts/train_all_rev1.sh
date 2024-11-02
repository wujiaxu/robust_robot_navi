#!/bin/bash

conda_env=url_navi
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

result_dir=results_seed_"$seed" 

# Declare arrays with session names and their respective scripts
declare -a sessions_and_scripts=(
# group 1: room 361 5p nvs
    # - base model
    "$group1:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_sp_nvs_room361 \
    --load_config tuned_configs/$happo_5p_sp_nvs_room361.json \
    --cuda_device cuda:0"
    # - w/o constraint
    "$group1:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_3c_nvs_room361 \
    --load_config tuned_configs/$happo_5p_3c_nvs_room361.json \
    --cuda_device cuda:0"
    "$group1:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_6c_nvs_room361 \
    --load_config tuned_configs/$happo_5p_6c_nvs_room361.json \
    --cuda_device cuda:0"

    # - with constraint
    "$group1:scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_5p_3c_nvs_room361 \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_5p_3c_nvs_room361.json \
    --base_model_dir /home/dl/wu_ws/HARL/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_5p_sp_nvs_room361 \
    --cuda_device cuda:0"
    "$group1:scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_5p_6c_nvs_room361 \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_5p_6c_nvs_room361.json \
    --base_model_dir /home/dl/wu_ws/HARL/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_5p_sp_nvs_room361 \
    --cuda_device cuda:0"
    ### done group 1
# group 2: room 361 5p rvs
    # - base model
    "$group2:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_sp_rvs_room361 \
    --load_config tuned_configs/$happo_5p_sp_rvs_room361.json \
    --cuda_device cuda:0"
    # - w/o constraint
    "$group2:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_3c_rvs_room361 \
    --load_config tuned_configs/$happo_5p_3c_rvs_room361.json \
    --cuda_device cuda:0"
    "$group2:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_6c_rvs_room361 \
    --load_config tuned_configs/$happo_5p_6c_rvs_room361.json \
    --cuda_device cuda:0"

    # - with constraint
    "$group2:scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_5p_3c_rvs_room361 \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_5p_3c_rvs_room361.json \
    --base_model_dir /home/dl/wu_ws/HARL/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_5p_sp_rvs_room361 \
    --cuda_device cuda:0"
    "$group2:scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_5p_6c_rvs_room361 \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_5p_6c_rvs_room361.json \
    --base_model_dir /home/dl/wu_ws/HARL/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_5p_sp_rvs_room361 \
    --cuda_device cuda:0"
    ### done group 2

# group 3: circle cross 5p nvs
    # - base model
    "$group3:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_sp_nvs_circlecross \
    --load_config tuned_configs/$happo_5p_sp_nvs_circlecross.json \
    --cuda_device cuda:1"
    # - w/o constraint
    "$group3:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_3c_nvs_circlecross \
    --load_config tuned_configs/$happo_5p_3c_nvs_circlecross.json \
    --cuda_device cuda:1"
    "$group3:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_6c_nvs_circlecross \
    --load_config tuned_configs/$happo_5p_6c_nvs_circlecross.json \
    --cuda_device cuda:1"

    # - with constraint
    "$group3:scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_5p_3c_nvs_circlecross \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_5p_3c_nvs_circlecross.json \
    --base_model_dir /home/dl/wu_ws/HARL/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_5p_sp_nvs_circlecross \
    --cuda_device cuda:1"
    "$group3:scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_5p_6c_nvs_circlecross \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_5p_6c_nvs_circlecross.json \
    --base_model_dir /home/dl/wu_ws/HARL/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_5p_sp_nvs_circlecross \
    --cuda_device cuda:1"
    ### done group 3

# group 4: circle cross 5p rvs
    # - base model
    "$group4:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_sp_rvs_circlecross \
    --load_config tuned_configs/$happo_5p_sp_rvs_circlecross.json \
    --cuda_device cuda:1"
    # - w/o constraint
    "$group4:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_3c_rvs_circlecross \
    --load_config tuned_configs/$happo_5p_3c_rvs_circlecross.json \
    --cuda_device cuda:1"
    "$group4:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_5p_6c_rvs_circlecross \
    --load_config tuned_configs/$happo_5p_6c_rvs_circlecross.json \
    --cuda_device cuda:1"

    # - with constraint
    "$group4:scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_5p_3c_rvs_circlecross \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_5p_3c_rvs_circlecross.json \
    --base_model_dir /home/dl/wu_ws/HARL/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_5p_sp_rvs_circlecross \
    --cuda_device cuda:1"
    "$group4:scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_5p_6c_rvs_circlecross \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_5p_6c_rvs_circlecross.json \
    --base_model_dir /home/dl/wu_ws/HARL/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_5p_sp_rvs_circlecross \
    --cuda_device cuda:1"
    ### done group 3

# group 5 need larger time limit
# group 5: circle cross 10p rvs
    # # - base model
    "$group5:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_10p_sp_rvs_circlecross \
    --load_config tuned_configs/$happo_10p_sp_rvs_circlecross.json \
    --cuda_device cuda:2"
    # - w/o constraint
    "$group5:scripts/train_robust_agent.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name $happo_10p_3c_rvs_circlecross \
    --load_config tuned_configs/$happo_10p_3c_rvs_circlecross.json \
    --cuda_device cuda:2"

    # - with constraint
    "$group5:scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_$happo_10p_3c_rvs_circlecross \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/$happo_10p_3c_rvs_circlecross.json \
    --base_model_dir /home/dl/wu_ws/HARL/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/$happo_10p_sp_rvs_circlecross \
    --cuda_device cuda:2"
    ### done group 5
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
