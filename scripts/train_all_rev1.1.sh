#!/bin/bash

# Parameters
conda_env=url_navi
seed=1
cuda_device_base="cuda:0"
cuda_device_alt="cuda:1"
cuda_device_sss="cuda:2"

# Groups and session scripts organized into associative arrays
declare -A groups=(
    [group1]="room361_5p_nvs"
    [group2]="room361_5p_rvs"
    [group3]="circle_cross_5p_nvs"
    [group4]="circle_cross_5p_rvs"
    [group5]="circle_cross_10p_rvs"
)

declare -A configs=(
    [happo_5p_sp_nvs]="happo_5p_sp_nvs_room361 happo_5p_3c_nvs_room361 happo_5p_6c_nvs_room361"
    [happo_5p_sp_rvs]="happo_5p_sp_rvs_room361 happo_5p_3c_rvs_room361 happo_5p_6c_rvs_room361"
    [happo_5p_sp_nvs_circlecross]="happo_5p_sp_nvs_circlecross happo_5p_3c_nvs_circlecross happo_5p_6c_nvs_circlecross"
    [happo_5p_sp_rvs_circlecross]="happo_5p_sp_rvs_circlecross happo_5p_3c_rvs_circlecross happo_5p_6c_rvs_circlecross"
    [happo_10p_sp_rvs_circlecross]="happo_10p_sp_rvs_circlecross happo_10p_3c_rvs_circlecross"
)

# Declare an array for all scripts
declare -a sessions_and_scripts

# Function to append training scripts for both constrained and unconstrained models
append_script3() {
    local group=$1
    local config_name=$2
    local cuda_device=$3
    local cmd

    # Base model (unconstrained)
    cmd="scripts/train_robust_agent.py \
        --seed $seed \
        --algo robot_crowd_happo \
        --env crowd_env \
        --exp_name $config_name \
        --load_config tuned_configs/$config_name.json \
        --cuda_device $cuda_device"
    sessions_and_scripts+=("$group:$cmd")

    # Constrained model
    cmd="scripts/train_robust_agent_cmdp.py \
        --seed $seed \
        --algo robot_crowd_happo \
        --env crowd_env \
        --exp_name $config_name \
        --optimality 0.95 \
        --lagrangian_k_p 1.0 \
        --lagrangian_k_i 0.003 \
        --load_config tuned_configs/$config_name.json \
        --base_model_dir /home/dl/wu_ws/HARL/results/crowd_env/crowd_navi/robot_crowd_happo/$config_name/$seed \
        --cuda_device $cuda_device"
    sessions_and_scripts+=("$group:$cmd")
}

# Populate session script map for all groups and configs
for group in "${!groups[@]}"; do
    group_name=${groups[$group]}

    for config in ${configs[$group_name]}; do
        if [[ "$group_name" =~ "circlecross_5p" ]]; then
            append_script "$group_name" "$config" "$cuda_device_alt"
        elif [[ "$group_name" =~ "circlecross_10p" ]]; then
            append_script "$group_name" "$config" "$cuda_device_sss"
        else
            append_script "$group_name" "$config" "$cuda_device_base"
        fi
    done
done

# Create the session script map
declare -A session_scripts_map
for item in "${sessions_and_scripts[@]}"; do
    IFS=':' read -r session_name script_cmd <<< "$item"
    session_scripts_map["$session_name"]+="python $script_cmd && "
done

# Execute all scripts for each session in a Screen session
for session_name in "${!session_scripts_map[@]}"; do
    script_chain="${session_scripts_map[$session_name]::-4}"  # Remove trailing ' && '

    # Start a new Screen session with the specified name
    screen -dmS "$session_name" bash -c "source activate $conda_env && $script_chain; exec bash"

    # Optional: Provide feedback about the session creation
    echo "Started Screen session '$session_name' running scripts: $script_chain in conda environment '$conda_env'"

    sleep 2  # Adjust sleep to reduce unnecessary waiting
done
