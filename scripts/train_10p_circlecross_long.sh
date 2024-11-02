#!/bin/bash

seed=1
result_dir=results_seed_"$seed"
conda_env=url_navi
optimality=0.95
optimalitystr=$(printf "%.2f" "$optimality")
declare -a sessions_and_scripts=(
    "10p_cc:scripts/train_robust_agent_cmdp.py \
    --log_dir $result_dir \
    --seed $seed \
    --algo robot_crowd_happo \
    --env crowd_env \
    --exp_name c"$optimalitystr"_happo_10p_3c_rvs_circlecross_long \
    --optimality $optimality \
    --lagrangian_k_p 1.0 \
    --lagrangian_k_i 0.003 \
    --load_config tuned_configs/happo_10p_3c_rvs_circlecross_long.json \
    --base_model_dir /home/dl/wu_ws/HARL/$result_dir/crowd_env/crowd_navi/robot_crowd_happo/happo_10p_sp_rvs_circlecross \
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