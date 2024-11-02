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
result_dir="results_vis_aware"
seed=1

declare -a sessions_and_scripts=(
    "1dis_5:training_runner/on_policy_single_vis_agent_runner.py \
    --log_dir $result_dir \
    --seed $seed \
    --distracted_human_shooting_range 1.0 \
    --distracted_human_separation_penalty_factor 1.5 \
    --load_config configs/room_361/happo_1r4p1dis_3c_rvs_room361.json \
    --exp_name train_on_vis_1point5_1m_ai_090_4p_3c_rvs_room361\
    --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361 \
    --cuda_device cuda:0"
    "1dis_2point5:training_runner/on_policy_single_vis_agent_runner.py \
    --log_dir $result_dir \
    --seed $seed \
    --distracted_human_shooting_range 1.0 \
    --distracted_human_separation_penalty_factor 2.5 \
    --load_config configs/room_361/happo_1r4p1dis_3c_rvs_room361.json \
    --exp_name train_on_vis_2point5_1m_ai_090_4p_3c_rvs_room361\
    --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361 \
    --cuda_device cuda:1"
    # "2ai_4p_3crvs_room361:training_runner/on_policy_single_vis_agent_runner.py \
    # --load_config configs/room_361/happo_4p_3c_rvs_room361.json \
    # --exp_name train_on_vis_ablation_ai_090_4p_3c_rvs_room361\
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361 \
    # --cuda_device cuda:1"
    # "3ai_4p_6crvs_room361:training_runner/on_policy_single_agent_runner.py \
    # --load_config configs/room_361/happo_4p_3c_rvs_room361.json \
    # --exp_name train_on_ai_095_4p_3c_rvs_room361\
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.95_happo_5p_3c_rvs_room361 \
    # --cuda_device cuda:2"
    # "4ai_4p_6crvs_room361:training_runner/on_policy_single_agent_runner.py \
    # --load_config configs/room_361/happo_4p_6c_rvs_room361.json \
    # --exp_name train_on_ai_085_4p_6c_rvs_room361\
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.85_happo_5p_6c_rvs_room361 \
    # --cuda_device cuda:0"
    # "5ai_4p_6crvs_room361:training_runner/on_policy_single_agent_runner.py \
    # --load_config configs/room_361/happo_4p_6c_rvs_room361.json \
    # --exp_name train_on_ai_095_4p_6c_rvs_room361\
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.95_happo_5p_6c_rvs_room361 \
    # --cuda_device cuda:1"
    # "sfm_4p1d_room361:training_runner/on_policy_single_agent_sfm_runner.py \
    #     --log_dir $result_dir \
    #     --seed $seed \
    #     --algo robot_crowd_happo \
    #     --env crowd_env_vis_ablation_1 \
    #     --exp_name dis_1p \
    #     --load_config configs/room_361/happo_1r4p1dis_sp_rvs_room361.json \
    #     --distracted_human_num 1 \
    #     --cuda_device cuda:0"
    # "sfm_4p2d_room361:training_runner/on_policy_single_agent_sfm_runner.py \
    #     --log_dir $result_dir \
    #     --seed $seed \
    #     --algo robot_crowd_happo \
    #     --env crowd_env_vis_ablation_1 \
    #     --exp_name dis_2p \
    #     --load_config configs/room_361/happo_1r4p1dis_sp_rvs_room361.json \
    #     --distracted_human_num 2 \
    #     --cuda_device cuda:1"
    # "sfm_4p3d_room361:training_runner/on_policy_single_agent_sfm_runner.py \
    #     --log_dir $result_dir \
    #     --seed $seed \
    #     --algo robot_crowd_happo \
    #     --env crowd_env_vis_ablation_1 \
    #     --exp_name dis_3p \
    #     --load_config configs/room_361/happo_1r4p1dis_sp_rvs_room361.json \
    #     --distracted_human_num 3 \
    #     --cuda_device cuda:2"
    # "ai_4p_sp_rvs_room361:training_runner/on_policy_single_agent_runner.py \
    # --exp_name train_on_ai_4p_sp_rvs_room361\
    # --load_config configs/room_361/happo_4p_sp_rvs_room361.json \
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
    # --cuda_device cuda:2"

    # "ai_4p_6crvs_cc:training_runner/on_policy_single_agent_runner.py \
    # --exp_name train_on_ai_4p_6c_rvs_circlecross\
    # --load_config configs/circle_cross_4P/happo_4p_6c_rvs_circlecross.json \
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_6c_rvs_circlecross \
    # --cuda_device cuda:1"
    # "sfm_4p_cc:training_runner/on_policy_single_agent_sfm_runner.py \
    # --load_config configs/circle_cross_4P/happo_4p_sp_rvs_circlecross.json \
    # --cuda_device cuda:1"
    # "ai_4p_sp_rvs_cc:training_runner/on_policy_single_agent_runner.py \
    # --exp_name train_on_ai_4p_sp_rvs_circlecross\
    # --load_config configs/circle_cross_4P/happo_4p_sp_rvs_circlecross.json \
    # --human_model_dir /home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_circlecross \
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