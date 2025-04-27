#!/bin/bash


#!/bin/bash

# Define the root directory
root_dir="/home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/results/crowd_env/crowd_navi/robot_crowd_happo"

# Define subpaths for each model directory
# model_subdirs=(
#     # "train_on_sfm_crowd/seed-00001-2024-09-28-20-00-11"
#     # "train_on_sfm_crowd_room361/seed-00001-2024-09-25-02-46-00" #room361
#     # "train_on_ai_4p_sp_rvs_room361/seed-00001-2024-09-28-20-00-00"
#     # "train_on_ai_090_4p_6c_rvs_room361/seed-00001-2024-09-28-19-59-06"
#     # "train_on_ai_085_4p_3c_rvs_room361/seed-00001-2024-09-29-17-59-51"
#     # "train_on_ai_090_4p_3c_rvs_room361/seed-00001-2024-09-29-17-59-46"
#     # "train_on_ai_095_4p_3c_rvs_room361/seed-00001-2024-09-29-17-59-56"
#     # "train_on_ai_085_4p_6c_rvs_room361/seed-00001-2024-09-29-18-00-02"
#     # "train_on_ai_095_4p_6c_rvs_room361/seed-00001-2024-09-29-18-00-07"
#     # "train_on_ai_4p_sp_rvs_circlecross/seed-00001-2024-09-28-20-00-16" #old
#     # "train_on_ai_090_4p_6c_rvs_circlecross/seed-00001-2024-09-28-20-00-06"
#     # "train_on_ai_090_4p_3c_rvs_circlecross/seed-00001-2024-10-10-10-49-34" #old
#     # "train_on_ai_090_4p_3c_rvs_circlecross"
#     # "train_on_ai_090_4p_6c_rvs_circlecross"
#     # "train_on_ai_4p_sp_rvs_circlecross"
#     # "train_on_sfm_crowd"
#     # "train_on_ai_090_4p_3c_rvs_room361"
#     # "train_on_ai_4p_sp_rvs_room361"
#     # "train_on_sfm_crowd_room361" 
#     "train_on_ai_4p_ccp_rvs_room361"
#     # "train_on_ai_4p_12c_rvs_room361"
# )

human_num=4
seed=1
exp_name="train_on_ai_4p_ccp_rvs_room361"
model_dir="$root_dir/$exp_name"

# Define a function to run the test
conda_env=url_navi
declare -a sessions_and_scripts=(

    "s1:test_runner/ad_hoc_crowd_model_test_runner.py \
        --exp_name "${exp_name}_vs_sfm_5_1point5_r361_5p" \
        --model_dir "$model_dir" \
        --scenario room_361 \
        --sfm_v0 5 \
        --sfm_sigma 1.5 \
        --cuda_device cuda:0 \
        --human_num 5"

    "s2:test_runner/ad_hoc_crowd_model_test_runner.py \
        --exp_name "${exp_name}_vs_sfm_10_03_r361_5" \
        --model_dir "$model_dir" \
        --scenario room_361 \
        --sfm_v0 10 \
        --sfm_sigma 0.3 \
        --cuda_device cuda:0 \
        --human_num 5"

    "s3:test_runner/ad_hoc_crowd_model_test_runner.py \
        --exp_name "${exp_name}_vs_sfm_10_03_cc_5p" \
        --model_dir "$model_dir" \
        --scenario circle_cross \
        --sfm_v0 10 \
        --sfm_sigma 0.3 \
        --cuda_device cuda:0 \
        --human_num 5"
    
    # python test_runner/ad_hoc_crowd_model_test_runner.py \
    #     --exp_name "${exp_name}_vs_sfm_5_1point5_cc_${human_num}p" \
    #     --model_dir "$model_dir" \
    #     --scenario circle_cross \
    #     --sfm_v0 5 \
    #     --sfm_sigma 1.5 \
    #     --human_num "$human_num"

    # "s4:test_runner/ai_crowd_model_test_runner.py\
    #     --seed $seed \
    #     --exp_name "${exp_name}_vs_c090_happo_5p_3c_rvs_circlecross" \
    #     --scenario circle_cross \
    #     --model_dir "$model_dir" \
    #     --human_num "$human_num" \
    #     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
    #     --cuda_device cuda:0"

    # "s5:test_runner/ai_crowd_model_test_runner.py\
    #     --seed $seed \
    #     --exp_name "${exp_name}_vs_happo_5p_sp_rvs_circlecross" \
    #     --scenario circle_cross \
    #     --model_dir "$model_dir" \
    #     --human_num "$human_num" \
    #     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_circlecross \
    #     --cuda_device cuda:2"

    # "s6:test_runner/ai_crowd_model_test_runner.py\
    #     --seed $seed \
    #     --exp_name "${exp_name}_vs_c090_happo_5p_3c_rvs_room361" \
    #     --scenario room_361 \
    #     --model_dir "$model_dir" \
    #     --human_num "$human_num" \
    #     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361 \
    #     --cuda_device cuda:2"

    # "s7:test_runner/ai_crowd_model_test_runner.py\
    #     --seed $seed \
    #     --exp_name "${exp_name}_vs_happo_5p_sp_rvs_room361" \
    #     --scenario room_361 \
    #     --model_dir "$model_dir" \
    #     --human_num "$human_num" \
    #     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
    #     --cuda_device cuda:2"

)

# # Loop through the model directories
# for model_subdir in "${model_subdirs[@]}"; do
#     model_dir="$root_dir/$model_subdir"
    
#     # Extract exp_name (the first part of model_subdir)
#     exp_name=$(echo "$model_subdir" | cut -d'/' -f1)
#     # exp_name="$model_subdir"
#     # Call the test function
#     run_test "$exp_name" "$model_dir" "$human_num"
# done


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
