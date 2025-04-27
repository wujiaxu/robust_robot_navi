#!/bin/bash
conda_env=url_navi
declare -a sessions_and_scripts=(

# "ss_sp:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/sp_4p_room361_2 \
#     --sfm_v0 5 \
#     --sfm_sigma 1.5 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:0"
# "ss1:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/proposed_4p_room361_2 \
#     --sfm_v0 5 \
#     --sfm_sigma 1.5 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:1"
# "ss_ccp:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/ccp_4p_room361_2 \
#     --sfm_v0 5 \
#     --sfm_sigma 1.5 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:2"
"ss_sfm:drl_vo_test_adhoc.py \
    --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/SFMC_4p_room361 \
    --sfm_v0 5 \
    --sfm_sigma 1.5 \
    --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
    --algo robot_crowd_happo \
    --cuda_device cuda:0"

# "ss_sp3:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/sp_4p_room361_2 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:0"
# "ss3:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/proposed_4p_room361_2 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:1"
# "ss_ccp3:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/ccp_4p_room361_2 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:2"
"ss_sfm2:drl_vo_test_adhoc.py \
    --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/SFMC_4p_room361 \
    --sfm_v0 10 \
    --sfm_sigma 0.3 \
    --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
    --algo robot_crowd_happo \
    --cuda_device cuda:0"


# "ss_sp2:drl_vo_test.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/sp_4p_room361_2 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:0"
# "ss2:drl_vo_test.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/proposed_4p_room361_2 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:1"
# "ss_ccp2:drl_vo_test.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/ccp_4p_room361_2 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:2"
"ss_sfm4:drl_vo_test.py \
    --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/SFMC_4p_room361 \
    --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_circlecross \
    --algo robot_crowd_happo \
    --cuda_device cuda:2"
"ss_sfm6:drl_vo_test.py \
    --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/SFMC_4p_room361 \
    --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/happo_5p_sp_rvs_room361 \
    --algo robot_crowd_happo \
    --cuda_device cuda:2"

# "ss_sp2:drl_vo_test.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/sp_4p_room361_2 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:0"
# "ss2:drl_vo_test.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/proposed_4p_room361_2 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:1"
# "ss_ccp2:drl_vo_test.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/ccp_4p_room361_2 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:2"
"ss_sfm5:drl_vo_test.py \
    --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/SFMC_4p_room361 \
    --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
    --algo robot_crowd_happo \
    --cuda_device cuda:1"
"ss_sfm7:drl_vo_test.py \
    --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/SFMC_4p_room361 \
    --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361 \
    --algo robot_crowd_happo \
    --cuda_device cuda:1"

# "ss_sp2:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/sp_4p_room361_2 \
#     --sfm_v0 5 \
#     --sfm_sigma 1.5 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:0"
# "ss2:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/proposed_4p_room361_2 \
#     --sfm_v0 5 \
#     --sfm_sigma 1.5 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:1"
# "ss_ccp2:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/ccp_4p_room361_2 \
#     --sfm_v0 5 \
#     --sfm_sigma 1.5 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:2"

# "ss_sp2:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/sp_4p_room361_2 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:0"
# "ss2:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/proposed_4p_room361_2 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:1"
# "ss_ccp2:drl_vo_test_adhoc.py \
#     --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/ccp_4p_room361_2 \
#     --sfm_v0 10 \
#     --sfm_sigma 0.3 \
#     --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
#     --algo robot_crowd_happo \
#     --cuda_device cuda:2"
"sfm3:drl_vo_test_adhoc.py \
    --model_dir /home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/SFMC_4p_room361 \
    --sfm_v0 10 \
    --sfm_sigma 0.3 \
    --human_model_dir /home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross \
    --algo robot_crowd_happo \
    --cuda_device cuda:0"

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