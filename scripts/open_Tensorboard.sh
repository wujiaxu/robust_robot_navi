#!/bin/bash

# Set the root directory for logs
ROOT_DIR="/home/dl/wu_ws/HARL/results/crowd_env/crowd_navi/robot_crowd_happo"

# Define the log directories and custom experiment names (name:dir)
LOG_DIRS=(
    "happo_5p_3c:/ped_sim/seed-00001-2024-09-20-04-03-41/log"
    "happo_5p_6c:/ped_sim/seed-00001-2024-09-20-10-42-59/log"
    "happo_5p_rvs_sp:/ped_sim/seed-00001-2024-09-21-00-52-59/log"
    "happo_10p_3c:/ped_sim/seed-00001-2024-09-20-04-03-46/log"
    "happo_10p_3c001:/ped_sim/seed-00001-2024-09-20-10-43-04/log"
    "happo_10p_rvs_sp:/ped_sim/seed-00001-2024-09-21-00-53-04/log"
    "happo_5p_6crvs:seed-00001-2024-09-22-01-50-11" #high aux_reward result a lot of timeout
    "happo_5p_6crvs_room361:seed-00001-2024-09-22-01-50-16" #same with above
    "chappo_5p_6c:c_ped_sim/seed-00001-2024-09-24-02-12-48"
    "happo_5p_sp_room361:/ped_sim_base/seed-00001-2024-09-24-02-12-53"
    "happo_5p_sp:/ped_sim_base/seed-00001-2024-09-24-03-21-00"

    # constrained
    
)

# Construct the TensorBoard logdir_spec argument
LOGDIR_ARG=""
for entry in "${LOG_DIRS[@]}"; do
    # Split the entry into experiment name and directory
    IFS=":" read -r name dir <<< "$entry"
    LOGDIR_ARG+="$ROOT_DIR$dir,"
done

# Run TensorBoard with the constructed logdir_spec argument
tensorboard --logdir_spec=$LOGDIR_ARG --host localhost --port 6006