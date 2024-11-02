import json
from pathlib import Path

import os

def process_result_file(file):
    with open(file, encoding="utf-8") as file:
        results = json.load(file)

    average_time = None
    success_rate = None
    collision_rate = None
    for key in results.keys():
        if "test_robot_navi_performance/average_navi_time" in key:
            average_time = results[key][-1][-1]
        if "test_robot_navi_performance/success_rate" in key:
            success_rate = results[key][-1][-1]
        if "test_robot_navi_performance/collision_rate" in key:
            collision_rate = results[key][-1][-1]

    return average_time,success_rate,collision_rate

if __name__=="__main__":
    root_dir = Path("/home/dl/wu_ws/HARL/crowd_navi_bench/results/crowd_env/crowd_navi/robot_crowd_happo")
    result_dirs = ["train_on_sfm_crowd_vs_sfm_5_1point5_r361_5p",
                   "train_on_sfm_crowd_vs_sfm_10_03_r361_5p",
                   "train_on_sfm_crowd_vs_sfm_10_03_cc_5p",
                   "train_on_ai_4p_sp_rvs_room361_vs_sfm_5_1point5_r361_5p",
                   "train_on_ai_4p_sp_rvs_room361_vs_sfm_10_03_r361_5p",
                   "train_on_ai_4p_sp_rvs_room361_vs_sfm_10_03_cc_5p",
                #    "train_on_ai_090_4p_6c_rvs_room361_vs_sfm_5_1point5_r361_5p",
                #    "train_on_ai_090_4p_6c_rvs_room361_vs_sfm_10_03_r361_5p",
                #    "train_on_ai_090_4p_6c_rvs_room361_vs_sfm_10_03_cc_5p",
                #    "train_on_ai_085_4p_3c_rvs_room361_vs_sfm_5_1point5_r361_5p",
                #    "train_on_ai_085_4p_3c_rvs_room361_vs_sfm_10_03_r361_5p",
                #    "train_on_ai_085_4p_3c_rvs_room361_vs_sfm_10_03_cc_5p",
                   "train_on_ai_090_4p_3c_rvs_room361_vs_sfm_5_1point5_r361_5p",
                   "train_on_ai_090_4p_3c_rvs_room361_vs_sfm_10_03_r361_5p",
                   "train_on_ai_090_4p_3c_rvs_room361_vs_sfm_10_03_cc_5p",
                #    "train_on_ai_095_4p_3c_rvs_room361_vs_sfm_5_1point5_r361_5p",
                #    "train_on_ai_095_4p_3c_rvs_room361_vs_sfm_10_03_r361_5p",
                #    "train_on_ai_095_4p_3c_rvs_room361_vs_sfm_10_03_cc_5p",
                #    "train_on_ai_085_4p_6c_rvs_room361_vs_sfm_5_1point5_r361_5p",
                #    "train_on_ai_085_4p_6c_rvs_room361_vs_sfm_10_03_r361_5p",
                #    "train_on_ai_085_4p_6c_rvs_room361_vs_sfm_10_03_cc_5p",
                #    "train_on_ai_095_4p_6c_rvs_room361_vs_sfm_5_1point5_r361_5p",
                #    "train_on_ai_095_4p_6c_rvs_room361_vs_sfm_10_03_r361_5p",
                #    "train_on_ai_095_4p_6c_rvs_room361_vs_sfm_10_03_cc_5p",
                   "train_on_ai_4p_sp_rvs_room361_vs_c090_happo_5p_3c_rvs_room361",
                   "train_on_ai_090_4p_3c_rvs_room361_vs_c090_happo_5p_3c_rvs_room361",
                   "train_on_ai_4p_sp_rvs_circlecross_vs_sfm_5_1point5_cc_5p",
                   "train_on_ai_4p_sp_rvs_circlecross_vs_sfm_10_03_cc_5p",
                   "train_on_ai_4p_sp_rvs_circlecross_vs_sfm_10_03_r361_5p",
                   "train_on_ai_090_4p_6c_rvs_circlecross_vs_sfm_5_1point5_cc_5p",
                   "train_on_ai_090_4p_6c_rvs_circlecross_vs_sfm_10_03_cc_5p",
                   "train_on_ai_090_4p_6c_rvs_circlecross_vs_sfm_10_03_r361_5p"
                   ]
    
    for result_dir in result_dirs:
        seed_dir = os.listdir(root_dir/Path(result_dir))[0]
        file = root_dir/Path(result_dir)/Path(seed_dir)/"logs/summary.json"
        average_time,success_rate,collision_rate = process_result_file(file)
        exp_name = result_dir.split("/")[0]
        print(exp_name.split("_vs_")[0],exp_name.split("_vs_")[1],average_time,success_rate,collision_rate)