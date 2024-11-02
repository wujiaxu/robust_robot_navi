from harl.common.data_recorder import DataRecorder


a = DataRecorder(save_dir="/home/dl/wu_ws/HARL/crowd_navi_bench/data_generation/crowd_env/crowd_navi/robot_crowd_happo/train_on_ai_090_4p_3c_rvs_room361_vs_c090_happo_5p_3c_rvs_room361_data/seed-00001-2024-09-30-16-39-46/logs")
a.load_data()
print(a.episode_buffer[2][0])