from harl.envs.robot_crowd_sim.crowd_env_vis import RobotCrowdSimVis
import os
import json
import numpy as np
from harl.common.video import VideoRecorder
video_recorder = VideoRecorder(".")
config_file="/home/dl/wu_ws/HARL/tuned_configs/happo_5p_3c_nvs_room361.json"
with open(config_file, encoding="utf-8") as file:
    all_config = json.load(file)
algo_args = all_config["algo_args"]
env_args = all_config["env_args"]
env_args["human_num"] = 2
env_args["robot_num"] = 1
env = RobotCrowdSimVis(env_args,"test",1,0,vis_scan=True)

env.reset()
video_recorder.init(env)
for i in range(50):
    actions = np.array([[1.0,0.1],[0.,0.5],[1,-0.2]])
    env.step(actions)
    video_recorder.record(env)

video_recorder.save("test.mp4")