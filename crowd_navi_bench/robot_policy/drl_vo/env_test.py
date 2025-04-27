import numpy as np
import gym
import gym.spaces
import json
import torch
# import rospy
import turtlebot_gym
import os
from harl.envs.robot_crowd_sim.crowd_env_wrapper import RobotCrowdSimWrapper
from harl.algorithms.actors import ALGO_REGISTRY
from harl.common.video import VideoRecorder
from stable_baselines3.common.env_util import make_vec_env

config_file = "/home/dl/wu_ws/robust_robot_navi/train_configs/happo_CNN_1D_5-5_3c_rvs_room361.json"

with open(config_file, encoding="utf-8") as file:
    all_config = json.load(file)
env_args = all_config["env_args"]
env_args["max_episode_length"] = 512
env_args["human_num"] = 4
env_args["robot_num"] = 1
algo_args = all_config["algo_args"]
human_model_dir = "/home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361/seed-00001-2024-09-27-10-55-57"
algo_args["train"]["model_dir"] = os.path.join(human_model_dir,"models")
algo_args["algo"]["human_preference_vector_dim"] = env_args["human_preference_vector_dim"]
# load ai model 
crowd_model = []
human_agent = ALGO_REGISTRY["robot_crowd_happo"](
                {**algo_args["model"], **algo_args["algo"]},
                gym.spaces.Box(-np.inf,np.inf,(726,)),
                gym.spaces.Box(-np.inf,np.inf,(2,)),
                device="cuda:0",
            )
human_policy_actor_state_dict = torch.load(
            str(algo_args["train"]["model_dir"])
            + "/actor_agent1"
            + ".pt"
        )
human_agent.actor.load_state_dict(human_policy_actor_state_dict)
for _ in range(env_args["human_num"]):
    crowd_model.append(human_agent)
env_core = RobotCrowdSimWrapper(crowd_model,env_args,phase="test",nenv=1,time_step=0.05)
env = gym.make('drl_vo_env-v0')
env.configure(env_core)

video_recorder = VideoRecorder("/home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/")
obs = env.reset(seed=5)
video_recorder.init(env, enabled=True)
for _ in range(512):
    obs,reward,done,info = env.step(np.array([-0.9,0.]))
    video_recorder.record(env)
    if done:break
video_recorder.save("env_test.mp4",save_pdf=False)


