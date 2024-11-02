from crowd_navi_bench.density_model.auto_encoder import VAE
from harl.envs.robot_crowd_sim.crowd_env_vis import RobotCrowdSim
import os
import json
import numpy as np
from harl.common.video import VideoRecorder
from harl.common.data_recorder import DataRecorder
import copy
import torch
# video_recorder = VideoRecorder(".")
# config_file="/home/dl/wu_ws/HARL/tuned_configs/happo_10p_3c_rvs_circlecross.json"
# with open(config_file, encoding="utf-8") as file:
#     all_config = json.load(file)
# algo_args = all_config["algo_args"]
# env_args = all_config["env_args"]
# env = RobotCrowdSim(env_args,"test")
import matplotlib.pyplot as plt
device = torch.device("cuda:0")
vae = VAE(device="cuda:0", learning_rate=1e-4,latent_dim=32)
vae.load_model("/home/dl/wu_ws/HARL/crowd_navi_bench/density_model/090_4p_6c_rvs_circlecross_vae_model_799.pth")
vae.eval()

from harl.algorithms.critics.twin_continuous_q_critic import TwinContinuousQCritic
from harl.algorithms.actors.recovery_ddpg import RecoveryDDPG
from pathlib import Path

model_dir="/home/dl/wu_ws/HARL/crowd_navi_bench/single_life_results/crowd_env/crowd_navi/robot_crowd_happo/train_recovery_policy/seed-00001-2024-10-12-22-45-19"
import gym.spaces
config_file = Path(model_dir)/"config.json"
with open(config_file, encoding="utf-8") as file:
        all_config = json.load(file)
        algo_args = all_config["algo_args"]
pr= RecoveryDDPG(
                {**algo_args["model"], **algo_args["algo"]},
                gym.spaces.Box(-np.inf,np.inf,(726,)),
                gym.spaces.Box(-1,1,(2,)),
                device=device,
            )
critic_ldm = TwinContinuousQCritic(
            {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
            gym.spaces.Box(-np.inf,np.inf,(726,)),
            [gym.spaces.Box(-1,1,(2,))],
            1,
            "EP",
            device=device,
        )
pr.restore(Path(model_dir)/"models",0)
critic_ldm.restore(Path(model_dir)/"models")
# return pr, critic_ldm
pr.turn_off_grad()
critic_ldm.turn_off_grad()

dataset = DataRecorder(save_dir="/home/dl/wu_ws/HARL/crowd_navi_bench/data_generation/crowd_env/crowd_navi/robot_crowd_happo/train_on_ai_090_4p_6c_rvs_circlecross_vs_c090_happo_5p_6c_rvs_circlecross_data/seed-00001-2024-10-09-12-27-03/logs")
dataset.load_data()
data_loader = dataset.get_data_generator(1)
with torch.no_grad():
    for data in data_loader:
        data_gg = copy.deepcopy(data)
        log_prob = vae.estimate_log_probability(data,100,plot_recon=False)
        
        print(-log_prob.item())
        G = critic_ldm.get_values(data[:,:726],data[:,-2:])
        print(G.item())
        input("next")



