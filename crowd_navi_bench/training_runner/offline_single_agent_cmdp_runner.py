from harl.algorithms.actors.recovery_ddpg import RecoveryDDPG
import os
import time
import torch
import numpy as np
import copy
import setproctitle
from harl.common.valuenorm import ValueNorm
from torch.distributions import Categorical
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config, get_task_name
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics import CRITIC_REGISTRY
from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
from harl.common.buffers.off_policy_buffer_fp import OffPolicyBufferFP
from crowd_navi_bench.density_model.auto_encoder import VAE
from harl.common.data_recorder import DataRecorder
from harl.algorithms.actors.robot_crowd_happo import RobotCrowdHAPPO
from harl.algorithms.critics.twin_continuous_q_critic import TwinContinuousQCritic
import gym.spaces
import torch.nn.functional as F
from pathlib import Path
class OfflineRecoveryPolicyRunner:

    def __init__(self, args, algo_args, env_args):

        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        if "policy_freq" in self.algo_args["algo"]:
            self.policy_freq = self.algo_args["algo"]["policy_freq"]
        else:
            self.policy_freq = 1

        self.state_type = env_args.get("state_type", "EP")

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"],self.args["cuda_device"])
        self.task_name = get_task_name(args["env"], env_args)

        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
        os.makedirs(os.path.join(self.save_dir,"actor"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir,"ldm"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir,"prox"), exist_ok=True)
        save_config(args, algo_args, env_args, self.run_dir)
        self.log_file = open(
            os.path.join(self.run_dir, "progress.txt"), "w", encoding="utf-8"
        )
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # init dataset
        # (s,a,logp,c,s')
        self.dataset = DataRecorder(
            save_dir="/home/dl/wu_ws/HARL/crowd_navi_bench/data_generation/crowd_env/crowd_navi/robot_crowd_happo/train_on_ai_090_4p_6c_rvs_circlecross_vs_c090_happo_5p_6c_rvs_circlecross_data/seed-00001-2024-10-09-12-27-03/logs")
        self.dataset.load_data()

        # init density model
        self.density_model = VAE(device="cuda:0", learning_rate=1e-4,latent_dim=32)
        self.density_model.load_model("/home/dl/wu_ws/HARL/crowd_navi_bench/density_model/090_4p_6c_rvs_circlecross_vae_model_7799.pth")
        self.density_model.eval()

        # init actor 
        # self.actor = RobotCrowdHAPPO(
        #             {**robot_algo_args["model"], **robot_algo_args["algo"]},
        #             gym.spaces.Box(-np.inf,np.inf,(726,)),
        #             gym.spaces.Box(-np.inf,np.inf,(2,)),
        #             device=self.device,
        #         )
        # robot_policy_actor_state_dict = torch.load(
        #     str(robot_algo_args["train"]["model_dir"])
        #     + "/actor_agent0"
        #     + ".pt"
        # )
        # self.actor.actor.load_state_dict(robot_policy_actor_state_dict)

        # init ldm critic and value normalizer
        self.critic_ldm = TwinContinuousQCritic(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                gym.spaces.Box(-np.inf,np.inf,(726,)),
                [gym.spaces.Box(-1,1,(2,))],
                1,
                self.state_type,
                device=self.device,
            )
        # self.ldm_value_normalizer = ValueNorm(1, device=self.device)
        

        # init collision critic and value normalizer
        # self.critic_prox = TwinContinuousQCritic(
        #         {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
        #         gym.spaces.Box(-np.inf,np.inf,(726,)),
        #         [gym.spaces.Box(-1,1,(2,))],
        #         1,
        #         self.state_type,
        #         device=self.device,
        #     )
        # self.prox_value_normalizer = ValueNorm(1, device=self.device)

        # balancer
        self.balance_factor = 1

        # init offpolicy buffer using dataset
        self.buffer = OffPolicyBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    gym.spaces.Box(-np.inf,np.inf,(726,)),
                    1,
                    [gym.spaces.Box(-np.inf,np.inf,(726,))],
                    [gym.spaces.Box(-1,1,(2,))],
                )
        
        self.agent_death = np.zeros(
            (1, 1, 1) #self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1
        )
        print("total episode",len(list(self.dataset.episode_buffer.keys())))
        for episode_id in self.dataset.episode_buffer.keys():
            episode_data = self.dataset.episode_buffer[episode_id]
            for i, frame in enumerate(episode_data[:-1]):
                # collision avoidance reward
                min_dist = np.min(frame['observation'][0][6:726])
                reward=np.expand_dims(
                    np.expand_dims(
                        min(min_dist-1.3,0),axis=0),axis=0)
                obs=np.expand_dims(
                    np.expand_dims(
                        frame['observation'][0][:726],axis=0),axis=0)
                action=np.expand_dims(np.expand_dims(
                        np.clip(frame['action'][0],np.array([-1,-1]),np.array([1,1])),axis=0),axis=0)
                # reward=frame["reward"]
                # print(reward)
                next_obs=np.expand_dims(
                    np.expand_dims(
                        episode_data[i+1]['observation'][0][:726],axis=0),axis=0)
                next_action=np.expand_dims(np.expand_dims(
                        np.clip(episode_data[i+1]['action'][0],np.array([-1,-1]),np.array([1,1])),axis=0),axis=0)
                done = np.array([[False]])
                term = np.array([[False]])
                dones_env = np.array([[False]]) 
                available_action = None
                next_available_action = None
                valid_transition = 1 - self.agent_death
                self.agent_death = np.expand_dims(done, axis=-1)
                data = (
                    obs[0],
                    obs,
                    action,
                    available_action,
                    reward[0],
                    dones_env,
                    valid_transition.transpose(1, 0, 2),
                    term,
                    next_obs[0],
                    next_obs,
                    next_action,
                    next_available_action,
                )
                self.buffer.insert_offline(data)
                """
                (
                    share_obs,  # (n_threads, n_agents, share_obs_dim)
                    obs,  # (n_agents, n_threads, obs_dim)
                    actions,  # (n_agents, n_threads, action_dim)
                    available_actions,  # None or (n_agents, n_threads, action_number)
                    rewards,  # (n_threads, n_agents, 1)
                    dones,  # (n_threads, n_agents)
                    infos,  # type: list, shape: (n_threads, n_agents)
                    next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
                    next_obs,  # (n_threads, n_agents, next_obs_dim)
                    next_available_actions,  # None or (n_agents, n_threads, next_action_number)
                ) = data
                ==>
                data = (
                    share_obs[:, 0],  # (n_threads, share_obs_dim)
                    obs,  # (n_agents, n_threads, obs_dim)
                    actions,  # (n_agents, n_threads, action_dim)
                    available_actions,  # None or (n_agents, n_threads, action_number)
                    rewards[:, 0],  # (n_threads, 1)
                    np.expand_dims(dones_env, axis=-1),  # (n_threads, 1)
                    valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
                    terms,  # (n_threads, 1)
                    next_share_obs[:, 0],  # (n_threads, next_share_obs_dim)
                    next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
                    next_available_actions,  # None or (n_agents, n_threads, next_action_number)
                )
                """
            # at final step
            frame = episode_data[-1]
            # print(frame["reward"])
            # input()
            if frame["reward"][0][0]<0:
                reward = np.array([[-1]])
            else:
                min_dist = np.min(frame['observation'][0][6:726])
                reward=np.expand_dims(
                    np.expand_dims(
                        min(min_dist-1.3,0),axis=0),axis=0)
            obs=np.expand_dims(
                np.expand_dims(
                    frame['observation'][0][:726],axis=0),axis=0)
            action=np.expand_dims(np.expand_dims(
                    np.clip(frame['action'][0],np.array([-1,-1]),np.array([1,1])),axis=0),axis=0)
            # reward=frame["reward"]
            next_obs=copy.deepcopy(obs)
            next_action=copy.deepcopy(action)
            done = np.array([[True]])
            term = np.array([[True]])
            dones_env = np.array([[True]]) 
            available_action = None
            next_available_action = None
            valid_transition = 1 - self.agent_death
            self.agent_death = np.expand_dims(done, axis=-1)
            data = (
                obs[0],
                obs,
                action,
                available_action,
                reward[0],
                dones_env,
                valid_transition.transpose(1, 0, 2),
                term,
                next_obs[0],
                next_obs,
                next_action,
                next_available_action,
            )
            #reset agent alive
            self.agent_death = np.zeros(
                (1, 1, 1) #self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1
            )
            
            self.buffer.insert_offline(data)

        # log buffer
        import matplotlib.pyplot as plt
        plt.hist(self.buffer.rewards[:self.buffer.idx,0],bins=100)
        plt.savefig(os.path.join(self.log_dir,"reward_hist.png"))
        plt.close()
            
        # total iteration
        self.total_it = 0  

        return 
    
    def run(self):

        steps = (
            self.algo_args["train"]["num_env_steps"]//1
        )
        update_num = int(  # update number per train
            self.algo_args["train"]["update_per_train"]
            * self.algo_args["train"]["train_interval"]
        )
        for step in range(1,steps+1):
            if step % self.algo_args["train"]["train_interval"]==0:
                loss_ldms = []
                loss_proxs = []
                loss_actors = []
                ldm_values,prox_values = [],[]
                for _ in range(update_num):
                    loss_ldm, loss_prox, loss_actor,ldm_value,prox_value = self.train()
                    # print(loss_ldm, loss_prox)
                    if loss_actor is not None:
                        loss_actors.append(loss_actor)
                    if ldm_value is not None:
                        ldm_values.append(ldm_value)
                    if prox_value is not None:
                        prox_values.append(prox_value)
                    loss_proxs.append(loss_prox)
                    loss_ldms.append(loss_ldm)
                # self.writter.add_scalar(
                #     "loss_ldm", np.average(loss_ldms), step
                # )
                # self.writter.add_scalar(
                #     "loss_prox", np.average(loss_proxs), step
                # )
                # self.writter.add_scalar(
                #     "loss_actors", np.average(loss_actors), step
                # )
                # self.writter.add_scalar(
                #     "loss_actors", np.average(ldm_values), step
                # )
                # self.writter.add_scalar(
                #     "loss_actors", np.average(prox_values), step
                # )
                print(step,"/",steps, ":",
                      np.average(loss_ldms))
            if step % self.algo_args["train"]["eval_interval"]==0:
                print("save model")
                self.save()
        return 
    
    @torch.no_grad()
    def get_log_prob(self,input_data,num_samples=10):

        input_data = torch.from_numpy(input_data).to(**dict(dtype=torch.float32, device=self.device))
        input_data = input_data.squeeze(0)
        mu, logvar = self.density_model.encode(input_data)
        
        # Initialize the log likelihood estimate
        log_prob_sum = 0.0
        
        # Perform Monte Carlo sampling
        for _ in range(num_samples):
            # Sample z from the Gaussian posterior q(z|x)
            z = self.density_model.reparameterize(mu, logvar)
            
            # Reconstruct the input from z
            recon_x = self.density_model.decoder(z)
            
            # Compute log p(x|z) (reconstruction log likelihood)
            # Assuming a Gaussian distribution for p(x|z) with mean recon_x and variance 1
            recon_log_prob = -F.mse_loss(recon_x, input_data, reduction='none').sum(dim=-1)
            # print(recon_log_prob.shape)
            # Add the reconstruction log probability to the sum
            log_prob_sum += recon_log_prob
        
        # Compute the average over all samples
        log_prob_estimate = log_prob_sum / num_samples
        # Compute the KL divergence D_KL(q(z|x) || p(z))
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        # print(log_prob_estimate - kl_divergence)
        # Estimate the log probability: log p(x) ≈ log p(x|z) - KL divergence
        log_prob = log_prob_estimate - kl_divergence
        return log_prob.cpu().numpy()
    
    def train(self):
        self.total_it += 1
        data = self.buffer.sample_offline()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_actions,
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data

        next_actions=torch.from_numpy(sp_next_actions)
        self.critic_ldm.turn_on_grad()
        

        # get ldm reward using density model:= -logP(s,a)
        ldm_reward = np.expand_dims(
            - self.get_log_prob(np.concatenate([sp_obs,sp_actions],axis=-1)),axis=-1)
        
        loss_ldm = self.critic_ldm.train_ldm(
                sp_share_obs,
                sp_actions,
                ldm_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )
        loss_prox = 0
        self.critic_ldm.turn_off_grad()
        
        actor_loss = None
        value_pred_ldm = None
        value_pred_prox = None
        # if self.total_it % self.policy_freq == 0:
        #     # with torch.no_grad():
        #     self.actor.turn_on_grad()
        #     action=self.actor.get_actions(
        #             sp_obs, False
        #         ).squeeze(0)
        #         # print(action.shape)
        #         # actions shape: (batch_size, dim)
        #     # print(sp_share_obs[0])
        #     # input()
        #     value_pred_ldm = self.critic_ldm.get_values(sp_share_obs, action)
        #     # value_pred_prox = self.critic_prox.get_values(sp_share_obs, action)
        #     actor_loss = torch.mean(value_pred_ldm)
        #     self.actor.actor_optimizer.zero_grad()
        #     actor_loss.backward()
        #     self.actor.actor_optimizer.step()
        #     self.actor.turn_off_grad()

        #     # soft update
        #     self.actor.soft_update()
        self.critic_ldm.soft_update()
        # self.critic_prox.soft_update()
        return (loss_ldm, 
                loss_prox,
                actor_loss.item() if actor_loss is not None else actor_loss,
                torch.mean(value_pred_ldm).item() if value_pred_ldm is not None else value_pred_ldm,
                torch.mean(value_pred_prox).item() if value_pred_prox is not None else value_pred_prox
                )
    
    def close(self):
        self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
        self.writter.close()
    
    def save(self):
        """Save the model"""
        
        # self.actor.save(Path(self.save_dir)/"actor", 0)
        self.critic_ldm.save(Path(self.save_dir) / "ldm")
        # self.critic_prox.save(Path(self.save_dir)/"prox")
        

if __name__ == "__main__":
    """Train an algorithm."""
    import argparse
    import json
    import os
    from harl.utils.configs_tools import get_defaults_yaml_args, update_args,find_seed_directories
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="robot_crowd_happo",
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="crowd_env",
        choices=[
            "crowd_env",
            "crowd_env_vis",
            "crowd_env_vis_ablation_1",
            "crowd_env_vis_ablation_2",
        ],
        help="Environment name. Choose from: crowd_sim",
    )
    parser.add_argument(
        "--exp_name", type=str, default="ldm_model_train_on_ai_090_4p_6c_rvs_circlecross", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="configs/single_life/config.json",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    # parser.add_argument(
    #     "--robot_model_dir",
    #     type=str,
    #     default="/home/dl/wu_ws/HARL/crowd_navi_bench/results/crowd_env/crowd_navi/robot_crowd_happo/train_on_ai_090_4p_6c_rvs_circlecross",
    #     help="If set, load existing experiment config file instead of reading from yaml config file.",
    # )

    parser.add_argument("--cuda_device",type=str,default="cuda:0")
    args, unparsed_args = parser.parse_known_args()
    # robot_model_dir = find_seed_directories(args.robot_model_dir,args.seed)[0]
    # robot_model_config_file = os.path.join(robot_model_dir,"config.json")
    def process(arg):
        try:
            return eval(arg)
        except:
            return arg
        
    # with open(robot_model_config_file, encoding="utf-8") as file:
    #     human_model_all_config = json.load(file)
    # robot_model_algo_args = human_model_all_config["algo_args"]
    # robot_model_algo_args["train"]["model_dir"] = os.path.join(robot_model_dir,"models")

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        # args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])

        # load config from existing config file
    
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line
    runner = OfflineRecoveryPolicyRunner(args, algo_args, env_args)
    runner.run()
    runner.close()