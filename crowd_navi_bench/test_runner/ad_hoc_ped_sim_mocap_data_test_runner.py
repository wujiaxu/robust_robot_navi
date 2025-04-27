"""Base runner for on-policy algorithms."""

import time
import numpy as np
from pathlib import Path
import random
import torch
import setproctitle
from harl.envs.robot_crowd_sim.evaluation.data import RawData
from harl.envs.robot_crowd_sim.evaluation.evaluation import compute_min_NN_distance,compute_winding_angle,compute_ade,compute_fde
from harl.envs.robot_crowd_sim.utils.info import *
from harl.algorithms.actors import ALGO_REGISTRY
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config
from harl.envs import LOGGER_REGISTRY

from harl.common.video import VideoRecorder
from crowd_navi_bench.crowd_policy.orca import ORCA,PenType
from crowd_navi_bench.crowd_policy.socialforce import SocialForce

class OnPolicyTestRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.algo_args["eval"]["n_eval_rollout_threads"] = 1
        # self.env_args["scenario"] = args["scenario"]
        # self.env_args["human_num"] = args["human_num"]
        # self.env_args["robot_num"] = 0
        self.env_args["human_policy"] = self.args["human_policy"]
        self.env_args["max_episode_length"] = 100
        # self.crowd_preference = args["crowd_preference"] if args["crowd_preference"]>=0 else None

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.human_data_file = self.args["human_data_file"]
        if os.path.exists(self.human_data_file):
            print("load data")
            data = np.load(self.human_data_file, allow_pickle=True)
            init_positions, trajs, wds, min_dists = data
            self.human_data=trajs["{}p".format(self.env_args["human_num"])]#+trajs["3p"]
            
        else:
            raise RuntimeError
            self.human_data = None
        
        # create test dir
        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
            args["env"],
            env_args,
            args["algo"],
            args["exp_name"],
            algo_args["seed"]["seed"],
            logger_path=algo_args["logger"]["log_dir"],
        )
        save_config(args, algo_args, env_args, self.run_dir)
        # set the title of the process
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # set the config of env
        self.manual_expand_dims = True
        self.env_num = 1

        from harl.envs.robot_crowd_sim.crowd_env import RobotCrowdSim
        self.envs = RobotCrowdSim(env_args,phase="test",nenv=1)

        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        # actor 
        # self.actor = []
        # # crowd (ai simulator) #TODO
        # human_agent = ALGO_REGISTRY[args["algo"]](
        #         {**algo_args["model"], **algo_args["algo"]},
        #         self.envs.observation_space[1],
        #         self.envs.action_space[1],
        #         device=self.device,
        #     )
        # human_policy_actor_state_dict = torch.load(
        #     str(algo_args["train"]["model_dir"])
        #     + "/actor_agent1"
        #     + ".pt"
        # )
        # human_agent.actor.load_state_dict(human_policy_actor_state_dict)
        # for _ in range(self.env_args["human_num"]):
        #     self.actor.append(human_agent)
        if self.args["human_policy"] == "ORCA":
            self.human_policy = ORCA(self.envs)
        elif self.args["human_policy"] == "SFM":
            self.human_policy = SocialForce(self.envs,self.args["sfm_v0"],self.args["sfm_sigma"])
        else:
            raise NotImplementedError

        self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )
        # if self.algo_args["train"]["model_dir"] is not None:  # restore model
        #     self.restore()

        # init video recorder for debug
        self.video_recorder = VideoRecorder(self.log_dir)

    def run(self):
        """Run the rendering pipeline."""
        if self.human_data is not None:
            self.render_with_data()
        else:
            self.render()

    @torch.no_grad()
    def render_with_data(self):
        """Render the model."""
        print("start rendering")
        assert self.human_data is not None
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            self.generated_human_data = []
            self.evaluations = []
            num_collision = 0
            num_reachgoal = 0
            for e,scene in enumerate(self.human_data): #TODO multiple dataset
                agent_init=[]
                num_steps = []
                v_pref_radius = []
                generated_trajs = []
                ades = np.zeros((20,self.num_agents))
                fdes = np.zeros((20,self.num_agents))
                wds = []
                nn_dists= []
                p_maxs = []
                p_mins = []
                for traj in scene:
                    traj = np.array(traj)
                    p_max = np.max(traj[:,:2],axis=0)
                    p_min = np.min(traj[:,:2],axis=0)
                    p_maxs.append(p_max)
                    p_mins.append(p_min)
                offset = (np.max(np.array(p_maxs),axis=0)+np.min(np.array(p_min),axis=0))/2
                print(offset)
                for traj in scene:
                    traj = np.array(traj)
                    px,py = traj[0,0]-offset[0],traj[0,1]-offset[1]
                    velo = (traj[1:,:2]-traj[:-1,:2])/0.25
                    speed = np.linalg.norm(velo,axis=-1)
                    max_speed = np.max(speed)
                    print(max_speed)
                    size = 0.25
                    v_pref_radius.append((max_speed,size))
                    vx,vy = (traj[1,0]-traj[0,0])/0.25,(traj[1,1]-traj[0,1])/0.25
                    gx,gy = traj[-1,0]-offset[0],traj[-1,1]-offset[1]
                    theta = np.arctan2(gy-py,gx-px)
                    agent_init.append((px,py,gx,gy,vx,vy,theta))
                    num_steps.append(len(traj))
                
                for v in range(20):
                    np.random.seed(v)
                    h_types = []
                    preference = []
                    for i in range(self.num_agents):
                        h_types.append(random.choice(list(PenType))) 
                        preference.append(h_types[-1].value-1)
                    eval_obs, _, eval_available_actions = self.envs.reset(
                                                                    seed=e,
                                                                    preference=preference,
                                                                    random_attribute=v_pref_radius,
                                                                    agent_init=agent_init)
                    self.video_recorder.init(self.envs, enabled=True)

                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    
                    rewards = 0
                    step = 0
                    masks = []
                    while step<max(num_steps):
                        eval_actions_collector = []
                        for agent_id in range(self.num_agents):

                            if self.args["human_policy"] == "ORCA":
                                action = self.human_policy.predict(agent_id,h_types[agent_id])
                            elif self.args["human_policy"] == "SFM":
                                action = self.human_policy.predict(agent_id)
                            else:
                                raise NotImplementedError
                            
                            eval_actions_collector.append(np.array(action))
                        eval_actions = np.array(eval_actions_collector)
                        
                        (
                            eval_obs,
                            eval_share_obs,
                            eval_rewards,
                            eval_dones,
                            eval_infos,
                            eval_available_actions,
                        ) = self.envs.step(eval_actions)
                        eval_data = (
                        eval_obs,
                        eval_share_obs,
                        eval_rewards,
                        eval_dones,
                        [eval_infos],
                        eval_available_actions,
                        )
                        # TODO change to use my info analyzer
                        self.logger.test_per_step(
                            eval_data
                        )  # logger callback at each step of evaluation
                        rewards += eval_rewards[0][0]
                        eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                        masks.append(eval_dones)
                        # TODO save to video recoder
                        self.video_recorder.record(self.envs)
                        step +=1
                        for info in eval_infos:
                            if isinstance(info["episode_info"],Collision):
                                num_collision+=1
                            elif isinstance(info["episode_info"],ReachGoal):
                                num_reachgoal+=1
                        # if np.all(eval_dones):
                        #     print(f"total reward of this episode: {rewards}")
                        #     # SFM:92%
                        #     self.logger.test_log(e)
                        #     # self.logger.eval_log(
                        #     #     e
                        #     # )  # logger callback at the end of evaluation
                            
                        #     break
                    
                    self.logger.test_log(e) #TODO dont overwrite
                    masks = ~np.array(masks)
                    generated_trajs.append(([np.array(self.envs._render.traj_data[agent_id])+offset for agent_id in range(self.num_agents)],masks))
                    for agent_id in range(self.num_agents):
                        gt_traj = np.array(scene[agent_id])[:,:2]
                        pred_traj = np.array(self.envs._render.traj_data[agent_id])+offset
                        ade = compute_ade(pred_traj[1:gt_traj.shape[0]],gt_traj[1:],masks[:,agent_id][:gt_traj.shape[0]-1])
                        fde = compute_fde(pred_traj[1:gt_traj.shape[0]],gt_traj[1:])
                        ades[v,agent_id] = ade
                        fdes[v,agent_id] = fde
                        for other_id in range(self.num_agents):
                            if other_id == agent_id: continue
                            other_traj = np.array(self.envs._render.traj_data[other_id])+offset
                            wd = compute_winding_angle(pred_traj,other_traj)
                            nn_dist = compute_min_NN_distance(pred_traj[1:],other_traj[1:],masks[:,agent_id])
                            wds.append(wd)
                            nn_dists.append(nn_dist)
                    # save video 
                    self.video_recorder.save(
                        "test_episode_{}_{}_{}.mp4".format(
                        e,v,''.join(map(str, preference))),
                        save_pdf=True)
                k_ades = np.min(ades,axis=0)
                k_fdes = np.min(fdes,axis=0)
                # print((k_ades,k_fdes,wds,nn_dists))
                # input()
                self.evaluations.append((k_ades,k_fdes,wds,nn_dists,num_collision,num_reachgoal))
                self.generated_human_data.append(generated_trajs)

            data = np.array((self.generated_human_data,self.evaluations), dtype=object)
            np.save(Path(self.log_dir) / "generated_data.npy", data)
            print(f'Saved processed data to {Path(self.log_dir) / "generated_data.npy"}\n')
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            raise NotImplementedError

    @torch.no_grad()
    def render(self):
        """Render the model."""
        print("start rendering")
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for e in range(self.algo_args["render"]["render_episodes"]):

                for v in range(10):
                    np.random.seed(v)
                    preference = [np.random.randint(0, 
                                                    self.envs._human_preference_vector_dim) 
                                  for _ in range(len(self.envs.humans))]
                    eval_obs, _, eval_available_actions = self.envs.reset(seed=e,preference=preference)
                    self.video_recorder.init(self.envs, enabled=True)

                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_rnn_states = np.zeros(
                        (
                            self.env_num,
                            self.num_agents,
                            self.recurrent_n,
                            self.rnn_hidden_size,
                        ),
                        dtype=np.float32,
                    )
                    eval_masks = np.ones(
                        (self.env_num, self.num_agents,1), dtype=np.float32
                    )
                    rewards = 0
                    while True:
                        eval_actions_collector = []
                        for agent_id in range(self.num_agents):
                            eval_actions, temp_rnn_state = self.actor[agent_id].act(
                                eval_obs[:, agent_id],
                                eval_rnn_states[:, agent_id],
                                eval_masks[:, agent_id],
                                None,
                                deterministic=True,
                            )
                            eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                            eval_actions_collector.append(_t2n(eval_actions))
                        eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)[0]
                        
                        (
                            eval_obs,
                            eval_share_obs,
                            eval_rewards,
                            eval_dones,
                            eval_infos,
                            eval_available_actions,
                        ) = self.envs.step(eval_actions)
                        eval_data = (
                        eval_obs,
                        eval_share_obs,
                        eval_rewards,
                        eval_dones,
                        [eval_infos],
                        eval_available_actions,
                        )
                        # TODO change to use my info analyzer
                        self.logger.test_per_step(
                            eval_data
                        )  # logger callback at each step of evaluation
                        rewards += eval_rewards[0][0]
                        eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                        
                        # TODO save to video recoder
                        self.video_recorder.record(self.envs)
            
                        if np.all(eval_dones):
                            print(f"total reward of this episode: {rewards}")
                            # SFM:92%
                            self.logger.test_log(e)
                            # self.logger.eval_log(
                            #     e
                            # )  # logger callback at the end of evaluation
                            
                            break
                    
                    # save video 
                    self.video_recorder.save(
                        "test_episode_{}_{}_{}.mp4".format(
                        e,v,''.join(map(str, preference))),
                        save_pdf=True)
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            raise NotImplementedError

    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()


if __name__ == "__main__":
    """test an algorithm."""
    import argparse
    import json
    from harl.utils.configs_tools import get_defaults_yaml_args, update_args,find_seed_directories
    import os

    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "robot_crowd_happo"
        ],
        help="Algorithm name. Choose from: robot_crowd_happo.",
    )
    parser.add_argument(
        "--sfm_v0", type=float, default=5, help="Experiment iteration."
    )
    parser.add_argument(
        "--sfm_sigma", type=float, default=1.5, help="Experiment iteration."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="crowd_env",
        choices=[
            "crowd_env",
        ],
        help="Environment name. Choose from: crowd_sim",
    )
    parser.add_argument(
        "--human_policy", type=str, default="ORCA", help="human policy."
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="model seed."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="circle_cross",
        choices=[
            "circle_cross",
            "corridor",
            "ucy_students",
            "room_256",
        ],
        help="scenario name",
    )
    parser.add_argument(
        "--exp_name", type=str, default="ad_hoc_crowdsim_2p_rvs_pt_room256", help="Experiment name."
    )
    parser.add_argument(
        "--test_episode", type=int, default=10, help="Experiment iteration."
    )
    parser.add_argument(
        "--human_num", type=int, default=5, help="Experiment iteration."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_2p_3c_rvs_room256",
        # default="/home/dl/wu_ws/robust_robot_navi/room256_results_ver_1_seed_1/crowd_env_ccp/crowd_navi/robot_crowd_ppo/ppo_3p_ccp_rvs_room256",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--human_data_file",
        type=str,
        default="/home/dl/wu_ws/robust_robot_navi/harl/envs/robot_crowd_sim/data_origin/2p3p_dataset.npy",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    # parser.add_argument( #TODO
    #     "--crowd_preference", type=int, default=0, help="should be list"
    # )
    parser.add_argument("--cuda_device",type=str,default="cuda:2")
    args, unparsed_args = parser.parse_known_args()
    test_episode = args.test_episode
    print(os.getcwd())
    model_dir = find_seed_directories(args.model_dir,args.seed)[0]
    config_file = os.path.join(model_dir,"config.json")
    def process(arg):
        try:
            return eval(arg)
        except:
            return arg
    
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict

    # load config from existing config file
    with open(config_file, encoding="utf-8") as file:
        all_config = json.load(file)
    args["algo"] = all_config["main_args"]["algo"]
    args["env"] = all_config["main_args"]["env"]
    algo_args = all_config["algo_args"]
    env_args = all_config["env_args"]

    
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line #TODO check it

    algo_args["train"]["model_dir"] = os.path.join(model_dir,"models")
    algo_args["render"]["render_episodes"] = test_episode
    runner = OnPolicyTestRunner(args, algo_args, env_args)
    runner.run()
    runner.close()
    
