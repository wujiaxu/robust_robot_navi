"""Base runner for on-policy algorithms."""

import time

import numpy as np
import torch
import setproctitle

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


class OnPolicyTestRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        algo_args["eval"]["n_eval_rollout_threads"] = 1
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.env_args["scenario"] = args["scenario"]
        self.env_args["human_num"] = args["human_num"]
        self.env_args["robot_num"] = 1
        self.env_args["human_policy"] = self.args["human_policy"]
        self.env_args["max_episode_length"] = 100
        # self.crowd_preference = args["crowd_preference"] if args["crowd_preference"]>=0 else None

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        # self.action_aggregation = algo_args["algo"]["action_aggregation"]
        # self.state_type = env_args.get("state_type", "EP")
        # self.share_param = algo_args["algo"]["share_param"]
        # self.fixed_order = algo_args["algo"]["fixed_order"]
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        
        # create test dir
        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
            args["env"],
            env_args,
            args["algo"],
            args["exp_name"],
            algo_args["seed"]["seed"],
            logger_path=algo_args["logger"]["log_dir"],
        )
        save_config(args, algo_args, self.env_args, self.run_dir)
        # set the title of the process
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # set the config of env
        self.manual_expand_dims = True
        self.env_num = 1
        if args["env"] == "crowd_env":
            from harl.envs.robot_crowd_sim.crowd_env import RobotCrowdSim
            self.envs = RobotCrowdSim(self.env_args,phase="test",nenv=1)
        elif args["env"] == "crowd_env_vis":
            from harl.envs.robot_crowd_sim.crowd_env_vis import RobotCrowdSimVis
            self.envs = RobotCrowdSimVis(self.env_args,phase="test",nenv=1)
        elif args["env"] == "crowd_env_vis_ablation_1":
            from harl.envs.robot_crowd_sim.crowd_env_vis_ablation_1 import RobotCrowdSimVis
            self.envs = RobotCrowdSimVis(self.env_args,phase="test",nenv=1)
        elif args["env"] == "crowd_env_vis_ablation_2":
            from harl.envs.robot_crowd_sim.crowd_env_vis_ablation_2 import RobotCrowdSimVis
            self.envs = RobotCrowdSimVis(self.env_args,phase="test",nenv=1)
        else: raise NotImplementedError

        self.num_agents = get_num_agents(args["env"], self.env_args, self.envs)

        # actor 
        algo_args["algo"]["use_discriminator"] = False #since we test robot
        self.actor = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
        # crowd (ad hoc simulator)
        if self.args["human_policy"] == "SFM":
            from crowd_navi_bench.crowd_policy.socialforce import SocialForce
            self.human_policy = SocialForce(self.envs,self.args["sfm_v0"],self.args["sfm_sigma"])
        elif self.args["human_policy"] == "ORCA":
            from crowd_navi_bench.crowd_policy.orca import ORCA
            self.human_policy = ORCA(self.envs)
        else:
            raise NotImplementedError

        self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, self.env_args, self.num_agents, self.writter, self.run_dir
            )
        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()

        # init video recorder for debug
        self.video_recorder = VideoRecorder(self.log_dir)

    def run(self):
        """Run the rendering pipeline."""
        self.render()

    @torch.no_grad()
    def render(self):
        """Render the model."""
        print("start rendering")
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for e in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                self.video_recorder.init(self.envs, enabled=True)

                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, 1), dtype=np.float32
                )
                rewards = 0
                time_step = 0
                while True:
                    eval_actions_collector = []
                    eval_actions, temp_rnn_state = self.actor.act(
                        eval_obs[:,0],
                        eval_rnn_states,
                        eval_masks,
                        None,
                        deterministic=True,
                    )
                    eval_rnn_states = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions.squeeze(0)))
                    for agent_id in range(1,self.num_agents):
                        action = self.human_policy.predict(agent_id)
                        eval_actions_collector.append(action)
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
                    time_step+=1
                    # TODO change to use my info analyzer
                    self.logger.test_per_step(
                        eval_data
                    )  # logger callback at each step of evaluation
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    
                    # TODO save to video recoder
                    self.video_recorder.record(self.envs)

                    if eval_dones[0]:
                        print(f"episode {e} end at {time_step},total reward: {rewards}")
                        print(eval_infos[0]["episode_info"])
                        # SFM:92%
                        self.logger.test_log(e)
                        # self.logger.eval_log(
                        #     e
                        # )  # logger callback at the end of evaluation
                        
                        break
                
                # save video 
                self.video_recorder.save("test_episode_{}_{}.mp4".format(e,eval_infos[0]["episode_info"]),save_pdf=True)
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            raise NotImplementedError

    def restore(self):
        """Restore model parameters."""
        policy_actor_state_dict = torch.load(
            str(self.algo_args["train"]["model_dir"])
            + "/actor_agent0"
            + ".pt"
        )
        self.actor.actor.load_state_dict(policy_actor_state_dict)

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
        default="robot_crowd_happo",
        choices=[
            "robot_crowd_happo"
        ],
        help="Algorithm name. Choose from: robot_crowd_happo.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="model seed."
    )
    parser.add_argument(
        "--human_policy", type=str, default="SFM", help="human policy."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="crowd_env",
        choices=[
            "crowd_env",
            "crowd_env_vis",
            "crowd_env_vis_ablation_1",
            "crowd_env_vis_ablation_2"
        ],
        help="Environment name. Choose from: crowd_sim",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="circle_cross",
        choices=[
            "circle_cross",
            "corridor",
            "room_361"
        ],
        help="scenario name",
    )
    parser.add_argument(
        "--exp_name", type=str, default="sfm_trained_vs_SFM_crowd", help="Experiment name."
    )
    parser.add_argument(
        "--test_episode", type=int, default=500, help="Experiment iteration."
    )
    parser.add_argument(
        "--sfm_v0", type=float, default=5, help="Experiment iteration."
    )
    parser.add_argument(
        "--sfm_sigma", type=float, default=1.5, help="Experiment iteration."
    )
    parser.add_argument(
        "--human_num", type=int, default=5, help="Experiment iteration."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/dl/wu_ws/HARL/crowd_navi_bench/results/crowd_env/crowd_navi/robot_crowd_happo/train_on_sfm_crowd/seed-00001-2024-09-28-20-00-11",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    # parser.add_argument(
    #     "--crowd_preference", type=int, default=0, help="hahaha muda muda this test model can't test different ai human"
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
    # args["env"] = all_config["main_args"]["env"]
    algo_args = all_config["algo_args"]
    env_args = all_config["env_args"]
    
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line
    algo_args["train"]["model_dir"] = os.path.join(model_dir,"models")
    algo_args["render"]["render_episodes"] = test_episode
    runner = OnPolicyTestRunner(args, algo_args, env_args)
    runner.run()
    runner.close()
    
