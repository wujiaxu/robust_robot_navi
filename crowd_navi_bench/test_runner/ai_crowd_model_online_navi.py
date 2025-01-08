"""Base runner for on-policy algorithms."""

from harl.envs.robot_crowd_sim.utils.info import ReachGoal
import numpy as np
import torch
import setproctitle
from harl.envs.robot_crowd_sim.crowd_env import RobotCrowdSim
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
from pathlib import Path
from harl.common.video import VideoRecorder
# from crowd_navi_bench.density_model.auto_encoder import VAE
from harl.algorithms.critics.twin_continuous_q_critic import TwinContinuousQCritic
# from harl.algorithms.actors.recovery_ddpg import RecoveryDDPG
import matplotlib.pyplot as plt
import pickle
from crowd_navi_bench.crowd_policy.socialforce import SocialForce

class OnPolicyTestRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args,
                 human_model_algo_args):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.logc = args["logc"]
        algo_args["eval"]["n_eval_rollout_threads"] = 1
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.env_args["scenario"] = args["scenario"]
        self.env_args["human_num"] = args["human_num"]
        self.env_args["robot_num"] = 1
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

        self.load_recovery(self.device)
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
        self.envs = RobotCrowdSim(self.env_args,phase="test",nenv=1)
        self.rec_policy = SocialForce(self.envs,self.args["sfm_v0"],self.args["sfm_sigma"],self.args["sfm_initial_speed"])
        
        self.num_agents = get_num_agents(args["env"], self.env_args, self.envs)

        # actor 
        algo_args["algo"]["use_discriminator"] = False #since we test robot
        self.actor = []
        self.actor.append(ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            ))
        policy_actor_state_dict = torch.load(
            str(self.algo_args["train"]["model_dir"])
            + "/actor_agent0"
            + ".pt"
        )
        self.actor[0].actor.load_state_dict(policy_actor_state_dict)
        # crowd (ai simulator) #TODO
        human_agent = ALGO_REGISTRY[args["algo"]](
                {**human_model_algo_args["model"], **human_model_algo_args["algo"]},
                self.envs.observation_space[1],
                self.envs.action_space[1],
                device=self.device,
            )
        human_policy_actor_state_dict = torch.load(
            str(human_model_algo_args["train"]["model_dir"])
            + "/actor_agent0"
            + ".pt"
        )
        human_agent.actor.load_state_dict(human_policy_actor_state_dict)
        for _ in range(self.env_args["human_num"]):
            self.actor.append(human_agent)


        self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )
        # if self.algo_args["train"]["model_dir"] is not None:  # restore model
        #     self.restore()

        # init video recorder for debug
        self.video_recorder = VideoRecorder(self.log_dir)

        # self.vae = VAE(device=self.args["cuda_device"], learning_rate=1e-4,latent_dim=32)
        # self.vae.load_model("/home/dl/wu_ws/HARL/crowd_navi_bench/density_model/090_4p_6c_rvs_circlecross_vae_model_7799.pth")
        # self.vae.eval()

    def load_recovery(self,device,
                  model_dir="/home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/single_life_results/crowd_env/crowd_navi/robot_crowd_happo/ldm_model_train_on_ai_090_4p_6c_rvs_circlecross/seed-00001-2024-10-16-23-13-37"):
        import gym.spaces
        config_file = Path(model_dir)/"config.json"
        with open(config_file, encoding="utf-8") as file:
                all_config = json.load(file)
                algo_args = all_config["algo_args"]
        # self.pr= RecoveryDDPG(
        #                 {**algo_args["model"], **algo_args["algo"]},
        #                 gym.spaces.Box(-np.inf,np.inf,(726,)),
        #                 gym.spaces.Box(-1,1,(2,)),
        #                 device=device,
        #             )
        self.critic_ldm = TwinContinuousQCritic(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    gym.spaces.Box(-np.inf,np.inf,(726,)),
                    [gym.spaces.Box(-1,1,(2,))],
                    1,
                    "EP",
                    device=device,
                )
        # self.pr.restore(Path(model_dir)/"models"/"actor",0)
        self.critic_ldm.restore(Path(model_dir)/"models"/"ldm")
        # return pr, critic_ldm
        # self.pr.turn_off_grad()
        self.critic_ldm.turn_off_grad()

    def run(self):
        """Run the rendering pipeline."""
        self.render()

    @torch.no_grad()
    def render(self):
        """Render the model."""
        print("start rendering")
        record = {}
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for e in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                # self.video_recorder.init(self.envs, enabled=True)
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
                time_step = 0
                log_probs = []
                Gs = []
                G_recs = []
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
                    robot_action = np.clip(eval_actions_collector[0],
                                           np.array([-1,-1]),
                                           np.array([1,1]))
                    # log_prob = self.vae.estimate_log_probability(np.concatenate([eval_obs[:,0,:726],robot_action],axis=-1))
                    # log_probs.append(-log_prob.item())

                    # rev_act = self.pr.get_actions(eval_obs[:,0,:726],False)
                    G = self.critic_ldm.get_values(eval_obs[:,0,:726],robot_action)
                    # G_rec = self.critic_ldm.get_values(eval_obs[:,0,:726],rev_act)
                    rec = False
                    if G.item()>self.logc and self.logc!=200000:
                        # print("rec",time_step)
                        rec = True
                        v_pref = self.envs.agents[0].v_pref 
                        self.envs.agents[0].v_pref = 1.0
                        eval_actions_collector[0] = np.expand_dims(self.rec_policy.predict(0),axis=0)
                        self.envs.agents[0].v_pref = v_pref
                    elif self.logc==200000:
                        _r = np.random.rand()
                        if _r<0.5:
                            rec = True
                            v_pref = self.envs.agents[0].v_pref 
                            self.envs.agents[0].v_pref = 1.0
                            eval_actions_collector[0] = np.expand_dims(self.rec_policy.predict(0),axis=0)
                            self.envs.agents[0].v_pref = v_pref
                        else:
                            rec = False

                    Gs.append(G.item())
                    # G_recs.append(G_rec.item())
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)[0]
                    
                    (
                        eval_obs,
                        eval_share_obs,
                        eval_rewards,
                        eval_dones,
                        eval_infos,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions,rec)
                    eval_data = (
                    eval_obs,
                    eval_share_obs,
                    eval_rewards,
                    eval_dones,
                    [eval_infos],
                    eval_available_actions,
                    )
                    time_step+=1
                    self.logger.test_per_step(
                            eval_data
                        )  # logger callback at each step of evaluation
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    
                    # TODO save to video recoder
                    self.video_recorder.record(self.envs)
                    # self.envs.record_current_log_prob(0,np.exp(log_prob.item()/728))

                    if eval_dones[0]:
                        print(f"this episode {e} end at {time_step} total reward: {rewards}")
                        print(eval_infos[0]["episode_info"])
                        # SFM:92%
                        self.logger.test_log(e)
                        # self.logger.eval_log(
                        #     e
                        # )  # logger callback at the end of evaluation
                        # print(log_probs)
                        # print(Gs)
                        # print(G_recs)
                        if isinstance(eval_infos[0]["episode_info"],ReachGoal):
                            color="g"
                        else:
                            color="r"
                        record[e] = (log_probs,Gs,G_recs)
                        break
                
                # save video 
                self.video_recorder.save("test_episode_{}_{}.mp4".format(e,eval_infos[0]["episode_info"]),save_pdf=True)
                # input("next")
            with open(Path(self.log_dir)/"record.pkl","wb") as f:
                pickle.dump((record,self.logger.eval_info_moniter),f)
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            raise NotImplementedError

    def restore(self): #not used
        """Restore model parameters."""
        policy_actor_state_dict = torch.load(
            str(self.algo_args["train"]["model_dir"])
            + "/actor_agent0"
            + ".pt"
        )
        self.actor[0].actor.load_state_dict(policy_actor_state_dict)

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
    from harl.utils.configs_tools import get_defaults_yaml_args, update_args
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
        "--human_policy", type=str, default="ai", help="human policy."
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
        "--logc", type=float, default=80, help="Experiment iteration."
    )
    parser.add_argument(
        "--sfm_v0", type=float, default=5, help="Experiment iteration."
    )
    parser.add_argument(
        "--sfm_sigma", type=float, default=0.8, help="Experiment iteration."
    )
    parser.add_argument(
        "--sfm_initial_speed", type=float, default=0.8, help="Experiment iteration."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="room_256",
        choices=[
            "room_361",
            "room_256",
            "circle_cross",
            "corridor",
        ],
        help="scenario name",
    )
    parser.add_argument(
        "--exp_name", type=str, 
        # default="train_on_ai_090_4p_6c_rvs_circlecross_vs_c090_happo_5p_6c_rvs_circlecross_online", 
        default="train_on_ai_090_4p_6c_rvs_circlecross_vs_happo_5p_sp_rvs_circlecross_online", 
        help="Experiment name."
    )
    parser.add_argument(
        "--test_episode", type=int, default=3, help="Experiment iteration."
    )
    parser.add_argument(
        "--human_num", type=int, default=4, help="Experiment iteration."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/results/crowd_env/crowd_navi/robot_crowd_happo/train_on_ai_090_4p_3c_rvs_circlecross/seed-00001-2024-10-10-10-49-34",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--human_model_dir",
        type=str,
        default="/home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross/seed-00001-2024-09-27-10-55-42",
        # default="/home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_6c_rvs_circlecross/seed-00001-2024-09-27-14-39-00",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    # parser.add_argument( #TODO
    #     "--crowd_preference", type=int, default=0, help="should be list"
    # )
    parser.add_argument("--cuda_device",type=str,default="cuda:2")
    args, unparsed_args = parser.parse_known_args()
    test_episode = args.test_episode
    print(os.getcwd())
    config_file = os.path.join(args.model_dir,"config.json")
    human_model_config_file = os.path.join(args.human_model_dir,"config.json")
    def process(arg):
        try:
            return eval(arg)
        except:
            return arg
    
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    human_model_args = args = vars(args)  # convert to dict

    # load config from existing config file
    with open(config_file, encoding="utf-8") as file:
        all_config = json.load(file)
    args["algo"] = all_config["main_args"]["algo"]
    args["env"] = all_config["main_args"]["env"]
    algo_args = all_config["algo_args"]
    # env_args = all_config["env_args"]

    # load config from existing config file
    with open(human_model_config_file, encoding="utf-8") as file:
        human_model_all_config = json.load(file)
    human_model_args["algo"] = human_model_all_config["main_args"]["algo"]
    human_model_args["env"] = human_model_all_config["main_args"]["env"]
    human_model_algo_args = human_model_all_config["algo_args"]
    env_args = human_model_all_config["env_args"]
    
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line #TODO check it

    algo_args["train"]["model_dir"] = os.path.join(args["model_dir"],"models")
    algo_args["render"]["render_episodes"] = test_episode
    human_model_algo_args["train"]["model_dir"] = os.path.join(args["human_model_dir"],"models")
    runner = OnPolicyTestRunner(args, algo_args, env_args,human_model_algo_args)
    runner.run()
    runner.close()
    
