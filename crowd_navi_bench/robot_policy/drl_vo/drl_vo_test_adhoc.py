from stable_baselines3 import PPO
from crowd_navi_bench.robot_policy.drl_vo.custom_cnn_full import *
import json
from harl.envs.robot_crowd_sim.crowd_env_wrapper import RobotCrowdSimWrapper
from harl.algorithms.actors import ALGO_REGISTRY
from harl.common.video import VideoRecorder
import turtlebot_gym
from harl.envs.robot_crowd_sim.utils.info import *
# for reproducibility, we seed the rng
#       
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

if __name__ == "__main__":
    import argparse
    import json
    from harl.utils.configs_tools import get_defaults_yaml_args, update_args,find_seed_directories
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )   
    parser.add_argument(
        "--human_policy", type=str, default="SFM", help="human policy."
    )
    parser.add_argument(
        "--test_episode", type=int, default=500, help="Experiment iteration."
    )
    parser.add_argument(
        "--exp_name", type=str, default="", help="Experiment name." #proposed_4p_room361_5_vs_3c_rvs_room361
    )
    parser.add_argument(
        "--sfm_v0", type=float, default=5, help="Experiment iteration."
    )
    parser.add_argument(
        "--sfm_sigma", type=float, default=1.5, help="Experiment iteration."
    )
    parser.add_argument(
        "--human_model_dir",
        type=str,
        default="/home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
            "--algo",
            type=str,
            default="robot_crowd_happo",
            help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
        )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model/proposed_4p_room361_2",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument("--cuda_device",type=str,default="cuda:0")
    args, unparsed_args = parser.parse_known_args()
    
    def process(arg):
            try:
                return eval(arg)
            except:
                return arg
            

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    human_model_args = args = vars(args)  # convert to dict]

    cuda_device = args["cuda_device"]#proposed:"cuda:1"
    robot_model_dir = args["model_dir"]
    human_model_dir = find_seed_directories(
                            args["human_model_dir"],
                            seed=1)[0] 
    exp_name = robot_model_dir.split("/")[-1]+"_vs_"+args["human_policy"]+"_{}_{}".format(args["sfm_v0"],args["sfm_sigma"])+"_"+args["human_model_dir"].split("_")[-1]+"_"+args["exp_name"] #"proposed_4p_room361_5"
    print(exp_name)
    # load model
    human_model_config_file = os.path.join(human_model_dir,"config.json")
    with open(human_model_config_file, encoding="utf-8") as file:
        human_model_all_config = json.load(file)
    human_model_args["algo"] = human_model_all_config["main_args"]["algo"]
    human_model_args["env"] = human_model_all_config["main_args"]["env"]
    human_model_algo_args = human_model_all_config["algo_args"]
    human_model_env_args = human_model_all_config["env_args"]
    human_model_algo_args["train"]["model_dir"] = human_model_dir
    
    #config_file = "/home/dl/wu_ws/robust_robot_navi/train_configs/happo_CNN_1D_5-5_3c_rvs_room361.json"
    # with open(config_file, encoding="utf-8") as file:
    #     all_config = json.load(file)
    # env_args = all_config["env_args"]

    human_model_env_args["max_episode_length"] = 240
    human_model_env_args["human_num"] = 5
    human_model_env_args["robot_num"] = 1
    human_model_env_args["human_policy"] = args["human_policy"]
    human_model_env_args["sfm_v0"] = args["sfm_v0"]
    human_model_env_args["sfm_sigma"] = args["sfm_sigma"]
    
    crowd_model = None
    # human_model_algo_args["train"]["model_dir"] = os.path.join(human_model_dir,"models")
    # human_model_algo_args["algo"]["human_preference_vector_dim"] = human_model_env_args["human_preference_vector_dim"]
    # # load ai model 
    # human_agent = ALGO_REGISTRY[args["algo"]](
    #                 {**human_model_algo_args["model"], **human_model_algo_args["algo"]},
    #                 gym.spaces.Box(-np.inf,np.inf,(729,)),
    #                 gym.spaces.Box(-np.inf,np.inf,(2,)),
    #                 device=cuda_device,
    #             )
    # human_policy_actor_state_dict = torch.load(
    #             str(human_model_algo_args["train"]["model_dir"])
    #             + "/actor_agent0"
    #             + ".pt"
    #         )
    # human_agent.actor.load_state_dict(human_policy_actor_state_dict)
    # for _ in range(human_model_env_args["human_num"]):
    #     crowd_model.append(human_agent)


    log_dir = os.path.join("/home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/test_log",exp_name)
    os.makedirs(log_dir, exist_ok=True)
    video_recorder = VideoRecorder(log_dir,fps=100)
    
    env_core = RobotCrowdSimWrapper(crowd_model,human_model_env_args,phase="test",time_step=0.05)
    env = gym.make('drl_vo_env-v0')
    env.configure(env_core)

    model = PPO.load(os.path.join(robot_model_dir,"best_model.zip"),device=cuda_device)
    
    
    sr = []
    cl = []
    to = []
    nt = []
    
    for e in range(args["test_episode"]):
        obs = env.reset()
        video_recorder.init(env, enabled=True)
        while True:
            action,_ = model.predict(obs)
            # print(action)
            obs,reward,done,info = env.step(action)
            video_recorder.record(env)
            if done:
                break
        nt.append(env.env.env.global_time)
        if isinstance(info["episode_info"],ReachGoal):
            sr.append(e)
        elif isinstance(info["episode_info"],Collision):
            cl.append(e)
        elif isinstance(info["episode_info"],Timeout):
            to.append(e)
        else:
            raise NotImplementedError
        print("test_episode_{}_{}".format(e,info["episode_info"]))
        if args["exp_name"] == "video":
            video_recorder.save("test_episode_{}_{}.mp4".format(e,info["episode_info"]))

    # Data to be written
    results_dictionary = {
        "success_episode": sr,
        "success_rate": len(sr)/args["test_episode"],
        "collision_episode": cl,
        "collision_rate": len(cl)/args["test_episode"],
        "navigation_time":nt
    }
    print(results_dictionary)
    # Serializing json
    json_object = json.dumps(results_dictionary, indent=4)
    
    # Writing to sample.json
    with open(os.path.join(log_dir,"results.json"), "w") as outfile:
        outfile.write(json_object)
