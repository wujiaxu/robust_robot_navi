"""Train an algorithm."""
import argparse
import json
import os
from harl.utils.configs_tools import get_defaults_yaml_args, update_args,find_seed_directories
from pathlib import Path

def main():
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
        ],
        help="Environment name. Choose from: crowd_sim",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--lagrangian_k_p", type=float, default=1.0, help="lagrange parameter"
    )
    parser.add_argument(
        "--lagrangian_k_i", type=float, default=0.003, help="lagrange parameter"
    )
    parser.add_argument(
        "--lagrangian_k_d", type=float, default=0.0, help="lagrange parameter"
    )
    parser.add_argument(
        "--lagrangian_upper_bound", type=float, default=1000, help="lagrange parameter"
    )
    parser.add_argument(
        "--lagrangian_lower_bound", type=float, default=1, help="lagrange parameter"
    )
    parser.add_argument(
        "--optimality", type=float, default=0.95, help="optimality constrain of the target policy"
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="/home/dl/wu_ws/HARL/results/crowd_env/crowd_navi/robot_crowd_happo/ped_sim_base/seed-00001-2024-09-24-03-21-00",
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
    base_model_args = args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    # load model
    model_dir = find_seed_directories(args["base_model_dir"],
                                    algo_args["seed"]["seed"])[0] #use the latest one 
    base_model_config_file = os.path.join(model_dir,"config.json")
    with open(base_model_config_file, encoding="utf-8") as file:
        base_model_all_config = json.load(file)
    base_model_args["algo"] = base_model_all_config["main_args"]["algo"]
    base_model_args["env"] = base_model_all_config["main_args"]["env"]
    base_model_algo_args = base_model_all_config["algo_args"]

    # # identify base model
    base_model_algo_args["train"]["model_dir"] = model_dir
    
    # # consider seed:
    
    print(args["base_model_dir"])

    # start training
    from harl.runners.crowd_sim_cmdp_runner import CrowdSimCMDPRunner

    runner = CrowdSimCMDPRunner(args, algo_args, env_args,base_model_algo_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
