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

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from crowd_navi_bench.robot_policy.drl_vo.custom_cnn_full import *
import multiprocessing
from gym.wrappers import RecordVideo

# from zmq import device
# multiprocessing.set_start_method("fork", force=True) 

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
          os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
                  
        # save model every 100000 timesteps:
        if self.n_calls % (20000) == 0:
          # Retrieve training reward
          path = self.save_path + '_model' + str(self.n_calls)
          self.model.save(path)
	  
        return True
    
def make_env(rank,nenv,env_args,seed=0,log_dir=None):
   def _init():
      env_core = RobotCrowdSimWrapper(crowd_model,env_args,phase="train",nenv=nenv,thisSeed=rank,time_step=0.05)
      env = gym.make('drl_vo_env-v0')
      env.configure(env_core)
      if log_dir is not None:
        env = Monitor(env, os.path.join(log_dir,"{}.monitor.csv".format(rank)))
      return env
   return _init

def step_decay_schedule(initial_lr=1e-3, final_lr=5e-5, total_steps=4):
    """
    Step decay schedule for SB3.
    
    :param initial_lr: Initial learning rate (1e-3).
    :param final_lr: Final learning rate (5e-5).
    :param total_steps: Number of decay steps.
    :return: Learning rate function for SB3.
    """
    gamma = (final_lr / initial_lr) ** (1 / total_steps)  # Compute decay factor
    step_size = 1.0 / total_steps  # Fraction of training before decay

    def lr_schedule(progress_remaining):
        """ SB3 expects progress_remaining to go from 1 to 0 """
        current_step = 1 - progress_remaining  # Convert to 0 -> 1 scale
        num_decays = current_step // step_size  # Count how many times to decay
        lr = initial_lr * (gamma ** num_decays)  # Apply decay
        return lr

    return lr_schedule


if __name__ == '__main__':
  import argparse
  import json
  from harl.utils.configs_tools import get_defaults_yaml_args, update_args,find_seed_directories
  """Main function."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )   
  parser.add_argument(
      "--seed", type=int, default=1, help="model seed."
  )
  parser.add_argument(
      "--human_policy", type=str, default="ai", help="human policy."
  )
  parser.add_argument(
      "--exp_name", type=str, default="SFMC_4p_room361", help="Experiment name."
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
  human_model_args = args = vars(args)  # convert to dict

  nenv = 4
  total_train_step = 4000000
  cuda_device = args["cuda_device"] #proposed:"cuda:1"
  exp_name = args["exp_name"] #"proposed_4p_room361_5"
  
  # load model
  human_model_dir = find_seed_directories(
                            args["human_model_dir"],
                            seed=1)[0] 
  
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

  human_model_env_args["max_episode_length"] = 240 #512
  human_model_env_args["human_num"] = 4
  human_model_env_args["robot_num"] = 1
  human_model_env_args["human_policy"] = "SFM"
  human_model_env_args["sfm_v0"] = args["sfm_v0"]
  human_model_env_args["sfm_sigma"] = args["sfm_sigma"]
  # algo_args = all_config["algo_args"]
  # human_model_dir = "/home/dl/wu_ws/robust_robot_navi/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_room361/seed-00001-2024-09-27-10-55-57"
#   human_model_algo_args["train"]["model_dir"] = os.path.join(human_model_dir,"models")
#   human_model_algo_args["algo"]["human_preference_vector_dim"] = human_model_env_args["human_preference_vector_dim"]
  # load ai model 
  crowd_model = []
#   human_agent = ALGO_REGISTRY[args["algo"]](
#                   {**human_model_algo_args["model"], **human_model_algo_args["algo"]},
#                   gym.spaces.Box(-np.inf,np.inf,(729,)),
#                   gym.spaces.Box(-np.inf,np.inf,(2,)),
#                   device=cuda_device,
#               )
#   human_policy_actor_state_dict = torch.load(
#               str(human_model_algo_args["train"]["model_dir"])
#               + "/actor_agent0"
#               + ".pt"
#           )
#   human_agent.actor.load_state_dict(human_policy_actor_state_dict)
#   for _ in range(human_model_env_args["human_num"]):
#       crowd_model.append(human_agent)


  log_dir = os.path.join("/home/dl/wu_ws/robust_robot_navi/crowd_navi_bench/robot_policy/drl_vo/model",args["exp_name"])
  os.makedirs(log_dir, exist_ok=True)
  # vec_env = make_vec_env("drl_vo_env-v0", n_envs=32,monitor_dir=log_dir)
  # for env in vec_env.envs:
  #     env.configure(env_core)

  #, allow_early_resets=True)  # in order to get rollout log data
  vec_env  = SubprocVecEnv([make_env(rank=i,
                                     nenv=nenv,
                                     env_args=human_model_env_args,
                                     log_dir=log_dir) for i in range(nenv)])
#   obs = vec_env.reset()

  # policy parameters:
  policy_kwargs = dict(
      features_extractor_class=CustomCNN,
      features_extractor_kwargs=dict(features_dim=256),
      net_arch=[dict(pi=[256], vf=[128])]
  )

  # raw training:
  model = PPO("CnnPolicy", vec_env , 
              policy_kwargs=policy_kwargs, 
              learning_rate=step_decay_schedule(total_steps=total_train_step), 
              verbose=2, tensorboard_log=log_dir, 
              n_steps=512, n_epochs=10, 
              batch_size=128, device=cuda_device) #, gamma=0.96, ent_coef=0.1, vf_coef=0.4) 

  # # continue training:
  # kwargs = {'tensorboard_log':log_dir, 'verbose':2, 'n_epochs':10, 'n_steps':512, 'batch_size':128,'learning_rate':5e-5}
  # model_file = rospy.get_param('~model_file', "./model/drl_pre_train.zip")
  # model = PPO.load(model_file, env=env, **kwargs)

  # Create the callback: check every 1000 steps
  callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
  model.learn(total_timesteps=total_train_step, 
              log_interval=5, 
              tb_log_name='drl_vo_policy', 
              callback=callback, reset_num_timesteps=True)

  # Saving final model
  model.save("drl_vo_model")
  print("Training finished.")
  vec_env.close()


