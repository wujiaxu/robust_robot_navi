"""Base runner for on-policy algorithms."""

import time
import numpy as np
import torch
import setproctitle
from harl.common.valuenorm import ValueNorm
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
# from harl.common.buffers.on_policy_critic_buffer_dis import OnPolicyCriticBufferDIS
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics.v_critic import VCritic
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


class OnPolicyAiCrowdRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args,human_model_algo_args):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.human_num = algo_args["model"]["human_num"] = env_args["human_num"]
        self.robot_num = algo_args["model"]["robot_num"] = env_args["robot_num"]

        algo_args["algo"]["human_preference_vector_dim"] = env_args["human_preference_vector_dim"]
        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"],self.args["cuda_device"])
        if not self.algo_args["render"]["use_render"]:  # train, not render
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
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        print(self.num_agents,self.human_num,self.robot_num)
        # print("share_observation_space: ", self.envs.share_observation_space)
        # print("observation_space: ", self.envs.observation_space)
        # print("action_space: ", self.envs.action_space)

        # actor #TODO add human policy
        self.actor = []
        assert self.robot_num == 1
        algo_args["algo"]["use_discriminator"] = False
        if algo_args["algo"]["use_vis_aware"] == True:
            print("use vis aware observation for robot")
        agent = ALGO_REGISTRY[args["algo"]](
            {**algo_args["model"], **algo_args["algo"]},
            self.envs.observation_space[0],
            self.envs.action_space[0],
            device=self.device,
        )
        self.actor.append(agent)
        
        human_agent = ALGO_REGISTRY[args["algo"]](
                {**human_model_algo_args["model"], **human_model_algo_args["algo"]},
                self.envs.observation_space[1],
                self.envs.action_space[1],
                device=self.device,
            )
        human_policy_actor_state_dict = torch.load(
            str(human_model_algo_args["train"]["model_dir"])
            + "/models"
            + "/actor_agent1"
            + ".pt"
        )
        human_agent.actor.load_state_dict(human_policy_actor_state_dict)
        for _ in range(self.human_num):
            self.actor.append(human_agent)


        if self.algo_args["render"]["use_render"] is False:  # train, not render
            self.actor_buffer = []
            for agent_id in range(self.num_agents):
                ac_bu = OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"],**algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                self.actor_buffer.append(ac_bu)

            self.critic = VCritic(
                {**algo_args["model"], **algo_args["algo"]},
                centralized=False,
                device=self.device,
            )
    
            # EP stands for Environment Provided, as phrased by MAPPO paper.
            # In EP, the global states for all agents are the same.
            self.critic_buffer = OnPolicyCriticBufferEP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[0],
            )

            if self.algo_args["train"]["use_valuenorm"] is True:
                self.value_normalizer = ValueNorm(1, device=self.device)
            else:
                self.value_normalizer = None

            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )

        # init video recorder for debug
        self.video_recorder = VideoRecorder(self.log_dir)

    def run(self):
        """Run the training (or rendering) pipeline."""
        print("start running")
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init(episodes)  # logger callback at the beginning of training

        for episode in range(1, episodes + 1):
            if self.algo_args["train"][
                "use_linear_lr_decay"
            ]:  # linear decay of learning rate

                self.actor[0].lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)

            self.logger.episode_init(
                episode
            )  # logger callback at the beginning of each episode

            self.prep_rollout()  # change to eval mode
            for step in range(self.algo_args["train"]["episode_length"]):
                # Sample actions from actors and values from critics
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                
                # actions: (n_threads, n_agents, action_dim)
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    np.zeros_like(rewards),#aux_reward
                )

                self.logger.per_step(data)  # logger callback at each step

                self.insert(data)  # insert data into buffer

            # compute return and update network
            self.compute()
            self.prep_training()  # change to train mode

            actor_train_infos, critic_train_info = self.train()
            
            # log information
            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

            # eval
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save()

            self.after_update()

    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()

        self.critic_buffer.share_obs[0] = obs[:, 0].copy()

    @torch.no_grad()
    def collect(self, step):
        """Collect actions and values from actors and critics.
        Args:
            step: step in the episode.
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        # collect actions, action_log_probs, rnn_states from n actors
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        for agent_id in range(self.num_agents):
            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
                self.actor_buffer[agent_id].obs[step],
                self.actor_buffer[agent_id].rnn_states[step],
                self.actor_buffer[agent_id].masks[step],
                self.actor_buffer[agent_id].available_actions[step]
                if self.actor_buffer[agent_id].available_actions is not None
                else None,
            )
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))

        # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        # collect values, rnn_states_critic from 1 critic
        value, rnn_state_critic = self.critic.get_values(
            self.critic_buffer.share_obs[step],
            self.critic_buffer.rnn_states_critic[step],
            self.critic_buffer.masks[step],
        )
        # (n_threads, dim)
        values = _t2n(value)
        rnn_states_critic = _t2n(rnn_state_critic)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """Insert data into buffer."""
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            rnn_states,  # (n_threads, n_agents, dim)
            rnn_states_critic,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)s
            _, # aux_reward (not used)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        rnn_states[
            dones_env == True
        ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # If env is done, then reset rnn_state_critic to all zero
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )

        # masks use 0 to mask out threads that just finish.
        # this is used for denoting at which point should rnn state be reset
        masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # bad_masks use 0 to denote truncation and 1 to denote termination
        bad_masks = np.array(
            [
                [0.0]
                if "bad_transition" in info[0].keys()
                and info[0]["bad_transition"] == True
                else [1.0]
                for info in infos
            ]
        )

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id], #TODO check impact
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

        self.critic_buffer.insert(
            obs[:, 0],
            rnn_states_critic,
            values,
            rewards[:, 0],
            masks[:, 0],
            bad_masks,
        )


    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        next_value, _ = self.critic.get_values(
            self.critic_buffer.share_obs[-1],
            self.critic_buffer.rnn_states_critic[-1],
            self.critic_buffer.masks[-1],
        )
        next_value = _t2n(next_value)

        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def train(self):
        """Train the model."""
        actor_train_infos = {}

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )
        agent_id = 0 #only train robot agent
        self.actor_buffer[agent_id].update_factor(
            factor
        )  # current actor save factor

        # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
        available_actions = (
            None
            if self.actor_buffer[agent_id].available_actions is None
            else self.actor_buffer[agent_id]
            .available_actions[:-1]
            .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
        )

        # update actor

        actor_train_info = self.actor[agent_id].train(
            self.actor_buffer[agent_id], advantages.copy(), "EP"
        )

        actor_train_infos[agent_id]= actor_train_info

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return [actor_train_infos[0]]+[{}]*self.human_num, critic_train_info

    def after_update(self):
        """Do the necessary data operations after an update.
        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        """Evaluate the model."""
        self.logger.eval_init()  # logger callback at the beginning of evaluation

        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        self.video_recorder.init(self.eval_envs, enabled=True)

        eval_rnn_states = np.zeros(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None
                    else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(
                eval_data
            )  # logger callback at each step of evaluation
            
            self.video_recorder.record(self.eval_envs)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[
                eval_dones_env == True
            ] = np.zeros(  # if env is done, then reset rnn_state to all zero
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(
                        eval_i
                    )  # logger callback when an episode is done
                    # save video of this thread?
                
            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(
                    eval_episode
                )  # logger callback at the end of evaluation
                self.video_recorder.save("eval_at_episode_{}.mp4".format(self.logger.episode))
                break

    def prep_rollout(self):
        """Prepare for rollout."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        """Prepare for training."""
        self.actor[0].prep_training()
        self.critic.prep_training()

    def save(self):
        """Save model parameters."""
        policy_actor = self.actor[0].actor
        torch.save(
            policy_actor.state_dict(),
            str(self.save_dir) + "/actor_agent" + str(0) + ".pt",
        )
        policy_critic = self.critic.critic
        torch.save(
            policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + ".pt"
        )
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer" + ".pt",
            )

    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()


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
        default="crowd_env_vis",
        choices=[
            "crowd_env_vis",
        ],
        help="Environment name. Choose from: crowd_sim",
    )
    parser.add_argument(
        "--exp_name", type=str, default="train_on_ai_crowd", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--human_model_dir",
        type=str,
        default="/home/dl/wu_ws/HARL/results/crowd_env/crowd_navi/robot_crowd_happo//ped_sim/seed-00001-2024-09-20-10-42-59",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument("--cuda_device",type=str,default="cuda:0")
    args, unparsed_args = parser.parse_known_args()
    # human_model_config_file = os.path.join(args.human_model_dir,"config.json")
    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    human_model_args = args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    # load model
    human_model_dir = find_seed_directories(args["human_model_dir"],
                                    algo_args["seed"]["seed"])[0] #use the latest one 
    human_model_config_file = os.path.join(human_model_dir,"config.json")
    with open(human_model_config_file, encoding="utf-8") as file:
        human_model_all_config = json.load(file)
    human_model_args["algo"] = human_model_all_config["main_args"]["algo"]
    # human_model_args["env"] = human_model_all_config["main_args"]["env"]
    human_model_algo_args = human_model_all_config["algo_args"]
    human_model_env_args = human_model_all_config["env_args"]
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line
    # human_model_algo_args["train"]["model_dir"] = os.path.join(args["human_model_dir"],"models")

    # # identify base model
    human_model_algo_args["train"]["model_dir"] = human_model_dir
    runner = OnPolicyAiCrowdRunner(args, algo_args, env_args,human_model_algo_args)
    runner.run()
    runner.close()