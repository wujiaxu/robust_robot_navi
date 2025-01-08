"""Base runner for on-policy algorithms."""

import time
import numpy as np
import torch
import setproctitle
from harl.common.valuenorm import ValueNorm
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_crowd_ep import OnPolicyCriticBufferCrowdEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
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


class CrowdSimBaseRunner:
    """Base runner for crowd sim using on-policy algorithms."""

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
        self.human_num = algo_args["model"]["human_num"] = env_args["human_num"]
        algo_args["algo"]["human_preference_vector_dim"] = env_args["human_preference_vector_dim"]

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.centralized = algo_args["algo"]["centralized"]
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        self.use_discriminator = algo_args["algo"].get("use_discriminator",False)
        if self.centralized:
            print("CTDE")
        else:
            print("Decentralize training")
        if self.use_discriminator:
            assert self.env_args["use_discriminator"] == True
            print("use_discriminator")
        else:
            print("no discriminator is used")

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"],self.args["cuda_device"])

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
        self.num_agents = self.human_num #get_num_agents(args["env"], env_args, self.envs)

        self.init_actor(args,algo_args)
        self.init_critic(args,algo_args)

        self.logger = LOGGER_REGISTRY[args["env"]](
            args, algo_args, env_args, 1, self.writter, self.run_dir
        )

        # init video recorder for debug
        self.video_recorder = VideoRecorder(self.log_dir)

    def init_actor(self,args,algo_args):

        # actor: share param
        self.actor = []
        self.actor_buffer = []
        human_agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
        for agent_id in range(self.human_num):
            self.actor.append(human_agent) 
            ac_bu = OnPolicyActorBuffer(
                {**algo_args["train"], **algo_args["model"],**algo_args["algo"]},
                self.envs.observation_space[agent_id],
                self.envs.action_space[agent_id],
            )
            self.actor_buffer.append(ac_bu)

        print("actor and actor buffer ready")

        return 
    
    def init_critic(self,args,algo_args):

        self.critic = VCritic(
            {**algo_args["model"], **algo_args["algo"]},
            centralized = self.centralized,
            device=self.device,
        )
        """
        if base_model_name == "CNN_1D":--> FP; central/decentral
                base = RobotCrowdBase
            elif base_model_name == "MLP": --> FP; central/decentral
                base = MLPCrowdBase
            elif base_model_name == "EGCL": --> crowd EP; central
                base = EGCLCrowdBase
        """
        if self.centralized:
            share_observation_space = self.envs.share_observation_space[0]
            if algo_args["model"]["base_model_name"] == "EGCL":
                self.critic_buffer = OnPolicyCriticBufferCrowdEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    self.num_agents,
                )
            else:
                self.critic_buffer = OnPolicyCriticBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    self.num_agents,
                )
        else:
            assert  algo_args["model"]["base_model_name"] != "EGCL"
            observation_space = self.envs.observation_space[0]
            self.critic_buffer = OnPolicyCriticBufferFP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                observation_space,
                self.num_agents,
            )

        if self.algo_args["train"]["use_valuenorm"] is True:
            self.value_normalizer = ValueNorm(1, device=self.device)
            self.aux_value_normalizer = None
        else:
            self.value_normalizer = None
            self.aux_value_normalizer = None

        print("critic and critic buffer ready")
        return 

    def run(self):
        """Run the training pipeline."""
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
                    aux_rewards,
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
                if self.use_discriminator:
                    rewards = rewards + aux_rewards[:,:,np.newaxis]*self.algo_args["algo"]["intrinsic_reward_scale"]
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
                    aux_rewards
                )

                self.logger.per_step(data)  # logger callback at each step

                self.insert(data)  # insert data into buffer

            # compute return and update network
            self.compute()
            self.prep_training()  # change to train mode

            if self.centralized:
                actor_train_infos, critic_train_info = self.train_centralized()
            else:
                actor_train_infos, critic_train_info = self.train_decentralized()
            
            # log information
            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )
                print(self.value_normalizer.running_mean_var())

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
            new_obs = self.actor_buffer[agent_id].human_feature_extractor(obs[:, agent_id].copy())
            self.actor_buffer[agent_id].obs[0] = new_obs
        if self.state_type == "EP":
            new_share_obs = self.critic_buffer.human_feature_extractor(share_obs[:, 0].copy())
            self.critic_buffer.share_obs[0] = new_share_obs
        elif self.state_type == "FP":
            new_share_obs = self.critic_buffer.human_feature_extractor(share_obs.copy())
            self.critic_buffer.share_obs[0] = new_share_obs

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
        aux_rewards = np.zeros((self.algo_args["train"]["n_rollout_threads"],self.num_agents))
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

            #TODO implement
            if not self.use_discriminator:
                pass
            else:
                loss = self.actor[agent_id].get_aux_reward(
                                        self.actor_buffer[agent_id].obs[step],
                                        rnn_state,
                                        action,
                                        self.actor_buffer[agent_id].active_masks[step])
                aux_rewards[:,agent_id] = -np.log(1./self.env_args["human_preference_vector_dim"])-_t2n(loss)
        
        # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        # collect values, rnn_states_critic from 1 critic
        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(
                self.critic_buffer.share_obs[step],
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                # self.critic_buffer.masks[step],
                np.concatenate(self.critic_buffer.masks[step]),
            )
            
            # split (n_threads * n_agents, dim) into (n_threads, n_agents, dim)
            # values = np.array(
            #     np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"])
            # )
            # # (n_threads, dim)
            # rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                np.concatenate(self.critic_buffer.masks[step]),
            )  # concatenate (n_threads, n_agents, dim) into (n_threads * n_agents, dim)
            # split (n_threads * n_agents, dim) into (n_threads, n_agents, dim)
        values = np.array(
            np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"])
        )
        rnn_states_critic = np.array(
            np.split(
                _t2n(rnn_state_critic), self.algo_args["train"]["n_rollout_threads"]
            )
        )

        return aux_rewards,values, actions, action_log_probs, rnn_states, rnn_states_critic

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
            rnn_states_critic,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            aux_rewards,
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
        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), 
                 self.num_agents,
                 self.recurrent_n, 
                 self.rnn_hidden_size),
                dtype=np.float32,
            )
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
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
        # if self.state_type == "EP":
        #     bad_masks = np.array(
        #         [
        #             [0.0]
        #             if "bad_transition" in info[0].keys()
        #             and info[0]["bad_transition"] == True
        #             else [1.0]
        #             for info in infos
        #         ]
        #     )
        # elif self.state_type == "FP":
        bad_masks = np.array(
            [
                [
                    [0.0]
                    if "bad_transition" in info[agent_id].keys()
                    and info[agent_id]["bad_transition"] == True
                    else [1.0]
                    for agent_id in range(self.num_agents)
                ]
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
        if self.centralized:
            if self.state_type == "EP":
                self.critic_buffer.insert(
                    share_obs[:, 0],
                    rnn_states_critic,
                    values,
                    rewards,#[:, 0],
                    masks,#[:, 0],
                    bad_masks,
                )
            elif self.state_type == "FP":
                self.critic_buffer.insert(
                    share_obs, rnn_states_critic, values, rewards, masks, bad_masks
                )
        else:
            self.critic_buffer.insert(
                obs, rnn_states_critic, values, rewards, masks, bad_masks
            )

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        if self.state_type == "EP": #EGCL only
            next_value, _ = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
                np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                np.concatenate(self.critic_buffer.masks[-1]),
            )
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[-1]),
                np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                np.concatenate(self.critic_buffer.masks[-1]),
            )
        next_value = np.array(
            np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"])
        )
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def train_centralized(self):
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

        # normalize advantages for FP
        # if self.state_type == "FP":
        active_masks_collector = [
            self.actor_buffer[i].active_masks for i in range(self.num_agents)
        ]
        active_masks_array = np.stack(active_masks_collector, axis=2)
        advantages_copy = advantages.copy()
        advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())
        for agent_id in agent_order:
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

            # compute action log probs for the actor before update.
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update actor
            # if self.state_type == "EP":
            #     actor_train_info = self.actor[agent_id].train(
            #         self.actor_buffer[agent_id], advantages.copy(), "EP"
            #     )
            # elif self.state_type == "FP":
            actor_train_info = self.actor[agent_id].train(
                self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
            )

            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update factor for next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            actor_train_infos[agent_id]= actor_train_info

        average_actor_train_info = {}
        for key in actor_train_infos[0]:
            value = []
            for agent_id in range(self.num_agents):
                value.append(actor_train_infos[agent_id][key])
            average_actor_train_info[key] = sum(value)/len(value)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return [average_actor_train_info], critic_train_info
    
    def train_decentralized(self):

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
        # # normalize advantages for FP
        # if self.state_type == "FP":
        active_masks_collector = [
            self.actor_buffer[i].active_masks for i in range(self.num_agents)
        ]
        active_masks_array = np.stack(active_masks_collector, axis=2)
        advantages_copy = advantages.copy()
        advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # if self.fixed_order:
        agent_order = list(range(self.num_agents))
        # else:
        #     agent_order = list(torch.randperm(self.num_agents).numpy())
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(
                factor
            )  # current actor save factor

            # update actor
            # if self.state_type == "EP":
            #     actor_train_info = self.actor[agent_id].train(
            #         self.actor_buffer[agent_id], advantages.copy(), "EP"
            #     )
            # elif self.state_type == "FP":
            actor_train_info = self.actor[agent_id].train(
                self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
            )

            actor_train_infos[agent_id]= actor_train_info
        average_actor_train_info = {}
        for key in actor_train_info[0]:
            value = []
            for agent_id in range(self.num_agents):
                value.append(actor_train_infos[agent_id][key])
            average_actor_train_info[key] = sum(value)/len(value)
        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return [average_actor_train_info], critic_train_info

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
                    self.actor_buffer[agent_id].human_feature_extractor(
                        eval_obs[:, agent_id]),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None
                    else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))
            #fuck!!! actions need padding
            pad_size = max([action.shape[-1] for action in eval_actions_collector])
            eval_actions_collector_padded = []
            for action in eval_actions_collector:
                if action.shape[-1]<pad_size:
                    action_padded = np.concatenate([action, 
                                                    np.zeros((action.shape[0],
                                                            pad_size-action.shape[-1]),
                                                            dtype=np.float32)],
                                                    axis=-1)
                else:
                    action_padded = action
                eval_actions_collector_padded.append(action_padded)
            eval_actions = np.array(eval_actions_collector_padded).transpose(1, 0, 2)

            #eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

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
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
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
            print(self.value_normalizer.state_dict())

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
