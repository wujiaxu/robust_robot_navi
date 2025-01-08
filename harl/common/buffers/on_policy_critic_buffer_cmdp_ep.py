"""On-policy buffer for critic that uses Feature-Pruned (EP) state."""
import torch
import numpy as np
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.utils.trans_tools import _flatten, _ma_cast,_sa_cast
from harl.common.buffers.on_policy_critic_buffer_crowd_ep import OnPolicyCriticBufferCrowdEP

class OnPolicyCriticBufferCMDPEP(OnPolicyCriticBufferCrowdEP):
    """On-policy buffer for critic that uses Feature-Pruned (EP) state.
    shared_obs --> crowd reward, auxiliary reward
    """

    def __init__(self, args, share_obs_space, num_agents):
        """Initialize on-policy critic buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
            num_agents: (int) number of agents
        """
        super(OnPolicyCriticBufferCMDPEP,self).__init__(args, share_obs_space, num_agents)

        # Buffer for value predictions made by this aux_critic
        self.aux_value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )

        # Buffer for aux_returns calculated at each timestep
        self.aux_returns = np.zeros_like(self.value_preds)

        # Buffer for rewards received by agents at each timestep
        self.aux_rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
    def insert(
        self, share_obs, rnn_states_critic, 
        value_preds, aux_value_preds, 
        rewards, aux_rewards, masks, bad_masks
    ):
        """Insert data into buffer."""
        share_obs = self.human_feature_extractor(share_obs)
        self.share_obs[self.step + 1] = share_obs.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.aux_value_preds[self.step] = aux_value_preds.copy()
        self.aux_rewards[self.step] = aux_rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.episode_length
    
    def human_feature_extractor(self,share_obs):
        if self.down_sample_lidar_scan_bin == 720: return share_obs
        obs = share_obs[...,:-self.num_agents*2].reshape(self.n_rollout_threads,self.num_agents,726+self.human_preference_dim)
        state = obs[...,:6]
        scan = obs[...,6:726]
        indices = np.linspace(0, 720 - 1, self.down_sample_lidar_scan_bin, dtype=int)
        down_sampled_scan = scan[...,indices]
        pref = obs[...,726:]
        new_obs = np.concatenate([state,down_sampled_scan,pref],axis=-1).reshape(self.n_rollout_threads,-1)
        new_obs = np.concatenate([new_obs,share_obs[...,-self.num_agents*2:]],axis=-1)
        return new_obs
    
    
    def compute_aux_returns(self,next_aux_value,value_normalizer=None):

        if (
            self.use_proper_time_limits
        ):  # consider the difference between truncation and termination
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_aux_value
                gae = 0
                for step in reversed(range(self.aux_rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            self.aux_rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.aux_returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            self.aux_rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.aux_returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                self.aux_returns[-1] = next_aux_value
                for step in reversed(range(self.aux_rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        self.aux_returns[step] = (
                            self.aux_returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.aux_rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        self.aux_returns[step] = (
                            self.aux_returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.aux_rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:  # do not consider the difference between truncation and termination, i.e. all done episodes are terminated
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_aux_value
                gae = 0
                for step in reversed(range(self.aux_rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            self.aux_rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.aux_returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            self.aux_rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.aux_returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                self.aux_returns[-1] = next_aux_value
                for step in reversed(range(self.aux_rewards.shape[0])):
                    self.aux_returns[step] = (
                        self.aux_returns[step + 1] * self.gamma * self.masks[step + 1]
                        + self.aux_rewards[step]
                    )

        return

    def recurrent_generator_critic(self, critic_num_mini_batch, data_chunk_length):
        """Training data generator for critic that uses RNN network.
        This generator splits the trajectories into chunks of length data_chunk_length,
        and therefore maybe more efficient than the naive_recurrent_generator_actor in training.
        Args:
            critic_num_mini_batch: (int) Number of mini batches for critic.
            data_chunk_length: (int) Length of data chunks.
        """

        # get episode_length, n_rollout_threads, num_agents, and mini_batch_size
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length #* num_agents
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // critic_num_mini_batch

        assert (
            episode_length % data_chunk_length == 0
        ), f"episode length ({episode_length}) must be a multiple of data chunk length ({data_chunk_length})."
        assert data_chunks >= 2, "need larger batch size"

        # shuffle indices
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        # The following data operations first transpose the first three dimensions of the data (episode_length, n_rollout_threads, num_agents)
        # to (n_rollout_threads, num_agents, episode_length), then reshape the data to (n_rollout_threads * num_agents * episode_length, *dim).
        # Take share_obs shape as an example:
        # (episode_length + 1, n_rollout_threads, num_agents, *share_obs_shape) --> (episode_length, n_rollout_threads, num_agents, *share_obs_shape)
        # --> (n_rollout_threads, num_agents, episode_length, *share_obs_shape) --> (n_rollout_threads * num_agents * episode_length, *share_obs_shape)
        # if len(self.share_obs.shape) > 3:
        #     share_obs = (
        #         self.share_obs[:-1]
        #         .transpose(1, 0, 2, 3, 4)
        #         .reshape(-1, *self.share_obs.shape[2:])
        #     )
        # else:
        share_obs = _sa_cast(self.share_obs[:-1])
        value_preds = self.value_preds[:-1].transpose(1, 2, 0, 3).reshape(-1, *self.value_preds.shape[2:])
        returns = self.returns[:-1].transpose(1, 2, 0, 3).reshape(-1, *self.returns.shape[2:])
        masks = self.masks[:-1].transpose(1, 2, 0, 3).reshape(-1, *self.masks.shape[2:])
        # (L+1,R,N,M,H) -> (R*L,N,M,H)
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.rnn_states_critic.shape[2:])
        )

        aux_value_preds = self.aux_value_preds[:-1].transpose(1, 2, 0, 3).reshape(-1, *self.aux_value_preds.shape[2:])
        aux_returns = self.aux_returns[:-1].transpose(1, 2, 0, 3).reshape(-1, *self.aux_returns.shape[2:])
        # generate mini-batches
        for indices in sampler:
            share_obs_batch = []
            rnn_states_critic_batch = []
            value_preds_batch = []
            return_batch = []
            aux_value_preds_batch = []
            aux_return_batch = []
            masks_batch = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                aux_value_preds_batch.append(aux_value_preds[ind : ind + data_chunk_length])
                aux_return_batch.append(aux_returns[ind : ind + data_chunk_length])
                rnn_states_critic_batch.append(
                    rnn_states_critic[ind]
                )  # only the beginning rnn states are needed

            L, N = data_chunk_length, mini_batch_size
            # These are all ndarrays of size (data_chunk_length, mini_batch_size, *dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            aux_value_preds_batch = np.stack(aux_value_preds_batch, axis=1)
            aux_return_batch = np.stack(aux_return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            # rnn_states_critic_batch is a (mini_batch_size, *dim) ndarray
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[2:]
            )

            # Flatten the (data_chunk_length, mini_batch_size, *dim) ndarrays to (data_chunk_length * mini_batch_size, *dim)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            value_preds_batch = _flatten(L*N,self.num_agents,_flatten(L, N, value_preds_batch))
            return_batch = _flatten(L*N,self.num_agents,_flatten(L, N, return_batch))
            aux_value_preds_batch = _flatten(L*N,self.num_agents,_flatten(L, N, aux_value_preds_batch))
            aux_return_batch = _flatten(L*N,self.num_agents,_flatten(L, N, aux_return_batch))
            masks_batch = _flatten(L*N,self.num_agents,_flatten(L, N, masks_batch))
            rnn_states_critic_batch = _flatten(N,self.num_agents,rnn_states_critic_batch)
            yield (share_obs_batch, 
                    rnn_states_critic_batch,
                    value_preds_batch, 
                    aux_value_preds_batch, 
                    return_batch, 
                    aux_return_batch,
                    masks_batch)
