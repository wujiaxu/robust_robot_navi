# if use discriminator
"""On-policy buffer for critic that uses Feature-Pruned (FP) state and discriminator."""
from collections import deque
import torch
from torch import nn
import numpy as np
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.models.base.rnn import RNNLayer
from harl.models.base.mlp import MLPLayer
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method
from harl.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)
from harl.utils.trans_tools import _flatten, _ma_cast,_sa_cast
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def push(self, item):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()
class Net(nn.Module):
    """
    Defines the discriminator network and the training objective for the
    discriminator. Through the Habitat Baselines auxiliary loss registry, this
    is automatically added to the policy class and the loss is computed in the
    policy update.
    """

    def __init__(
        self,
        input_size, #global obs of crowd
        joint_action_dim,
        hidden_sizes,
        recurrent_n,
        initialization_method,
        behavior_latent_dim,
        device
    ):
        super().__init__()
        self.input_dim = input_size
        self.action_input_dim = joint_action_dim
        self.behavior_latent_dim = behavior_latent_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.base = MLPLayer(input_size+joint_action_dim,hidden_sizes,initialization_method,"relu")
        init_method = get_init_method(initialization_method)

        self.rnn = RNNLayer(
                hidden_sizes[-1],
                hidden_sizes[-1],
                recurrent_n,
                initialization_method,
            )
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        
        self.discrim_output = nn.Sequential(
            init_(nn.Linear(hidden_sizes[-1], hidden_sizes[-1])),
            nn.ReLU(True),
            init_(nn.Linear(hidden_sizes[-1], hidden_sizes[-1])),
            nn.ReLU(True),
            init_(nn.Linear(hidden_sizes[-1], behavior_latent_dim)),
        )
        self.to(device)

    def pred_logits(self, policy_features):
        return self.discrim_output(policy_features)

    def forward(self,cent_obs,joint_action, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        joint_action = check(joint_action).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        crowd_obs = cent_obs[...,
                             -self.behavior_latent_dim-self.input_dim
                             :-self.behavior_latent_dim]
        behav_ids = torch.argmax(cent_obs[...,-self.behavior_latent_dim:], -1)
        input_discrim = torch.cat([crowd_obs,joint_action],dim=-1)
        discrim_features = self.base(input_discrim)
        discrim_features, rnn_states = self.rnn(discrim_features, rnn_states, masks)
        pred_logits = self.pred_logits(discrim_features)
        # if self.training:
        #     # print(cent_obs[...,-self.behavior_latent_dim:])
        #     # input()
        #     N = rnn_states.size(0)
        #     T = int(input_discrim.size(0) / N)

        #     # unflatten
        #     x = input_discrim.view(T, N, input_discrim.size(1))
        #     print(x[:,1],pred_logits[0],behav_ids[0])
        #     # input()
        #     print(x[:,2],pred_logits[1],behav_ids[1])
        #     # input()
        #     print(x[:,3],pred_logits[2],behav_ids[2])
        #     # input()
        loss = F.cross_entropy(pred_logits, behav_ids,reduction="none")
        return loss,rnn_states

class BehavDiscrim():
    def __init__(self, input_dim,action_input_dim,args,device):
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = 10 #args["critic_num_mini_batch"]
        self.data_chunk_length = args["data_chunk_length"]

        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.train_every = 5
        self.episode = 0
        self.sample_buffer = ReplayMemory(capacity=10000)
        self.num_mini_batch = 1000
        self.warmup_episode = 25

        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]

        self.net = Net(input_dim,
                       action_input_dim,
                        self.hidden_sizes,
                        self.recurrent_n,
                        self.initialization_method,
                        args["human_preference_vector_dim"], 
                        device
                        )
        
        self.discrim_optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

        return
    
    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.discrim_optimizer, episode, episodes, self.critic_lr)

    def get_values(self, cent_obs, joint_action,rnn_states_discrim, masks):
        losses, rnn_states_discrim = self.net(cent_obs, joint_action,rnn_states_discrim, masks)
        return losses, rnn_states_discrim
    
    def update(self,sample):
        
        (
            share_obs_batch,
            joint_action_batch,
            rnn_states_discrim_batch,
            masks_batch,
        ) = sample

        discrim_losses,_ = self.net(
            share_obs_batch, 
            joint_action_batch,
            rnn_states_discrim_batch, 
            masks_batch)

        self.discrim_optimizer.zero_grad()
        discrim_loss = discrim_losses.mean()
        discrim_loss.backward()

        if self.use_max_grad_norm:
            discrim_grad_norm = nn.utils.clip_grad_norm_(
                self.net.parameters(), self.max_grad_norm
            )
        else:
            discrim_grad_norm = get_grad_norm(self.net.parameters())

        self.discrim_optimizer.step()

        return discrim_loss,discrim_grad_norm
     
    def train(self,critic_buffer):
        train_info = {}
        train_info["discrim_loss"] = 0
        train_info["discrim_grad_norm"] = 0
        self.episode+=1
        for _ in range(self.critic_epoch):
            data_generator = critic_buffer.recurrent_generator_discrim(
                    self.critic_num_mini_batch, self.data_chunk_length
                )
            for sample in data_generator:
                self.sample_buffer.push(sample)
                
        if self.episode%self.train_every ==0 and self.episode>self.warmup_episode:
            print("update discriminator",self.sample_buffer.position)
            if self.sample_buffer.is_full():
                sampler = torch.randperm(self.sample_buffer.capacity).numpy()
            else:
                sampler = torch.randperm(self.sample_buffer.position).numpy()
            for i in range(self.num_mini_batch):
                sample = self.sample_buffer[sampler[i%len(sampler)]]
                discrim_loss, discrim_grad_norm = self.update(sample)#sample for replay buffer
                train_info["discrim_loss"] += discrim_loss.item()
                train_info["discrim_grad_norm"] += discrim_grad_norm

        num_updates = self.num_mini_batch

        for k, _ in train_info.items():
            train_info[k] /= num_updates
        return train_info
    
    def prep_training(self):
        """Prepare for training."""
        self.net.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.net.eval()
    
class OnPolicyCriticBufferDIS(OnPolicyCriticBufferFP):
    """On-policy buffer for critic that uses Feature-Pruned (FP) state.
    When FP state is used, the critic takes different global state as input for different actors. Thus, OnPolicyCriticBufferFP has an extra dimension for number of agents compared to OnPolicyCriticBufferEP.
    """

    def __init__(self, args, share_obs_space, num_agents,device=torch.device("cpu")):
        """Initialize on-policy critic buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
            num_agents: (int) number of agents
        """
        super(OnPolicyCriticBufferDIS, self).__init__(
            args, share_obs_space, num_agents
        )
        

        self.discrim = BehavDiscrim(num_agents*2,
                                    num_agents*2,
                                        args,
                                        device
                                        )
        
        self.rnn_states_discrim = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        self.joint_action = np.zeros(
            (
                self.episode_length,
                self.n_rollout_threads,
                num_agents,
                2,
            ),
            dtype=np.float32,
        )

    def insert(
            self, share_obs, joint_action, rnn_states_critic, rnn_state_discrim, value_preds, rewards, masks, bad_masks
        ):
            """Insert data into buffer."""
            self.share_obs[self.step + 1] = share_obs.copy()
            self.joint_action[self.step] = joint_action.copy()
            self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
            self.rnn_states_discrim[self.step+1] = rnn_state_discrim.copy()
            self.value_preds[self.step] = value_preds.copy()
            self.rewards[self.step] = rewards.copy()
            self.masks[self.step + 1] = masks.copy()
            self.bad_masks[self.step + 1] = bad_masks.copy()

            self.step = (self.step + 1) % self.episode_length

    def recurrent_generator_discrim(self, critic_num_mini_batch, data_chunk_length):
        """Training data generator for critic that uses RNN network.
        This generator splits the trajectories into chunks of length data_chunk_length,
        and therefore maybe more efficient than the naive_recurrent_generator_actor in training.
        Args:
            critic_num_mini_batch: (int) Number of mini batches for critic.
            data_chunk_length: (int) Length of data chunks.
        """

        # get episode_length, n_rollout_threads, num_agents, and mini_batch_size 
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3] #30,64,6
        batch_size = n_rollout_threads * episode_length #1920
        data_chunks = batch_size // data_chunk_length #1920//10 = 192 chuncks
        mini_batch_size = data_chunks // critic_num_mini_batch #192/1

        assert (
            episode_length % data_chunk_length == 0
        ), f"episode length ({episode_length}) must be a multiple of data chunk length ({data_chunk_length})."
        assert data_chunks >= 2, "need larger batch size"

        # shuffle indices (shuffle chunk id)
        rand = torch.randperm(data_chunks).numpy()
        # decide which chunks to involve for each batch
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]
        # print(self.share_obs[:-1,:,0][0,0,-22:])
        # print(self.share_obs[:-1,:,0][1,0,-22:])
        # input()
        share_obs = _sa_cast(self.share_obs[:-1,:,0])
        # print(share_obs[:10,-22:])
        # input()
        
        # (episode_length, n_rollout_threads, joint_action_dim)
        # --> (n_rollout_threads * episode_length, joint_action_dim)
        joint_action = _sa_cast(self.joint_action.reshape(
                    self.joint_action.shape[0],
                    self.joint_action.shape[1],
                    -1))
        
        masks = _sa_cast(self.masks[:-1,:,0])
        rnn_states_discrim = (
            self.rnn_states_discrim[:-1,:,0]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.rnn_states_discrim.shape[3:])
        )

        # generate mini-batches
        for indices in sampler: #since only one element in sampler.... the generator only yield onece
            share_obs_batch = []
            joint_action_batch = []
            rnn_states_discrim_batch = []
            masks_batch = []
            # loop over data chunk ids inside each batch
            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                joint_action_batch.append(joint_action[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                rnn_states_discrim_batch.append(
                    rnn_states_discrim[ind]
                )  # only the beginning rnn states are needed

            L, N = data_chunk_length, mini_batch_size #20, 96
            # These are all ndarrays of size (data_chunk_length, mini_batch_size, *dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            joint_action_batch = np.stack(joint_action_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            # rnn_states_discrim_batch is a (mini_batch_size, *dim) ndarray
            rnn_states_discrim_batch = np.stack(rnn_states_discrim_batch).reshape(
                N, *self.rnn_states_critic.shape[3:]
            )

            # Flatten the (data_chunk_length, mini_batch_size, *dim) ndarrays to (data_chunk_length * mini_batch_size, *dim)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            joint_action_batch = _flatten(L, N, joint_action_batch)
            masks_batch = _flatten(L, N, masks_batch)

            yield share_obs_batch, joint_action_batch, rnn_states_discrim_batch, masks_batch