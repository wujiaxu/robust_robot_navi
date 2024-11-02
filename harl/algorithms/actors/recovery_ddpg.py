from copy import deepcopy
import torch
from torch import nn
from harl.utils.envs_tools import check
from harl.algorithms.actors.off_policy_base import OffPolicyBase
from harl.models.base.custom import ScanEncoder,StateEncoder
from harl.models.base.plain_mlp import PlainMLP

class RobotBase(nn.Module):

    def __init__(self,args,action_space):
        super(RobotBase,self).__init__()

        self.robot_scan_shape = 720
        self.robot_state_shape = 6

        self.scan_encoder=ScanEncoder(self.robot_scan_shape,args)
        self.state_encoder=StateEncoder(self.scan_encoder.repr_dim+self.robot_state_shape,args)
        self.repr_dim = self.state_encoder.repr_dim

    def forward(self, x):
        
        robot_state = x[...,:self.robot_state_shape]
        robot_scan = x[...,self.robot_state_shape:self.robot_state_shape+self.robot_scan_shape]
        h = self.state_encoder(torch.cat([robot_state,self.scan_encoder(robot_scan)], dim=-1))

        return h
    
class DeterministicPolicy(nn.Module):
    """Deterministic policy network for continuous action space."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize DeterministicPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
 
        self.feature_extractor = RobotBase(args,action_space)
            
        act_dim = action_space.shape[0]
        pi_sizes = [self.feature_extractor.repr_dim] + list(hidden_sizes) + [act_dim]
        self.pi = PlainMLP(pi_sizes, activation_func, final_activation_func)
        low = torch.tensor(action_space.low).to(**self.tpdv)
        high = torch.tensor(action_space.high).to(**self.tpdv)
        self.scale = (high - low) / 2
        self.mean = (high + low) / 2
        self.to(device)

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        if self.feature_extractor is not None:
            x = self.feature_extractor(obs)
        else:
            x = obs
        if torch.sum(torch.isnan(x))>0:
            print(x)
            raise ValueError
        x = self.pi(x)
        if torch.sum(torch.isnan(x))>0:
            print(x)
            raise ValueError
        x = self.scale * x + self.mean
        if torch.sum(torch.isnan(x))>0:
            print(x,self.scale,self.mean)
            raise ValueError
        return x
    
class RecoveryDDPG(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        assert (
            act_space.__class__.__name__ == "Box"
        ), f"only continuous action space is supported by {self.__class__.__name__}."
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.expl_noise = args["expl_noise"]

        self.actor = DeterministicPolicy(args, obs_space, act_space, device)
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.low = torch.tensor(act_space.low).to(**self.tpdv)
        self.high = torch.tensor(act_space.high).to(**self.tpdv)
        self.scale = (self.high - self.low) / 2
        self.mean = (self.high + self.low) / 2
        self.turn_off_grad()

    def get_actions(self, obs, add_noise):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            add_noise: (bool) whether to add noise
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.actor(obs)
        if add_noise:
            actions += torch.randn_like(actions) * self.expl_noise * self.scale
            actions = torch.clamp(actions, self.low, self.high)
        return actions

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        return self.target_actor(obs)
    
