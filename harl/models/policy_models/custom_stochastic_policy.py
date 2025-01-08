import torch
import torch.nn as nn
from harl.utils.envs_tools import check
# from harl.models.base.custom import ScanEncoder,StateEncoder
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.models.base.robot_crowd_base import RobotCrowdBase,VisRobotBase,MLPCrowdBase
from harl.utils.envs_tools import get_shape_from_obs_space

# class RobotBase(nn.Module):

#     def __init__(self,args,action_space):
#         super(RobotBase,self).__init__()

#         self.robot_scan_shape = 720
#         self.robot_state_shape = 6

#         self.scan_encoder=ScanEncoder(self.robot_scan_shape,args)
#         self.state_encoder=StateEncoder(self.scan_encoder.repr_dim+self.robot_state_shape,args)
#         self.repr_dim = self.state_encoder.repr_dim

#     def forward(self, x):
        
#         robot_state = x[...,:self.robot_state_shape]
#         robot_scan = x[...,self.robot_state_shape:self.robot_state_shape+self.robot_scan_shape]
#         h = self.state_encoder(torch.cat([robot_state,self.scan_encoder(robot_scan)], dim=-1))

#         return h

class CustomStochasticPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(CustomStochasticPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.use_discriminator = args["use_discriminator"]
        self.human_preference_dim = args["human_preference_vector_dim"]
        # print(self.human_preference_dim)

        self.use_vis_aware = args.get("use_vis_aware",False)
        if not self.use_vis_aware:
            base_model_name = args.get("base_model_name","CNN_1D")
            if base_model_name == "CNN_1D":
                base = RobotCrowdBase
            elif base_model_name == "MLP" or base_model_name == "EGCL":
                base = MLPCrowdBase
            else:
                raise NotImplementedError
        else:
            base = VisRobotBase
        args["use_discriminator"] = False #actor use late fusion
        self.base = base(args, centralized=False)

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        self.rnn = RNNLayer(
                self.base.repr_dim,
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1]+self.human_preference_dim if self.use_discriminator else self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )

        self.to(device)

    def get_actor_features(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            """
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            """
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_discriminator:
            actor_features = torch.cat([actor_features,obs[...,-self.human_preference_dim:]],dim=-1)

        return actor_features # (T, N, -1)

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_discriminator:
            actor_features = torch.cat([actor_features,obs[...,-self.human_preference_dim:]],dim=-1)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_discriminator:
            actor_features_with_pref_vector = torch.cat([actor_features,obs[...,-self.human_preference_dim:]],dim=-1)

            action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
                actor_features_with_pref_vector,
                action,
                available_actions,
                active_masks=active_masks if self.use_policy_active_masks else None,
            )
        else:
            action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=active_masks if self.use_policy_active_masks else None,
            )

        return action_log_probs, dist_entropy, action_distribution,actor_features
