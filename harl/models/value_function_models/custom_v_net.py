from sklearn import neighbors
import torch
import torch.nn as nn
from harl.models.base.custom import ScanEncoder,StateEncoder
from harl.models.base.rnn import RNNLayer
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method
from harl.models.base.robot_crowd_base import RobotCrowdBase,MLPCrowdBase,EGCLCrowdBase,VisRobotBase

class CustomVNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, centralized=True, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            centralized: bool
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(CustomVNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(self.initialization_method)

        self.use_vis_aware = args.get("use_vis_aware",False)

        if not self.use_vis_aware:
            base_model_name = args.get("base_model_name","CNN_1D")
            if base_model_name == "CNN_1D":
                base = RobotCrowdBase
            elif base_model_name == "MLP":
                base = MLPCrowdBase
            elif base_model_name == "EGCL":
                base = EGCLCrowdBase
            else:
                raise NotImplementedError
        else:
            assert centralized == False
            base = VisRobotBase
        self.base = base(args, centralized)

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        self.rnn = RNNLayer(
            self.base.repr_dim,
            self.hidden_sizes[-1],
            self.recurrent_n,
            self.initialization_method,
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """Compute actions from the given inputs.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        
        
        values = self.v_out(critic_features)

        return values, rnn_states
    
class CustomDoubleVNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, centralized, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            centralized: bool
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(CustomDoubleVNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(self.initialization_method)

        base_model_name = args.get("base_model_name","CNN_1D")
        if base_model_name == "CNN_1D":
            base = RobotCrowdBase
        elif base_model_name == "MLP":
            base = MLPCrowdBase
        elif base_model_name == "EGCL":
            assert centralized == True
            base = EGCLCrowdBase
        else:
            raise NotImplementedError
        self.base = base(args, centralized)

        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        self.rnn = RNNLayer(
            self.base.repr_dim,
            self.hidden_sizes[-1],
            self.recurrent_n,
            self.initialization_method,
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out_1 = init_(nn.Linear(self.hidden_sizes[-1], 1))
        self.v_out_2 = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """Compute actions from the given inputs.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values_1 = self.v_out_1(critic_features)
        values_2 = self.v_out_2(critic_features)

        return values_1,values_2, rnn_states
