import torch
import torch.nn as nn
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.custom import ScanEncoder,StateEncoder

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
    
def get_combined_dim(cent_obs_feature_dim, act_spaces):
    """Get the combined dimension of central observation and individual actions."""
    combined_dim = cent_obs_feature_dim
    for space in act_spaces:
        if space.__class__.__name__ == "Box":
            combined_dim += space.shape[0]
        elif space.__class__.__name__ == "Discrete":
            combined_dim += space.n
        else:
            action_dims = space.nvec
            for action_dim in action_dims:
                combined_dim += action_dim
    return combined_dim


class ContinuousQNet(nn.Module):
    """Q Network for continuous and discrete action space. Outputs the q value given global states and actions.
    Note that the name ContinuousQNet emphasizes its structure that takes observations and actions as input and outputs
    the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space.
    """

    def __init__(self, args, cent_obs_space, act_spaces, device=torch.device("cpu")):
        super(ContinuousQNet, self).__init__()
        activation_func = args["activation_func"]
        hidden_sizes = args["hidden_sizes"]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.feature_extractor = RobotBase(args,act_spaces)
        cent_obs_feature_dim = self.feature_extractor.repr_dim
        
        sizes = (
            [get_combined_dim(cent_obs_feature_dim, act_spaces)]
            + list(hidden_sizes)
            + [1]
        )
        self.mlp = PlainMLP(sizes, activation_func)
        self.to(device)

    def forward(self, cent_obs, actions):
        if self.feature_extractor is not None:
            feature = self.feature_extractor(cent_obs)
        else:
            feature = cent_obs
        if torch.sum(torch.isnan(feature))>0:
            raise ValueError
        concat_x = torch.cat([feature, actions], dim=-1)
        if torch.sum(torch.isnan(concat_x))>0:
            print(actions)
            raise ValueError
        q_values = self.mlp(concat_x)
        return q_values
