import torch
import torch.nn as nn
from harl.models.base.custom import ScanEncoder,StateEncoder
from harl.models.base.rnn import RNNLayer
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method

class DecRobotCrowdBase(nn.Module):

    def __init__(self,args,obs_shape):
        super(DecRobotCrowdBase,self).__init__()
        self.robot_scan_shape = 720
        self.robot_state_shape = 6
        # self.robot_num = args["robot_num"]
        # assert self.robot_num == 1

        self.robot_obs_shape = self.robot_scan_shape + self.robot_state_shape 

        self.scan_encoder=ScanEncoder(self.robot_scan_shape,args)
        self.state_encoder=StateEncoder(self.scan_encoder.repr_dim+self.robot_state_shape,args)

        
        self.repr_dim = (self.state_encoder.repr_dim)


    def forward(self, x):
        # print(x.shape)
        i=0
        robot_state = x[...,i*self.robot_obs_shape
                            :i*self.robot_obs_shape+self.robot_state_shape]
        robot_scan = x[...,i*self.robot_obs_shape+self.robot_state_shape
                            :i*self.robot_obs_shape+self.robot_state_shape+self.robot_scan_shape]
        
        h =self.state_encoder(torch.cat([robot_state,
                                            self.scan_encoder(robot_scan)]
                                    , dim=-1)
                            )
        
        return h
    
class RobotCrowdBase(nn.Module):

    def __init__(self,args,cent_obs_shape):
        super(RobotCrowdBase,self).__init__()
        self.human_scan_shape = 720
        self.human_state_shape = 6
        self.robot_scan_shape = 720
        self.robot_state_shape = 6
        self.human_num = args["human_num"]
        self.robot_num = args["robot_num"]
        self.human_preference_dim = args["human_preference_vector_dim"]
        self.use_discriminator = args["use_discriminator"]

        self.robot_obs_shape = self.robot_scan_shape + self.robot_state_shape + self.human_preference_dim
        self.human_obs_shape = self.human_scan_shape + self.human_state_shape + self.human_preference_dim

        self.scan_encoder=ScanEncoder(self.robot_scan_shape,args)
        self.state_encoder=StateEncoder(self.scan_encoder.repr_dim+self.robot_state_shape,args)

        self.human_scan_encoder=ScanEncoder(self.human_scan_shape,args)
        self.human_state_encoder=StateEncoder(self.human_scan_encoder.repr_dim+self.human_state_shape,args)
        
        if self.use_discriminator:
            self.repr_dim = (self.state_encoder.repr_dim+2)*self.robot_num
            self.repr_dim += (self.human_state_encoder.repr_dim+self.human_preference_dim+2)*self.human_num
        else:
            self.repr_dim = (self.state_encoder.repr_dim+2)*self.robot_num
            self.repr_dim += (self.human_state_encoder.repr_dim+2)*self.human_num

    def forward(self, x):
        # print(x.shape)
        hs = []
        for i in range(self.robot_num):
            robot_state = x[...,i*self.robot_obs_shape
                                :i*self.robot_obs_shape+self.robot_state_shape]
            robot_scan = x[...,i*self.robot_obs_shape+self.robot_state_shape
                                :i*self.robot_obs_shape+self.robot_state_shape+self.robot_scan_shape]
            hs.append(
                self.state_encoder(torch.cat([robot_state,
                                              self.scan_encoder(robot_scan)]
                                        , dim=-1)
                                )
                    )
        
        for i in range(self.human_num):
            human_state = x[...,
                            i*self.human_obs_shape+self.robot_obs_shape*self.robot_num
                            :i*self.human_obs_shape+self.robot_obs_shape*self.robot_num+self.human_state_shape]
            human_scan = x[...,
                            i*self.human_obs_shape+self.robot_obs_shape*self.robot_num+self.human_state_shape\
                            :i*self.human_obs_shape+self.robot_obs_shape*self.robot_num+self.human_state_shape+self.human_scan_shape] 
            hs.append(self.human_state_encoder(torch.cat([human_state,self.human_scan_encoder(human_scan)],dim=-1)))
            if self.use_discriminator:
                hs.append(
                    x[...,
                    i*self.human_obs_shape+self.robot_obs_shape*self.robot_num+self.human_state_shape+self.human_scan_shape\
                    :(i+1)*self.human_obs_shape+self.robot_obs_shape*self.robot_num] 
                )
            # print(hs[-1].shape,hs[-2].shape)

        global_state = x[...,-2*(self.human_num+self.robot_num):]
        hs.append(global_state)
        h = torch.cat(hs, dim=-1)
        return h
    
class VisRobotCrowdBase(nn.Module):

    def __init__(self,args,cent_obs_shape):
        super(VisRobotCrowdBase,self).__init__()
        self.human_scan_shape = 720
        self.human_state_shape = 6
        self.robot_scan_shape = 720
        self.robot_state_shape = 6
        self.human_num = args["human_num"]
        self.robot_num = args["robot_num"]
        self.human_preference_dim = args["human_preference_vector_dim"]
        self.use_discriminator = args["use_discriminator"]

        self.input_channel = 2
        self.robot_obs_shape = self.robot_scan_shape * self.input_channel+ self.robot_state_shape + self.human_preference_dim
        self.human_obs_shape = self.human_scan_shape * self.input_channel+ self.human_state_shape + self.human_preference_dim
        
        self.scan_encoder=ScanEncoder(self.robot_scan_shape,args,input_channel=self.input_channel)
        self.state_encoder=StateEncoder(self.scan_encoder.repr_dim+self.robot_state_shape,args)

        self.human_scan_encoder=ScanEncoder(self.human_scan_shape,args,input_channel=self.input_channel)
        self.human_state_encoder=StateEncoder(self.human_scan_encoder.repr_dim+self.human_state_shape,args)
        
        if self.use_discriminator:
            self.repr_dim = (self.state_encoder.repr_dim+2)*self.robot_num
            self.repr_dim += (self.human_state_encoder.repr_dim+self.human_preference_dim+2)*self.human_num
        else:
            self.repr_dim = (self.state_encoder.repr_dim+2)*self.robot_num
            self.repr_dim += (self.human_state_encoder.repr_dim+2)*self.human_num

    def forward(self, x):
        # print(x.shape)
        hs = []
        for i in range(self.robot_num):
            robot_state = x[...,i*self.robot_obs_shape
                                :i*self.robot_obs_shape+self.robot_state_shape]
            robot_scan = x[...,i*self.robot_obs_shape+self.robot_state_shape
                                :i*self.robot_obs_shape
                                    +self.robot_state_shape
                                    +self.robot_scan_shape
                                    *self.input_channel]#.view(-1,self.input_channel,self.robot_scan_shape)
            hs.append(
                self.state_encoder(torch.cat([robot_state,
                                              self.scan_encoder(robot_scan)]
                                        , dim=-1)
                                )
                    )
        
        for i in range(self.human_num):
            human_state = x[...,
                            i*self.human_obs_shape+self.robot_obs_shape*self.robot_num
                            :i*self.human_obs_shape+self.robot_obs_shape*self.robot_num+self.human_state_shape]
            human_scan = x[...,
                            i*self.human_obs_shape+self.robot_obs_shape*self.robot_num+self.human_state_shape\
                            :i*self.human_obs_shape
                                +self.robot_obs_shape
                                *self.robot_num
                                +self.human_state_shape
                                +self.human_scan_shape*self.input_channel]
            # # .view(
            #                         -1,self.input_channel,self.human_scan_shape
            #                         ) 
            hs.append(self.human_state_encoder(torch.cat([human_state,self.human_scan_encoder(human_scan)],dim=-1)))
            if self.use_discriminator:
                hs.append(
                    x[...,
                    i*self.human_obs_shape+self.robot_obs_shape*self.robot_num+self.human_state_shape+self.human_scan_shape\
                    :(i+1)*self.human_obs_shape+self.robot_obs_shape*self.robot_num] 
                )
            # print(hs[-1].shape,hs[-2].shape)

        global_state = x[...,-2*(self.human_num+self.robot_num):]
        hs.append(global_state)
        h = torch.cat(hs, dim=-1)
        return h
    

class CustomDecVNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(CustomDecVNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(self.initialization_method)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = DecRobotCrowdBase
        self.base = base(args, obs_shape)

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

class CustomDoubleDecVNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) decentralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(CustomDoubleDecVNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(self.initialization_method)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = DecRobotCrowdBase
        self.base = base(args, obs_shape)

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
      
class CustomVNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
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

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if not self.use_vis_aware:
            base = RobotCrowdBase
        else:
            base = VisRobotCrowdBase
        self.base = base(args, cent_obs_shape)

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

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
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

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = RobotCrowdBase
        self.base = base(args, cent_obs_shape)

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
