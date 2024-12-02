from sklearn import neighbors
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

class ConvEGNN3(nn.Module):
    """
        change log: modified NN structures
        see update in ver3
    """
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.hid_dim=hid_dim
        # self.cuda = cuda
        
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(in_dim*2+1, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.SiLU())
        
        # update position
        self.f_x = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1))
        
        # self.f_a = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, 1))
        
        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(in_dim+hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim))
    
    def forward(self, ped_features, h_st, h_neigh, rela_features, neigh_mask):
        """
            ped_features: [bs, N, 6] -> [bs, 2]
            h_st: [bs, N, dim_h] -> [bs, dim_h]
            h_neigh: [bs, N, k, dim_h] -> [bs, N-1, dim_h]
            rela_features: [bs, N, k, 6] -> [bs, N-1, 2]
            neigh_index: [bs, N, k] -> [bs, N-1]
        """
        # dists = torch.norm(rela_features[..., :2], dim=-1) # bs, N, k -> bs, N-1
        dists = torch.norm(rela_features, dim=-1) # bs, N, k -> bs, N-1
        neigh_num = neigh_mask.sum(dim=-1) # bs, N -> bs
        # print(h_neigh.shape, h_st.unsqueeze(-2).repeat(( 1, dists.shape[-1], 1)).shape, dists.unsqueeze(-1).shape)
        # compute messages
        # tmp = torch.cat([h_st.unsqueeze(-2).repeat((1, 1, dists.shape[-1], 1)), 
        #                     h_neigh, dists.unsqueeze(-1)], dim=-1) #bs, N, k, dim_h*2+1
        tmp = torch.cat([h_st.unsqueeze(-2).repeat(( 1, dists.shape[-1], 1)), 
                            h_neigh, dists.unsqueeze(-1)], dim=-1) #bs, N, k, dim_h*2+1 -> bs, N-1, dim_h*2+1
        m_ij = self.f_e(tmp) #bs, N, k, dim_h

        # update in ver3
        m_ij[~neigh_mask.bool()] = 0.
        
        # predict edges
        # agg = rela_features[..., :2] * self.f_x(m_ij) # bs, N, k, 2
        agg = rela_features * self.f_x(m_ij) # bs, N-1, 2
        # agg[~neigh_mask.bool()] = 0. # bs, N, k, 2 # deleted in ver3
        agg = 1/(neigh_num.unsqueeze(-1) + 1e-6) * agg.sum(dim=-2) # bs, N, 2 -> bs, 2
        # a_new = self.f_a(h_st) * ped_features[...,4: ] + agg
        # v_new = ped_features[...,2:4] + a_new
        # x_new = ped_features[..., :2] + v_new # bs, N, 2
        x_new = ped_features + agg # bs, N, 2 -> bs, 2
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = m_ij.sum(dim=-2) # bs, N, dim_h -> bs, dim_h
        
        # update hidden representations (with residual)
        h_st = h_st + self.f_h(torch.cat([h_st, m_i], dim=-1))

        #return torch.cat([x_new, v_new, a_new] ,dim=-1), h_st
        return x_new, h_st
    
# class NetEGNN_hid2(nn.Module, DATA.Pedestrians):
#     def __init__(self, in_dim=3+8+8, hid_dim=64, out_dim=1, n_layers=3, cuda=True):
#         super().__init__()
#         self.hid_dim=hid_dim
#         self.k_dim = 3
#         self.encode_v = nn.Linear(1, 8) 
#         self.encode_a = nn.Linear(1, 8) 
#         self.emb = nn.Linear(in_dim, hid_dim) 

#         self.gnn = nn.ModuleList(ConvEGNN3(hid_dim, hid_dim, cuda=cuda) for _ in range(n_layers))
#         # self.gnn = nn.Sequential(*self.gnn)
        
#         # self.pre_mlp = nn.Sequential(
#         #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
#         #     nn.Linear(hid_dim, hid_dim))
        
#         # self.post_mlp = nn.Sequential(
#         #     nn.Dropout(0.4),
#         #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
#         #     nn.Linear(hid_dim, out_dim))

#         if cuda: self.cuda()
#         self.cuda = cuda
    
#     def forward(self, context, k_emb):
#         """
#             context: (list), 
#                 [0]ped_features (bs, N, 6), 
#                 [1]neigh_mask (bs, N, k), 
#                 [2]neigh_index (bs, N, k)
#             k_emb: [bs, N, 3]
#         """
#         ped_features = context[0]
#         neigh_mask = context[1]
#         neigh_index = context[2]
        
#         # print('before gnn velo',ped_features[...,2:4].isnan().sum())
#         # print('before gnn acce',ped_features[...,4: ].isnan().sum())
#         # print('before gnn kemb',k_emb.isnan().sum())

        
#         h_initial = torch.cat((self.encode_v(torch.norm(ped_features[...,2:4],dim=-1, keepdim=True)),
#                                self.encode_a(torch.norm(ped_features[...,4: ],dim=-1, keepdim=True)),
#                                k_emb), dim=-1) # bs, N, 19 
#         # print('before gnn h_initial',h_initial.isnan().sum())
#         h_st = self.emb(h_initial) # bs, N, dim_h
#         # print('emb layer weight', list(self.emb.parameters())[0].isnan().sum())
#         # print('emb layer bias', list(self.emb.parameters())[1].isnan().sum())
#         # print('before gnn h_st',h_st.isnan().sum())
#         h_neigh = torch.zeros(h_st.shape[:2]+neigh_index.shape[-1:]+h_st.shape[-1:])
#         acce = ped_features[..., 4:].clone()

#         for i, model in enumerate(self.gnn):
#             # h_neigh = torch.zeros(neigh_index.shape+[h_st.shape[-1]]) # bs, N, k, dim_h
#             # relative_features = torch.zeros(neigh_index.shape+[ped_features.shape[-1]]) # bs, N, k, 6
#             # print('layer:',i)

#             # print('h_st',h_st.isnan().sum())

#             h_neigh = torch.gather(h_st.unsqueeze(1).expand(h_st.shape[:2]+h_st.shape[1:]), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
#             # h_neigh = torch.gather(h_st.unsqueeze(1).repeat(1,h_st.shape[1],1,1), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h


#             relative_features = self.get_relative_quantity(ped_features, ped_features) # bs, N, N, 6
#             dim = neigh_index.dim()
#             neigh_index2 = neigh_index.unsqueeze(-1).repeat(*([1]*dim + [relative_features.shape[-1]]))  # bs,n,k,6
#             relative_features = torch.gather(relative_features, -2, neigh_index2)  # bs,n,k,6
            
#             h_neigh[~neigh_mask.bool()]=0.
#             relative_features[~neigh_mask.bool()]=0.
#             # print('h_neigh',h_neigh.isnan().sum())
#             #print('relative_features',relative_features.isnan().sum())
            
#             #ped_features, h_st = model(ped_features, h_st, h_neigh, relative_features, neigh_mask)
#             ped_features, h_st = model(torch.cat((ped_features[...,:4], acce), dim=-1), h_st, h_neigh, relative_features, neigh_mask)
#             #print('acce',ped_features[...,4:].isnan().sum())
#             #print('h_st',h_st.isnan().sum())

#         # output = ped_features[...,4:]
#         output = h_st

#         return output # bs, N, 6; bs, N, dim_hd
    
class EGCLRobotCrowdBase(nn.Module):

    def __init__(self,args,cent_obs_shape):
        super(EGCLRobotCrowdBase,self).__init__()
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

        # self.scan_encoder=ScanEncoder(self.robot_scan_shape,args)
        # self.state_encoder=StateEncoder(self.scan_encoder.repr_dim+self.robot_state_shape,args)

        self.human_scan_encoder=ScanEncoder(self.human_scan_shape,args)
        self.human_state_encoder=StateEncoder(self.human_scan_encoder.repr_dim+self.human_state_shape,args)

        # if self.use_discriminator:
        #     self.repr_dim = (self.state_encoder.repr_dim+2)*self.robot_num
        #     self.repr_dim += (self.human_state_encoder.repr_dim+self.human_preference_dim+2)*self.human_num
        # else:
        #     self.repr_dim = (self.state_encoder.repr_dim+2)*self.robot_num
        #     self.repr_dim += (self.human_state_encoder.repr_dim+2)*self.human_num
        if self.use_discriminator:
            self.gnn = ConvEGNN3(self.human_state_encoder.repr_dim+self.human_preference_dim, self.human_state_encoder.repr_dim+self.human_preference_dim)
            self.repr_dim = self.human_state_encoder.repr_dim+self.human_preference_dim
        else:
            self.gnn = ConvEGNN3(self.human_state_encoder.repr_dim, self.human_state_encoder.repr_dim)
            self.repr_dim = self.human_state_encoder.repr_dim
            
    def forward(self, x):

        human_obs = x[...,:self.human_num*self.human_obs_shape].reshape(*x.shape[:-1],self.human_num,self.human_obs_shape)
        human_states = human_obs[...,:self.human_state_shape].reshape(-1,self.human_state_shape)
        human_scans = human_obs[...,self.human_state_shape:self.human_state_shape+self.human_scan_shape].reshape(-1,self.human_scan_shape)
        human_features = self.human_state_encoder(torch.cat([human_states,
                                              self.human_scan_encoder(human_scans)]
                                        , dim=-1)
                                )
        # if self.use_discriminator:
        #     human_prefs = human_obs[...,-self.human_preference_dim:].reshape(-1,self.human_preference_dim)
        #     human_features = torch.cat([human_features,human_prefs])
        human_features = human_features.reshape(*x.shape[:-1],self.human_num,-1)
        if self.use_discriminator:
            human_prefs = human_obs[...,-self.human_preference_dim:]
            human_features = torch.cat([human_features,human_prefs],dim=-1)
        global_obs = x[...,-2*self.human_num:].reshape(*x.shape[:-1],self.human_num,2)
        neigh_mask = global_obs[...,1:,0]!=999
        self_position = global_obs[...,0,:]
        relative_position = global_obs[...,1:,:]-self_position.unsqueeze(-2)
        
        self_position, h_st = self.gnn(self_position, human_features[...,0,:], human_features[...,1:,:], relative_position, neigh_mask)

        
        
        return h_st
    
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
