from sklearn import neighbors
import torch
import torch.nn as nn
from harl.models.base.custom import ScanEncoder,StateEncoder

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
            ped_features: [bs, N, 6]
            h_st: [bs, N, dim_h]
            h_neigh: [bs, N, k, dim_h]
            rela_features: [bs, N, k, 6]
            neigh_index: [bs, N, k]
        """
        dists = torch.norm(rela_features[..., :2], dim=-1) # bs, N, k
        neigh_num = neigh_mask.sum(dim=-1) # bs, N
        
        # compute messages
        tmp = torch.cat([h_st.unsqueeze(-2).repeat((1, 1, dists.shape[-1], 1)), 
                            h_neigh, dists.unsqueeze(-1)], dim=-1) #bs, N, k, dim_h*2+1
        m_ij = self.f_e(tmp) #bs, N, k, dim_h

        # update in ver3
        m_ij[~neigh_mask.bool()] = 0.
        
        # predict edges
        agg = rela_features[..., :2] * self.f_x(m_ij) # bs, N, k, 2
        # agg[~neigh_mask.bool()] = 0. # bs, N, k, 2 # deleted in ver3
        agg = 1/(neigh_num.unsqueeze(-1) + 1e-6) * agg.sum(dim=-2) # bs, N, 2
        x_new = ped_features + agg
        # a_new = self.f_a(h_st) * ped_features[...,4: ] + agg
        # v_new = ped_features[...,2:4] + a_new
        # x_new = ped_features[..., :2] + v_new # bs, N, 2
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = m_ij.sum(dim=-2) # bs, N, dim_h
        
        # update hidden representations (with residual)
        h_st = h_st + self.f_h(torch.cat([h_st, m_i], dim=-1))

        return x_new, h_st

class EGCLCrowdBase(nn.Module): #TODO
    # for crowd only and need ep buffer
    def __init__(self,args,centralized=True):
        super(EGCLCrowdBase,self).__init__()
        self.human_scan_shape = 11
        self.human_state_shape = 6
        n_layers = 3
        self.human_num = args["human_num"]
        self.human_preference_dim = args["human_preference_vector_dim"]
        self.use_discriminator = args["use_discriminator"]

        self.human_obs_shape = self.human_scan_shape + self.human_state_shape + self.human_preference_dim

        # self.human_scan_encoder=ScanEncoder(self.human_scan_shape,args)
        self.human_state_encoder=StateEncoder(self.human_scan_shape+self.human_state_shape,
                                              args,
                                              hidden_size=128,
                                              num_layer=2)

        if self.use_discriminator:
            self.gnn = nn.ModuleList(
                            ConvEGNN3(self.human_state_encoder.repr_dim+self.human_preference_dim, 
                                        self.human_state_encoder.repr_dim+self.human_preference_dim) 
                                 for _ in range(n_layers))
            self.repr_dim = self.human_state_encoder.repr_dim+self.human_preference_dim
        else:
            self.gnn = nn.ModuleList(
                            ConvEGNN3(self.human_state_encoder.repr_dim, 
                                      self.human_state_encoder.repr_dim) 
                            for _ in range(n_layers))
            self.repr_dim = self.human_state_encoder.repr_dim

    @staticmethod
    def get_relative_quantity(A, B):
        """
        Function:
            The relative amount among all objects in A and all objects in B at
            each moment get relative vector xj - xi fof each pedestrain i (B - A)
        Args:
            A: (*c, N, dim)
            B: (*c, M, dim)
        Return:
            relative_A: (*c, t, N, M, dim)
        """
        dim = A.dim()
        A = A.unsqueeze(-2).repeat(*([1]*(dim-1) + [B.shape[-2]] + [1]))  # *c, t, N, M, dim
        B = B.unsqueeze(-3).repeat(*([1]*(dim-2) + [A.shape[-3]] + [1, 1]))
        relative_A = B - A

        return relative_A.contiguous()
    
    def get_nearby_obj(self, position, objects, k):
        """
        Function: get The k closest people's index at time t:
            Calculate the relative position between every two people, and then look
            at the angle between this relative position and the speed direction of the
            current person. If the angle is within the threshold range, then it is
            judged to be in the field of view.
        Args:
            k: get the nearest k persons
            position: (*c, N, 2)
            objects: (*c, M, 2)
        Return:
            neighbor_index: (*c, N, k), The k closest objects' index at time t
        """

        relative_pos = self.get_relative_quantity(position, objects)  # *c,t,N,M,2
        relative_pos[relative_pos.isnan()] = float('inf')
        distance = torch.norm(relative_pos, p=2, dim=-1)  # *c,t,N,M

        sorted_dist, indices = torch.sort(distance, dim=-1)

        return sorted_dist[..., :k], indices[..., :k]
    
    def get_filtered_features(self, features, nearby_idx, nearby_dist, dist_threshold):
        """
        features: (*c, N, M, dim)
        nearby_idx: (*c, N, k)
        nearby_dist: (*c, N, k)
        """
        dim = nearby_idx.dim()
        nearby_idx = nearby_idx.unsqueeze(-1).repeat(*([1]*dim + [features.shape[-1]]))  # t,n,k,dim
        features = torch.gather(features, -2, nearby_idx)  # t,n,k,dim

        dist_filter = torch.ones(features.shape, device=features.device)
        nearby_dist = nearby_dist.unsqueeze(-1).repeat(*([1]*dim + [features.shape[-1]]))
        dist_filter[nearby_dist > dist_threshold] = 0
        features[dist_filter == 0] = 0  # nearest neighbor less than k --> zero padding

        return features, dist_filter[...,0]
    
    def get_relative_features(
            self, position,
            topk_ped=6,dist_threshold_ped=4
    ):
        """
            position: bs, N, 2
           
        Return:
            
        """

        near_ped_dist, near_ped_idx = self.get_nearby_obj(
            position, position,topk_ped) # top_k=6
        ped_features = self.get_relative_quantity(position, position)  # *c t N N dim
        ped_features, neigh_ped_mask = self.get_filtered_features(
            ped_features, near_ped_idx, near_ped_dist, dist_threshold_ped)

        return ped_features, near_ped_idx, neigh_ped_mask
            
    def forward(self, x):
        out_of_bound = torch.tensor(float('nan'),device=x.device)
        human_obs = x[...,:self.human_num*self.human_obs_shape].reshape(*x.shape[:-1],self.human_num,self.human_obs_shape)
        human_states = human_obs[...,:self.human_state_shape].reshape(-1,self.human_state_shape)
        human_scans = human_obs[...,self.human_state_shape:self.human_state_shape+self.human_scan_shape].reshape(-1,self.human_scan_shape)
        h_st = self.human_state_encoder(torch.cat([human_states,
                                              human_scans]
                                        , dim=-1)
                                )
        h_st = h_st.reshape(*x.shape[:-1],self.human_num,-1) # bs, N, dim_h
        if self.use_discriminator:
            human_prefs = human_obs[...,-self.human_preference_dim:]
            h_st = torch.cat([h_st,human_prefs],dim=-1) # bs, N, dim_h
        p_mask = torch.any(human_obs[...,:self.human_state_shape]!=0,-1) # bs, N
        p_mask_ = p_mask.unsqueeze(-1).repeat(1,1,2)
        ped_features = x[...,-2*self.human_num:].reshape(*x.shape[:-1],self.human_num,2) #(bs,N,2)
        ped_features[p_mask_==0] = out_of_bound
        (_, 
         neigh_index, 
         neigh_mask
         ) = self.get_relative_features(ped_features)
        

        # h_neigh = torch.zeros(h_st.shape[:2]+(self.human_num,)+h_st.shape[-1:]) # bs, N, N, dim_h
        
        for i, model in enumerate(self.gnn):
            h_neigh = torch.gather(
                h_st.unsqueeze(1).expand(
                    h_st.shape[:2]+h_st.shape[1:]
                    ), 
                2, 
                neigh_index.unsqueeze(-1).repeat((1,1,1,self.repr_dim)))

            relative_features = self.get_relative_quantity(ped_features, ped_features) # bs, N, N, 2
            dim = neigh_index.dim()
            neigh_index2 = neigh_index.unsqueeze(-1).repeat(*([1]*dim + [relative_features.shape[-1]]))  # bs,n,k,6
            relative_features = torch.gather(relative_features, -2, neigh_index2)  # bs,n,k,2
            
            h_neigh[~neigh_mask.bool()]=0.
            relative_features[~neigh_mask.bool()]=0.
            # print('h_neigh',h_neigh.isnan().sum())
            #print('relative_features',relative_features.isnan().sum())
            
            #ped_features, h_st = model(ped_features, h_st, h_neigh, relative_features, neigh_mask)
            ped_features, h_st = model(ped_features, h_st, h_neigh, relative_features, neigh_mask)
            #print('acce',ped_features[...,4:].isnan().sum())
            #print('h_st',h_st.isnan().sum())

        
        
        return h_st.reshape(-1,self.repr_dim)

class MLPCrowdBase(nn.Module):
    # for FP only
    def __init__(self,args,centralized):
        super(MLPCrowdBase,self).__init__()
        assert "down_sample_lidar_scan_bin" in args.keys()
        self.human_scan_shape = args["down_sample_lidar_scan_bin"]
        self.human_state_shape = 6
        self.centralized = centralized
        if centralized:
            self.human_num = args["human_num"]
        else:
            self.human_num = 1
        self.human_preference_dim = args["human_preference_vector_dim"]
        self.use_discriminator = args["use_discriminator"]

        self.human_obs_shape = self.human_scan_shape + self.human_state_shape + self.human_preference_dim

        # self.human_scan_encoder=ScanEncoder(self.human_scan_shape,args)
        self.human_state_encoder=StateEncoder(self.human_scan_shape+self.human_state_shape,
                                              args,
                                              hidden_size=128,
                                              num_layer=2)
        
        if centralized:
            if self.use_discriminator:
                self.repr_dim = (self.human_state_encoder.repr_dim+self.human_preference_dim+2)*self.human_num
            else:
                self.repr_dim = (self.human_state_encoder.repr_dim+2)*self.human_num
        else:
            if self.use_discriminator:
                self.repr_dim = (self.human_state_encoder.repr_dim+self.human_preference_dim)*self.human_num#=1
            else:
                self.repr_dim = (self.human_state_encoder.repr_dim)*self.human_num#=1

    def forward(self, x):
        # print(x.shape)
        human_obs = x[...,:self.human_num*self.human_obs_shape].reshape(*x.shape[:-1],self.human_num,self.human_obs_shape)
        human_state_scans = human_obs[...,:self.human_state_shape+self.human_scan_shape].reshape(-1,self.human_state_shape+self.human_scan_shape)
        human_features = self.human_state_encoder(human_state_scans)
        human_features = human_features.reshape(*x.shape[:-1],self.human_num,-1)
        if self.use_discriminator:
            human_prefs = human_obs[...,-self.human_preference_dim:]
            human_features = torch.cat([human_features,human_prefs],dim=-1)
            human_features = human_features.reshape(*x.shape[:-1],self.human_num*(self.human_state_encoder.repr_dim+self.human_preference_dim))
        else:
            human_features = human_features.reshape(*x.shape[:-1],self.human_num*self.human_state_encoder.repr_dim)
        
        if self.centralized:
            global_state = x[...,-2*(self.agent_num):]
            h = torch.cat([human_features,global_state], dim=-1)
            return h
        else:
            h = human_features
            return h
        
class RobotCrowdBase(nn.Module):
    # for FP and 720 bin lidar only
    def __init__(self,args,centralized):
        super(RobotCrowdBase,self).__init__()
        self.scan_shape = 720
        self.state_shape = 6
        self.centralized = centralized
        if centralized:
            self.agent_num = args["human_num"]
        else:
            self.agent_num = 1
        
        self.human_preference_dim = args["human_preference_vector_dim"]
        self.use_discriminator = args["use_discriminator"]

        self.obs_shape = self.scan_shape + self.state_shape + self.human_preference_dim

        self.scan_encoder=ScanEncoder(self.scan_shape,args)
        self.state_encoder=StateEncoder(self.scan_encoder.repr_dim+self.state_shape,args)

        if self.centralized:
            if self.use_discriminator:
                self.repr_dim = (self.state_encoder.repr_dim+self.human_preference_dim+2)*self.agent_num
            else:
                self.repr_dim = (self.state_encoder.repr_dim+2)*self.agent_num
        else:
            if self.use_discriminator:
                self.repr_dim = (self.state_encoder.repr_dim+self.human_preference_dim)*self.agent_num#=1
            else:
                self.repr_dim = (self.state_encoder.repr_dim)*self.agent_num#=1

    def forward(self, x):
        # print(x.shape)
        obs = x[...,:self.agent_num*self.obs_shape].reshape(*x.shape[:-1],self.agent_num,self.obs_shape)
        states = obs[...,:self.state_shape].reshape(-1,self.state_shape)
        scans = obs[...,self.state_shape:self.state_shape+self.scan_shape].reshape(-1,self.scan_shape)
        features = self.state_encoder(torch.cat([states,
                                              self.scan_encoder(scans)]
                                        , dim=-1)
                                ).reshape(*x.shape[:-1],self.agent_num,self.state_encoder.repr_dim)
        if self.use_discriminator:
            human_prefs = obs[...,-self.human_preference_dim:] #B,N,dim
            # print(features.shape,human_prefs.shape)
            features = torch.cat([features,human_prefs],dim=-1)
            features = features.reshape(*x.shape[:-1],self.agent_num*(self.state_encoder.repr_dim+self.human_preference_dim))
        else:
            features = features.reshape(*x.shape[:-1],self.agent_num*self.state_encoder.repr_dim)
        
        if self.centralized:
            global_state = x[...,-2*(self.agent_num):]
            h = torch.cat([features,global_state], dim=-1)
            return h
        else:
            h = features
            return h

class VisRobotBase(nn.Module): #TODO

    def __init__(self,args,centralized):
        super(VisRobotBase,self).__init__()

        self.robot_scan_shape = 720
        self.robot_state_shape = 6
        self.input_channel = 2

        self.scan_encoder=ScanEncoder(self.robot_scan_shape,args,input_channel=self.input_channel)
        self.state_encoder=StateEncoder(self.scan_encoder.repr_dim+self.robot_state_shape,args)
        self.repr_dim = self.state_encoder.repr_dim

    def forward(self, x):
        
        robot_state = x[...,:self.robot_state_shape]
        robot_scan = x[...,self.robot_state_shape:self.robot_state_shape
                           +self.robot_scan_shape*self.input_channel]#.view(-1,self.input_channel,self.robot_scan_shape)
        h = self.state_encoder(torch.cat([robot_state,self.scan_encoder(robot_scan)], dim=-1))

        return h
      
# class VisRobotCrowdBase(nn.Module):

#     def __init__(self,args,centralized):
#         super(VisRobotCrowdBase,self).__init__()
#         assert centralized==False
#         self.human_scan_shape = 720
#         self.human_state_shape = 6
#         self.robot_scan_shape = 720
#         self.robot_state_shape = 6
#         self.human_num = args["human_num"]
#         self.robot_num = args["robot_num"]
#         self.human_preference_dim = args["human_preference_vector_dim"]
#         self.use_discriminator = args["use_discriminator"]

#         self.input_channel = 2
#         self.robot_obs_shape = self.robot_scan_shape * self.input_channel+ self.robot_state_shape + self.human_preference_dim
#         self.human_obs_shape = self.human_scan_shape * self.input_channel+ self.human_state_shape + self.human_preference_dim
        
#         self.scan_encoder=ScanEncoder(self.robot_scan_shape,args,input_channel=self.input_channel)
#         self.state_encoder=StateEncoder(self.scan_encoder.repr_dim+self.robot_state_shape,args)

#         self.human_scan_encoder=ScanEncoder(self.human_scan_shape,args,input_channel=self.input_channel)
#         self.human_state_encoder=StateEncoder(self.human_scan_encoder.repr_dim+self.human_state_shape,args)
        
#         if self.use_discriminator:
#             self.repr_dim = (self.state_encoder.repr_dim+2)*self.robot_num
#             self.repr_dim += (self.human_state_encoder.repr_dim+self.human_preference_dim+2)*self.human_num
#         else:
#             self.repr_dim = (self.state_encoder.repr_dim+2)*self.robot_num
#             self.repr_dim += (self.human_state_encoder.repr_dim+2)*self.human_num

#     def forward(self, x):
#         # print(x.shape)
#         hs = []
#         for i in range(self.robot_num):
#             robot_state = x[...,i*self.robot_obs_shape
#                                 :i*self.robot_obs_shape+self.robot_state_shape]
#             robot_scan = x[...,i*self.robot_obs_shape+self.robot_state_shape
#                                 :i*self.robot_obs_shape
#                                     +self.robot_state_shape
#                                     +self.robot_scan_shape
#                                     *self.input_channel]#.view(-1,self.input_channel,self.robot_scan_shape)
#             hs.append(
#                 self.state_encoder(torch.cat([robot_state,
#                                               self.scan_encoder(robot_scan)]
#                                         , dim=-1)
#                                 )
#                     )
        
#         for i in range(self.human_num):
#             human_state = x[...,
#                             i*self.human_obs_shape+self.robot_obs_shape*self.robot_num
#                             :i*self.human_obs_shape+self.robot_obs_shape*self.robot_num+self.human_state_shape]
#             human_scan = x[...,
#                             i*self.human_obs_shape+self.robot_obs_shape*self.robot_num+self.human_state_shape\
#                             :i*self.human_obs_shape
#                                 +self.robot_obs_shape
#                                 *self.robot_num
#                                 +self.human_state_shape
#                                 +self.human_scan_shape*self.input_channel]
#             # # .view(
#             #                         -1,self.input_channel,self.human_scan_shape
#             #                         ) 
#             hs.append(self.human_state_encoder(torch.cat([human_state,self.human_scan_encoder(human_scan)],dim=-1)))
#             if self.use_discriminator:
#                 hs.append(
#                     x[...,
#                     i*self.human_obs_shape+self.robot_obs_shape*self.robot_num+self.human_state_shape+self.human_scan_shape\
#                     :(i+1)*self.human_obs_shape+self.robot_obs_shape*self.robot_num] 
#                 )
#             # print(hs[-1].shape,hs[-2].shape)

#         global_state = x[...,-2*(self.human_num+self.robot_num):]
#         hs.append(global_state)
#         h = torch.cat(hs, dim=-1)
#         return h
    