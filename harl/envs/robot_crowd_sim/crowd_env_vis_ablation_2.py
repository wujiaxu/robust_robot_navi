import gym.spaces
from harl.envs.robot_crowd_sim.utils.agent import Agent
import numpy as np
import typing as tp
import copy
import gym
from harl.envs.robot_crowd_sim.crowd_env import RobotCrowdSim
from harl.envs.robot_crowd_sim.utils.info import *
# macro
EPISODE_TIME_STATE_DIM = 2

class RobotCrowdSimVis(RobotCrowdSim): #ablation on adversarial agent
    
    def __init__(self,
                 args,
                 phase:str,
                 nenv:int=1,
                 thisSeed:int=0,
                 vis_scan:bool=False
                 ) -> None:
        super(RobotCrowdSimVis,self).__init__(args,phase,nenv,thisSeed)
        
        # add distracted human group
        # the sensing range of distracted people will be randomly shorten on each reset call
        self.distracted_humans = []

        # overlaod sensor (fov using masking on 360 degree sensing results)
        for agent_id in self.distracted_humans:
            self.agent_sensors[agent_id].laser_angle_resolute = np.pi*2/self.n_laser

        # overload human observation space
        self.robot_obs_dim = self.n_laser*2+6+self.discriminator_dim
        self.human_obs_dim = self.n_laser*2+6+self.discriminator_dim
        self.padded_obs_dim = max(self.robot_obs_dim,self.human_obs_dim) # TODO delete padding
        self.robot_obs_space = gym.spaces.Box(-np.inf,np.inf,(self.padded_obs_dim,))
        self.human_obs_sapce = gym.spaces.Box(-np.inf,np.inf,(self.padded_obs_dim,))
        self.observation_space = [self.robot_obs_space]*self._robot_num+[self.human_obs_sapce]*self._human_num
        self.share_observation_space = [gym.spaces.Box(-np.inf,np.inf,
                                                       ((self.robot_obs_dim+2)*self._robot_num\
                                                        +(self.human_obs_dim+2)*self._human_num,))
                                        ]*(self.n_agents)
        
        # add scan for render
        self.vis_scan = vis_scan
        self.current_scans = {}
        self.attented_values = {}
    
    # def reset(self, seed: tp.Optional[int]=None, 
    #                 preference: tp.Optional[tp.List[int]]=None,
    #                 random_attribute_seed:tp.Optional[int]=None):
    #     for agent_id in self.distracted_humans:
    #         self.agent_sensors[agent_id].laser_max_range \
    #             = self._distracted_human_shooting_range #sensing range for robot 
    #         self.agent_sensors[agent_id].distracted_range \
    #                 = np.random.uniform(self.agents[agent_id].radius+0.3,
    #                                 self._distracted_max_sensing_range) #sensing range for human
    #         self.agents[agent_id].rotation_constraint = self._distracted_max_w

    #     return super(RobotCrowdSimVis,self).reset(seed,preference,random_attribute_seed)

    
    # def step(self,actions):
    #     assert len(self.robots) <2
    #     # change goals of distracted people to robot position 
    #     # if robot inside their shotting range 
    #     if len(self.robots) == 1:
    #         for agent_id in self.distracted_humans:
                
    #             # self_position = np.array(self.agents[agent_id].get_position())
    #             # robot_position = np.array(self.agents[self.robots[0]].get_position())
    #             # if np.linalg.norm(self_position-robot_position)< self._distracted_human_shooting_range:
    #             self.agents[agent_id].set_goal(self.agents[self.robots[0]].px,
    #                                            self.agents[self.robots[0]].py)
                    
    #     return super(RobotCrowdSimVis,self).step(actions)

    def _get_agent_obs(self,agent_id):
        if self.agents[agent_id].task_done:
            if self.agents[agent_id].agent_type == "robot":
                return np.zeros(self.robot_obs_dim,dtype=np.float32)
            elif self.agents[agent_id].agent_type == "human":
                return np.zeros(self.human_obs_dim,dtype=np.float32)
        if agent_id in self.distracted_humans:
            self.agents[agent_id].set_goal(self.agents[self.robots[0]].px,
                                               self.agents[self.robots[0]].py)
        scan, scan_end = self.agent_sensors[agent_id].getScan()
        state = np.array(self.agents[agent_id].get_transformed_state(),
                          dtype=np.float32)
        prefer = np.zeros(self.discriminator_dim) if self.agents[agent_id].agent_type == "robot"\
                    else self._crowd_preference[agent_id]

        # self visibility awareness
        # need to aware human attention on the robot, and map to its scan directions
        # let use scan_end to judge whether a ray hit human
        self.current_scans[agent_id] = []
        self.attented_values[agent_id] = np.ones(scan_end.shape[0])*self.agent_sensors[agent_id].laser_max_range

        for i in range(scan_end.shape[0]):
            
            # masking by fov for normal human
            # if self.agents[agent_id].agent_type == "human" and agent_id not in self.distracted_humans: 
            #     in_fov = False
            #     self_half_fov = (self.human_fov/2)/180.*np.pi
            #     # check if ray in side fov
            #     ray_n = scan_end[i]-np.array(self.agents[agent_id].get_position())
            #     ray_n_cosin = ray_n.dot(
            #         np.array([np.cos(self.agents[agent_id].theta),
            #                 np.sin(self.agents[agent_id].theta)]) )/np.linalg.norm(ray_n)
            #     if ray_n_cosin>=np.cos(self_half_fov):
            #         in_fov = True

            #     if not in_fov: 
            #         scan[i] = 0
            #         self.attented_values[agent_id][i] = 0

            
            if self.agents[agent_id].agent_type == "human" and agent_id in self.distracted_humans:
                hit_robot = False
                # in_fov = False
                self.attented_values[agent_id][i] = 1

                # self_half_fov = (self.human_fov/2)/180.*np.pi
                # # check if ray in side fov
                # ray_n = scan_end[i]-np.array(self.agents[agent_id].get_position())
                # ray_n_cosin = ray_n.dot(
                #     np.array([np.cos(self.agents[agent_id].theta),
                #             np.sin(self.agents[agent_id].theta)]) )/np.linalg.norm(ray_n)
                # if ray_n_cosin>=np.cos(self_half_fov):
                #     in_fov = True

                #check if hitted robot
                for id in self.robots:
                    robot_position = np.array(self.agents[self.robots[id]].get_position())
                    if np.linalg.norm(scan_end[i]-robot_position) < self.agents[self.robots[id]].radius+0.01:
                        hit_robot = True
                        break

                if not hit_robot:
                    # scan[i] = max(scan[i],self.agent_sensors[agent_id].distracted_range)
                    # # modify scan_end
                    # ray_n = ray_n/np.linalg.norm(ray_n)*self.agent_sensors[agent_id].distracted_range
                    # scan_end[i] = ray_n + np.array(self.agents[agent_id].get_position())
                    self.attented_values[agent_id][i] = 0

                # if not in_fov: #and not hit_robot: 
                #     scan[i] = 0
                    
                
                        
            self.current_scans[agent_id].append(
                [(self.agents[agent_id].px, self.agents[agent_id].py), \
                    (scan_end[i,0],scan_end[i,1])]
            )

            if self.agents[agent_id].agent_type == "human":
                continue

            # attented value only for robot
            attented = False
            for other in self.agents:
                if other.id not in self.distracted_humans:continue
                other_center = np.array(other.get_position())
                dist = np.linalg.norm(other_center-scan_end[i])
                if dist < other.radius+0.05: 
                    # hitted = True
                    attented = True
                # else:
                #     hitted = False
                # other_to_self = np.array(self.agents[agent_id].get_position())-other_center
                # if self.agent_sensors[other.id].distracted_range is not None:
                #     if np.linalg.norm(other_to_self)-self.agents[agent_id].radius< self.agent_sensors[other.id].distracted_range:
                #         in_fov_r = True
                #     else:
                #         in_fov_r = False
                # else:
                #     if np.linalg.norm(other_to_self)-self.agents[agent_id].radius< self.agent_sensors[other.id].laser_max_range:
                #         in_fov_r = True
                #     else:
                #         in_fov_r = False
                
                # other_half_fov = (self.human_fov/2)/180.*np.pi
                # other_to_self_cosin = other_to_self.dot(
                #     np.array([np.cos(other.theta),np.sin(other.theta)])
                #     )/np.linalg.norm(other_to_self)
                # #TODO: check whether holomonic model update theta correctly
                # if other_to_self_cosin >= np.cos(other_half_fov):
                #     in_fov_a = True
                # else:
                #     in_fov_a = False
                # if hitted and in_fov_a and in_fov_r:
                #     attented = True
                #     break
            if attented:
                self.attented_values[agent_id][i] = 1
            else:
                self.attented_values[agent_id][i] = 0

        if self.vis_scan:
            self._render.add_scan_data(agent_id, self.current_scans[agent_id],
                                    self.attented_values[agent_id])

        return np.hstack([state,scan,self.attented_values[agent_id],prefer]) 

    