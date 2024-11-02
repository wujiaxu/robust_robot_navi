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

class RobotCrowdSimVis(RobotCrowdSim):
    
    def __init__(self,
                 args,
                 phase:str,
                 nenv:int=1,
                 thisSeed:int=0,
                 vis_scan:bool=False
                 ) -> None:
        self._distracted_human_num = args.get("distracted_human_num",1)
        self._distracted_max_sensing_range = args.get("distracted_max_sensing_range",1.0)
        self._distracted_max_w = args.get("distracted_max_w",None)
        self._distracted_max_v = args.get("distracted_max_v",None)
        self._distracted_human_shooting_range = args.get("distracted_human_shooting_range",1.5)
        assert self._distracted_human_shooting_range>=self._distracted_max_sensing_range
        super(RobotCrowdSimVis,self).__init__(args,phase,nenv,thisSeed)
        
        # add distracted human group
        # the sensing range of distracted people will be randomly shorten on each reset call
        self.distracted_humans = [self._robot_num+i for i in range(self._distracted_human_num)]

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


    def reset(self, seed: tp.Optional[int]=None, 
                    preference: tp.Optional[tp.List[int]]=None,
                    random_attribute_seed:tp.Optional[int]=None):
        
        # for agent_id in self.distracted_humans:
            # self.agent_sensors[agent_id].laser_max_range \
            #     = self._distracted_human_shooting_range #sensing range for robot 
            # self.agent_sensors[agent_id].distracted_range \
            #         = np.random.uniform(self.agents[agent_id].radius+0.3,
            #                         self._distracted_max_sensing_range) #sensing range for human
            # self.agents[agent_id].rotation_constraint = self._distracted_max_w

        available_actions = None
        self._num_episode_steps = 0
        train_seed_begin = [0, 10, 100, 1000, 10000]
        val_seed_begin = [0, 10, 100, 1000, 10000]
        test_seed_begin = [0, 10, 100, 1000, 10000]
        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'] + train_seed_begin[1],
                     'val': 0 + val_seed_begin[1], 'test': self.case_capacity['val']+test_seed_begin[2]+1000}

        if self.phase == "test" and seed is not None:
            self.random_seed = seed+base_seed[self.phase]
        else:
            self.random_seed = base_seed[self.phase] + self.case_counter[self.phase] + self.thisSeed
        np.random.seed(self.random_seed)
        
        # assert self._robot_num == 1 #TODO multiple robots
        if self.phase == "test" and seed is not None:
            if seed == -1:
                self.agents[0].set(0, -(self._map_size/2-1), 0, (self._map_size/2-1), 0, 0, np.pi/2)
                self.agents[1].set(3, 0.1, -3, -0, 0,0, np.pi)
                self.agents[2].set(-3, -0.1, 3, -0, 0,0, -np.pi)
            else:
                for agent in self.agents:
                    self._spawner.spawnAgent(agent)
                    agent.task_done = False
        else:
            for agent in self.agents:
                self._spawner.spawnAgent(agent)
                agent.task_done = False

        # randomize agent attributes
        if self.phase == "test" and random_attribute_seed is not None:
            np.random.seed(random_attribute_seed)
            if self.robot_random_pref_v_and_size:
                for agent_id in self.robots:
                    self.agents[agent_id].sample_random_attributes()
            if self.human_random_pref_v_and_size:
                for agent_id in self.humans:
                    self.agents[agent_id].sample_random_attributes()
            np.random.seed(self.random_seed)
        else:
            if self.robot_random_pref_v_and_size:
                for agent_id in self.robots:
                    self.agents[agent_id].sample_random_attributes()
            if self.human_random_pref_v_and_size:
                for agent_id in self.humans:
                    self.agents[agent_id].sample_random_attributes()

        # randomize crowd preference vector
        for i, agent_id in enumerate(self.humans):
            self._crowd_preference[agent_id] = self._initHumanPreferenceVector(preference[i] if preference != None else None)
        # setup episode 
        self._num_episode_steps = 0
        self.global_time = 0
        self.case_counter[self.phase] = (self.case_counter[self.phase] + int(1*self.nenv)) % self.case_size[self.phase]

        for agent_id in self.distracted_humans:
            self.agents[agent_id].set_goal(self.agents[self.robots[0]].px,self.agents[self.robots[0]].py)
            if self._distracted_max_v is not None:
                self.agents[agent_id].v_pref = self._distracted_max_v
            if self._distracted_max_w is not None:
                self.agents[agent_id].rotation_constraint = self._distracted_max_w
        
        obs, share_obs = self._genObservation()

        self._render.reset()

        return obs, share_obs,available_actions
    
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
    
    def _calAgentReward(self, agent: Agent):
        if agent.id in self.distracted_humans:
            if agent.task_done:
                return 0.,True,{"episode_info":Nothing(),"bad_transition":False}
            collision,min_dist = self._checkAgentCollision(agent)
            truncation = False
            if agent.dg<self.agents[self.robots[0]].radius:
                reward = self._reward_goal #* (1-self._num_episode_steps/self._max_episode_length)
                done = True
                episode_info = ReachGoal()
                agent.task_done = True
                collision = False
            elif collision:
                reward = self._penalty_collision
                done = True
                episode_info = Collision()
                agent.task_done = True
            elif self._num_episode_steps >= self._max_episode_length:
                reward = 0#-self._reward_goal
                done = True
                truncation = True
                episode_info = Timeout()
                agent.task_done = True
            else:
                reward = 10*(agent.prev_dg-agent.dg)
                done = False
                episode_info = Nothing()
            return reward, done,{"episode_info":episode_info,"bad_transition":truncation}
            
        return super()._calAgentReward(agent)
    