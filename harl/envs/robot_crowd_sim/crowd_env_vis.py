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
        # self._distracted_max_sensing_range = args.get("distracted_max_sensing_range",1.0)
        self._distracted_max_w = args.get("distracted_max_w",None)
        self._distracted_max_v = args.get("distracted_max_v",None)
        self._distracted_human_minimum_separation = args.get("distracted_human_shooting_range",1.0)
        print(self._distracted_human_minimum_separation)
        self._distracted_human_separation_penalty_factor = args.get("distracted_human_separation_penalty_factor",0.5)
        # assert self._distracted_human_shooting_range>=self._distracted_max_sensing_range
        super(RobotCrowdSimVis,self).__init__(args,phase,nenv,thisSeed)
        
        # add distracted human group
        # the sensing range of distracted people will be randomly shorten on each reset call

        # overlaod sensor (fov using masking on 360 degree sensing results)
        # for agent_id in self.distracted_humans:
        #     self.agent_sensors[agent_id].laser_angle_resolute = np.pi*2/self.n_laser

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


    def reset(self, seed: tp.Optional[int]=None, 
                    preference: tp.Optional[tp.List[int]]=None,
                    random_attribute_seed:tp.Optional[int]=None):

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

        self.current_distracted_human_num = np.random.randint(0, self._distracted_human_num+1)
        self.distracted_humans = [self._robot_num+i for i in range(self.current_distracted_human_num)]

        for agent_id in self.distracted_humans:
            # self.agents[agent_id].set_goal(self.agents[self.robots[0]].px,self.agents[self.robots[0]].py)
            if self._distracted_max_v is not None:
                self.agents[agent_id].v_pref = self._distracted_max_v
            if self._distracted_max_w is not None:
                self.agents[agent_id].rotation_constraint = self._distracted_max_w
        
        obs, share_obs = self._genObservation()

        self._render.reset(distracted_humans=self.distracted_humans)

        return obs, share_obs,available_actions

    def _get_agent_obs(self,agent_id):
        if self.agents[agent_id].task_done:
            if self.agents[agent_id].agent_type == "robot":
                return np.zeros(self.robot_obs_dim,dtype=np.float32)
            elif self.agents[agent_id].agent_type == "human":
                return np.zeros(self.human_obs_dim,dtype=np.float32)
            
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
                
            if attented:
                self.attented_values[agent_id][i] = 1
            else:
                self.attented_values[agent_id][i] = 0

        if self.vis_scan:
            self._render.add_scan_data(agent_id, self.current_scans[agent_id],
                                    self.attented_values[agent_id])

        return np.hstack([state,scan,self.attented_values[agent_id],prefer]) 
    
    def _checkAgentCollision(self,agent:Agent):
        collision = self._map.checkCollision(agent)
        dists = {}
        for other in self.agents:
            if agent.id == other.id: continue
            dist = np.linalg.norm([agent.px-other.px,agent.py-other.py])-agent.radius-other.radius
            dists[other.id] = dist
            if dist <0:
                collision = True
        return collision, dists
    
    def _calAgentReward(self,agent:Agent):
        if agent.task_done:
            return 0.,True,{"episode_info":Nothing(),"bad_transition":False}
        collision,dists = self._checkAgentCollision(agent)
        # Get the key with the minimum value
        min_id = min(dists, key=dists.get)

        # Get the minimum value
        min_dist = dists[min_id]
        truncation = False
        if collision:
            reward = self._penalty_collision
            done = True
            episode_info = Collision()
            agent.task_done = True
        elif agent.dg<self._goal_range:
            reward = self._reward_goal #* (1-self._num_episode_steps/self._max_episode_length)
            done = True
            episode_info = ReachGoal()
            agent.task_done = True
        elif self._num_episode_steps >= self._max_episode_length:
            reward = 0#-self._reward_goal
            done = True
            truncation = True
            episode_info = Timeout()
            agent.task_done = True
        else:
            reward = self._goal_factor*(agent.prev_dg-agent.dg)
            done = False
            episode_info = Nothing()
            if agent.id in self.robots: 
                #only robot use this reward
                danger = False
                danger_dists = []
                discomfort = False
                for other_id in dists:
                    other_dist = dists[other_id]
                    if other_id not in self.distracted_humans:
                        if other_dist<self._discomfort_dist:
                            reward += self._discomfort_penalty_factor*(other_dist-self._discomfort_dist)
                            discomfort=True
                    else:
                        if other_dist<self._distracted_human_minimum_separation:
                            reward += self._distracted_human_separation_penalty_factor*(other_dist-self._distracted_human_minimum_separation)
                            danger=True
                            danger_dists.append(other_dist)
                if danger:
                    episode_info = Danger(min(danger_dists))
                elif discomfort:
                    episode_info = Discomfort(min_dist)
        return reward, done,{"episode_info":episode_info,"bad_transition":truncation}
    