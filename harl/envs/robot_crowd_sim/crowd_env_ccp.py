import gym.spaces
import numpy as np
import typing as tp
import copy
import gym
import time
from harl.envs.robot_crowd_sim.utils.agent import Agent
from harl.envs.robot_crowd_sim.utils.map import Map
from harl.envs.robot_crowd_sim.utils.sensor import LiDAR
from harl.envs.robot_crowd_sim.utils.render import Render
from harl.envs.robot_crowd_sim.utils.spawner import CircleSpawner,RectangleSpawner,Room361Spawner
from harl.envs.robot_crowd_sim.utils.action import ActionVW,ActionXY
from harl.envs.robot_crowd_sim.utils.info import *
from harl.envs.robot_crowd_sim.crowd_env import RobotCrowdSim
# macro
EPISODE_TIME_STATE_DIM = 2

class RobotCrowdSimCCP(RobotCrowdSim):

    def __init__(self,
                 args,
                 phase:str,
                 nenv:int=1,
                 thisSeed:int=0,
                 ) -> None:
        super(RobotCrowdSimCCP,self).__init__(args,phase,nenv,thisSeed)
        self.wca_max = 3.5/1.25
        self.wca_min = (0.5+3.5)/2.5
        self.wg_max = 1.8
        self.wg_min = 0.1
        self.goal_weight = self.wg_min
        self.collision_weight =self.wca_max
        self._reward_goal = 1.0
        self._penalty_collision = -0.5
        self._penalty_backward = -0.00025/0.04*0.25
        self._goal_factor = 0.00075/0.04*0.25
        self._penalty_living = -0.00015/0.04*0.25
        self.reward_weight_changing_countdown = 100
        self.current_time = time.time()
        return
    
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
        t = time.time()
        self.reward_weight_changing_countdown -= (t-self.current_time)
        self.current_time = t
        if preference:
            for i, agent_id in enumerate(self.humans):
                self._crowd_preference[agent_id] = preference[i]
        elif preference is None and self.reward_weight_changing_countdown<0:
            normalized_weights = self._initHumanPreferenceVector()
            self.reward_weight_changing_countdown = 100
            for i, agent_id in enumerate(self.humans):
                self._crowd_preference[agent_id] = normalized_weights
        else:
            normalized_weights = np.array([(self.goal_weight-self.wg_min)/(self.wg_max-self.wg_min),
                            (self.collision_weight-self.wca_min)/(self.wca_max-self.wca_min)])
            for i, agent_id in enumerate(self.humans):
                self._crowd_preference[agent_id] = normalized_weights
        
        # setup episode 
        self._num_episode_steps = 0
        self.global_time = 0
        self.case_counter[self.phase] = (self.case_counter[self.phase] + int(1*self.nenv)) % self.case_size[self.phase]
        
        obs, share_obs = self._genObservation()

        self._render.reset()

        return obs, share_obs,available_actions

    def _initHumanPreferenceVector(self,preference=None):
        assert self._human_preference_type == "ccp"
        assert self._human_preference_vector_dim == 2
        #TODO set reward weights from outside    
        # Create a one-hot vector
        self.goal_weight = np.random.uniform(self.wg_min,self.wg_max)
        self.collision_weight = np.random.uniform(self.wca_min,self.wca_max)
        task = np.array([(self.goal_weight-self.wg_min)/(self.wg_max-self.wg_min),
                            (self.collision_weight-self.wca_min)/(self.wca_max-self.wca_min)])
            
        return task
    
    def _calAgentReward(self,agent:Agent):
        if agent.task_done:
            return 0.,True,{"episode_info":Nothing(),"bad_transition":False}
        collision,min_dist = self._checkAgentCollision(agent)
        truncation = False
        goal_weight = self._crowd_preference[agent.id][0]*(self.wg_max-self.wg_min)+self.wg_min
        collision_weight = self._crowd_preference[agent.id][1]*(self.wca_max-self.wca_min)+self.wca_min
        if collision:
            reward = self._penalty_collision*collision_weight
            done = True
            episode_info = Collision()
            agent.task_done = True
        elif agent.dg<self._goal_range:
            reward = self._reward_goal*self.goal_weight #* (1-self._num_episode_steps/self._max_episode_length)
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
            if agent.prev_dg>agent.dg and (agent.hf<np.pi/4 and agent.hf>-np.pi/4):
                reward = self._goal_factor+self._penalty_living
            else:
                reward = self._penalty_backward+self._penalty_living
            done = False
            episode_info = Nothing()
            if min_dist<self._discomfort_dist and agent.id in self.robots: 
                reward += self._discomfort_penalty_factor*(min_dist-self._discomfort_dist)
                episode_info = Discomfort(min_dist)
            reward *= goal_weight
        return reward, done,{"episode_info":episode_info,"bad_transition":truncation}