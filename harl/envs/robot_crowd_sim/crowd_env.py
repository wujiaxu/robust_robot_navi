import gym.spaces
import numpy as np
import typing as tp
import copy
import gym
from harl.envs.robot_crowd_sim.utils.agent import Agent
from harl.envs.robot_crowd_sim.utils.map import Map
from harl.envs.robot_crowd_sim.utils.sensor import LiDAR
from harl.envs.robot_crowd_sim.utils.render import Render
from harl.envs.robot_crowd_sim.utils.spawner import CircleSpawner,RectangleSpawner,Room361Spawner
from harl.envs.robot_crowd_sim.utils.action import ActionVW,ActionXY
from harl.envs.robot_crowd_sim.utils.info import *

# macro
EPISODE_TIME_STATE_DIM = 2

class RobotCrowdSim:
    
    def __init__(self,
                 args,
                 phase:str,
                 nenv:int=1,
                 thisSeed:int=0,
                 ) -> None:
        
        # setup env
        self.nenv = nenv
        self.thisSeed = thisSeed
        self.phase = phase
        self.human_policy = args.get("human_policy","ai")
        self.human_random_pref_v_and_size = args.get("human_random_pref_v_and_size",False)
        self.robot_random_pref_v_and_size = args.get("robot_random_pref_v_and_size",False)
        self.args = copy.deepcopy(args)
        # map setup
        # self._with_static_obstacle = cfg.with_static_obstacle TODO
        # self._regen_map_every = cfg.regen_map_every TODO
        self._scenario = self.args["scenario"]
        self._map_size = self.args["map_size"]
        if self._scenario == "circle_cross":
            self._map = Map(self._map_size,self._map_size) 
        elif self._scenario == "corridor":
            self._map = Map(self._map_size/2,self._map_size*1.5)
        elif self._scenario == "room_361":
            self._map = Map(3.7,5.6)
        else:
            raise NotImplementedError
        self._time_step = 0.25
        # setup agents (robot and crowd)
        self._human_num = self.args["human_num"]
        self._robot_num = self.args["robot_num"]
        self.n_agents = self._human_num+self._robot_num #TODO change self.robot to self.robots: List[Agent] to adapt self play
        
        self.agents:tp.List[Agent] = []
        self.robots = []
        self.humans = []
        self.distracted_humans = []
        agent_id = 0
        for _ in range(self._robot_num):
            self.robots.append(agent_id)
            self.agents.append(Agent(agent_id,"robot","unicycle",self._time_step,self.args["robot_visible"],self.args["robot_v_pref"],self.args["robot_radius"],self.args["robot_rotation_constrain"]))
            agent_id+=1
        for _ in range(self._human_num):
            self.humans.append(agent_id)
            if self.human_policy == "ai":
                self.agents.append(Agent(agent_id,"human","unicycle",self._time_step,self.args["human_visible"],self.args["human_v_pref"],self.args["human_radius"],self.args["human_rotation_constrain"]))
            else:
                self.agents.append(Agent(agent_id,"human","holonomic",self._time_step,self.args["human_visible"],self.args["human_v_pref"],self.args["human_radius"],self.args["human_rotation_constrain"])) 
            agent_id+=1
        self.agent_log_probs = [None for _ in range(len(self.agents))]
        # for human in self.crowd:
        #     human.sample_random_attributes()
        self._use_human_preference = self.args["use_discriminator"]
        self._human_preference_vector_dim = self.args["human_preference_vector_dim"]
        self._human_preference_type = self.args["human_preference_type"]
        self._crowd_preference = {}

        if self._scenario == "circle_cross":
            self._spawner = CircleSpawner(self.args["map_size"],self.agents) #TODO add corridor spawner
        elif self._scenario == "corridor":
            self._spawner = RectangleSpawner(self._map._map_width,self._map._map_hight,self.agents)
        elif self._scenario == "room_361":
            self._spawner = Room361Spawner(self._map._map_width,self._map._map_hight,self.agents)
        else:
            raise NotImplementedError

        # sensor setup
        self.n_laser=self.args["n_laser"]
        self.laser_angle_resolute=self.args["laser_angle_resolute"]
        self.laser_min_range=self.args["laser_min_range"]
        self.laser_max_range=self.args["laser_max_range"]
        self.human_fov = self.args["human_fov"]
        self.agent_sensors:tp.List[LiDAR] = []
        for agent in self.agents:
            others = []
            for other in self.agents:
                if other.id != agent.id:
                    others.append(other)
            if agent.agent_type == "robot":
                self.agent_sensors.append(
                    LiDAR(self._map,agent,others,
                            self.n_laser,self.laser_angle_resolute,
                            self.laser_min_range,self.laser_max_range)
                )
            else:
                self.agent_sensors.append(
                    LiDAR(self._map,agent,others,
                            self.n_laser,(self.human_fov/180*np.pi)/self.n_laser,
                            0,8)
                )

        """
        observation space [O_robot, O_crowd]
        O_robot = [robot_transformed_state,robot_lidar_scan] -> dim=6+720
        shared observation space [O_robot, robot_position, O_crowd, crowd_position]
        """
        self.discriminator_dim = self._human_preference_vector_dim #if self._use_human_preference else 0
        self.robot_obs_dim = self.n_laser+6+self.discriminator_dim
        self.human_obs_dim = self.n_laser+6+self.discriminator_dim
        self.padded_obs_dim = max(self.robot_obs_dim,self.human_obs_dim) # TODO delete padding
        self.robot_obs_space = gym.spaces.Box(-np.inf,np.inf,(self.padded_obs_dim,))
        self.human_obs_sapce = gym.spaces.Box(-np.inf,np.inf,(self.padded_obs_dim,))
        self.observation_space = [self.robot_obs_space]*self._robot_num+[self.human_obs_sapce]*self._human_num
        self.share_observation_space = [gym.spaces.Box(-np.inf,np.inf,
                                                       ((self.robot_obs_dim+2)*self._robot_num\
                                                        +(self.human_obs_dim+2)*self._human_num,))
                                        ]*(self.n_agents)
        # action space [A_robot, A_crowd]
        self.action_space = [gym.spaces.Box(-np.inf,np.inf,(2,))]*(self.n_agents)

        # setup reward
        self._penalty_collision = self.args["penalty_collision"]
        self._reward_goal = self.args["reward_goal"]
        self._goal_factor = self.args["goal_factor"]
        self._goal_range = self.args["goal_range"]
        self._penalty_backward = self.args["penalty_backward"]
        self._velo_factor = self.args["velo_factor"]
        self._discomfort_penalty_factor = self.args["discomfort_penalty_factor"]
        self._discomfort_dist = self.args["discomfort_dist"]

        # setup episode
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 2000,
                          'test': 1000}
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        self.global_time : float = 0
        self._num_episode_steps : int = 0
        self._max_episode_length: int = self.args["max_episode_length"]

        # setup render
        self._render = Render(self._map,self.agents) #TODO multiple robots

    def seed(self,seed): #TODO check if this match the code of init_env in envs_tools.py line 102
        self.thisSeed = seed
        return 
    
    def set_agent_size(self,agent_id,r):
        self.agents[agent_id].radius = r
    
    def set_agent_v_pref(self,agent_id,v_pref):
        self.agents[agent_id].v_pref = v_pref

    def record_current_log_prob(self,agent_id,log_prob):
        self.agent_log_probs[agent_id] = log_prob

    
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
        
        obs, share_obs = self._genObservation()

        self._render.reset()

        return obs, share_obs,available_actions
    
    def step(self, actions, rec=False):
        available_actions = None
        self._render.robot_rec = rec
        # robot step
        for agent_id in self.robots:
            assert self.agents[agent_id].agent_type == "robot"
            robot_action = actions[agent_id]
            if rec==False:
                robot_action = np.clip(robot_action,np.array([-1,-1]),np.array([1,1]))
                robot_action = ActionVW(
                                ((robot_action[0]+1.)/2.)*self.agents[agent_id].v_pref,
                                robot_action[1]*self.agents[agent_id].rotation_constraint*np.pi/180.
                                )
                if not self.agents[agent_id].task_done:
                    self.agents[agent_id].step(robot_action)
            else:
                self.agents[agent_id].kinematics="holonomic"
                robot_action = ActionXY(
                                    robot_action[0],
                                    robot_action[1]
                                    )
                if not self.agents[agent_id].task_done:
                    self.agents[agent_id].step(robot_action)     
                self.agents[agent_id].kinematics="unicycle"   
        
        # crowd step
        for agent_id in self.humans:
            assert self.agents[agent_id].agent_type == "human"
            human_action = actions[agent_id]
            if self.human_policy == "ai":
                human_action = np.clip(human_action,np.array([-1,-1]),np.array([1,1]))
                human_action = ActionVW(
                                    ((human_action[0]+1.)/2.)*self.agents[agent_id].v_pref,
                                    human_action[1]*self.agents[agent_id].rotation_constraint*np.pi/180.
                                    )
            else:
                human_action = ActionXY(
                                    human_action[0],
                                    human_action[1]
                                    )
            if not self.agents[agent_id].task_done:
                self.agents[agent_id].step(human_action)

        self.global_time+=self._time_step
        self._num_episode_steps+=1
        # if self.global_time>12:
        #     print("nani",self.global_time,self._time_step)
        #     input("cao ni ma")
        obs, share_obs = self._genObservation()
        rewards = []
        dones = []
        infos = []
        for agent in self.agents:
            reward, done, info = self._calAgentReward(agent)
            rewards.append([reward])
            dones.append(done)
            infos.append(info)

        return (
            obs, share_obs, 
            rewards, 
            dones, 
            infos,
            available_actions
        )
    
    def render(self, mode="rgb_array"):
        return self._render.rend(mode,self.global_time,self._crowd_preference,self.agent_log_probs)
    
    def close(self):

        return
    
    def _initHumanPreferenceVector(self,preference=None):
        if self._human_preference_type == "category":
            if preference is None:
                random_index = np.random.randint(0, self._human_preference_vector_dim)
            else:
                assert preference<self._human_preference_vector_dim and isinstance(preference,int)
                random_index = preference
            # Create a one-hot vector
            task = np.zeros(self._human_preference_vector_dim)
            task[random_index] = 1
        else:
            raise NotImplementedError
            task = np.random.randn(self._human_preference_vector_dim)
            task = task / np.linalg.norm(task)
        return task
    
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
        return np.hstack([state,scan,prefer])
    
    def _genObservation(self):
        """
        observation space [O_robot, O_crowd]
        O_robot = [robot_transformed_state,robot_lidar_scan] -> dim=6+720
        shared observation space [O_robot, robot_position, O_crowd, crowd_position]
        """
        # TODO: memorize scan end for rendering sensor
  
        obs = []
        for agent_id in range(len(self.agents)):
            obs.append(
                self._get_agent_obs(agent_id)
            )
            

        s_obs = []
        s_obs.append(np.hstack(obs))
        for i,agent in enumerate(self.agents):
            s_obs.append(np.array(agent.get_position(),dtype=np.float32))
        s_obs = np.hstack(s_obs)

        return obs,[s_obs]*(self.n_agents)
    
    def _checkAgentCollision(self,agent:Agent):
        collision = self._map.checkCollision(agent)
        min_dist = np.inf
        for other in self.agents:
            if agent.id == other.id: continue
            dist = np.linalg.norm([agent.px-other.px,agent.py-other.py])-agent.radius-other.radius
            if dist < min_dist:
                min_dist = dist
            if dist <0:
                collision = True
        return collision, min_dist
    
    def _calAgentReward(self,agent:Agent):
        if agent.task_done:
            return 0.,True,{"episode_info":Nothing(),"bad_transition":False}
        collision,min_dist = self._checkAgentCollision(agent)
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
            if min_dist<self._discomfort_dist and agent.id in self.robots: 
                #only robot use this reward?
                #this result better since crowd have more diversity
                #crowd with this panelty result in robot bad perform on test
                # # 0.86 0.11 0.03 0.785 7.483 0.891
                reward += self._discomfort_penalty_factor*(min_dist-self._discomfort_dist)
                episode_info = Discomfort(min_dist)
        return reward, done,{"episode_info":episode_info,"bad_transition":truncation}