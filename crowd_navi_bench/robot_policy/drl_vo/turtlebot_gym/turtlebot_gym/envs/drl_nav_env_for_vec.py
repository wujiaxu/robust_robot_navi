import gym
import typing as tp
import numpy as np
import random
from gym import spaces
from harl.envs.robot_crowd_sim.crowd_env import RobotCrowdSim
# from harl.envs.robot_crowd_sim.crowd_env_ccp import RobotCrowdSimCCP
NUM_TP = 10     # the number of timestamps
NUM_PEDS = 34+1

class DRLNavEnvForVec(gym.Env):
    """
    Gazebo env converts standard openai gym methods into Gazebo commands

    To check any topic we need to have the simulations running, we need to do two things:
        1)Unpause the simulation: without that the stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2)If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation and need to be reseted to work properly.
    """
    def __init__(self):

        self.env:tp.Optional[RobotCrowdSim] = None

        self.high_action = np.array([1, 1])
        self.low_action = np.array([-1, -1])
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)

        self.ROBOT_RADIUS = 0.3
        self.GOAL_RADIUS = 0.3 #0.3
        self.DIST_NUM = 10
        self.dist_to_goal_reg = np.zeros(self.DIST_NUM)

        self.ts_cnt = 0
        self.ped_pos = []
        self.scan = [] #np.zeros(720)
        self.scan_buffer = []
        self.mht_peds = []
        self.goal = []
        self.observation_space = spaces.Box(low=-1, high=1, shape=(19202,), dtype=np.float32)

        return
    
    def configure(self,env:RobotCrowdSim):

        self.env = env
        assert self.env._time_step == 0.05
        # assert self.env._max_episode_length == 512

        return
    
    def step(self, actions):

        (
            obs, share_obs, 
            rewards, 
            dones, 
            infos,
            available_actions
        ) = self.env.step(actions)
        
        obs_drl_vo = self._get_observation(obs)
        reward = self._compute_reward(obs)
        
        return obs_drl_vo,obs[1:],reward,dones,infos
    
    def reset(self,seed=None):
        # print("reset")
        self.ts_cnt = 0
        self.dist_to_goal_reg = np.zeros(self.DIST_NUM)
        self.scan_buffer = []
        obs = self.env.reset(seed=seed)
        obs_drl_vo = self._get_observation(obs)

        return obs_drl_vo, obs[1:]
    
    def render(self, mode="human"):
        return self.env.render()
    
    def _compute_reward(self,obs):

        # reward parameters:
        r_arrival = 20 #15
        r_waypoint = 3.2 #2.5 #1.6 #2 #3 #1.6 #6 #2.5 #2.5
        r_collision = -20 #-15
        r_scan = -0.2 #-0.15 #-0.3
        r_angle = 0.6 #0.5 #1 #0.8 #1 #0.5
        r_rotation = -0.1 #-0.15 #-0.4 #-0.5 #-0.2 # 0.1

        angle_thresh = np.pi/6
        w_thresh = 1 # 0.7

        # reward parts:
        r_g = self._goal_reached_reward(r_arrival, r_waypoint)
        r_c = self._obstacle_collision_punish(obs[6:726], r_scan, r_collision)
        r_w = self._angular_velocity_punish(self.env.agents[0].w,  r_rotation, w_thresh)
        r_t = self._theta_reward(self.goal, self.mht_peds, self.env.agents[0].v, r_angle, angle_thresh)
        # print(r_g,r_c , r_t , r_w)
        reward = r_g + r_c + r_t + r_w #+ r_v # + r_p

        return reward
    
    def _goal_reached_reward(self, r_arrival, r_waypoint):
        """
        Returns positive reward if the robot reaches the goal.
        :param transformed_goal goal position in robot frame
        :param k reward constant
        :return: returns reward colliding with obstacles
        """
        # distance to goal:
        dist_to_goal = self.env.agents[0].dg
        # t-1 id:
        t_1 = self.env._num_episode_steps % self.DIST_NUM
        # initialize the dist_to_goal_reg:
        if(self.env._num_episode_steps == 0):
            self.dist_to_goal_reg = np.ones(self.DIST_NUM)*dist_to_goal

        max_iteration = 512 #800 
        # reward calculation:
        if(dist_to_goal <= self.GOAL_RADIUS):  # goal reached: t = T
            reward = r_arrival
        elif(self.env._num_episode_steps >= max_iteration):  # failed to the goal
            reward = -r_arrival
        else:   # on the way
            reward = r_waypoint*(self.env.agents[0].prev_dg - dist_to_goal)

        # storage the robot pose at t-1:
        #if(self.num_iterations % 40 == 0):
        self.dist_to_goal_reg[t_1] = dist_to_goal #self.curr_pose
    
        return reward

    def _obstacle_collision_punish(self, scan, r_scan, r_collision):
        """
        Returns negative reward if the robot collides with obstacles.
        :param scan containing obstacles that should be considered
        :param k reward constant
        :return: returns reward colliding with obstacles
        """
        min_scan_dist = np.amin(scan[scan!=0])
        #if(self.bump_flag == True): #or self.pos_valid_flag == False):
        if(min_scan_dist <= self.ROBOT_RADIUS and min_scan_dist >= 0.02):
            reward = r_collision
        elif(min_scan_dist < 3*self.ROBOT_RADIUS):
            reward = r_scan * (3*self.ROBOT_RADIUS - min_scan_dist)
        else:
            reward = 0.0

        return reward

    def _angular_velocity_punish(self, w_z,  r_rotation, w_thresh):
        """
        Returns negative reward if the robot turns.
        :param w roatational speed of the robot
        :param fac weight of reward punish for turning
        :param thresh rotational speed > thresh will be punished
        :return: returns reward for turning
        """
        if(abs(w_z) > w_thresh):
            reward = abs(w_z) * r_rotation
        else:
            reward = 0.0

        return reward

    def _theta_reward(self, goal, mht_peds, v_x, r_angle, angle_thresh):
        """
        Returns negative reward if the robot turns.
        :param w roatational speed of the robot
        :param fac weight of reward punish for turning
        :param thresh rotational speed > thresh will be punished
        :return: returns reward for turning
        """
        # prefer goal theta:
        theta_pre = np.arctan2(goal[1], goal[0])
        d_theta = theta_pre

        # get the pedstrain's position:
        if(len(mht_peds) != 0):  # tracker results
            d_theta = np.pi/2 #theta_pre
            N = 60
            theta_min = 1000
            for i in range(N):
                theta = random.uniform(-np.pi, np.pi)
                free = True
                for ped in mht_peds:
                    #ped_id = ped.track_id 
                    # create pedestrian's postion costmap: 10*10 m
                    p_x = ped[0]
                    p_y = ped[1]
                    p_vx = ped[2]
                    p_vy = ped[3]
                    
                    ped_dis = np.linalg.norm([p_x, p_y])
                    if(ped_dis <= 7):
                        ped_theta = np.arctan2(p_y, p_x)
                        vo_theta = np.arctan2(3*self.ROBOT_RADIUS, np.sqrt(max(0, ped_dis**2 - (3 * self.ROBOT_RADIUS)**2)))
                        # collision cone:
                        theta_rp = np.arctan2(v_x*np.sin(theta)-p_vy, v_x*np.cos(theta) - p_vx)
                        if(theta_rp >= (ped_theta - vo_theta) and theta_rp <= (ped_theta + vo_theta)):
                            free = False
                            break

                # reachable available theta:
                if(free):
                    theta_diff = (theta - theta_pre)**2
                    if(theta_diff < theta_min):
                        theta_min = theta_diff
                        d_theta = theta
                
        else: # no obstacles:
            d_theta = theta_pre

        reward = r_angle*(angle_thresh - abs(d_theta))

        
        return reward 
    
    def _get_observation(self,obs):

        self.ped_pos = self._get_ped_pos_map()
        self.goal = self._get_goal()
        self.vel = np.array([self.env.agents[0].v,self.env.agents[0].w])

        self.scan_buffer.append(obs[6:726].tolist())
        if(self.ts_cnt == 0): 
            for i in range(NUM_TP-1):
                self.scan_buffer.append(obs[6:726].tolist())
        self.scan = [float(val) for sublist in self.scan_buffer for val in sublist]
        self.scan_buffer = self.scan_buffer[1:NUM_TP]

        # ped map:
        # MaxAbsScaler:
        v_min = -2
        v_max = 2
        self.ped_pos = np.array(self.ped_pos, dtype=np.float32)
        self.ped_pos = 2 * (self.ped_pos - v_min) / (v_max - v_min) + (-1)

        # scan map:
        # MaxAbsScaler:
        temp = np.array(self.scan, dtype=np.float32)
        scan_avg = np.zeros((20,80))
        for n in range(10):
            scan_tmp = temp[n*720:(n+1)*720]
            for i in range(80):
                scan_avg[2*n, i] = np.min(scan_tmp[i*9:(i+1)*9])
                scan_avg[2*n+1, i] = np.mean(scan_tmp[i*9:(i+1)*9])
        
        scan_avg = scan_avg.reshape(1600)
        scan_avg_map = np.tile(scan_avg,(1,4))
        self.scan = scan_avg_map.reshape(6400)
        s_min = 0
        s_max = 30
        self.scan = 2 * (self.scan - s_min) / (s_max - s_min) + (-1)
        
        # goal:
        # MaxAbsScaler:
        g_min = -2
        g_max = 2
        self.goal = np.array(self.goal, dtype=np.float32)
        self.goal = 2 * (self.goal - g_min) / (g_max - g_min) + (-1)
        #self.goal = self.goal.tolist()

        # observation:
        self.observation = np.concatenate((self.ped_pos, self.scan, self.goal), axis=None) #list(itertools.chain(self.ped_pos, self.scan, self.goal))
        #self.observation = np.concatenate((self.scan, self.goal), axis=None)

        return self.observation
    
    def _get_ped_pos_map(self,):

        ped_pos_map_tmp = np.zeros((2,80,80))

        robot_pos = np.zeros(3)
        robot_pos[:2] = np.array([self.env.agents[0].px,self.env.agents[0].py])
        robot_pos[2] =  self.env.agents[0].theta    
        map_R_robot = np.array([[np.cos(robot_pos[2]), -np.sin(robot_pos[2])],
                                [np.sin(robot_pos[2]),  np.cos(robot_pos[2])],
                            ])
        map_T_robot = np.array([[np.cos(robot_pos[2]), -np.sin(robot_pos[2]), robot_pos[0]],
                                [np.sin(robot_pos[2]),  np.cos(robot_pos[2]), robot_pos[1]],
                                [0, 0, 1]])
        # robot_T_map = (map_T_robot)^(-1)
        robot_R_map = np.linalg.inv(map_R_robot)
        robot_T_map = np.linalg.inv(map_T_robot)

        self.mht_peds = []
        for human in self.env.agents[1:]:
            ped_pos = np.array([human.px, human.py, 1])
            ped_vel = np.array([human.vx,human.vy])
            ped_pos_in_robot = np.matmul(robot_T_map, ped_pos.T)
            ped_vel_in_robot = np.matmul(robot_R_map, ped_vel.T) 
            x = ped_pos_in_robot[0]
            y = ped_pos_in_robot[1]
            vx = ped_vel_in_robot[0]
            vy = ped_vel_in_robot[1]
            self.mht_peds.append([x,y,vx,vy])
            # 20m * 20m occupancy map:
            if(x >= 0 and x <= 20 and np.abs(y) <= 10):
                # bin size: 0.25 m
                c = int(np.floor(-(y-10)/0.25))
                r = int(np.floor(x/0.25))

                if(r == 80):
                    r = r - 1
                if(c == 80):
                    c = c - 1
                # cartesian velocity map
                ped_pos_map_tmp[0,r,c] = vx
                ped_pos_map_tmp[1,r,c] = vy

        return ped_pos_map_tmp
    
    def _get_goal(self,):
        x = np.array([self.env.agents[0].px,self.env.agents[0].py])
        theta = self.env.agents[0].theta
        goal = np.array([self.env.agents[0].gx,self.env.agents[0].gy])
        map_T_robot = np.array([[np.cos(theta), -np.sin(theta), x[0]],
                                    [np.sin(theta), np.cos(theta), x[1]],
                                    [0, 0, 1]])

        goal = np.matmul(np.linalg.inv(map_T_robot), np.array([[goal[0]],[goal[1]],[1]])) #np.dot(np.linalg.inv(map_T_robot), np.array([goal[0], goal[1],1])) #
        goal = goal[0:2]
        return goal
    
    

    
