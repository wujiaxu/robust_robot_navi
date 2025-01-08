from pdb import run
from harl.envs.robot_crowd_sim.utils.info import *
import numpy as np
from collections import deque

class InfoMonitor:

    def __init__(self,env_num,human_num,robot_num,time_step,buffer_len=10000,max_episode_len=100):

        self._nenv = env_num
        self.human_num = human_num
        self.robot_num = robot_num
        self._time_step = time_step
        self._buffer_len = buffer_len
        self._max_episode_len = max_episode_len
        self.reset()

    def reset(self):

        self.global_step = 0

        self.robot_step = {}
        for id in range(self.robot_num):
            self.robot_step[id] = [0]*self._nenv

        self.robot_episode = 0
        self.robot_success = deque(maxlen=self._buffer_len)
        self.robot_collide = deque(maxlen=self._buffer_len)
        self.robot_timeout = deque(maxlen=self._buffer_len)
        self.robot_invasion = deque(maxlen=self._buffer_len*self._max_episode_len)
        self.robot_danger = deque(maxlen=self._buffer_len*self._max_episode_len)
        self.robot_time = deque(maxlen=self._buffer_len)

        self.crowd_step = {}
        for id in range(self.human_num):
            self.crowd_step[id] = []# [0]*self._nenv
            for env_id in range(self._nenv):
                self.crowd_step[id].append(0)
            
        self.crowd_episode = 0
        self.crowd_success = deque(maxlen=self._buffer_len)
        self.crowd_collide = deque(maxlen=self._buffer_len)
        self.crowd_timeout = deque(maxlen=self._buffer_len)
        self.crowd_invasion = deque(maxlen=self._buffer_len*self._max_episode_len)
        self.crowd_time = deque(maxlen=self._buffer_len)

        return
    def reset_episode_info(self):
        self.robot_invasion = deque(maxlen=self._buffer_len*self._max_episode_len)
        self.robot_danger = deque(maxlen=self._buffer_len*self._max_episode_len)
        return
    
    def evaluateRobotInfo(self):
        # assert (self.robot_collide+self.robot_success+self.robot_timeout)==self.robot_episode
        # if output_separation:
        return (
                round(sum(self.robot_success)/len(self.robot_success) if len(self.robot_success)>0 else 0,3),
                round(sum(self.robot_collide)/len(self.robot_collide) if len(self.robot_collide)>0 else 0,3),
                round(sum(self.robot_timeout)/len(self.robot_timeout) if len(self.robot_timeout)>0 else 0,3),
                round((len(self.robot_invasion)-self.robot_invasion.count(0))/len(self.robot_invasion) if len(self.robot_invasion)>0 else 0,3),
                round(np.mean(list(self.robot_time)) if len(self.robot_time)>0 else 0,3),
                round(np.std(list(self.robot_time)) if len(self.robot_time)>0 else 0,3),
                round(sum(self.robot_invasion)/(len(self.robot_invasion)-self.robot_invasion.count(0)) if (len(self.robot_invasion)-self.robot_invasion.count(0))>0 else 0,3),
                round(sum(self.robot_danger)/(len(self.robot_danger)-self.robot_danger.count(0)) if (len(self.robot_danger)-self.robot_danger.count(0))>0 else 0,3),
                round((len(self.robot_danger)-self.robot_danger.count(0))/len(self.robot_danger) if len(self.robot_danger)>0 else 0,3),
            )
        # return (
        #     round(sum(self.robot_success)/len(self.robot_success) if len(self.robot_success)>0 else 0,3),
        #     round(sum(self.robot_collide)/len(self.robot_collide) if len(self.robot_collide)>0 else 0,3),
        #     round(sum(self.robot_timeout)/len(self.robot_timeout) if len(self.robot_timeout)>0 else 0,3),
        #     round((len(self.robot_invasion)-self.robot_invasion.count(0))/len(self.robot_invasion) if len(self.robot_invasion)>0 else 0,3),
        #     round(np.mean(list(self.robot_time)) if len(self.robot_time)>0 else 0,3),
        #     round(np.std(list(self.robot_time)) if len(self.robot_time)>0 else 0,3)
        # )

    def evaluateCrowdInfo(self):
        # assert (self.crowd_collide+self.crowd_success+self.crowd_timeout)==self.crowd_episode
        return (
            round(sum(self.crowd_success)/len(self.crowd_success) if len(self.crowd_success)>0 else 0,3),
            round(sum(self.crowd_collide)/len(self.crowd_collide) if len(self.crowd_collide)>0 else 0,3),
            round(sum(self.crowd_timeout)/len(self.crowd_timeout) if len(self.crowd_timeout)>0 else 0,3),
            round(sum(self.crowd_invasion)/len(self.crowd_invasion) if len(self.crowd_invasion)>0 else 0,3),
            round(np.mean(list(self.crowd_time)) if len(self.crowd_time)>0 else 0,3),
            round(np.std(list(self.crowd_time)) if len(self.crowd_time)>0 else 0,3)
        )
    
    def saveInfoVecEnv(self,infos):
        """
        crowd_infos_dict: 
            key: human_id
            value: infos of each human, infos: vec human info from vec_env
        """
        self.global_step+=self._nenv
        for env_id,info_env in enumerate(infos):
            for agent_id, info in enumerate(info_env):
                if agent_id<self.robot_num:
                    self._readRobotInfo(info["episode_info"],agent_id,env_id)
                elif agent_id<self.robot_num+self.human_num:
                    self._readCrowdInfo(info["episode_info"],agent_id-self.robot_num,env_id)
                else:
                    raise ValueError
    
    def _robotNewEpisode(self):
        self.robot_episode+=1
        self.robot_success.append(0)
        self.robot_collide.append(0)
        self.robot_timeout.append(0)

    def _crowdNewEpisode(self):
        self.crowd_episode+=1
        self.crowd_success.append(0)
        self.crowd_collide.append(0)
        self.crowd_timeout.append(0)

    def _readRobotInfo(self,info,robot_id,env_id):
        self.robot_step[robot_id][env_id] += self._time_step
        if isinstance(info,Discomfort) or isinstance(info,Danger):
            
            if isinstance(info,Danger):
                self.robot_danger.append(info.min_dist)
            else:
                self.robot_danger.append(0)
                self.robot_invasion.append(info.min_dist)
            return
        self.robot_invasion.append(0)
        self.robot_danger.append(0)
        if isinstance(info,Nothing): 
            return
        self._robotNewEpisode()
        if isinstance(info,ReachGoal):
            self.robot_time.append(info.time)
            self.robot_success[-1]+=1
        elif isinstance(info,Collision):
            self.robot_collide[-1]+=1
        elif isinstance(info,Timeout):
            self.robot_timeout[-1]+=1
        else:
            raise ValueError
        self.robot_step[robot_id][env_id] = 0
        return
    
    def _readCrowdInfo(self,info,human_id,env_id):
        
        self.crowd_step[human_id][env_id] += self._time_step
        if isinstance(info,Discomfort):
            self.crowd_invasion.append(1)
            return
        self.crowd_invasion.append(0)
        if isinstance(info,Nothing):
            return
        self._crowdNewEpisode()
        if isinstance(info,ReachGoal):
            if info.time>self._max_episode_len*self._time_step:
                print(self.crowd_step[human_id][env_id])
                print(info)
                raise RuntimeError
            self.crowd_time.append(info.time)
            self.crowd_success[-1]+=1
        elif isinstance(info,Collision):
            self.crowd_collide[-1]+=1
        elif isinstance(info,Timeout):
            self.crowd_timeout[-1]+=1
        else:
            raise ValueError
        self.crowd_step[human_id][env_id] = 0
        return
