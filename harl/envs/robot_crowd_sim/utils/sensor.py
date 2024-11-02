from harl.envs.robot_crowd_sim.utils.C_library.motion_plan_lib import *
import numpy as np
from typing import List
from harl.envs.robot_crowd_sim.utils.agent import Agent
from harl.envs.robot_crowd_sim.utils.map import Map

class LiDAR:

    def __init__(self,map,self_agent,other_agents,n_laser,laser_angle_resolute,laser_min_range,laser_max_range):
        self._map:Map = map
        self._self_agent:Agent = self_agent
        self._other_agents:List[Agent] = other_agents

        self.n_laser=n_laser
        self.laser_angle_resolute=laser_angle_resolute
        self.laser_min_range=laser_min_range
        self.laser_max_range=laser_max_range
        self.distracted_range = None
        
        return
    
    def getScan(self):
        num_line = 4
        num_circle = len(self._other_agents)
        scan = np.zeros(self.n_laser, dtype=np.float32)
        scan_end = np.zeros((self.n_laser, 2), dtype=np.float32)
        pose = np.array([self._self_agent.px, self._self_agent.py, self._self_agent.theta])
        
        InitializeEnv(num_line, num_circle, self.n_laser, self.laser_angle_resolute)
        x_b,y_b = self._map.getBoundary()
        for i in range(4):
            line = [(x_b[i],y_b[i]),(x_b[i+1],y_b[i+1])]
            set_lines(4 * i    , line[0][0])
            set_lines(4 * i + 1, line[0][1])
            set_lines(4 * i + 2, line[1][0])
            set_lines(4 * i + 3, line[1][1])
        
        i=0
        for agent in self._other_agents:
            set_circles(3 * i    , agent.px)
            set_circles(3 * i + 1, agent.py)
            set_circles(3 * i + 2, agent.radius)
            i+=1
        set_robot_pose(pose[0], pose[1], pose[2])
        cal_laser() #memory leak
        
        for i in range(self.n_laser):
            scan[i] = get_scan(i)
            scan_end[i, :] = np.array([get_scan_line(4 * i + 2), get_scan_line(4 * i + 3)])
            ### used for visualization
            # self.scan_intersection.append([(get_scan_line(4 * i + 0), get_scan_line(4 * i + 1)), \
            #                                (get_scan_line(4 * i + 2), get_scan_line(4 * i + 3))])
            # ### used for visualization
        ReleaseEnv()

        cliped_scan = np.clip(scan,self.laser_min_range,self.laser_max_range).astype(np.float32)
        scan_vector = scan_end-np.array(self._self_agent.get_position())
        cliped_vector = scan_vector/scan.reshape(-1,1)*cliped_scan.reshape(-1,1)
        return cliped_scan, cliped_vector+np.array(self._self_agent.get_position()) #need self position to clip this value
