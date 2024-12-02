import abc
from typing import List
import numpy as np
from harl.envs.robot_crowd_sim.utils.agent import Agent

class Spawner(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def spawnAgent(self,agent:Agent):

        return
    
class RectangleArea:

    def __init__(self,x_min, x_max, y_min, y_max,margin=0.5):
        self.x_min, self.x_max, self.y_min, self.y_max, self.margin = x_min, x_max, y_min, y_max,margin
    def get_random_position(self):

        x = np.random.uniform(self.x_min+self.margin, self.x_max-self.margin)
        y = np.random.uniform(self.y_min+self.margin, self.y_max-self.margin)
        return x, y

    
class Room361Spawner(Spawner):

    def __init__(self,width:float,hight,agents:List[Agent]):
        self._agents:List[Agent]=agents
        self._map_width = width
        self._map_hight = hight

        # 6 area => 3 row 2 col
        self.area = {}
        self.goal_area_mapping = {} #spawn area id to goal area id
        self.area_width = self._map_width/2.
        self.area_hight = self._map_hight/3.
        for i in range(2):
            for j in range(3):
                area_id = i*3+j
                self.area[area_id] = RectangleArea(i*self.area_width-width/2,
                                                         (i+1)*self.area_width-width/2,
                                                         j*self.area_hight-hight/2,
                                                         (j+1)*self.area_hight-hight/2)
        self.goal_area_mapping={0:5,1:4,2:3,3:2,4:1,5:0}


        return 
    
    def spawnAgent(self, agent: Agent):
        counter = 2e7
        while True:
            spawn_area_id = np.random.randint(0, 6)
            goal_area_id = self.goal_area_mapping[spawn_area_id]
            px,py = self.area[spawn_area_id].get_random_position()
            gx,gy = self.area[goal_area_id].get_random_position()
            # TODO consider obstacle
            collide = False
            for other in self._agents:
                if other.id >= agent.id:
                    continue
                min_dist = agent.radius + other.radius + 0.2#self._discomfort_dist
                if np.linalg.norm((px - other.px, py - other.py)) < min_dist or \
                        np.linalg.norm((gx - other.gx, gy - other.gy)) < min_dist:
                    collide = True
                    break
            if not collide or counter<0:
                break
            counter-=1
        # robot.start_pos.append((px, py))
        theta_noise = (np.random.random()-0.5) * agent.rotation_constraint*np.pi/180
        theta = np.arctan2(gy-py,gx-px)+theta_noise
        vx = agent.v_pref*np.cos(theta)
        vy = agent.v_pref*np.sin(theta)
        agent.set(px, py, gx, gy, 0,0, theta)
        return 
    
class Room256Spawner(Spawner):

    def __init__(self,width:float,hight,agents:List[Agent]):
        self._agents:List[Agent]=agents
        self._map_width = width #3
        self._map_hight = hight #3

        # 6 area => 3 row 2 col
        self.area = {}
        
        self.area_width = 0.8
        
        # left
        self.area[0] = RectangleArea(-width/2,
                                    -width/2+self.area_width,
                                    -hight/2+0.2,
                                    hight/2-0.2,
                                    0.3)
        # top
        self.area[1] = RectangleArea(-width/2,
                                    width/2,
                                    hight/2-self.area_width,
                                    hight/2,
                                    0.3)
        # right
        self.area[2] = RectangleArea(width/2-self.area_width,
                                    width/2,
                                    -hight/2+0.2,
                                    hight/2-0.2,
                                    0.3)
        # top
        self.area[3] = RectangleArea(-width/2,
                                    width/2,
                                    -hight/2,
                                    -hight/2+self.area_width,
                                    0.3)


        return 
    
    def spawnAgent(self, agent: Agent):
        counter = 2e7
        while True:
            spawn_area_id = np.random.randint(0, 4)
            px,py = self.area[spawn_area_id].get_random_position()
            gx,gy = -px,-py
            # TODO consider obstacle
            collide = False
            for other in self._agents:
                if other.id >= agent.id:
                    continue
                min_dist = agent.radius + other.radius + 0.2#self._discomfort_dist
                if np.linalg.norm((px - other.px, py - other.py)) < min_dist or \
                        np.linalg.norm((gx - other.gx, gy - other.gy)) < min_dist:
                    collide = True
                    break
            if not collide or counter<0:
                break
            counter-=1
        # robot.start_pos.append((px, py))
        theta_noise = (np.random.random()-0.5) * agent.rotation_constraint*np.pi/180
        theta = np.arctan2(gy-py,gx-px)+theta_noise
        vx = agent.v_pref*np.cos(theta)
        vy = agent.v_pref*np.sin(theta)
        agent.set(px, py, gx, gy, 0,0, theta)
        return 
    
class UCYstudentsSpawner(Spawner):

    def __init__(self,width:float,hight,agents:List[Agent]):
        self._agents:List[Agent]=agents
        self._map_width = width
        self._map_hight = hight

        # 6 area => 3 row 2 col
        self.area = {}
        
        self.area_width = 2
        
        # left
        self.area[0] = RectangleArea(-width/2,
                                    -width/2+self.area_width,
                                    -hight/2+self.area_width,
                                    hight/2-self.area_width)
        # top
        self.area[1] = RectangleArea(-width/2+self.area_width,
                                    width/2-self.area_width,
                                    hight/2-self.area_width,
                                    hight/2)
        # right
        self.area[2] = RectangleArea(width/2-self.area_width,
                                    width/2,
                                    -hight/2+self.area_width,
                                    hight/2-self.area_width)
        # top
        self.area[3] = RectangleArea(-width/2+self.area_width,
                                    width/2-self.area_width,
                                    -hight/2,
                                    -hight/2+self.area_width)
        # middle
        self.area[4] = RectangleArea(-width/2+self.area_width,
                                    +width/2-self.area_width,
                                    -hight/2+self.area_width,
                                    +hight/2-self.area_width)
        

        return 
    
    def spawnAgent(self, agent: Agent):
        counter = 2e7
        while True:
            spawn_area_id = np.random.randint(0, 4)
            stop_at_middle = np.random.random()
            if stop_at_middle<0.01:
                goal_area_id = np.random.randint(0, 5)
                while goal_area_id == spawn_area_id:
                    goal_area_id = np.random.randint(0, 5)
            else:
                goal_area_id = np.random.randint(0, 4)
                while goal_area_id == spawn_area_id:
                    goal_area_id = np.random.randint(0, 4)

            px,py = self.area[spawn_area_id].get_random_position()
            gx,gy = self.area[goal_area_id].get_random_position()
            # TODO consider obstacle
            collide = False
            for other in self._agents:
                if other.id >= agent.id:
                    continue
                min_dist = agent.radius + other.radius + 0.2#self._discomfort_dist
                if np.linalg.norm((px - other.px, py - other.py)) < min_dist or \
                        np.linalg.norm((gx - other.gx, gy - other.gy)) < min_dist:
                    collide = True
                    break
            if not collide or counter<0:
                break
            counter-=1
        # robot.start_pos.append((px, py))
        theta_noise = (np.random.random()-0.5) * agent.rotation_constraint*np.pi/180
        theta = np.arctan2(gy-py,gx-px)+theta_noise
        vx = agent.v_pref*np.cos(theta)
        vy = agent.v_pref*np.sin(theta)
        agent.set(px, py, gx, gy, 0,0, theta)
        return 
    
class RectangleSpawner(Spawner):

    def __init__(self,width:float,hight,agents:List[Agent]):
        self._agents:List[Agent]=agents
        self._map_width = width
        self._map_hight = hight

        return 
    
    def spawnAgent(self, agent: Agent):
        counter = 2e7
        while True:
            px = (np.random.random()-0.5) * (self._map_width-1)
            gx = (np.random.random()-0.5) * (self._map_width-1)
            if np.random.random()-0.5<0:
                py = -np.random.uniform(self._map_hight/2-3,self._map_hight/2-1)
                gy = np.random.uniform(self._map_hight/2-3,self._map_hight/2-1)
            else:
                py = np.random.uniform(self._map_hight/2-3,self._map_hight/2-1)
                gy = -np.random.uniform(self._map_hight/2-3,self._map_hight/2-1)
            # TODO consider obstacle
            collide = False
            for other in self._agents:
                if other.id >= agent.id:
                    continue
                min_dist = agent.radius + other.radius + 0.2#self._discomfort_dist
                if np.linalg.norm((px - other.px, py - other.py)) < min_dist or \
                        np.linalg.norm((gx - other.gx, gy - other.gy)) < min_dist:
                    collide = True
                    break
            if not collide or counter<0:
                break
            counter-=1
        # robot.start_pos.append((px, py))
        theta = np.arctan2(gy-py,gx-px)
        vx = agent.v_pref*np.cos(theta)
        vy = agent.v_pref*np.sin(theta)
        agent.set(px, py, gx, gy, 0,0, theta)
        return 
    
class CircleSpawner(Spawner):

    def __init__(self,circle_size:float,agents:List[Agent]):
        self._agents:List[Agent]=agents
        self._circle_size = circle_size

    def spawnAgent(self, agent: Agent):

        counter = 2e7
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * agent.v_pref
            py_noise = (np.random.random() - 0.5) * agent.v_pref
            px = (self._circle_size /2. - 1)* np.cos(angle) + px_noise
            py = (self._circle_size /2. - 1) * np.sin(angle) + py_noise

            # TODO consider obstacle
            collide = False
            for other in self._agents:
                if other.id >= agent.id:
                    continue
                min_dist = agent.radius + other.radius + 0.2#self._discomfort_dist
                if np.linalg.norm((px - other.px, py - other.py)) < min_dist or \
                        np.linalg.norm((-px - other.gx, -py - other.gy)) < min_dist:
                    collide = True
                    break
            if not collide or counter<0:
                break
            counter-=1
        # robot.start_pos.append((px, py))
        theta = np.arctan2(-py,-px)
        vx = agent.v_pref*np.cos(theta)
        vy = agent.v_pref*np.sin(theta)
        agent.set(px, py, -px, -py, 0,0, theta)

        return 