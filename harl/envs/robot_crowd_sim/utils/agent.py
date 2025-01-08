import abc
import numpy as np
from numpy.linalg import norm
from shapely.geometry import Point
import typing as tp
from harl.envs.robot_crowd_sim.utils.action import ActionRot,ActionVW,ActionXY

# macro
AGENT_FULL_STATE_DIM = 8
AGENT_OBSERVABLE_STATE_DIM = 5
AGENT_HIDDEN_STATE_DIM = 3
class Agent(object):
    def __init__(self, agent_id,agent_type, kinematics,time_step, visible,v_pref,radius,rotation_constraint=np.pi*2):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = visible
        self.v_pref = v_pref
        self.radius = radius
        self.kinematics = kinematics
        self.time_step = time_step
        self.px : float 
        self.py : float
        self.gx : float
        self.gy : float
        self.vx : float
        self.vy : float
        self.theta : float
        self.goal_theta : float
        self.goal_v : float
        self.w : float
        self.rotation_constraint=rotation_constraint
        self.id = agent_id
        self.agent_type = agent_type
        self.dg: float
        self.prev_dg:float
        self.hf: float
        self.min_dist: float

        #TODO deal with different shape
        self.collider = None 
        self.task_done = False

        self.time_sampling = 10

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.7, 1.4)
        self.radius = np.random.uniform(0.2, 0.3)

    def set(self, px:float, py:float, gx:float, gy:float, vx:float, vy:float, theta:float, w:float=0.,
            radius:tp.Optional[float]=None, v_pref:tp.Optional[float]=None,goal_theta:tp.Optional[float]=None,goal_v:tp.Optional[float]=None):
        self.px = px
        self.py = py
        self.sx = px
        self.sy = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.w = w
        self.dg = np.linalg.norm((gx-px,gy-py))
        dxy = np.array(self.get_goal_position())-np.array(self.get_position())
        goal_direction = np.arctan2(dxy[1],dxy[0])
        hf = (self.theta-goal_direction)% (2 * np.pi)
        if hf > np.pi:
            hf -= 2 * np.pi
        self.hf = hf

        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref
        if goal_theta is not None:
            self.goal_theta = goal_theta
        if goal_v is not None:
            self.goal_v = goal_v
        self.collider = Point(px, py).buffer(self.radius)

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_start_position(self):
        return self.sx, self.sy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    def set_goal(self,gx,gy):
        self.gx = gx
        self.gy = gy
        self.dg = np.linalg.norm((self.gx-self.px,self.gy-self.py))

    def get_observable_state(self):
        return self.px,self.py,self.vx,self.vy,self.radius
    
    def get_full_state(self):
        return self.px,self.py,self.vx,self.vy,self.radius, self.gx,self.gy,self.v_pref
    
    def get_transformed_state(self):
        dxy = np.array(self.get_goal_position())-np.array(self.get_position())
        dg = np.linalg.norm(dxy)
        goal_direction = np.arctan2(dxy[1],dxy[0])
        hf = (self.theta-goal_direction)% (2 * np.pi)
        if hf > np.pi:
            hf -= 2 * np.pi
        # transform dxy to base frame
        vx = (self.vx * np.cos(goal_direction) + self.vy * np.sin(goal_direction))
        vy = (self.vy * np.cos(goal_direction) - self.vx * np.sin(goal_direction)) 

        return dg, hf, vx, vy, self.radius, self.v_pref

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        elif self.kinematics == 'unicycle':
            assert isinstance(action,ActionVW)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        elif self.kinematics == 'unicycle':
            self.theta = (self.theta + action.w * delta_t) % (2 * np.pi)
            px = self.px + np.cos(self.theta) * action.v * delta_t
            py = self.py + np.sin(self.theta) * action.v * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.prev_dg = self.dg
        x,y = self.px,self.py
        self.check_validity(action)
        for i in range(self.time_sampling):
            pos = self.compute_position(action, self.time_step/self.time_sampling)
            self.px, self.py = pos
            if self.kinematics == 'holonomic':
                self.vx = action.vx
                self.vy = action.vy
            elif self.kinematics == 'unicycle':
                self.vx = action.v * np.cos(self.theta)
                self.vy = action.v * np.sin(self.theta)
                self.w  = action.w
            else:
                self.theta = (self.theta + action.r) % (2 * np.pi)
                self.vx = action.v * np.cos(self.theta)
                self.vy = action.v * np.sin(self.theta)
        self.collider = Point(self.px, self.py).buffer(self.radius)
        self.dg = np.linalg.norm((self.gx-self.px,self.gy-self.py))

        dxy = np.array(self.get_goal_position())-np.array(self.get_position())
        goal_direction = np.arctan2(dxy[1],dxy[0])
        hf = (self.theta-goal_direction)% (2 * np.pi)
        if hf > np.pi:
            hf -= 2 * np.pi
        self.hf = hf

        
    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

