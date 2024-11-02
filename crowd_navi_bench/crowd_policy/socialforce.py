import numpy as np
from crowd_navi_bench.crowd_policy import socialforcelib
from harl.envs.robot_crowd_sim.crowd_env import RobotCrowdSim

class SocialForce():
    def __init__(self,env:RobotCrowdSim,v0=5,sigma=1.5,initial_speed = 1.5):
        self.time_step = 0.25
        self.name = 'SocialForce'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.initial_speed = initial_speed # control traction force from goal
        self.v0 = v0 # magnitude of the force 10for quick reaction #5 for slow
        self.sigma = sigma# 0.3for short collision avoidance distance #1.5 for long
        self.sim = None
        self.static_obstacle = None
        self.env = env
        self.set_static_obstacle()
        print("using sfm parameter:",v0,sigma)

    def set_static_obstacle(self):
        """
        x,y = self._layout["boundary"].coords.xy
        human.policy.set_static_obstacle(self._layout["vertices"]+[[(x[i],y[i]) for i in range(len(x))]])
        """
        x,y = self.env._map.getBoundary()
        obstacles = [[(x[i],y[i]) for i in range(len(x))]]
        self.static_obstacle = []
        for polygon in obstacles:
            #polygon: a loop of a list of vertices 
            for i in range(len(polygon)-1):
                edge = np.linspace(polygon[i],polygon[i+1],500)
                self.static_obstacle.append(edge)
        return
    def predict(self,agent_id):
        """

        :param state:
        :return:
        """
        agents = self.env.agents
        sf_state = []
        (self_state_px, self_state_py,
         self_state_vx, self_state_vy,
         self_state_radius, 
         self_state_gx, self_state_gy,
         self_state_v_pref) = agents[agent_id].get_full_state()
        sf_state.append((self_state_px, self_state_py, self_state_vx, self_state_vy, self_state_gx, self_state_gy))
        for agent in agents:
            if agent.id == agent_id: continue
            if agent.id in self.env.robots and agent_id in self.env.distracted_humans: continue
            (human_state_px, 
            human_state_py, 
            human_state_vx, 
            human_state_vy,
            human_state_radius) = agent.get_observable_state()
            # approximate desired direction with current velocity
            if human_state_vx == 0 and human_state_vy == 0:
                gx = np.random.random()
                gy = np.random.random()
            else:
                gx = human_state_px + human_state_vx
                gy = human_state_py + human_state_vy
            sf_state.append((human_state_px, human_state_py, human_state_vx, human_state_vy, gx, gy))
        
        sim = socialforcelib.Simulator(np.array(sf_state), 
                                    delta_t=self.time_step, 
                                    initial_speed=self.initial_speed,
                                    v0=self.v0, sigma=self.sigma,
                                    ped_space=socialforcelib.PedSpacePotential(self.static_obstacle))
        sim.step()
        
        #clip according to preferred speed
        velo = np.array([sim.state[0, 2], sim.state[0, 3]])
        speed = np.linalg.norm(velo)
        if speed > self_state_v_pref:
            velo = velo/speed * self_state_v_pref
        
        action = velo #ActionXY(velo[0],velo[1])

        return action


class CentralizedSocialForce(SocialForce):
    """
    Centralized socialforce, a bit different from decentralized socialforce, where the goal position of other agents is
    set to be (0, 0)
    """
    def __init__(self):
        super().__init__()

    def predict(self, state):
        sf_state = []
        for agent_state in state:
            sf_state.append((agent_state.px, agent_state.py, agent_state.vx, agent_state.vy,
                             agent_state.gx, agent_state.gy))

        sim = socialforcelib.Simulator(np.array(sf_state), delta_t=self.time_step, initial_speed=self.initial_speed,
                                    v0=self.v0, sigma=self.sigma)
        sim.step()
        actions = [np.array([sim.state[i, 2], sim.state[i, 3]]) for i in range(len(state))]
        del sim

        return actions
