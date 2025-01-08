import numpy as np
import rvo2
from enum import Enum
from harl.envs.robot_crowd_sim.crowd_env import RobotCrowdSim

class PenType(Enum):
    pen_Psych=1#[ 15   ,       40   ,      38     ,     0.4  ,   1.55]
    pen_Extrav=2#[ 15    ,      23    ,     32    ,      0.4  ,   1.55]
    pen_Neuro=3#[  15    ,      9   ,       29     ,     1.6  ,   1.25]

class ORCA():
    def __init__(self,env:RobotCrowdSim):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        B =
            (   
                w1 * Aggressive
                w2 * Assertive
                w3 * Shy
                w4 * Active
                w5 * T ense
                w6 * Impulsive  
            )

        = RVOmat * (
            1/13.5 (Neighbor Dist - 15)
            1/49.5(M ax. Neighbors - 10)
            1/14.5(Planning Horiz. - 30)
            1/0.85 (Radius - 0.8)
            1/0.5(P ref. Speed - 1.4)
        )
        RVOmat =(
            -0.02 0.32 0.13 -0.41 1.02
            0.03 0.22 0.11 -0.28 1.05
            -0.04 -0.08 0.02 0.58 -0.88
            -0.06 0.04 0.04 -0.16 1.07
            0.10 0.07 -0.08 0.19 0.15
            0.03 -0.15 0.03 -0.23 0.23
        )

        Psychoticism
        Extraversion
        Neuroticism
        Apen =(
            0.00 0.08 0.08 -0.32 0.63
            -0.02 0.13 0.08 -0.22 1.06
            0.03 -0.01 -0.03 0.39 -0.37
        )

        Trait       Neigh Dist. Num Neigh. Plan Horiz. Radius. Speed
        pen_Psych.  15          40         38          0.4     1.55
        pen_Extrav. 15          23         32          0.4     1.55
        pen_Neuro.  15          9          29          1.6     1.25
        Aggres.     15 20 31 0.6 1.55
        Assert. 15 23 32 0.5 1.55
        Shy 15 7 30 1.1 1.25
        Active 13 17 40 0.4 1.55
        Tense 29 63 12 1.6 1.55
        Impul. 30 2 90 0.4 1.55
        """
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = 'holonomic'
        self.safety_space = 0.15
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.max_speed = 0.5#1
        self.time_step = 0.25
        self.sim = None
        self.env = env
        x,y = self.env._map.getBoundary()
        self.static_obstacle = [[(x[i],y[i]),(x[i+1],y[i+1])] for i in range(len(x)-1)]
        # print(self.static_obstacle)
        # self.static_obstacle = []
        # for polygon in obstacles:
        #     #polygon: a loop of a list of vertices 
        #     for i in range(len(polygon)-1):
        #         edge = np.linspace(polygon[i],polygon[i+1],500)
        #         self.static_obstacle.append(edge)

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, agent_id, agent_type=None):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        agents = self.env.agents
        self.max_neighbors = len(self.env.agents)-1
        other_states = []
        # (self_state_px, self_state_py,
        #  self_state_vx, self_state_vy,
        #  self_state_radius, 
        #  self_state_gx, self_state_gy,
        #  self_state_v_pref)
        self_state = list(agents[agent_id].get_full_state())
        for agent in agents:
            if agent.id == agent_id: continue
            if agent.id in self.env.robots and agent_id in self.env.distracted_humans: continue
            [human_state_px, 
            human_state_py, 
            human_state_vx, 
            human_state_vy,
            human_state_radius] = agent.get_observable_state()
            # approximate desired direction with current velocity
            if human_state_vx == 0 and human_state_vy == 0:
                gx = np.random.random()
                gy = np.random.random()
            else:
                gx = human_state_px + human_state_vx
                gy = human_state_py + human_state_vy
            other_states.append([human_state_px, human_state_py, human_state_vx, human_state_vy, human_state_radius])

        if agent_type==None:
            params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        elif agent_type == PenType.pen_Extrav:
            params = 15,23,32,self.time_horizon_obst        
            self_state[4]=0.4     
            self_state[7] = 1.55
            self.max_speed = 1.55
        elif agent_type == PenType.pen_Psych:
            params = 15 , 40, 38 ,self.time_horizon_obst 
            self_state[4]=0.4     
            self_state[7] = 1.55
            self.max_speed = 1.55
        elif agent_type == PenType.pen_Neuro:
            params = 15,9 , 29,self.time_horizon_obst             
            self_state[4]=1.6   
            self_state[7] = 1.25
            self.max_speed = 1.25
        else:
            raise ValueError
        # if self.sim is not None and self.sim.getNumAgents() != len(other_states) + 1:
        #     del self.sim
        #     self.sim = None
        # if self.sim is None:
        self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
        for obst in self.static_obstacle:
            self.sim.addObstacle(obst) 
        self.sim.processObstacles() #TODO
        self.sim.addAgent((self_state[0],self_state[1]), *params, self_state[4],
                            self_state[7], (self_state[2],self_state[3]))
        for other_state in other_states:
            self.sim.addAgent((other_state[0],other_state[1]), *params, other_state[4],
                                self.max_speed, (other_state[2],other_state[3]))
        # else:
        #     self.sim.setAgentPosition(0, robot_state.position)
        #     self.sim.setAgentVelocity(0, robot_state.velocity)
        #     for i, human_state in enumerate(state.human_states):
        #         self.sim.setAgentPosition(i + 1, human_state.position)
        #         self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((self_state[5] - self_state[0], self_state[6] - self_state[1]))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, _ in enumerate(other_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = self.sim.getAgentVelocity(0)
        # self.last_state = state
        return action


# class CentralizedORCA(ORCA):
#     def __init__(self):
#         super().__init__()

#     def predict(self, state):
#         """ Centralized planning for all agents """
#         params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
#         if self.sim is not None and self.sim.getNumAgents() != len(state):
#             del self.sim
#             self.sim = None

#         if self.sim is None:
#             self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
#             for agent_state in state:
#                 self.sim.addAgent(agent_state.position, *params, agent_state.radius + 0.01 + self.safety_space,
#                                   self.max_speed, agent_state.velocity)
#         else:
#             for i, agent_state in enumerate(state):
#                 self.sim.setAgentPosition(i, agent_state.position)
#                 self.sim.setAgentVelocity(i, agent_state.velocity)

#         # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
#         for i, agent_state in enumerate(state):
#             velocity = np.array((agent_state.gx - agent_state.px, agent_state.gy - agent_state.py))
#             speed = np.linalg.norm(velocity)
#             pref_vel = velocity / speed if speed > 1 else velocity
#             self.sim.setAgentPrefVelocity(i, (pref_vel[0], pref_vel[1]))

#         self.sim.doStep()
#         actions = [self.sim.getAgentVelocity(i) for i in range(len(state))]

#         return actions


if __name__ == "__main__":
    import os
    import json
    import random
    from harl.common.video import VideoRecorder
    video_recorder = VideoRecorder(".")
    """
     (   
                w1 * Aggressive
                w2 * Assertive
                w3 * Shy
                w4 * Active
                w5 * T ense
                w6 * Impulsive  
            )
            1/13.5 (Neighbor Dist - 15)
            1/49.5(M ax. Neighbors - 10)
            1/14.5(Planning Horiz. - 30)
            1/0.85 (Radius - 0.8)
            1/0.5(P ref. Speed - 1.4)
    """
    # personality = np.array([0.,0.,1.,0.,0.,0])
    # RVOmat = np.array([
    #         [-0.02, 0.32, 0.13, -0.41, 1.02],
    #         [0.03, 0.22, 0.11, -0.28, 1.05],
    #         [-0.04, -0.08, 0.02, 0.58, -0.88],
    #         [-0.06, 0.04, 0.04, -0.16, 1.07],
    #         [0.10, 0.07, -0.08, 0.19, 0.15],
    #         [0.03, -0.15, 0.03, -0.23, 0.23],
    #     ])
    # pen_mat = np.array([[0.00, 0.08, 0.08, -0.32, 0.63],
    #                     [-0.02, 0.13, 0.08, -0.22, 1.06],
    #                     [0.03, -0.01, -0.03, 0.39, -0.37]])
    
    # weights = np.array([13.5, 
    #                     49.5,
    #                     14.5,
    #                     0.85,
    #                     0.5])
    # b = np.array([15,10,30,0.8,1.4])
    # parameters = personality.dot(np.linalg.pinv(RVOmat).T)*weights+b
    # print(parameters)
    model_dir="/home/dl/wu_ws/HARL/results_seed_1/crowd_env/crowd_navi/robot_crowd_happo/c0.90_happo_5p_3c_rvs_circlecross/seed-00001-2024-09-27-10-55-42"
    config_file = os.path.join(model_dir,"config.json")
    with open(config_file, encoding="utf-8") as file:
        all_config = json.load(file)
    algo_args = all_config["algo_args"]
    env_args = all_config["env_args"]
    env_args["human_policy"] = "ORCA"
    env_args["human_num"] = 4
    env = RobotCrowdSim(env_args,"test",1,0)
    orca_planner = ORCA(env)

    # env.reset()
    # video_recorder.init(env)
    # for i in range(50):
    #     actions = []
    #     for agent_id in range(env.n_agents):
            
    #         action = orca_planner.predict(agent_id,PenType.pen_Neuro)
    #         actions.append(action)

    #     action = np.array(actions)
    #     # print(action)
    #     # input()
    #     env.step(action)
    #     video_recorder.record(env)

    # video_recorder.save("test_neuro.mp4")

    # env.reset()
    # video_recorder.init(env)
    # for i in range(50):
    #     actions = []
    #     for agent_id in range(env.n_agents):
            
    #         action = orca_planner.predict(agent_id,PenType.pen_Extrav)
    #         actions.append(action)

    #     action = np.array(actions)
    #     # print(action)
    #     # input()
    #     env.step(action)
    #     video_recorder.record(env)

    # video_recorder.save("test_Extrav.mp4")

    # env.reset()
    # video_recorder.init(env)
    # for i in range(50):
    #     actions = []
    #     for agent_id in range(env.n_agents):
            
    #         action = orca_planner.predict(agent_id,PenType.pen_Psych)
    #         actions.append(action)

    #     action = np.array(actions)
    #     # print(action)
    #     # input()
    #     env.step(action)
    #     video_recorder.record(env)
    # video_recorder.save("test_psych.mp4")

    env.reset()
    video_recorder.init(env)
    h_types = []
    for i in range(env.n_agents):
        h_types.append(random.choice(list(PenType))) 
    print(h_types)
    for i in range(50):
        actions = []
        for agent_id in range(env.n_agents):
            
            action = orca_planner.predict(agent_id,h_types[agent_id])
            actions.append(action)

        action = np.array(actions)
        # print(action)
        # input()
        env.step(action)
        video_recorder.record(env)

    video_recorder.save("test_mix_7.mp4")
