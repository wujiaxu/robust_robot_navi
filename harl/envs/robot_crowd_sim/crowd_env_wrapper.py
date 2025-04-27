from ast import Tuple
import typing as tp
from harl.envs.robot_crowd_sim.crowd_env import RobotCrowdSim
from crowd_navi_bench.crowd_policy.socialforce import SocialForce
import numpy as np
from harl.utils.trans_tools import _t2n
# from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer

class RobotCrowdSimWrapper(RobotCrowdSim):


    def __init__(self,
                 crowd_model,
                 args,
                 phase:str,
                 nenv:int=1,
                 thisSeed:int=0,
                 time_step:float = 0.25
                 ) -> None:
        super(RobotCrowdSimWrapper,self).__init__(args,phase,nenv,thisSeed,time_step)
        if self.human_policy == "ai":
            self.crowd_model = crowd_model
        elif self.human_policy == "SFM":
            self.crowd_model = SocialForce(self,self.args["sfm_v0"],self.args["sfm_sigma"])
        else:
            raise NotImplementedError
        self.wca_max = 3.5/1.25
        self.wca_min = (0.5+3.5)/2.5
        self.wg_max = 1.8
        self.wg_min = 0.1
        self.goal_weight = self.wg_min
        self.collision_weight =self.wca_max

        self.eval_obs = None
        self.eval_rnn_states = None
        self.eval_masks = None
        self.human_actions = []

        return 
    
    def _initHumanPreferenceVector(self,preference=None):
        task = np.zeros(self._human_preference_vector_dim)
        if self._use_human_preference:
            if self._human_preference_type == "category":
                if preference is None:
                    random_index = np.random.randint(0, self._human_preference_vector_dim)
                else:
                    assert preference<self._human_preference_vector_dim and isinstance(preference,int)
                    random_index = preference
                # Create a one-hot vector
                task[random_index] = 1
               
            elif self._human_preference_type == "ccp":
                assert self._human_preference_vector_dim == 2
                #TODO set reward weights from outside    
                # Create a one-hot vector
                self.goal_weight = np.random.uniform(self.wg_min,self.wg_max)
                self.collision_weight = np.random.uniform(self.wca_min,self.wca_max)
                task = np.array([(self.goal_weight-self.wg_min)/(self.wg_max-self.wg_min),
                                    (self.collision_weight-self.wca_min)/(self.wca_max-self.wca_min)])
            else:
                raise NotImplementedError
                task = np.random.randn(self._human_preference_vector_dim)
                task = task / np.linalg.norm(task)
        return task
    
    def reset(self, 
              seed: tp.Optional[int]=None, 
            preference: tp.Optional[tp.List[int]]=None,
            random_attribute_seed:tp.Optional[int]=None,
            agent_init: tp.Optional[tp.List[Tuple]]=None,
            random_attribute: tp.Optional[tp.List[Tuple]]=None):
        
        obs, share_obs,available_actions = super().reset(seed, preference, random_attribute_seed, agent_init, random_attribute)

        human_obs = obs[1:]
        self.eval_obs = np.expand_dims(np.array(human_obs), axis=0)
        if self.human_policy == "ai":
            self.eval_rnn_states = np.zeros(
                (
                    1,
                    self._human_num,
                    self.crowd_model[0].actor.recurrent_n,
                    self.crowd_model[0].actor.hidden_sizes[-1],
                ),
                dtype=np.float32,
            )
            self.eval_masks = np.ones(
                (1, self._human_num,1), dtype=np.float32
            )
        elif self.human_policy == "SFM":
            self.human_actions = []
        
        return obs[0]
    
    def step(self, robot_action:np.ndarray, rec=False, new_human=None):
        
        if self.human_policy == "ai":
            eval_actions_collector = [np.expand_dims(robot_action,0)]
            for agent_id in range(self._human_num):
                eval_actions, temp_rnn_state = self.crowd_model[agent_id].act(
                    self.eval_obs[:, agent_id],
                    self.eval_rnn_states[:, agent_id],
                    self.eval_masks[:, agent_id],
                    None,
                    deterministic=True,
                )
                self.eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))
            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)[0]
        elif self.human_policy == "SFM":
            eval_actions_collector = [robot_action]
            if self._num_episode_steps%5==0: #since 0.25 = 5X0.05
                # print("human decide")
                self.human_actions = []
                for agent_id in range(1,self._human_num+1):
                    action = self.crowd_model.predict(agent_id)
                    eval_actions_collector.append(action)
                    self.human_actions.append(action)
            else:
                # print("human move",len(self.human_actions))
                for action in self.human_actions:
                    eval_actions_collector.append(action)
            eval_actions = np.array(eval_actions_collector)

        (
            obs, share_obs, 
            rewards, 
            dones, 
            infos,
            available_actions
        ) = super().step(eval_actions, rec, new_human)

        human_obs = obs[1:]
        self.eval_obs = np.expand_dims(np.array(human_obs), axis=0)

        return obs[0], rewards[0], dones[0], infos[0]