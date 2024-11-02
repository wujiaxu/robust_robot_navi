from harl.common.base_logger import BaseLogger
from harl.envs.robot_crowd_sim.utils.monitor import InfoMonitor

class CrowdEnvLogger(BaseLogger):

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)
        self.eval_agent_num = self.train_agent_num = env_args["human_num"]+env_args["robot_num"]

        self.train_info_moniter = InfoMonitor(env_num=algo_args["train"]["n_rollout_threads"],
                                              human_num=env_args["human_num"],
                                              robot_num=env_args["robot_num"],
                                              time_step=0.25,
                                              max_episode_len=env_args["max_episode_length"])
        self.eval_info_moniter = InfoMonitor(env_num=algo_args["eval"]["n_eval_rollout_threads"],
                                              human_num=env_args["human_num"],
                                              robot_num=env_args["robot_num"],
                                              time_step=0.25,
                                              max_episode_len=env_args["max_episode_length"])

    def get_task_name(self):
        return f"{self.env_args['scenario']}-{self.env_args['task']}"
    
    def per_step(self, data):
        (
        obs,
        share_obs,
        rewards,
        dones,
        infos,
        available_actions,
        values,
        actions,
        action_log_probs,
        rnn_states,
        rnn_states_critic,
        aux_rewards,
        ) = data

        self.train_info_moniter.saveInfoVecEnv(infos)

        return super().per_step(data)
    
    def lagrange_per_step(self,cost_limit,
                                cost,
                                balancing_factor,
                                pid_i):
        print("L:{},C:{},lamdba:{},pid_i:{}".format(cost_limit,cost,balancing_factor,pid_i))
        lagrange_info = {
            "value lower limit":-cost_limit,
            "current value":-cost,
            "lambda":balancing_factor,
            "error_integration":pid_i
        }
        for k, v in lagrange_info.items():
            lagrangian_k = "lagrangian/" + k
            self.writter.add_scalars(lagrangian_k, {lagrangian_k: v}, self.total_num_steps)
        return
    
    def log_train(self, actor_train_infos, critic_train_info,discrim_train_info=None):
        sr,cr,tr,fi,nt_mean,nt_std,_,_,_ = self.train_info_moniter.evaluateRobotInfo()
        print(sr,cr,tr,fi,nt_mean,nt_std)
        robot_navi_performance_info = {"success_rate": sr, 
                 "collision_rate":cr,
                 "timeout_rate":tr,
                 "frequency_of_invasion":fi,
                 "average_navi_time":nt_mean}
        for k, v in robot_navi_performance_info.items():
            robot_navi_performance_k = "robot_navi_performance/" + k
            self.writter.add_scalars(robot_navi_performance_k, {robot_navi_performance_k: v}, self.total_num_steps)
        
        sr,cr,tr,fi,nt_mean,nt_std = self.train_info_moniter.evaluateCrowdInfo()
        print(sr,cr,tr,fi,nt_mean,nt_std)
        crowd_navi_performance_info =  {"success_rate": sr, 
                 "collision_rate":cr,
                 "timeout_rate":tr,
                 "frequency_of_invasion":fi,
                 "average_navi_time":nt_mean}
        for k, v in crowd_navi_performance_info.items():
            crowd_navi_performance_info_k = "crowd_navi_performance/" + k
            self.writter.add_scalars(crowd_navi_performance_info_k, {crowd_navi_performance_info_k: v}, self.total_num_steps)
       
        return super().log_train(actor_train_infos, critic_train_info,discrim_train_info)
    
    def test_per_step(self,eval_data):
        (
        eval_obs,
        eval_share_obs,
        eval_rewards,
        eval_dones,
        eval_infos,
        eval_available_actions,
        ) = eval_data

        self.eval_info_moniter.saveInfoVecEnv(eval_infos)
        return 
    def eval_per_step(self, eval_data):
        """
        add pushing info
        for eval_info in eval_infos:
                robot_info, crowd_info = eval_info[0],eval_info[1]
                print(robot_info["episode_info"],crowd_info["episode_info"])
        """
        (
        eval_obs,
        eval_share_obs,
        eval_rewards,
        eval_dones,
        eval_infos,
        eval_available_actions,
        ) = eval_data
        
        self.eval_info_moniter.saveInfoVecEnv(eval_infos)
        return super().eval_per_step(eval_data)
    
    def test_log(self,eval_episode):
        sr,cr,tr,fi,nt_mean,nt_std,dist_normal,dist_danger,fd = self.eval_info_moniter.evaluateRobotInfo()
        print(sr,cr,tr,fi,nt_mean,nt_std,dist_normal,dist_danger,fd)
        self.eval_info_moniter.reset_episode_info()
        eval_robot_navi_performance_info = {"success_rate": sr, 
                 "collision_rate":cr,
                 "timeout_rate":tr,
                 "frequency_of_invasion":fi,
                 "average_navi_time":nt_mean,
                 "dist_to_ped":dist_normal,
                 "dist_to_danger_ped":dist_danger,
                 "frequency_of_danger":fd}
        for k, v in eval_robot_navi_performance_info.items():
            eval_robot_navi_performance_info_k = "test_robot_navi_performance/" + k
            self.writter.add_scalars(eval_robot_navi_performance_info_k, 
                                     {eval_robot_navi_performance_info_k: v},eval_episode)
        
        sr,cr,tr,fi,nt_mean,nt_std = self.eval_info_moniter.evaluateCrowdInfo()
        print(sr,cr,tr,fi,nt_mean,nt_std)
        eval_crowd_navi_performance_info = {"success_rate": sr, 
                 "collision_rate":cr,
                 "timeout_rate":tr,
                 "frequency_of_invasion":fi,
                 "average_navi_time":nt_mean}
        for k, v in eval_crowd_navi_performance_info.items():
            eval_crowd_navi_performance_info_k = "test_crowd_navi_performance/" + k
            self.writter.add_scalars(eval_crowd_navi_performance_info_k, 
                                     {eval_crowd_navi_performance_info_k: v}, eval_episode)
            
        return 
    def eval_log(self, eval_episode):
        """
        sum(self.crowd_success)/len(self.crowd_success),
            sum(self.crowd_collide)/len(self.crowd_collide),
            sum(self.crowd_timeout)/len(self.crowd_timeout),
            sum(self.crowd_invasion)/len(self.crowd_invasion),
            np.mean(list(self.crowd_time)),
            np.std(list(self.crowd_time))
        """
        
        sr,cr,tr,fi,nt_mean,nt_std,dist_normal,dist_danger,fd = self.eval_info_moniter.evaluateRobotInfo()
        print(sr,cr,tr,fi,nt_mean,nt_std,dist_normal,dist_danger,fd)
        self.eval_info_moniter.reset_episode_info()
        eval_robot_navi_performance_info = {"success_rate": sr, 
                 "collision_rate":cr,
                 "timeout_rate":tr,
                 "frequency_of_invasion":fi,
                 "average_navi_time":nt_mean,
                 "dist_to_ped":dist_normal,
                 "dist_to_danger_ped":dist_danger,
                 "frequency_of_danger":fd}
        for k, v in eval_robot_navi_performance_info.items():
            eval_robot_navi_performance_info_k = "eval_robot_navi_performance/" + k
            self.writter.add_scalars(eval_robot_navi_performance_info_k, 
                                     {eval_robot_navi_performance_info_k: v}, self.total_num_steps)
        
        sr,cr,tr,fi,nt_mean,nt_std = self.eval_info_moniter.evaluateCrowdInfo()
        print(sr,cr,tr,fi,nt_mean,nt_std)
        eval_crowd_navi_performance_info = {"success_rate": sr, 
                 "collision_rate":cr,
                 "timeout_rate":tr,
                 "frequency_of_invasion":fi,
                 "average_navi_time":nt_mean}
        for k, v in eval_crowd_navi_performance_info.items():
            eval_crowd_navi_performance_info_k = "eval_crowd_navi_performance/" + k
            self.writter.add_scalars(eval_crowd_navi_performance_info_k, 
                                     {eval_crowd_navi_performance_info_k: v}, self.total_num_steps)
        return super().eval_log(eval_episode)
