
from harl.utils.configs_tools import save_config
import os
def create_config(
    save_dir = "./train_configs",
    critic_model_id = 0,
    human_pref_type_id = 0,
    scenario_id = 0,
    use_discriminator = True,
    human_num = 12,
    random_human_num = False,
    minimum_human_num = 2,
    human_preference_vector_dim= 3,
    num_env_steps=8000000,
    map_size=8,
    max_episode_length=48,
    intrinsic_reward_scale = 0.1,
    centralized_critic = True,  
    ):

    critic_model_choices = ["EGCL", "MLP", "CNN_1D"]
    down_sample_lidar_scan_bin = 720
    state_type = "FP"#EP, FP
    if centralized_critic and critic_model_choices[critic_model_id]=="EGCL":
        state_type = "EP"
    if critic_model_choices[critic_model_id]=="EGCL" or critic_model_choices[critic_model_id]=="MLP":
        down_sample_lidar_scan_bin = 11

    # env
    human_pref_type_choices = ["category","ccp"]
    human_preference_type=	human_pref_type_choices[human_pref_type_id]
    if human_preference_type == "ccp":
        main_algo = "robot_crowd_ppo"
        main_env = "crowd_env_ccp"
        human_preference_vector_dim = 2
    else:
        main_algo = "robot_crowd_happo"
        main_env = "crowd_env"
    scenario_choices = ["circle_cross","room_256","room_361","ucy_students"]
    scenario=scenario_choices[scenario_id]

    if centralized_critic:
        algo_name = "happo"
    else:
        algo_name = "ppo"
    if not use_discriminator:
        multi_modal_name = "sp"
    else:
        multi_modal_name = "{}c".format(human_preference_vector_dim) if human_preference_type=="category" else "ccp"

    if random_human_num:
        human_config_name = str(minimum_human_num) \
                    +"-" \
                    +str(human_num)
    else:
        human_config_name = str(human_num) \
                    +"-" \
                    +str(human_num)

    config_name = algo_name \
                    +"_"\
                    +critic_model_choices[critic_model_id]\
                    +"_"\
                    +human_config_name\
                    +"_"\
                    +multi_modal_name \
                    +"_"\
                    +"rvs_"+scenario.replace("_", "")+".json"

    config = {
        "algo_args":	{
            "algo":	{
                "centralized": centralized_critic,
                "action_aggregation":	"prod",
                "actor_num_mini_batch":	1,
                "clip_param":	0.2,
                "critic_epoch":	5,
                "critic_num_mini_batch":	1,
                "entropy_coef":	0.01,
                "fixed_order":	False,
                "gae_lambda":	0.95,
                "gamma":	0.99,
                "huber_delta":	10.0,
                "intrinsic_reward_scale":	intrinsic_reward_scale,
                "discriminator_loss_scale": 0.1,
                "max_grad_norm":	10.0,
                "ppo_epoch":	5,
                "share_param":	True,
                "use_clipped_value_loss":	True,
                "use_discriminator":	use_discriminator,
                "use_gae":	True,
                "use_huber_loss":	True,
                "use_max_grad_norm":	True,
                "use_policy_active_masks":	True,
                "value_loss_coef":	1
            },
            "device":	{
                "cuda":	True,
                "cuda_deterministic":	True,
                "torch_threads":	8
            },
            "eval":	{
                "eval_episodes":	10,
                "n_eval_rollout_threads":	9,
                "use_eval":	True
            },
            "logger":	{
                "log_dir":	"./results"
            },
            "model":	{
                "down_sample_lidar_scan_bin":down_sample_lidar_scan_bin,
                "activation_func":	"relu",
                "critic_lr":	0.0005,
                # "custom":	True,
                "base_model_name": critic_model_choices[critic_model_id],
                "data_chunk_length":	10,
                "gain":	0.01,
                "hidden_sizes":	[
                    128,
                    128
                ],
                "initialization_method":	"orthogonal_",
                "lr":	0.0005,
                "opti_eps":	1e-05,
                "recurrent_n":	1,
                "std_x_coef":	1,
                "std_y_coef":	0.5,
                "use_feature_normalization":	True,
                "use_naive_recurrent_policy":	False,
                "use_recurrent_policy":	True,
                "weight_decay":	0
            },
            "render":	{
                "render_episodes":	10,
                "use_render":	False
            },
            "seed":	{
                "seed":	1,
                "seed_specify":	True
            },
            "train":	{
                "episode_length":	40,
                "eval_interval":	25,
                "log_interval":	5,
                "model_dir":	None,
                "n_rollout_threads":	32,
                "num_env_steps":	num_env_steps,
                "use_linear_lr_decay":	True,
                "use_proper_time_limits":	True,
                "use_valuenorm":	True
            }
        },
        "env_args":	{
            "continuous_actions":	True,
            "discomfort_dist":	1.0,
            "discomfort_penalty_factor":	0.2,
            "goal_factor":	2,
            "goal_range":	0.3,
            "human_fov":	110,
            "human_n_scan":	720,
            "human_num":	human_num,
            "human_preference_type":	human_preference_type,
            "human_preference_vector_dim":	human_preference_vector_dim,
            "human_random_pref_v_and_size": True,
            "random_human_num":random_human_num,
            "minimum_human_num":minimum_human_num,
            "human_radius":	0.3,
            "human_rotation_constrain":	180,
            "human_v_pref":	1.0,
            "human_visible":	True,
            "laser_angle_resolute":	0.008726646,
            "laser_max_range":	4.0,
            "laser_min_range":	0.0,
            "map_size":	map_size,
            "max_episode_length":	max_episode_length,
            "n_laser":	720,
            "penalty_backward":	0.2,
            "penalty_collision":	-20,
            "reward_goal":	10,
            "robot_num":	0,
            "robot_radius":	0.3,
            "robot_rotation_constrain":	180,
            "robot_v_pref":	1.0,
            "robot_visible":	True,
            "robot_random_pref_v_and_size": False,
            "scenario":	scenario,
            "state_type":	state_type,
            "task":	"crowd_navi",
            "use_discriminator":	use_discriminator,
            "velo_factor":	0.2,
            "with_static_obstacle":	True
        },
        "main_args":	{
                "algo":	main_algo, 
                "cuda_device":	"cuda:2", #default
                "env":	main_env,
                "exp_name":	"robust_navi", #default
                "load_config":	""  #default
        }
    }
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_config(config["main_args"],config["algo_args"],config["env_args"],save_dir,config_name)
    return 

def create_pedsim_eval_config():
    create_config(save_dir = "./train_configs",critic_model_id = 0,
        human_pref_type_id = 0,
        scenario_id = 3,
        max_episode_length = 48,
        use_discriminator = False,
        human_num = 12,
        random_human_num = True,
        minimum_human_num = 6,
        human_preference_vector_dim= 3,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 0,
        human_pref_type_id = 0,
        scenario_id = 3,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 12,
        random_human_num = True,
        minimum_human_num = 6,
        human_preference_vector_dim= 3,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 0,
        human_pref_type_id = 0,
        scenario_id = 0,
        max_episode_length = 48,
        use_discriminator = False,
        human_num = 10,
        random_human_num = False,
        human_preference_vector_dim= 3,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 0,
        human_pref_type_id = 0,
        scenario_id = 0,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 10,
        random_human_num = False,
        minimum_human_num = 6,
        human_preference_vector_dim= 3,
        centralized_critic = True,  
    )

    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 2,
        max_episode_length = 48,
        use_discriminator = False,
        human_num = 5,
        random_human_num = False,
        human_preference_vector_dim= 3,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 2,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 5,
        random_human_num = False,
        human_preference_vector_dim= 3,
        centralized_critic = True,  
        intrinsic_reward_scale=0.1,
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 2,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 5,
        random_human_num = False,
        human_preference_vector_dim= 6,
        centralized_critic = True,  
        intrinsic_reward_scale=0.1,
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 2,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 5,
        random_human_num = False,
        human_preference_vector_dim= 12,
        centralized_critic = True,  
        intrinsic_reward_scale=0.1,
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 2,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 5,
        random_human_num = False,
        human_preference_vector_dim= 24,
        centralized_critic = True,  
        intrinsic_reward_scale=0.1,
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 1,
        max_episode_length = 48,
        use_discriminator = False,
        human_num = 6,
        random_human_num = False,
        human_preference_vector_dim= 6,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 1,
        max_episode_length = 48,
        use_discriminator = False,
        human_num = 8,
        random_human_num = False,
        human_preference_vector_dim= 6,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 1,
        max_episode_length = 48,
        use_discriminator = False,
        human_num = 10,
        random_human_num = False,
        human_preference_vector_dim= 6,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 1,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 6,
        random_human_num = False,
        human_preference_vector_dim= 6,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 1,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 8,
        random_human_num = False,
        human_preference_vector_dim= 6,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 1,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 10,
        random_human_num = False,
        human_preference_vector_dim= 6,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 1,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 6,
        random_human_num = False,
        human_preference_vector_dim= 3,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 1,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 8,
        random_human_num = False,
        human_preference_vector_dim= 3,
        centralized_critic = True,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 1,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 10,
        random_human_num = False,
        human_preference_vector_dim= 3,
        centralized_critic = True,  
    )


    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 0,
        max_episode_length = 48,
        use_discriminator = False,
        human_num = 7,
        random_human_num = False,
        human_preference_vector_dim= 3,
        centralized_critic = True,  
    )

    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 0,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 7,
        random_human_num = False,
        human_preference_vector_dim= 3,
        centralized_critic = True,  
    )

    return 

def create_navi_eval_config():

    return

def create_ablation_study_config():
    # create_config(save_dir = "./train_configs",critic_model_id = 1,
    #     human_pref_type_id = 0,
    #     scenario_id = 3,
    #     max_episode_length = 60,
    #     use_discriminator = False,
    #     human_num = 12,
    #     random_human_num = True,
    #     minimum_human_num = 6,
    #     human_preference_vector_dim= 3,
    #     centralized_critic = False,  
    # )
    # create_config(save_dir = "./train_configs",critic_model_id = 1,
    #     human_pref_type_id = 0,
    #     scenario_id = 3,
    #     max_episode_length = 60,
    #     use_discriminator = True,
    #     human_num = 12,
    #     random_human_num = True,
    #     minimum_human_num = 6,
    #     human_preference_vector_dim= 3,
    #     centralized_critic = False,  
    # )

    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 1,
        use_discriminator = True,
        human_num = 2,
        minimum_human_num = 2,
        human_preference_vector_dim= 6,
        centralized_critic = False,  
    )
    create_config(save_dir = "./train_configs",critic_model_id = 2,
        human_pref_type_id = 0,
        scenario_id = 1,
        use_discriminator = True,
        human_num = 3,
        minimum_human_num = 2,
        human_preference_vector_dim= 6,
        centralized_critic = False,  
    )


    return

def create_ccp_eval_config():

    create_config(
        save_dir = "./train_configs",
        critic_model_id = 2,
        human_pref_type_id = 1,
        scenario_id = 0, #2
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 5,
        random_human_num = False,
        minimum_human_num = 5,
        human_preference_vector_dim= 2,
        centralized_critic = False,  
        num_env_steps=4000000,
    )
    create_config(
        save_dir = "./train_configs",
        critic_model_id = 2,
        human_pref_type_id = 1,
        scenario_id = 2,
        max_episode_length = 48,
        use_discriminator = True,
        human_num = 5,
        random_human_num = False,
        minimum_human_num = 5,
        human_preference_vector_dim= 2,
        centralized_critic = False,  
        num_env_steps=4000000,
    )


    return 

if __name__ == "__main__":
    # create_pedsim_eval_config()
    create_ablation_study_config()
    # create_ccp_eval_config()