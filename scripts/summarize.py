
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from harl.utils.configs_tools import find_seed_directories
from scipy import stats

def process_result_dir(root,dir):
    dir= Path(root)/Path(dir)
    log_dir_str = find_seed_directories(dir,1)[0]+"/logs"
    file = Path(log_dir_str)/"summary.json"
    with open(file, encoding="utf-8") as file:
        results = json.load(file)

    average_time = None
    success_rate = None
    collision_rate = None
    for key in results.keys():
        if "test_robot_navi_performance/average_navi_time" in key:
            average_time = np.array(results[key])[:,-1]
        if "test_robot_navi_performance/success_rate" in key:
            success_rate = results[key][-1][-1]
        if "test_robot_navi_performance/collision_rate" in key:
            collision_rate = np.array(results[key])[:,-1]
        if "test_robot_navi_performance/timeout_rate" in key:
            timeout_rate = np.array(results[key])[:,-1]

    return average_time,success_rate,collision_rate,timeout_rate


def perform_z_test(results1,results2,N=500):
    result_model1  = 0
    result_model2  = 0
    n_model1 = 0
    n_model2 = 0 
    for r1 in results1:
        result_model1+=r1*N
        n_model1+=N
    for r2 in results2:
        result_model2+=r2*N
        n_model2+=N

    p1 = result_model1 / n_model1
    p2 = result_model2 / n_model2

    # Combined proportion
    p_combined = (result_model1 + result_model2) / (n_model1 + n_model2)

    # Calculate z-statistic
    z_stat = (p1 - p2) / np.sqrt(p_combined * (1 - p_combined) * (1/n_model1 + 1/n_model2))

    # Calculate p-value from z-statistic (two-tailed test)
    p_value_one_tailed = 1 - stats.norm.cdf(z_stat)

    return p_value_one_tailed

def extract_data(file,keys):

    with open(file, encoding="utf-8") as file:
        results = json.load(file)
    data = []
    for key in keys:
        for k in results:
            if key in k:
                data.append(np.array(results[k])[:,-1])
    
    return data

def draw_discriminator_curve(root,dirs,labels,agent_num=5,figure_name="discriminator_loss"):

    keys = ["agent{}/discriminator_loss/agent{}/discriminator_loss".format(i,i) for i in range(agent_num)]
    fig, ax = plt.subplots(figsize=(8, 8))
    for dir,label in zip(dirs,labels):
        dir = Path(root)/Path(dir)
        log_dir_str = find_seed_directories(dir,1)[0]+"/logs"
        loss_per_agent = extract_data(Path(log_dir_str)/"summary.json",keys)
        average_per_step = np.mean(np.array(loss_per_agent),axis=0)
        plt.plot([i for i in range(average_per_step.shape[0])],average_per_step,label=label)

    plt.legend(loc='upper center', bbox_to_anchor=(.5, -.05), ncol=3,fontsize=10)
    with PdfPages("result_figures/{}.pdf".format(figure_name)) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    plt.close(fig)  # Close the figure to free memory

    return 

def draw_crowd_success_rate(root,dirs,labels,figure_name="success_rate"):
    key = "eval_crowd_navi_performance/success_rate"
    fig, ax = plt.subplots(figsize=(8, 8))
    for dir,label in zip(dirs,labels):
        dir = Path(root)/Path(dir)
        log_dir_str = find_seed_directories(dir,1)[0]+"/logs"
        success_rate = extract_data(Path(log_dir_str)/"summary.json",[key])[0]
        plt.plot([i for i in range(success_rate.shape[0])],success_rate,label=label)
    plt.legend(loc='upper center', bbox_to_anchor=(.5, -.05), ncol=3,fontsize=10)
    with PdfPages("result_figures/{}.pdf".format(figure_name)) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    plt.close(fig)  # Close the figure to free memory

    return

def draw_robot_danger_dist(root,dir1,dir2,labels,figure_name="dist_to_danger_ped_with_vis_vs_wo_vis"):
    key = "test_robot_navi_performance/dist_to_danger_ped"
    fig, ax = plt.subplots(figsize=(8, 8))
    
    
    dir1= Path(root)/Path(dir1)
    dir2= Path(root)/Path(dir2)
    log_dir_str1 = find_seed_directories(dir1,1)[0]+"/logs"
    log_dir_str2 = find_seed_directories(dir2,1)[0]+"/logs"
    dist1 = extract_data(Path(log_dir_str1)/"summary.json",[key])[0]
    dist2 = extract_data(Path(log_dir_str2)/"summary.json",[key])[0]
    dist1_non_zero = dist1[dist1!=0]
    dist2_non_zero = dist2[dist2!=0]
    t_stat, p_value = stats.ttest_ind(dist1_non_zero, dist2_non_zero,alternative="greater")

    # Plot histograms for both model results
    ax.hist(dist1_non_zero, bins=15, alpha=0.5, label=labels[0], color='blue')
    ax.hist(dist2_non_zero, bins=15, alpha=0.5, label=labels[1], color='orange')

    # Display p-value on the plot
    p_text = f'p-value: {p_value:.5f}'
    ax.text(0.45, 0.99, p_text, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, fontsize=12)

    # Set title and labels
    # ax.set_title('Histogram of Results for Model 1 and Model 2')
    ax.set_xlabel('Test Results')
    ax.set_ylabel('Frequency')

    # Add a legend
    ax.legend()

    with PdfPages("result_figures/{}.pdf".format(figure_name)) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    plt.close(fig)  # Close the figure to free memory

    return

def draw_robot_danger_frequency(root,dir1,dir2,labels,
                                figure_name="frequency_closing_danger_ped_with_vis_vs_wo_vis"):
    key = "test_robot_navi_performance/frequency_of_danger"
    fig, ax = plt.subplots(figsize=(8, 8))
    
    
    dir1= Path(root)/Path(dir1)
    dir2= Path(root)/Path(dir2)
    log_dir_str1 = find_seed_directories(dir1,1)[0]+"/logs"
    log_dir_str2 = find_seed_directories(dir2,1)[0]+"/logs"
    dist1 = extract_data(Path(log_dir_str1)/"summary.json",[key])[0]
    dist2 = extract_data(Path(log_dir_str2)/"summary.json",[key])[0]
    # dist1_non_zero = dist1[dist1!=0]
    # dist2_non_zero = dist2[dist2!=0]
    t_stat, p_value = stats.ttest_ind(dist1, dist2,alternative="less")

    # Plot histograms for both model results
    ax.hist(dist1, bins=15, alpha=0.5, label=labels[0], color='blue')
    ax.hist(dist2, bins=15, alpha=0.5, label=labels[1], color='orange')

    # Display p-value on the plot
    p_text = f'p-value: {p_value:.5f}'
    ax.text(0.45, 0.99, p_text, verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, fontsize=12)

    # Set title and labels
    # ax.set_title('Histogram of Results for Model 1 and Model 2')
    ax.set_xlabel('Test Results')
    ax.set_ylabel('Frequency')

    # Add a legend
    ax.legend()

    with PdfPages("result_figures/{}.pdf".format(figure_name)) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    plt.close(fig)  # Close the figure to free memory

    return

def perform_t_test(results1,results2,alternative="less"):
    """
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.
    """
    
    
    t_stat, p_value = stats.ttest_ind(np.concatenate(results1),
                                       np.concatenate(results2),
                                       alternative=alternative)

    return p_value

def draw_crowd_collision_rate(root,dirs,labels,figure_name="collision_rate"):
    key = "eval_crowd_navi_performance/collision_rate"
    fig, ax = plt.subplots(figsize=(8, 8))
    for dir,label in zip(dirs,labels):
        dir = Path(root)/Path(dir)
        log_dir_str = find_seed_directories(dir,1)[0]+"/logs"
        success_rate = extract_data(Path(log_dir_str)/"summary.json",[key])[0]
        plt.plot([i for i in range(success_rate.shape[0])],success_rate,label=label)
    plt.legend(loc='upper center', bbox_to_anchor=(.5, -.05), ncol=3,fontsize=10)
    with PdfPages("result_figures/{}.pdf".format(figure_name)) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    plt.close(fig)  # Close the figure to free memory

    return

def draw_lagrange_curve(root,dirs,figure_name="lagrange_multiplier"):

    keys = [
        "lagrangian/value lower limit/lagrangian/value lower limit",
        "/lagrangian/current value/lagrangian/current value",
        "lagrangian/lambda/lagrangian/lambda",
    ]
    labels =[
        "constraints",
        "current value",
        "lagrange multiplier"
    ]   
    fig, ax = plt.subplots(figsize=(8, 8))
    for dir,label in zip(dirs,labels):
        dir = Path(root)/Path(dir)
        log_dir_str = find_seed_directories(dir,1)[0]+"/logs"
        curves = extract_data(Path(log_dir_str)/"summary.json",keys)
        for label , curve in zip(labels,curves):
            plt.plot([i for i in range(curve.shape[0])],curve,label=label)
    plt.legend(loc='upper center', bbox_to_anchor=(.5, -.05), ncol=3,fontsize=10)
    with PdfPages("result_figures/{}.pdf".format(figure_name)) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    plt.close(fig)  # Close the figure to free memory

    return

def test_robot_navi_room361():
    print("room361")
    homogeneous = ["happo_5p_sp_rvs_room361",
                   "happo_5p_sp_rvs_circlecross",
            "sfm_5_1point5_r361_5p",
            "sfm_10_03_cc_5p",
            "sfm_10_03_r361_5p"]
    heterogeneous = ["c090_happo_5p_3c_rvs_room361",
                     "c090_happo_5p_3c_rvs_circlecross"]
    test_target = {
        "train_on_ai_090_4p_3c_rvs_room361":[
            "c090_happo_5p_3c_rvs_room361",
            "c090_happo_5p_3c_rvs_circlecross",
            "happo_5p_sp_rvs_room361",
            "happo_5p_sp_rvs_circlecross",
            "sfm_5_1point5_r361_5p",
            "sfm_10_03_cc_5p",
            "sfm_10_03_r361_5p",
        ],
        "train_on_ai_4p_sp_rvs_room361":[
            "c090_happo_5p_3c_rvs_room361",
            "c090_happo_5p_3c_rvs_circlecross",
            "happo_5p_sp_rvs_room361",
            "happo_5p_sp_rvs_circlecross",
            "sfm_5_1point5_r361_5p",
            "sfm_10_03_cc_5p",
            "sfm_10_03_r361_5p",
        ],
        "train_on_sfm_crowd_room361":[
            "c090_happo_5p_3c_rvs_room361",
            "c090_happo_5p_3c_rvs_circlecross",
            "happo_5p_sp_rvs_room361",
            "happo_5p_sp_rvs_circlecross",
            "sfm_5_1point5_r361_5p",
            "sfm_10_03_cc_5p",
            "sfm_10_03_r361_5p",
        ],
    }
    test_success_rates={
        "train_on_ai_090_4p_3c_rvs_room361":{},
        "train_on_ai_4p_sp_rvs_room361":{},
        "train_on_sfm_crowd_room361":{}
    }
    test_collision_rates={
        "train_on_ai_090_4p_3c_rvs_room361":{},
        "train_on_ai_4p_sp_rvs_room361":{},
        "train_on_sfm_crowd_room361":{}
    }
    test_ave_navitimes={
        "train_on_ai_090_4p_3c_rvs_room361":{},
        "train_on_ai_4p_sp_rvs_room361":{},
        "train_on_sfm_crowd_room361":{}
    }
    test_navitimes={
        "train_on_ai_090_4p_3c_rvs_room361":{},
        "train_on_ai_4p_sp_rvs_room361":{},
        "train_on_sfm_crowd_room361":{}
    }
    for model in test_target:
        for d in test_target[model]:
            (average_time,
             success_rate,
             collision_rate,
             timeout_rate
             ) = process_result_dir(
                 "crowd_navi_bench/results/crowd_env/crowd_navi/robot_crowd_happo",
                 model+"_vs_"+d
             )
            # print(np.sum(np.logical_and(average_time[1:]==average_time[:-1],
            #                              collision_rate[1:]>collision_rate[:-1]) ),success_rate)
            if np.logical_or(collision_rate[0]!=0,
                                timeout_rate[0]!=0):
                success_episode_id = []
            else:
                success_episode_id = [0]
            fail_cases = np.logical_or(collision_rate[1:]>collision_rate[:-1],
                                timeout_rate[1:]>timeout_rate[:-1])
            for i in range(len(fail_cases)):
                if not fail_cases[i]:
                    success_episode_id.append(i)
            # print(model,d,average_time[-1],success_episode_id)
            average_time = average_time[success_episode_id]
            test_collision_rates[model][d] = collision_rate[-1]
            test_success_rates[model][d] = success_rate
            test_ave_navitimes[model][d] = average_time[-1]
            average_time_0 = np.zeros_like(average_time)
            average_time_0[1:] = average_time[:-1]
            navi_time = np.array(
                [i for i in range(1,average_time.shape[0]+1)]
                )*average_time-average_time_0
            test_navitimes[model][d] = navi_time
    print(test_success_rates)
    print(test_collision_rates)
    print(test_ave_navitimes)
    for k1 in test_success_rates:
        for k2 in test_success_rates:
            if k1==k2:continue
            results1 = []
            results2 = []
            #homogeneous 
            for k in homogeneous:
                results1.append(test_success_rates[k1][k])
            for k in test_success_rates[k2]:
                results2.append(test_success_rates[k2][k])
            p_value = perform_z_test(results1,results2)
            print("homogeneous success_rate",k1,k2,p_value)
            results1 = []
            results2 = []
            #heterogeneous
            for k in homogeneous+heterogeneous:
                results1.append(test_success_rates[k1][k])
            for k in test_success_rates[k2]:
                results2.append(test_success_rates[k2][k])
            p_value = perform_z_test(results1,results2)
            print("heterogeneous success_rate",k1,k2,p_value)
            results1 = []
            results2 = []
            for k in homogeneous+heterogeneous:
                results1.append(test_success_rates[k1][k])
            for k in test_success_rates[k2]:
                results2.append(test_success_rates[k2][k])
            p_value = perform_z_test(results1,results2)
            print("all success_rate",k1,k2,p_value)
    for k1 in test_navitimes:
        for k2 in test_navitimes:
            if k1==k2:continue
            results1 = []
            results2 = []
            for k in homogeneous:
                results1.append(test_navitimes[k1][k])
            for k in homogeneous:
                results2.append(test_navitimes[k2][k])
            p_value = perform_t_test(results1,results2)
            print("homogeneous navi_time",k1,k2,p_value)
            results1 = []
            results2 = []
            for k in heterogeneous:
                results1.append(test_navitimes[k1][k])
            for k in heterogeneous:
                results2.append(test_navitimes[k2][k])
            p_value = perform_t_test(results1,results2)
            print("heterogeneous navi_time",k1,k2,p_value)
            results1 = []
            results2 = []
            for k in homogeneous+heterogeneous:
                results1.append(test_navitimes[k1][k])
            for k in homogeneous+heterogeneous:
                results2.append(test_navitimes[k2][k])
            p_value = perform_t_test(results1,results2)
            print("all navi_time",k1,k2,p_value)
    # for k1 in test_collision_rates:
    #     for k2 in test_collision_rates:
    #         if k1==k2:continue
    #         results1 = []
    #         results2 = []
    #         for k in test_collision_rates[k1]:
    #             results1.append(test_collision_rates[k1][k])
    #         for k in test_collision_rates[k2]:
    #             results2.append(test_collision_rates[k2][k])
    #         p_value = perform_z_test(results1,results2)
    #         print("collision_rate",k1,k2,p_value)

    return

def test_robot_navi_circlecross():
    print("circlecross")
    homogeneous = ["happo_5p_sp_rvs_room361",
                   "happo_5p_sp_rvs_circlecross",
            "sfm_5_1point5_cc_5p",
            "sfm_10_03_cc_5p",
            "sfm_10_03_r361_5p"]
    heterogeneous = ["c090_happo_5p_3c_rvs_room361",
                     "c090_happo_5p_3c_rvs_circlecross"]
    test_target = {
        "train_on_ai_090_4p_3c_rvs_circlecross":[
            "c090_happo_5p_3c_rvs_room361",
            "c090_happo_5p_3c_rvs_circlecross",
            "happo_5p_sp_rvs_room361",
            "happo_5p_sp_rvs_circlecross",
            "sfm_5_1point5_cc_5p",
            "sfm_10_03_cc_5p",
            "sfm_10_03_r361_5p",
        ],
        # "train_on_ai_090_4p_6c_rvs_circlecross":[
        #     "c090_happo_5p_3c_rvs_room361",
        #     "c090_happo_5p_3c_rvs_circlecross",
        #     "happo_5p_sp_rvs_room361",
        #     "happo_5p_sp_rvs_circlecross",
        #     "sfm_5_1point5_cc_5p",
        #     "sfm_10_03_cc_5p",
        #     "sfm_10_03_r361_5p",
        # ],
        "train_on_ai_4p_sp_rvs_circlecross":[
            "c090_happo_5p_3c_rvs_room361",
            "c090_happo_5p_3c_rvs_circlecross",
            "happo_5p_sp_rvs_room361",
            "happo_5p_sp_rvs_circlecross",
            "sfm_5_1point5_cc_5p",
            "sfm_10_03_cc_5p",
            "sfm_10_03_r361_5p",
        ],
        "train_on_sfm_crowd":[
            "c090_happo_5p_3c_rvs_room361",
            "c090_happo_5p_3c_rvs_circlecross",
            "happo_5p_sp_rvs_room361",
            "happo_5p_sp_rvs_circlecross",
            "sfm_5_1point5_cc_5p",
            "sfm_10_03_cc_5p",
            "sfm_10_03_r361_5p",
        ],
    }
    test_success_rates={
        "train_on_ai_090_4p_3c_rvs_circlecross":{},
        "train_on_ai_4p_sp_rvs_circlecross":{},
        "train_on_sfm_crowd":{}
    }
    test_collision_rates={
        "train_on_ai_090_4p_3c_rvs_circlecross":{},
        "train_on_ai_4p_sp_rvs_circlecross":{},
        "train_on_sfm_crowd":{}
    }
    test_ave_navitimes={
        "train_on_ai_090_4p_3c_rvs_circlecross":{},
        "train_on_ai_4p_sp_rvs_circlecross":{},
        "train_on_sfm_crowd":{}
    }
    test_navitimes={
        "train_on_ai_090_4p_3c_rvs_circlecross":{},
        "train_on_ai_4p_sp_rvs_circlecross":{},
        "train_on_sfm_crowd":{}
    }
    for model in test_target:
        for d in test_target[model]:
            (average_time,
             success_rate,
             collision_rate,
             timeout_rate
             ) = process_result_dir(
                 "crowd_navi_bench/results/crowd_env/crowd_navi/robot_crowd_happo",
                 model+"_vs_"+d
             )
            # print(np.sum(np.logical_and(average_time[1:]==average_time[:-1],
            #                              collision_rate[1:]>collision_rate[:-1]) ),success_rate)
            if np.logical_or(collision_rate[0]!=0,
                                timeout_rate[0]!=0):
                success_episode_id = []
            else:
                success_episode_id = [0]
            fail_cases = np.logical_or(collision_rate[1:]>collision_rate[:-1],
                                timeout_rate[1:]>timeout_rate[:-1])
            for i in range(len(fail_cases)):
                if not fail_cases[i]:
                    success_episode_id.append(i)
            # print(model,d,average_time[-1],success_episode_id)
            average_time = average_time[success_episode_id]
            test_collision_rates[model][d] = collision_rate[-1]
            test_success_rates[model][d] = success_rate
            test_ave_navitimes[model][d] = average_time[-1]
            average_time_0 = np.zeros_like(average_time)
            average_time_0[1:] = average_time[:-1]
            navi_time = np.array(
                [i for i in range(1,average_time.shape[0]+1)]
                )*average_time-average_time_0
            test_navitimes[model][d] = navi_time
    print(test_success_rates)
    print(test_collision_rates)
    print(test_ave_navitimes)
    for k1 in test_success_rates:
        for k2 in test_success_rates:
            if k1==k2:continue
            results1 = []
            results2 = []
            #homogeneous 
            for k in homogeneous:
                results1.append(test_success_rates[k1][k])
            for k in test_success_rates[k2]:
                results2.append(test_success_rates[k2][k])
            p_value = perform_z_test(results1,results2)
            print("homogeneous success_rate",k1,k2,p_value)
            #heterogeneous
            for k in homogeneous+heterogeneous:
                results1.append(test_success_rates[k1][k])
            for k in test_success_rates[k2]:
                results2.append(test_success_rates[k2][k])
            p_value = perform_z_test(results1,results2)
            print("heterogeneous success_rate",k1,k2,p_value)
            for k in homogeneous+heterogeneous:
                results1.append(test_success_rates[k1][k])
            for k in test_success_rates[k2]:
                results2.append(test_success_rates[k2][k])
            p_value = perform_z_test(results1,results2)
            print("all success_rate",k1,k2,p_value)
    for k1 in test_navitimes:
        for k2 in test_navitimes:
            if k1==k2:continue
            results1 = []
            results2 = []
            for k in homogeneous:
                results1.append(test_navitimes[k1][k])
            for k in homogeneous:
                results2.append(test_navitimes[k2][k])
            p_value = perform_t_test(results1,results2)
            print("homogeneous navi_time",k1,k2,p_value)
            results1 = []
            results2 = []
            for k in heterogeneous:
                results1.append(test_navitimes[k1][k])
            for k in heterogeneous:
                results2.append(test_navitimes[k2][k])
            p_value = perform_t_test(results1,results2)
            print("heterogeneous navi_time",k1,k2,p_value)
            results1 = []
            results2 = []
            for k in homogeneous+heterogeneous:
                results1.append(test_navitimes[k1][k])
            for k in homogeneous+heterogeneous:
                results2.append(test_navitimes[k2][k])
            p_value = perform_t_test(results1,results2)
            print("all navi_time",k1,k2,p_value)
    # for k1 in test_collision_rates:
    #     for k2 in test_collision_rates:
    #         if k1==k2:continue
    #         results1 = []
    #         results2 = []
    #         for k in test_collision_rates[k1]:
    #             results1.append(test_collision_rates[k1][k])
    #         for k in test_collision_rates[k2]:
    #             results2.append(test_collision_rates[k2][k])
    #         p_value = perform_z_test(results1,results2)
    #         print("collision_rate",k1,k2,p_value)

    return

def test_distracted_aware_robot_navi_room361():
    draw_robot_danger_dist(
        "crowd_navi_bench/results_vis_aware/crowd_env_vis/crowd_navi/robot_crowd_happo",
        "train_on_vis_1point5_1m_ai_090_4p_3c_rvs_room361_vs_1d_ai_env",
        "0d_wo_vis_trained_vs_1d_ai_env",
        ["distract_aware_1.5_1m","vanilla"],
        figure_name="dist_to_danger_ped_with_vis_vs_wo_vis_1point5_1m"
    )

    draw_robot_danger_frequency(
        "crowd_navi_bench/results_vis_aware/crowd_env_vis/crowd_navi/robot_crowd_happo",
        "train_on_vis_1point5_1m_ai_090_4p_3c_rvs_room361_vs_1d_ai_env",
        "0d_wo_vis_trained_vs_1d_ai_env",
        ["distract_aware_1.5_1m","vanilla"],
        figure_name="frequency_closing_danger_ped_with_vis_vs_wo_vis_1point5_1m"
    )

    draw_robot_danger_dist(
        "crowd_navi_bench/results_vis_aware/crowd_env_vis/crowd_navi/robot_crowd_happo",
        "train_on_vis_2point5_1m_ai_090_4p_3c_rvs_room361_vs_1d_ai_env",
        "0d_wo_vis_trained_vs_1d_ai_env",
        ["distract_aware_2.5_1m","vanilla"],
        figure_name="dist_to_danger_ped_with_vis_vs_wo_vis_2point5_1m"
    )

    draw_robot_danger_frequency(
        "crowd_navi_bench/results_vis_aware/crowd_env_vis/crowd_navi/robot_crowd_happo",
        "train_on_vis_2point5_1m_ai_090_4p_3c_rvs_room361_vs_1d_ai_env",
        "0d_wo_vis_trained_vs_1d_ai_env",
        ["distract_aware_2.5_1m","vanilla"],
        figure_name="frequency_closing_danger_ped_with_vis_vs_wo_vis_2point5_1m"
    )

    test_target = ["train_on_vis_1point5_1m_ai_090_4p_3c_rvs_room361_vs_1d_ai_env",
                   "train_on_vis_2point5_1m_ai_090_4p_3c_rvs_room361_vs_1d_ai_env",
                    "0d_wo_vis_trained_vs_1d_ai_env"]
    labels = {"train_on_vis_1point5_1m_ai_090_4p_3c_rvs_room361_vs_1d_ai_env":"proposed",
              "train_on_vis_2point5_1m_ai_090_4p_3c_rvs_room361_vs_1d_ai_env":"variant",
              "0d_wo_vis_trained_vs_1d_ai_env":"vanilla"}

    sr = {}
    cr = {}
    ant = {}
    nt = {}
    for target in test_target:
        (average_time,
            success_rate,
            collision_rate,
            timeout_rate
            ) = process_result_dir(
                "crowd_navi_bench/results_vis_aware/crowd_env_vis/crowd_navi/robot_crowd_happo",
                target
            )
        # print(np.sum(np.logical_and(average_time[1:]==average_time[:-1],
        #                              collision_rate[1:]>collision_rate[:-1]) ),success_rate)
        if np.logical_or(collision_rate[0]!=0,
                            timeout_rate[0]!=0):
            success_episode_id = []
        else:
            success_episode_id = [0]
        fail_cases = np.logical_or(collision_rate[1:]>collision_rate[:-1],
                            timeout_rate[1:]>timeout_rate[:-1])
        for i in range(len(fail_cases)):
            if not fail_cases[i]:
                success_episode_id.append(i)
        # print(model,d,average_time[-1],success_episode_id)
        average_time = average_time[success_episode_id]
        cr[labels[target]] = collision_rate[-1]
        sr[labels[target]] = success_rate
        ant[labels[target]] = average_time[-1]
        average_time_0 = np.zeros_like(average_time)
        average_time_0[1:] = average_time[:-1]
        navi_time = np.array(
            [i for i in range(1,average_time.shape[0]+1)]
            )*average_time-average_time_0
        nt[labels[target]] = navi_time
    print(sr)
    print(cr)
    print(ant)
    for k1 in sr:
        for k2 in sr:
            if k1==k2:continue
            p_value = perform_z_test([sr[k1]],[sr[k2]])
            print("all success_rate",k1,k2,p_value)
            p_value = perform_t_test([nt[k1]],[nt[k2]],"two-sided")
            print("all navi_time",k1,k2,p_value)

    return 

def test_cmdp_crowd_sim():
    draw_discriminator_curve(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        ["c0.85_happo_5p_3c_rvs_room361",
         "c0.90_happo_5p_3c_rvs_room361",
         "c0.95_happo_5p_3c_rvs_room361",
         "happo_5p_3c_rvs_room361"],
        ["optimality:0.85",
         "optimality:0.90",
         "optimality:0.95",
         "without constraint"],
        figure_name="discrimination_loss_5human_small_room_3c"
    )
    draw_discriminator_curve(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        ["c0.85_happo_5p_3c_rvs_circlecross",
         "c0.90_happo_5p_3c_rvs_circlecross",
         "c0.95_happo_5p_3c_rvs_circlecross",
         "happo_5p_3c_rvs_circlecross"],
        ["optimality:0.85",
         "optimality:0.90",
         "optimality:0.95","without constraint"],
        figure_name="discrimination_loss_5human_circle_cross_3c"
    )
    draw_discriminator_curve(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        ["c0.85_happo_10p_3c_rvs_circlecross",
         "c0.90_happo_10p_3c_rvs_circlecross",
         "c0.95_happo_10p_3c_rvs_circlecross",
         "happo_10p_3c_rvs_circlecross"],
        ["optimality:0.85",
         "optimality:0.90",
         "optimality:0.95","without constraint"],
        agent_num=10,
        figure_name="discrimination_loss_10human_circle_cross_3c"
    )
    draw_crowd_success_rate(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        ["c0.85_happo_5p_3c_rvs_room361",
         "c0.90_happo_5p_3c_rvs_room361",
         "c0.95_happo_5p_3c_rvs_room361",
         "happo_5p_3c_rvs_room361",
         "happo_5p_sp_rvs_room361"],
        ["optimality:0.85",
         "optimality:0.90",
         "optimality:0.95",
         "without constraint",
         "optimal"],
        figure_name="success_rate_5human_small_room_3c"
    )
    draw_crowd_success_rate(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        ["c0.85_happo_5p_3c_rvs_circlecross",
         "c0.90_happo_5p_3c_rvs_circlecross",
         "c0.95_happo_5p_3c_rvs_circlecross",
         "happo_5p_3c_rvs_circlecross",
         "happo_5p_sp_rvs_circlecross"],
        ["optimality:0.85",
         "optimality:0.90",
         "optimality:0.95","without constraint","optimal"],
        figure_name="success_rate_5human_circle_cross_3c"
    )
    draw_crowd_success_rate(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        ["c0.85_happo_10p_3c_rvs_circlecross",
         "c0.90_happo_10p_3c_rvs_circlecross",
         "c0.95_happo_10p_3c_rvs_circlecross",
         "happo_10p_3c_rvs_circlecross",
         "happo_10p_sp_rvs_circlecross"],
        ["optimality:0.85",
         "optimality:0.90",
         "optimality:0.95","without constraint","optimal"],
        figure_name="success_rate_10human_circle_cross_3c"
    )
    draw_crowd_collision_rate(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        ["c0.85_happo_5p_3c_rvs_room361",
         "c0.90_happo_5p_3c_rvs_room361",
         "c0.95_happo_5p_3c_rvs_room361",
         "happo_5p_3c_rvs_room361",
         "happo_5p_sp_rvs_room361"],
        ["optimality:0.85",
         "optimality:0.90",
         "optimality:0.95",
         "without constraint",
         "optimal"],
        figure_name="collision_rate_5human_small_room_3c"
    )
    draw_crowd_collision_rate(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        ["c0.85_happo_5p_3c_rvs_circlecross",
         "c0.90_happo_5p_3c_rvs_circlecross",
         "c0.95_happo_5p_3c_rvs_circlecross",
         "happo_5p_3c_rvs_circlecross",
         "happo_5p_sp_rvs_circlecross"],
        ["optimality:0.85",
         "optimality:0.90",
         "optimality:0.95","without constraint","optimal"],
        figure_name="collision_rate_5human_circle_cross_3c"
    )
    draw_crowd_collision_rate(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        ["c0.85_happo_10p_3c_rvs_circlecross",
         "c0.90_happo_10p_3c_rvs_circlecross",
         "c0.95_happo_10p_3c_rvs_circlecross",
         "happo_10p_3c_rvs_circlecross",
         "happo_10p_sp_rvs_circlecross"],
        ["optimality:0.85",
         "optimality:0.90",
         "optimality:0.95","without constraint","optimal"],
        figure_name="collision_rate_10human_circle_cross_3c"
    )
    return 

def test_crowd_sim_editable_factor_class():
    draw_discriminator_curve(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        ["c0.90_happo_5p_3c_rvs_circlecross",
         "c0.90_happo_5p_6c_rvs_circlecross",
         "c0.90_happo_5p_3c_rvs_room361",
         "c0.90_happo_5p_6c_rvs_room361"],
        ["circle_cross_3_class",
         "circle_cross_6_class",
         "small_room_3_class",
         "small_room_6_class",],
        figure_name="discrimination_loss_5human_3c_vs_6c"
    )
    draw_crowd_success_rate(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        ["c0.90_happo_5p_3c_rvs_circlecross",
         "c0.90_happo_5p_6c_rvs_circlecross",
         "c0.90_happo_5p_3c_rvs_room361",
         "c0.90_happo_5p_6c_rvs_room361"],
        ["circle_cross_3_class",
         "circle_cross_6_class",
         "small_room_3_class",
         "small_room_6_class",],
        figure_name="succees_rate_5human_3c_vs_6c"
    )

    return 

def test_crowd_sim_editable_factor_time():
    draw_discriminator_curve(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        [
         "c_happo_10p_3c_rvs_circlecross_long",
         "happo_10p_3c_rvs_circlecross"],
        ["time limit: 15[s]",
         "time limit: 12[s]"],
        agent_num=10,
        figure_name="discrimination_loss_10human_circle_cross_15s_vs_12s"
    )
    draw_crowd_success_rate(
        "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
        [
         "c_happo_10p_3c_rvs_circlecross_long",
         "happo_10p_3c_rvs_circlecross"],
        ["time limit: 15[s]",
         "time limit: 12[s]"],
        figure_name="success_rate_10human_circle_cross_15s_vs_12s"
    )
    return 

if __name__ =="__main__":
    # 1. compare with and without constraints
    # test_cmdp_crowd_sim()
    # shows: the proposed method well balanced performance and diversity across all scenarios

    # 2. compare different optimality same as 1

    # shows: the proposed method could use optimality factor to control the balance between performance and diversity

    # 3. compare different scene 1

    # shows: diversity objective was easier to optimize in scene with larger free space
    # # it probably since that larger scene provide human more degree of freedom on choosing different actions

    # 4. compare different time
    # test_crowd_sim_editable_factor_time()
    # shows: given larger time limit the proposed conditional policy can achieve higher diversity 

    # 5. different class number
    # test_crowd_sim_editable_factor_class()
    # shows: large number of category may hinder the training

    # 6 test different between model trained on different simulator
    # 6.1 room361
    test_robot_navi_room361()
    # # 6.2 circlecross
    test_robot_navi_circlecross()

    # 7 distance between robot and cell phone walker (with vis vs without vis)
    # test_distracted_aware_robot_navi_room361()

    # 8 Lagrange
    # draw_lagrange_curve(
    #     "results_seed_1/crowd_env/crowd_navi/robot_crowd_happo",
    #     ["c0.90_happo_5p_3c_rvs_room361"]
    # )
