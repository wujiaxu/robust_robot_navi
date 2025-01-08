import numpy as np
import matplotlib.pyplot as plt
from harl.envs.robot_crowd_sim.evaluation.evaluation import compute_winding_angle,compute_min_NN_distance
from harl.envs.robot_crowd_sim.evaluation.metrics import mmd_with_time_mask
from matplotlib.backends.backend_pdf import PdfPages

def process_real_world_data(file="/home/dl/wu_ws/robust_robot_navi/harl/envs/robot_crowd_sim/data_origin/2p3p_dataset.npy"):
    print("load data")
    data = np.load(file, allow_pickle=True)
    init_positions, trajs, wds, min_dists = data
    human_data_2p=trajs["2p"]
    human_data_3p=trajs["3p"]

    wds_2p = []
    wds_3p = []
    nn_2p = []
    nn_3p = []
    fig, ax = plt.subplots(figsize=(5, 5))
    for scene in human_data_2p:
        traj1,traj2 = scene[0],scene[1]
        ax.plot(traj1[:,0],traj1[:,1])
        ax.plot(traj2[:,0],traj2[:,1])
        wd = compute_winding_angle(traj1,traj2)
        wds_2p.append(wd)
        
    for scene in human_data_3p:
        traj1,traj2,traj3 = scene[0],scene[1],scene[2]
        wd12 = compute_winding_angle(traj1,traj2)
        wd23 = compute_winding_angle(traj2,traj3)
        wd31 = compute_winding_angle(traj3,traj1)
        ax.plot(traj1[:,0],traj1[:,1])
        ax.plot(traj2[:,0],traj2[:,1])
        ax.plot(traj3[:,0],traj3[:,1])
        wds_3p.append(wd12)
        wds_3p.append(wd23)
        wds_3p.append(wd31)

    ax.set_title(f"Collected Trajectories",fontsize=20)
    ax.set_xlabel("x [m]",fontsize=15)
    ax.set_ylabel("y [m]",fontsize=15)
    ax.legend()
    plt.xticks(fontsize=15)  # Larger font for x-axis ticks
    plt.yticks(fontsize=15) 
    plt.tight_layout()
    fig.savefig(f"results_figure/real_world_trajs.png")
    with PdfPages(f"results_figure/real_world_trajs.pdf") as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    return wds_2p,wds_3p,min_dists["2p"],min_dists["3p"]

def process_method(files):
    all_k_ades,all_k_fdes,all_wds,all_nn_dists,all_num_collision,all_num_reachgoal = [],[],[],[],[],[]
    for file in files:
        data = np.load(file, allow_pickle=True)
        #generated_human_data: scene: trajs X 20
        #evaluation: k_ades,k_fdes,wds,nn_dists
        generated_human_datas, evaluations = data
        print(len(evaluations))
        for generated_human_data,evaluation in zip(generated_human_datas,evaluations):
            #20 crowd trajs with mask
            # trajs = []
            # masks = []
            # for traj, mask in generated_human_data:
            #     pass
            # """
            # p: (*c, t, N, feature_dim)
            # q: (*c, t, N, feature_dim)
            # """
            # mmd_with_time_mask()
            k_ades,k_fdes,wds,nn_dists,num_collision,num_reachgoal = evaluation
            all_k_ades.extend(k_ades)
            all_k_fdes.extend(k_fdes) 
            all_wds.extend(wds)
            all_nn_dists.extend(nn_dists)
            all_num_collision.append(num_collision)
            all_num_reachgoal.append(num_reachgoal)
    
    data = {"ADE20":all_k_ades,"FDE20":all_k_fdes,"wd":all_wds,"nn-dist":all_nn_dists,"num_collision":all_num_collision,"num_reachgoal":all_num_reachgoal}
    
    print("ADE20",np.mean(all_k_ades),np.std(all_k_ades))
    print("FDE20",np.mean(all_k_fdes),np.std(all_k_fdes))
    return data

def process_scene(proposed,ccp,pt,scene_name,rw_wds,rw_mds):
    real_world_data = {"wd":rw_wds,"nn-dist":rw_mds}
    result_proposed= process_method(proposed)
    result_ccp= process_method(ccp)
    result_pt= process_method(pt)
    # print(len(result_pt["nn-dist"]),len(result_ccp["nn-dist"]))
    datasets = [result_proposed, 
                result_ccp,
                result_pt,
                ]
    for dataset in datasets:
        print("collision:",sum(dataset["num_collision"]))
    dataset_names = ["Proposed", "CCP","PT-ORCA"]
    list_names = ['ADE20', 'FDE20', 'wd', 'nn-dist']
    # Combine data for boxplots and histograms
    boxplot_data1 = [d['ADE20'] for d in datasets]
    boxplot_data2 = [d['FDE20'] for d in datasets]



    histogram_data1 = [item for d in datasets for item in d['wd']]
    histogram_data2 = [item for d in datasets for item in d['nn-dist']]

    # Plotting
    # fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # # Boxplot for 'list1'
    # axes[0, 0].boxplot(boxplot_data1, labels=dataset_names)
    # axes[0, 0].set_title("Boxplot for ADE20")

    # # Boxplot for 'list2'
    # axes[0, 1].boxplot(boxplot_data2, labels=dataset_names)
    # axes[0, 1].set_title("Boxplot for FDE20")

    # # Histogram for 'list3'
    # # for i, d in enumerate(datasets):
    # axes[1, 0].hist([d['wd'] for d in datasets], bins=10, range=(-2*np.pi,np.pi*2),alpha=0.7, label=dataset_names)
    # axes[1, 0].set_title("Histogram for wd")
    # axes[1, 0].legend()

    # # Histogram for 'list4'
    # # for i, d in enumerate(datasets):
    # axes[1, 1].hist([d['nn-dist'] for d in datasets], bins=10, range=(0.5,3.5),alpha=0.7, label=dataset_names)
    # axes[1, 1].set_title("Histogram for nn-dist")
    # axes[1, 1].legend()

    # # Adjust layout and show plot
    # plt.tight_layout()
    # plt.savefig("results_figure/mocap_ped_sim_evaluation_{}.png".format(scene_name))
    # plt.close()
    # Set histogram bins and range
    
    fontsize = 20
    # Plotting Boxplots and Saving Figures
    for list_name in list_names[:2]:
        # Collect data for the current list from all datasets
        list_data = [d[list_name] for d in datasets]

        # Boxplot
        fig_box = plt.figure(figsize=(5, 5))
        plt.boxplot(list_data, labels=dataset_names)
        plt.title(f"Boxplot for {list_name}",fontsize=fontsize)
        plt.ylabel("Erros [m]",fontsize=15)
        plt.xticks(fontsize=fontsize)  # Larger font for x-axis ticks
        plt.yticks(fontsize=15) 
        plt.tight_layout()
        fig_box.savefig(f"results_figure/mocap_ped_sim_evaluation_{scene_name}_{list_name}_boxplot.png")
        with PdfPages(f"results_figure/mocap_ped_sim_evaluation_{scene_name}_{list_name}_boxplot.pdf") as pdf:
            pdf.savefig(fig_box, bbox_inches='tight')
        plt.close(fig_box)  # Close the figure after saving

    data_ranges = {"wd":(-6.2,6.2),"nn-dist":(0.2,3.5)}
    dataset_names = ["Real world","Proposed", "CCP","PT-ORCA"]
    for list_name in list_names[2:]:
        # Collect data for the current list from all datasets
        list_data = [real_world_data[list_name]]+[d[list_name] for d in datasets]
        fig_hist, axes = plt.subplots(1, 4, figsize=(20, 4))  # 1 row, 3 columns for datasets

        # Plot histogram for each dataset in a different subplot
        for i, (data, ax) in enumerate(zip(list_data, axes)):
            # Plot 2D histogram
            n, bins, patches = ax.hist(data, bins=30, range=data_ranges[list_name], alpha=0.7, label=dataset_names[i])
            ax.set_title(f"{dataset_names[i]}",fontsize=fontsize)
            ax.set_xlabel(f"{list_name} [m]",fontsize=15)
            ax.set_ylabel("Frequency",fontsize=15)
            ax.legend()
        plt.xticks(fontsize=15)  # Larger font for x-axis ticks
        plt.yticks(fontsize=15) 
        plt.tight_layout()
        fig_hist.savefig(f"results_figure/mocap_ped_sim_evaluation_{scene_name}_{list_name}_histogram.png")
        with PdfPages(f"results_figure/mocap_ped_sim_evaluation_{scene_name}_{list_name}_histogram.pdf") as pdf:
            pdf.savefig(fig_hist, bbox_inches='tight')
        plt.close(fig_hist)  # Close the figure after saving

if __name__ == "__main__":
    data_path_2p_sim_proposed = "ped_sim/crowd_env/crowd_navi/robot_crowd_happo/ai_crowdsim_2p_rvs_6c_room256/seed-00001-2024-12-14-23-07-48/logs/generated_data.npy"
    data_path_3p_sim_proposed = "ped_sim/crowd_env/crowd_navi/robot_crowd_happo/ai_crowdsim_3p_rvs_6c_room256/seed-00001-2024-12-14-23-07-53/logs/generated_data.npy"
    data_path_2p_sim_ccp = "ped_sim/crowd_env_ccp/crowd_navi/robot_crowd_ppo/ai_crowdsim_2p_rvs_ccp_room256/seed-00001-2024-12-14-23-07-58/logs/generated_data.npy"
    data_path_3p_sim_ccp = "ped_sim/crowd_env_ccp/crowd_navi/robot_crowd_ppo/ai_crowdsim_3p_rvs_ccp_room256/seed-00001-2024-12-14-23-37-42/logs/generated_data.npy"
    data_path_2p_sim_pt = "ped_sim/crowd_env/crowd_navi/robot_crowd_happo/ad_hoc_crowdsim_2p_rvs_pt_room256/seed-00001-2024-12-14-23-08-08/logs/generated_data.npy"
    data_path_3p_sim_pt = "ped_sim/crowd_env/crowd_navi/robot_crowd_happo/ad_hoc_crowdsim_3p_rvs_pt_room256/seed-00001-2024-12-14-23-08-13/logs/generated_data.npy"
    
    wds_2p,wds_3p,min_dists_2p,min_dists_3p = process_real_world_data()
    # process_scene(
    #     [data_path_2p_sim_proposed],
    #     [data_path_2p_sim_ccp],
    #     [data_path_2p_sim_pt],
    #     "2p",
    #     wds_2p,
    #     min_dists_2p
    # )
    # process_scene(
    #     [data_path_3p_sim_proposed],
    #     [data_path_3p_sim_ccp],
    #     [data_path_3p_sim_pt],
    #     "3p",
    #     wds_3p,
    #     min_dists_3p
    # )
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'
    })
    process_scene(
        [data_path_2p_sim_proposed,data_path_3p_sim_proposed],
        [data_path_2p_sim_ccp,data_path_3p_sim_ccp],
        [data_path_2p_sim_pt,data_path_3p_sim_pt],
        "all",
        wds_2p+wds_3p,
        min_dists_2p+min_dists_3p
    )