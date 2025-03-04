from tkinter import W
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
from .trajectory_utils import prediction_output_to_trajectories
#import visualization
from matplotlib import pyplot as plt
import pdb

def calculate_density_zero_centered(agents, W, H, s, ws):
    # Number of grid cells in each dimension
    rows = int(np.ceil(H / s))
    cols = int(np.ceil(W / s))
    
    # Initialize the density matrix
    density = np.zeros((rows, cols))
    
    # Half window size
    half_ws = ws / 2
    
    # Shift factor for zero-centered coordinates
    shift_x = W / 2
    shift_y = H / 2
    
    # Assign agents to grid cells within the window
    for agent in agents:
        x,y = agent.get_position()
        # Shift coordinates to positive space
        x_shifted = x + shift_x
        y_shifted = y + shift_y
        
        # Determine the range of grid indices affected by the window
        i_min = max(0, int((y_shifted - half_ws) // s))
        i_max = min(rows - 1, int((y_shifted + half_ws) // s))
        j_min = max(0, int((x_shifted - half_ws) // s))
        j_max = min(cols - 1, int((x_shifted + half_ws) // s))
        
        # Increment density for all cells within the window
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                density[i, j] += 1
    
    # Normalize by window area (optional)
    density /= (ws * ws)
    
    return np.max(density)

def compute_winding_angle(traj_a,traj_b):
    length = min(traj_a.shape[0],traj_b.shape[0])
    traj_a = traj_a[:length]
    traj_b = traj_b[:length]
    # winding_angle=0
    wabs = traj_a-traj_b
    angles = np.arctan2(wabs[:,1],wabs[:,0])
    # for frame in traj_a:
    #     t = frame[-1]
    #     corr_frame = traj_b[traj_b[:,2]==t,:]
    #     if corr_frame.size==0:continue
    #     wab = corr_frame[0,:2]-frame[:2]
    #     wabs.append(np.arctan2(wab[1],wab[0]))
    # wabs = np.array(wabs)
    # Unwrap angles to ensure continuity
    unwrapped_angles = np.unwrap(angles)

    # Calculate winding angle (total angular displacement)
    winding_angle = unwrapped_angles[-1] - unwrapped_angles[0]
    return winding_angle

def compute_min_NN_distance(traj_a,traj_b,mask):
    try:
    # winding_angle=0
        dists = np.linalg.norm(traj_a-traj_b,axis=-1)[mask==1]
        min_dist = np.min(dists)
    except Exception as e:
        print(traj_a,traj_b,mask)
        raise RuntimeError
    # for frame in traj_a:
    #     t = frame[-1]
    #     corr_frame = traj_b[traj_b[:,2]==t,:]
    #     if corr_frame.size==0:continue
    #     dist = corr_frame[0,:2]-frame[:2]
    #     dists.append(np.linalg.norm(dist))
    # print(dists)
    return min_dist

def compute_ade(predicted_trajs, gt_traj, mask, oper_nodes='mean'):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    
    return np.mean(error[mask==1])
    # elif oper_nodes=='sum':
    #     error = np.sum(error,axis=-1)
    # elif oper_nodes==None:
    #     final_error = final_error
    # else:
    #     raise('not supported operation mode on ade')
    # ade = np.mean(error, axis=-1)
    # return ade.flatten()


def compute_fde(predicted_trajs, gt_traj, oper_nodes='mean'):
    final_error = np.linalg.norm(predicted_trajs[-1] - gt_traj[-1], axis=-1)
    return final_error
    # if oper_nodes=='mean':
    #     final_error = np.mean(final_error, axis=-1)
    # elif oper_nodes=='sum':
    #     final_error = np.sum(final_error,axis=-1)
    # elif oper_nodes==None:
    #     final_error = final_error
    # else:
    #     raise('not supported operation mode on fde')
    # return final_error.flatten()


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[1]),
                                         range(obs_map.shape[0]),
                                         binary_dilation(obs_map.T, iterations=4),
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs


def compute_batch_statistics(prediction_output_dict,
                             dt,
                             max_hl,
                             ph,
                             node_type_enum,
                             kde=True,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False,
                             best_of=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] = {'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            #pdb.set_trace()
            #target_shape =
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of: 
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].extend(list(ade_errors))
            batch_error_dict[node.type]['fde'].extend(list(fde_errors))
            batch_error_dict[node.type]['kde'].extend([kde_ll])
            batch_error_dict[node.type]['obs_viols'].extend([obs_viols])

    return batch_error_dict

def compute_batch_statistics2(prediction_output,
                              gt,
                              best_of=True):
    ade_errors = compute_ade(prediction_output, gt)
    fde_errors = compute_fde(prediction_output, gt)
    if best_of: 
        ade_errors = np.min(ade_errors, keepdims=True)
        fde_errors = np.min(fde_errors, keepdims=True)
    return ade_errors,fde_errors
    
# def log_batch_errors(batch_errors_list, log_writer, namespace, curr_iter, bar_plot=[], box_plot=[]):
#     for node_type in batch_errors_list[0].keys():
#         for metric in batch_errors_list[0][node_type].keys():
#             metric_batch_error = []
#             for batch_errors in batch_errors_list:
#                 metric_batch_error.extend(batch_errors[node_type][metric])

#             if len(metric_batch_error) > 0:
#                 log_writer.add_histogram(f"{node_type.name}/{namespace}/{metric}", metric_batch_error, curr_iter)
#                 log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error), curr_iter)
#                 log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error), curr_iter)

#                 if metric in bar_plot:
#                     pd = {'dataset': [namespace] * len(metric_batch_error),
#                                   metric: metric_batch_error}
#                     kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
#                     visualization.visualization_utils.plot_barplots(ax, pd, 'dataset', metric)
#                     log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_bar_plot", kde_barplot_fig, curr_iter)

#                 if metric in box_plot:
#                     mse_fde_pd = {'dataset': [namespace] * len(metric_batch_error),
#                                   metric: metric_batch_error}
#                     fig, ax = plt.subplots(figsize=(5, 5))
#                     visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', metric)
#                     log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_box_plot", fig, curr_iter)


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error))
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error))


def batch_pcmd(prediction_output_dict,
               dt,
               max_hl,
               ph,
               node_type_enum,
               kde=True,
               obs=False,
               map=None,
               prune_ph_to_future=False,
               best_of=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] =  {'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].append(np.array(ade_errors))
            batch_error_dict[node.type]['fde'].append(np.array(fde_errors))

    return batch_error_dict
