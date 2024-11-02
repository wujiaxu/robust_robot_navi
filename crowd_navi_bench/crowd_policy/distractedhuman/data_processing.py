#TODO
from harl.envs.robot_crowd_sim.crowd_env import RobotCrowdSim
import numpy as np
import os 
def set_scene(current_frame:np.ndarray,env:RobotCrowdSim):

    return 

def process_traj(txy:np.ndarray):

    return 

# want maximum num ped in one frame

def preprocess(file):

    # read file 
    data_frames = np.genfromtxt(file,delimiter=" ")
    human_ids = np.unique(data_frames[:,0]).tolist()

    # get each person traj
    for id in human_ids:
        human_frames = data_frames[data_frames[:,0]==id,:]
        human_type = human_frames[0,-1]
        human_goal = human_frames[-1,2:4]*0.01 #cm to m
        human_times = human_frames[:,1]
        human_xs = human_frames[:,2]
        human_ys = human_frames[:,3]

    # add velocity, size, goal, theta, maximum speed

    # 

    return 

if __name__ == "__main__":
    root_dir = "raw_data"
    files = os.listdir(root_dir)
    preprocess(os.path.join(root_dir,files[0]))