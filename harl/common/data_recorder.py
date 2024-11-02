import pickle
from pathlib import Path
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DataRecorder:
    def __init__(self, save_interval=100, save_dir="rollout_data"):
        
        self.save_interval = save_interval  # Save every `save_interval` episodes
        self.save_dir = Path(save_dir) / 'episode_data' # Path to save the data file
        self.save_dir.mkdir(exist_ok=True)
        self.episode_count = 0  # Keep track of the episode number
        self.episode_buffer = {self.episode_count:[]}  # List to store data per episode

    def record_step(self, observation, action, reward):
        """Record a single step's data in the current episode."""
        # if len(self.episode_buffer) == self.episode_count:

        self.episode_buffer[self.episode_count].append({
            'observation': observation,
            'action': action,
            'reward': reward
        })

    def end_episode(self):
        """Mark the end of the current episode and start a new one."""
        self.episode_count += 1

        # Save and clear buffer if the save interval is reached
        if self.episode_count % self.save_interval == 0:
            self.save_data()
            self.clear_buffer()
        self.episode_buffer[self.episode_count]=[]  # Add a new episode

    def save_data(self):
        """Save the episode data to a file."""
        save_path = self.save_dir/"data_{}.pkl".format(self.episode_count)
        with open(save_path, 'ab') as f:
            pickle.dump(self.episode_buffer, f)
        print(f'Saved {len(self.episode_buffer)} episodes to {save_path}.')

    def clear_buffer(self):
        """Clear the episode buffer to save memory."""
        self.episode_buffer = {}
        print(f'Buffer cleared after {self.episode_count} episodes.')

    def load_data(self): #TODO read all file
        """Load previously saved data from the file."""
        pkl_files = glob.glob(os.path.join(self.save_dir, '*.pkl'))
        print(pkl_files)
        # Print the list of .pkl files
        for file in pkl_files:
            
            with open(file, 'rb') as f:
                while True:
                    try:
                        self.episode_buffer.update(pickle.load(f))
                    except EOFError:
                        break
        return 


    def get_data_generator(self, batch_size):
        """Sample observation-action pairs from the episode buffer."""
        # Flatten the episode buffer to extract all observation-action pairs
        all_data = []
        for episode in self.episode_buffer:
            final_step = self.episode_buffer[episode][-1]
            robot_reward = final_step["reward"][0][0]
            if robot_reward <=0:
                print("skip")
                continue
            for step in self.episode_buffer[episode]:
                obs_action_pair = np.concatenate([step['observation'][0][:726], np.clip(step['action'][0],np.array([-1,-1]),np.array([1,1]))],dtype=np.float32)
                all_data.append(obs_action_pair)

        class EpisodeDataset(Dataset):
            def __init__(self, all_data):
                """Initialize with the episode buffer and sequence length."""
                self.data_buffer = all_data

            def __len__(self):
                """Return the number of valid sequences."""
                return len(self.data_buffer)

            def __getitem__(self, idx):
                """Get a sequence of observation-action pairs."""
                return self.data_buffer[idx]
        

        return DataLoader(EpisodeDataset(all_data), batch_size=batch_size, shuffle=True)

# Example usage:
# recorder = DataRecorder(save_interval=100, save_path='rollout_data.pkl')
# for episode in range(1000):
#     for step in range(episode_length):
#         recorder.record_step(observation, action, reward)
#     recorder.end_episode()