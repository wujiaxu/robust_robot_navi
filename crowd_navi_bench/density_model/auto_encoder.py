from harl.models.policy_models.custom_stochastic_policy import RobotBase
from harl.common.data_recorder import DataRecorder
from harl.utils.envs_tools import check
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
class HybridDecoder(nn.Module):
    def __init__(self, feature_dim: int, cnn_input_dim:int, lidar_output_shape: int, state_output_dim: int):
        super().__init__()
        
        # Parameters to match the encoder
        self.feature_dim = feature_dim
        self.lidar_output_shape = lidar_output_shape
        # print(cnn_input_dim)
        # Fully connected layer for LiDAR reconstruction
        self.fc_lidar = nn.Sequential(nn.Linear(feature_dim, 256),
                                      nn.Linear(256,cnn_input_dim))
        
        # Transposed convolutional layers for LiDAR reconstruction
        self.deconv1 = nn.ConvTranspose1d(in_channels=32, 
                                           out_channels=32, 
                                           kernel_size=5, 
                                           stride=2)
        self.deconv2 = nn.ConvTranspose1d(in_channels=32, 
                                           out_channels=1,  # Output channels = input channels
                                           kernel_size=4, 
                                           stride=2)

        # Fully connected layers for internal state reconstruction
        self.fc_state = nn.Sequential(
            nn.Linear(feature_dim,32),
            nn.ReLU(),
            nn.Linear(32, state_output_dim)
        )
        self.fc_action = nn.Sequential(
            nn.Linear(feature_dim,32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, z):
        N,D = z.shape
        # First fully connected layer for LiDAR reconstruction
        lidar_x = F.relu(self.fc_lidar(z))
        lidar_x = lidar_x.view(N, 32, -1)  # Reshape for deconvolution layers

        # Transposed convolutional layers for LiDAR reconstruction
        lidar_x = F.relu(self.deconv1(lidar_x))
       
        lidar_reconstructed = self.deconv2(lidar_x)  # Final output for LiDAR
        lidar_reconstructed = torch.flatten(lidar_reconstructed, 1)
        # Internal state reconstruction
        state_reconstructed = self.fc_state(z)  # Directly reconstruct internal state

        # Crop to match the output shape if necessary
        lidar_reconstructed = lidar_reconstructed[..., :self.lidar_output_shape]  # Crop if necessary

        action_reconstructed = self.fc_action(z)

        # print(state_reconstructed.shape,lidar_reconstructed.shape,action_reconstructed.shape)

        return torch.cat([state_reconstructed,lidar_reconstructed,action_reconstructed],dim=-1)
    
# Define the VAE model
class VAE(nn.Module):
    def __init__(self, device="cuda:0", latent_dim=10,learning_rate=1e-4):
        super(VAE, self).__init__()
        args = {}
        # Encoder
        self.encoder = RobotBase(args,2)
        
        # Latent space (mean and log variance)
        self.fc_mu = nn.Linear(self.encoder.repr_dim+2, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.repr_dim+2, latent_dim)
        
        # Decoder
        self.decoder = HybridDecoder(feature_dim=latent_dim, 
                                     cnn_input_dim=self.encoder.scan_encoder.cnn_output_dim,
                                     lidar_output_shape=self.encoder.robot_scan_shape, 
                                     state_output_dim=self.encoder.robot_state_shape)
    
        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(self.device)
        self.tpdv = dict(dtype=torch.float32, device=device)
    
    def encode(self, x):
        h = self.encoder(x[...,:-2])
        mu = self.fc_mu(torch.cat([h,x[...,-2:]],dim=-1))
        logvar = self.fc_logvar(torch.cat([h,x[...,-2:]],dim=-1))
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Reparameterization trick
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def train_batch(self,dataloader):
        total_loss = 0
        for batch_data in dataloader:
            batch_data = torch.from_numpy(batch_data) if isinstance(batch_data, np.ndarray) else batch_data
            batch_data = batch_data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch_data)
            loss = vae_loss(recon_batch, batch_data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()

        return total_loss / len(dataloader)

    def load_model(self,path):
        self.load_state_dict(torch.load(path))
        return 
    
    @torch.no_grad()
    def estimate_log_probability(self, input_data, num_samples=100,plot_recon=False):
        """
        Estimate the log probability of input data using the VAE.
        
        Args:
            model: The trained VAE model.
            input_data: The input data for which to estimate the log probability.
            num_samples: Number of Monte Carlo samples for estimating the log likelihood.
        
        Returns:
            log_prob: The estimated log probability of the input.
        """
        input_data = check(input_data).to(**self.tpdv)
        # Encode the input to get the mean and log variance of q(z|x)
        mu, logvar = self.encode(input_data)
        
        # Initialize the log likelihood estimate
        log_prob_sum = 0.0
        
        # Perform Monte Carlo sampling
        for _ in range(num_samples):
            # Sample z from the Gaussian posterior q(z|x)
            z = self.reparameterize(mu, logvar)
            
            # Reconstruct the input from z
            recon_x = self.decoder(z)
            
            if plot_recon:
                plt.plot(np.linspace(0,728,728),input_data.cpu().numpy()[0],color="b")
                plt.plot(np.linspace(0,728,728),recon_x.cpu().numpy()[0],color="r")
                plt.savefig("density_model/check.png")
                plt.close()
                plt.clf()
                input("check")
            # Compute log p(x|z) (reconstruction log likelihood)
            # Assuming a Gaussian distribution for p(x|z) with mean recon_x and variance 1
            recon_log_prob = -F.mse_loss(recon_x, input_data, reduction='none').sum(dim=-1)
            
            # Add the reconstruction log probability to the sum
            log_prob_sum += recon_log_prob
        
        # Compute the average over all samples
        log_prob_estimate = log_prob_sum / num_samples
        # Compute the KL divergence D_KL(q(z|x) || p(z))
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        # Estimate the log probability: log p(x) ≈ log p(x|z) - KL divergence
        log_prob = log_prob_estimate - kl_divergence
        return log_prob.mean()


# Loss function for the VAE
def vae_loss(recon_x, x, mu, logvar):
    weight = torch.ones(728,dtype=torch.float32,device="cuda:0")
    weight[0] = 10.
    weight[1] = 10.
    weight[2] = 10.
    weight[3] = 10.
    weight[4] = 10.
    weight[5] = 10.
    weight[-2] = 10.
    weight[-1] = 10.
    # weight = weight/torch.sum(weight)
    # Reconstruction loss (MSE or BCE for normalized data)
    # recon_loss = nn.functional.mse_loss(recon_x, x,reduction="sum")
    recon_loss = torch.sum(weight.unsqueeze(0)*(recon_x - x)**2)
    # KL Divergence loss
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_div

def train_vae(vae:VAE, dataset:DataRecorder, epochs=500, batch_size=256):
    
    dataloader = dataset.get_data_generator(batch_size)
    for epoch in range(epochs): 
        avg_loss = vae.train_batch(dataloader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
        if (epoch+1) % 100 == 0:
            print("eval")
            vae.eval()
            with torch.no_grad():
                for batch_data in dataloader:
                    reconstruction,_,_ = vae.forward(batch_data.to(vae.device))
                    reconstruction = reconstruction.cpu().numpy()
                    for i in range(5):
                        plt.plot(np.linspace(0,728,728),batch_data[i],color="b")
                        plt.plot(np.linspace(0,728,728),reconstruction[i],color="r")
                        plt.savefig("density_model/{}.png".format(i))
                        plt.close()
                        plt.clf()
                    break
            torch.save(vae.state_dict(), 'density_model/090_4p_6c_rvs_circlecross_vae_model_{}.pth'.format(epoch))
            vae.train()


if __name__ == "__main__":
    print("train vae")
    dataset = DataRecorder(save_dir="/home/dl/wu_ws/HARL/crowd_navi_bench/data_generation/crowd_env/crowd_navi/robot_crowd_happo/train_on_ai_090_4p_6c_rvs_circlecross_vs_c090_happo_5p_6c_rvs_circlecross_data/seed-00001-2024-10-09-12-27-03/logs")
    dataset.load_data()
    vae = VAE(device="cuda:0", learning_rate=1e-3,latent_dim=32)
    train_vae(vae,dataset,epochs=10000)