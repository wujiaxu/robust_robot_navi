import torch
from torch import nn
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self, input_shape,args) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_shape, 32),
                                   nn.LayerNorm(32))
        self.repr_dim = 32

    def forward(self,x):
        return self.net(x)

class ScanEncoder(nn.Module):
    def __init__(self, input_shape,args,feature_dim:int = 128,input_channel=1) -> None:
        super().__init__()
        
        self.input_shape = input_shape
        self.input_channels = input_channel
        out_channels_1:int =32
        kernel_size_1:int =5
        stride_1:int =2
        out_channels_2:int =32
        kernel_size_2:int =3
        stride_2:int =2
        liner_layer_1:int = 256
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=self.input_channels, 
                               out_channels=out_channels_1, 
                               kernel_size=kernel_size_1, 
                               stride=stride_1)
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=out_channels_1, 
                               out_channels=out_channels_2, 
                               kernel_size=kernel_size_2, 
                               stride=stride_2)
        self.cnn_output_dim = self._get_conv_output_size(self.input_channels)
        # Fully connected layer
        self.fc1 = nn.Linear(in_features=self.cnn_output_dim, 
                             out_features=liner_layer_1)
        # Output layer (if needed, depends on the task)
        self.fc2 = nn.Linear(in_features=liner_layer_1, 
                             out_features=feature_dim)
        
        self.repr_dim = feature_dim

    def _get_conv_output_size(self, input_channels):
        # Function to compute the size of the output from the conv layers
        # Assuming input size is (N, input_channels, L) where L is the length of the sequence
        dummy_input = torch.zeros(1, input_channels, self.input_shape)  # Replace 100 with an appropriate sequence length
        dummy_output = self._forward_conv_layers(dummy_input)
        return int(torch.flatten(dummy_output, 1).size(1))
    
    def _forward_conv_layers(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
    
    def forward(self, x):
        # x = x.unsqueeze(-1).transpose(-2, -1)
        # if len(x.shape)==4:
        #     N,L,C,S = x.shape
        #     print(N,L,C,S)
        #     x = x.view(-1,C,S)
        # else:
        #     L = None
        #     N,C,S = x.shape
        if len(x.shape)==3:
            N,L,S = x.shape
            x  =x.view(N,L,self.input_channels,-1)
            S = x.shape[-1]
            x  = x.view(-1,self.input_channels,S)
        else:
            L = None
            N,S = x.shape
            x  =x.view(N,self.input_channels,-1)
        x = self._forward_conv_layers(x)
        x = torch.flatten(x, 1)  # Flatten the tensor except for the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if L is not None:
            x = x.view(N,L,-1)
        return x