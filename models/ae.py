import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
# sys.path is a list of absolute path strings
sys.path.append('/root/Diffusion-Project-3DVolume/dataset')
from voxel_data import VoxelDataset
import numpy as np
import os
from tqdm import tqdm
device = 'cpu'

class VAE3D128(nn.Module):
    def __init__(self):
        super(VAE3D128, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=6, stride=2, padding=0),   # 62
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=6, stride=2, padding=0),  # 29
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=0),  # 13
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=4, stride=2, padding=0),  # 5
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0),  # 3
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=2, stride=2, padding=0), # 1
            nn.Flatten() # 64, 128
            # nn.Conv3d(128, 128, kernel_size=2, stride=2, padding=0),  # 128 2 2 2 
        )

        
        self.mu = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )

        self.log_var = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )


        # Decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0), # 14
            # nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=0), # 32
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.ConvTranspose3d(64, 64, kernel_size=6, stride=2, padding=0),  # 68
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm3d(32),  
            nn.ConvTranspose3d(32, 8, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),  
            nn.ConvTranspose3d(8, 1, kernel_size=6, stride=2, padding=0), 
            nn.Sigmoid()
        )

    def forward(self, x):
        # print('here')
        # print(x.shape)
        z = self.encoder(x)
        # print(z.shape)
        print("z shape")
        print(z.shape)
        mu = self.mu(z)
        log_variance = self.log_var(z)

        var = log_variance.exp()
        dev = var.sqrt()

        eps = torch.randn(mu.shape).to(device)

        res = mu + dev * eps
        # print(res.shape)
        res = res.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        out = self.decoder(res)
        return out


class VAE3D1282(nn.Module):
    def __init__(self):
        super(VAE3D1282, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=6, stride=2, padding=0),   # 62
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=6, stride=2, padding=0),  # 29
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=0),  # 13
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=4, stride=2, padding=0),  # 5
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0),  # 3
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=2, stride=2, padding=0), # 1
            # nn.Conv3d(128, 128, kernel_size=2, stride=2, padding=0),  # 128 2 2 2 
        )

        # Decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0), # 14
            # nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=0), # 32
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.ConvTranspose3d(64, 64, kernel_size=6, stride=2, padding=0),  # 68
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm3d(32),  
            nn.ConvTranspose3d(32, 8, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),  
            nn.ConvTranspose3d(8, 1, kernel_size=6, stride=2, padding=0),    
        )

    def forward(self, x):
        print('here')
        print(x.shape)
        z = self.encoder(x)
        # print(z.shape)
        out = self.decoder(z)
        return out
