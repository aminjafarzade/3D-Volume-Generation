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
from data import Dataset_3d
device = 'cuda'

class VAE3D128_copy(nn.Module):
    def __init__(self):
        super(VAE3D128_copy, self).__init__()

        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 24, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv3d(in_channels = 24, out_channels = 48, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv3d(in_channels = 48, out_channels = 96, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 96, out_channels = 96, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv3d(in_channels = 96, out_channels = 128, kernel_size = 4, stride = 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Flatten(),

            nn.Linear(in_features = 128, out_features=64)
        )

        self.bottleneck=nn.Sequential(            
            nn.Linear(in_features = 64, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.decoder = nn.Sequential(            
            nn.ConvTranspose3d(in_channels = 128, out_channels = 96, kernel_size = 4, stride = 1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 96, out_channels = 96, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 96, out_channels = 48, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 48, out_channels = 24, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 24, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z=self.bottleneck(z)
        z=z.reshape(-1, 128, 1, 1, 1)

        out = self.decoder(z)
        return out