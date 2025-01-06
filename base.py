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
from diffusion.data import Dataset_3d_single
device = 'cuda'

class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)

class VAE3D128(nn.Module):
    def __init__(self):
        super(VAE3D128, self).__init__()

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
            
            Lambda(lambda x: x.reshape(x.shape[0], -1)),

            nn.Linear(in_features = 128, out_features=64)
        )

        # self.bottleneck=nn.Sequential(            
        #     nn.Linear(in_features = 64, out_features=128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # )

        self.decoder = nn.Sequential(       
            nn.Linear(in_features = 64, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            Lambda(lambda x: x.reshape(-1, 128, 1, 1, 1)),

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
        x = x.reshape((-1, 1, 64, 64, 64))
        z = self.encoder(x)
        # print(z.shape)
        # z=self.bottleneck(z)
        # print(z.shape)
        # z=z.reshape(-1, 128, 1, 1, 1)

        out = self.decoder(z)
        return out.squeeze()