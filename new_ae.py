from diffusion.data import Dataset_3d_single
from tqdm import tqdm
import os
import numpy as np
from voxel_data import VoxelDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
# sys.path is a list of absolute path strings
sys.path.append('/root/Diffusion-Project-3DVolume/dataset')
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

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=24,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=24, out_channels=48,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=48, out_channels=96,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=96, out_channels=96,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=96, out_channels=128,
                      kernel_size=4, stride=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            Lambda(lambda x: x.reshape(x.shape[0], -1)),

            nn.Linear(in_features=128, out_features=64)
        )

        # self.bottleneck=nn.Sequential(
        #     nn.Linear(in_features = 64, out_features=128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            Lambda(lambda x: x.reshape(-1, 128, 1, 1, 1)),

            nn.ConvTranspose3d(
                in_channels=128, out_channels=96, kernel_size=4, stride=1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=96, out_channels=96,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=96, out_channels=48,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=48, out_channels=24,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=24, out_channels=1,
                               kernel_size=4, stride=2, padding=1),
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


class VAE3D128_diff(nn.Module):
    def __init__(self):
        super(VAE3D128_diff, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=12,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(12),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=12, out_channels=24,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=24, out_channels=24,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=24, out_channels=48,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=48, out_channels=16,
                      kernel_size=3, stride=1, padding=1),


            # nn.Conv3d(in_channels = 96, out_channels = 128, kernel_size = 4, stride = 1),
            # nn.BatchNorm3d(128),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Lambda(lambda x: x.reshape(x.shape[0], -1)),

            # nn.Linear(in_features = 24, out_features=16)
        )

        # self.bottleneck=nn.Sequential(
        #     nn.Linear(in_features = 64, out_features=128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # )

        self.decoder = nn.Sequential(
            # nn.Linear(in_features = 16, out_features=48),
            # nn.BatchNorm1d(48),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Lambda(lambda x: x.reshape(-1, 48, 1, 1, 1)),
            nn.ConvTranspose3d(in_channels=16, out_channels=48,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=48, out_channels=24,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=24, out_channels=24,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=24, out_channels=12,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(12),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=12, out_channels=1,
                               kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm3d(24),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.ConvTranspose3d(in_channels = 24, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.reshape((-1, 1, 64, 64, 64))
        # print(x.shape)
        z = self.encoder(x)
        # print(z.shape)
        # print(z.shape)
        # z=self.bottleneck(z)
        # print(z.shape)
        # z=z.reshape(-1, 128, 1, 1, 1)
        # print(z.shape)

        out = self.decoder(z)
        return out


criterion = nn.BCELoss()
dataloader_sz = 132


# table_data = np.load('/data/hdf5_data/airplane_voxels_train.npy')
# airplane_data = np.load('/data/hdf5_data/airplane_voxels_train.npy')
# chair_data = np.load('/data/hdf5_data/chair_voxels_train.npy')
# table_data = np.load('/data/hdf5_data/table_voxels_train.npy')

plane_path = '/data/hdf5_data/airplane_voxels_train.npy'
chair_path = '/data/hdf5_data/chair_voxels_train.npy'
table_path = '/data/hdf5_data/table_voxels_train.npy'

batch_size = 64
table_data_size = 3834
table_data = np.load('/data/hdf5_data/table_voxels_train.npy')


def get_batch():
    idxs_table = np.random.randint(low=0, high=table_data_size, size=(64,))
    # idxs_chair = np.random.randint(low=0, high=chair_data_size, size=(5,))
    # idxs_airplane = np.random.randint(low=0, high=airplane_data_size, size=(5,))
    # print(f"table data : {table_data.shape}")
    # print(idxs_table)
    # print("table")
    # print(table_data.shape)
    # batch_table = table_data[idxs_table, :, :, :]
    batch_table = []
    for i in idxs_table:
        # print(i)
        batch_table.append(torch.tensor(table_data[i]).unsqueeze(0).to("cuda"))
    batch_table = torch.cat(batch_table, dim=0)

    return batch_table


data = Dataset_3d_single(chair_path, plane_path, table_path)
train_loader = DataLoader(data, shuffle=True, batch_size=batch_size)
# example_table = table_data[0, :, :, :]
# example_airplane = airplane_data[0, :, :, :]
# example_chair = chair_data[0, :, :, :]
# print(torch.tensor(table_data[0]).shape)

# example_table = torch.tensor(table_data[0]).unsqueeze(0).unsqueeze(0).to(device)
# example_airplane = torch.tensor(airplane_data[0]).unsqueeze(0).unsqueeze(0).to(device)
# example_chair = torch.tensor(chair_data[0]).unsqueeze(0).unsqueeze(0).to(device)


# def eval(model, epoch):
#     path = f'/root/Diffusion-Project-3DVolume/models/reconstruct/{epoch}'
#     os.makedirs(path, exist_ok=True)
#     with torch.no_grad():
#         print(f"Getting results for epoch {epoch + 1}")

#         # model.eval()
#         eval = torch.cat([example_chair, example_airplane, example_table], dim=0)
#         res = model(eval)

#         res_final = (res >= 0.5).int()

#         output_table = res_final[2]
#         output_chair = res_final[0]
#         output_airplane = res_final[1]

#         output_table = output_table.squeeze(0).cpu().numpy()
#         output_airplane = output_airplane.squeeze(0).cpu().numpy()
#         output_chair = output_chair.squeeze(0).cpu().numpy()

#         np.save(f"{path}/table.npy", output_table)
#         np.save(f"{path}/airplane.npy", output_airplane)
#         np.save(f"{path}/chair.npy", output_chair)

#         print(f"Results saved in {path} for epoch {epoch + 1}")

path = "/data/ae_ckpts_new_single_getbatch_basemodel"
os.makedirs(path, exist_ok=True)

# Training function


def train_model(model, optimizer, criterion, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):

        model.train()
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(
                path, f"model_{epoch}.pt"))
            torch.save(optimizer.state_dict(),
                       os.path.join(path, f"optim_{epoch}.pt"))
            # eval(model, epoch)

        # running_loss = 0.0
        for i, batch_ in enumerate(tqdm(train_loader)):

            # print(batch_table.shape)
            # print(batch.shape)
            batch = get_batch()
            # batch=batch.reshape((-1, 1, 64, 64, 64))
            optimizer.zero_grad()
            # batch = get_batch()

            # Move data to device
            inputs = batch.to(device)
            # print(inputs.shape)
            # print(inputs.shape)
            # print(inputs.shape)
            # print(inputs.shape)

            # Forward pass
            outputs = model(inputs)
            # print(outputs.squeeze().shape)
            # print(outputs.shape)
            # print(inputs.shape)
            # # # Compute loss
            # print("Outputs min:", outputs.min().item(), "max:", outputs.max().item())
            # print("Inputs min:", inputs.min().item(), "max:", inputs.max().item())
            # outputs = torch.sigmoid(outputs)  # Apply sigmoid to convert logits to probabilities

            loss = criterion(outputs, inputs)

            # # # Backward pass and optimization
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)

            # # Accumulate loss
            # running_loss += loss.item()

        # Print epoch loss
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / dataloader_sz:.4f}")
    print("Training complete!")


# Parameters
batch_size = 64
learning_rate = 0.001
# beta1 = 0.9
num_epochs = 5000

# Instantiate the model, optimizer, and dataloader
model = VAE3D128()  # Use VAE3D32 or VAE3D64 depending on the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example random dataset (replace with real data)
# random_data = np.random.rand(1000, 1, 32, 32, 32)  # 1000 samples of 32³ voxel grids

# ls = ['table_voxels_train.npy', "chair_voxels_train.npy", 'airplane_voxels_train.npy']

# dataset = VoxelDataset(ls)

# print(f"length of dataset {len(dataset)}")
# print(f"4 th element is {dataset[4]}")


# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, optimizer, criterion, num_epochs, device)
