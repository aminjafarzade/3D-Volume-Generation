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
device = 'cuda'

class VAE3D128(nn.Module):
    def __init__(self):
        super(VAE3D128, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, stride=1, padding=0),   # 124
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0), #62
            nn.Conv3d(32, 32, kernel_size=5, stride=1, padding=0),  # 58
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0), #29
            # nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=5, stride=1, padding=0),  # 24
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0), #12
            nn.Conv3d(64, 64, kernel_size=5, stride=1, padding=0),  # 8
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0) #4
            # nn.BatchNorm3d(64),
            # nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0),  # 3
            # nn.ReLU(),
            # nn.Conv3d(64, 128, kernel_size=2, stride=2, padding=0), # 1
            # nn.Flatten() # 64, 128
            # nn.Conv3d(128, 128, kernel_size=2, stride=2, padding=0),  # 128 2 2 2 
        )

        
        # self.mu = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.BatchNorm1d(128)
        # )

        # self.log_var = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.BatchNorm1d(128)
        # )


        # Decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0), # 14
            # nn.ReLU(),
            # nn.ConvTranspose3d(64, 64, kernel_size=1, stride=1, padding=0), # 4
            # nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True), #8
            nn.ConvTranspose3d(64, 64, kernel_size=5, stride=1, padding=0), # 12
            nn.ReLU(),
            # nn.BatchNorm3d(64),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True), #24
            nn.ConvTranspose3d(64, 32, kernel_size=5, stride=1, padding=0),  # 28
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True), #56
            nn.ConvTranspose3d(32, 8, kernel_size=5, stride=1, padding=0), #60
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True), #120
            # nn.BatchNorm3d(32),  
            nn.ConvTranspose3d(8, 1, kernel_size=9, stride=1, padding=0),
            nn.ReLU(),  
            # nn.ConvTranspose3d(8, 1, kernel_size=6, stride=2, padding=0), 
            nn.Sigmoid()
        )

    def forward(self, x):
        # print('here')
        # print(x.shape)
        z = self.encoder(x)
        # print(z.shape)
        # mu = self.mu(z)
        # log_variance = self.log_var(z)

        # var = log_variance.exp()
        # dev = var.sqrt()

        # eps = torch.randn(mu.shape).to(device)

        # res = mu + dev * eps
        # # print(res.shape)
        # res = res.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print(z.shape)
        out = self.decoder(z)
        # print(out.shape)
        return out


# class VAE3D128_Updated(nn.Module):
#     def __init__(self):
#         self.encoder = nn.Sequential(
#             nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=0),
#             nn.ELU(),
#             nn.BatchNorm3d(8),
#             nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
#             nn.ELU(),
#             nn.BatchNorm3d(16),
#             nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0),
#             nn.ELU(),
#             nn.BatchNorm3d(32),
#             nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ELU(),
#             nn.BatchNorm3d(64),
#             nn.Flatten(),
#             nn.Linear(in_features=64, out_features=343),  # Replace `enc_conv4_output_shape` with the flattened input size
#             nn.BatchNorm1d(num_features=343),
#             nn.Linear(in_features=343, out_features=128),
#             nn.BatchNorm1d(num_features=128)
#         )

criterion = nn.BCELoss()
table_data_size = 3834
chair_data_size = 2654
airplane_data_size = 1957
dataloader_sz = 132


# table_data = np.load('/data/hdf5_data/airplane_voxels_train.npy')
airplane_data = np.load('/data/hdf5_data/airplane_voxels_train.npy', mmap_mode='r')
chair_data = np.load('/data/hdf5_data/chair_voxels_train.npy', mmap_mode='r')
table_data = np.load('/data/hdf5_data/table_voxels_train.npy', mmap_mode='r')

# example_table = table_data[0, :, :, :]
# example_airplane = airplane_data[0, :, :, :]
# example_chair = chair_data[0, :, :, :]
# print(torch.tensor(table_data[0]).shape)

example_table = torch.tensor(table_data[0]).unsqueeze(0).unsqueeze(0).to(device)
example_airplane = torch.tensor(airplane_data[0]).unsqueeze(0).unsqueeze(0).to(device)
example_chair = torch.tensor(chair_data[0]).unsqueeze(0).unsqueeze(0).to(device)

def eval(model, epoch):
    path = f'/root/Diffusion-Project-3DVolume/models/reconstruct2/{epoch}'
    os.makedirs(path, exist_ok=True)
    with torch.no_grad():
        print(f"Getting results for epoch {epoch + 1}")

        # model.eval()
        eval = torch.cat([example_chair, example_airplane, example_table], dim=0)
        res = model(eval)

        res_final = (res >= 0.5).int()

        output_table = res_final[2]
        output_chair = res_final[0]
        output_airplane = res_final[1]
        
        output_table = output_table.squeeze(0).cpu().numpy()
        output_airplane = output_airplane.squeeze(0).cpu().numpy()
        output_chair = output_chair.squeeze(0).cpu().numpy()

        np.save(f"{path}/table.npy", output_table)
        np.save(f"{path}/airplane.npy", output_airplane)
        np.save(f"{path}/chair.npy", output_chair)

        print(f"Results saved in {path} for epoch {epoch + 1}")





def get_batch():
    idxs_table = np.random.randint(low=0, high=table_data_size, size=(1, 6))
    idxs_chair = np.random.randint(low=0, high=chair_data_size, size=(1, 5))
    idxs_airplane = np.random.randint(low=0, high=airplane_data_size, size=(1, 5))
    # print(f"table data : {table_data.shape}")

    batch_table = table_data[idxs_table, :, :, :]
    batch_airplane = airplane_data[idxs_airplane, :, :, :]
    batch_chair = chair_data[idxs_chair, :, :, :]

    # print(f"batch_tabel : {batch_table.shape}")

    # print(batch_chair.shape)

    batch = np.concatenate([batch_table, batch_airplane, batch_chair], axis=1)

    
    batch = torch.from_numpy(batch)
    batch = batch.squeeze(axis=0).unsqueeze(axis=1)
    # print(batch.shape)
    permutation = torch.randperm(batch.size(0))  # Size of the 0th axis

    # Shuffle the tensor along axis=0
    shuffled_batch = batch[permutation]

    return shuffled_batch


print(table_data.shape)

# os.makedirs("ae_ckpts", exist_ok=True)
path="/root/Diffusion-Project-3DVolume/models/ae_ckpts2"
os.makedirs(path, exist_ok=True)

# Training function
def train_model(model, optimizer, criterion, num_epochs, device):
    model.to(device)
    
    
    for epoch in range(num_epochs):

        model.train()
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), os.path.join(path, f"model_{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(path, f"optim_{epoch}.pt"))
            eval(model, epoch)

        
        # running_loss = 0.0
        for i in tqdm(range(dataloader_sz)):
            
            # print(batch_table.shape)

            batch = get_batch()
            
            # Move data to device
            inputs = batch.to(device)
            # print(inputs.shape)
            # print(inputs.shape)
            # print(inputs.shape)
            
            # Forward pass
            outputs = model(inputs)
            # print(outputs.shape)
            # print(inputs.shape)
            # # # Compute loss
            # print("Outputs min:", outputs.min().item(), "max:", outputs.max().item())
            # print("Inputs min:", inputs.min().item(), "max:", inputs.max().item())
            # outputs = torch.sigmoid(outputs)  # Apply sigmoid to convert logits to probabilities

            optimizer.zero_grad()
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
batch_size = 16
learning_rate = 0.001
beta1 = 0.9
num_epochs = 100

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, 0.999))
# optimizer = optim.Adadelta(model.parameters())

# Train the model
train_model(model, optimizer, criterion, num_epochs, device)



