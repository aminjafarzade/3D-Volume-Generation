import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('/root/Diffusion-Project-3DVolume/dataset')
from voxel_data import VoxelDataset
import numpy as np
import os
from tqdm import tqdm
# sys.path is a list of absolute path strings
def main():
    from ae import VAE3D128

    device="cuda"
    table_data = np.load('/data/hdf5_data/airplane_voxels_train.npy')
    print(table_data.shape)
    data=torch.tensor(table_data[1]).unsqueeze(0).unsqueeze(0)

    model=VAE3D128()
    model.load_state_dict(torch.load("/root/Diffusion-Project-3DVolume/models/ae_ckpts_updated/model_30.pt", map_location="cpu"))
    model.eval()
    output=model(data)
    # output = torch.sigmoid(output)
    print(output.shape)
    print("loss")
    loss=nn.BCELoss()(output, data)
    print(loss)
    output=output.squeeze(0).detach().numpy()
    np.save("/root/Diffusion-Project-3DVolume/models/reconstruct/data_new.npy", output)

if __name__=="__main__":
    main()


