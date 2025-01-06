import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE3D32(nn.Module):
    def __init__(self):
        super(VAE3D32, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=6, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(64),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 32, kernel_size=6, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=8, stride=4, padding=0)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class VAE3D64(nn.Module):
    def __init__(self):
        super(VAE3D64, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=6, stride=2, padding=0),   # 30
            nn.Conv3d(32, 32, kernel_size=6, stride=2, padding=0),  # 13
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=0),  # 5
            nn.Conv3d(64, 64, kernel_size=4, stride=2, padding=0),  # 1
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0),  # 6
            nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0),  # 6

        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, padding=0), # 14
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=0), # 32
            nn.BatchNorm3d(32),
            nn.ConvTranspose3d(32, 32, kernel_size=6, stride=2, padding=0),  # 68
            nn.ConvTranspose3d(32, 32, kernel_size=6, stride=2, padding=0),  # 68
            nn.BatchNorm3d(32),
            nn.ConvTranspose3d(32, 32, kernel_size=8, stride=2, padding=0),  # 68
            nn.ConvTranspose3d(32, 1, kernel_size=8, stride=2, padding=0)    
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class VAE3D128(nn.Module):
    def __init__(self):
        super(VAE3D128, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=6, stride=2, padding=0),   # 31
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=6, stride=2, padding=0),  # 13
            nn.ReLU(),
            nn.BatchNorm3d(64),
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
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=0), # 4
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.ConvTranspose3d(64, 64, kernel_size=6, stride=2, padding=0),  # 12
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=8, stride=2, padding=0),  # 30
            nn.ReLU(),
            nn.BatchNorm3d(32),  
            nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=0),  # 62
            nn.ReLU(),  
            nn.ConvTranspose3d(32, 1, kernel_size=6, stride=2, padding=0),    
        )

    def forward(self, x):
        z = self.encoder(x)
        print(z.shape)
        out = self.decoder(z)
        return out



# Example usage
if __name__ == "__main__":
    # Instantiate the models
    model_128 = VAE3D128().to("cuda")
    # model_64 = VAE3D64()

    # Create dummy inputs
    # input_32 = torch.randn(1, 1, 32, 32, 32)  # Batch size of 1, single channel, 32^3 volume
    # input_64 = torch.randn(1, 1, 64, 64, 64)  # Batch size of 1, single channel, 64^3 volume

    input_128 = torch.randn(32, 1, 128, 128, 128).to("cuda")

    # Forward pass
    output_128 = model_128(input_128)
    # output_64 = model_64(input_64)

    # print("Output shape for 32^3 model:", output_32.shape)
    print("Output shape for 128^3 model:", output_128.shape)


