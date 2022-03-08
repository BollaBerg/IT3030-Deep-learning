import numpy as np
import torch
from torch import nn

# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class _Reshape(nn.Module):
    """Custom class to reshape a passing through"""
    def __init__(self, shape):
        super(_Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class AutoEncoder(nn.Module):
    def __init__(self, channels: int = 1):
        super(AutoEncoder, self).__init__()
        self.channels = channels
        
        ##### Define model #####
        # Follows, at the time of writing, the structure given in
        # https://www.researchgate.net/publication/320658590_Deep_Clustering_with_Convolutional_Autoencoders
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,          # "Layers" of input data
                out_channels=32,        # "Filters" the convolutional layer applied. What it learns
                kernel_size=(2, 2),     # Size of kernel. Play around with this!
                stride=2,               # "Length" of steps taken. 1=all inputs
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(2, 2),
                stride=2
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(2, 2),
                stride=2
            ),
            nn.Flatten(),
            nn.Linear(1152, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 1152),
            _Reshape((-1, 128, 3, 3)),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(2, 2),
                stride=2,
                output_padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=(2, 2),
                stride=2,
            )
        )
    
    def forward(self, x : torch.Tensor):
        if self.channels == 1:
            z = self.encoder(x)
            x_hat = self.decoder(z)
            output = nn.Sigmoid()(x_hat)
            return output
        
        else:
            output = torch.zeros(x.size(), device=x.device)
            for i in range(self.channels):
                inputs = x[:, i, :, :].view(-1, 1, 28, 28)
                z = self.encoder(inputs)
                x_hat = self.decoder(z)
                outputs = nn.Sigmoid()(x_hat).view(-1, 28, 28)
                output[:, i, :, :] = outputs
            return output

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x_hat = self.decoder(z)
        output = nn.Sigmoid()(x_hat)
        return output


if __name__ == "__main__":
    encoder = AutoEncoder().to(device)
    print(encoder)

    inputs = torch.randint(0, 2, (1, 1, 28, 28), dtype=torch.float)

    output = encoder(inputs)
    print(output.size())

    input_decode = torch.randint(0, 2, (10,), dtype=torch.float)
    output_decode = encoder.decode(input_decode)
    print(output_decode.size())
