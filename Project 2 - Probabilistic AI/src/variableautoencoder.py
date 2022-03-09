from src.autoencoder import _Reshape, device

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy

def elbo_loss_function(recon_x, x, mu, logvar):
    # https://github.com/pytorch/examples/blob/a74badde33f924c2ce5391141b86c40483150d5a/vae/main.py#L73
    BCE = binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class VariableAutoEncoder(nn.Module):
    def __init__(self, channels: int = 1):
        super(VariableAutoEncoder, self).__init__()
        self.channels = channels
        
        ##### Define model #####
        # Loosely follows example found here
        # https://github.com/pytorch/examples/blob/a74badde33f924c2ce5391141b86c40483150d5a/vae/main.py
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
            nn.Flatten()
        )
        self.mu = nn.Linear(1152, 28)
        self.logvar = nn.Linear(1152, 28)

        self.decoder = nn.Sequential(
            nn.Linear(28, 1152),
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
    
    def encode(self, x):
        out = self.encoder(x)
        return self.mu(out), self.logvar(out)
    
    def reparameterize(self, mu, logvar):
        """Convert a N(0,1) sample to a N(mu, logvar) sample"""
        standard_deviation = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(standard_deviation)
        return epsilon * standard_deviation + mu
    
    def decode(self, z):
        x_hat = self.decoder(z)
        output = nn.Sigmoid()(x_hat)
        return output
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == "__main__":
    encoder = VariableAutoEncoder()
    print(encoder)

    inputs = torch.randint(0, 2, (1, 1, 28, 28), dtype=torch.float)

    output = encoder(inputs)
    print(output[0].size())

    input_decode = torch.randint(0, 2, (28,), dtype=torch.float)
    output_decode = encoder.decode(input_decode)
    print(output_decode.size())
