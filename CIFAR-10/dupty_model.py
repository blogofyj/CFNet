import torch
import torch.nn as nn
import torch.nn.functional as F
from denoisingBlock import denoising_block
from torch.distributions.normal import Normal


# Class for convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=0, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


# Class for de-convolution block
class DeConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=0, stride=1, padding=0, out_padding=0):
        super(DeConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, out_padding, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class VAE(nn.Module):
    def __init__(self, image_size=28, image_channels=1, h_dim=4096, z_dim=128):
        super(VAE, self).__init__()
        self.model_name = 'VAE'
        self.image_size = image_size
        self.image_channels = image_channels
        self.h_dim = h_dim
        self.z_dim = z_dim
        # module for encoder
        # d = (d - kernel_size + 2 * padding) / stride + 1
        self.c1 = ConvBlock(self.image_channels, 64, 5, 1, 2)   # Bx64x28x28
        self.c2 = ConvBlock(64, 64, 4, 2, 3)                    # Bx64x16x16
        self.c3 = ConvBlock(64, 128, 4, 2, 1)                   # Bx128x8x8
        self.c4 = ConvBlock(128, 256, 4, 2, 1)                  # Bx256x4x4
        self.denoising_block1 = denoising_block(in_planes=64, ksize=3, filter_type="Mean_Filter")
        # self.denoising_block2 = denoising_block(in_planes=64, ksize=3, filter_type="Gaussian_Filter")
        # self.denoising_block3 = denoising_block(in_planes=128, ksize=3, filter_type="Gaussian_Filter")
        # self.updemo = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.downdemo = nn.AvgPool2d(2)
        self.e_module = nn.Sequential(self.c1, self.denoising_block1, self.c2, self.c3, self.c4)

        # self.e_module = nn.Sequential(self.c1, self.c2, self.c3, self.c4)
        self.mu = nn.Linear(self.h_dim, self.z_dim)
        self.sigma = nn.Linear(self.h_dim, self.z_dim)
        # module for image decoder
        self.linear = nn.Linear(self.z_dim, self.h_dim)
        self.d1 = DeConvBlock(256, 128, 4, 2, 1)
        self.d2 = DeConvBlock(128, 64, 4, 2, 1)
        self.d3 = DeConvBlock(64, 64, 4, 2, 3)
        self.d4 = nn.ConvTranspose2d(64, self.image_channels, 5, 1, 2, bias=False)
        self.img_module = nn.Sequential(self.d1, self.d2, self.d3, self.d4)
        self.fc = nn.Linear(784, 10)


    # Encoder
    def encode(self, x):
        self.batch_size = x.size(0)
        x = self.e_module(x)
        x = x.view(self.batch_size, -1)
        mean = self.mu(x)
        var = self.sigma(x)

        # distribution = Normal(mean.softmax(dim=-1), var.softmax(dim=-1))
        distribution = Normal(mean, abs(var))
        # distribution = Normal(mean, var)

        return mean, var, distribution

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # Decoder for image denoising
    def img_decode(self, z):
        self.batch_size = z.size(0)
        x = F.relu(self.linear(z))
        x = x.view(self.batch_size, 256, 4, 4)

        return torch.sigmoid(self.img_module(x))
        # return torch.sigmoid(self.d4(self.d3(self.d2(self.d1(x)))))

    # Forward function
    def forward(self, x):
        mean, var, dist = self.encode(x)
        z = self.reparameterize(mean, var)
        output = self.img_decode(z)
        out = output.view(output.size(0), -1)
        out = self.fc(out)
        return output, mean, var, z, out, dist.mean, dist.stddev

    def getFeatures(self, x):
        features = []
        features.append(x)
        x = self.c1(x)
        features.append(x)
        x = self.c2(x)
        features.append(x)
        x = self.c3(x)
        features.append(x)
        x = self.c4(x)
        features.append(x)
        x = x.view(self.batch_size, -1)
        mean = self.mu(x)
        var = self.sigma(x)
        z = self.reparameterize(mean, var)
        x = F.relu(self.linear(z))
        x = x.view(self.batch_size, 256, 4, 4)
        features.append(x)
        x = self.d1(x)
        features.append(x)
        x = self.d2(x)
        features.append(x)

        x = self.d3(x)
        features.append(x)
        x = self.d4(x)
        features.append(x)
        x = torch.sigmoid(x)
        features.append(x)

        return features