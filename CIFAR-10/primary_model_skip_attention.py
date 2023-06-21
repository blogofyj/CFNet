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


class Mix(nn.Module):
    def __init__(self, m=-0.80, size=1):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()
        self.denoising_block = denoising_block(in_planes=size, ksize=3, filter_type="Mean_Filter")

    def forward(self, fea1, fea2, denoising):
        mix_factor = self.mix_block(self.w)
        if denoising:
            fea1 = self.denoising_block(fea1)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


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
        # self.denoising_block1 = denoising_block(in_planes=64, ksize=3, filter_type="Mean_Filter")

        # self.e_module = nn.Sequential(self.c1, self.denoising_block1, self.c2, self.c3, self.c4)
        self.e_module = nn.Sequential(self.c1, self.c2, self.c3, self.c4)

        self.mu = nn.Linear(self.h_dim, self.z_dim)
        self.sigma = nn.Linear(self.h_dim, self.z_dim)
        # module for image decoder
        self.linear = nn.Linear(self.z_dim, self.h_dim)
        self.d1 = DeConvBlock(256, 128, 4, 2, 1)                #128x8x8
        self.d2 = DeConvBlock(128, 64, 4, 2, 1)                 #64x16x16
        self.d3 = DeConvBlock(64, 64, 4, 2, 3)                  #64x28x28
        self.d4 = nn.ConvTranspose2d(64, self.image_channels, 5, 1, 2, bias=False)  #1x28x28
        self.img_module = nn.Sequential(self.d1, self.d2, self.d3, self.d4)
        self.fc = nn.Linear(784, 10)
        # self.d0 = ConvBlock(8, 256, kernel_size=3)
        self.mix1 = Mix(m=-1, size=256)
        self.mix2 = Mix(m=-0.6, size=128)
        self.mix3 = Mix(m=-0.4, size=64)

        self.block1 = DehazeBlock(default_conv, 64, 3)
        self.block2 = DehazeBlock(default_conv, 128, 3)
        self.block3 = DehazeBlock(default_conv, 256, 3)


    # Encoder
    def encode(self, x):
        self.batch_size = x.size(0)
        out1 = self.c1(x)
        out2 = self.c2(out1)
        outD1 = self.block1(out2)
        outD2 = self.block1(outD1)
        out3 = self.c3(outD2)
        outD3 = self.block2(out3)
        outD4 = self.block2(outD3)
        out4 = self.c4(outD4)
        outD5 = self.block3(out4)
        outD6 = self.block3(outD5)

        # x = self.e_module(x)
        x = outD6.view(self.batch_size, -1)
        # mean = self.relu(self.mu(x))
        # var = self.relu(self.sigma(x))
        mean = self.mu(x)
        var = self.sigma(x)

        # distribution = Normal(mean.softmax(dim=-1), var.softmax(dim=-1))
        distribution = Normal(mean, abs(var))
        # distribution = Normal(mean, var)

        return mean, var, distribution, out4, out3, out2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # Decoder for image denoising
    def img_decode(self, z, out4, out3, out2):
        self.batch_size = z.size(0)
        x = F.relu(self.linear(z))
        # x = z.view(self.batch_size, 8, 4, 4)
        # x = self.d0(x)

        x = x.view(self.batch_size, 256, 4, 4)
        x_out = self.mix1(out4, x, False)
        x_out = self.d1(x_out)
        x_out = self.mix2(out3, x_out, True)
        x_out = self.d2(x_out)
        x_out = self.mix3(out2, x_out, True)
        x_out = self.d3(x_out)
        x_out = self.d4(x_out)

        return torch.sigmoid((x_out))


        # return torch.sigmoid(self.img_module(x))
        # return torch.sigmoid(self.d4(self.d3(self.d2(self.d1(x)))))

    # Forward function
    def forward(self, x):
        mean, var, dist, out4, out3, out2 = self.encode(x)
        z = self.reparameterize(mean, var)
        output = self.img_decode(z, out4, out3, out2)
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

