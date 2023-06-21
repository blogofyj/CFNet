import torch, sys, os
import torch.nn as nn
sys.path.append(os.pardir)
# import sys
# sys.path.append("..")
import torch.nn.functional as F
from torch.nn import init
import functools
from purifier_vae.deconv import FastDeconv
from purifier_vae.DCNv2.dcn_v2 import DCN



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

class DCNBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DCNBlock, self).__init__()
        self.dcn = DCN(in_channel, out_channel, kernel_size=(3,3), stride=1, padding=1).cuda()
    def forward(self, x):
        return self.dcn(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class Dehaze(nn.Module):
    def __init__(self):
        super(Dehaze, self).__init__()

        ###### downsample
        self.down1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))


        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(64, 6, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(6),
                                 nn.ReLU(True))
        self.outc = nn.Conv2d(6, 3, kernel_size=1)

        # self.inc = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #                          nn.BatchNorm2d(64),
        #                          nn.ReLU(inplace=True),
        #                          nn.MaxPool2d(2),
        #                          nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #                          nn.BatchNorm2d(64),
        #                          nn.ReLU(inplace=True)) #112
        # self.down1 = nn.Sequential(nn.MaxPool2d(2),
        #                            nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #                            nn.BatchNorm2d(128),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #                            nn.BatchNorm2d(128),
        #                            nn.ReLU(inplace=True)
        #                            ) #56
        # self.down2 = nn.Sequential(nn.MaxPool2d(2),
        #                            nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(inplace=True)
        #                            ) #28
        # self.down3 = nn.Sequential(nn.MaxPool2d(2),
        #                            nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #                            nn.BatchNorm2d(512),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #                            nn.BatchNorm2d(512),
        #                            nn.ReLU(inplace=True)
        #                            ) #14
        # self.down4 = nn.Sequential(nn.MaxPool2d(2),
        #                            nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #                            nn.BatchNorm2d(512),
        #                            nn.ReLU(inplace=True),
        #                            nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #                            nn.BatchNorm2d(512),
        #                            nn.ReLU(inplace=True)
        #                            ) #7

        ###### FFA blocks
        self.block = DehazeBlock(default_conv, 256, 3)

        ###### upsample
        # self.up1 = Up(1024, 256) #14
        # self.up2 = Up(512, 128) #28
        # self.up3 = Up(256, 64) #56
        # self.up4 = Up(128, 64) #112
        # self.up5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                          nn.Conv2d(64, 32, kernel_size=3, padding=1),
        #                          nn.BatchNorm2d(32),
        #                          nn.ReLU(inplace=True),
        #                          nn.Conv2d(32, 6, kernel_size=3, padding=1),
        #                          nn.BatchNorm2d(6),
        #                          nn.ReLU(inplace=True)
        #                          )
        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)
        # self.mix3 = Mix(m=-0.7)
        # self.mix4 = Mix(m=-0.6)

        self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)
        self.dcn_block = DCNBlock(256, 256)


    def forward(self, input):
        x_deconv = self.deconv(input)
        x_down1 = self.down1(x_deconv)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)

        x1 = self.block(x_down3)
        x2 = self.block(x1)
        x3 = self.block(x2)
        x4 = self.block(x3)
        x5 = self.block(x4)
        x6 = self.block(x5)

        x_dcn1 = self.dcn_block(x2)
        x_dcn2 = self.dcn_block(x_dcn1)

        x_out_mix = self.mix1(x_down3, x_dcn2)
        x_up1 = self.up1(x_out_mix)  # [bs, 128, 128, 128]

        x_up1_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_up1_mix)  # [bs, 64, 256, 256]


        out = self.up3(x_up2)
        final_out = self.outc(out)

        # final_out = torch.sigmoid(final_out)
        # out = torch.sigmoid(out)
        return out, final_out

        # x_down1 = self.inc(input) # [bs, 64, 28, 28]
        # x_down2 = self.down1(x_down1) # [bs, 64, 16, 16]
        # x_down3 = self.down2(x_down2) # [bs, 128, 8, 8]
        # x_down4 = self.down3(x_down3)
        # x_down5 = self.down4(x_down4)
        #
        # x1 = self.block(x_down5)
        # x2 = self.block(x1)
        # # x3 = self.block(x2)
        # # x4 = self.block(x3)
        # # x5 = self.block(x4)
        # # x6 = self.block(x5)
        #
        # x_dcn1 = self.dcn_block(x2)
        # # x_dcn2 = self.dcn_block(x_dcn1)
        #
        # x = self.up1(x_dcn1, x_down4)
        # x = self.up2(x, x_down3)
        # x = self.up3(x, x_down2)
        # x = self.up4(x, x_down1)
        # out = self.up5(x)
        # final_out = self.outc(out)
        # out = torch.sigmoid(out)
        # final_out = torch.sigmoid(final_out)
        #
        # return out, final_out