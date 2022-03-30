#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Time    :   2022/3/29
# Author  :   XavierYorke
# Contact :   mzlxavier1230@gmail.com

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool3d(2)

    def forward(self, x):
        bn2 = self.conv(x)

        return bn2


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Res_block3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Res_block3d, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(out_ch),
                                   nn.LeakyReLU(inplace=True),
                                   )
        self.conv2 = nn.Sequential(nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(out_ch),
                                   nn.LeakyReLU(inplace=True),
                                   )
        self.conv3 = nn.Sequential(nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(out_ch))

        self.leak_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        out = self.leak_relu(conv1 + conv3)

        return out


class ResUnet(nn.Module):
    def __init__(self, in_ch=1, num_class=1):
        super(ResUnet, self).__init__()
        # encoder
        self.conv1 = ConvBlock(in_ch, 32)
        self.conv2 = Res_block3d(32, 64)
        self.conv3 = Res_block3d(64, 128)
        self.conv4 = nn.Sequential(nn.Conv3d(128, 256, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(256),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv3d(256, 256, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(256),
                                   nn.LeakyReLU(inplace=True)
                                   )

        # decoder
        self.up4 = nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.conv_up4 = DoubleConv(384, 128)

        self.up3 = nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.conv_up3 = DoubleConv(192, 64)

        self.up2 = nn.ConvTranspose3d(64, 64, 2, stride=2)
        self.conv_up2 = DoubleConv(96, 32)

        self.conv_up = nn.Conv3d(32, num_class, 1)
        self.pool = nn.AvgPool3d(2)

        self.h3 = nn.Sequential(nn.ConvTranspose3d(in_channels=128, out_channels=1, kernel_size=4, stride=4))

    def forward(self, x):
        # encoder
        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        h3 = self.h3(conv3)

        pool3 = self.pool(conv3)
        conv4 = self.conv4(pool3)

        # decoder
        up4 = self.up4(conv4)
        merge4 = torch.cat([up4, conv3], dim=1)
        conv_up4 = self.conv_up4(merge4)

        up3 = self.up3(conv_up4)
        merge3 = torch.cat([up3, conv2], dim=1)
        conv_up3 = self.conv_up3(merge3)

        up2 = self.up2(conv_up3)
        merge2 = torch.cat([conv1, up2], dim=1)
        conv_up2 = self.conv_up2(merge2)

        conv_up = self.conv_up(conv_up2)

        out = nn.Sigmoid()(conv_up)
        h3 = nn.Sigmoid()(h3)

        # return out, h3
        return out


if __name__ == '__main__':
    from torchsummary import summary

    model = ResUnet()

    summary(model, (1, 48, 48, 48), 2, device='cpu')
