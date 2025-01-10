import torch.nn as nn
import torch

import torch
import torch.nn as nn

# Double Convolution (2x{Conv + BN + Activation})
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))  # Dropout layer
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


# Down-sampling Block (2xConv + Pooling)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_dropout)
        )

    def forward(self, x):
        return self.pool_conv(x)


# Up-sampling Block (Upsample + Skip Connection + Double Conv)
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, use_dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure x1 and x2 have the same size
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Output Layer (1x1 Conv)
class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# Full U-Net Architecture
class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(UNet, self).__init__()
        self.in_conv = DoubleConv(in_channels, 64)  # Initial convolution
        self.down1 = Down(64, 128)  # Down-sampling blocks
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048, use_dropout=True)
        self.down6 = Down(2048, 4096, use_dropout=True)

        self.up1 = Up(4096, 2048, use_dropout=True)  # Up-sampling blocks
        self.up2 = Up(2048, 1024, use_dropout=True)
        self.up3 = Up(1024, 512)
        self.up4 = Up(512, 256)
        self.up5 = Up(256, 128)
        self.up6 = Up(128, 64)

        self.out_conv = OutConv(64, num_classes)  # Output layer

    def forward(self, x):
        # Encoder
        x1 = self.in_conv(x)  # Input to DoubleConv
        x2 = self.down1(x1)  # Down1
        x3 = self.down2(x2)  # Down2
        x4 = self.down3(x3)  # Down3
        x5 = self.down4(x4)  # Down4
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        # Decoder
        x = self.up1(x7, x6)  # Up1
        x = self.up2(x, x5)  # Up2
        x = self.up3(x, x4)  # Up3
        x = self.up4(x, x3)  # Up4
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        # Output
        x = self.out_conv(x)
        return x

if __name__ == '__main__':
# 计算 UNet 的网络参数个数
    a = torch.randn(5, 3, 256, 256)
    net = UNet(in_channels=3, num_classes=4)
    print(net(a).shape)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))