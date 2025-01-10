import torch
import torch.nn as nn
from torch.nn import functional as F

class Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.layer(x)

class DownSampling(nn.Module):
    def __init__(self, channel):
        super(DownSampling, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSampling(nn.Module):

    def __init__(self, channel):
        super(UpSampling, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)
    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.layer(up)
        return torch.cat((x, feature_map), 1)


class UNet(nn.Module):
    def __init__(self, input_channels, input_size, num_classes):
        super(UNet, self).__init__()

        # 下采样
        self.C1 = Conv(input_channels, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 1, 3, 1, 1)
        self.fc1 = torch.nn.Linear(torch.tensor(input_size).prod().item(), num_classes)
    def forward(self, x):

        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))


        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))
        x1 = self.pred(O4)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.fc1(x1)

        return self.Th(x2)


if __name__ == '__main__':
    a = torch.randn(5, 3, 256, 256)
    net = UNet(input_channels=3, input_size=(256,256), num_classes=2)
    print(net(a).shape)
