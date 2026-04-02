import torch.nn as nn
import torch
class UpsamplingSubmodule(nn.Module):
    def __init__(self, in_channels):
        super(UpsamplingSubmodule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)  # 降低特征维度
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.upsample(x)
        return x
class Atmospheric_Prior_Module(nn.Module):
    def __init__(self,in_channels):
        super(Atmospheric_Prior_Module, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, 4, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(12, in_channels, kernel_size=3, padding=1)
    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        G = torch.cat([x1, x3, x5], dim=1)
        G = self.fuse(G)
        J = G * x - G + 1  # 元素级操作
        return J

class Clear_Channel_Guided_Module(nn.Module):
    def __init__(self, in_channel):
        super(Clear_Channel_Guided_Module, self).__init__()
        self.Uplayer = UpsamplingSubmodule(in_channel)
        self.Conv = nn.Conv2d(in_channel//8, 3, 3, padding=1)
        self.APM = Atmospheric_Prior_Module(in_channel//8)
    def forward(self, x):
        x = self.Uplayer(x)
        x = self.APM(x)
        x = self.Conv(x)
        return x