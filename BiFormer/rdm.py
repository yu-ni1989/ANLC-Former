
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class RDM(nn.Module):
    def __init__(self, cs_dim=128, rdm_dim=64, out_dim=64):
        super(RDM, self).__init__()
        self.cs_dim = cs_dim
        self.rdm_dim = rdm_dim
        self.out_dim = out_dim

        self.red_l1 = nn.Sequential(
            nn.Conv2d(in_channels=self.cs_dim, out_channels=self.rdm_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.rdm_dim),
            nn.ReLU()
        )

        self.apool_l = nn.AdaptiveAvgPool2d(1)
        self.cconv = nn.Sequential(
            nn.Conv2d(2*self.rdm_dim, self.rdm_dim, 1),
            nn.BatchNorm2d(self.rdm_dim),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_low, x_high):
        size_low = x_low.shape
        size_high = x_high.shape

        size_high_cos = x_high.detach().shape
        x_low_temp = F.interpolate(x_low, size_high_cos[-2:], mode='bilinear', align_corners=True)
        x_low_temp = x_low_temp.view(size_low[0], -1, size_high_cos[2] * size_high_cos[3])
        x_high_cos = x_high.detach().view(size_high_cos[0], -1, size_high_cos[2] * size_high_cos[3]).permute(0, 2, 1)
        x_low_temp_n = torch.norm(x_low_temp, p='fro', dim=2).unsqueeze(2) + 1e-12
        x_high_cos_n = torch.norm(x_high_cos, p='fro', dim=1).unsqueeze(1) + 1e-12
        cos = torch.bmm(x_low_temp / x_low_temp_n, x_high_cos / x_high_cos_n)
        cos = F.softmax(cos, dim=2)
        x_high_cos = x_high_cos.permute(0, 2, 1)
        x_high_cos = torch.bmm(cos, x_high_cos)
        x_high_cos = x_high_cos.view(size_low[0], size_low[1], size_high_cos[2], size_high_cos[3])
        x_high_cos = F.interpolate(x_high_cos, size_low[-2:], mode='bilinear', align_corners=True)

        x_high_red = self.red_l1(x_high.detach())
        x_high_red = F.interpolate(x_high_red, size_low[-2:], mode='bilinear', align_corners=True)

        x_att_v = self.apool_l(torch.cat((x_low, x_high_red), dim=1))
        x_att_v = self.cconv(x_att_v)
        x_high_red = x_high_red * x_att_v

        x_diff = []
        x_diff.append(self.sigmoid(x_low) - self.sigmoid(x_high_cos))
        x_diff.append(self.sigmoid(x_low) - self.sigmoid(x_high_red))

        x_diff = torch.cat(x_diff, 1)
        x_diff = self.relu(x_diff)

        return x_diff


class IS(nn.Module):
    def __init__(self, in_dim, ds_dim=192):
        super(IS, self).__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=ds_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(ds_dim),
            nn.ReLU()
        )

        self.gcd = np.gcd(in_dim, ds_dim)
        self.dwconvm = nn.Sequential(
            nn.Conv2d(in_dim, ds_dim, kernel_size=3, stride=1, padding=1, groups=self.gcd, bias=False),
            nn.BatchNorm2d(ds_dim),
            nn.ReLU(inplace=True)
        )
        self.apool_d = nn.AdaptiveAvgPool2d(1)
        self.conv_d = nn.Conv2d(
            1, 1, kernel_size=3, padding=1, bias=False)

        self.sigmoid_d = nn.Sigmoid()
        self.relu_lm = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.dconv(x)

        x2 = self.dwconvm(x)
        lamb = self.apool_d(x2)
        lamb = self.conv_d(lamb.transpose(-1, -3)).transpose(-1, -3)
        lamb = self.sigmoid_d(lamb)

        x = x1 + x2 * lamb.expand_as(x2)
        x = self.relu_lm(x)

        return x



