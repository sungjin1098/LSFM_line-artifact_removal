import torch
from torch import nn
import torch.nn.functional as F


# DenseLayer class 는 x와, x-conv-relu 거친 것을, channel이 두꺼워지게 쌓음
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


# G: growth rate, RDB 내에서 각 conv의 출력 feature 수
# C: num_layers, RDB 내 conv layer의 수
# RDB 내에서 dense해질수록 feature 두께가 in_channels + G*(c-1)이 됨
class RDB(nn.Module):
    def __init__(self, in_channels, G, C):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + G * i, G) for i in range(C)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + G * C, G, kernel_size=1)

    def forward(self, x):
        # local residual learning: RDB 거친 뒤에 local feature fusion까지 수행
        return x + self.lff(self.layers(x))


# G0: num_features, 처음 RDB는 input feature 수가 G0
# D: number of RDBs
class RDN(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks: RDBs + local feature fusions
        # 처음 RDB 는 input feature 수가 G0임
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])

        # 그 이후로 D-1 개의 RDB 는 input feature 수가 G
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion1
        # 각 RDB 마다 feature G개씩, 총 RDB 수 D 개가 들어가서 output은 G0개
        # 처음 sfe1 통과한 결과인, G0개 들어가서 output은 G0개 인 것과 합치기
        self.gff1 = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # global featuer fusion2
        # gff1 결과는 G0 인데 conv 통과하면 input channel 수인 1이어야 함 (grayscale 인 경우)
        # forward 단계에서 이 gff2 결과랑 처음 input 과 합칠 예정
        self.gff2 = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        a = x
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        # RDB D 개에 대해 반복
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff1(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.gff2(x) + a

        # to prevent divided-by-zero
        r = 5
        mask_adjusted = (a + r * torch.ones_like(a)) / (x + r * torch.ones_like(x))

        return x, mask_adjusted


class STN(nn.Module):
    def __init__(self, data_shape, device):
        super(STN, self).__init__()
        self.b, self.w, self.h = data_shape
        self.device = device
        self.theta = nn.Parameter(torch.zeros(1, dtype=torch.float, device=self.device).requires_grad_(True))

    def forward(self, x):
        theta = torch.zeros((self.b, 2, 3), dtype=torch.float, device=self.device)
        theta[:, 0, 0] = torch.cos(self.theta)
        theta[:, 1, 1] = torch.cos(self.theta)
        theta[:, 0, 1] = -torch.sin(self.theta)
        theta[:, 1, 0] = torch.sin(self.theta)
        
        x_reshape = x.view(-1, 1, self.w, self.h)        # (b, 1, w, h) except last batch
        theta = theta[:x_reshape.shape[0]]               # for last batch
        grid = F.affine_grid(theta, x_reshape.size())
        # mask after STN = x_reg
        x_reg = F.grid_sample(x_reshape, grid).view(-1, 1, self.w, self.h)    # (b, inp) except last batch

        return x_reg
        
