import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_features=16):
        super(UNet, self).__init__()

        # Conv + BatchNorm + ReLU
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, padding_mode='reflect', bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            CBR = nn.Sequential(*layers)

            return CBR

        # Conv + ReLU
        def CR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, padding_mode='reflect', bias=bias)]
            # layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            CBR = nn.Sequential(*layers)

            return CBR

        # U-Net contracting path
        self.enc1_1 = CR2d(in_channels=1, out_channels=n_features)
        self.enc1_2 = CR2d(in_channels=n_features, out_channels=n_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=n_features, out_channels=n_features*2)
        self.enc2_2 = CBR2d(in_channels=n_features*2, out_channels=n_features*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=n_features*2, out_channels=n_features*4)
        self.enc3_2 = CBR2d(in_channels=n_features*4, out_channels=n_features*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=n_features*4, out_channels=n_features*8)
        self.enc4_2 = CBR2d(in_channels=n_features*8, out_channels=n_features*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=n_features*8, out_channels=n_features*16)

        # U-Net expansive path
        self.dec5_1 = CBR2d(in_channels=n_features*16, out_channels=n_features*8)

        self.unpool4 = nn.ConvTranspose2d(in_channels=n_features*8, out_channels=n_features*8, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_2 = CBR2d(in_channels=n_features*16, out_channels=n_features*8)
        self.dec4_1 = CBR2d(in_channels=n_features*8, out_channels=n_features*4)

        self.unpool3 = nn.ConvTranspose2d(in_channels=n_features*4, out_channels=n_features*4, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_2 = CBR2d(in_channels=n_features*8, out_channels=n_features*4)
        self.dec3_1 = CBR2d(in_channels=n_features*4, out_channels=n_features*2)

        self.unpool2 = nn.ConvTranspose2d(in_channels=n_features*2, out_channels=n_features*2, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_2 = CBR2d(in_channels=n_features*4, out_channels=n_features*2)
        self.dec2_1 = CBR2d(in_channels=n_features*2, out_channels=n_features)

        self.unpool1 = nn.ConvTranspose2d(in_channels=n_features, out_channels=n_features, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_2 = CR2d(in_channels=n_features*2, out_channels=n_features)
        self.dec1_1 = CR2d(in_channels=n_features, out_channels=n_features)

        self.fc = nn.Conv2d(in_channels=n_features, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x


class STN(nn.Module):
    def __init__(self, data_shape, device, k=1):
        super(STN, self).__init__()
        self.b, self.w, self.h = data_shape
        # self.inp = self.w * self.h
        # self.ln = nn.Linear(self.inp, k, bias=False)
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
        x_reg = F.grid_sample(x_reshape, grid).view(-1, 1, self.w, self.h)    # (b, inp) except last batch
        # L = self.ln(x_reg) @ self.ln.weight
        return x_reg, self.theta


class UNet_with_STN(nn.Module):
    def __init__(self, data_shape, device, n_features=16):
        super(UNet_with_STN, self).__init__()

        ### STN ###
        self.b, self.w, self.h = data_shape
        self.device = device
        self.theta = torch.zeros(1, dtype=torch.float, device=self.device).requires_grad_(True)
        ###########

        # Conv + BatchNorm + ReLU
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, padding_mode='reflect', bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            CBR = nn.Sequential(*layers)

            return CBR

        # Conv + ReLU
        def CR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, padding_mode='reflect', bias=bias)]
            # layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            CBR = nn.Sequential(*layers)

            return CBR
        
        def fc(in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            # layers += [nn.Sigmoid()]
            
            fc = nn.Sequential(*layers)
            
            return fc
        
        # U-Net contracting path
        self.enc1_1 = CR2d(in_channels=1, out_channels=n_features)
        self.enc1_2 = CR2d(in_channels=n_features, out_channels=n_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=n_features, out_channels=n_features*2)
        self.enc2_2 = CBR2d(in_channels=n_features*2, out_channels=n_features*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=n_features*2, out_channels=n_features*4)
        self.enc3_2 = CBR2d(in_channels=n_features*4, out_channels=n_features*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=n_features*4, out_channels=n_features*8)
        self.enc4_2 = CBR2d(in_channels=n_features*8, out_channels=n_features*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=n_features*8, out_channels=n_features*16)

        # U-Net expansive path
        self.dec5_1 = CBR2d(in_channels=n_features*16, out_channels=n_features*8)

        self.unpool4 = nn.ConvTranspose2d(in_channels=n_features*8, out_channels=n_features*8, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_2 = CBR2d(in_channels=n_features*16, out_channels=n_features*8)
        self.dec4_1 = CBR2d(in_channels=n_features*8, out_channels=n_features*4)

        self.unpool3 = nn.ConvTranspose2d(in_channels=n_features*4, out_channels=n_features*4, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_2 = CBR2d(in_channels=n_features*8, out_channels=n_features*4)
        self.dec3_1 = CBR2d(in_channels=n_features*4, out_channels=n_features*2)

        self.unpool2 = nn.ConvTranspose2d(in_channels=n_features*2, out_channels=n_features*2, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_2 = CBR2d(in_channels=n_features*4, out_channels=n_features*2)
        self.dec2_1 = CBR2d(in_channels=n_features*2, out_channels=n_features)

        self.unpool1 = nn.ConvTranspose2d(in_channels=n_features, out_channels=n_features, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_2 = CR2d(in_channels=n_features*2, out_channels=n_features)
        self.dec1_1 = CR2d(in_channels=n_features, out_channels=n_features)

        # self.fc = nn.Conv2d(in_channels=n_features, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc = fc(in_channels=n_features, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
	
    def forward(self, x):
        img = x
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        # network output = x
        x = self.fc(dec1_1)

        # mask; line artifact = x
        mask = img / x

        ### STN ###
        theta = torch.zeros((self.b, 2, 3), dtype=torch.float, device=self.device)
        theta[:, 0, 0] = torch.cos(self.theta)
        theta[:, 1, 1] = torch.cos(self.theta)
        theta[:, 0, 1] = -torch.sin(self.theta)
        theta[:, 1, 0] = torch.sin(self.theta)

        x_reshape = mask.view(-1, 1, self.w, self.h)  # (b, 1, w, h) except last batch
        theta = theta[:x_reshape.shape[0]]  # for last batch
        grid = F.affine_grid(theta, x_reshape.size())
        # mask after STN = x_reg
        x_reg = F.grid_sample(x_reshape, grid).view(-1, 1, self.w, self.h)  # (b, 1, w, h) except last batch
        ###########

        return x, mask, x_reg
