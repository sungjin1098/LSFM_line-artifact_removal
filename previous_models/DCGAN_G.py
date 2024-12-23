from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from torch.utils.data import DataLoader
from DCGAN_dataloader import LineDataset_S_is_added_Multiple_updates
from torchvision.utils import save_image, make_grid

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=2000, help='number of training epochs')
parser.add_argument("--valid_save_period", type=int, default=100, help='save the valid results for every N epochs')
parser.add_argument("--model_save_period", type=int, default=100, help='save the model pth file for every N epochs')

parser.add_argument("--lrG", type=float, default=0.0002, help="default 0.0002, learning rate of Generator")
parser.add_argument("--lrD", type=float, default=0.0002, help="default 0.0002, learning rate of Discriminator")

parser.add_argument("--batch_size", type=int, default=64, help="default 64, training batch size")
parser.add_argument("--image_size", type=int, default=64, help="default 64, training patch size")
parser.add_argument("--nz", type=int, default=100, help="default 100, latent vector size")
parser.add_argument("--ngf", type=int, default=64, help="default 64, # of channels in Generator")
parser.add_argument("--ndf", type=int, default=64, help="default 64, # of channels in Discriminator")

parser.add_argument("--in_dir", type=str,
                    default='./dataset/230321_6-2/6-2_20170915_LSM_zStackFish_vol20-130_crop1000-1800.tif',
                    help="dataset path")
parser.add_argument("--out_dir", type=str, default='./230913_03',
                    help='hyperparameters are saved at the end automatically')

parser.add_argument("--num_G", type=int, default=2, help="number of Generator learning iterations")

opt = parser.parse_args()

root = f'{opt.out_dir}_DCGAN_G{opt.num_G}_lrG_{opt.lrG}_lrD_{opt.lrD}_b{opt.batch_size}_p{opt.image_size}_nz{opt.nz}_ngf{opt.ngf}_ndf{opt.ndf}'
model_path = root + '/saved_models'
os.makedirs(model_path, exist_ok=True)

# dataloader에서 사용할 쓰레드 수
workers = 2
# 이미지의 채널 수
nc = 1
# Adam 옵티마이저의 beta1 하이퍼파라미터
beta1 = 0.5

train_data = LineDataset_S_is_added_Multiple_updates(img_path=opt.in_dir, synthetic_S_path=opt.in_dir,
                                                     patch_size=opt.image_size, num_update=opt.num_G)
dataloader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)

# GPU 사용여부를 결정해 줍니다
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ``netG`` 와 ``netD`` 에 적용시킬 커스텀 가중치 초기화 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다.
            nn.ConvTranspose2d(opt.nz, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(opt.ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 위의 계층을 통과한 데이터의 크기. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)


netG = Generator().to(device)
netG = nn.DataParallel(netG)
# 모든 가중치의 평균을 0( ``mean=0`` ), 분산을 0.02( ``stdev=0.02`` )로 초기화하기 위해 ``weight_init`` 함수를 적용시킵니다
netG.apply(weights_init)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 입력 데이터의 크기는 ``(nc) x 64 x 64`` 입니다
            nn.Conv2d(nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf) x 32 x 32``
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*2) x 16 x 16``
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*4) x 8 x 8``
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*8) x 4 x 4``
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



netD = Discriminator().to(device)
netD = nn.DataParallel(netD)
netD.apply(weights_init)

# ``BCELoss`` 함수의 인스턴스를 초기화합니다
criterion = nn.BCELoss()
criterion_MSE = nn.MSELoss()

# 생성자의 학습상태를 확인할 잠재 공간 벡터를 생성합니다
fixed_noise = torch.randn(64, opt.nz, 1, 1, device=device)

# 학습에 사용되는 참/거짓의 라벨을 정합니다
real_label = 1.
fake_label = 0.

# G와 D에서 사용할 Adam옵티마이저를 생성합니다
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
total_loss_check_valid_ep = []
iters = 0

print("Starting Training Loop...")
# 에폭(epoch) 반복
for epoch in range(opt.num_epochs):
    # 한 에폭 내에서 배치 반복
    total_loss_check_valid = []
    G_record = []
    D_record = []
    for i, data in enumerate(dataloader, 0):
        data = data.unsqueeze(1)
        if i % opt.num_G == 0:
            ############################
            # (1) D 신경망을 업데이트 합니다: log(D(x)) + log(1 - D(G(z)))를 최대화 합니다
            ###########################
            ## 진짜 데이터들로 학습을 합니다
            netD.zero_grad()
            # 배치들의 사이즈나 사용할 디바이스에 맞게 조정합니다
            data = data.to(device)
            high_data = torch.quantile(data, 0.99)
            real_cpu = data / high_data

            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # 진짜 데이터들로 이루어진 배치를 D에 통과시킵니다
            output = netD(real_cpu).view(-1)
            # 손실값을 구합니다
            errD_real = criterion(output, label)
            # 역전파의 과정에서 변화도를 계산합니다
            errD_real.backward()
            D_x = output.mean().item()

            ## 가짜 데이터들로 학습을 합니다
            # 생성자에 사용할 잠재공간 벡터를 생성합니다
            noise = torch.randn(b_size, opt.nz, 1, 1, device=device)
            # G를 이용해 가짜 이미지를 생성합니다
            fake = netG(noise)
            label.fill_(fake_label)
            # D를 이용해 데이터의 진위를 판별합니다
            output = netD(fake.detach()).view(-1)
            # D의 손실값을 계산합니다
            errD_fake = criterion(output, label)
            # 역전파를 통해 변화도를 계산합니다. 이때 앞서 구한 변화도에 더합니다(accumulate)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # 가짜 이미지와 진짜 이미지 모두에서 구한 손실값들을 더합니다
            # 이때 errD는 역전파에서 사용되지 않고, 이후 학습 상태를 리포팅(reporting)할 때 사용합니다
            errD = errD_real + errD_fake
            # D를 업데이트 합니다
            optimizerD.step()

        ############################
        # (2) G 신경망을 업데이트 합니다: log(D(G(z)))를 최대화 합니다
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # 생성자의 손실값을 구하기 위해 진짜 라벨을 이용할 겁니다
        # 우리는 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과시킵니다.
        # 이때 G는 업데이트되지 않았지만, D가 업데이트 되었기 때문에 앞선 손실값가 다른 값이 나오게 됩니다
        noise = torch.randn(b_size, opt.nz, 1, 1, device=device)
        # G를 이용해 가짜 이미지를 생성합니다
        fake = netG(noise)
        output = netD(fake).view(-1)
        # G의 손실값을 구합니다
        errG = criterion(output, label)
        # G의 변화도를 계산합니다
        errG.backward()
        D_G_z2 = output.mean().item()
        # G를 업데이트 합니다
        optimizerG.step()

        ### loss for check if 'GT' and our 'output X' is same. ###
        loss_check_valid = criterion_MSE(real_cpu, fake)
        total_loss_check_valid.append(loss_check_valid.item())
        ##########################################################

        # 훈련 상태를 출력합니다
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, opt.num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # 이후 그래프를 그리기 위해 손실값들을 저장해둡니다
        G_record.append(errG.item())
        D_record.append(errD.item())

        iters += 1

    total_loss_check_valid = sum(total_loss_check_valid) / len(total_loss_check_valid)
    G_record = sum(G_record) / len(G_record)
    D_record = sum(D_record) / len(D_record)

    total_loss_check_valid_ep.append(total_loss_check_valid)
    G_losses.append(G_record)
    D_losses.append(D_record)

    plt.close()
    plt.figure()
    plt.plot(total_loss_check_valid_ep)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MSE loss of {GT, output}')
    plt.savefig(f'{root}/check_loss.png')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')

    ax[0].plot(G_losses)
    ax[0].set_title("Generator Loss During Training")
    ax[1].plot(D_losses)
    ax[1].set_title("Discriminator Loss During Training")

    fig.supxlabel("Epoch")
    fig.supylabel("Loss")
    plt.savefig(f'{root}/traning_cruve.png')

    if (epoch + 1) % opt.valid_save_period == 0:
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            img_grid = make_grid(fake, padding=2, pad_value=1)
            save_image(img_grid, f'{root}/e{epoch + 1}.png')

    if (epoch + 1) % opt.model_save_period == 0:
        torch.save(netG.state_dict(), f"{model_path}/netG_e{epoch + 1}.pth")
        torch.save(netD.state_dict(), f"{model_path}/netD_e{epoch + 1}.pth")
