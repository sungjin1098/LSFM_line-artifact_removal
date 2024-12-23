from __future__ import print_function
#%matplotlib inline
import argparse
import os
import argparse
from skimage import io
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
from torchvision.utils import save_image, make_grid

from synthetic_dataloader_3Dpre import LineDataset_S_is_added_Multiple_updates, LineDataset_valid
from torch.utils.data import DataLoader
# from model_UNetv8 import UNet_3Plus_for_S, UNet_3Plus_for_X_train, STN, Discriminator, MLP


import os
import torch
import argparse
import warnings
import torchvision
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from skimage import io
from torch import nn
from os import path
from torch.utils.data import DataLoader
import shutil

# from model_UNetv7 import Discriminator
from dataloader_test_discriminator_by_using_Y import LineDataset_3D, LineDatasetValid

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1000, help='number of training epochs')
parser.add_argument("--lrDis", type=float, default=1e-4, help="Discriminator learning rate")

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay of optimizer")
parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
parser.add_argument("--patch_size", type=int, default=256, help="training patch size")

parser.add_argument("--in_dir", type=str, default='./dataset/230919_testGAN/230919_input_220vol.tif', help="dataset path")
parser.add_argument("--valid_dir", type=str, default='./dataset/230919_testGAN/valid_input', help="dataset path")
parser.add_argument("--out_dir", default='./230919_testGAN', type=str)

opt = parser.parse_args()

# save path
root = opt.out_dir
plot_path = root + '/loss_curve'
os.makedirs(plot_path, exist_ok=True)

# Prepare for use of CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataloader
train_data = LineDataset_3D(opt.in_dir, opt.patch_size)
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)

# Sample image for model definition and valid image for validation
valid_data = LineDatasetValid(opt.valid_dir)
valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)

# Prepare for use of CUDA

# 코드 실행결과의 동일성을 위해 무작위 시드를 설정합니다
manualSeed = 999
#manualSeed = random.randint(1, 10000) # 만일 새로운 결과를 원한다면 주석을 없애면 됩니다
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

workers = 2

batch_size = 8

image_size = 256

nc = 1

nz = 100

ngf = 64

ndf = 64

num_epochs = 6000

# 옵티마이저의 학습률
lr = opt.lrDis

#dis
# lr1 = opt.lr1
# lr2 = 0.0001

# Adam 옵티마이저의 beta1 하이퍼파라미터
beta1 = 0.5

# 사용가능한 gpu 번호. CPU를 사용해야 하는경우 0으로 설정하세요
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 학습 데이터들 중 몇가지 이미지들을 화면에 띄워봅시다

# ``netG`` 와 ``netD`` 에 적용시킬 커스텀 가중치 초기화 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# 생성자 코드
# Dataloader
# dataset = LineDataset_S_is_added_Multiple_updates(img_path=opt.in_dir, synthetic_S_path=opt.synthetic_S_dir, gt_path=opt.in_dir, patch_size=image_size, num_update=1)
# dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# valid_data = LineDataset_valid(index_path=opt.valid_idx_path, valid_path=opt.in_dir, synthetic_S_path=opt.synthetic_S_dir, gt_path=opt.gt_dir, patch_size=image_size)
# valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True,drop_last=True)

# # 생성자를 만듭니다
# netG = Generator().to(device)
# netG_S = Generator().to(device)
#
# # 필요한 경우 multi-GPU를 설정 해주세요
# if (device.type == 'cuda') and (ngpu > 1):
#     netG = nn.DataParallel(netG, list(range(ngpu)))
#
# # 모든 가중치의 평균을 0( ``mean=0`` ), 분산을 0.02( ``stdev=0.02`` )로 초기화하기 위해
# # ``weight_init`` 함수를 적용시킵니다
# netG.apply(weights_init)
# netG_S.apply(weights_init)

# 모델의 구조를 출력합니다


# 구분자 코드
# Loss function for L1 loss (reconstruction loss)
criterion = nn.BCELoss()


# Total variation

# optimizers
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 입력 데이터의 크기는 ``(nc) x 64 x 64`` 입니다
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 16, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# 구분자를 만듭니다
modelDiscriminator = Discriminator().to(device)
# netD_S = Discriminator().to(device)
modelDiscriminator = nn.DataParallel(modelDiscriminator)
# # 필요한 경우 multi-GPU를 설정 해주세요
# if (device.type == 'cuda') and (ngpu > 1):
#     netD = nn.DataParallel(netD, list(range(ngpu)))

# 모든 가중치의 평균을 0( ``mean=0`` ), 분산을 0.02( ``stdev=0.02`` )로 초기화하기 위해
# ``weight_init`` 함수를 적용시킵니다
modelDiscriminator.apply(weights_init)
# netD_S.apply(weights_init)

# 모델의 구조를 출력합니다

# # ``BCELoss`` 함수의 인스턴스를 초기화합니다
# criterion = nn.BCELoss()

# optimizers
optimDiscriminator = torch.optim.Adam(modelDiscriminator.parameters(), lr=opt.lrDis, weight_decay=opt.weight_decay)

# schedulers
schedulerDiscriminator = torch.optim.lr_scheduler.StepLR(optimDiscriminator, step_size=50, gamma=0.5)

# To draw loss graph
train_loss_ep = []
epoch_ep = []
value_record1 = []
value_record2 = []
accuracy_ep = []

# To record best accuracy
best = 0

### train
for e in range(opt.epoch):
    epoch_ep.append(e + 1)
    is_best = False

    modelDiscriminator.train()

    total_loss = []
    temp1 = []
    temp2 = []

    for idx, img in enumerate(train_loader):
        # (b, h, w) --> (b, c, h, w) in this case, (b, h, w) --> (b, 1, h, w)
        img = img.unsqueeze(1)

        # Normalize using 99% percentile value (0~65525 -> 0~1.6)
        high = torch.quantile(img, 0.99)
        img = img / high

        img = img.to(device)

        dis_output = modelDiscriminator(img.clone().detach())
        dis_output_rotated = modelDiscriminator(torch.rot90(img.clone().detach(), k=1, dims=[2,3]))

        loss = torch.mean((dis_output_rotated - 1)**2) + torch.mean((dis_output - 0)**2)

        temp1.append(np.mean(dis_output.detach().cpu().numpy()))
        temp2.append(np.mean(dis_output_rotated.detach().cpu().numpy()))

        total_loss.append(loss.item())

        optimDiscriminator.zero_grad()
        loss.backward(retain_graph=True)
        optimDiscriminator.step()

    schedulerDiscriminator.step()

    total_loss_mean = sum(total_loss) / len(total_loss)
    print(f'[{e}/{opt.epoch}] Loss: {total_loss_mean:.3f}')

    temp1_mean= sum(temp1) / len(temp1)
    temp2_mean = sum(temp2) / len(temp2)

    train_loss_ep.append(total_loss_mean)
    value_record1.append(temp1_mean)
    value_record2.append(temp2_mean)


    ### validation #################
    accuracy_cnt = 0

    for idx, (img, img_name) in enumerate(valid_loader):
        modelDiscriminator.eval()

        with torch.no_grad():
            img = img.unsqueeze(0)

            # Normalize using 99% percentile value (0~65525 -> 0~1.6)
            high = torch.quantile(img, 0.99)
            img = img / high

            img = img.to(device)

            dis_output = modelDiscriminator(img.clone().detach())
            dis_output_rotated = modelDiscriminator(torch.rot90(img.clone().detach(), k=1, dims=[2, 3]))

            img_vis = img.squeeze().detach().cpu()
            img_vis = (img_vis * high).numpy().astype('uint16')  # denormalize using only max

            img_rot_vis = torch.rot90(img, k=1, dims=[2, 3]).squeeze().detach().cpu()
            img_rot_vis = (img_rot_vis * high).numpy().astype('uint16')

            # oo
            if dis_output < 0.5 and dis_output_rotated > 0.5:
                accuracy_cnt += 1

    accuracy = accuracy_cnt / len(valid_loader) * 100
    accuracy_ep.append(accuracy)

    if accuracy > best:
        best = accuracy
        is_best = True
        if path.exists(root + f'/best'):
            shutil.rmtree(root + f'/best')

    print(f'Accuracy: {accuracy:.3f}, Best: {best:.3f}')

    if is_best == True:
        mis_path_oo = root + f'/best/epoch{str(e).zfill(3)}' + '/oo'
        mis_path_ox = root + f'/best/epoch{str(e).zfill(3)}' + '/ox'
        mis_path_xo = root + f'/best/epoch{str(e).zfill(3)}' + '/xo'
        mis_path_xx = root + f'/best/epoch{str(e).zfill(3)}' + '/xx'
        os.makedirs(mis_path_oo, exist_ok=True)
        os.makedirs(mis_path_ox, exist_ok=True)
        os.makedirs(mis_path_xo, exist_ok=True)
        os.makedirs(mis_path_xx, exist_ok=True)
        mis_path_oo_line = root + f'/best/epoch{str(e).zfill(3)}' + '/oo_line'
        mis_path_ox_line = root + f'/best/epoch{str(e).zfill(3)}' + '/ox_line'
        mis_path_xo_line = root + f'/best/epoch{str(e).zfill(3)}' + '/xo_line'
        mis_path_xx_line = root + f'/best/epoch{str(e).zfill(3)}' + '/xx_line'
        os.makedirs(mis_path_oo_line, exist_ok=True)
        os.makedirs(mis_path_ox_line, exist_ok=True)
        os.makedirs(mis_path_xo_line, exist_ok=True)
        os.makedirs(mis_path_xx_line, exist_ok=True)

        accuracy_record = root + f'/best/epoch{str(e).zfill(3)}_acc{accuracy:.3f}'
        os.makedirs(accuracy_record, exist_ok=True)

        model_save_dir = root + '/best/model.pth'
        torch.save(modelDiscriminator.state_dict(), model_save_dir)

        for idx, (img, img_name) in enumerate(valid_loader):
            modelDiscriminator.eval()

            with torch.no_grad():
                img = img.unsqueeze(0)

                # Normalize using 99% percentile value (0~65525 -> 0~1.6)
                high = torch.quantile(img, 0.99)
                img = img / high

                img = img.to(device)

                dis_output = modelDiscriminator(img.clone().detach())
                dis_output_rotated = modelDiscriminator(torch.rot90(img.clone().detach(), k=1, dims=[2, 3]))

                img_vis = img.squeeze().detach().cpu()
                img_vis = (img_vis * high).numpy().astype('uint16')  # denormalize using only max

                img_rot_vis = torch.rot90(img, k=1, dims=[2, 3]).squeeze().detach().cpu()
                img_rot_vis = (img_rot_vis * high).numpy().astype('uint16')

                if idx >= 110:
                    if dis_output < 0.5 and dis_output_rotated > 0.5:
                        io.imsave(f'{mis_path_oo}/{img_name}_{dis_output.item():.3f}.tif', img_vis)
                        io.imsave(f'{mis_path_oo}/{img_name}_rot_{dis_output_rotated.item():.3f}.tif', img_rot_vis)
                    elif dis_output < 0.5 and dis_output_rotated < 0.5:
                        io.imsave(f'{mis_path_ox}/{img_name}_{dis_output.item():.3f}.tif', img_vis)
                        io.imsave(f'{mis_path_ox}/{img_name}_rot_{dis_output_rotated.item():.3f}.tif', img_rot_vis)
                    elif dis_output > 0.5 and dis_output_rotated > 0.5:
                        io.imsave(f'{mis_path_xo}/{img_name}_{dis_output.item():.3f}.tif', img_vis)
                        io.imsave(f'{mis_path_xo}/{img_name}_rot_{dis_output_rotated.item():.3f}.tif', img_rot_vis)
                    else:
                        io.imsave(f'{mis_path_xx}/{img_name}_{dis_output.item():.3f}.tif', img_vis)
                        io.imsave(f'{mis_path_xx}/{img_name}_rot_{dis_output_rotated.item():.3f}.tif', img_rot_vis)
                else:
                    if dis_output < 0.5 and dis_output_rotated > 0.5:
                        io.imsave(f'{mis_path_oo_line}/{img_name}_{dis_output.item():.3f}.tif', img_vis)
                        io.imsave(f'{mis_path_oo_line}/{img_name}_rot_{dis_output_rotated.item():.3f}.tif', img_rot_vis)
                    elif dis_output < 0.5 and dis_output_rotated < 0.5:
                        io.imsave(f'{mis_path_ox_line}/{img_name}_{dis_output.item():.3f}.tif', img_vis)
                        io.imsave(f'{mis_path_ox_line}/{img_name}_rot_{dis_output_rotated.item():.3f}.tif', img_rot_vis)
                    elif dis_output > 0.5 and dis_output_rotated > 0.5:
                        io.imsave(f'{mis_path_xo_line}/{img_name}_{dis_output.item():.3f}.tif', img_vis)
                        io.imsave(f'{mis_path_xo_line}/{img_name}_rot_{dis_output_rotated.item():.3f}.tif', img_rot_vis)
                    else:
                        io.imsave(f'{mis_path_xx_line}/{img_name}_{dis_output.item():.3f}.tif', img_vis)
                        io.imsave(f'{mis_path_xx_line}/{img_name}_rot_{dis_output_rotated.item():.3f}.tif', img_rot_vis)
    ############################################

    ############################################
    # Save loss curves
    epoch_ep_n = np.array(epoch_ep)

    plt.clf()
    plt.plot(epoch_ep_n, np.array(value_record1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'dis_output_should_be_0')
    plt.savefig(f'{plot_path}/dis_output.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(value_record2))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'dis_output_rotate_should_be_1')
    plt.savefig(f'{plot_path}/dis_output_rotate.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_loss_ep), label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss')
    plt.savefig(f'{plot_path}/loss_curve.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(accuracy_ep), label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Classification accuracy')
    plt.savefig(f'{plot_path}/accuracy.png')

    ############################################

    if (e + 1) % 100 == 0:
        torch.save(modelDiscriminator.state_dict(), f"{root}/model_e{e + 1}.pth")
