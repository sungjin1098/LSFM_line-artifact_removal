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
from model_UNetv8 import UNet_3Plus_for_S, UNet_3Plus_for_X_train, STN, Discriminator, MLP, Generator


parser = argparse.ArgumentParser()
parser.add_argument("--valid_save_period", type=int, default=10, help='save the valid results for every N epochs')
parser.add_argument("--model_save_period", type=int, default=100, help='save the model pth file for every N epochs')
parser.add_argument("--valid_batch_size", type=int, default=10, help="validation image saving batch size")


parser.add_argument("--in_dir", type=str, default='./dataset/230817_LSM_simulation/230817_input_Y_110vol.tif', help="dataset path")
parser.add_argument("--synthetic_S_dir", type=str, default='./dataset/230817_LSM_simulation/230817_synthetic_S_110vol.tif')
parser.add_argument("--gt_dir", type=str, default='./dataset/230817_LSM_simulation/230817_gt_110vol.tif')
parser.add_argument("--valid_idx_path", type=str, default='./dataset/230817_LSM_simulation/index_list.txt')
parser.add_argument("--out_dir", type=str, default='./230905_03', help='hyperparameters are saved at the end automatically')
parser.add_argument("--lr", type=float, default=0.0001, help='hyperparameters are saved at the end automatically')
parser.add_argument("--lr1", type=float, default=0.001, help='hyperparameters are saved at the end automatically')
parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay of optimizer")


opt = parser.parse_args()

# save path
root = opt.out_dir
model_path = root + '/saved_models'
plot_path = root + '/loss_curve'
image_path = root + '/output_images'
temp_path = root + '/temp'
os.makedirs(model_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)
os.makedirs(temp_path, exist_ok=True)

# Prepare for use of CUDA

# 코드 실행결과의 동일성을 위해 무작위 시드를 설정합니다
manualSeed = 999
#manualSeed = random.randint(1, 10000) # 만일 새로운 결과를 원한다면 주석을 없애면 됩니다
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

workers = 2

batch_size = 10

image_size = 64

nc = 1

nz = 100

ngf = 64

ndf = 64

num_epochs = 6000

# 옵티마이저의 학습률
lr = opt.lr

#dis
lr1 = opt.lr1
lr2 = 0.0001

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
dataset = LineDataset_S_is_added_Multiple_updates(img_path=opt.in_dir, synthetic_S_path=opt.synthetic_S_dir, gt_path=opt.in_dir, patch_size=image_size, num_update=1)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_data = LineDataset_valid(index_path=opt.valid_idx_path, valid_path=opt.in_dir, synthetic_S_path=opt.synthetic_S_dir, gt_path=opt.gt_dir, patch_size=image_size)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True,drop_last=True)

# 생성자를 만듭니다
netG = Generator().to(device)
netG_S = Generator().to(device)

# 필요한 경우 multi-GPU를 설정 해주세요
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# 모든 가중치의 평균을 0( ``mean=0`` ), 분산을 0.02( ``stdev=0.02`` )로 초기화하기 위해
# ``weight_init`` 함수를 적용시킵니다
netG.apply(weights_init)
netG_S.apply(weights_init)

# 모델의 구조를 출력합니다


# 구분자 코드
# Loss function for L1 loss (reconstruction loss)
criterion = nn.BCELoss()


# Total variation

# optimizers

# 구분자를 만듭니다
netD = Discriminator().to(device)
netD_S = Discriminator().to(device)

# 필요한 경우 multi-GPU를 설정 해주세요
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 모든 가중치의 평균을 0( ``mean=0`` ), 분산을 0.02( ``stdev=0.02`` )로 초기화하기 위해
# ``weight_init`` 함수를 적용시킵니다
netD.apply(weights_init)
netD_S.apply(weights_init)

# 모델의 구조를 출력합니다

# ``BCELoss`` 함수의 인스턴스를 초기화합니다
criterion = nn.BCELoss()

# 생성자의 학습상태를 확인할 잠재 공간 벡터를 생성합니다
fixed_noise = torch.randn(10, nz, 1, 1, device=device)
fixed_noise_S = torch.randn(10, nz, 1, 1, device=device)

# 학습에 사용되는 참/거짓의 라벨을 정합니다
real_label = 1.
fake_label = 0.

# G와 D에서 사용할 Adam옵티마이저를 생성합니다
optimizerD = optim.Adam(netD.parameters(), lr=lr1, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD_S = optim.Adam(netD_S.parameters(), lr=lr2, betas=(beta1, 0.999))
optimizerG_S = optim.Adam(netG_S.parameters(), lr=lr, betas=(beta1, 0.999))


modelUNet_for_S = UNet_3Plus_for_S(in_channels=1, out_channels=1, feature_scale=2, is_deconv=True, is_batchnorm=True)
modelUNet_for_X = UNet_3Plus_for_X_train(in_channels=1, out_channels=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
modelUNet_for_S = nn.DataParallel(modelUNet_for_S)
modelUNet_for_X = nn.DataParallel(modelUNet_for_X)
modelUNet_for_S = modelUNet_for_S.to(device)
modelUNet_for_X = modelUNet_for_X.to(device)
optimUNet_for_S = torch.optim.Adam(modelUNet_for_S.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
optimUNet_for_X = torch.optim.Adam(modelUNet_for_X.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

# 학습상태를 체크하기 위해 손실값들을 저장합니다
img_list = []
G_losses = []
D_losses = []
G_S_losses = []
D_S_losses = []
iters = 0
print("Starting Training Loop...")
### train
for epoch in range(num_epochs):
    # 한 에폭 내에서 배치 반복
    for i, datas in enumerate(dataloader, 0):
        
        data, gt, S = datas
        ############################
        # (1) D 신경망을 업데이트 합니다: log(D(x)) + log(1 - D(G(z)))를 최대화 합니다
        ###########################
        ## 진짜 데이터들로 학습을 합니다
        netD.zero_grad()

        # 배치들의 사이즈나 사용할 디바이스에 맞게 조정합니다
        data= data.unsqueeze(1)
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label,
                           dtype=torch.float, device=device)
        # 진짜 데이터들로 이루어진 배치를 D에 통과시킵니다
        output = netD(real_cpu).view(-1)

        # 손실값을 구합니다
        errD_real = criterion(output, label)*10
        # 역전파의 과정에서 변화도를 계산합니다
        errD_real.backward()
        D_x = output.mean().item()

        ## 가짜 데이터들로 학습을 합니다
        # 생성자에 사용할 잠재공간 벡터를 생성합니다
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # G를 이용해 가짜 이미지를 생성합니다
        fake = netG(noise)
        fake = modelUNet_for_X(fake)
        label.fill_(fake_label)
        # D를 이용해 데이터의 진위를 판별합니다
        output = netD(fake.detach()).view(-1)
        # D의 손실값을 계산합니다
        errD_fake = criterion(output, label)*10
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
        optimUNet_for_X.zero_grad()

        label.fill_(real_label)  # 생성자의 손실값을 구하기 위해 진짜 라벨을 이용할 겁니다
        # 우리는 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과시킵니다.
        # 이때 G는 업데이트되지 않았지만, D가 업데이트 되었기 때문에 앞선 손실값가 다른 값이 나오게 됩니다
        output = netD(fake).view(-1)
        # G의 손실값을 구합니다
        errG = criterion(output, label)
        # G의 변화도를 계산합니다
        errG.backward()
        D_G_z2 = output.mean().item()
        # G를 업데이트 합니다
        optimizerG.step()
        optimUNet_for_X.step()

        
        
        
        ########################################################################################################################
        netD_S.zero_grad()
        high_s = torch.max(S)
        S = S / high_s
        # 배치들의 사이즈나 사용할 디바이스에 맞게 조정합니다
        S = S.unsqueeze(1)
        real_cpu1 = S.to(device)
        b_size1 = real_cpu1.size(0)
        label = torch.full((b_size1,), real_label,
                           dtype=torch.float, device=device)
        # 진짜 데이터들로 이루어진 배치를 D에 통과시킵니다
        output = netD_S(real_cpu1).view(-1)

        # 손실값을 구합니다
        errD_S_real = criterion(output, label)
        # 역전파의 과정에서 변화도를 계산합니다
        errD_S_real.backward()
        D_S_x = output.mean().item()

        ## 가짜 데이터들로 학습을 합니다
        # 생성자에 사용할 잠재공간 벡터를 생성합니다
        noise = torch.randn(b_size1, nz, 1, 1, device=device)
        # G를 이용해 가짜 이미지를 생성합니다
        fake = netG_S(noise)
        _, fake = modelUNet_for_S(fake)

        label.fill_(fake_label)
        # D를 이용해 데이터의 진위를 판별합니다
        output = netD_S(fake.detach()).view(-1)
        # D의 손실값을 계산합니다
        errD_S_fake = criterion(output, label)
        # 역전파를 통해 변화도를 계산합니다. 이때 앞서 구한 변화도에 더합니다(accumulate)
        errD_S_fake.backward()
        D_G_S_z1 = output.mean().item()
        # 가짜 이미지와 진짜 이미지 모두에서 구한 손실값들을 더합니다
        # 이때 errD는 역전파에서 사용되지 않고, 이후 학습 상태를 리포팅(reporting)할 때 사용합니다
        errD_S = errD_S_real + errD_S_fake
        # D를 업데이트 합니다
        optimizerD_S.step()
        
        ###########################
        netG_S.zero_grad()
        optimUNet_for_S.zero_grad()

        label.fill_(real_label)  # 생성자의 손실값을 구하기 위해 진짜 라벨을 이용할 겁니다
        # 우리는 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과시킵니다.
        # 이때 G는 업데이트되지 않았지만, D가 업데이트 되었기 때문에 앞선 손실값가 다른 값이 나오게 됩니다
        output = netD_S(fake).view(-1)
        # G의 손실값을 구합니다
        errG_S = criterion(output, label)
        # G의 변화도를 계산합니다
        errG_S.backward()

        D_G_S_z2 = output.mean().item()
        # G를 업데이트 합니다
        optimizerG_S.step()   
        optimUNet_for_S.step()

        
        
        
        
    

        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'  % (epoch, num_epochs, i, len(dataloader),
                     errD_S.item(), errG.item(), D_S_x, D_G_S_z1, D_G_S_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        G_S_losses.append(errG_S.item())
        D_S_losses.append(errD_S.item())


    if (epoch + 1) % opt.valid_save_period == 0:
        valid_save_path = f'{image_path}/epoch{epoch+ 1}'
        valid_array_save_path = f'{temp_path}/epoch{epoch+1}'
        os.makedirs(valid_save_path, exist_ok=True)
        os.makedirs(valid_array_save_path, exist_ok=True)
    with torch.no_grad():
        for _, datas in enumerate(valid_loader):
            # Dataloader = (img, synthetic S)
            _, synthetic_S, gt, idx = datas

            # Normalize using 99% percentile value (0~65525 -> 0~1.6)
            high_gt = torch.quantile(gt, 0.99)
            gt = gt / high_gt
            gt= gt.unsqueeze(1).to(device)

            synthetic_S=synthetic_S.unsqueeze(1).to(device)
            output = netG(fixed_noise)
            output = modelUNet_for_X(output)

            output_S = netG_S(fixed_noise_S)
            _, output_S = modelUNet_for_S(output_S)
            high_s = torch.max(synthetic_S)
            synthetic_S = synthetic_S / high_s


            ### loss for check if 'GT' and our 'output X' is same. ###

            ##########################################################

            if (epoch + 1) % opt.valid_save_period == 0:
                ############# PNG Image Array Save #############
                #img_grid = make_grid(img2, padding=2, pad_value=1)
                #img_grid1 = make_grid(imidxg, padding=2, pad_value=1)

                gt_grid = make_grid(gt, padding=2, pad_value=1)
                output_temp = output * high_gt
                output_temp_flat = torch.flatten(output_temp, start_dim=1, end_dim=-1)
                output_temp /= torch.quantile(output_temp_flat, 0.99, dim=1)[..., None, None, None]
                output_grid = make_grid(output_temp, padding=2, pad_value=1)

                save_grid = torch.cat([ gt_grid, output_grid], dim=1)
                save_image(save_grid, f'{valid_array_save_path}/e{epoch + 1}_{idx[0]}-{idx[-1]}.png')
                
                
                synthetic_S_grid = make_grid(synthetic_S, padding=2, pad_value=1)
                mask_grid = make_grid(output_S, padding=2, pad_value=1)
                save_grid = torch.cat([ synthetic_S_grid, mask_grid], dim=1)
                save_image(save_grid, f'{valid_array_save_path}/e{epoch + 1}_{idx[0]}-{idx[-1]}_S.png')
                ################################################

                for i in range(opt.valid_batch_size):
                    valid_idx = idx[i]


                    output_vis = output[i].squeeze().detach().cpu()
                    output_vis = (output_vis * high_gt).numpy().astype('uint16')  # denormalize using only max
                    io.imsave(f'{valid_save_path}/e{epoch+1}_{valid_idx}_outputX.tif', output_vis)
                    
                    mask_vis = output_S[i].squeeze().detach().cpu().numpy()
                    io.imsave(f'{valid_save_path}/e{epoch+1}_{valid_idx}_maskS.tif', mask_vis)
                    ######################################

        ### loss for check if 'GT' and our 'output X' is same. ###
        ##########################################################

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{plot_path}/Loss_X.png')

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_S_losses,label="G")
    plt.plot(D_S_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{plot_path}/Loss_S.png')


    ############################################

    if (epoch + 1) % opt.model_save_period == 0:
        torch.save(netD.state_dict(), f"{model_path}/UNet_for_X_D_{epoch+1}.pth")
        torch.save(netG.state_dict(), f"{model_path}/UNet_for_X_G_{epoch+1}.pth")
    if (epoch + 1) % opt.model_save_period == 0:
        torch.save(netD_S.state_dict(), f"{model_path}/UNet_for_S_D_{epoch+1}.pth")
        torch.save(netG_S.state_dict(), f"{model_path}/UNet_for_S_G_{epoch+1}.pth")