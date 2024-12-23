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
from tqdm import tqdm
from torch.fft import fftn, ifftn, fftshift, ifftshift
from torch.utils.data import DataLoader
from torchmetrics import TotalVariation

from model_UNetv5 import UNet_3Plus_for_S, UNet_3Plus_for_X, STN
from dataloader_3Dpre import LineDataset

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=500, help='number of training epochs')
parser.add_argument("--lrUNet", type=float, default=1e-4, help="UNet_3Plus learning rate")
parser.add_argument("--lrSTN", type=float, default=1e-4, help="STN learning rate")

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay of optimizer")
parser.add_argument("--patch_size", type=int, default=256, help="training patch size")
parser.add_argument("--batch_size", type=int, default=4, help="training batch size")

parser.add_argument("--in_dir", type=str,
                    default='./dataset/230321_6-2/6-2_20170915_LSM_zStackFish_vol20-130_crop1000-1800.tif',
                    help="dataset path")
parser.add_argument("--out_dir", default='./230706_01_UNetv5-1_1e-5', type=str)

parser.add_argument("--a", type=float, default=1e-5, help="weight for loss2 (NN, TV of S'' should be large)")
# parser.add_argument("--b", type=float, default=1e-1, help="weight for loss3 (NN, S' = S''')")
# parser.add_argument("--c", type=float, default=1e-1, help="wieght ofr loss4 (NN, S = S'''')")


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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataloader
train_data = LineDataset(opt.in_dir, patch_size=opt.patch_size)
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)

# Sample image for model definition and valid image for validation
sample_img = next(iter(train_loader))
valid_img = io.imread('./dataset/6-2_valid_sample_256.tif')
valid_img = torch.Tensor(valid_img.astype('float32'))

# Define models
modelUNet_for_S = UNet_3Plus_for_S(in_channels=1, out_channels=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
modelUNet_for_X = UNet_3Plus_for_X(in_channels=1, out_channels=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
modelSTN = STN(data_shape=sample_img.shape, device=device)

# If you use MULTIPLE GPUs, use 'DataParallel'
modelUNet_for_S = nn.DataParallel(modelUNet_for_S)
modelUNet_for_X = nn.DataParallel(modelUNet_for_X)
modelSTN = nn.DataParallel(modelSTN)

# to device
modelUNet_for_S = modelUNet_for_S.to(device)
modelUNet_for_X = modelUNet_for_X.to(device)
modelSTN = modelSTN.to(device)

# Loss function for L1 loss (reconstruction loss)
criterion = nn.MSELoss().to(device)

# Total variation
tv = TotalVariation().to(device)

# optimizers
optimUNet_for_S = torch.optim.Adam(modelUNet_for_S.parameters(), lr=opt.lrUNet, weight_decay=opt.weight_decay)
optimUNet_for_X = torch.optim.Adam(modelUNet_for_X.parameters(), lr=opt.lrUNet, weight_decay=opt.weight_decay)
optimSTN = torch.optim.Adam(modelSTN.parameters(), lr=opt.lrSTN, weight_decay=opt.weight_decay)

# schedulers
schedulerUNet_for_S = torch.optim.lr_scheduler.StepLR(optimUNet_for_S, step_size=20, gamma=0.5)
schedulerUNet_for_X = torch.optim.lr_scheduler.StepLR(optimUNet_for_X, step_size=20, gamma=0.5)
schedulerSTN = torch.optim.lr_scheduler.StepLR(optimSTN, step_size=20, gamma=0.5)

# To draw loss graph
train_loss_ep = []
train_loss1_ep = []
train_loss2_ep = []
train_loss3_ep = []
train_loss3_pure_ep = []
train_loss4_ep = []
epoch_ep = []

### train
for e in range(opt.epoch):
    epoch_ep.append(e + 1)

    modelUNet_for_X.train()
    modelUNet_for_S.train()
    modelSTN.train()

    total_train_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0
    total_loss3_pure = 0
    total_loss4 = 0

    for idx, img in enumerate(train_loader):
        # (b, h, w) --> (b, c, h, w) in this case, (b, h, w) --> (b, 1, h, w)
        img = img.unsqueeze(1)

        # Normalize using 99% percentile value (0~65525 -> 0~1.6)
        high = torch.quantile(img, 0.99)
        img = img / high

        img = img.to(device)

        # 1. Y' --[STN]-> Y''
        # img_afterSTN, theta = modelSTN(img.clone().detach())
        img_afterSTN = img

        # 2. Y'' --[Enc1]-> 1D vector S --[extend]-> S'
        mask_1D = modelUNet_for_S(img_afterSTN.clone().detach())
        mask = mask_1D.repeat(1, 1, opt.patch_size, 1)

        # 3. Y'' --[Enc2]-> 2D feature X --[Dec2]-> X'
        output = modelUNet_for_X(img_afterSTN.clone().detach())

        output_inv_STN = output
        mask_inv_STN = mask

        # 4. S' --[Inverse STN]-> S     and     X' --[Inverse STN]-> X
        # theta = theta[0]   # If we use 3 GPUs, theta becomes (1, 3) even if its real size is (1)
        # theta_m = torch.zeros((mask.shape[0], 2, 3), dtype=torch.float, device=device)  # theta size: (b, 2, 3)
        # theta_m[:, 0, 0] = torch.cos(theta)
        # theta_m[:, 1, 1] = torch.cos(theta)
        # theta_m[:, 0, 1] = torch.sin(theta)
        # theta_m[:, 1, 0] = -torch.sin(theta)
        # grid = F.affine_grid(theta_m, mask.size())
        # # The size of 'mask_inv_STN' and 'output_inv_STN': (b, 1, w, h) except last batch
        # mask_inv_STN = F.grid_sample(mask, grid).view(-1, 1, opt.patch_size, opt.patch_size)
        # output_inv_STN = F.grid_sample(output, grid).view(-1, 1, opt.patch_size, opt.patch_size)

        # # mean_projection of S to obtain S'' and S'''
        # mean_proj = torch.mean(mask_out, dim=2)   # (b, 1, w)
        # mean_proj_extend = mean_proj.unsqueeze(2).repeat(1, 1, opt.patch_size, 1)   # (b, 1, h, w)


        # loss1: {input} and {output * mask} should be similar. --> UNet, STN 학습에 사용
        loss1 = criterion(img, mask_inv_STN * output_inv_STN)
        # loss2: mask_1D 의 Total Variation 커야 함 --> UNet 학습에 사용
        loss2 = -opt.a * tv(mask_1D)
        # # loss3: STN 거친 S' = S''' 이어야 함 --> UNet, STN 학습에 사용. STN update에는 weight 따로 붙이지 않음
        # loss3 = opt.b * criterion(mask_out, mean_proj_extend)
        # loss3_pure = criterion(mask_out, mean_proj_extend)
        # # loss4: loss1이 {input}과 {mask}를 곱해서 쓰기 때문에 생기는 loss. S와 S''''는 비슷해야 함 --> UNet 학습에 사용
        # loss4 = opt.c * criterion(mask, mask_inv_STN)

        # Now, we use only loss1 to update three networks.
        loss = loss1 + loss2
        # UNet_for_S_loss = loss1
        # UNet_for_X_loss = loss1
        # STN_loss = loss1

        # to print loss values
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        # total_loss3 += loss3.item()
        # total_loss3_pure += loss3_pure.item()
        # total_loss4 += loss4.item()

        optimUNet_for_S.zero_grad()
        optimUNet_for_X.zero_grad()
        optimSTN.zero_grad()
        loss.backward()
        optimUNet_for_S.step()
        optimUNet_for_X.step()
        optimSTN.step()

    schedulerUNet_for_S.step()
    schedulerUNet_for_X.step()
    schedulerSTN.step()

    total_train_loss = total_loss1

    print(f'Epoch: {e + 1} / {opt.epoch}, Loss: {total_train_loss:.3f}, Loss1: {total_loss1:.3f}, Loss2: {total_loss2:.3f}')

    train_loss_ep.append(total_train_loss)
    train_loss1_ep.append(total_loss1)
    train_loss2_ep.append(total_loss2)
    # train_loss3_ep.append(total_loss3)
    # train_loss3_pure_ep.append(total_loss3_pure)
    # train_loss4_ep.append(total_loss4)

    ### validation
    if e % 10 == 0:
        modelUNet_for_S.eval()
        modelUNet_for_X.eval()
        modelSTN.eval()

        with torch.no_grad():
            img = valid_img.unsqueeze(0).unsqueeze(0)

            # Normalize using 99% percentile value (0~65525 -> 0~1.6)
            high = torch.quantile(img, 0.99)
            img = img / high

            img = img.to(device)

            # 1. Y' --[STN]-> Y''
            # img_afterSTN, theta = modelSTN(img.clone().detach())
            img_afterSTN = img

            # 2. Y'' --[Enc1]-> 1D vector S --[extend]-> S'
            mask_1D = modelUNet_for_S(img_afterSTN.clone().detach())
            mask = mask_1D.repeat(1, 1, opt.patch_size, 1)

            # 3. Y'' --[Enc2]-> 2D feature X --[Dec2]-> X'
            output = modelUNet_for_X(img_afterSTN.clone().detach())

            output_inv_STN = output
            mask_inv_STN = mask

            # # 4. S' --[Inverse STN]-> S     and     X' --[Inverse STN]-> X
            # theta = theta[0]  # If we use 3 GPUs, theta becomes (1, 3) even if its real size is (1)
            # theta_m = torch.zeros((mask.shape[0], 2, 3), dtype=torch.float, device=device)  # theta size: (b, 2, 3)
            # theta_m[:, 0, 0] = torch.cos(theta)
            # theta_m[:, 1, 1] = torch.cos(theta)
            # theta_m[:, 0, 1] = torch.sin(theta)
            # theta_m[:, 1, 0] = -torch.sin(theta)
            # grid = F.affine_grid(theta_m, mask.size())
            # # The size of 'mask_inv_STN' and 'output_inv_STN': (b, 1, w, h) except last batch
            # mask_inv_STN = F.grid_sample(mask, grid).view(-1, 1, opt.patch_size, opt.patch_size)
            # output_inv_STN = F.grid_sample(output, grid).view(-1, 1, opt.patch_size, opt.patch_size)

            ### DFT part ##################################################
            # mask_out_fft = fftshift(fftn(mask_out))
            # mask_out_fft_mag = torch.abs(mask_out_fft)  # magnitude
            #
            # # mask_out_fft_mag.shape = (b, c, h ,w) = (64, 1, 512, 512)
            # proj_x, _ = torch.max(mask_out_fft_mag, dim=2)
            # proj_y, _ = torch.max(mask_out_fft_mag, dim=3)
            # proj_x_l1 = torch.norm(proj_x, 1)
            # proj_y_l1 = torch.norm(proj_y, 1)
            ###############################################################

            # # mean_projection of S to obtain S'' and S'''
            # mean_proj = torch.mean(mask_out, dim=2)  # (b, 1, w)
            # mean_proj_extend = mean_proj.unsqueeze(2).repeat(1, 1, opt.patch_size, 1)  # (b, 1, h, w)

            ############# Image Save ######################
            mask_inv_STN_vis = mask_inv_STN[0].squeeze().detach().cpu().numpy()
            io.imsave(f'{temp_path}/S_{e}.tif', mask_inv_STN_vis)

            # mask_out_vis = mask_out[0].squeeze().detach().cpu().numpy()
            # io.imsave(f'{temp_path}/S1_{e}.tif', mask_out_vis)
            #
            # mean_proj_extend_vis = mean_proj_extend[0].squeeze().detach().cpu().numpy()
            # io.imsave(f'{temp_path}/S3_{e}.tif', mean_proj_extend_vis)
            #
            # mask_inv_STN_vis = mask_inv_STN[0].squeeze().detach().cpu().numpy()
            # io.imsave(f'{temp_path}/S4_{e}.tif', mask_inv_STN_vis)

            output_inv_STN_vis = output_inv_STN[0].squeeze().detach().cpu()
            output_inv_STN_vis = (output_inv_STN_vis * high).numpy().astype('uint16')  # denormalize using only max
            io.imsave(f'{temp_path}/output_{e}.tif', output_inv_STN_vis)
            ############################################

            ### DFT part #########################
            # mask_out_fft_mag_vis = mask_out_fft_mag[0].squeeze().detach().cpu().numpy()
            # io.imsave(f'{temp_path}/mask_out_fft_linear_{e}.tif', mask_out_fft_mag_vis)
            #
            # mask_out_fft_mag_log_vis = 20 * np.log(mask_out_fft_mag_vis)
            # io.imsave(f'{temp_path}/mask_out_fft_log_{e}.tif', mask_out_fft_mag_log_vis)
            #
            # plt.figure()
            # ax1 = plt.subplot(1, 2, 1)
            # plt.plot(range(opt.patch_size), 20 * np.log(proj_x[0].squeeze().detach().cpu().numpy()))
            # plt.title('proj_x (log-scale)')
            # ax2 = plt.subplot(1, 2, 2, sharey=ax1)
            # plt.plot(range(opt.patch_size), 20 * np.log(proj_y[0].squeeze().detach().cpu().numpy()))
            # plt.title('proj_y (log-scale)')
            # plt.savefig(f'{plot_path}/xy_proj_{e}.png')
            ############################################

    ############################################
    # Save loss curves
    epoch_ep_n = np.array(epoch_ep)

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_loss_ep), label='total train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total train loss')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/total_loss_curve.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_loss1_ep), label='loss1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss1')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss1_curve.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_loss2_ep), label='a * loss2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss2, a: {str(opt.a)}')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss2_curve.png')
    #
    # plt.clf()
    # plt.plot(epoch_ep_n, np.array(train_loss3_ep), label='b * loss3')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title(f'Loss3 for UNet, b: {str(opt.b)}')
    # plt.legend(loc='upper right')
    # plt.savefig(f'{plot_path}/loss3_curve_forUNet.png')
    #
    # plt.clf()
    # plt.plot(epoch_ep_n, np.array(train_loss3_pure_ep), label='loss3')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title(f'Loss3 for STN')
    # plt.legend(loc='upper right')
    # plt.savefig(f'{plot_path}/loss3_curve_forSTN.png')
    #
    # plt.clf()
    # plt.plot(epoch_ep_n, np.array(train_loss4_ep), label='loss4')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title(f'Loss4, c: {str(opt.c)}')
    # plt.legend(loc='upper right')
    # plt.savefig(f'{plot_path}/loss4_curve.png')

    ############################################

    if e % 100 == 0:
        torch.save(modelUNet_for_S.state_dict(), f"{model_path}/UNet_for_S_{e}_{total_train_loss:.4f}.pth")
        torch.save(modelUNet_for_X.state_dict(), f"{model_path}/UNet_for_X_{e}_{total_train_loss:.4f}.pth")
        torch.save(modelSTN.state_dict(), f"{model_path}/STN_{e}_{total_loss2:.4f}.pth")
