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

from model_UNetv7 import UNet_3Plus_for_S, UNet_3Plus_for_X, STN, Discriminator
from dataloader_3Dpre import LineDataset_S_is_added

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=500, help='number of training epochs')
parser.add_argument("--lrUNet", type=float, default=1e-4, help="UNet_3Plus learning rate")
parser.add_argument("--lrSTN", type=float, default=1e-4, help="STN learning rate")
parser.add_argument("--lrDis", type=float, default=1e-4, help="Discriminator learning rate")

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay of optimizer")
parser.add_argument("--patch_size", type=int, default=256, help="training patch size")
parser.add_argument("--batch_size", type=int, default=4, help="training batch size")

parser.add_argument("--in_dir", type=str,
                    default='./dataset/230321_6-2/6-2_20170915_LSM_zStackFish_vol20-130_crop1000-1800.tif',
                    help="dataset path")
parser.add_argument("--out_dir", default='./230719_08_UNetv7-3_1_1e-2_10_100', type=str)
parser.add_argument("--synthetic_S_dir", type=str, default='./dataset/synthetic_S/synthetic_S_110vol.tif')

parser.add_argument("--a", type=float, default=1, help="weight for loss2 (UNet_S, TV of S should be large)")
parser.add_argument("--b", type=float, default=1e-2, help="weight for loss3 (UNet_S, Deviation from 1 loss of mask S)")
parser.add_argument("--c", type=float, default=10, help="weight for loss4 (UNet_X, Generator loss for X)")
parser.add_argument("--d", type=float, default=100, help="weight for loss5 (UNet_S, Generator loss for S)")


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
train_data = LineDataset_S_is_added(img_path=opt.in_dir, synthetic_S_path= opt.synthetic_S_dir, patch_size=opt.patch_size)
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)

# Sample image for model definition and valid image for validation
sample_img = next(iter(train_loader))
valid_img = io.imread('./dataset/6-2_valid_sample_256.tif')
valid_img = torch.Tensor(valid_img.astype('float32'))

# Define models
modelUNet_for_S = UNet_3Plus_for_S(in_channels=1, out_channels=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
modelUNet_for_X = UNet_3Plus_for_X(in_channels=1, out_channels=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
# modelSTN = STN(data_shape=sample_img.shape, device=device)
modelDiscriminator_for_S = Discriminator(input_dim=1, n_layer=3, dim=16, activ='relu', pad='zero', num_scales=2)
modelDiscriminator_for_X = Discriminator(input_dim=1, n_layer=3, dim=16, activ='relu', pad='zero', num_scales=2)

# If you use MULTIPLE GPUs, use 'DataParallel'
modelUNet_for_S = nn.DataParallel(modelUNet_for_S)
modelUNet_for_X = nn.DataParallel(modelUNet_for_X)
# modelSTN = nn.DataParallel(modelSTN)
modelDiscriminator_for_S = nn.DataParallel(modelDiscriminator_for_S)
modelDiscriminator_for_X = nn.DataParallel(modelDiscriminator_for_X)

# to device
modelUNet_for_S = modelUNet_for_S.to(device)
modelUNet_for_X = modelUNet_for_X.to(device)
# modelSTN = modelSTN.to(device)
modelDiscriminator_for_S = modelDiscriminator_for_S.to(device)
modelDiscriminator_for_X = modelDiscriminator_for_X.to(device)

# Loss function for L1 loss (reconstruction loss)
criterion = nn.MSELoss().to(device)
criterion_SSE = nn.MSELoss(reduction='sum').to(device)

# Total variation
tv = TotalVariation().to(device)

# optimizers
optimUNet_for_S = torch.optim.Adam(modelUNet_for_S.parameters(), lr=opt.lrUNet, weight_decay=opt.weight_decay)
optimUNet_for_X = torch.optim.Adam(modelUNet_for_X.parameters(), lr=opt.lrUNet, weight_decay=opt.weight_decay)
# optimSTN = torch.optim.Adam(modelSTN.parameters(), lr=opt.lrSTN, weight_decay=opt.weight_decay)
optimDiscriminator_for_S = torch.optim.Adam(modelDiscriminator_for_S.parameters(), lr=opt.lrDis, weight_decay=opt.weight_decay)
optimDiscriminator_for_X = torch.optim.Adam(modelDiscriminator_for_X.parameters(), lr=opt.lrDis, weight_decay=opt.weight_decay)

# schedulers
schedulerUNet_for_S = torch.optim.lr_scheduler.StepLR(optimUNet_for_S, step_size=50, gamma=0.5)
schedulerUNet_for_X = torch.optim.lr_scheduler.StepLR(optimUNet_for_X, step_size=50, gamma=0.5)
# schedulerSTN = torch.optim.lr_scheduler.StepLR(optimSTN, step_size=20, gamma=0.5)
schedulerDiscriminator_for_S = torch.optim.lr_scheduler.StepLR(optimDiscriminator_for_S, step_size=50, gamma=0.5)
schedulerDiscriminator_for_X = torch.optim.lr_scheduler.StepLR(optimDiscriminator_for_X, step_size=50, gamma=0.5)

# To draw loss graph
train_loss_ep = []
train_loss1_ep = []
train_loss2_ep = []
train_loss3_ep = []
train_loss4_ep = []
train_loss5_ep = []
train_lossDisX_ep = []
train_lossDisS_ep = []
epoch_ep = []

### train
for e in range(opt.epoch):
    epoch_ep.append(e + 1)

    modelUNet_for_X.train()
    modelUNet_for_S.train()
    # modelSTN.train()
    modelDiscriminator_for_S.train()
    modelDiscriminator_for_X.train()

    total_train_loss = 0
    total_loss1 = []
    total_loss2 = []
    total_loss3 = []
    total_loss4 = []
    total_loss5 = []
    total_lossDisX = []
    total_lossDisS = []

    for idx, datas in enumerate(train_loader):
        # Dataloader = (img, synthetic S)
        img, synthetic_S = datas

        # (b, h, w) --> (b, c, h, w) in this case, (b, h, w) --> (b, 1, h, w)
        img = img.unsqueeze(1)
        synthetic_S = synthetic_S.unsqueeze(1)

        # Normalize using 99% percentile value (0~65525 -> 0~1.6)
        high = torch.quantile(img, 0.99)
        img = img / high

        # Normalize S using max value (0~65525 -> 0~1)
        high_s = torch.max(synthetic_S)
        low_s = torch.min(synthetic_S)
        synthetic_S = (synthetic_S - low_s) / (high_s - low_s)

        img = img.to(device)
        synthetic_S = synthetic_S.to(device)

        # 1. Y'' --[Enc1]-> 1D vector S --[extend]-> S'
        mask_1D, mask = modelUNet_for_S(img.clone().detach())

        # 2. Y'' --[Enc2]-> 2D feature X --[Dec2]-> X'
        output = modelUNet_for_X(img.clone().detach())

        # 3. {X, rotated X} --[Discriminator_for_X]-> 0/1
        dis_outputX = modelDiscriminator_for_X(output.clone().detach())
        dis_outputX_rotated = modelDiscriminator_for_X(torch.rot90(output.clone().detach(), k=1, dims=[2,3]))

        # 4. {S, synthetic S} --[Discriminator_for_S]-> 0/1
        dis_outputS = modelDiscriminator_for_S(mask.clone().detach())
        dis_outputS_synthetic = modelDiscriminator_for_S(synthetic_S.clone().detach())

        # loss1: {input} and {output * mask} should be similar. --> UNet, STN 학습에 사용
        loss1 = criterion(img, mask * output)
        # loss2: mask_1D 의 Total Variation 커야 함 --> UNet 학습에 사용
        loss2 = -opt.a * tv(mask_1D)
        # loss3: Deviation from 1 loss for mask S (n*100 % percentile 값에만 적용)
        n = 0.95
        top_n_value = torch.quantile(img.clone().detach(), n)
        top_n_mask = torch.where(img.clone().detach() > top_n_value, mask, 0)
        top_n_gt = torch.where(img.clone().detach() > top_n_value, 1, 0).to(torch.float32)
        loss3 = opt.b * criterion_SSE(top_n_gt, top_n_mask)
        # loss4: GAN loss for UNet_for_X
        loss4 = opt.c * torch.mean((dis_outputX - 1)**2)
        # GAN loss for Discriminator X
        lossDisX = torch.mean((dis_outputX_rotated - 1)**2) + torch.mean((dis_outputX - 0)**2)
        # loss5: GAN loss for UNet_for_S
        loss5 = opt.d * torch.mean((dis_outputS - 1)**2)
        # GAN loss for Discriminator S
        lossDisS = torch.mean((dis_outputS_synthetic - 1)**2) + torch.mean((dis_outputS - 0)**2)

        # Now, we use only loss1 to update three networks.
        loss_UNet = loss1 + loss2 + loss3 + loss4 + loss5
        loss_Discriminator_for_X = lossDisX
        loss_Discriminator_for_S = lossDisS

        # STN_loss = loss1

        # to print loss values
        total_loss1.append(loss1.item())
        total_loss2.append(loss2.item())
        total_loss3.append(loss3.item())
        total_loss4.append(loss4.item())
        total_loss5.append(loss5.item())
        total_lossDisX.append(lossDisX.item())
        total_lossDisS.append(lossDisS.item())

        optimUNet_for_X.zero_grad()
        optimUNet_for_S.zero_grad()
        loss_UNet.backward(retain_graph=True)
        optimUNet_for_X.step()
        optimUNet_for_S.step()

        optimDiscriminator_for_X.zero_grad()
        loss_Discriminator_for_X.backward(retain_graph=True)
        optimDiscriminator_for_X.step()

        optimDiscriminator_for_S.zero_grad()
        loss_Discriminator_for_S.backward(retain_graph=True)
        optimDiscriminator_for_S.step()


    schedulerUNet_for_S.step()
    schedulerUNet_for_X.step()
    # schedulerSTN.step()
    schedulerDiscriminator_for_X.step()
    schedulerDiscriminator_for_S.step()


    total_loss1 = sum(total_loss1) / len(total_loss1)
    total_loss2 = sum(total_loss2) / len(total_loss2)
    total_loss3 = sum(total_loss3) / len(total_loss3)
    total_loss4 = sum(total_loss4) / len(total_loss4)
    total_loss5 = sum(total_loss5) / len(total_loss5)
    total_lossDisX = sum(total_lossDisX) / len(total_lossDisX)
    total_lossDisS = sum(total_lossDisS) / len(total_lossDisS)

    total_train_loss = total_loss1 + total_loss2 + total_loss3 + total_loss4 + total_loss5

    print(f'Epoch: {e + 1} / {opt.epoch}, Total_loss: {total_train_loss:.3f}, Loss1: {total_loss1:.3f}, '
          f'Loss2: {total_loss2:.3f}, Loss3: {total_loss3:.3f}, Loss4: {total_loss4:.3f}, Loss5: {total_loss5:.3f}')
    print(f'Loss_for_DisX: {total_lossDisX:.3f}, Loss_for_DisS: {total_lossDisS:.3f}')

    train_loss_ep.append(total_train_loss)
    train_loss1_ep.append(total_loss1)
    train_loss2_ep.append(total_loss2)
    train_loss3_ep.append(total_loss3)
    train_loss4_ep.append(total_loss4)
    train_loss5_ep.append(total_loss5)
    train_lossDisX_ep.append(total_lossDisX)
    train_lossDisS_ep.append(total_lossDisS)

    ### validation
    if e % 10 == 0:
        modelUNet_for_S.eval()
        modelUNet_for_X.eval()
        # modelSTN.eval()

        with torch.no_grad():
            img = valid_img.unsqueeze(0).unsqueeze(0)

            # Normalize using 99% percentile value (0~65525 -> 0~1.6)
            high = torch.quantile(img, 0.99)
            img = img / high

            img = img.to(device)

            # 1. Y'' --[Enc1]-> 1D vector S --[extend]-> S'
            mask_1D, mask = modelUNet_for_S(img.clone().detach())

            # 2. Y'' --[Enc2]-> 2D feature X --[Dec2]-> X'
            output = modelUNet_for_X(img.clone().detach())

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
            mask_vis = mask[0].squeeze().detach().cpu().numpy()
            io.imsave(f'{temp_path}/S_{e}.tif', mask_vis)

            # mask_out_vis = mask_out[0].squeeze().detach().cpu().numpy()
            # io.imsave(f'{temp_path}/S1_{e}.tif', mask_out_vis)
            #
            # mean_proj_extend_vis = mean_proj_extend[0].squeeze().detach().cpu().numpy()
            # io.imsave(f'{temp_path}/S3_{e}.tif', mean_proj_extend_vis)
            #
            # mask_inv_STN_vis = mask_inv_STN[0].squeeze().detach().cpu().numpy()
            # io.imsave(f'{temp_path}/S4_{e}.tif', mask_inv_STN_vis)

            output_vis = output[0].squeeze().detach().cpu()
            output_vis = (output_vis * high).numpy().astype('uint16')  # denormalize using only max
            io.imsave(f'{temp_path}/output_{e}.tif', output_vis)
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

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_loss3_ep), label='b * loss3')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss3, b: {str(opt.b)}')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss3_curve.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_loss4_ep), label='c * loss4')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss4, c:{str(opt.c)}')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss4_curve.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_loss5_ep), label='loss5')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss5')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss5_curve.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_lossDisS_ep), label='loss_DisS')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss of the discriminator for S')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss_DisS_curve.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_lossDisX_ep), label='loss_DisX')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss of the discrminator for X')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss_DisX_curve.png')

    ############################################

    if e % 100 == 0:
        torch.save(modelUNet_for_S.state_dict(), f"{model_path}/UNet_for_S_{e}_{total_train_loss:.4f}.pth")
        torch.save(modelUNet_for_X.state_dict(), f"{model_path}/UNet_for_X_{e}_{total_train_loss:.4f}.pth")
        # torch.save(modelSTN.state_dict(), f"{model_path}/STN_{e}_{total_loss2:.4f}.pth")
        torch.save(modelDiscriminator_for_S.state_dict(), f"{model_path}/Discriminator_for_S_{e}_{total_lossDisS:.4f}.pth")
        torch.save(modelDiscriminator_for_X.state_dict(), f"{model_path}/Discriminator_for_X_{e}_{total_lossDisX:.4f}.pth")
