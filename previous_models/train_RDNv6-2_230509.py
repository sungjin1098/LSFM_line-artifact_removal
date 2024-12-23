import os
import torch
import argparse
import warnings
import torchvision
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from skimage import io
from torch import nn
from tqdm import tqdm
from torch.fft import fftn, ifftn, fftshift, ifftshift
from torch.utils.data import DataLoader

from model_RDNv6 import RDN, STN
from dataloader_3D import LineDataset


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='number of training epochs')
parser.add_argument("--lrUNet", type=float, default=1e-4, help="RDN learning rate")
parser.add_argument("--lrSTN", type=float, default=1e-4, help="STN learning rate")

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay of optimizer")
parser.add_argument("--patch_size", type=int, default=512, help="training patch size")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")

parser.add_argument("--in_dir", type=str, default='./dataset/230321_6-2/6-2_20170915_LSM_zStackFish_vol20-130_crop1000-1800.tif', help="dataset path")
parser.add_argument("--out_dir", default='./230511_6-2_11_a1b1c1d1_losschanged', type=str)

parser.add_argument("--a", type=float, default=1e-1, help="weight for loss2 term (L1 norm should be small)")
parser.add_argument("--b", type=float, default=1e-1, help="weight for loss3 term (L1 norm should be large)")
parser.add_argument("--c", type=float, default=1e-1, help="weight for loss4 term (std(diff(mean_proj)) should be large)")
parser.add_argument("--d", type=float, default=1e-1, help="weight for loss5 term (S'=S''')")



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
valid_img = io.imread('./dataset/6-2_valid_sample.tif')
valid_img = torch.Tensor(valid_img.astype('float32'))

# Define models
modelRDN = RDN(num_channels=1, num_features=16, growth_rate=16, num_blocks=4, num_layers=2)
modelSTN = STN(data_shape=sample_img.shape, device=device)

# If you use MULTIPLE GPUs, use 'DataParallel'
modelRDN = nn.DataParallel(modelRDN)
modelSTN = nn.DataParallel(modelSTN)

# to device
modelRDN = modelRDN.to(device)
modelSTN = modelSTN.to(device)

# Loss function for L1 loss (reconstruction loss)
criterion = nn.MSELoss().to(device)

# optimizers
optimRDN = torch.optim.Adam(modelRDN.parameters(), lr=opt.lrUNet, weight_decay=opt.weight_decay)
optimSTN = torch.optim.Adam(modelSTN.parameters(), lr=opt.lrSTN, weight_decay=opt.weight_decay)

# schedulers
schedulerRDN = torch.optim.lr_scheduler.StepLR(optimRDN, step_size=50, gamma=0.5)
schedulerSTN = torch.optim.lr_scheduler.StepLR(optimSTN, step_size=50, gamma=0.5)

# To draw loss graph
train_loss_ep = []
train_loss1_ep = []
train_loss2_ep = []
train_loss3_ep = []
train_loss4_ep = []
train_loss5_ep = []
epoch_ep = []

### train
for e in range(opt.epoch):
    epoch_ep.append(e+1)

    modelRDN.train()
    modelSTN.train()

    total_train_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0
    total_loss4 = 0
    total_loss5 = 0

    for idx, img in enumerate(train_loader):
        img = img.unsqueeze(1)

        # Normalize using 99% percentile value (0~65525 -> 0~1.6)
        high = torch.quantile(img, 0.99)
        img = img / high

        img = img.to(device)

        output, mask = modelRDN(img.clone().detach())
        mask_out = modelSTN(mask.clone().detach())

        mask_out_fft = fftshift(fftn(mask_out))
        mask_out_fft_mag = torch.abs(mask_out_fft)    # magnitude

        # mask_out_fft_mag.shape = (b, c, h ,w) = (64, 1, 512, 512)
        proj_x, _ = torch.max(mask_out_fft_mag, dim=2)
        proj_y, _ = torch.max(mask_out_fft_mag, dim=3)
        proj_x_l1 = torch.norm(proj_x, 1)
        proj_y_l1 = torch.norm(proj_y, 1)

        # the smaller value becomes loss2. projection along x-axis vs y-axis
        if proj_x_l1 < proj_y_l1:
            loss2 = proj_x_l1
            # loss3 = torch.norm(proj_y - proj_x, 1)
            loss3 = torch.sum(proj_y - proj_x)
            mean_proj = torch.mean(mask_out, dim=3)
            mean_proj_extend = torch.tile(mean_proj, (1, 1, opt.patch_size, 1)).permute(0, 1, 3, 2)
            print('if')
            print(mean_proj.detach().cpu().numpy().shape)
            print(mean_proj_extend.detach().cpu().numpy().shape)
        else:
            loss2 = proj_y_l1
            # loss3 = torch.norm(proj_x - proj_y, 1)
            loss3 = torch.sum(proj_x - proj_y)
            mean_proj = torch.mean(mask_out, dim=2)
            mean_proj_extend = mean_proj.unsqueeze(2).repeat(1, 1, opt.patch_size, 1)
            print('else')
            print(mean_proj.detach().cpu().numpy().shape)
            print(mean_proj_extend.detach().cpu().numpy().shape)


        # loss1: {input} and {output * mask} should be similar.
        loss1 = criterion(img, output * mean_proj_extend)
        # loss2: L1 norm of 1D-projected vector should be small.
        loss2 = opt.a * loss2
        # loss3: the difference between two 1D-projected vectors should be large
        loss3 = -opt.b * loss3
        # loss4: std(diff(mean_projection)) should be large
        loss4 = -opt.c * torch.std(torch.diff(mean_proj))
        # loss5: mask after STN (S') should be similar to extended mean projection of S' (S''')
        loss5 = opt.d * criterion(mask_out, mean_proj_extend)

        # To train RDN, SubNet1, SubNet2 --> loss1, loss2, loss3, loss4
        RDN_loss = loss1 + loss2 + loss3 + loss4

        # To train STN --> loss2, loss5
        STN_loss = loss2 + loss5
        
        # to print loss values
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss3 += loss3.item()
        total_loss4 += loss4.item()
        total_loss5 += loss5.item()

        optimRDN.zero_grad()
        RDN_loss.backward(retain_graph=True)
        optimRDN.step()

        optimSTN.zero_grad()
        STN_loss.backward()
        optimSTN.step()

    schedulerRDN.step()
    schedulerSTN.step()

    total_train_loss = total_loss1 + total_loss2 + total_loss3 + total_loss4 + total_loss5

    print(f'Epoch: {e + 1} / {opt.epoch}, Loss: {total_train_loss:.3f}, '
          f'Loss1: {total_loss1:.3f}, Loss2: {total_loss2:.3f}, Loss3: {total_loss3:.3f}, '
          f'Loss4: {total_loss4:.3f}, Loss5: {total_loss5:.3f}')

    train_loss_ep.append(total_train_loss)
    train_loss1_ep.append(total_loss1)
    train_loss2_ep.append(total_loss2)
    train_loss3_ep.append(total_loss3)
    train_loss4_ep.append(total_loss4)
    train_loss5_ep.append(total_loss5)

    ### validation
    if e % 10 == 0:
        modelRDN.eval()
        modelSTN.eval()

        with torch.no_grad():
            img = valid_img.unsqueeze(0).unsqueeze(0)

            # Normalize using 99% percentile value (0~65525 -> 0~1.6)
            high = torch.quantile(img, 0.99)
            img = img / high

            img = img.to(device)

            output, mask = modelRDN(img.clone().detach())
            mask_out = modelSTN(mask.clone().detach())

            mask_out_fft = fftshift(fftn(mask_out))
            mask_out_fft_mag = torch.abs(mask_out_fft)  # magnitude

            # mask_out_fft_mag.shape = (b, c, h ,w) = (64, 1, 512, 512)
            proj_x, _ = torch.max(mask_out_fft_mag, dim=2)
            proj_y, _ = torch.max(mask_out_fft_mag, dim=3)
            proj_x_l1 = torch.norm(proj_x, 1)
            proj_y_l1 = torch.norm(proj_y, 1)

            ############################################
            mask_vis = mask[0].squeeze().detach().cpu().numpy()
            io.imsave(f'{temp_path}/mask_{e}.tif', mask_vis)

            mask_out_vis = mask_out[0].squeeze().detach().cpu().numpy()
            io.imsave(f'{temp_path}/mask_out_{e}.tif', mask_out_vis)

            mask_out_fft_mag_vis = mask_out_fft_mag[0].squeeze().detach().cpu().numpy()
            io.imsave(f'{temp_path}/mask_out_fft_linear_{e}.tif', mask_out_fft_mag_vis)

            mask_out_fft_mag_log_vis = 20 * np.log(mask_out_fft_mag_vis)
            io.imsave(f'{temp_path}/mask_out_fft_log_{e}.tif', mask_out_fft_mag_log_vis)

            output_vis = output[0].squeeze().detach().cpu()
            output_vis = (output_vis * high).numpy().astype('uint16')     # denormalize using only max
            io.imsave(f'{temp_path}/output_{e}.tif', output_vis)
            
            mean_proj_extend_vis = mean_proj_extend[0].squeeze().detach().cpu().numpy()
            print(mean_proj_extend_vis.shape)
            exit()
            io.imsave(f'{temp_path}/mean_proj_{e}.tif', mean_proj_extend_vis)

            plt.figure()
            ax1 = plt.subplot(1, 2, 1)
            plt.plot(range(opt.patch_size), 20*np.log(proj_x[0].squeeze().detach().cpu().numpy()))
            plt.title('proj_x (log-scale)')
            ax2 = plt.subplot(1, 2, 2, sharey=ax1)
            plt.plot(range(opt.patch_size), 20*np.log(proj_y[0].squeeze().detach().cpu().numpy()))
            plt.title('proj_y (log-scale)')
            plt.savefig(f'{plot_path}/xy_proj_{e}.png')
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
    plt.title(f'Loss4, c: {str(opt.c)}')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss4_curve.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_loss5_ep), label='d * loss5')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss5, d: {str(opt.d)}')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss5_curve.png')
    ############################################

    if e % 100 == 0:
        torch.save(modelRDN.state_dict(), f"{model_path}/UNet_{e}_{total_train_loss:.4f}.pth")
        torch.save(modelSTN.state_dict(), f"{model_path}/STN_{e}_{total_loss2:.4f}.pth")
