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

from model_RDN import RDN, STN
from dataloader_3D import LineDataset


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=2000, help='number of training epochs')
parser.add_argument("--lrUNet", type=float, default=1e-4, help="RDN learning rate")
parser.add_argument("--lrSTN", type=float, default=1e-4, help="STN learning rate")

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay of optimizer")
parser.add_argument("--patch_size", type=int, default=512, help="training patch size")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")

parser.add_argument("--in_dir", type=str, default='./dataset/230321_6-2/6-2_20170915_LSM_zStackFish_vol20-130_crop1000-1800.tif', help="dataset path")
parser.add_argument("--out_dir", default='./230322_loss134_78_again', type=str)

parser.add_argument("--a", type=float, default=1e-7, help="weight for loss2 (L1-loss) term")
parser.add_argument("--b", type=float, default=1e-7, help="weight for loss3 term")
parser.add_argument("--c", type=float, default=1e-8, help="weight for loss4 term")
parser.add_argument("--d", type=float, default=1e-7, help="weight for loss5 term")

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

sample_img = next(iter(train_loader))

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
        origin_img = img

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

        # the smaller value becomes loss. projection along x-axis vs y-axis
        if proj_x_l1 < proj_y_l1:
            loss2 = proj_x_l1
            loss5 = torch.sum(proj_y - proj_x)
        else:
            loss2 = proj_y_l1
            loss5 = torch.sum(proj_x - proj_y)

        # loss1: input and output must be similar.
        loss1 = criterion(img, output)
        # loss2: L1 norm of 1D-projected vector must be small.
        loss2 = opt.a * loss2
        # loss3: output must be positive.
        loss3 = abs(torch.sum(torch.where(output < 0, output, 0)))
        loss3 = opt.b * loss3
        # loss4: deviation from 1 loss.
        loss4 = torch.sum(torch.where(mask < 1, 0.5*(mask-1)**2, 5*(mask-1)**2))
        loss4 = opt.c * loss4
        # loss5: the difference between two 1D-projected vectors must be large
        loss5 = -opt.d * loss5

        # total loss except loss2
        # loss = loss1 + loss3 + loss4 + loss5
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        # to print loss values
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss3 += loss3.item()
        total_loss4 += loss4.item()
        total_loss5 += loss5.item()

        optimUNet.zero_grad()
        loss.backward(retain_graph=True)
        optimUNet.step()

        optimSTN.zero_grad()
        loss2.backward()
        optimSTN.step()

    total_train_loss = total_loss1 + total_loss2 + total_loss3 + total_loss4 + total_loss5

    print(f'Epoch: {e + 1} / {opt.epoch}, Loss: {total_train_loss:.3f}, '
          f'Loss1: {total_loss1:.3f}, Loss2: {total_loss2:.3f}, '
          f'Loss3: {total_loss3:.3f}, Loss4: {total_loss4:.3f}, Loss5: {total_loss5:.3f}')

    train_loss_ep.append(total_train_loss)
    train_loss1_ep.append(total_loss1)
    train_loss2_ep.append(total_loss2)
    train_loss3_ep.append(total_loss3)
    train_loss4_ep.append(total_loss4)
    train_loss5_ep.append(total_loss5)

    ############################################
    # Just for visualization
    if e % 10 == 0:
        mask_vis = mask[0].squeeze().detach().cpu().numpy()
        # # Normalization
        # minval = np.min(mask_vis)
        # maxval = np.max(mask_vis)
        # mask_vis = ((mask_vis - minval) / (maxval - minval)) * 255
        # mask_vis = np.clip(np.round(mask_vis), 0, 255).astype(np.uint8)
        io.imsave(f'{temp_path}/mask_{e}.tif', mask_vis)

        mask_out_vis = mask_out[0].squeeze().detach().cpu().numpy()
        # minval = np.min(mask_out_vis)
        # maxval = np.max(mask_out_vis)
        # mask_out_vis = ((mask_out_vis - minval) / (maxval - minval)) * 255
        # mask_out_vis = np.clip(np.round(mask_out_vis), 0, 255).astype(np.uint8)
        io.imsave(f'{temp_path}/mask_out_{e}.tif', mask_out_vis)

        mask_out_fft_mag_vis = mask_out_fft_mag[0].squeeze().detach().cpu().numpy()
        # minval = np.min(mask_out_fft_mag_vis)
        # maxval = np.max(mask_out_fft_mag_vis)
        # mask_out_fft_mag_vis = ((mask_out_fft_mag_vis - minval) / (maxval - minval)) * 255
        # mask_out_fft_mag_vis = np.clip(np.round(mask_out_fft_mag_vis), 0, 255).astype(np.uint8)
        io.imsave(f'{temp_path}/mask_out_fft_linear_{e}.tif', mask_out_fft_mag_vis)

        mask_out_fft_mag_log_vis = 20 * np.log(mask_out_fft_mag_vis)
        io.imsave(f'{temp_path}/mask_out_fft_log_{e}.tif', mask_out_fft_mag_log_vis)

        input_vis = origin_img[0].squeeze().detach().cpu().numpy().astype('uint16')
        io.imsave(f'{temp_path}/input_{e}.tif', input_vis)

        output_vis = output[0].squeeze().detach().cpu()
        # Denormalize
        # output_vis = (output_vis * (high - low) + low).numpy().astype('uint16')     # denormalize using min and max
        output_vis = (output_vis * high).numpy().astype('uint16')     # denormalize using only max
        # output_vis = np.where(output_vis < 0, 0, output_vis)
        # output_vis = (output_vis * max_value).astype('uint16')
        io.imsave(f'{temp_path}/output_{e}.tif', output_vis)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(range(opt.patch_size), proj_x[0].squeeze().detach().cpu().numpy())
        plt.title('proj_x')
        plt.subplot(1, 2, 2)
        plt.plot(range(opt.patch_size), proj_y[0].squeeze().detach().cpu().numpy())
        plt.title('proj_y')
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
    plt.title(f'Loss3, a: {str(opt.b)}')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss3_curve.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_loss4_ep), label='c * loss4')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss4, a: {str(opt.c)}')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss4_curve.png')

    plt.clf()
    plt.plot(epoch_ep_n, np.array(train_loss5_ep), label='d * loss5')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss5, a: {str(opt.d)}')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/loss5_curve.png')
    ############################################

    if e % 100 == 0:
        torch.save(modelRDN.state_dict(), f"{model_path}/UNet_{e}_{total_train_loss:.4f}.pth")
        torch.save(modelSTN.state_dict(), f"{model_path}/STN_{e}_{total_loss2:.4f}.pth")