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
from torchvision.utils import save_image, make_grid

from model_UNetv13 import UNet_3Plus_for_S, UNet_3Plus_for_X, STN, Discriminator
from dataloader_3Dpre import LineDataset_S_is_added, LineDataset_valid

warnings.filterwarnings("ignore")


def gen_index(input_size, patch_size, overlap_size):
    indices = []
    for k in range(2):
        z_range = list(range(0, input_size[k] - patch_size[k] + 1, overlap_size[k]))
        if input_size[k] - patch_size[k] > z_range[-1]:
            z_range.append(input_size[k] - patch_size[k])
        indices.append(z_range)
    return indices


def infer(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define models
    modelUNet_for_S = UNet_3Plus_for_S(in_channels=1, out_channels=1, feature_scale=4)
    modelUNet_for_X = UNet_3Plus_for_X(in_channels=1, out_channels=1, feature_scale=4)

    # Data parallel
    modelUNet_for_S = nn.DataParallel(modelUNet_for_S)
    modelUNet_for_X = nn.DataParallel(modelUNet_for_X)

    # Load model
    modelUNet_for_S.load_state_dict(torch.load(opt.modelUNet_for_S_path))
    modelUNet_for_X.load_state_dict(torch.load(opt.modelUNet_for_X_path))

    # to device
    modelUNet_for_S = modelUNet_for_S.to(device)
    modelUNet_for_X = modelUNet_for_X.to(device)

    # eval mode
    modelUNet_for_S.eval()
    modelUNet_for_X.eval()

    # load 3D image
    img_3D = io.imread(opt.in_dir)
    img_3D_output = torch.zeros_like(torch.Tensor(img_3D.astype('float32')))
    mask_3D_output = torch.zeros_like(torch.Tensor(img_3D.astype('float32')))

    with torch.no_grad():
        for i in tqdm(range(img_3D.shape[0])):
            img_2D = torch.Tensor(img_3D[i].astype('float32'))

            # (h, w) --> (b, c, h, w)
            img = img_2D.unsqueeze(0).unsqueeze(0)

            # Normalize using 99% percentile value (0~65525 -> 0~1.6)
            high = torch.quantile(img_2D, 0.99)
            img = img / high
            
            # add margin
            img_pad = nn.functional.pad(img, (opt.pad, opt.pad, opt.pad, opt.pad), mode='reflect')

            val_ind = gen_index(img_pad.size()[2:], [opt.patch+opt.pad+opt.pad, opt.patch+opt.pad+opt.pad], [opt.patch, opt.patch])

            h = img.size()[2]
            w = img.size()[3]
            x_remain = h % opt.patch
            y_remain = w % opt.patch
            if x_remain == 0:
                x_remain = opt.patch
            if y_remain == 0:
                y_remain = opt.patch

            imgs = torch.zeros(img.size(), dtype=torch.float32)
            masks = torch.zeros(img.size(), dtype=torch.float32)

            for xi in range(len(val_ind[0])):
                for yi in range(len(val_ind[1])):
                    img_small = img_pad[:, :, val_ind[0][xi]:val_ind[0][xi] + opt.patch + opt.pad + opt.pad, val_ind[1][yi]:val_ind[1][yi] + opt.patch + opt.pad + opt.pad].to(device)

                    _, mask = modelUNet_for_S(img_small)
                    output = modelUNet_for_X(img_small)

                    if xi != len(val_ind[0]) - 1 and yi == len(val_ind[1]) - 1:
                        imgs[:, :, xi * opt.patch:(xi + 1) * opt.patch, yi * opt.patch:] = output[:, :, opt.pad:-opt.pad, opt.pad + opt.patch - y_remain:-opt.pad]
                        masks[:, :, xi * opt.patch:(xi + 1) * opt.patch, yi * opt.patch:] = mask[:, :, opt.pad:-opt.pad, opt.pad + opt.patch - y_remain:-opt.pad]
                    elif xi == len(val_ind[0]) - 1 and yi != len(val_ind[1]) - 1:
                        imgs[:, :, xi * opt.patch:, yi * opt.patch:(yi + 1) * opt.patch] = output[:, :, opt.pad + opt.patch - x_remain:-opt.pad, opt.pad:-opt.pad]
                        masks[:, :, xi * opt.patch:, yi * opt.patch:(yi + 1) * opt.patch] = mask[:, :, opt.pad + opt.patch - x_remain:-opt.pad, opt.pad:-opt.pad]
                    elif xi == len(val_ind[0]) - 1 and yi == len(val_ind[1]) - 1:
                        imgs[:, :, xi * opt.patch:, yi * opt.patch:] = output[:, :, opt.pad + opt.patch - x_remain:-opt.pad, opt.pad + opt.patch - y_remain:-opt.pad]
                        masks[:, :, xi * opt.patch:, yi * opt.patch:] = mask[:, :, opt.pad + opt.patch - x_remain:-opt.pad, opt.pad + opt.patch - y_remain:-opt.pad]
                    else:
                        imgs[:, :, xi * opt.patch:(xi + 1) * opt.patch, yi * opt.patch:(yi + 1) * opt.patch] = output[:, :, opt.pad:-opt.pad, opt.pad:-opt.pad]
                        masks[:, :, xi * opt.patch:(xi + 1) * opt.patch, yi * opt.patch:(yi + 1) * opt.patch] = mask[:, :, opt.pad:-opt.pad, opt.pad:-opt.pad]

            img_3D_output[i] = (imgs.squeeze().squeeze().detach().cpu() * high)
            mask_3D_output[i] = masks.squeeze().squeeze().detach().cpu()

        io.imsave(f'{opt.out_dir}/test_output_3D.tif', img_3D_output.numpy().astype('uint16'))
        io.imsave(f'{opt.out_dir}/test_mask_3D.tif', mask_3D_output.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input options: No PSF / Synthetic / Real
    parser.add_argument('--in_dir', type=str, default='./dataset/231030_LSM_simulation_ExcludePSF/231030_input_Y_110vol_ExcludePSF.tif')
    # parser.add_argument('--in_dir', type=str, default='./dataset/230925_LSM_simulation_ExcludePSF/230925_input_Y_110vol_ExcludePSF.tif')
    # parser.add_argument('--in_dir', type=str, default='./dataset/230817_LSM_simulation/230817_input_Y_110vol.tif')
    # parser.add_argument('--in_dir', type=str, default='./dataset/230321_6-2/6-2_20170915_LSM_zStackFish_vol20-130_crop1000-1800.tif')
    parser.add_argument('--modelUNet_for_S_path', type=str, default='./UNet_for_S.pth')
    parser.add_argument('--modelUNet_for_X_path', type=str, default='./UNet_for_X.pth')
    parser.add_argument('--out_dir', type=str, default='./test_output')
    parser.add_argument("--patch", type=int, default=128, help='patch size, 256 is default')
    parser.add_argument("--pad", type=int, default=192, help='padding, 512 is default')
    ### (patch size + 2*padding) % 256 = 0이어야 함
    opt = parser.parse_args()

    if not os.path.exists(opt.out_dir): os.makedirs(opt.out_dir)

    infer(opt)
