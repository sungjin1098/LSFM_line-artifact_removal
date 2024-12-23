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

from model_UNetv7 import Discriminator
from dataloader_test_discriminator_by_using_Y import LineDataset, LineDatasetValid

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--in_dir", type=str, default='./dataset/Data_for_discriminator/ConfocalForDiscriminator', help="dataset path")
parser.add_argument("--out_dir", type=str, default='./230717_test_discriminator_inference')
parser.add_argument("--model_path", type=str, default='./model.pth', help="inference model path")

opt = parser.parse_args()

# save path
root = opt.out_dir
os.makedirs(root, exist_ok=True)

# Prepare for use of CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inference data
valid_data = LineDatasetValid(opt.in_dir)
valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)

# Define models
modelDiscriminator = Discriminator(input_dim=1, n_layer=3, dim=16, activ='relu', pad='zero', num_scales=2)

# Data parallel
modelDiscriminator = nn.DataParallel(modelDiscriminator)

# to device
modelDiscriminator = modelDiscriminator.to(device)

# Load trained model
modelDiscriminator.load_state_dict(torch.load("./model.pth"))

### validation #################
accuracy_cnt = 0
dis_output_graph = []
dis_output_rotated_graph = []

for idx, (img, img_name) in enumerate(valid_loader):
    modelDiscriminator.eval()

    with torch.no_grad():
        mis_path_oo = root + '/oo'
        mis_path_ox = root + '/ox'
        mis_path_xo = root + '/xo'
        mis_path_xx = root + '/xx'
        os.makedirs(mis_path_oo, exist_ok=True)
        os.makedirs(mis_path_ox, exist_ok=True)
        os.makedirs(mis_path_xo, exist_ok=True)
        os.makedirs(mis_path_xx, exist_ok=True)

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

        # To draw histogram
        dis_output_graph.append(dis_output[0].item())
        dis_output_rotated_graph.append(dis_output_rotated[0].item())

accuracy = accuracy_cnt / len(valid_loader) * 100
print(f'Accuracy: {accuracy:.3f}')

# Save histogram
plt.clf()
plt.hist(dis_output_graph, alpha=0.5, label='Original (should be 0)', range=(0, 1), bins=20)
plt.hist(dis_output_rotated_graph, alpha=0.5, label='Rotated (should be 1)', range=(0, 1), bins=20)
plt.xlabel('Classification score (0~1)')
plt.ylabel('# of images')
plt.title(f'Classification accuracy: {accuracy:.3f}')
plt.legend()
plt.savefig(root + '/histogram.png')