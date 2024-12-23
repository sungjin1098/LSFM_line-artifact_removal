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
parser.add_argument('--epoch', type=int, default=1000, help='number of training epochs')
parser.add_argument("--lrDis", type=float, default=1e-4, help="Discriminator learning rate")

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay of optimizer")
parser.add_argument("--batch_size", type=int, default=4, help="training batch size")

parser.add_argument("--in_dir", type=str, default='./dataset/Data_for_discriminator/TestDataForDiscriminator', help="dataset path")
parser.add_argument("--out_dir", default='./230717_test_discriminator_save_pth', type=str)

opt = parser.parse_args()

# save path
root = opt.out_dir
plot_path = root + '/loss_curve'
os.makedirs(plot_path, exist_ok=True)

# Prepare for use of CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataloader
train_data = LineDataset(opt.in_dir)
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)

# Sample image for model definition and valid image for validation
valid_data = LineDatasetValid(opt.in_dir)
valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)

# Define models
modelDiscriminator = Discriminator(input_dim=1, n_layer=3, dim=16, activ='relu', pad='zero', num_scales=2)

# If you use MULTIPLE GPUs, use 'DataParallel'
modelDiscriminator = nn.DataParallel(modelDiscriminator)

# to device
modelDiscriminator = modelDiscriminator.to(device)

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

                # oo
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
