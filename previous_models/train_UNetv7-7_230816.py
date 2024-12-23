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
from torch.utils.tensorboard import SummaryWriter

from model_UNetv7 import UNet_3Plus_for_S, UNet_3Plus_for_X, STN, Discriminator
from dataloader_3Dpre import LineDataset_S_is_added, LineDataset_valid

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=2000, help='number of training epochs')
parser.add_argument("--lrUNetX", type=float, default=1e-5, help="UNet_3Plus learning rate")
parser.add_argument("--lrUNetS", type=float, default=1e-5, help="UNet_3Plus learning rate")
parser.add_argument("--lrSTN", type=float, default=1e-4, help="STN learning rate")
parser.add_argument("--lrDisX", type=float, default=1e-8, help="Discriminator learning rate")
parser.add_argument("--lrDisS", type=float, default=1e-10, help="Discriminator learning rate")

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay of optimizer")
parser.add_argument("--patch_size", type=int, default=256, help="training patch size")
parser.add_argument("--batch_size", type=int, default=4, help="training batch size")
parser.add_argument("--valid_batch_size", type=int, default=5, help="validation image saving batch size")

parser.add_argument("--in_dir", type=str,
                    default='./dataset/230321_6-2/6-2_20170915_LSM_zStackFish_vol20-130_crop1000-1800.tif',
                    help="dataset path")
parser.add_argument("--out_dir", default='./230818_01_UNetv7-7_1e-1_1e-1_10_5_lrX_1e-8_lrS_1e-10_lrUX_1e-5_lrUS_1e-5', type=str)
parser.add_argument("--synthetic_S_dir", type=str, default='./dataset/synthetic_S/230726_synthetic_S_110vol.tif')
parser.add_argument("--valid_dir", type=str, default='./dataset/230321_6-2/ValidDataset')

parser.add_argument("--a", type=float, default=1e-1, help="weight for loss2 (UNet_S, TV of S should be large)")
parser.add_argument("--b", type=float, default=1e-1, help="weight for loss3 (UNet_S, Deviation from 1 loss of mask S)")
parser.add_argument("--c", type=float, default=10, help="weight for loss4 (UNet_X, Generator loss for X)")
parser.add_argument("--d", type=float, default=5, help="weight for loss5 (UNet_S, Generator loss for S)")

opt = parser.parse_args()
writer = SummaryWriter(log_dir="log/{}".format(opt.out_dir[2:11]))

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
train_data = LineDataset_S_is_added(img_path=opt.in_dir, synthetic_S_path=opt.synthetic_S_dir, patch_size=opt.patch_size)
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
valid_data = LineDataset_valid(valid_path=opt.valid_dir, synthetic_S_path=opt.synthetic_S_dir, patch_size=opt.patch_size)
valid_loader = DataLoader(dataset=valid_data, batch_size=opt.valid_batch_size, shuffle=False)

# Define models
modelUNet_for_S = UNet_3Plus_for_S(in_channels=1, out_channels=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
modelUNet_for_X = UNet_3Plus_for_X(in_channels=1, out_channels=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
modelDiscriminator_for_S = Discriminator(input_dim=1, n_layer=3, dim=16, activ='relu', pad='zero', num_scales=2)
modelDiscriminator_for_X = Discriminator(input_dim=1, n_layer=3, dim=16, activ='relu', pad='zero', num_scales=2)

# If you use MULTIPLE GPUs, use 'DataParallel'
modelUNet_for_S = nn.DataParallel(modelUNet_for_S)
modelUNet_for_X = nn.DataParallel(modelUNet_for_X)
modelDiscriminator_for_S = nn.DataParallel(modelDiscriminator_for_S)
modelDiscriminator_for_X = nn.DataParallel(modelDiscriminator_for_X)

# to device
modelUNet_for_S = modelUNet_for_S.to(device)
modelUNet_for_X = modelUNet_for_X.to(device)
modelDiscriminator_for_S = modelDiscriminator_for_S.to(device)
modelDiscriminator_for_X = modelDiscriminator_for_X.to(device)

# Loss function for L1 loss (reconstruction loss)
criterion = nn.MSELoss().to(device)
criterion_SSE = nn.MSELoss(reduction='sum').to(device)
criterion_BCE = nn.BCELoss().to(device)

# Total variation
tv = TotalVariation().to(device)

# optimizers
optimUNet_for_S = torch.optim.Adam(modelUNet_for_S.parameters(), lr=opt.lrUNetS, weight_decay=opt.weight_decay)
optimUNet_for_X = torch.optim.Adam(modelUNet_for_X.parameters(), lr=opt.lrUNetX, weight_decay=opt.weight_decay)
optimDiscriminator_for_S = torch.optim.Adam(modelDiscriminator_for_S.parameters(), lr=opt.lrDisS, weight_decay=opt.weight_decay)
optimDiscriminator_for_X = torch.optim.Adam(modelDiscriminator_for_X.parameters(), lr=opt.lrDisX, weight_decay=opt.weight_decay)

# schedulers
schedulerUNet_for_S = torch.optim.lr_scheduler.StepLR(optimUNet_for_S, step_size=50, gamma=0.5)
schedulerUNet_for_X = torch.optim.lr_scheduler.StepLR(optimUNet_for_X, step_size=50, gamma=0.5)
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

        # Normalize synthetic S using max value (0~65525 -> 0.5~1) --> normalize [a, b] = (b-a)*(x-Minx)/(Maxx-Minx)+a
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
        # loss3: Deviation from 1 loss for mask S (refer n*100 % percentile values from mask S)
        n = 0.95
        loss3 = 0
        for i in range(mask.size()[0]):
            top_n_value = torch.quantile(mask_1D[i].clone().detach(), n)
            top_n_gt = torch.where(mask_1D[i].clone().detach() > top_n_value, 1, 0).to(torch.float32)
            top_n_gt = top_n_gt.repeat(1, opt.patch_size, 1)
            top_n_mask = torch.mul(top_n_gt, mask[i])
            loss3 += criterion_SSE(top_n_gt, top_n_mask)
        loss3 = opt.b * loss3

        tensor_one = torch.ones_like(dis_outputX).to(device)
        tensor_zero = torch.zeros_like(dis_outputX).to(device)

        # loss4: GAN loss for UNet_for_X
        loss4 = opt.c * criterion_BCE(dis_outputX, tensor_one)
        # GAN loss for Discriminator X
        lossDisX = criterion_BCE(dis_outputX_rotated, tensor_one) + criterion_BCE(dis_outputX, tensor_zero)
        # loss5: GAN loss for UNet_for_S
        loss5 = opt.d * criterion_BCE(dis_outputS, tensor_one)
        # GAN loss for Discriminator S
        lossDisS = criterion_BCE(dis_outputS_synthetic, tensor_one) + criterion_BCE(dis_outputS, tensor_zero)

        # # loss4: GAN loss for UNet_for_X
        # loss4 = opt.c * torch.sum((dis_outputX - 1) ** 2)
        # # GAN loss for Discriminator X
        # lossDisX = torch.sum((dis_outputX_rotated - 1) ** 2) + torch.sum((dis_outputX - 0) ** 2)
        # # loss5: GAN loss for UNet_for_S
        # loss5 = opt.d * torch.sum((dis_outputS - 1) ** 2)
        # # GAN loss for Discriminator S
        # lossDisS = torch.sum((dis_outputS_synthetic - 1) ** 2) + torch.sum((dis_outputS - 0) ** 2)

        # Now, we use only loss1 to update three networks.
        loss_UNet_for_X = loss1 + loss4
        loss_UNet_for_S = loss2 + loss3 + loss5
        loss_Discriminator_for_X = lossDisX
        loss_Discriminator_for_S = lossDisS

        # to print loss values
        total_loss1.append(loss1.item())
        total_loss2.append(loss2.item())
        total_loss3.append(loss3.item())
        total_loss4.append(loss4.item())
        total_loss5.append(loss5.item())
        total_lossDisX.append(lossDisX.item())
        total_lossDisS.append(lossDisS.item())

        optimUNet_for_X.zero_grad()
        loss_UNet_for_X.backward(retain_graph=True)
        optimUNet_for_X.step()

        optimUNet_for_S.zero_grad()
        loss_UNet_for_S.backward(retain_graph=True)
        optimUNet_for_S.step()

        optimDiscriminator_for_X.zero_grad()
        loss_Discriminator_for_X.backward(retain_graph=True)
        optimDiscriminator_for_X.step()

        optimDiscriminator_for_S.zero_grad()
        loss_Discriminator_for_S.backward(retain_graph=True)
        optimDiscriminator_for_S.step()

    schedulerUNet_for_X.step()
    schedulerUNet_for_S.step()
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

    writer.add_scalar('Total train loss', total_train_loss, e)
    writer.add_scalar('Loss1', total_loss1, e)
    writer.add_scalar('Loss2', total_loss2, e)
    writer.add_scalar('Loss3', total_loss3, e)
    writer.add_scalar('Loss4', total_loss4, e)
    writer.add_scalar('Loss5', total_loss5, e)
    writer.add_scalar('Loss DisX', total_lossDisX, e)
    writer.add_scalar('Loss DisS', total_lossDisS, e)

    ### Validation
    if (e + 1) % 50 == 0:
        valid_save_path = f'{image_path}/epoch{e + 1}'
        valid_array_save_path = f'{temp_path}/epoch{e+1}'
        os.makedirs(valid_save_path, exist_ok=True)
        os.makedirs(valid_array_save_path, exist_ok=True)

        modelUNet_for_S.eval()
        modelUNet_for_X.eval()
        modelDiscriminator_for_S.eval()
        modelDiscriminator_for_X.eval()

        dis_outputX_his = []
        dis_outputX_rotated_his = []
        dis_outputS_his = []
        dis_outputS_synthetic_his = []

        with torch.no_grad():
            for idx, datas in enumerate(valid_loader):
                # Dataloader = (img, synthetic S)
                img, synthetic_S, idx = datas

                # (b, h, w) --> (b, c, h, w) in this case, (b, h, w) --> (b, 1, h, w)
                img = img.unsqueeze(1)
                synthetic_S = synthetic_S.unsqueeze(1)

                # Normalize using 99% percentile value (0~65525 -> 0~1.6)
                high = torch.quantile(img, 0.99)
                img = img / high

                # Normalize synthetic S using max value (0~65525 -> 0.5~1) --> normalize [a, b] = (b-a)*(x-Minx)/(Maxx-Minx)+a
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
                dis_outputX_rotated = modelDiscriminator_for_X(torch.rot90(output.clone().detach(), k=1, dims=[2, 3]))

                # 4. {S, synthetic S} --[Discriminator_for_S]-> 0/1
                dis_outputS = modelDiscriminator_for_S(mask.clone().detach())
                dis_outputS_synthetic = modelDiscriminator_for_S(synthetic_S.clone().detach())

                ############# PNG Image Array Save #############
                img_grid = make_grid(img, padding=2, pad_value=1)
                synthetic_S_grid = make_grid(synthetic_S, padding=2, pad_value=1)
                mask_grid = make_grid(mask, padding=2, pad_value=1)
                output_temp = output * high
                output_temp_flat = torch.flatten(output_temp, start_dim=1, end_dim=-1)
                output_temp /= torch.quantile(output_temp_flat, 0.99, dim=1)[..., None, None, None]
                output_grid = make_grid(output_temp, padding=2, pad_value=1)
                color_grid = make_grid(torch.cat([output, img, torch.zeros_like(img)], dim=1), padding=2, pad_value=1)

                save_grid = torch.cat([img_grid, synthetic_S_grid, mask_grid, output_grid, color_grid], dim=1)
                save_image(save_grid, f'{valid_array_save_path}/e{e + 1}_{idx[0]}-{idx[-1]}.png')
                ################################################

                for i in range(opt.valid_batch_size):
                    valid_idx = idx[i]

                    ############# Discriminator histogram #############
                    dis_outputX_temp = dis_outputX[i].item()
                    dis_outputX_rotated_temp = dis_outputX_rotated[i].item()
                    dis_outputS_temp = dis_outputS[i].item()
                    dis_outputS_synthetic_temp = dis_outputS_synthetic[i].item()

                    dis_outputX_his.append(dis_outputX_temp)
                    dis_outputX_rotated_his.append(dis_outputX_rotated_temp)
                    dis_outputS_his.append(dis_outputS_temp)
                    dis_outputS_synthetic_his.append(dis_outputS_synthetic_temp)
                    ###################################################

                    ############# Image Save #############
                    mask_vis = mask[i].squeeze().detach().cpu().numpy()
                    io.imsave(f'{valid_save_path}/e{e+1}_{valid_idx}_maskS_{dis_outputS_temp:.3f}.tif', mask_vis)

                    synthetic_S_vis = synthetic_S[i].squeeze().detach().cpu().numpy()
                    io.imsave(f'{valid_save_path}/e{e+1}_{valid_idx}_syntheticS_{dis_outputS_synthetic_temp:.3f}.tif', synthetic_S_vis)

                    output_vis = output[i].squeeze().detach().cpu()
                    output_vis = (output_vis * high).numpy().astype('uint16')  # denormalize using only max
                    io.imsave(f'{valid_save_path}/e{e+1}_{valid_idx}_outputX_{dis_outputX_temp:.3f}_r{dis_outputX_rotated_temp:.3f}.tif', output_vis)
                    ######################################

        ############# Discriminator histogram #############
        plt.figure()
        plt.hist(dis_outputX_his, alpha=0.5, label='X (0)', range=(0, 1), bins=20)
        plt.hist(dis_outputX_rotated_his, alpha=0.5, label='rotated X (1)', range=(0, 1), bins=20)
        plt.title('Discriminator output of {X, rotated X}')
        plt.xlabel('Discriminator output value (0~1)')
        plt.ylabel('# of images')
        plt.legend()
        plt.savefig(f'{valid_array_save_path}/e{e+1}_histogram_X.png')

        plt.figure()
        plt.hist(dis_outputS_his, alpha=0.5, label='S (0)', range=(0, 1), bins=20)
        plt.hist(dis_outputS_synthetic_his, alpha=0.5, label='synthetic S (1)', range=(0, 1), bins=20)
        plt.title('Discriminator output of {S, synthetic S}')
        plt.xlabel('Discriminator output value (0~1)')
        plt.ylabel('# of images')
        plt.legend()
        plt.savefig(f'{valid_array_save_path}/e{e + 1}_histogram_S.png')
        ###################################################


    ############################################
    # Save loss curves [[1, 2, 3, total], [4, X, 5, S]]
    epoch_ep_n = np.array(epoch_ep)
    fig, ax = plt.subplots(2, 4, figsize=(20, 8), layout='constrained')

    ax[0, 0].plot(epoch_ep_n, np.array(train_loss1_ep))
    ax[0, 0].set_title(f'Loss1')

    ax[0, 1].plot(epoch_ep_n, np.array(train_loss2_ep))
    ax[0, 1].set_title(f'{str(opt.a)} * Loss2')

    ax[0, 2].plot(epoch_ep_n, np.array(train_loss3_ep))
    ax[0, 2].set_title(f'{str(opt.b)} * Loss3')

    ax[0, 3].set_title('Total loss')
    ax[0, 3].plot(epoch_ep_n, np.array(train_loss_ep))

    ax[1, 0].plot(epoch_ep_n, np.array(train_loss4_ep))
    ax[1, 0].set_title(f'{str(opt.c)} * Loss4')

    ax[1, 1].plot(epoch_ep_n, np.array(train_lossDisX_ep))
    ax[1, 1].set_title(f'Loss DisX')

    ax[1, 2].plot(epoch_ep_n, np.array(train_loss5_ep))
    ax[1, 2].set_title(f'{str(opt.d)} * Loss5')

    ax[1, 3].plot(epoch_ep_n, np.array(train_lossDisS_ep))
    ax[1, 3].set_title('Loss DisS')

    fig.supxlabel('Epoch')
    fig.supylabel('Loss')
    plt.savefig(f'{plot_path}/loss_curve.png')

    ############################################

    if (e+1) % 100 == 0:
        torch.save(modelUNet_for_S.state_dict(), f"{model_path}/UNet_for_S_{e+1}_{total_train_loss:.4f}.pth")
        torch.save(modelUNet_for_X.state_dict(), f"{model_path}/UNet_for_X_{e+1}_{total_train_loss:.4f}.pth")
        torch.save(modelDiscriminator_for_S.state_dict(), f"{model_path}/Discriminator_for_S_{e+1}_{total_lossDisS:.4f}.pth")
        torch.save(modelDiscriminator_for_X.state_dict(), f"{model_path}/Discriminator_for_X_{e+1}_{total_lossDisX:.4f}.pth")
