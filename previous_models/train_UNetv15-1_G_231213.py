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

from model_UNetv15 import UNet_3Plus_for_S, UNet_3Plus_for_X, Discriminator_revised
from dataloader_3Dpre import LineDataset_S_is_added, LineDataset_valid

### S 네트워크, X 네트워크는 v13과 같음
### Discriminator는 수정됨
## loss2 수정됨. 더이상 latent의 tv를 계산하지 않음

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=2000, help='number of training epochs')
parser.add_argument("--valid_save_period", type=int, default=200, help='save the valid results for every N epochs')
parser.add_argument("--model_save_period", type=int, default=200, help='save the model pth file for every N epochs')

parser.add_argument("--lrDisS", type=float, default=1e-5, help="Discriminator learning rate")
parser.add_argument("--lrUNetX", type=float, default=1e-4, help="UNet_3Plus learning rate")
parser.add_argument("--lrUNetS", type=float, default=1e-4, help="UNet_3Plus learning rate")
parser.add_argument("--lrSTN", type=float, default=1e-4, help="STN learning rate")

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay of optimizer")
parser.add_argument("--patch_size", type=int, default=256, help="training patch size")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
parser.add_argument("--valid_batch_size", type=int, default=5, help="validation image saving batch size")

parser.add_argument("--in_dir", type=str,
                    default='./dataset/230321_6-2/6-2_20170915_LSM_zStackFish_vol20-130_crop1000-1800.tif',
                    help="dataset path")
parser.add_argument("--synthetic_S_dir", type=str, default='./dataset/synthetic_S/230726_synthetic_S_110vol.tif')
parser.add_argument("--valid_dir", type=str, default='./dataset/230321_6-2/ValidDataset')
parser.add_argument("--out_dir", type=str, default='./231213_00', help='hyperparameters are saved at the end automatically')

parser.add_argument("--a", type=float, default=10, help="weight for loss2 (UNet_S, TV of S should be large)")
parser.add_argument("--b", type=float, default=100, help="weight for loss3 (UNet_S, Deviation from 1 loss of mask S)")
parser.add_argument("--c", type=float, default=1, help="weight for loss4 (UNet_X, mask of X should be 1-tensor)")
parser.add_argument("--d", type=float, default=1, help="weight for loss5 (UNet_S, Generator loss for S)")
parser.add_argument("--num_G", type=int, default=2, help="number of Generator learning iterations")

opt = parser.parse_args()
writer = SummaryWriter(log_dir="log/{}".format(opt.out_dir[2:11]))

# NaN이 생기면 즉시 학습을 중단하고, 오류가 생긴 위치를 출력
torch.autograd.set_detect_anomaly(True)

# save path
root = f'{opt.out_dir}_UNetv15-1_G{opt.num_G}_{opt.a}_{opt.b}_{opt.c}_{opt.d}_lrDS_{opt.lrDisS}_lrUX_{opt.lrUNetX}_lrUS_{opt.lrUNetS}_b{opt.batch_size}'
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
modelUNet_for_S = UNet_3Plus_for_S(in_channels=1, out_channels=1, feature_scale=4)
modelUNet_for_X = UNet_3Plus_for_X(in_channels=1, out_channels=1, feature_scale=4)
modelDiscriminator_for_S = Discriminator_revised(input_dim=1, dim=64)

# If you use MULTIPLE GPUs, use 'DataParallel'
modelUNet_for_S = nn.DataParallel(modelUNet_for_S)
modelUNet_for_X = nn.DataParallel(modelUNet_for_X)
modelDiscriminator_for_S = nn.DataParallel(modelDiscriminator_for_S)

# Load pretrained model
modelUNet_for_S.load_state_dict(torch.load('./BEST_PTH/UNet_for_S_200_v13-1_unsupervised.pth'))
modelUNet_for_X.load_state_dict(torch.load('./BEST_PTH/UNet_for_X_200_v13-1_unsupervised.pth'))
### Discriminator는 v13과 비교했을 때 모델 구조가 바뀌어서 pretrained 불러올 수 없음
# modelDiscriminator_for_S.load_state_dict(torch.load('./BEST_PTH/Discriminator_for_S_200_v13-1_unsupervised.pth'))

# to device
modelUNet_for_S = modelUNet_for_S.to(device)
modelUNet_for_X = modelUNet_for_X.to(device)
modelDiscriminator_for_S = modelDiscriminator_for_S.to(device)

# Loss function for L1 loss (reconstruction loss)
criterion = nn.MSELoss().to(device)
criterion_BCE = nn.BCELoss().to(device)

# Total variation
tv = TotalVariation(reduction='mean').to(device)

# optimizers
optimUNet_for_S = torch.optim.Adam(modelUNet_for_S.parameters(), lr=opt.lrUNetS, weight_decay=opt.weight_decay)
optimUNet_for_X = torch.optim.Adam(modelUNet_for_X.parameters(), lr=opt.lrUNetX, weight_decay=opt.weight_decay)
optimDiscriminator_for_S = torch.optim.Adam(modelDiscriminator_for_S.parameters(), lr=opt.lrDisS, weight_decay=opt.weight_decay)

# schedulers
schedulerUNet_for_S = torch.optim.lr_scheduler.StepLR(optimUNet_for_S, step_size=50, gamma=0.5)
schedulerUNet_for_X = torch.optim.lr_scheduler.StepLR(optimUNet_for_X, step_size=50, gamma=0.5)
schedulerDiscriminator_for_S = torch.optim.lr_scheduler.StepLR(optimDiscriminator_for_S, step_size=50, gamma=0.5)

# To draw loss graph
train_loss_ep = []
train_loss1_ep = []
train_loss2_ep = []
train_loss3_ep = []
train_loss4_ep = []
train_loss5_ep = []
train_lossDisS_ep = []
epoch_ep = []

### train
for e in range(opt.epoch):
    epoch_ep.append(e + 1)

    modelUNet_for_X.train()
    modelUNet_for_S.train()
    modelDiscriminator_for_S.train()

    total_train_loss = 0
    total_loss1 = []
    total_loss2 = []
    total_loss3 = []
    total_loss4 = []
    total_loss5 = []
    total_lossDisS = []

    for _, datas in enumerate(train_loader):
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
        synthetic_S = synthetic_S / high_s

        img = img.to(device)
        synthetic_S = synthetic_S.to(device)

        # 1. Y'' --[Enc1]-> 1D vector S --[Dec1]-> S'
        mask_1D, mask = modelUNet_for_S(img)

        # 2. Y'' --[Enc2]-> 2D feature X --[Dec2]-> X
        output = modelUNet_for_X(img)

        # 3. X --[Enc1][Dec1]-> mask of X
        _, mask_of_X = modelUNet_for_S(output)

        # 4. {S, synthetic S} --[Discriminator_for_S]-> 0/1
        dis_outputS = modelDiscriminator_for_S(mask.clone())
        dis_outputS_synthetic = modelDiscriminator_for_S(synthetic_S.clone())

        # loss1: {input} and {output * mask} should be similar. --> UNet, STN 학습에 사용
        loss1 = criterion(img, mask * output)
        # loss2: mask_1D 의 Total Variation 커야 함 --> UNet 학습에 사용
        mask_1D, _ = torch.max(mask, dim=2)     # (b, 1, w)
        mask_1D = mask_1D.unsqueeze(1)          # (b, 1, 1, w)
        loss2 = -opt.a * tv(mask_1D) / (opt.patch_size - 1)
        # loss3: Deviation from 1 loss for mask S (refer n*100 % percentile values from mask S)
        n = 0.01
        num_pixel = int(opt.patch_size * opt.patch_size * n)
        one_tensor = torch.ones(num_pixel).to(device)
        loss3_Y = 0
        loss3_S = 0
        for i in range(img.size()[0]):
            img_temp = torch.sort(torch.flatten(img[i]), descending=True)
            img_temp_idx = img_temp[1][:num_pixel]
            mask_temp = torch.flatten(mask[i])[img_temp_idx]
            loss3_Y += criterion(mask_temp, one_tensor)

            mask_temp = torch.sort(torch.flatten(mask[i]), descending=True)
            mask_temp = mask_temp[0][:num_pixel]
            loss3_S += criterion(mask_temp, one_tensor)
        loss3_Y = loss3_Y / (img.size()[0] * num_pixel)
        loss3_S = loss3_S / (img.size()[0] * num_pixel)
        loss3 = opt.b * ((opt.epoch - e) * loss3_Y + e * loss3_S) / opt.epoch

        tensor_one_for_loss4 = torch.ones_like(mask_of_X).to(device)
        tensor_one = torch.ones_like(dis_outputS).to(device)
        tensor_zero = torch.zeros_like(dis_outputS).to(device)

        # loss4: mask of X should be 1.
        loss4 = opt.c * criterion(mask_of_X, tensor_one_for_loss4)
        # loss5: GAN loss for UNet_for_S
        loss5 = opt.d * criterion_BCE(dis_outputS, tensor_one)
        # GAN loss for Discriminator S
        lossDisS = (criterion_BCE(dis_outputS_synthetic, tensor_one) + criterion_BCE(dis_outputS, tensor_zero))/2

        # Now, we use only loss1 to update three networks.
        loss_UNet_for_X = loss1 + loss4
        loss_UNet_for_S = loss1 + loss2 + loss3 + loss5
        loss_Discriminator_for_S = lossDisS

        # to print loss values
        total_loss1.append(loss1.item())
        total_loss2.append(loss2.item())
        total_loss3.append(loss3.item())
        total_loss4.append(loss4.item())
        total_loss5.append(loss5.item())
        total_lossDisS.append(lossDisS.item())

        optimUNet_for_X.zero_grad()
        optimUNet_for_S.zero_grad()

        loss_UNet_for_S.backward(retain_graph=True)
        optimUNet_for_S.step()

        loss_UNet_for_X.backward(retain_graph=True)
        optimUNet_for_X.step()

        if i % opt.num_G == 0:
            optimDiscriminator_for_S.zero_grad()
            loss_Discriminator_for_S.backward(retain_graph=True)
            optimDiscriminator_for_S.step()

    schedulerUNet_for_X.step()
    schedulerUNet_for_S.step()
    schedulerDiscriminator_for_S.step()

    total_loss1 = sum(total_loss1) / len(total_loss1)
    total_loss2 = sum(total_loss2) / len(total_loss2)
    total_loss3 = sum(total_loss3) / len(total_loss3)
    total_loss4 = sum(total_loss4) / len(total_loss4)
    total_loss5 = sum(total_loss5) / len(total_loss5)
    total_lossDisS = sum(total_lossDisS) / len(total_lossDisS)

    total_train_loss = total_loss1 + total_loss2 + total_loss3 + total_loss4 + total_loss5

    print(f'Epoch: {e + 1} / {opt.epoch}, Total_loss: {total_train_loss:.3f}, Loss1: {total_loss1:.3f}, '
          f'Loss2: {total_loss2:.3f}, Loss3: {total_loss3:.3f}, Loss4: {total_loss4:.3f}, Loss5: {total_loss5:.3f}, Loss_for_DisS: {total_lossDisS:.3f}')

    train_loss_ep.append(total_train_loss)
    train_loss1_ep.append(total_loss1)
    train_loss2_ep.append(total_loss2)
    train_loss3_ep.append(total_loss3)
    train_loss4_ep.append(total_loss4)
    train_loss5_ep.append(total_loss5)
    train_lossDisS_ep.append(total_lossDisS)

    writer.add_scalar('Total train loss', total_train_loss, e)
    writer.add_scalar('Loss1', total_loss1, e)
    writer.add_scalar('Loss2', total_loss2, e)
    writer.add_scalar('Loss3', total_loss3, e)
    writer.add_scalar('Loss4', total_loss4, e)
    writer.add_scalar('Loss5', total_loss5, e)
    writer.add_scalar('Loss DisS', total_lossDisS, e)

    ### Validation
    if (e + 1) % opt.valid_save_period == 0:
        valid_save_path = f'{image_path}/epoch{e + 1}'
        valid_array_save_path = f'{temp_path}/epoch{e+1}'
        valid_plot_path = f'{plot_path}/epoch{e+1}'
        os.makedirs(valid_save_path, exist_ok=True)
        os.makedirs(valid_array_save_path, exist_ok=True)
        os.makedirs(valid_plot_path, exist_ok=True)

        modelUNet_for_S.eval()
        modelUNet_for_X.eval()
        modelDiscriminator_for_S.eval()

        dis_outputS_his = []
        dis_outputS_synthetic_his = []

        with torch.no_grad():
            for _, datas in enumerate(valid_loader):
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
                synthetic_S = synthetic_S / high_s

                img = img.to(device)
                synthetic_S = synthetic_S.to(device)

                # 1. Y'' --[Enc1]-> 1D vector S --[Dec1]-> S'
                mask_1D, mask = modelUNet_for_S(img)

                # 2. Y'' --[Enc2]-> 2D feature X --[Dec2]-> X
                output = modelUNet_for_X(img)

                # 3. X --[Enc1][Dec1]-> mask of X
                _, mask_of_X = modelUNet_for_S(output)

                # 4. {S, synthetic S} --[Discriminator_for_S]-> 0/1
                dis_outputS = modelDiscriminator_for_S(mask.clone())
                dis_outputS_synthetic = modelDiscriminator_for_S(synthetic_S.clone())

                ############# PNG Image Array Save #############
                img_grid = make_grid(img, padding=2, pad_value=1)
                synthetic_S_grid = make_grid(synthetic_S, padding=2, pad_value=1)
                mask_grid = make_grid(mask, padding=2, pad_value=1)
                mask_of_X_grid = make_grid(mask_of_X, padding=2, pad_value=1)
                output_temp = output * high
                output_temp_flat = torch.flatten(output_temp, start_dim=1, end_dim=-1)
                output_temp /= torch.quantile(output_temp_flat, 0.99, dim=1)[..., None, None, None]
                output_grid = make_grid(output_temp, padding=2, pad_value=1)
                color_grid = make_grid(torch.cat([output, img, torch.zeros_like(img)], dim=1), padding=2, pad_value=1)

                save_grid = torch.cat([img_grid, synthetic_S_grid, mask_grid, mask_of_X_grid, output_grid, color_grid], dim=1)
                save_image(save_grid, f'{valid_array_save_path}/e{e + 1}_{idx[0]}-{idx[-1]}.png')
                ################################################

                for i in range(opt.valid_batch_size):
                    valid_idx = idx[i]

                    ############# Discriminator histogram #############
                    dis_outputS_temp = dis_outputS[i].item()
                    dis_outputS_synthetic_temp = dis_outputS_synthetic[i].item()

                    dis_outputS_his.append(dis_outputS_temp)
                    dis_outputS_synthetic_his.append(dis_outputS_synthetic_temp)
                    ###################################################

                    ############# Image Save #############
                    mask_vis = mask[i].squeeze().detach().cpu().numpy()
                    io.imsave(f'{valid_save_path}/e{e+1}_{valid_idx}_maskS.tif', mask_vis)

                    synthetic_S_vis = synthetic_S[i].squeeze().detach().cpu().numpy()
                    io.imsave(f'{valid_save_path}/e{e+1}_{valid_idx}_syntheticS.tif', synthetic_S_vis)

                    mask_of_X_vis = mask_of_X[i].squeeze().detach().cpu().numpy()
                    io.imsave(f'{valid_save_path}/e{e + 1}_{valid_idx}_mask_of_X.tif', mask_of_X_vis)

                    output_vis = output[i].squeeze().detach().cpu()
                    output_vis = (output_vis * high).numpy().astype('uint16')  # denormalize using only max
                    io.imsave(f'{valid_save_path}/e{e+1}_{valid_idx}_outputX.tif', output_vis)
                    ######################################

                    ############### Plot profile ###############
                    plt.figure(figsize=(10,10))

                    plt.subplot(2, 2, 1)
                    plt.title('Mask S')
                    plt.imshow(mask_vis, cmap='gray')
                    plt.clim(0, 1)
                    plt.plot(range(opt.patch_size), [opt.patch_size//2]*opt.patch_size)

                    plt.subplot(2, 2, 2)
                    plt.title('Synthetic S')
                    plt.imshow(synthetic_S_vis, cmap='gray')
                    plt.clim(0, 1)
                    plt.plot(range(opt.patch_size), [opt.patch_size//2]*opt.patch_size)

                    plt.subplot(2, 2, 3)
                    plt.plot(range(opt.patch_size), mask_vis[opt.patch_size//2][:])
                    plt.ylim((0, 1))
                    plt.grid(True)

                    plt.subplot(2, 2, 4)
                    plt.plot(range(opt.patch_size), synthetic_S_vis[opt.patch_size // 2][:])
                    plt.ylim((0, 1))
                    plt.grid(True)

                    plt.savefig(f'{valid_plot_path}/e{e+1}_{valid_idx}.png')
                    ############################################

        ############# Discriminator histogram #############
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
    # Save loss curves [[1, 2, 3, total], [4, 5, S]]
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

    ax[1, 1].plot(epoch_ep_n, np.array(train_loss5_ep))
    ax[1, 1].set_title(f'{str(opt.d)} * Loss5')

    ax[1, 2].plot(epoch_ep_n, np.array(train_lossDisS_ep))
    ax[1, 2].set_title('Loss DisS')

    fig.supxlabel('Epoch')
    fig.supylabel('Loss')
    plt.savefig(f'{plot_path}/loss_curve.png')

    ############################################

    if (e + 1) % opt.model_save_period == 0:
        torch.save(modelUNet_for_S.state_dict(), f"{model_path}/UNet_for_S_{e+1}_{total_train_loss:.4f}.pth")
        torch.save(modelUNet_for_X.state_dict(), f"{model_path}/UNet_for_X_{e+1}_{total_train_loss:.4f}.pth")
        torch.save(modelDiscriminator_for_S.state_dict(), f"{model_path}/Discriminator_for_S_{e+1}_{total_lossDisS:.4f}.pth")