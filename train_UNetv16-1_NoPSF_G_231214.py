import os
import torch
import argparse
import warnings
import torchvision
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from skimage import io
from torch import nn
from tqdm import tqdm
from torch.fft import fftn, ifftn, fftshift, ifftshift
from torch.utils.data import DataLoader
from torchmetrics import TotalVariation
from torchvision.utils import save_image, make_grid

from model_UNetv16 import UNet_3Plus_for_S, UNet_3Plus_for_X, Discriminator_revised,UNet_for_D
from dataloader_3Dpre import LineDataset_S_is_added, LineDataset_valid
from synthetic_dataloader_3Dpre import LineDataset_S_is_added_syn
def gen_index(input_size, patch_size, overlap_size):
    indices = []
    for k in range(2):
        z_range = list(range(0, input_size[k] - patch_size[k] + 1, overlap_size[k]))
        if input_size[k] - patch_size[k] > z_range[-1]:
            z_range.append(input_size[k] - patch_size[k])
        indices.append(z_range)
    return indices
sss=0


criterion_BCE1 = nn.BCEWithLogitsLoss(reduction='none')

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=4000, help='number of training epochs')
parser.add_argument("--valid_save_period", type=int, default=20, help='save the valid results for every N epochs')
parser.add_argument("--model_save_period", type=int, default=20, help='save the model pth file for every N epochs')
parser.add_argument("--gt_dir", type=str, default='./dataset/231030_LSM_simulation_ExcludePSF/231030_gt_110vol_ExcludePSF.tif')

parser.add_argument("--lrDisS", type=float, default=1e-6, help="Discriminator learning rate")
parser.add_argument("--lrUNetX", type=float, default=1e-6, help="UNet_3Plus learning rate")
parser.add_argument("--lrUNetS", type=float, default=1e-6, help="UNet_3Plus learning rate")
parser.add_argument("--lrSTN", type=float, default=1e-4, help="STN learning rate")

parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay of optimizer")
parser.add_argument("--patch_size", type=int, default=256, help="training patch size")
parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
parser.add_argument("--valid_batch_size", type=int, default=5, help="validation image saving batch size")


parser.add_argument("--in_dir", type=str,
                    default='./dataset/230321_6-2/6-2_20170915_LSM_zStackFish_vol20-130_crop1000-1800.tif',
                    help="dataset path")
#parser.add_argument("--in_dir", type=str,
#                    default='./dataset/input_3638.tif',   
#                    help="dataset path")
parser.add_argument ("--synthetic_S_dir", type=str, default='./dataset/synthetic_S/synthetic_S_110.tif')
parser.add_argument("--valid_dir", type=str, default='./dataset/230321_6-2/ValidDataset')
parser.add_argument("--out_dir", type=str, default='./241203', help='hyperparameters are saved at the end automatically')
parser.add_argument("--pystripe_dir", type=str,
                    default='./dataset/230321_6-2/pystripe_real_output.tif',
                    help="dataset path")
parser.add_argument("--a", type=float, default=1, help="weight for loss2 (UNet_S, TV of S should be large)")
parser.add_argument("--b", type=float, default=40, help="weight for losss3 (UNet_S, Deviation from 1 loss of mask S)")
parser.add_argument("--c", type=float, default=1, help="weight for loss3 (UNet_X)")
parser.add_argument("--d", type=float, default=1, help="weight for loss4 (UNet_S, Generator loss for S)")
parser.add_argument("--num_G", type=int, default=2, help="number of Generatsor learning iterations")

opt = parser.parse_args()

#writer = SummaryWriter(log_dir="log/{}".format(opt.out_dir[2:11]))

# NaN이 생기면 즉시 학습을 중단하고, 오류가 생긴 위치를 출력
torch.autograd.set_detect_anomaly(True)


#save path

# save path
root = f'{opt.out_dir}_SyntheticUNetv16-1_NoPSF_G{opt.num_G}_{opt.a}_{opt.b}_{opt.c}_{opt.d}_lrDS_{opt.lrDisS}_lrUX_{opt.lrUNetX}_lrUS_{opt.lrUNetS}_b{opt.batch_size}'
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
train_data = LineDataset_S_is_added(img_path=opt.in_dir, synthetic_S_path=opt.synthetic_S_dir, patch_size=opt.patch_size,pystripe = opt.pystripe_dir)
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True) 
train_loader_for_S = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
valid_data = LineDataset_valid( valid_path=opt.valid_dir, synthetic_S_path=opt.synthetic_S_dir, patch_size=opt.patch_size)
valid_loader = DataLoader(dataset=valid_data, batch_size=opt.valid_batch_size, shuffle=False)

# Define models
modelUNet_for_S = UNet_3Plus_for_S(in_channels=1, out_channels=1, feature_scale=2)
modelUNet_for_X = UNet_3Plus_for_X(in_channels=1, out_channels=1, feature_scale=4)
modelDiscriminator_for_S = Discriminator_revised(input_dim=1, dim=64)
modelDiscriminator_for_patch = UNet_for_D(in_channels=1, out_channels=1, feature_scale=4)

# If you use MULTIPLE GPUs, use 'DataParallel'
modelUNet_for_S = nn.DataParallel(modelUNet_for_S)
modelUNet_for_X = nn.DataParallel(modelUNet_for_X)
modelDiscriminator_for_S = nn.DataParallel(modelDiscriminator_for_S)
modelDiscriminator_for_patch = nn.DataParallel(modelDiscriminator_for_patch)

# Load pretrained model

modelUNet_for_S.load_state_dict(torch.load('./UNet_for_S_1520.pth'))
modelDiscriminator_for_S.load_state_dict(torch.load('./Discriminator_for_S_1520.pth'))
modelDiscriminator_for_patch.load_state_dict(torch.load('./Discriminator_for_patch_1520.pth'))
modelUNet_for_X.load_state_dict(torch.load('./UNet_for_X_1520.pth'))


# to device
modelUNet_for_S = modelUNet_for_S.to(device)
modelUNet_for_X = modelUNet_for_X.to(device)
modelDiscriminator_for_S = modelDiscriminator_for_S.to(device)
modelDiscriminator_for_patch  = modelDiscriminator_for_patch.to(device)

# Loss function for L1 loss (reconstruction loss)
criterion = nn.MSELoss().to(device)
criterion_BCE_patch = nn.BCELoss(reduction='sum').to(device)
criterion_BCE = nn.BCELoss().to(device)
criterion_L1_patch = nn.L1Loss(reduction='sum').to(device)

criterion_L1 = nn.L1Loss().to(device)
# Total variation
tv = TotalVariation(reduction='mean').to(device)
    
# optimizers
optimUNet_for_S = torch.optim.Adam(modelUNet_for_S.parameters(), lr=opt.lrUNetS, weight_decay=opt.weight_decay)
optimUNet_for_X = torch.optim.Adam(modelUNet_for_X.parameters(), lr=opt.lrUNetX, weight_decay=opt.weight_decay)
optimDiscriminator_for_S = torch.optim.Adam(modelDiscriminator_for_S.parameters(), lr=1e-4, weight_decay=opt.weight_decay)
optimDiscriminator_for_patch = torch.optim.Adam(modelDiscriminator_for_patch.parameters(), lr=1e-4, weight_decay=opt.weight_decay)

# schedulers
schedulerUNet_for_S = torch.optim.lr_scheduler.StepLR(optimUNet_for_S, step_size=50, gamma=0.5)
schedulerUNet_for_X = torch.optim.lr_scheduler.StepLR(optimUNet_for_X, step_size=50, gamma=0.5)
schedulerDiscriminator_for_S = torch.optim.lr_scheduler.StepLR(optimDiscriminator_for_S, step_size=50, gamma=0.5)
schedulerDiscriminator_for_patch = torch.optim.lr_scheduler.    StepLR(optimDiscriminator_for_patch, step_size=50, gamma=0.5)

# To draw loss graph
train_loss_ep = []
train_loss1_ep = []
train_loss2_ep = []
train_loss3_ep = []
train_loss4_ep = []
train_losspatch_ep = []
train_lossDisS_ep = []
epoch_ep = []

### loss for check if 'GT' and our 'output X' is same. ###
total_loss_check_train_ep = []
total_loss_check_valid_ep = []
total_loss2_check_train_ep = []
total_loss2_check_valid_ep = []
##########################################################

### train
for e in range(opt.epoch):
    epoch_ep.append(e + 1)

    modelUNet_for_X.train()
    modelUNet_for_S.train()
    modelDiscriminator_for_S.train()
    modelDiscriminator_for_patch.train()

    total_train_loss = 0
    total_loss1 = []
    total_loss2 = []
    total_loss3 = []
    total_loss4 = []
    total_lossDisS = []
    total_losspatch = []

    ### loss for check if 'GT' and our 'output X' is same. ###
    total_loss_check_train = 0
    total_loss_check_valid = 0
    total_loss2_check_train = 0
    total_loss2_check_valid = 0
    ##########################################################

    for ii, datas in enumerate(zip(train_loader, train_loader_for_S)):
        # Dataloader = (img, synthetic S)
        img= datas[0]
        _ = datas[1]
        # (b, h, w) --> (b, c, h, w) in this case, (b, h, w) --> (b, 1, h, w)
        img = img.unsqueeze(1)

        # Normalize using 99% percentile value (0~65525 -> 0~1.6)
        high = torch.quantile(img, 0.99)
        img = img / high

        img = img.to(device)
        mask_1D, mask = modelUNet_for_S(img)
        output = modelUNet_for_X(img)
        _, mask_of_X = modelUNet_for_S(output)

        loss1 = criterion_L1(img, mask*output)*5+criterion(img, mask*output)*5


        dis_outputS = modelDiscriminator_for_S(output)
        dis_outputS_synthetic = modelDiscriminator_for_S(torch.rot90(output, k=1, dims=[2, 3]))

        tensor_one = torch.ones_like(dis_outputS).to(device)
        tensor_zero = torch.zeros_like(dis_outputS).to(device)
        tensor_half = tensor_one*0.5
 


        loss4= (criterion_BCE(dis_outputS, tensor_half)+criterion_BCE(dis_outputS_synthetic, tensor_half))*10/2
        lossDisS= (criterion_BCE(dis_outputS_synthetic, tensor_one) + criterion_BCE(dis_outputS, tensor_zero))*10/2

        tensor_one_for_loss4 = torch.ones_like(mask_of_X).to(device)
        loss3 = criterion(mask_of_X, tensor_one_for_loss4)*2


        dis_outputpatch = modelDiscriminator_for_patch(output)
        tensor_one_for_patch = torch.ones_like(dis_outputpatch).to(device)  
        epsilon = 1e-6  # 작은 상수
        dis_outputpatch = torch.clamp(dis_outputpatch, 0 + epsilon, 1 - epsilon)


        losspatch = criterion_BCE(dis_outputpatch,mask)
        loss2 = criterion_BCE(dis_outputpatch, tensor_one_for_patch)

         ### loss for check if 'GT' and our 'output X' is same. ###  
        loss2_check_train = criterion(img,mask*output)
        total_loss2_check_train += loss2_check_train.item()
        ##########################################################

        loss_UNet_for_X = loss1+loss2+loss3+loss4
        loss_UNet_for_S = loss1+loss3
        loss_Discriminator_for_S = lossDisS
        loss_Discriminator_for_patch = losspatch

        # to print loss valuesloss_dis = loss_dis= 3.69 dds=0.871loss_dis  =
        total_loss1.append(loss1.item())
        total_loss2.append(loss2.item())
        total_loss3.append(loss3.item()) 
        total_loss4.append(loss4.item())
        total_lossDisS.append(lossDisS.item())
        total_losspatch.append(losspatch.item())

        optimUNet_for_X.zero_grad()
        optimUNet_for_S.zero_grad()

        loss_UNet_for_S.backward(retain_graph=True)

        loss_UNet_for_X.backward(retain_graph=True)
        optimUNet_for_S.step()

        optimUNet_for_X.step()

        optimDiscriminator_for_S.zero_grad()
        loss_Discriminator_for_S.backward(retain_graph=True)
        optimDiscriminator_for_S.step()

        optimDiscriminator_for_patch.zero_grad()
        loss_Discriminator_for_patch.backward(retain_graph=True)
        optimDiscriminator_for_patch.step()


    schedulerUNet_for_X.step()
    schedulerUNet_for_S.step()
    schedulerDiscriminator_for_S.step()
    schedulerDiscriminator_for_patch.step()

    total_loss1 = sum(total_loss1) / len(total_loss1)
    total_loss2 = sum(total_loss2) / len(total_loss2)
    total_loss3 = sum(total_loss3) / len(total_loss3)
    total_loss4 = sum(total_loss4) / len(total_loss4)
    total_lossDisS = sum(total_lossDisS) / len(total_lossDisS)
    total_losspatch = sum(total_losspatch) / len(total_losspatch)

    total_train_loss = total_loss1 + total_loss2 + total_loss3 + total_loss4

    print(f'Epoch: {e + 1} / {opt.epoch}, Total_loss: {total_train_loss:.3f}, Loss1: {total_loss1:.3f}', 
                f'Loss2: {total_loss2:.3f}, Losspatch: {total_losspatch:.3f}',
                f'Loss4: {total_loss4:.3f}, Loss_for_DisS: {total_lossDisS:.3f}, Loss3: {total_loss3:.3f}')

    train_loss_ep.append(total_train_loss)
    train_loss1_ep.append(total_loss1)
    train_loss2_ep.append(total_loss2)
    train_loss3_ep.append(total_loss3)
    train_loss4_ep.append(total_loss4)
    train_lossDisS_ep.append(total_lossDisS)
    train_losspatch_ep.append(total_losspatch)

    ### loss for check if 'GT' and our 'output X' is same. ###
    total_loss_check_train /= len(train_loader)
    total_loss2_check_train /= len(train_loader)
    total_loss_check_train_ep.append(total_loss_check_train)
    total_loss2_check_train_ep.append(total_loss2_check_train)
    ##########################################################

    #writer.add_scalar('Total train loss', total_train_loss, e)
    #writer.add_scalar('Loss1', total_loss1, e)
    #writer.add_scalar('Loss2', total_loss2, e)
    #writer.add_scalar('Loss3', total_loss3, e)
    #writer.add_scalar('loss3', total_loss4, e)
    #writer.add_scalar('loss4', total_loss5, e)
    #writer.add_scalar('Loss DisS', total_lossDisS, e)
    #writer.add_scalar('train MSE loss of {GT, output}', total_loss_check_train, e)
    #writer.add_scalar('train MSE loss of {input, output * mask', total_loss2_check_train, e)

    ### Validation
    if (e + 1) % opt.valid_save_period == 0:
        valid_save_path = f'{image_path}/epoch{e + 1}'
        valid_array_save_path = f'{temp_path}/epoch{e+1}'
        valid_plot_path = f'{plot_path}/epoch{e+1}'
        os.makedirs(valid_save_path, exist_ok=True)
        os.makedirs(valid_array_save_path, exist_ok=True)
        os.makedirs(valid_plot_path, exist_ok=True)

        dis_outputS_his = []
        dis_outputS_synthetic_his = []


    with torch.no_grad():
        modelUNet_for_S.eval()
        modelUNet_for_X.eval()
        modelDiscriminator_for_S.eval()
        modelDiscriminator_for_patch.eval()

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
            # (optional) to make sure that 'synthetic S' has values more than 0.1 at least.
            # synthetic_S = torch.maximum(synthetic_S, torch.ones_like(synthetic_S) * 0.1)

            img = img.to(device)
            synthetic_S = synthetic_S.to(device)

            # 1. Y'' --[Enc1]-> 1D vector S --[Dec1]-> S'
            mask_1D, mask = modelUNet_for_S(img)
            # 2. Y'' --[Enc2]-> 2D feature X --[Dec2]-> X
            output= modelUNet_for_X(img)

            # 3. X --[Enc1][Dec1]-> mask of X
            _, mask_of_X = modelUNet_for_S(output)

            # 4. {S, synthetic S} --[Discriminator_for_S]-> 0/1
            dis_outputS = modelDiscriminator_for_S(output.clone())
            dis_outputS_synthetic = modelDiscriminator_for_S(synthetic_S.clone())
            dis_outputpatch = modelDiscriminator_for_patch(output.clone())

            ### loss for check if 'GT' and our 'output X' is same. ###
            loss2_check_valid = criterion(img, mask * output)
            total_loss2_check_valid += loss2_check_valid.item()
            ##########################################################

            if (e + 1) % opt.valid_save_period == 0:
                ############# PNG Image Array Save #############
                img_grid = make_grid(img, padding=2, pad_value=1)
                mask_grid = make_grid(mask, padding=2, pad_value=1)
                mask_of_X_grid = make_grid(dis_outputpatch, padding=2, pad_value=1)

                output_temp = output * high
                output_temp_flat = torch.flatten(output_temp, start_dim=1, end_dim=-1)
                output_temp /= torch.quantile(output_temp_flat, 0.99, dim=1)[..., None, None, None]
                output_grid = make_grid(output_temp, padding=2, pad_value=1)

                save_grid = torch.cat([img_grid, mask_grid, mask_of_X_grid, output_grid], dim=1)
                save_image(save_grid, f'{valid_array_save_path}/e{e + 1}_{idx[0]}-{idx[-1]}.png')
                ################################################

                for i in range(opt.valid_batch_size):
                    valid_idx = idx[i]

                    ############# Discriminator histogram #############
                    dis_outputS_temp = torch.mean(dis_outputS[i]).item()
                    dis_outputS_synthetic_temp = torch.mean(dis_outputS_synthetic[i]).item()

                    dis_outputS_his.append(dis_outputS_temp)
                    dis_outputS_synthetic_his.append(dis_outputS_synthetic_temp)
                    ###################################################

                    ############# Image Save #############
                    mask_vis = mask[i].squeeze().detach().cpu().numpy()
                    io.imsave(f'{valid_save_path}/e{e+1}_{valid_idx}_maskS.tif', mask_vis)
                    
                    mask_of_X_vis = mask_of_X[i].squeeze().detach().cpu().numpy()
                    io.imsave(f'{valid_save_path}/e{e + 1}_{valid_idx}_mask_of_X.tif', mask_of_X_vis)
                    
                    mask_patch = dis_outputpatch[i].squeeze().detach().cpu().numpy()
                    io.imsave(f'{valid_save_path}/e{e+1}_{valid_idx}_mask_patch_dis.tif', mask_patch)

                    
                    output_vis1 = output[i].squeeze().detach().cpu()
                    output_vis1 = (output_vis1 * high).numpy().astype('uint16')  # denormalize using only max
                    
                    io.imsave(f'{valid_save_path}/e{e+1}_{valid_idx}_{mask_score[i]}_outputX1.tif', output_vis1)


                    ############### Plot profile ###############

                    ##############

        ### loss for check if 'GT' and our 'output X' is same. ###
        total_loss_check_valid /= len(valid_loader)
        total_loss2_check_valid /= len(valid_loader)
        total_loss_check_valid_ep.append(total_loss_check_valid)
        total_loss2_check_valid_ep.append(total_loss2_check_valid)
        ##########################################################

        if (e + 1) % opt.valid_save_period == 0:
            ############# Discriminator histogram #############
            # plt.figure()
            # plt.hist(dis_outputX_his, alpha=0.5, label='X (0)', range=(0, 1), bins=20)
            # plt.hist(dis_outputX_rotated_his, alpha=0.5, label='rotated X (1)', range=(0, 1), bins=20)
            # plt.title('Discriminator output of {X, rotated X}')
            # plt.xlabel('Discriminator output value (0~1)')
            # plt.ylabel('# of images')
            # plt.legend()
            # plt.savefig(f'{valid_array_save_path}/e{e+1}_histogram_X.png')

            plt.figure()
            plt.hist(dis_outputS_his, alpha=0.5, label='S (0)', range=(0, 1), bins=20)
            plt.hist(dis_outputS_synthetic_his, alpha=0.5, label='synthetic S (1)', range=(0, 1), bins=20)
            plt.title('Discriminator output of {S, synthetic S}')
            plt.xlabel('Discriminator output value (0~1)')
            plt.ylabel('# of images')
            plt.legend()
            plt.savefig(f'{valid_array_save_path}/e{e + 1}_histogram_S.png')
            ###################################################

    #writer.add_scalar('valid MSE loss of {GT, output}', total_loss_check_valid, e)ss e)

    ############################################
    # Save loss curves [[1, 2, 3, total], [4, 5, S]]

    epoch_ep_n = np.array(epoch_ep)
    fig, ax = plt.subplots(2, 5, figsize=(20, 8), layout='constrained')

    ax[0, 0].plot(epoch_ep_n, np.array(train_loss1_ep))
    ax[0, 0].set_title(f'Loss1')

    ax[0, 1].plot(epoch_ep_n, np.array(train_loss2_ep))
    ax[0, 1].set_title(f'Loss2')

    ax[0, 2].plot(epoch_ep_n, np.array(train_losspatch_ep))
    ax[0, 2].set_title(f'Losspatch')

    ax[0, 3].plot(epoch_ep_n, np.array(train_losspatch_ep),color='red')
    ax[0, 3].plot(epoch_ep_n, np.array(train_loss2_ep),color='blue')
    ax[0, 3].set_title('overlay')
    
    ax[1, 0].plot(epoch_ep_n, np.array(train_loss3_ep))
    ax[1, 0].set_title(f'{str(opt.c)} * loss3')

    ax[1, 1].plot(epoch_ep_n, np.array(train_loss4_ep))
    ax[1, 1].set_title(f'{str(opt.d)} * loss4')

    ax[1, 2].plot(epoch_ep_n, np.array(train_lossDisS_ep))
    ax[1, 2].set_title('Loss DisS')

    ax[1, 3].plot(epoch_ep_n, np.array(train_lossDisS_ep),color='red')
    ax[1, 3].plot(epoch_ep_n, np.array(train_loss4_ep),color='blue')
    ax[1, 3].set_title('overlay')


    fig.supxlabel('Epoch')
    fig.supylabel('Loss')
    plt.savefig(f'{plot_path}/loss_curve.png')

    ### loss for check if 'GT' and our 'output X' is same. ###

    plt.figure()
    plt.plot(epoch_ep_n, np.array(total_loss2_check_train_ep), label='train loss')
    plt.plot(epoch_ep_n, np.array(total_loss2_check_valid_ep), label='valid loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MSE loss of {input, output * mask}')
    plt.legend(loc='upper right')
    plt.savefig(f'{plot_path}/check_loss2.png')
    ##########################################################

