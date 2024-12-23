import os

from skimage import io
import torch
import numpy as np
# from dataloader_3D import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


def gen_index(input_size, patch_size, overlap_size):
    # B, C, W, H = input_size
    indices = []
    for k in range(2):
        z_range = list(range(0, input_size[k] - patch_size[k] + 1, overlap_size[k]))
        if input_size[k] - patch_size[k] > z_range[-1]:
            z_range.append(input_size[k] - patch_size[k])
        indices.append(z_range)
    return indices


# ### Generate valid patches to test discriminator (patch 256x256, NO overlap)
# img_3D = io.imread('./dataset/Data_for_discriminator/5_raw_211002_SCAPE_casperGFP_4dpf_S1_R2_801planex1vol.tif')
# index = gen_index(img_3D.shape[1:], [256, 256], [256, 256])
#
# for i in tqdm(range(img_3D.shape[0])):
#     img_2D = img_3D[i]
#     for xi in range(len(index[0])):
#         for yi in range(len(index[1])):
#             patch = img_2D[index[0][xi]:index[0][xi]+256, index[1][yi]:index[1][yi]+256]
#             order = xi * len(index[1]) + yi
#             io.imsave(f'./dataset/Data_for_discriminator/LineDataForDiscriminator/slice{i}_{order}.tif', patch)


# ### Generate valid patches (patch 256x256, overlap 128)
# img_3D = io.imread('D:/Data_line_artifact/6-1_20170915_LSM_GCaMPfish_singlePlane_3.tif')
#
# # 6-1 데이터의 0-th volume, {222}th~{222+768}th column, {94}th~{94+1024}th row
# img_2D = img_3D[0, 222:222+768, 94:94+1024]
# index = gen_index(img_2D.shape, [256, 256], [128, 128])
#
# for xi in range(len(index[0])):
#     for yi in range(len(index[1])):
#         patch = img_2D[index[0][xi]:index[0][xi]+256, index[1][yi]:index[1][yi]+256]
#         order = xi * len(index[1]) + yi
#         io.imsave(f'./dataset/230321_6-2/ValidDataset/6-1_{order}.tif', patch)


# ### Generate valid patches (patch 256x256, no overlap)
# img_3D = io.imread('./dataset/230321_6-2/6-2_20170915_LSM_zStackFish_vol20-130_crop1000-1800.tif')
#
# h_origin, w_origin = img_3D.shape[1:]
#
# for i in range(110):
#     for j in range(3):
#         while True:
#             h = random.randint(0, h_origin-256)
#             w = random.randint(0, w_origin-256)
#             patch = img_3D[i, h:h+256, w:w+256]
#             if np.mean(patch) > 30:
#                 break
#         idx = str(i*3 + j).zfill(3)
#         io.imsave(f'./dataset/230321_6-2/ValidDataset2/{idx}_c{i}_h{h}_w{w}.tif', patch)


# #### 2048x2048 이미지를 1150x1900으로 만들기
# dir = './dataset/231216_LSM_simulation_ExcludePSF'
# for name in ['synthetic_S', 'gt', 'input_Y']:
#     img_3D = io.imread(f'{dir}/231216_{name}_110vol_ExcludePSF.tif')
#     img_3D_crop = img_3D[:, 449:-449, 148:]
#     io.imsave(f'{dir}/231216_{name}_110vol_ExcludePSF.tif', img_3D_crop)


# ## Valid dataset을 위한 index list 만들기
# img_3D = io.imread('./dataset/231216_LSM_simulation_ExcludePSF/231216_gt_110vol_ExcludePSF.tif')
# h, w = img_3D.shape[1:]
#
# import pickle
# import random
#
# temp = []
# for i in range(110):
#     j = 0
#     while j < 3:
#         idx = 3*i + j
#         top = random.randint(0, h - 256)
#         left = random.randint(0, w - 256)
#         img = img_3D[i, top:top+256, left:left+256]
#         if np.mean(img) < 1200:
#             continue
#         temp.append([idx, i, top, left])
#         j += 1
#
# file_name = './dataset/231216_LSM_simulation_ExcludePSF/index_list.txt'
# with open(file_name, 'wb') as file:
#     pickle.dump(temp, file)


# ### Index list를 이용해서, valid input이랑 valid gt는 따로 미리 저장해두기
# import pickle
# index_path = './dataset/231216_LSM_simulation_ExcludePSF/index_list.txt'
# img_3D = io.imread('./dataset/231216_LSM_simulation_ExcludePSF/231216_input_Y_110vol_ExcludePSF.tif')
# gt_3D = io.imread('./dataset/231216_LSM_simulation_ExcludePSF/231216_gt_110vol_ExcludePSF.tif')
# patch_size = 256
#
# with open(index_path, 'rb') as file:
#     index_list = pickle.load(file)
#
# for content in index_list:
#     index, c, h, w = content
#
#     index = str(index).zfill(3)
#     img = img_3D[c, h:h + patch_size, w:w + patch_size]
#     gt = gt_3D[c, h:h + patch_size, w:w + patch_size]
#
#     io.imsave(f'./dataset/231216_LSM_simulation_ExcludePSF/valid_input/{index}_c{c}_h{h}_w{w}.tif', img)
#     io.imsave(f'./dataset/231216_LSM_simulation_ExcludePSF/valid_gt/{index}_c{c}_h{h}_w{w}.tif', gt)


# ### Index list를 이용해서, testGAN 이미지 저장하기
# import pickle
# index_path = './dataset/230817_LSM_simulation/index_list.txt'
# img_3D = io.imread('./dataset/230919_testGAN/230919_input_220vol.tif')
# # gt_3D = io.imread('./dataset/230817_LSM_simulation/230817_gt_110vol.tif')
# patch_size = 256
#
# with open(index_path, 'rb') as file:
#     index_list = pickle.load(file)
#
# for c in range(len(index_list)):
#     if c == 110:
#         break
#     content = index_list[c]
#     index, c, h, w = content
#
#     index_img = str(index).zfill(3)
#     index_noisy = str(index+110).zfill(3)
#     img = img_3D[c, h:h + patch_size, w:w + patch_size]
#     # gt = gt_3D[c, h:h + patch_size, w:w + patch_size]
#     img_noisy = img_3D[c+110, h:h+patch_size, w:w+patch_size]
#
#     io.imsave(f'./dataset/230919_testGAN/valid_input/{index}_c{c}_h{h}_w{w}.tif', img)
#     io.imsave(f'./dataset/230919_testGAN/valid_input/{index_noisy}_c{c+110}_h{h}_w{w}.tif', img_noisy)

# ### 파일 이름 바꾸기
# dir = 'D:/210827_LSM_simulation/S_for_real_training'
# img_list = os.listdir(dir)
#
# for i in img_list:
#     if i[0] == '0':
#         continue
#     f = f'{dir}/{i}'
#     os.rename(f, f'{dir}/04_{i}')

### Calculate MSE
# img_dir = 'D:/230118_Line_Artifact_Removal/dataset/231030_LSM_simulation_ExcludePSF/valid_input'
# gt_dir = 'D:/230118_Line_Artifact_Removal/dataset/231030_LSM_simulation_ExcludePSF/valid_gt'
#
# img_list = sorted(os.listdir(img_dir))
# gt_list = sorted(os.listdir(gt_dir))
#
# MSE = nn.MSELoss()
#
# total = 0
# cnt = 0
# for i, g in zip(img_list, gt_list):
#     if cnt == 5:
#         break
#     img = io.imread(f'{img_dir}/{i}')
#     gt = io.imread(f'{gt_dir}/{g}')
#
#     img = torch.Tensor(img.astype('float32')).unsqueeze(1)
#     gt = torch.Tensor(gt.astype('float32')).unsqueeze(1)
#
#     high = torch.quantile(img, 0.99)
#     img = img / high
#     high_gt = torch.quantile(gt, 0.99)
#     gt = gt / high_gt
#
#     total += MSE(img, gt)
#     cnt += 1
#
# print(total)
# print(total / cnt)


# ### 3D tif --> 2D tifs
# from tqdm import tqdm
#
# img_path = 'D:/230118_Line_Artifact_Removal/My_result_Removing_stripes/data0321_input.tif'
# save_dir = 'D:/230118_Line_Artifact_Removal/My_result_Removing_stripes/data0321_input_Y'
# os.makedirs(save_dir, exist_ok=True)
#
# img_3D = io.imread(img_path)
#
# for i in tqdm(range(img_3D.shape[0])):
#     img_2D = img_3D[i]
#
#     io.imsave(f'{save_dir}/{str(i).zfill(3)}.tif', img_2D)


# ### 3D tif --> 2D tifs with rotate90 left
# from tqdm import tqdm
# import numpy as np
#
# img_path = 'D:/230118_Line_Artifact_Removal/My_result_Removing_stripes/231216_input_Y_110vol_ExcludePSF.tif'
# save_dir = 'D:/230118_Line_Artifact_Removal/My_result_PyStripe/data1216_input_Y_rotate'
# os.makedirs(save_dir, exist_ok=True)
#
# img_3D = io.imread(img_path)
#
# for i in tqdm(range(img_3D.shape[0])):
#     img_2D = img_3D[i]
#     img_2D = np.rot90(img_2D)
#
#     io.imsave(f'{save_dir}/{str(i).zfill(3)}.tif', img_2D)