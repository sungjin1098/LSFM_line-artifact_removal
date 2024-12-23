import os
import glob
import torch
import random
import numpy as np
from skimage import io


class LineDataset(torch.utils.data.Dataset):
    def __init__(self, path, patch_size):
        self.img_3D = io.imread(path)
        self.patch_size = patch_size

    def __len__(self):
        return self.img_3D.shape[0]

    def __getitem__(self, item):
        img = self.img_3D[item]
        h, w = img.shape[0], img.shape[1]

        # # preprocessing Y to obatin Y'
        # low = np.percentile(img, 10)
        # img = np.maximum(0, img-low)

        bool = True
        while bool == True:
            # random crop
            # random crop
            left = random.randint(0, w - self.patch_size)
            top = random.randint(0, h - self.patch_size)
            patch = img[top:top + self.patch_size, left:left + self.patch_size]
            if patch.sum() != 0:
                bool = False

        patch = torch.Tensor(patch.astype('float32'))

        return patch

class LineDataset_S_is_added(torch.utils.data.Dataset):
    def __init__(self, img_path, synthetic_S_path, patch_size,pystripe):
        self.img_3D = io.imread(img_path)
        self.synthetic_3D = io.imread(synthetic_S_path)
        self.patch_size = patch_size
        self.pystripe = io.imread(pystripe)

    def __len__(self):
        return self.img_3D.shape[0]

    def __getitem__(self, item):
        img = self.img_3D[item]
        h, w = img.shape[0], img.shape[1]
        #i=0
        #i=0

        # # preprocessing Y to obatin Y'
        # low = np.percentile(img, 10)
        # img = np.maximum(0, img-low)
        while True:
            # random crop
            left = random.randint(0, w - self.patch_size)
            top = random.randint(520, h - self.patch_size)
            patch = img[top:top + self.patch_size, left:left + self.patch_size]
            if patch.sum() < 1000000:
                continue   
            #io.imsave(f'./230612_exp/loss_curve/outputX{patch.sum()}.tif', patch)


            break

        patch = torch.Tensor(patch.astype('float32'))

        return patch 


class LineDataset_S_is_added_Multiple_updates(torch.utils.data.Dataset):
    def __init__(self, img_path, synthetic_S_path, patch_size, num_update):
        self.img_3D = io.imread(img_path)
        self.synthetic_3D = io.imread(synthetic_S_path)
        self.patch_size = patch_size
        self.num_update = num_update

    def __len__(self):
        return self.img_3D.shape[0] * self.num_update

    def __getitem__(self, item):
        img = self.img_3D[item//self.num_update]
        h, w = img.shape[0], img.shape[1]
        synthetic_s = self.synthetic_3D[item//self.num_update]
        sh, sw = synthetic_s.shape[0], synthetic_s.shape[1]

        # # preprocessing Y to obatin Y'
        # low = np.percentile(img, 10)
        # img = np.maximum(0, img - low)

        while True:
            # random crop
            print(h)
            left = random.randint(0, w - self.patch_size)
            top = random.randint(0, h - self.patch_size)
            patch = img[top:top + self.patch_size, left:left + self.patch_size]

            #if patch.sum() == 0:
            #    continue

            top = random.randint(0, sw - self.patch_size)
            left = random.randint(0, sh - self.patch_size)
            patch_s = synthetic_s[left:left + self.patch_size, top:top + self.patch_size]

            break

        patch = torch.Tensor(patch.astype('float32'))
        patch_s = torch.Tensor(patch_s.astype('float32'))

        return patch, patch_s



class LineDataset_valid(torch.utils.data.Dataset):
    def __init__(self, valid_path, synthetic_S_path, patch_size):
        self.valid_path = valid_path
        self.paths = sorted(os.listdir(valid_path))
        self.synthetic_3D = io.imread(synthetic_S_path)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        index = path[:3]
        img = io.imread(f'{self.valid_path}/{path}')

        synthetic_s = self.synthetic_3D[int(item//3)]
        sh, sw = synthetic_s.shape[0], synthetic_s.shape[1]

        # # preprocessing Y to obatin Y'
        # low = np.percentile(img, 10)
        # img = np.maximum(0, img-low)

        # random crop
        top = random.randint(0, sw - self.patch_size)
        left = random.randint(0, sh - self.patch_size)
        patch_s = synthetic_s[left:left + self.patch_size, top:top + self.patch_size]
        img = torch.Tensor(img.astype('float32'))

        patch_s = torch.Tensor(patch_s.astype('float32'))

        return img, patch_s, index