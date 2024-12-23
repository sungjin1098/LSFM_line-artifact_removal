import os

import torch
import random
import numpy as np
from skimage import io
import glob


class LineDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.img_list = glob.glob(f'{path}/*')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        img = io.imread(img_path)
        img = torch.Tensor(img.astype('float32'))
        return img


class LineDataset_3D(torch.utils.data.Dataset):
    def __init__(self, path, patch_size):
        self.img_3D = io.imread(path)
        self.patch_size = patch_size

    def __len__(self):
        return self.img_3D.shape[0]

    def __getitem__(self, item):
        img = self.img_3D[item]
        h, w = img.shape[0], img.shape[1]

        # preprocessing Y to obatin Y'
        low = np.percentile(img, 10)
        img = np.maximum(0, img-low)

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

class LineDatasetValid(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.img_list = glob.glob(f'{path}/*')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        img_name = img_path.split('/')[-1][:-4]
        img = io.imread(img_path)
        img = torch.Tensor(img.astype('float32'))
        return img, img_name
