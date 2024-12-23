import torch
import random
import numpy as np
from skimage import io


class LineDataset(torch.utils.data.Dataset):
    def __init__(self, path, patch_size):
        self.path = path
        self.img_3D = io.imread(path)
        self.patch_size = patch_size

    def __len__(self):
        return self.img_3D.shape[0]

    def __getitem__(self, item):
        img = self.img_3D[item]
        h, w = img.shape[0], img.shape[1]

        bool = True
        while bool == True:
            # random crop
            top = random.randint(0, w - self.patch_size)
            left = random.randint(0, h - self.patch_size)
            patch = img[left:left + self.patch_size, top:top + self.patch_size]
            if patch.sum() != 0:
                bool = False

        patch = torch.Tensor(patch.astype('float32'))

        return patch
