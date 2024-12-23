import os
import torch
import random
import pickle
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
            left = random.randint(0, w - self.patch_size)
            top = random.randint(0, h - self.patch_size)
            patch = img[top:top + self.patch_size, left:left + self.patch_size]
            if patch.sum() != 0:
                bool = False

        patch = torch.Tensor(patch.astype('float32'))

        return patch


class LineDataset_S_is_added_syn(torch.utils.data.Dataset):
    def __init__(self, img_path, synthetic_S_path, gt_path, patch_size):
        self.img_3D = io.imread(img_path)
        self.synthetic_3D = io.imread(synthetic_S_path)
        self.gt_3D = io.imread(gt_path)
        self.patch_size = patch_size

    def __len__(self):
        return self.img_3D.shape[0]

    def __getitem__(self, item):
        # the shape of img and synthetic_s should be same.
        img = self.img_3D[item]
        synthetic_s = self.synthetic_3D[item]
        gt = self.gt_3D[item]
        h, w = img.shape[0], img.shape[1]

        # # preprocessing Y to obatin Y'
        # low = np.percentile(img, 10)
        # img = np.maximum(0, img-low)

        while True:
            # random crop
            left = random.randint(0, w - self.patch_size)
            top = random.randint(0, h - self.patch_size)
            patch = img[top:top + self.patch_size, left:left + self.patch_size]
            patch_s = synthetic_s[top:top + self.patch_size, left:left + self.patch_size]
            patch_gt = gt[top:top + self.patch_size, left:left + self.patch_size]

            if patch.sum() == 0:
                continue

            break

        patch = torch.Tensor(patch.astype('float32'))
        patch_s = torch.Tensor(patch_s.astype('float32'))
        patch_gt = torch.Tensor(patch_gt.astype('float32'))

        return patch, patch_s, patch_gt


class LineDataset_S_is_added_Multiple_updates(torch.utils.data.Dataset):
    def __init__(self, img_path, synthetic_S_path, gt_path, patch_size, num_update):
        self.img_3D = io.imread(img_path)
        self.synthetic_3D = io.imread(synthetic_S_path)
        self.gt_3D = io.imread(gt_path)
        self.patch_size = patch_size
        self.num_update = num_update

    def __len__(self):
        return self.img_3D.shape[0] * self.num_update

    def __getitem__(self, item):
        # the shape of img and synthetic_s should be same.
        img = self.img_3D[item//self.num_update]
        synthetic_s = self.synthetic_3D[item//self.num_update]
        gt = self.gt_3D[item//self.num_update]
        h, w = img.shape[0], img.shape[1]

        # # preprocessing Y to obatin Y'
        # low = np.percentile(img, 10)
        # img = np.maximum(0, img - low)

        while True:
            # random crop
            left = random.randint(0, w - self.patch_size)
            top = random.randint(0, h - self.patch_size)
            patch = img[top:top + self.patch_size, left:left + self.patch_size]
            patch_s = synthetic_s[top:top + self.patch_size, left:left + self.patch_size]
            patch_gt = gt[top:top + self.patch_size, left:left + self.patch_size]

            if patch.sum() == 0:
                continue

            break

        patch = torch.Tensor(patch.astype('float32'))
        patch_s = torch.Tensor(patch_s.astype('float32'))
        patch_gt = torch.Tensor(patch_gt.astype('float32'))

        return patch, patch_s, patch_gt

class LineDataset_valid(torch.utils.data.Dataset):
    def __init__(self, index_path, valid_path, synthetic_S_path, gt_path, patch_size):
        self.img_3D = io.imread(valid_path)
        self.synthetic_3D = io.imread(synthetic_S_path)
        self.gt_3D = io.imread(gt_path)
        self.patch_size = patch_size
        with open(index_path, 'rb') as file:
            self.index_list = pickle.load(file)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, item):
        index, c, h, w = self.index_list[item]

        index = str(index).zfill(3)
        img = self.img_3D[c, h:h + self.patch_size, w:w + self.patch_size]
        patch_s = self.synthetic_3D[c, h:h + self.patch_size, w:w + self.patch_size]
        gt = self.gt_3D[c, h:h + self.patch_size, w:w + self.patch_size]

        # # preprocessing Y to obatin Y'
        # low = np.percentile(img, 10)
        # img = np.maximum(0, img-low)

        img = torch.Tensor(img.astype('float32'))
        patch_s = torch.Tensor(patch_s.astype('float32'))
        gt = torch.Tensor(gt.astype('float32'))

        return img, patch_s, gt, index