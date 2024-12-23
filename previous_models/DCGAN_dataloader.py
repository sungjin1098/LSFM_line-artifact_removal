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

        # preprocessing Y to obatin Y'
        low = np.percentile(img, 10)
        img = np.maximum(0, img-low)

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


class LineDataset_S_is_added_for_S(torch.utils.data.Dataset):
    def __init__(self, img_path, synthetic_S_path, patch_size):
        self.img_3D = io.imread(img_path)
        self.synthetic_3D = io.imread(synthetic_S_path)
        self.patch_size = patch_size

    @staticmethod
    def random_horizontal_flip(a, b):
        # probability 0.5
        if random.random() < 0.5:
            # stride를 -1로 해서 역정렬
            a = a[:, ::-1].copy()
            b = b[:, ::-1].copy()
        return a, b

    # random vertical flip
    @staticmethod
    def random_vertical_flip(a, b):
        if random.random() < 0.5:
            a = a[::-1, :].copy()
            b = b[::-1, :].copy()
        return a, b

    # random rotate 90 degrees
    @staticmethod
    def random_rotate(x, rot_times):
        x = np.rot90(x, rot_times, axes=(1, 0)).copy()
        return x

    @staticmethod
    def random_rotate_90(a, b):
        if random.random() < 0.5:
            a = LineDataset_S_is_added.random_rotate(a, 2)
            b = LineDataset_S_is_added.random_rotate(b, 2)
        return a, b


    # def random_gaussian(self, a):
    #     std = random.uniform(0, 0.1)
    #     noise = np.random.normal(0, std, (self.patch_size, self.patch_size))
    #     a = a + noise
    #     a = np.clip(0, 1, a)
    #     return a
    #
    # @staticmethod
    # def random_brightness(a):
    #     alpha = random.uniform(-0.2, 0.2)
    #     a = (1 + alpha) * a
    #     a = np.clip(0, 1, a)
    #     return a

    def __len__(self):
        return self.img_3D.shape[0]

    def __getitem__(self, item):
        img = self.img_3D[item]
        h, w = img.shape[0], img.shape[1]
        synthetic_s = self.synthetic_3D[item]
        sh, sw = synthetic_s.shape[0], synthetic_s.shape[1]

        # preprocessing Y to obatin Y'
        low = np.percentile(img, 10)
        img = np.maximum(0, img-low)

        while True:
            # random crop
            left = random.randint(0, w - self.patch_size)
            top = random.randint(0, h - self.patch_size)
            patch = img[top:top + self.patch_size, left:left + self.patch_size]
            top = random.randint(0, sw - self.patch_size)
            left = random.randint(0, sh - self.patch_size)
            patch_s = synthetic_s[left:left + self.patch_size, top:top + self.patch_size]

            break

        # data augmentation
        patch, patch_s = self.random_horizontal_flip(patch, patch_s)
        patch, patch_s = self.random_vertical_flip(patch, patch_s)
        patch, patch_s = self.random_rotate_90(patch, patch_s)
        # patch = self.random_gaussian(patch)
        # patch = self.random_brightness(patch)

        patch = torch.Tensor(patch.astype('float32'))
        patch_s = torch.Tensor(patch_s.astype('float32'))

        return patch

class LineDataset_S_is_added(torch.utils.data.Dataset):
    def __init__(self, img_path, synthetic_S_path, patch_size):
        self.img_3D = io.imread(img_path)
        self.synthetic_3D = io.imread(synthetic_S_path)
        self.patch_size = patch_size

    @staticmethod
    def random_horizontal_flip(a, b):
        # probability 0.5
        if random.random() < 0.5:
            # stride를 -1로 해서 역정렬
            a = a[:, ::-1].copy()
            b = b[:, ::-1].copy()
        return a, b

    # random vertical flip
    @staticmethod
    def random_vertical_flip(a, b):
        if random.random() < 0.5:
            a = a[::-1, :].copy()
            b = b[::-1, :].copy()
        return a, b

    # random rotate 90 degrees
    @staticmethod
    def random_rotate(x, rot_times):
        x = np.rot90(x, rot_times, axes=(1, 0)).copy()
        return x

    @staticmethod
    def random_rotate_90(a, b):
        if random.random() < 0.5:
            a = LineDataset_S_is_added.random_rotate(a, 2)
            b = LineDataset_S_is_added.random_rotate(b, 2)
        return a, b


    # def random_gaussian(self, a):
    #     std = random.uniform(0, 0.1)
    #     noise = np.random.normal(0, std, (self.patch_size, self.patch_size))
    #     a = a + noise
    #     a = np.clip(0, 1, a)
    #     return a
    #
    # @staticmethod
    # def random_brightness(a):
    #     alpha = random.uniform(-0.2, 0.2)
    #     a = (1 + alpha) * a
    #     a = np.clip(0, 1, a)
    #     return a

    def __len__(self):
        return self.img_3D.shape[0]

    def __getitem__(self, item):
        img = self.img_3D[item]
        h, w = img.shape[0], img.shape[1]
        synthetic_s = self.synthetic_3D[item]
        sh, sw = synthetic_s.shape[0], synthetic_s.shape[1]

        # preprocessing Y to obatin Y'
        low = np.percentile(img, 10)
        img = np.maximum(0, img-low)

        while True:
            # random crop
            left = random.randint(0, w - self.patch_size)
            top = random.randint(0, h - self.patch_size)
            patch = img[top:top + self.patch_size, left:left + self.patch_size]

            if patch.mean() <= 20:
                continue

            top = random.randint(0, sw - self.patch_size)
            left = random.randint(0, sh - self.patch_size)
            patch_s = synthetic_s[left:left + self.patch_size, top:top + self.patch_size]

            break

        # data augmentation
        patch, patch_s = self.random_horizontal_flip(patch, patch_s)
        patch, patch_s = self.random_vertical_flip(patch, patch_s)
        patch, patch_s = self.random_rotate_90(patch, patch_s)
        # patch = self.random_gaussian(patch)
        # patch = self.random_brightness(patch)

        patch = torch.Tensor(patch.astype('float32'))
        patch_s = torch.Tensor(patch_s.astype('float32'))

        return patch


class LineDataset_S_is_added_Multiple_updates(torch.utils.data.Dataset):
    def __init__(self, img_path, synthetic_S_path, patch_size, num_update):
        self.img_3D = io.imread(img_path)
        self.synthetic_3D = io.imread(synthetic_S_path)
        self.patch_size = patch_size
        self.num_update = num_update

    @staticmethod
    def random_horizontal_flip(a, b):
        # probability 0.5
        if random.random() < 0.5:
            # stride를 -1로 해서 역정렬
            a = a[:, ::-1].copy()
            b = b[:, ::-1].copy()
        return a, b

    # random vertical flip
    @staticmethod
    def random_vertical_flip(a, b):
        if random.random() < 0.5:
            a = a[::-1, :].copy()
            b = b[::-1, :].copy()
        return a, b

    # random rotate 90 degrees
    @staticmethod
    def random_rotate(x, rot_times):
        x = np.rot90(x, rot_times, axes=(1, 0)).copy()
        return x

    @staticmethod
    def random_rotate_90(a, b):
        if random.random() < 0.5:
            a = LineDataset_S_is_added_Multiple_updates.random_rotate(a, 2)
            b = LineDataset_S_is_added_Multiple_updates.random_rotate(b, 2)
        return a, b

    # def random_gaussian(self, a):
    #     std = random.uniform(0, 0.1)
    #     noise = np.random.normal(0, std, (self.patch_size, self.patch_size))
    #     a = a + noise
    #     a = np.clip(0, 1, a)
    #     return a
    #
    # @staticmethod
    # def random_brightness(a):
    #     alpha = random.uniform(-0.2, 0.2)
    #     a = (1 + alpha) * a
    #     a = np.clip(0, 1, a)
    #     return a

    def __len__(self):
        return self.img_3D.shape[0] * self.num_update

    def __getitem__(self, item):
        img = self.img_3D[item//self.num_update]
        h, w = img.shape[0], img.shape[1]
        synthetic_s = self.synthetic_3D[item//self.num_update]
        sh, sw = synthetic_s.shape[0], synthetic_s.shape[1]

        # preprocessing Y to obatin Y'
        low = np.percentile(img, 10)
        img = np.maximum(0, img - low)

        while True:
            # random crop
            left = random.randint(0, w - self.patch_size)
            top = random.randint(0, h - self.patch_size)
            patch = img[top:top + self.patch_size, left:left + self.patch_size]

            if patch.mean() <= 20:
                continue

            top = random.randint(0, sw - self.patch_size)
            left = random.randint(0, sh - self.patch_size)
            patch_s = synthetic_s[left:left + self.patch_size, top:top + self.patch_size]

            break

        # data augmentation
        patch, patch_s = self.random_horizontal_flip(patch, patch_s)
        patch, patch_s = self.random_vertical_flip(patch, patch_s)
        patch, patch_s = self.random_rotate_90(patch, patch_s)
        # patch = self.random_gaussian(patch)
        # patch = self.random_brightness(patch)

        patch = torch.Tensor(patch.astype('float32'))
        patch_s = torch.Tensor(patch_s.astype('float32'))

        return patch



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

        # preprocessing Y to obatin Y'
        low = np.percentile(img, 10)
        img = np.maximum(0, img-low)

        # random crop
        top = random.randint(0, sw - self.patch_size)
        left = random.randint(0, sh - self.patch_size)
        patch_s = synthetic_s[left:left + self.patch_size, top:top + self.patch_size]

        img = torch.Tensor(img.astype('float32'))
        patch_s = torch.Tensor(patch_s.astype('float32'))

        return img, patch_s, index