import torch.utils.data as data
import os.path
import cv2
import numpy as np
from data import common
from utils import base_path


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]

def npy_loader(path):
    return np.load(path)

IMG_EXTENSIONS = [
    '.png', '.npy',
]

# base_path = os.path.dirname(os.path.abspath('train.py'))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    dir = base_path + dir
    print(dir)
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class div2k(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt.scale
        self.root = os.path.join(base_path, self.opt.root)
        self.ext = self.opt.ext   # '.png' or '.npy'(default)
        self.train = True if self.opt.phase == 'train' else False
        self.repeat = 10#self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan()
        # Set standard size for patches
        self.standard_size = (self.opt.patch_size // self.scale) * self.scale

    def _set_filesystem(self, dir_data):
        self.root = dir_data + '/DF2K_decoded'
        self.dir_hr = os.path.join(self.root, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.root, 'DIV2K_train_LR_bicubic/X' + str(self.scale))

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.opt.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=self.opt.rgb_range)
        return lr_tensor, hr_tensor

    def __len__(self):
        if self.train:
            return self.opt.n_train * self.repeat

    def _get_index(self, idx):
        if self.train:
            return idx % self.opt.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar):
        patch_size = self.opt.patch_size
        scale = self.scale
        if self.train:
            # Get standardized patch
            input_size = patch_size // scale
            
            ih, iw = img_in.shape[:2]
            
            # Make sure we can get a valid patch
            if ih < input_size or iw < input_size:
                # Pad the image if too small
                pad_h = max(0, input_size - ih)
                pad_w = max(0, input_size - iw)
                padding = ((0, pad_h), (0, pad_w), (0, 0))
                img_in = np.pad(img_in, padding, mode='constant')
                img_tar = np.pad(img_tar, ((0, pad_h*scale), (0, pad_w*scale), (0, 0)), mode='constant')
                ih, iw = img_in.shape[:2]
            
            # Get random patch position
            ix = np.random.randint(0, max(0, iw - input_size) + 1)
            iy = np.random.randint(0, max(0, ih - input_size) + 1)
            
            # Extract patches with exact dimensions
            input_patch = img_in[iy:iy + input_size, ix:ix + input_size, :]
            target_patch = img_tar[iy*scale:(iy+input_size)*scale, ix*scale:(ix+input_size)*scale, :]
            
            # Apply augmentation
            input_patch, target_patch = common.augment(input_patch, target_patch)
            
            return input_patch, target_patch
        else:
            # For validation, make dimensions consistent
            ih, iw = img_in.shape[:2]
            
            # Make dimensions divisible by scale
            new_h = (ih // scale) * scale
            new_w = (iw // scale) * scale
            
            # Crop to consistent size
            img_in = img_in[:new_h // scale, :new_w // scale, :]
            img_tar = img_tar[:new_h, :new_w, :]
            
            return img_in, img_tar

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
        return lr, hr
