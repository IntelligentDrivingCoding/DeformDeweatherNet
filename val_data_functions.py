import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
from glob import glob
import os

class ValData(data.Dataset):
    def __init__(self, val_data_dir, rain_dir, gt_dir):
        super().__init__()
        self.input_names = sorted(glob(os.path.join(val_data_dir, rain_dir, "*.jpg")))
        self.gt_names = sorted(glob(os.path.join(val_data_dir, gt_dir, "*.jpg")))
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        input_img = Image.open(input_name)
        gt_img = Image.open(gt_name)

        # Resizing image in the multiple of 16"
        wd_new, ht_new = input_img.size
        if ht_new>wd_new and ht_new>1024:
            wd_new = int(np.ceil(wd_new*1024/ht_new))
            ht_new = 1024
        elif ht_new<=wd_new and wd_new>1024:
            ht_new = int(np.ceil(ht_new*1024/wd_new))
            wd_new = 1024
        wd_new = int(32*np.ceil(wd_new/32.0))
        ht_new = int(32*np.ceil(ht_new/32.0))
        input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

