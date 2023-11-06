import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
from os import path
import os
import numpy as np
import torch, random
from glob import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, train_data_dir, rain_dir, gt_dir, crop_size):
        super().__init__()
        self.input_names = sorted(glob(os.path.join(train_data_dir, rain_dir, "*.png")))
        self.gt_names = sorted(glob(os.path.join(train_data_dir, gt_dir, "*.png")))
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir
        print("[INFO] num of training data: {}".format(len(self.input_names)))

    def mixup_image(self, input_img, gt_img, input_alpha):
        input_img = np.array(input_img)
        gt_img = np.array(gt_img)
        target_input = input_alpha * input_img + (1 - input_alpha) * gt_img
        return Image.fromarray(target_input.astype(np.uint8))

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/',input_name)[-1][:-4]
        input_img = Image.open(input_name)
        try:
            gt_img = Image.open(gt_name)
        except:
            gt_img = Image.open(gt_name).convert('RGB')
        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)
        if list(input_im.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))
        return input_im, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

