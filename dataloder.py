import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageData(Dataset):
    def __init__(self, img_dir, file_paths, transform=None):
        self.matcontent = sio.loadmat(file_paths) # resnet101.mat
        self.image_files = np.squeeze(self.matcontent['image_files']) # 取出所有图像的文件名
        self.img_dir = img_dir
        self.transform = transform


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx][0]
        image_file = os.path.join(self.img_dir,
                                    '/'.join(image_file.split('/')[6:]))
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image