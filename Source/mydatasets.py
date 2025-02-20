import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import DatasetFolder
import torch.utils.data as data
from PIL import Image
import random
import os

import random
from glob import glob
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        raise(RuntimeError('No Module named accimage'))
    else:
        return pil_loader(path)
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class ImageData(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform = None,
            target_transform  = None,
            loader = default_loader,
            is_valid_file = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def __len__(self):
        return len(self.samples)

class WrapperDataset(data.Dataset):
    """Dataset class for the Imagenet-LT dataset."""

    def __init__(self, data_class, transform,task,loader=default_loader,target_transform=None):

        self.data_class = data_class
        self.transform = transform
        self.data = self.data_class.data
        self.targets = self.data_class.targets
        self.task = task
        self.class_to_idx = self.data_class.class_to_idx
        self.target_transform = target_transform
        self.loader = loader
        self.samples = list(zip(self.data,self.targets))




        # 统计类数量
        classes_num = {}
        id_to_class = {v: k for k, v in self.class_to_idx.items()}
        for i in self.targets:
            class_name = id_to_class[i]
            if class_name not in classes_num:
                classes_num[class_name] = 1
            else:
                classes_num[class_name] += 1

        self.cls_num_list = list(classes_num.values())
        self.class_names = list(classes_num.keys())
        self.id_to_name = id_to_class
    def __len__(self):
        return len(self.samples)


    def __getitem__(self,index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target