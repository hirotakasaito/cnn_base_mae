import sys
import os
import json
from glob import iglob
import math
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from collections import deque
import random

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, is_trans=True):

        self.img_paths = []
        for img_path in iglob(os.path.join(dataset_dir,"*")):
            img_path = self.img_paths.append(img_path)
        if len(self.img_paths) == 0:
            print(f"Notice: Your Images is not used")
            sys.exit(1)
        self.transform = nn.Sequential(
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ).to(torch.device('cuda'))
        self.convert_tensor = transforms.ToTensor()

    def __len__(self):

        return len(self.img_paths)

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img = self.convert_tensor(img)
        # img = torch.load(img_path)
        # img = img[0]

        return img
