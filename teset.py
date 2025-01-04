import time
import numpy as np
from torchvision import models
import torch.nn as nn

import patch_generator
import config
import os
import imageio
import torch
from PIL import Image
import torchvision.transforms as transforms
import datareader


'''dataset = datareader.DataReader()
data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=5)'''
img = Image.open('./000_sdv4_00020.png')
img = transforms.ToTensor()(img)
img = torch.reshape(img,[1,3,512,512])
