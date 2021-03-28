import torch
from random import shuffle
import os
import random
import time
import copy
from torchvision import datasets, models, transforms, utils
from random import shuffle
import numpy as np
from custom_utils import *
from train import *
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN8s(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        # these layers take input that is the output of the VGG MaxPool5
        self.fc6 = nn.Conv2d(512, 4096, 1)  # x_input: [bs, 512, 7x7]
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)  # x_input: [bs, 4096, 7x7]
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)


        # to convolve the shallowers layers: FOR THE FIRST FUSION
        self.conv_pool_4 = nn.Conv2d(512, n_class, 1)  # pool4: [bs, 512, 32, 54]
        self.first_upsampling = nn.ConvTranspose2d(n_class, n_class, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        # to convolve the shallowers layers: FOR THE SECOND FUSION
        self.conv_pool_3 = nn.Conv2d(256, n_class, 1)  # pool3: [bs, 256, 64, 108]
        self.second_upsampling = nn.ConvTranspose2d(n_class, n_class, kernel_size=3, stride=2, padding=1, dilation=1,
                                                   output_padding=1)

    def forward(self, x, pool4, pool3): # x is pool5

        x = self.drop6(self.relu6(self.fc6(x))) # here we take the output of the avgpool of the VGG x_input: [bs, 512, 7 ,7]
        x = self.drop7(self.relu7(self.fc7(x))) # [bs, 4096, 7, 7]
        x = self.score_fr(x)

        x = self.first_upsampling(x)
        pool4_preds = self.conv_pool_4(pool4)

        if self.training == False:
            pool4_preds = pool4_preds[:,:,:x.shape[2],:x.shape[3]]

        fuse_1 = (x+pool4_preds)
        pool3_preds = self.conv_pool_3(pool3)
        temp_upsampling = self.second_upsampling(fuse_1)

        if self.training == False:
            pool3_preds = pool3_preds[:,:,:temp_upsampling.shape[2],:temp_upsampling.shape[3]]

        result = (pool3_preds + temp_upsampling)

        return result


class VGG_custom(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG_custom, self).__init__()
        self.features = nn.ModuleList(list(features))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        results = []
        for ii, model in enumerate(self.features):   # self.features is the convolutional block of the VGG
            x = model(x)
            if ii in {16,23}:
                results.append(x)

        x = self.classifier(x, pool4 =results[1], pool3=results[0])

        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)