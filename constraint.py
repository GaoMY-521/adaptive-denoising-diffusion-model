import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import utils
import os
import argparse
from torchvision import datasets, transforms
from einops import rearrange, reduce, repeat
from kmeans_function import kmeans

class ThetaPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = nn.Sequential(nn.Linear(85, 256),
                                 nn.GELU(),
                                 nn.Linear(256, 2048)
                                 )
        self.fc1 = nn.Linear(2048, 256)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(256, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.ten = nn.Parameter(torch.tensor(10), requires_grad=False)
        self.sig = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, att):
        map_att = self.map(att)
        h = self.lrelu(self.fc1(map_att))
        h = self.fc2(h)
        # x = (h + 1) * 0.5
        # x = 3.5 * h * h + 4.5 * h + 2
        x = self.sig(h) * 10
        return x

def predict_steps(theta_prediction, attributes):
    predict_steps = theta_prediction(attributes.to(device)) # bs x 1
    predict_steps_ = predict_steps.mean()
    # print(predict_steps)
    final_timesteps = int(99 / predict_steps_) + 2
    times = torch.FloatTensor(final_timesteps, ).T
    for k in range(100):
        time_k = 99 - (k * predict_steps_)
        if time_k < 0:
            time_k = -1
            times[k] = time_k
            break
        elif time_k == 0:
            times[k] = time_k + 0.01
        else:
            times[k] = time_k
    time_pairs = list(zip(times[:-1], times[1:]))

    return time_pairs
