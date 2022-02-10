# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:36:44 2020

@author: kdu
"""

from parameters import param

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NET(nn.Module):
    def __init__(self, in_dim, 
                 n_hidden_1, n_hidden_2, n_hidden_3,
                 out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, out_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.softmax(x)
        return x
