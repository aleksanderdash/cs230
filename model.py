import os
import pickle
import time
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models

from video_dataset import VideoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class DeepPepegaNet(nn.Module):

    def __init__(self):
        super(DeepPepegaNet, self).__init__()
        self.inception = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        #del self.inception.AuxLogits
        #self.inception = nn.Sequential(*(list(self.inception.children())[:-1]))
        self.inception.fc = Identity()
        for param in self.inception.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(2048,512)
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(512, 2)

    def forward(self, x):
        """ 
            Expects that x is input of shape (batch_size, 6(num_frames), 3, 299, 299)
            Outputs (batch_size, 2), where the 2 are logits
        """
        batch_size, num_frames, _, _, _ = x.shape
        out = x.reshape(-1, 3, 299, 299)
        out = self.inception(out)
        out = out.reshape(batch_size, num_frames, 2048).permute(1, 0, 2)
        outputs, (hn, cn) = self.lstm(out)
        out = hn.reshape(-1, 512)
        out = self.dropout(out)
        out = self.linear(out)
        return out

