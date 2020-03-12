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



class DeepPepegaNet(nn.Module):

    def __init__(self):
        super(DeepPepegaNet, self).__init__()
        self.inception = torchvision.models.inception_v3(pretrained=True)
        del self.inception.AuxLogits
        self.inception = nn.Sequential(*(list(self.inception.children())[:-1]))
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
        out = x.reshape(-1, 3, 299, 299)
        out = self.inception(out)
        out = out.reshape(6, -1, 2048)
        outputs, (hn, cn) = self.lstm(out)
        out = hn.reshape(-1, 512)
        out = self.dropout(out)
        out = self.linear(out)
        return out

