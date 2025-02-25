import torch
import torch.nn as nn
from config import *

class CNN(nn.Module):
    def __init__(self, action_size):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(CONV1_IN_CHANNELS, CONV1_OUT_CHANNELS, kernel_size=CONV1_KERNEL_SIZE, stride=CONV1_STRIDE)
        self.conv2 = torch.nn.Conv2d(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=CONV2_KERNEL_SIZE, stride=CONV2_STRIDE)
        self.conv3 = torch.nn.Conv2d(CONV2_OUT_CHANNELS, CONV3_OUT_CHANNELS, kernel_size=CONV3_KERNEL_SIZE, stride=CONV3_STRIDE)
        self.fc1 = torch.nn.Linear(FC1_UNITS_IN, FC1_UNITS_OUT)
        self.fc2 = torch.nn.Linear(FC1_UNITS_OUT, ACTION_SIZE)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)