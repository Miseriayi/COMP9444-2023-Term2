"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.linear = nn.Linear(784,10)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = x.view(-1,784)
        x = self.linear(x)
        return self.softmax(x)

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.l1 = nn.Linear(784,250)
        self.l2 = nn.Linear(250,10)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        x = x.view(-1,784)
        x = nn.functional.tanh(self.l1(x))
        x = self.l2(x)
        return self.softmax(x)

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.b1 = nn.BatchNorm2d(num_features=32)
        self.maxp = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64*7*7,512)
        self.linear2 = nn.Linear(512,10)
        self.softmax = nn.LogSoftmax(dim=-1)
    def forward(self, x):
        x = self.maxp(F.relu(self.b1(self.conv1(x))))
        x = self.maxp2(F.relu(self.b2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.linear2(self.linear1(x))
        return self.softmax(x)
