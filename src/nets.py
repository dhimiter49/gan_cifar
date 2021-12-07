import torch
import torch.nn as nn
import torch.nn.functional as F


class DefaultGen(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        pass

    def forward(self, noise, label):
        pass


class DefaultDis(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        pass

    def forward(self, image):
        pass
