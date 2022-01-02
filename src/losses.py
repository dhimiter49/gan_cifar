import torch
from torch import nn

BCELoss = nn.BCELoss


class ExpSigmaLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return -0.5 * torch.div(prediction, 1 - prediction).mean()
