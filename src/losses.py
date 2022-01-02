import torch
from torch import nn

BCELoss = nn.BCELoss


class ExpSigmaLoss(nn.Module):
    """
    This loss is defined in https://arxiv.org/pdf/1701.00160.pdf as
                J^{(G)} = -0.5 * exp(sigmoid^{-1}(D(G(z)))).
    The inverse sigmoid function is defined as
                      sigmoid^{-1}(x) = log(y/(1-y)).
    Thus we get a cost function for the generator defined as
                J^{(G)} = -0.5 * D(G(z))) / (1 - D(G(z))).
    """

    def __init__(self):
        super().__init__()
        pass

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return -0.5 * torch.div(prediction, 1 - prediction).mean()


class WassersteinLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return (target * prediction).mean()
