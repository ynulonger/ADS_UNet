import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropy(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(SoftCrossEntropy, self).__init__()
        self.reduction=reduction
        self.weight = weight

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        log_likelihood = -F.log_softmax(inputs, dim=1)
        loss = torch.mul(log_likelihood, target)
        if self.weight != None:
            loss = loss * self.weight
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        return loss
