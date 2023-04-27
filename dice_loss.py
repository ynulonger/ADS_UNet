import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class mIoULoss(nn.Module):
    def __init__(self, size_average=True, n_classes=2, ignore_indix=None):
        super(mIoULoss, self).__init__()
        self.ignore_indix = ignore_indix
        if self.ignore_indix !=None:
            self.classes = n_classes-1
        else:
            self.classes = n_classes
    def forward(self, inputs, target_oneHot, weight=None):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        N = inputs.size()[0]
        if self.ignore_indix !=None:
            selected_idx = [i for i in range(self.classes)]
            inputs = inputs[:,selected_idx,:,:]
            target_oneHot = target_oneHot[:,selected_idx,:,:]
        # print(inputs.size(), self.classes)
        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs,dim=1)
        # Numerator Product
        inter = inputs * target_oneHot
        union= inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2)
        union = union.view(N,self.classes,-1).sum(2)
        mIoU = (inter/union).mean(1)
        loss = 1-mIoU
        if weight == None:
            return loss.mean()
        else:
            weight= weight.squeeze()
            loss = loss*weight
            loss = loss.sum()
            return loss

class IoULoss(nn.Module):
    def __init__(self, size_average=True, n_classes=2):
        super(IoULoss, self).__init__()

    def forward(self, inputs, target_oneHot, weight=None):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        N = inputs.size()[0]
        # predicted probabilities for each pixel along channel
        inputs = torch.sigmoid(inputs)
        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,-1).sum(1)
        #Denominator 
        union = inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,-1).sum(1)
        IoU = inter/union
        loss = 1-IoU
        if weight==None:
            w_loss = loss
        else:
            w_loss = loss*weight
        ## Return average loss over classes and batch
        return w_loss.sum()

class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2,classes=1):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.classes = classes
        # self.weight = weight
        # self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim()>2:
            inputs = inputs.contiguous().view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1,2)
            inputs = inputs.contiguous().view(-1, inputs.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)
        # compute the negative likelyhood
        if self.classes==2:
            logpt = -F.cross_entropy(inputs, target.long())
        else:
            logpt = -F.cross_entropy(inputs, target.argmax(dim=1).long())
        pt = torch.exp(logpt)
        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt
        # averaging (or not) loss
        # if self.size_average:
        #     return loss.mean()
        # else:
        #     return loss.sum()
        return loss