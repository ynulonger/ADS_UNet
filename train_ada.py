import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from metrics import *
from dice_loss import *
from utils.loss import *
from torch import optim
from utils.dataset import *
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.ticker as ticker
from torch.autograd import Variable
from collections import OrderedDict
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from unet.Adaboost_UNet import AdaBoost_UNet as AdaBoost_UNet
from thop import profile
from thop import clever_format

def count_your_model(model, x, y):
        # your rule here
        pass

ADS_1    = AdaBoost_UNet(3, 5, "1", skip_option='scse', filters=64, deep_sup=True, head=None)
ADS_2    = AdaBoost_UNet(3, 5, "2", skip_option='scse', filters=64, deep_sup=True, head=None)
ADS_3    = AdaBoost_UNet(3, 5, "3", skip_option='scse', filters=64, deep_sup=True, head=None)
ADS_4    = AdaBoost_UNet(3, 5, "4", skip_option='scse', filters=64, deep_sup=True, head=None)
ADS_1234 = AdaBoost_UNet(3, 5, "1234_pred", skip_option='scse', filters=64, deep_sup=True, head=None)

NETS = [ADS_1,ADS_2,ADS_3,ADS_4,ADS_1234]
for net in NETS:
    inputs = torch.randn(1, 3, 512, 512)
    flops, params = profile(net, inputs=(inputs, ), custom_ops={AdaBoost_UNet: count_your_model})
    flops, params = clever_format([flops, params], "%.3f")
    print("flops", flops, "params", params)