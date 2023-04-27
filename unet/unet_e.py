# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *
# from .init_weights import init_weights
from torchvision import models


class UNet_e(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, filters=16,bilinear=False,
                 is_deconv=True, is_batchnorm=True, flag='out'):
        super(UNet_e, self).__init__()
        self.n_channels = n_channels
        if n_classes==1:
            self.n_classes = 2
        else:
            self.n_classes = n_classes
        self.bilinear = bilinear
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.flag= flag

        # downsampling
        self.X_00 = DoubleConv(n_channels, filters)
        self.X_10 = Down(filters, 2*filters)
        self.X_20 = Down(2*filters, 4*filters)
        self.X_30 = Down(4*filters, 8*filters)
        self.X_40 = Down(8*filters, 16*filters)

        #UNet level 1
        self.up_10  = Up(2*filters, filters, self.bilinear)

        #UNet level 2
        self.up_20 = Up(4*filters, 2*filters, self.bilinear)
        self.up_11 = Up(2*filters, filters, self.bilinear)

        #UNet level 3
        self.up_30 = Up(8*filters, 4*filters, self.bilinear)
        self.up_21 = Up(4*filters, 2*filters, self.bilinear)
        self.up_12 = Up(2*filters, filters, self.bilinear)

        #UNet level 4
        self.up_40 = Up(16*filters, 8*filters, self.bilinear)
        self.up_31 = Up(8*filters, 4*filters, self.bilinear)
        self.up_22 = Up(4*filters, 2*filters, self.bilinear)
        self.up_13 = Up(2*filters, filters, self.bilinear)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters, self.n_classes, 1)
        self.final_2 = nn.Conv2d(filters, self.n_classes, 1)
        self.final_3 = nn.Conv2d(filters, self.n_classes, 1)
        self.final_4 = nn.Conv2d(filters, self.n_classes, 1)


    def forward(self, inputs):
        x_00 = self.X_00(inputs)
        x_10 = self.X_10(x_00)
        # column : 1
        x_01 = self.up_10(x_10, x_00)

        # column : 2
        x_20 = self.X_20(x_10)
        x_11 = self.up_20(x_20,x_10)
        x_02 = self.up_11(x_11,x_00)

        # column : 3
        x_30 = self.X_30(x_20)
        x_21 = self.up_30(x_30,x_20)
        x_12 = self.up_21(x_21,x_10)
        x_03 = self.up_12(x_12,x_00)

        # column : 4
        x_40 = self.X_40(x_30)
        x_31 = self.up_40(x_40,x_30)
        x_22 = self.up_31(x_31,x_20)
        x_13 = self.up_22(x_22,x_10)
        x_04 = self.up_13(x_13,x_00)
        
        if self.flag=='feat':
            return [x_01, x_02, x_03, x_04]

        # final layer
        final_1 = self.final_1(x_01)
        final_2 = self.final_2(x_02)
        final_3 = self.final_3(x_03)
        final_4 = self.final_4(x_04)
        # final = (final_1 + final_2 + final_3 + final_4) / 4
        if self.flag=='out':
            return [final_1, final_2, final_3, final_4]

# if __name__ == '__main__'
#     model = UNet2Plus()
#     print('# generator parameters:', 1.0 * sum(param.numel() for param in model.parameters()) / 1000000)
    
#     params = list(model.named_parameters())
#     for i in range(len(params)):
#         name, param = params[i]
#         print('name:', name, ' param.shape:', param.shape)