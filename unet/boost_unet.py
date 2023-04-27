""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from .unet_parts import *
from torch.nn import init
import torch.nn.functional as F

class AdaBoost_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, iteration, skip_option='None', filters=32):
        super(AdaBoost_UNet, self).__init__()
        self.width = 512
        self.iteration = iteration
        self.n_channels = n_channels
        if n_classes==2:
            self.n_classes = n_classes-1
        else:
            self.n_classes = n_classes
        self.X_00 = DoubleConv(n_channels, filters)

        self.X_10     = Down(filters, 2*filters)
        self.X_10_out = OutConv(2*filters, self.n_classes)
        self.X_20     = Down(2*filters, 4*filters)
        self.X_20_out = OutConv(4*filters, self.n_classes)
        self.X_30     = Down(4*filters, 8*filters)
        self.X_30_out = OutConv(8*filters, self.n_classes)
        self.X_40     = Down(8*filters, 16*filters)
        self.X_40_out = OutConv(16*filters, self.n_classes)

        self.X_01     = Up_skip(2*filters, filters, self.width, skip_option)
        self.X_01_out = OutConv(filters, self.n_classes)
        self.X_11     = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.X_11_out = OutConv(2*filters, self.n_classes)
        self.X_21     = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.X_21_out = OutConv(4*filters, self.n_classes)
        self.X_31     = Up_skip(16*filters, 8*filters, self.width//8, skip_option)
        self.X_31_out = OutConv(8*filters, self.n_classes)

        self.X_02     = Up_skip(2*filters, filters, self.width, skip_option)
        self.X_02_out = OutConv(filters, self.n_classes)
        self.X_12     = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.X_12_out = OutConv(2*filters, self.n_classes)
        self.X_22     = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.X_22_out = OutConv(4*filters, self.n_classes)

        self.X_03     = Up_skip(2*filters, filters, self.width, skip_option)
        self.X_03_out = OutConv(filters, self.n_classes)
        self.X_13     = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.X_13_out = OutConv(2*filters, self.n_classes)

        self.X_04     = Up_skip(2*filters, filters, self.width, skip_option)
        self.X_04_out = OutConv(filters, self.n_classes)

  #       self.pathes= [nn.Sequential(self.X_00),
  # 1                   nn.Sequential(self.X_00, self.X_10),
  # 2                   nn.Sequential(self.X_00, self.X_10, self.X_01),
  # 3                   nn.Sequential(self.X_00, self.X_10, self.X_20),
  # 4                   nn.Sequential(self.X_00, self.X_10, self.X_20, self.X_11),
  # 5                   nn.Sequential(self.X_00, self.X_10, self.X_20, self.X_11, self.X_02),
  # 6                   nn.Sequential(self.X_00, self.X_10, self.X_20, self.X_30),
  # 7                   nn.Sequential(self.X_00, self.X_10, self.X_20, self.X_30, self.X_21),
  # 8                   nn.Sequential(self.X_00, self.X_10, self.X_20, self.X_30, self.X_21, self.X_12),
  # 9                   nn.Sequential(self.X_00, self.X_10, self.X_20, self.X_30, self.X_21, self.X_12, self.X_03),
  # 10                  nn.Sequential(self.X_00, self.X_10, self.X_20, self.X_30, self.X_40),
  # 11                  nn.Sequential(self.X_00, self.X_10, self.X_20, self.X_30, self.X_40, self.X_31),
  # 12                  nn.Sequential(self.X_00, self.X_10, self.X_20, self.X_30, self.X_40, self.X_31, self.X_22),
  # 13                  nn.Sequential(self.X_00, self.X_10, self.X_20, self.X_30, self.X_40, self.X_31, self.X_22, self.X_13),
  # 14                  nn.Sequential(self.X_00, self.X_10, self.X_20, self.X_30, self.X_40, self.X_31, self.X_22, self.X_13, self.X_04)]
        # self.out_layers = [None, self.X_10_out, self.X_01_out, self.X_20_out, self.X_11_out,
        #                    self.X_02_out, self.X_30_out, self.X_21_out, self.X_12_out, self.X_03_out, 
        #                    self.X_40_out, self.X_31_out, self.X_22_out, self.X_13_out, self.X_04_out]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, input_x):
        x_00 = self.X_00(input_x)
        x_10 = self.X_10(x_00)
        if self.iteration==1:
            return self.X_10_out(x_10)
        elif self.iteration==2:
            x_01 = self.X_01(x_10, x_00)
            return self.X_01_out(x_01)
        elif self.iteration==3:
            x_20 = self.X_20(x_10)
            return self.X_20_out(x_20)
        elif self.iteration==4:
            x_20 = self.X_20(x_10)
            x_11 = self.X_11(x_20, x_10)
            return self.X_11_out(x_11)
        elif self.iteration==5:
            x_20 = self.X_20(x_10)
            x_11 = self.X_11(x_20, x_10)
            x_02 = self.X_02(x_11, x_00)
            return self.X_02_out(x_02)
        elif self.iteration==6:
            x_20 = self.X_20(x_10)
            x_30 = self.X_30(x_20)
            return self.X_30_out(x_30)
        elif self.iteration==7:
            x_20 = self.X_20(x_10)
            x_30 = self.X_30(x_20)
            x_21 = self.X_21(x_30, x_20)
            return self.X_21_out(x_21)
        elif self.iteration==8:
            x_20 = self.X_20(x_10)
            x_30 = self.X_30(x_20)
            x_21 = self.X_21(x_30, x_20)
            x_12 = self.X_12(x_21, x_10)
            return self.X_12_out(x_12)
        elif self.iteration==9:
            x_20 = self.X_20(x_10)
            x_30 = self.X_30(x_20)
            x_21 = self.X_21(x_30, x_20)
            x_12 = self.X_12(x_21, x_10)
            x_03 = self.X_03(x_12, x_00)
            return self.X_03_out(x_00)
        elif self.iteration==10:
            x_20 = self.X_20(x_10)
            x_30 = self.X_30(x_20)
            x_40 = self.X_40(x_30)
            return self.X_40_out(x_40)
        elif self.iteration==11:
            x_20 = self.X_20(x_10)
            x_30 = self.X_30(x_20)
            x_40 = self.X_40(x_30)
            x_31 = self.X_31(x_40,x_30)
            return self.X_31_out(x_31)
        elif self.iteration==12:
            x_20 = self.X_20(x_10)
            x_30 = self.X_30(x_20)
            x_40 = self.X_40(x_30)
            x_31 = self.X_31(x_40,x_30)
            x_22 = self.X_22(x_31,x_20)
            return self.X_22_out(x_22)
        elif self.iteration==13:
            x_20 = self.X_20(x_10)
            x_30 = self.X_30(x_20)
            x_40 = self.X_40(x_30)
            x_31 = self.X_31(x_40,x_30)
            x_22 = self.X_22(x_31,x_20)
            x_13 = self.X_13(x_22,x_10)
            return self.X_13_out(x_13)
        elif self.iteration==14:
            x_20 = self.X_20(x_10)
            x_30 = self.X_30(x_20)
            x_40 = self.X_40(x_30)
            x_31 = self.X_31(x_40,x_30)
            x_22 = self.X_22(x_31,x_20)
            x_13 = self.X_13(x_22,x_10)
            x_04 = self.X_04(x_13,x_00)
            return self.X_04_out(x_04)
        elif self.iteration>14:
            x_01 = self.X_01(x_10, x_00)

            x_20 = self.X_20(x_10)
            x_11 = self.X_11(x_20, x_10)
            x_02 = self.X_02(x_11, x_00)

            x_30 = self.X_30(x_20)
            x_21 = self.X_21(x_30, x_20)
            x_12 = self.X_12(x_21, x_10)
            x_03 = self.X_03(x_12, x_00)

            x_40 = self.X_40(x_30)
            x_31 = self.X_31(x_40,x_30)
            x_22 = self.X_22(x_31,x_20)
            x_13 = self.X_13(x_22,x_10)
            x_04 = self.X_04(x_13,x_00)
            return [self.X_10_out(x_10), self.X_01_out(x_01), self.X_20_out(x_20), self.X_11_out(x_11), self.X_02_out(x_02),
                    self.X_30_out(x_30), self.X_21_out(x_21), self.X_12_out(x_12), self.X_03_out(x_03), self.X_40_out(x_40),
                    self.X_31_out(x_31), self.X_22_out(x_22), self.X_13_out(x_13), self.X_04_out(x_04)]


def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)