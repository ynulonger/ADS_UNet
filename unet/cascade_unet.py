""" Full assembly of the parts to form the complete network """
import torch.nn.functional as F
from .unet_parts import *

class cascade_UNet(nn.Module):
	def __init__(self, n_channels, n_classes, level, skip_option='None', filters=32):
		super(cascade_UNet, self).__init__()
		self.width = 512
		self.level = level
		self.n_channels = n_channels
		if n_classes==2:
			self.n_classes = n_classes-1
		else:
			self.n_classes = n_classes
		self.X_00 = DoubleConv(n_channels, filters)
		self.X_10 = Down(filters, 2*filters)
		self.X_20 = Down(2*filters, 4*filters)
		self.X_30 = Down(4*filters, 8*filters)
		self.X_40 = Down(8*filters, 16*filters)

		self.up_40 = Up_skip(16*filters, 8*filters, self.width//8, skip_option)
		self.up_31 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
		self.up_22 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
		self.up_13 = Up_skip(2*filters, filters, self.width, skip_option)

		self.out_20= OutConv(4*filters, self.n_classes)
		self.out_30= OutConv(8*filters, self.n_classes)
		self.out_40= OutConv(16*filters, self.n_classes)
		self.out_31= OutConv(8*filters, self.n_classes)
		self.out_22= OutConv(4*filters, self.n_classes)
		self.out_13= OutConv(2*filters, self.n_classes)
		self.out_04 = OutConv(filters, self.n_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init_weights(m, init_type='kaiming')
			elif isinstance(m, nn.BatchNorm2d):
				init_weights(m, init_type='kaiming')
				
	def forward(self, input_x):
		x_00 = self.X_00(input_x)
		x_10 = self.X_10(x_00)
		if len(self.level)==1:
			if '1' in self.level:
				x_20 = self.X_20(x_10)
				out_20 = self.out_20(x_20)
				return [out_20]

			if '2' in self.level:
				x_20 = self.X_20(x_10)
				x_30 = self.X_30(x_20)
				out_30 = self.out_30(x_30)
				return [out_30]

			if '3' in self.level:
				x_20 = self.X_20(x_10)
				x_30 = self.X_30(x_20)
				x_40 = self.X_40(x_30)
				x_31 = self.up_40(x_40,x_30)
				x_22 = self.up_31(x_31,x_20)
				x_13 = self.up_22(x_22,x_10)
				x_04 = self.up_13(x_13,x_00)
				out_31 = self.out_31(x_31)
				out_22 = self.out_22(x_22)
				out_13 = self.out_13(x_13)
				out_04 = self.out_04(x_04)
				return [out_40, out_31, out_22, out_13, out_04]
		else:
			x_20 = self.X_20(x_10)
			x_30 = self.X_30(x_20)
			x_40 = self.X_40(x_30)
			x_31 = self.up_40(x_40,x_30)
			x_22 = self.up_31(x_31,x_20)
			x_13 = self.up_22(x_22,x_10)
			x_04 = self.up_13(x_13,x_00)
			out_04 = self.out_04(x_04)
			return out_04

import torch
import torch.nn as nn
from torch.nn import init

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