""" Full assembly of the parts to form the complete network """
import torch.nn.functional as F
from .unet_parts import *

class AdaBoost_UNet(nn.Module):
	def __init__(self, n_channels, n_classes, level, skip_option='None', filters=32):
		super(AdaBoost_UNet, self).__init__()
		self.width = 512
		self.level = level
		self.n_channels = n_channels
		if n_classes==2:
			self.n_classes = n_classes-1
		else:
			self.n_classes = n_classes
		# Encoders
		self.X_00 = DoubleConv(n_channels, filters)
		self.X_10 = Down(filters, 2*filters)
		self.X_20 = Down(2*filters, 4*filters)
		self.X_30 = Down(4*filters, 8*filters)
		self.X_40 = Down(8*filters, 16*filters)

		#UNet 1
		self.X_31 = Up_skip(16*filters, 8*filters, self.width//8, 'none')
		self.X_22 = Up_skip(8*filters, 4*filters, self.width//4, 'none')
		self.X_13 = Up_skip(4*filters, 2*filters, self.width//2, 'none')
		self.X_04 = Up_skip(2*filters, filters, self.width, 'none')
		self.out_04 = OutConv(filters, self.n_classes)

		#UNet 2
		self.X_21 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
		self.X_11 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
		self.X_01 = Up_skip(2*filters, filters, self.width, skip_option)
		self.out_01 = OutConv(filters, self.n_classes)

		#UNet 3
		self.X_12 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
		self.X_02 = Up_skip(2*filters, filters, self.width, skip_option)
		self.out_02 = OutConv(filters, self.n_classes)

		#UNet 4
		self.X_03 = Up_skip(2*filters, filters, self.width, skip_option)
		self.out_03 = OutConv(filters, self.n_classes)

		self.out_10= OutConv(2*filters, self.n_classes)
		self.out_20= OutConv(4*filters, self.n_classes)
		self.out_11= OutConv(2*filters, self.n_classes)
		self.out_02 = OutConv(filters, self.n_classes)
		self.out_30= OutConv(8*filters, self.n_classes)
		self.out_21= OutConv(4*filters, self.n_classes)
		self.out_12= OutConv(2*filters, self.n_classes)
		self.out_03 = OutConv(filters, self.n_classes)
		self.out_40= OutConv(16*filters, self.n_classes)
		self.out_31= OutConv(8*filters, self.n_classes)
		self.out_22= OutConv(4*filters, self.n_classes)
		self.out_13= OutConv(2*filters, self.n_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init_weights(m, init_type='kaiming')
			elif isinstance(m, nn.BatchNorm2d):
				init_weights(m, init_type='kaiming')
				
	def forward(self, input_x):
		x_00 = self.X_00(input_x)
		x_10 = self.X_10(x_00)
		x_20 = self.X_20(x_10)
		x_30 = self.X_30(x_20)
		x_40 = self.X_40(x_30)
		x_31 = self.X_31(x_40,x_30)

		if len(self.level)==1:
			if '1' in self.level:
				x_22 = self.X_22(x_31,x_20)
				x_13 = self.X_13(x_22,x_10)
				x_04 = self.X_04(x_13,x_00)
				out_10 = self.out_10(x_10)
				out_20 = self.out_20(x_20)
				out_30 = self.out_30(x_30)
				out_40 = self.out_40(x_40)
				out_31 = self.out_31(x_31)
				out_22 = self.out_22(x_22)
				out_13 = self.out_13(x_13)
				out_04 = self.out_04(x_04)
				return [out_10, out_20, out_30, out_40, out_31, out_22, out_13, out_04]

			if '2' in self.level:
				x_22 = self.X_22(x_31,x_20)
				x_13 = self.X_13(x_22,x_10)
				x_03 = self.X_03(x_13,x_00)
				out_03 = self.out_03(x_03)
				return [out_03]

			if '3' in self.level:
				x_22 = self.X_22(x_31, x_20)
				x_12 = self.X_12(x_22, x_10)
				x_02 = self.X_02(x_12, x_00)
				out_12 = self.out_12(x_12)
				out_02 = self.out_02(x_02)
				return [out_12, out_02]

			if '4' in self.level:
				x_21 = self.X_21(x_31,x_20)
				x_11 = self.X_11(x_21,x_10)
				x_01 = self.X_01(x_11,x_00)
				out_21 = self.out_21(x_21)
				out_11 = self.out_11(x_11)
				out_01 = self.out_01(x_01)
				return [out_21, out_11, out_01]
				# return out_04

		else:
			x_22 = self.X_22(x_31,x_20)
			x_13 = self.X_13(x_22,x_10)
			x_04 = self.X_04(x_13,x_00)
			out_04 = self.out_04(x_04)

			x_21 = self.X_21(x_31,x_20)
			x_11 = self.X_11(x_21,x_10)
			x_01 = self.X_01(x_11,x_00)
			out_01 = self.out_01(x_01)

			x_12 = self.X_12(x_22, x_10)
			x_02 = self.X_02(x_12, x_00)
			out_02 = self.out_02(x_02)

			x_03 = self.X_03(x_13, x_00)
			out_03 = self.out_03(x_03)
			return [out_04, out_03, out_02, out_01]

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