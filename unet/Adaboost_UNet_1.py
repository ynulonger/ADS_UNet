""" Full assembly of the parts to form the complete network """
import torch.nn.functional as F
from .unet_parts import *

class AdaBoost_UNet_1(nn.Module):
	def __init__(self, n_channels, n_classes, level, skip_option='None', filters=32):
		super(AdaBoost_UNet_1, self).__init__()
		self.width = 512
		self.level = level
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.X_00 = DoubleConv(n_channels, filters)
		self.X_10 = Down(filters, 2*filters)
		self.X_20 = Down(2*filters, 4*filters)
		self.X_30 = Down(4*filters, 8*filters)
		self.X_40 = Down(8*filters, 16*filters)

		#UNet level 1
		self.up_1  = Up_skip(2*filters, filters, self.width, skip_option)
		self.out_1 = OutConv(filters, n_classes)
		#UNet level 2
		self.up_20 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
		self.up_21 = Up_skip(2*filters, filters, self.width, skip_option)
		self.out_2 = OutConv(filters, n_classes)
		#UNet level 3
		self.up_30 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
		self.up_31 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
		self.up_32 = Up_skip(2*filters, filters, self.width, skip_option)
		self.out_3 = OutConv(filters, n_classes)
		#UNet level 4
		self.up_40 = Up_skip(16*filters, 8*filters, self.width//8, skip_option)
		self.up_41 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
		self.up_42 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
		self.up_43 = Up_skip(2*filters, filters, self.width, skip_option)
		self.out_4 = OutConv(filters, n_classes)

	def forward(self, input_x):
		x_00 = self.X_00(input_x)
		x_10 = self.X_10(x_00)
		if len(self.level)==1:
			if '1' in self.level:
				x_01 = self.up_1(x_10, x_00)
				out_1 = self.out_1(x_01)
				return out_1
			if '2' in self.level:
				x_20 = self.X_20(x_10)
				x_11 = self.up_20(x_20,x_10)
				x_02 = self.up_21(x_11,x_00)
				out_2 = self.out_2(x_02)
				return out_2

			if '3' in self.level:
				x_20 = self.X_20(x_10)
				x_30 = self.X_30(x_20)
				x_21 = self.up_30(x_30,x_20)
				x_12 = self.up_31(x_21,x_10)
				x_03 = self.up_32(x_12,x_00)
				out_3 = self.out_3(x_03)
				return out_3

			if '4' in self.level:
				x_20 = self.X_20(x_10)
				x_30 = self.X_30(x_20)
				x_40 = self.X_40(x_30)
				x_31 = self.up_40(x_40,x_30)
				x_22 = self.up_41(x_31,x_20)
				x_13 = self.up_42(x_22,x_10)
				x_04 = self.up_43(x_13,x_00)
				out_4 = self.out_4(x_04)
				return out_4
		else:
			x_01 = self.up_1(x_10, x_00)
			out_1 = self.out_1(x_01)

			x_20 = self.X_20(x_10)
			x_11 = self.up_20(x_20,x_10)
			x_02 = self.up_21(x_11,x_00)
			out_2 = self.out_2(x_02)

			x_30 = self.X_30(x_20)
			x_21 = self.up_30(x_30,x_20)
			x_12 = self.up_31(x_21,x_10)
			x_03 = self.up_32(x_12,x_00)
			out_3 = self.out_3(x_03)

			x_40 = self.X_40(x_30)
			x_31 = self.up_40(x_40,x_30)
			x_22 = self.up_41(x_31,x_20)
			x_13 = self.up_42(x_22,x_10)
			x_04 = self.up_43(x_13,x_00)
			out_4 = self.out_4(x_04)
			return [out_1, out_2, out_3, out_4]
