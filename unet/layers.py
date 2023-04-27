import torch
import torch.nn as nn
import torch.nn.functional as F
# from .init_weights import init_weights

class depth_seperate_conv(nn.Module):
    """docstring for depth_seperate_conv"""
    def __init__(self, ch_in, ch_out, kernel_size, padding):
        super(depth_seperate_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, padding=padding, groups=ch_in),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, groups=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetConv2, self).__init__()
        self.double_conv = nn.Sequential(
            # depth_seperate_conv(in_size, out_size,kernel_size=3, padding=1),
            # depth_seperate_conv(out_size, out_size,kernel_size=3, padding=1)
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(out_size * 2, out_size)
        
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')


    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('unetConv2') != -1: continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)

        diffY = input[0].size()[2] - outputs0.size()[2]
        diffX = input[0].size()[3] - outputs0.size()[3]

        outputs0 = F.pad(outputs0, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)