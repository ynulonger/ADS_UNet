""" Parts of the U-Net model """
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=1,kernel_size=1,padding=0),
            nn.BatchNorm2d(1))
    def forward(self,x):
        x=self.conv(x)
        x=torch.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0),
            nn.BatchNorm2d(int(out_channels/2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.activation = nn.ReLU()

    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        #print('channel',x.size())
        x=self.conv1(x)
        x=self.activation(x)
        x=self.conv2(x)
        x=torch.sigmoid(x)
        return x

class SCSE_Block(nn.Module):
    def __init__(self, out_channels):
        super(SCSE_Block, self).__init__()
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x):
        g1 = self.spatial_gate(x)
        # print('g1',g1.size())
        g2 = self.channel_gate(x)
        # print('g2',g2.size())
        x = g1 * x + g2 * x
        return x

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

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # depth_seperate_conv(in_channels, out_channels,kernel_size=3, padding=1),
            # depth_seperate_conv(out_channels, out_channels,kernel_size=3, padding=1)
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DoubleConv_2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_down(x)

class Up_skip_2(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, middle_channels, out_channels, skip_option='none'):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.skip_option = skip_option
        if self.skip_option == 'scse':
            self.scse = SCSE_Block(middle_channels)

        self.up = nn.ConvTranspose2d(in_channels , middle_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv_2(2*middle_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        if self.skip_option=='None':
            x = torch.cat([x2, x1], dim=1)
        elif self.skip_option=='scse':
            x = torch.cat([self.scse(x2), x1],dim=1)
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, classes, scale_factor):
        super(Up_2, self).__init__()

        # self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=kernel_size, stride=kernel_size)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class Up_noskip(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels) 

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class Up_skip(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, feature_width, skip_option='None'):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.skip_option = skip_option
        if self.skip_option == 'scalar':
            self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        if self.skip_option == 'diag':
            diag = torch.randn(out_channels,1, feature_width)
            self.beta = nn.Parameter(diag, requires_grad=True)
        if self.skip_option == 'diag1':
            diag = torch.randn(out_channels,1, feature_width)
            inverse_diag = torch.randn(out_channels, feature_width,1)
            self.beta = nn.Parameter(diag, requires_grad=True)
            self.beta_1 = nn.Parameter(inverse_diag, requires_grad=True)
        if self.skip_option == 'scse':
            self.scse = SCSE_Block(in_channels//2)

        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        if self.skip_option =='scalar':
            x = torch.cat([F.relu(x2*self.beta), x1], dim=1)
        elif self.skip_option =='diag':
            x = torch.cat([F.relu(x2*self.beta), x1],dim=1)
        elif self.skip_option =='diag1':
            x = torch.cat([F.relu(x2*self.beta + self.beta_1*x2), x1],dim=1)
        elif self.skip_option=='None':
            x = torch.cat([x2, x1], dim=1)
        elif self.skip_option=='scse':
            x = torch.cat([self.scse(x2), x1],dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CENet_Up_1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(CENet_Up_1, self).__init__()
        self.out_channels = out_channels
        self.up_1 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2),
                        # nn.Conv2d(in_channels, 42, kernel_size=3, padding=0, groups=1)
                        nn.Conv2d(in_channels, 171, kernel_size=3, padding=0, groups=1)
                    )
        self.up_2 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2),
                        # nn.Conv2d(in_channels, 43, kernel_size=1, padding=0, groups=1)
                        nn.Conv2d(in_channels, 170, kernel_size=1, padding=0, groups=1)
                    )

        self.conv = DoubleConv(in_channels, self.out_channels)
        # self.conv_1_1 = nn.Conv2d(in_channels//2, 43, kernel_size=1, padding=0, groups=1)
        self.conv_1_1 = nn.Conv2d(in_channels//2, 171, kernel_size=1, padding=0, groups=1)


    def forward(self, x1, x2, x3):
        x1 = self.up_1(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = self.conv_1_1(x2)
        x3 = self.up_2(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.conv(x)
        # H_out =(H_in−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
class CENet_Up_2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(CENet_Up_2, self).__init__()
        self.out_channels = out_channels
        # if bilinear, use the normal convolutions to reduce the number of channels
        # self.up_1 = nn.ConvTranspose2d(in_channels , 64, kernel_size=2, stride=2, dilation=1)
        # self.up_2 = nn.ConvTranspose2d(in_channels , 64, kernel_size=2, stride=2, dilation=1)
        # self.up_3 = nn.ConvTranspose2d(in_channels*2 , 64, kernel_size=2, stride=4, dilation=3)

        self.up_1 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
                        # nn.Conv2d(in_channels, 16, kernel_size=3, padding=0, groups=1)
                        nn.Conv2d(in_channels, 64, kernel_size=3, padding=0, groups=1)
                    )
        self.up_2 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
                        # nn.Conv2d(in_channels, 16, kernel_size=1, padding=0, groups=1)
                        nn.Conv2d(in_channels, 64, kernel_size=1, padding=0, groups=1)
                    )
        self.up_3 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=4, dilation=3),
                        # nn.Conv2d(in_channels, 16, kernel_size=1, padding=0, groups=1)
                        nn.Conv2d(in_channels, 64, kernel_size=1, padding=0, groups=1)
                    )

        self.conv = DoubleConv(in_channels, self.out_channels)
        # self.conv_1_1 = nn.Conv2d(in_channels//2, 16, kernel_size=1, padding=0, groups=1)
        self.conv_1_1 = nn.Conv2d(in_channels//2, 64, kernel_size=1, padding=0, groups=1)

    def forward(self, x1, x2, x3, x4):
        x1 = self.up_1(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = self.conv_1_1(x2)
        x3 = self.up_2(x3)
        x4 = self.up_3(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)
    
class CENet_Up_3(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(CENet_Up_3, self).__init__()
        self.out_channels = out_channels
        # if bilinear, use the normal convolutions to reduce the number of channels
        # self.up_1 = nn.ConvTranspose2d(in_channels , 26, kernel_size=2, stride=2, dilation=1)
        # self.up_2 = nn.ConvTranspose2d(in_channels , 26, kernel_size=2, stride=2, dilation=1)
        # self.up_3 = nn.ConvTranspose2d(in_channels*2 , 26, kernel_size=2, stride=4, dilation=3)
        # self.up_4 = nn.ConvTranspose2d(in_channels*4 , 25, kernel_size=2, stride=8, dilation=7)
        self.up_1 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2),
                        # nn.Conv2d(in_channels, 6, kernel_size=3, padding=0, groups=1)
                        nn.Conv2d(in_channels, 26, kernel_size=3, padding=0, groups=1)
                    )
        self.up_2 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2),
                        # nn.Conv2d(in_channels, 6, kernel_size=1, padding=0, groups=1)
                        nn.Conv2d(in_channels, 26, kernel_size=1, padding=0, groups=1)
                    )
        self.up_3 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels*2 , in_channels, kernel_size=2, stride=4, dilation=3),
                        # nn.Conv2d(in_channels, 6, kernel_size=1, padding=0, groups=1)
                        nn.Conv2d(in_channels, 26, kernel_size=1, padding=0, groups=1)
                    )
        self.up_4 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels*4 , in_channels, kernel_size=2, stride=8, dilation=7),
                        # nn.Conv2d(in_channels, 7, kernel_size=1, padding=0, groups=1)
                        nn.Conv2d(in_channels, 25, kernel_size=1, padding=0, groups=1)
                    )
        self.conv = DoubleConv(in_channels, self.out_channels)
        # self.conv_1_1 = nn.Conv2d(in_channels//2, 7, kernel_size=1, padding=0, groups=1)
        self.conv_1_1 = nn.Conv2d(in_channels//2, 25, kernel_size=1, padding=0, groups=1)

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.up_1(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = self.conv_1_1(x2)
        x3 = self.up_2(x3)
        x4 = self.up_3(x4)
        x5 = self.up_4(x5)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.conv(x)