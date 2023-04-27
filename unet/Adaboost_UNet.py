""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from .unet_parts import *
from torch.nn import init
from thop import *
import torch.nn.functional as F
from torchsummary import summary


# nnUNetTrainerBCSS_DP__nnUNetPlansv2.1
class AdaBoost_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, level, skip_option='None', filters=32, deep_sup=True, head=None):
        super(AdaBoost_UNet, self).__init__()
        self.width = 512
        self.level = level
        self.n_channels = n_channels
        self.deep_sup = deep_sup
        self.head = head
        if n_classes==1:
            self.n_classes = 2
        else:
            self.n_classes = n_classes
        self.X_00 = DoubleConv(n_channels, filters)
        self.X_10 = Down(filters, 2*filters)
        self.X_20 = Down(2*filters, 4*filters)
        self.X_30 = Down(4*filters, 8*filters)
        self.X_40 = Down(8*filters, 16*filters)

        #UNet level 1
        self.up_10  = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_01 = OutConv(filters, self.n_classes)

        #UNet level 2
        self.up_20 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_11 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_02 = OutConv(filters, self.n_classes)

        #UNet level 3
        self.up_30 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.up_21 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_12 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_03 = OutConv(filters, self.n_classes)

        #UNet level 4
        self.up_40 = Up_skip(16*filters, 8*filters, self.width//8, skip_option)
        self.up_31 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.up_22 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_13 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_04 = OutConv(filters, self.n_classes)

        if self.deep_sup:
            #UNet level 1
            self.out_10= OutConv(2*filters, self.n_classes)
            self.alpha_1 = nn.Parameter(torch.tensor(np.reshape(np.array([1/2]*2), [1,2]),dtype=torch.float32, requires_grad=True))
            #UNet level 2
            self.out_20= OutConv(4*filters, self.n_classes)
            self.out_11= OutConv(2*filters, self.n_classes)
            self.alpha_2 = nn.Parameter(torch.tensor(np.reshape(np.array([1/3]*3), [1,3]),dtype=torch.float32, requires_grad=True))
            #UNet level 3
            self.out_30= OutConv(8*filters, self.n_classes)
            self.out_21= OutConv(4*filters, self.n_classes)
            self.out_12= OutConv(2*filters, self.n_classes)
            self.alpha_3 = nn.Parameter(torch.tensor(np.reshape(np.array([1/4]*4), [1,4]),dtype=torch.float32, requires_grad=True))
            #UNet level 4
            self.out_40= OutConv(16*filters, self.n_classes)
            self.out_31= OutConv(8*filters, self.n_classes)
            self.out_22= OutConv(4*filters, self.n_classes)
            self.out_13= OutConv(2*filters, self.n_classes)
            self.alpha_4 = nn.Parameter(torch.tensor(np.reshape(np.array([1/5]*5), [1,5]),dtype=torch.float32, requires_grad=True))

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
                x_01 = self.up_10(x_10, x_00)
                out_01 = self.out_01(x_01)
                if self.deep_sup:
                    out_10 = self.out_10(x_10)
                    return [out_10, out_01], self.alpha_1
                else:
                    return out_01

            if '2' in self.level:
                x_20 = self.X_20(x_10)
                x_11 = self.up_20(x_20,x_10)
                x_02 = self.up_11(x_11,x_00)
                out_02 = self.out_02(x_02)
                if self.deep_sup:
                    out_20  = self.out_20(x_20)
                    out_11 = self.out_11(x_11)
                    return [out_20, out_11, out_02], self.alpha_2
                else:
                    return out_02

            if '3' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_21 = self.up_30(x_30,x_20)
                x_12 = self.up_21(x_21,x_10)
                x_03 = self.up_12(x_12,x_00)
                out_03 = self.out_03(x_03)
                if self.deep_sup:
                    out_30 = self.out_30(x_30)
                    out_21 = self.out_21(x_21)
                    out_12 = self.out_12(x_12)
                    return [out_30, out_21, out_12, out_03], self.alpha_3
                else:
                    return out_03

            if '4' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_31 = self.up_40(x_40,x_30)
                x_22 = self.up_31(x_31,x_20)
                x_13 = self.up_22(x_22,x_10)
                x_04 = self.up_13(x_13,x_00)
                out_04 = self.out_04(x_04)
                if self.deep_sup:
                    out_40 = self.out_40(x_40)
                    out_31 = self.out_31(x_31)
                    out_22 = self.out_22(x_22)
                    out_13 = self.out_13(x_13)
                    return [out_40, out_31, out_22, out_13, out_04], self.alpha_4
                else:
                    return out_04

        elif self.level=='1234_pred':
            x_01 = self.up_10(x_10, x_00)
            out_01 = self.out_01(x_01)

            x_20 = self.X_20(x_10)
            x_11 = self.up_20(x_20,x_10)
            x_02 = self.up_11(x_11,x_00)
            out_02 = self.out_02(x_02)

            x_30 = self.X_30(x_20)
            x_21 = self.up_30(x_30,x_20)
            x_12 = self.up_21(x_21,x_10)
            x_03 = self.up_12(x_12,x_00)
            out_03 = self.out_03(x_03)

            x_40 = self.X_40(x_30)
            x_31 = self.up_40(x_40,x_30)
            x_22 = self.up_31(x_31,x_20)
            x_13 = self.up_22(x_22,x_10)
            x_04 = self.up_13(x_13,x_00)
            out_04 = self.out_04(x_04)
            return [out_01, out_02, out_03, out_04]

        elif self.level=='1234_feat':
            x_01 = self.up_10(x_10, x_00)

            x_20 = self.X_20(x_10)
            x_11 = self.up_20(x_20,x_10)
            x_02 = self.up_11(x_11,x_00)

            x_30 = self.X_30(x_20)
            x_21 = self.up_30(x_30,x_20)
            x_12 = self.up_21(x_21,x_10)
            x_03 = self.up_12(x_12,x_00)

            x_40 = self.X_40(x_30)
            x_31 = self.up_40(x_40,x_30)
            x_22 = self.up_31(x_31,x_20)
            x_13 = self.up_22(x_22,x_10)
            x_04 = self.up_13(x_13,x_00)
            return [x_01, x_02, x_03, x_04]

        elif isinstance(self.level, list):
            x_01 = self.up_10(x_10, x_00)
            out_10 = self.out_10(x_10)
            out_01 = self.out_01(x_01)

            x_20 = self.X_20(x_10)
            x_11 = self.up_20(x_20,x_10)
            x_02 = self.up_11(x_11,x_00)
            out_20  = self.out_20(x_20)
            out_11 = self.out_11(x_11)
            out_02 = self.out_02(x_02)

            x_30 = self.X_30(x_20)
            x_21 = self.up_30(x_30,x_20)
            x_12 = self.up_21(x_21,x_10)
            x_03 = self.up_12(x_12,x_00)
            out_30 = self.out_30(x_30)
            out_21 = self.out_21(x_21)
            out_12 = self.out_12(x_12)
            out_03 = self.out_03(x_03)

            x_40 = self.X_40(x_30)
            x_31 = self.up_40(x_40,x_30)
            x_22 = self.up_31(x_31,x_20)
            x_13 = self.up_22(x_22,x_10)
            x_04 = self.up_13(x_13,x_00)
            out_40 = self.out_40(x_40)
            out_31 = self.out_31(x_31)
            out_22 = self.out_22(x_22)
            out_13 = self.out_13(x_13)
            out_04 = self.out_04(x_04)

            pred_list = {1:[self.alpha_1, [out_10, out_01]],
                         2:[self.alpha_2, [out_20, out_11, out_02]],
                         3:[self.alpha_3, [out_30, out_21, out_12, out_03]],
                         4:[self.alpha_4, [out_40, out_31, out_22, out_13, out_04]]
                        }
            return pred_list

        elif self.level=='outer':
            x_20 = self.X_20(x_10)
            x_30 = self.X_30(x_20)
            x_40 = self.X_40(x_30)
            x_31 = self.up_40(x_40,x_30)
            x_22 = self.up_31(x_31,x_20)
            x_13 = self.up_22(x_22,x_10)
            x_04 = self.up_13(x_13,x_00)
            return [x_00, x_10, x_20, x_30, x_40, x_31, x_22, x_13, x_04]

        elif self.level=='all':
            x_01 = self.up_10(x_10, x_00)
            x_20 = self.X_20(x_10)
            x_11 = self.up_20(x_20,x_10)
            x_02 = self.up_11(x_11,x_00)
            x_30 = self.X_30(x_20)
            x_21 = self.up_30(x_30,x_20)
            x_12 = self.up_21(x_21,x_10)
            x_03 = self.up_12(x_12,x_00)
            x_40 = self.X_40(x_30)
            x_31 = self.up_40(x_40,x_30)
            x_22 = self.up_31(x_31,x_20)
            x_13 = self.up_22(x_22,x_10)
            x_04 = self.up_13(x_13,x_00)
            pred_list = [x_00, x_01, x_02, x_03, x_04, x_10, x_11, x_12, x_13, x_20, x_21, x_22, x_30, x_31, x_40]
            return pred_list

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


class AdaBoost_UNet_deeper(nn.Module):
    def __init__(self, n_channels, n_classes, level, skip_option='None', filters=32, deep_sup=True, head=None):
        super(AdaBoost_UNet_deeper, self).__init__()
        self.width = 512
        self.level = level
        self.n_channels = n_channels
        self.deep_sup = deep_sup
        self.head = head
        if n_classes==1:
            self.n_classes = 2
        else:
            self.n_classes = n_classes
        self.X_00 = DoubleConv(n_channels, filters)
        self.X_10 = Down(filters, 2*filters)
        self.X_20 = Down(2*filters, 4*filters)
        self.X_30 = Down(4*filters, 8*filters)
        self.X_40 = Down(8*filters, 16*filters)
        self.X_50 = Down(16*filters, 32*filters)
        self.X_60 = Down(32*filters, 64*filters)

        #UNet level 1
        self.up_10  = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_01 = OutConv(filters, self.n_classes)

        #UNet level 2
        self.up_20 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_11 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_02 = OutConv(filters, self.n_classes)

        #UNet level 3
        self.up_30 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.up_21 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_12 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_03 = OutConv(filters, self.n_classes)

        #UNet level 4
        self.up_40 = Up_skip(16*filters, 8*filters, self.width//8, skip_option)
        self.up_31 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.up_22 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_13 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_04 = OutConv(filters, self.n_classes)

        #UNet level 5   
        self.up_50 = Up_skip(32*filters, 16*filters, self.width//16, skip_option)
        self.up_41 = Up_skip(16*filters, 8*filters, self.width//8, skip_option)
        self.up_32 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.up_23 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_14 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_05 = OutConv(filters, self.n_classes)

        #UNet level 6
        self.up_60 = Up_skip(64*filters, 32*filters, self.width//32, skip_option)   
        self.up_51 = Up_skip(32*filters, 16*filters, self.width//16, skip_option)
        self.up_42 = Up_skip(16*filters, 8*filters, self.width//8, skip_option)
        self.up_33 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.up_24 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_15 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_06 = OutConv(filters, self.n_classes)

        if self.deep_sup:
            #UNet level 1
            self.out_10= OutConv(2*filters, self.n_classes)
            self.alpha_1 = nn.Parameter(torch.tensor(np.reshape(np.array([1/2]*2), [1,2]),dtype=torch.float32, requires_grad=True))
            #UNet level 2
            self.out_20= OutConv(4*filters, self.n_classes)
            self.out_11= OutConv(2*filters, self.n_classes)
            self.alpha_2 = nn.Parameter(torch.tensor(np.reshape(np.array([1/3]*3), [1,3]),dtype=torch.float32, requires_grad=True))
            #UNet level 3
            self.out_30= OutConv(8*filters, self.n_classes)
            self.out_21= OutConv(4*filters, self.n_classes)
            self.out_12= OutConv(2*filters, self.n_classes)
            self.alpha_3 = nn.Parameter(torch.tensor(np.reshape(np.array([1/4]*4), [1,4]),dtype=torch.float32, requires_grad=True))
            #UNet level 4
            self.out_40= OutConv(16*filters, self.n_classes)
            self.out_31= OutConv(8*filters, self.n_classes)
            self.out_22= OutConv(4*filters, self.n_classes)
            self.out_13= OutConv(2*filters, self.n_classes)
            self.alpha_4 = nn.Parameter(torch.tensor(np.reshape(np.array([1/5]*5), [1,5]),dtype=torch.float32, requires_grad=True))
            #UNet level 5
            self.out_50= OutConv(32*filters, self.n_classes)
            self.out_41= OutConv(16*filters, self.n_classes)
            self.out_32= OutConv(8*filters, self.n_classes)
            self.out_23= OutConv(4*filters, self.n_classes)
            self.out_14= OutConv(2*filters, self.n_classes)
            self.alpha_5 = nn.Parameter(torch.tensor(np.reshape(np.array([1/6]*6), [1,6]),dtype=torch.float32, requires_grad=True))
            #UNet level 6
            self.out_60= OutConv(64*filters, self.n_classes)
            self.out_51= OutConv(32*filters, self.n_classes)
            self.out_42= OutConv(16*filters, self.n_classes)
            self.out_33= OutConv(8*filters, self.n_classes)
            self.out_24= OutConv(4*filters, self.n_classes)
            self.out_15= OutConv(2*filters, self.n_classes)
            self.alpha_6 = nn.Parameter(torch.tensor(np.reshape(np.array([1/7]*7), [1,7]),dtype=torch.float32, requires_grad=True))

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
                x_01 = self.up_10(x_10, x_00)
                out_01 = self.out_01(x_01)
                if self.deep_sup:
                    out_10 = self.out_10(x_10)
                    return [out_10, out_01], self.alpha_1
                else:
                    return out_01

            if '2' in self.level:
                x_20 = self.X_20(x_10)
                x_11 = self.up_20(x_20,x_10)
                x_02 = self.up_11(x_11,x_00)
                out_02 = self.out_02(x_02)
                if self.deep_sup:
                    out_20  = self.out_20(x_20)
                    out_11 = self.out_11(x_11)
                    return [out_20, out_11, out_02], self.alpha_2
                else:
                    return out_02

            if '3' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_21 = self.up_30(x_30,x_20)
                x_12 = self.up_21(x_21,x_10)
                x_03 = self.up_12(x_12,x_00)
                out_03 = self.out_03(x_03)
                if self.deep_sup:
                    out_30 = self.out_30(x_30)
                    out_21 = self.out_21(x_21)
                    out_12 = self.out_12(x_12)
                    return [out_30, out_21, out_12, out_03], self.alpha_3
                else:
                    return out_03

            if '4' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_31 = self.up_40(x_40,x_30)
                x_22 = self.up_31(x_31,x_20)
                x_13 = self.up_22(x_22,x_10)
                x_04 = self.up_13(x_13,x_00)
                out_04 = self.out_04(x_04)
                if self.deep_sup:
                    out_40 = self.out_40(x_40)
                    out_31 = self.out_31(x_31)
                    out_22 = self.out_22(x_22)
                    out_13 = self.out_13(x_13)
                    return [out_40, out_31, out_22, out_13, out_04], self.alpha_4
                else:
                    return out_04

            if '5' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_50 = self.X_50(x_40)
                x_41 = self.up_50(x_50,x_40)
                x_32 = self.up_41(x_41,x_30)
                x_23 = self.up_32(x_32,x_20)
                x_14 = self.up_23(x_23,x_10)
                x_05 = self.up_14(x_14,x_00)
                out_05 = self.out_05(x_05)
                if self.deep_sup:
                    out_50 = self.out_50(x_50)
                    out_41 = self.out_41(x_41)
                    out_32 = self.out_32(x_32)
                    out_23 = self.out_23(x_23)
                    out_14 = self.out_14(x_14)
                    return [out_50, out_41, out_32, out_23, out_14, out_05], self.alpha_5
                else:
                    return out_05

            if '6' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_50 = self.X_50(x_40)
                x_60 = self.X_60(x_50)
                x_51 = self.up_60(x_60,x_50)
                x_42 = self.up_51(x_51,x_40)
                x_33 = self.up_42(x_42,x_30)
                x_24 = self.up_33(x_33,x_20)
                x_15 = self.up_24(x_24,x_10)
                x_06 = self.up_15(x_15,x_00)
                out_06 = self.out_06(x_06)
                if self.deep_sup:
                    out_60 = self.out_60(x_60)
                    out_51 = self.out_51(x_51)
                    out_42 = self.out_42(x_42)
                    out_33 = self.out_33(x_33)
                    out_24 = self.out_24(x_24)
                    out_15 = self.out_15(x_15)
                    return [out_60, out_51, out_42, out_33, out_24, out_15, out_06], self.alpha_6
                else:
                    return out_06

        elif self.level=='1234_pred':
            x_01 = self.up_10(x_10, x_00)
            out_01 = self.out_01(x_01)

            x_20 = self.X_20(x_10)
            x_11 = self.up_20(x_20,x_10)
            x_02 = self.up_11(x_11,x_00)
            out_02 = self.out_02(x_02)

            x_30 = self.X_30(x_20)
            x_21 = self.up_30(x_30,x_20)
            x_12 = self.up_21(x_21,x_10)
            x_03 = self.up_12(x_12,x_00)
            out_03 = self.out_03(x_03)

            x_40 = self.X_40(x_30)
            x_31 = self.up_40(x_40,x_30)
            x_22 = self.up_31(x_31,x_20)
            x_13 = self.up_22(x_22,x_10)
            x_04 = self.up_13(x_13,x_00)
            out_04 = self.out_04(x_04)
            return [out_01, out_02, out_03, out_04]

        elif self.level=='1234_feat':
            x_01 = self.up_10(x_10, x_00)

            x_20 = self.X_20(x_10)
            x_11 = self.up_20(x_20,x_10)
            x_02 = self.up_11(x_11,x_00)

            x_30 = self.X_30(x_20)
            x_21 = self.up_30(x_30,x_20)
            x_12 = self.up_21(x_21,x_10)
            x_03 = self.up_12(x_12,x_00)

            x_40 = self.X_40(x_30)
            x_31 = self.up_40(x_40,x_30)
            x_22 = self.up_31(x_31,x_20)
            x_13 = self.up_22(x_22,x_10)
            x_04 = self.up_13(x_13,x_00)
            return [x_01, x_02, x_03, x_04]

        elif isinstance(self.level, list):
            x_01 = self.up_10(x_10, x_00)
            out_10 = self.out_10(x_10)
            out_01 = self.out_01(x_01)

            x_20 = self.X_20(x_10)
            x_11 = self.up_20(x_20,x_10)
            x_02 = self.up_11(x_11,x_00)
            out_20  = self.out_20(x_20)
            out_11 = self.out_11(x_11)
            out_02 = self.out_02(x_02)

            x_30 = self.X_30(x_20)
            x_21 = self.up_30(x_30,x_20)
            x_12 = self.up_21(x_21,x_10)
            x_03 = self.up_12(x_12,x_00)
            out_30 = self.out_30(x_30)
            out_21 = self.out_21(x_21)
            out_12 = self.out_12(x_12)
            out_03 = self.out_03(x_03)

            x_40 = self.X_40(x_30)
            x_31 = self.up_40(x_40,x_30)
            x_22 = self.up_31(x_31,x_20)
            x_13 = self.up_22(x_22,x_10)
            x_04 = self.up_13(x_13,x_00)
            out_40 = self.out_40(x_40)
            out_31 = self.out_31(x_31)
            out_22 = self.out_22(x_22)
            out_13 = self.out_13(x_13)
            out_04 = self.out_04(x_04)

            x_50 = self.X_50(x_40)
            x_41 = self.up_50(x_50,x_40)
            x_32 = self.up_41(x_41,x_30)
            x_23 = self.up_32(x_32,x_20)
            x_14 = self.up_23(x_23,x_10)
            x_05 = self.up_14(x_14,x_00)
            out_50 = self.out_50(x_50)
            out_41 = self.out_41(x_41)
            out_32 = self.out_32(x_32)
            out_23 = self.out_23(x_23)
            out_14 = self.out_14(x_14)
            out_05 = self.out_05(x_05)

            x_60 = self.X_60(x_50)
            x_51 = self.up_60(x_60,x_50)
            x_42 = self.up_51(x_51,x_40)
            x_33 = self.up_42(x_42,x_30)
            x_24 = self.up_33(x_33,x_20)
            x_15 = self.up_24(x_24,x_10)
            x_06 = self.up_15(x_15,x_00)
            out_60 = self.out_60(x_60)
            out_51 = self.out_51(x_51)
            out_42 = self.out_42(x_42)
            out_33 = self.out_33(x_33)
            out_24 = self.out_24(x_24)
            out_15 = self.out_15(x_15)
            out_06 = self.out_06(x_06)

            pred_list = {1:[self.alpha_1, [out_10, out_01]],
                         2:[self.alpha_2, [out_20, out_11, out_02]],
                         3:[self.alpha_3, [out_30, out_21, out_12, out_03]],
                         4:[self.alpha_4, [out_40, out_31, out_22, out_13, out_04]],
                         5:[self.alpha_5, [out_50, out_41, out_32, out_23, out_14, out_05]],
                         6:[self.alpha_6, [out_60, out_51, out_42, out_33, out_24, out_15, out_06]]
                        }
            return pred_list

class AdaBoost_UNet_d_BCSS(nn.Module):
    def __init__(self, n_channels, n_classes, level, skip_option='None', filters=32, deep_sup=True, head=None):
        super(AdaBoost_UNet_d_BCSS, self).__init__()
        self.width = 512
        self.level = level
        self.n_channels = n_channels
        self.deep_sup = deep_sup
        self.head = head
        if n_classes==1:
            self.n_classes = 2
        else:
            self.n_classes = n_classes
        self.X_00 = DoubleConv(n_channels, filters)
        self.X_10 = Down(filters, 2*filters)
        self.X_20 = Down(2*filters, 4*filters)
        self.X_30 = Down(4*filters, 8*filters)
        self.X_40 = Down(8*filters, 16*filters)
        self.X_50 = Down(16*filters, 32*filters)
        self.X_60 = Down(32*filters, 64*filters)
        self.X_70 = Down(64*filters, 128*filters)

        #UNet level 1
        self.up_10  = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_01 = OutConv(filters, self.n_classes)

        #UNet level 2
        self.up_20 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_11 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_02 = OutConv(filters, self.n_classes)

        #UNet level 3
        self.up_30 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.up_21 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_12 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_03 = OutConv(filters, self.n_classes)

        #UNet level 4
        self.up_40 = Up_skip(16*filters, 8*filters, self.width//8, skip_option)
        self.up_31 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.up_22 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_13 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_04 = OutConv(filters, self.n_classes)

        #UNet level 5   
        self.up_50 = Up_skip(32*filters, 16*filters, self.width//16, skip_option)
        self.up_41 = Up_skip(16*filters, 8*filters, self.width//8, skip_option)
        self.up_32 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.up_23 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_14 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_05 = OutConv(filters, self.n_classes)

        #UNet level 6
        self.up_60 = Up_skip(64*filters, 32*filters, self.width//32, skip_option)   
        self.up_51 = Up_skip(32*filters, 16*filters, self.width//16, skip_option)
        self.up_42 = Up_skip(16*filters, 8*filters, self.width//8, skip_option)
        self.up_33 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.up_24 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_15 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_06 = OutConv(filters, self.n_classes)

        #UNet level 7
        self.up_70 = Up_skip(128*filters, 64*filters, self.width//128, skip_option)   
        self.up_61 = Up_skip(64*filters, 32*filters, self.width//32, skip_option)   
        self.up_52 = Up_skip(32*filters, 16*filters, self.width//16, skip_option)
        self.up_43 = Up_skip(16*filters, 8*filters, self.width//8, skip_option)
        self.up_34 = Up_skip(8*filters, 4*filters, self.width//4, skip_option)
        self.up_25 = Up_skip(4*filters, 2*filters, self.width//2, skip_option)
        self.up_16 = Up_skip(2*filters, filters, self.width, skip_option)
        self.out_07 = OutConv(filters, self.n_classes)

        if self.deep_sup:
            #UNet level 1
            self.out_10= OutConv(2*filters, self.n_classes)
            self.alpha_1 = nn.Parameter(torch.tensor(np.reshape(np.array([1/2]*2), [1,2]),dtype=torch.float32, requires_grad=True))
            #UNet level 2
            self.out_20= OutConv(4*filters, self.n_classes)
            self.out_11= OutConv(2*filters, self.n_classes)
            self.alpha_2 = nn.Parameter(torch.tensor(np.reshape(np.array([1/3]*3), [1,3]),dtype=torch.float32, requires_grad=True))
            #UNet level 3
            self.out_30= OutConv(8*filters, self.n_classes)
            self.out_21= OutConv(4*filters, self.n_classes)
            self.out_12= OutConv(2*filters, self.n_classes)
            self.alpha_3 = nn.Parameter(torch.tensor(np.reshape(np.array([1/4]*4), [1,4]),dtype=torch.float32, requires_grad=True))
            #UNet level 4
            self.out_40= OutConv(16*filters, self.n_classes)
            self.out_31= OutConv(8*filters, self.n_classes)
            self.out_22= OutConv(4*filters, self.n_classes)
            self.out_13= OutConv(2*filters, self.n_classes)
            self.alpha_4 = nn.Parameter(torch.tensor(np.reshape(np.array([1/5]*5), [1,5]),dtype=torch.float32, requires_grad=True))
            #UNet level 5
            self.out_50= OutConv(32*filters, self.n_classes)
            self.out_41= OutConv(16*filters, self.n_classes)
            self.out_32= OutConv(8*filters, self.n_classes)
            self.out_23= OutConv(4*filters, self.n_classes)
            self.out_14= OutConv(2*filters, self.n_classes)
            self.alpha_5 = nn.Parameter(torch.tensor(np.reshape(np.array([1/6]*6), [1,6]),dtype=torch.float32, requires_grad=True))
            #UNet level 6
            self.out_60= OutConv(64*filters, self.n_classes)
            self.out_51= OutConv(32*filters, self.n_classes)
            self.out_42= OutConv(16*filters, self.n_classes)
            self.out_33= OutConv(8*filters, self.n_classes)
            self.out_24= OutConv(4*filters, self.n_classes)
            self.out_15= OutConv(2*filters, self.n_classes)
            self.alpha_6 = nn.Parameter(torch.tensor(np.reshape(np.array([1/7]*7), [1,7]),dtype=torch.float32, requires_grad=True))
            #UNet level 7
            self.out_70= OutConv(128*filters, self.n_classes)
            self.out_61= OutConv(64*filters, self.n_classes)
            self.out_52= OutConv(32*filters, self.n_classes)
            self.out_43= OutConv(16*filters, self.n_classes)
            self.out_34= OutConv(8*filters, self.n_classes)
            self.out_25= OutConv(4*filters, self.n_classes)
            self.out_16= OutConv(2*filters, self.n_classes)
            self.alpha_7 = nn.Parameter(torch.tensor(np.reshape(np.array([1/8]*8), [1,8]),dtype=torch.float32, requires_grad=True))

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
                x_01 = self.up_10(x_10, x_00)
                out_01 = self.out_01(x_01)
                if self.deep_sup:
                    out_10 = self.out_10(x_10)
                    return [out_10, out_01], self.alpha_1
                else:
                    return out_01

            if '2' in self.level:
                x_20 = self.X_20(x_10)
                x_11 = self.up_20(x_20,x_10)
                x_02 = self.up_11(x_11,x_00)
                out_02 = self.out_02(x_02)
                if self.deep_sup:
                    out_20  = self.out_20(x_20)
                    out_11 = self.out_11(x_11)
                    return [out_20, out_11, out_02], self.alpha_2
                else:
                    return out_02

            if '3' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_21 = self.up_30(x_30,x_20)
                x_12 = self.up_21(x_21,x_10)
                x_03 = self.up_12(x_12,x_00)
                out_03 = self.out_03(x_03)
                if self.deep_sup:
                    out_30 = self.out_30(x_30)
                    out_21 = self.out_21(x_21)
                    out_12 = self.out_12(x_12)
                    return [out_30, out_21, out_12, out_03], self.alpha_3
                else:
                    return out_03

            if '4' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_31 = self.up_40(x_40,x_30)
                x_22 = self.up_31(x_31,x_20)
                x_13 = self.up_22(x_22,x_10)
                x_04 = self.up_13(x_13,x_00)
                out_04 = self.out_04(x_04)
                if self.deep_sup:
                    out_40 = self.out_40(x_40)
                    out_31 = self.out_31(x_31)
                    out_22 = self.out_22(x_22)
                    out_13 = self.out_13(x_13)
                    return [out_40, out_31, out_22, out_13, out_04], self.alpha_4
                else:
                    return out_04

            if '5' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_50 = self.X_50(x_40)
                x_41 = self.up_50(x_50,x_40)
                x_32 = self.up_41(x_41,x_30)
                x_23 = self.up_32(x_32,x_20)
                x_14 = self.up_23(x_23,x_10)
                x_05 = self.up_14(x_14,x_00)
                out_05 = self.out_05(x_05)
                if self.deep_sup:
                    out_50 = self.out_50(x_50)
                    out_41 = self.out_41(x_41)
                    out_32 = self.out_32(x_32)
                    out_23 = self.out_23(x_23)
                    out_14 = self.out_14(x_14)
                    return [out_50, out_41, out_32, out_23, out_14, out_05], self.alpha_5
                else:
                    return out_05

            if '6' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_50 = self.X_50(x_40)
                x_60 = self.X_60(x_50)
                x_51 = self.up_60(x_60,x_50)
                x_42 = self.up_51(x_51,x_40)
                x_33 = self.up_42(x_42,x_30)
                x_24 = self.up_33(x_33,x_20)
                x_15 = self.up_24(x_24,x_10)
                x_06 = self.up_15(x_15,x_00)
                out_06 = self.out_06(x_06)
                if self.deep_sup:
                    out_60 = self.out_60(x_60)
                    out_51 = self.out_51(x_51)
                    out_42 = self.out_42(x_42)
                    out_33 = self.out_33(x_33)
                    out_24 = self.out_24(x_24)
                    out_15 = self.out_15(x_15)
                    return [out_60, out_51, out_42, out_33, out_24, out_15, out_06], self.alpha_6
                else:
                    return out_06

            if '7' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_50 = self.X_50(x_40)
                x_60 = self.X_60(x_50)
                x_70 = self.X_70(x_60)
                x_61 = self.up_70(x_70,x_60)
                x_52 = self.up_61(x_61,x_50)
                x_43 = self.up_52(x_52,x_40)
                x_34 = self.up_43(x_43,x_30)
                x_25 = self.up_34(x_34,x_20)
                x_16 = self.up_25(x_25,x_10)
                x_07 = self.up_16(x_16,x_00)
                out_07 = self.out_07(x_07)
                if self.deep_sup:
                    out_70 = self.out_70(x_70)
                    out_61 = self.out_61(x_61)
                    out_52 = self.out_52(x_52)
                    out_43 = self.out_43(x_43)
                    out_34 = self.out_34(x_34)
                    out_25 = self.out_25(x_25)
                    out_16 = self.out_16(x_16)
                    return [out_70, out_61, out_52, out_43, out_34, out_25, out_16, out_07], self.alpha_7
                else:
                    return out_07

        elif isinstance(self.level, list):
            x_01 = self.up_10(x_10, x_00)
            out_10 = self.out_10(x_10)
            out_01 = self.out_01(x_01)

            x_20 = self.X_20(x_10)
            x_11 = self.up_20(x_20,x_10)
            x_02 = self.up_11(x_11,x_00)
            out_20  = self.out_20(x_20)
            out_11 = self.out_11(x_11)
            out_02 = self.out_02(x_02)

            x_30 = self.X_30(x_20)
            x_21 = self.up_30(x_30,x_20)
            x_12 = self.up_21(x_21,x_10)
            x_03 = self.up_12(x_12,x_00)
            out_30 = self.out_30(x_30)
            out_21 = self.out_21(x_21)
            out_12 = self.out_12(x_12)
            out_03 = self.out_03(x_03)

            x_40 = self.X_40(x_30)
            x_31 = self.up_40(x_40,x_30)
            x_22 = self.up_31(x_31,x_20)
            x_13 = self.up_22(x_22,x_10)
            x_04 = self.up_13(x_13,x_00)
            out_40 = self.out_40(x_40)
            out_31 = self.out_31(x_31)
            out_22 = self.out_22(x_22)
            out_13 = self.out_13(x_13)
            out_04 = self.out_04(x_04)

            x_50 = self.X_50(x_40)
            x_41 = self.up_50(x_50,x_40)
            x_32 = self.up_41(x_41,x_30)
            x_23 = self.up_32(x_32,x_20)
            x_14 = self.up_23(x_23,x_10)
            x_05 = self.up_14(x_14,x_00)
            out_50 = self.out_50(x_50)
            out_41 = self.out_41(x_41)
            out_32 = self.out_32(x_32)
            out_23 = self.out_23(x_23)
            out_14 = self.out_14(x_14)
            out_05 = self.out_05(x_05)

            x_60 = self.X_60(x_50)
            x_51 = self.up_60(x_60,x_50)
            x_42 = self.up_51(x_51,x_40)
            x_33 = self.up_42(x_42,x_30)
            x_24 = self.up_33(x_33,x_20)
            x_15 = self.up_24(x_24,x_10)
            x_06 = self.up_15(x_15,x_00)
            out_60 = self.out_60(x_60)
            out_51 = self.out_51(x_51)
            out_42 = self.out_42(x_42)
            out_33 = self.out_33(x_33)
            out_24 = self.out_24(x_24)
            out_15 = self.out_15(x_15)
            out_06 = self.out_06(x_06)

            x_70 = self.X_70(x_60)
            x_61 = self.up_70(x_70,x_60)
            x_52 = self.up_61(x_61,x_50)
            x_43 = self.up_52(x_52,x_40)
            x_34 = self.up_43(x_43,x_30)
            x_25 = self.up_34(x_34,x_20)
            x_16 = self.up_25(x_25,x_10)
            x_07 = self.up_16(x_16,x_00)
            out_70 = self.out_70(x_70)
            out_61 = self.out_61(x_61)
            out_52 = self.out_52(x_52)
            out_43 = self.out_43(x_43)
            out_34 = self.out_34(x_34)
            out_25 = self.out_25(x_25)
            out_16 = self.out_16(x_16)
            out_07 = self.out_07(x_07)

            pred_list = {1:[self.alpha_1, [out_10, out_01]],
                         2:[self.alpha_2, [out_20, out_11, out_02]],
                         3:[self.alpha_3, [out_30, out_21, out_12, out_03]],
                         4:[self.alpha_4, [out_40, out_31, out_22, out_13, out_04]],
                         5:[self.alpha_5, [out_50, out_41, out_32, out_23, out_14, out_05]],
                         6:[self.alpha_6, [out_60, out_51, out_42, out_33, out_24, out_15, out_06]],
                         7:[self.alpha_7, [out_70, out_61, out_52, out_43, out_34, out_25, out_16, out_07]]
                        }
            return pred_list

class AdaBoost_UNet_nn(nn.Module):
    def __init__(self, n_channels, n_classes, level, skip_option='None', filters=32, deep_sup=True, head=None):
        super(AdaBoost_UNet_nn, self).__init__()
        self.width = 512
        self.level = level
        self.n_channels = n_channels
        self.deep_sup = deep_sup
        self.head = head
        if n_classes==1:
            self.n_classes = 2
        else:
            self.n_classes = n_classes
        self.X_00 = DoubleConv_2(n_channels, filters)   #3->32->32
        self.X_10 = Down_Conv(filters, 2*filters)     #32->64->64
        self.X_20 = Down_Conv(2*filters, 4*filters)   #64->128->128
        self.X_30 = Down_Conv(4*filters, 8*filters)   #128->256->256
        self.X_40 = Down_Conv(8*filters, 323)         #256->323->323
        self.X_50 = Down_Conv(323, 323)               #323->323->323
        self.X_60 = Down_Conv(323, 323)               #323->323->323
        self.X_70 = Down_Conv(323, 323)               #323->323->323

        #UNet level 1
        self.up_10  = Up_skip_2(2*filters, filters, filters, skip_option)
        self.out_01 = OutConv(filters, self.n_classes)

        #UNet level 2
        self.up_20 = Up_skip_2(4*filters, 2*filters, 2*filters, skip_option)
        self.up_11 = Up_skip_2(2*filters, filters, filters, skip_option)
        self.out_02 = OutConv(filters, self.n_classes)

        #UNet level 3
        self.up_30 = Up_skip_2(8*filters, 4*filters, 4*filters, skip_option)
        self.up_21 = Up_skip_2(4*filters, 2*filters, 2*filters, skip_option)
        self.up_12 = Up_skip_2(2*filters, filters, filters, skip_option)
        self.out_03 = OutConv(filters, self.n_classes)

        #UNet level 4
        self.up_40 = Up_skip_2(323, 8*filters, 8*filters, skip_option)
        self.up_31 = Up_skip_2(8*filters, 4*filters, 4*filters, skip_option)
        self.up_22 = Up_skip_2(4*filters, 2*filters, 2*filters, skip_option)
        self.up_13 = Up_skip_2(2*filters, filters, filters, skip_option)
        self.out_04 = OutConv(filters, self.n_classes)

        #UNet level 5   
        self.up_50 = Up_skip_2(323, 323, 323, skip_option)
        self.up_41 = Up_skip_2(323, 8*filters, 8*filters, skip_option)
        self.up_32 = Up_skip_2(8*filters, 4*filters, 4*filters, skip_option)
        self.up_23 = Up_skip_2(4*filters, 2*filters, 2*filters, skip_option)
        self.up_14 = Up_skip_2(2*filters, filters, filters, skip_option)
        self.out_05 = OutConv(filters, self.n_classes)

        #UNet level 6
        self.up_60 = Up_skip_2(323, 323, 323, skip_option)   
        self.up_51 = Up_skip_2(323, 323, 323, skip_option)
        self.up_42 = Up_skip_2(323, 8*filters, 8*filters, skip_option)
        self.up_33 = Up_skip_2(8*filters, 4*filters, 4*filters, skip_option)
        self.up_24 = Up_skip_2(4*filters, 2*filters, 2*filters, skip_option)
        self.up_15 = Up_skip_2(2*filters, filters, filters, skip_option)
        self.out_06 = OutConv(filters, self.n_classes)

        #UNet level 7
        self.up_70 = Up_skip_2(323, 323, 323, skip_option)   
        self.up_61 = Up_skip_2(323, 323, 323, skip_option)
        self.up_52 = Up_skip_2(323, 323, 323, skip_option)
        self.up_43 = Up_skip_2(323, 8*filters, 8*filters, skip_option)
        self.up_34 = Up_skip_2(8*filters, 4*filters, 4*filters, skip_option)
        self.up_25 = Up_skip_2(4*filters, 2*filters, 2*filters, skip_option)
        self.up_16 = Up_skip_2(2*filters, filters, filters, skip_option)
        self.out_07 = OutConv(filters, self.n_classes)

        if self.deep_sup:
            #UNet level 1
            self.out_10= OutConv(2*filters, self.n_classes)
            self.alpha_1 = nn.Parameter(torch.tensor(np.reshape(np.array([1/2]*2), [1,2]),dtype=torch.float32, requires_grad=True))
            #UNet level 2
            self.out_20= OutConv(4*filters, self.n_classes)
            self.out_11= OutConv(2*filters, self.n_classes)
            self.alpha_2 = nn.Parameter(torch.tensor(np.reshape(np.array([1/3]*3), [1,3]),dtype=torch.float32, requires_grad=True))
            #UNet level 3
            self.out_30= OutConv(8*filters, self.n_classes)
            self.out_21= OutConv(4*filters, self.n_classes)
            self.out_12= OutConv(2*filters, self.n_classes)
            self.alpha_3 = nn.Parameter(torch.tensor(np.reshape(np.array([1/4]*4), [1,4]),dtype=torch.float32, requires_grad=True))
            #UNet level 4
            self.out_40= OutConv(323, self.n_classes)
            self.out_31= OutConv(8*filters, self.n_classes)
            self.out_22= OutConv(4*filters, self.n_classes)
            self.out_13= OutConv(2*filters, self.n_classes)
            self.alpha_4 = nn.Parameter(torch.tensor(np.reshape(np.array([1/5]*5), [1,5]),dtype=torch.float32, requires_grad=True))
            #UNet level 5
            self.out_50= OutConv(323, self.n_classes)
            self.out_41= OutConv(323, self.n_classes)
            self.out_32= OutConv(8*filters, self.n_classes)
            self.out_23= OutConv(4*filters, self.n_classes)
            self.out_14= OutConv(2*filters, self.n_classes)
            self.alpha_5 = nn.Parameter(torch.tensor(np.reshape(np.array([1/6]*6), [1,6]),dtype=torch.float32, requires_grad=True))
            #UNet level 6
            self.out_60= OutConv(323, self.n_classes)
            self.out_51= OutConv(323, self.n_classes)
            self.out_42= OutConv(323, self.n_classes)
            self.out_33= OutConv(8*filters, self.n_classes)
            self.out_24= OutConv(4*filters, self.n_classes)
            self.out_15= OutConv(2*filters, self.n_classes)
            self.alpha_6 = nn.Parameter(torch.tensor(np.reshape(np.array([1/7]*7), [1,7]),dtype=torch.float32, requires_grad=True))
            #UNet level 7
            self.out_70= OutConv(323, self.n_classes)
            self.out_61= OutConv(323, self.n_classes)
            self.out_52= OutConv(323, self.n_classes)
            self.out_43= OutConv(323, self.n_classes)
            self.out_34= OutConv(8*filters, self.n_classes)
            self.out_25= OutConv(4*filters, self.n_classes)
            self.out_16= OutConv(2*filters, self.n_classes)
            self.alpha_7 = nn.Parameter(torch.tensor(np.reshape(np.array([1/8]*8), [1,8]),dtype=torch.float32, requires_grad=True))

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
                x_01 = self.up_10(x_10, x_00)
                out_01 = self.out_01(x_01)
                if self.deep_sup:
                    out_10 = self.out_10(x_10)
                    return [out_10, out_01], self.alpha_1
                else:
                    return out_01

            if '2' in self.level:
                x_20 = self.X_20(x_10)
                x_11 = self.up_20(x_20,x_10)
                x_02 = self.up_11(x_11,x_00)
                out_02 = self.out_02(x_02)
                if self.deep_sup:
                    out_20  = self.out_20(x_20)
                    out_11 = self.out_11(x_11)
                    return [out_20, out_11, out_02], self.alpha_2
                else:
                    return out_02

            if '3' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_21 = self.up_30(x_30,x_20)
                x_12 = self.up_21(x_21,x_10)
                x_03 = self.up_12(x_12,x_00)
                out_03 = self.out_03(x_03)
                if self.deep_sup:
                    out_30 = self.out_30(x_30)
                    out_21 = self.out_21(x_21)
                    out_12 = self.out_12(x_12)
                    return [out_30, out_21, out_12, out_03], self.alpha_3
                else:
                    return out_03

            if '4' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_31 = self.up_40(x_40,x_30)
                x_22 = self.up_31(x_31,x_20)
                x_13 = self.up_22(x_22,x_10)
                x_04 = self.up_13(x_13,x_00)
                out_04 = self.out_04(x_04)
                if self.deep_sup:
                    out_40 = self.out_40(x_40)
                    out_31 = self.out_31(x_31)
                    out_22 = self.out_22(x_22)
                    out_13 = self.out_13(x_13)
                    return [out_40, out_31, out_22, out_13, out_04], self.alpha_4
                else:
                    return out_04

            if '5' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_50 = self.X_50(x_40)
                x_41 = self.up_50(x_50,x_40)
                x_32 = self.up_41(x_41,x_30)
                x_23 = self.up_32(x_32,x_20)
                x_14 = self.up_23(x_23,x_10)
                x_05 = self.up_14(x_14,x_00)
                out_05 = self.out_05(x_05)
                if self.deep_sup:
                    out_50 = self.out_50(x_50)
                    out_41 = self.out_41(x_41)
                    out_32 = self.out_32(x_32)
                    out_23 = self.out_23(x_23)
                    out_14 = self.out_14(x_14)
                    return [out_50, out_41, out_32, out_23, out_14, out_05], self.alpha_5
                else:
                    return out_05

            if '6' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_50 = self.X_50(x_40)
                x_60 = self.X_60(x_50)
                x_51 = self.up_60(x_60,x_50)
                x_42 = self.up_51(x_51,x_40)
                x_33 = self.up_42(x_42,x_30)
                x_24 = self.up_33(x_33,x_20)
                x_15 = self.up_24(x_24,x_10)
                x_06 = self.up_15(x_15,x_00)
                out_06 = self.out_06(x_06)
                if self.deep_sup:
                    out_60 = self.out_60(x_60)
                    out_51 = self.out_51(x_51)
                    out_42 = self.out_42(x_42)
                    out_33 = self.out_33(x_33)
                    out_24 = self.out_24(x_24)
                    out_15 = self.out_15(x_15)
                    return [out_60, out_51, out_42, out_33, out_24, out_15, out_06], self.alpha_6
                else:
                    return out_06
            if '7' in self.level:
                x_20 = self.X_20(x_10)
                x_30 = self.X_30(x_20)
                x_40 = self.X_40(x_30)
                x_50 = self.X_50(x_40)
                x_60 = self.X_60(x_50)
                x_70 = self.X_70(x_60)
                x_61 = self.up_70(x_70,x_60)
                x_52 = self.up_61(x_61,x_50)
                x_43 = self.up_52(x_52,x_40)
                x_34 = self.up_43(x_43,x_30)
                x_25 = self.up_34(x_34,x_20)
                x_16 = self.up_25(x_25,x_10)
                x_07 = self.up_16(x_16,x_00)
                out_07 = self.out_07(x_07)
                if self.deep_sup:
                    out_70 = self.out_70(x_70)
                    out_61 = self.out_61(x_61)
                    out_52 = self.out_52(x_52)
                    out_43 = self.out_43(x_43)
                    out_34 = self.out_34(x_34)
                    out_25 = self.out_25(x_25)
                    out_16 = self.out_16(x_16)
                    return [out_70, out_61, out_52, out_43, out_34, out_25, out_16, out_07], self.alpha_7
                else:
                    return out_07

        elif isinstance(self.level, list):
            x_01 = self.up_10(x_10, x_00)
            out_10 = self.out_10(x_10)
            out_01 = self.out_01(x_01)

            x_20 = self.X_20(x_10)
            x_11 = self.up_20(x_20,x_10)
            x_02 = self.up_11(x_11,x_00)
            out_20  = self.out_20(x_20)
            out_11 = self.out_11(x_11)
            out_02 = self.out_02(x_02)

            x_30 = self.X_30(x_20)
            x_21 = self.up_30(x_30,x_20)
            x_12 = self.up_21(x_21,x_10)
            x_03 = self.up_12(x_12,x_00)
            out_30 = self.out_30(x_30)
            out_21 = self.out_21(x_21)
            out_12 = self.out_12(x_12)
            out_03 = self.out_03(x_03)

            x_40 = self.X_40(x_30)
            x_31 = self.up_40(x_40,x_30)
            x_22 = self.up_31(x_31,x_20)
            x_13 = self.up_22(x_22,x_10)
            x_04 = self.up_13(x_13,x_00)
            out_40 = self.out_40(x_40)
            out_31 = self.out_31(x_31)
            out_22 = self.out_22(x_22)
            out_13 = self.out_13(x_13)
            out_04 = self.out_04(x_04)

            x_50 = self.X_50(x_40)
            x_41 = self.up_50(x_50,x_40)
            x_32 = self.up_41(x_41,x_30)
            x_23 = self.up_32(x_32,x_20)
            x_14 = self.up_23(x_23,x_10)
            x_05 = self.up_14(x_14,x_00)
            out_50 = self.out_50(x_50)
            out_41 = self.out_41(x_41)
            out_32 = self.out_32(x_32)
            out_23 = self.out_23(x_23)
            out_14 = self.out_14(x_14)
            out_05 = self.out_05(x_05)

            x_60 = self.X_60(x_50)
            x_51 = self.up_60(x_60,x_50)
            x_42 = self.up_51(x_51,x_40)
            x_33 = self.up_42(x_42,x_30)
            x_24 = self.up_33(x_33,x_20)
            x_15 = self.up_24(x_24,x_10)
            x_06 = self.up_15(x_15,x_00)
            out_60 = self.out_60(x_60)
            out_51 = self.out_51(x_51)
            out_42 = self.out_42(x_42)
            out_33 = self.out_33(x_33)
            out_24 = self.out_24(x_24)
            out_15 = self.out_15(x_15)
            out_06 = self.out_06(x_06)

            x_70 = self.X_70(x_60)
            x_61 = self.up_70(x_70,x_60)
            x_52 = self.up_61(x_61,x_50)
            x_43 = self.up_52(x_52,x_40)
            x_34 = self.up_43(x_43,x_30)
            x_25 = self.up_34(x_34,x_20)
            x_16 = self.up_25(x_25,x_10)
            x_07 = self.up_16(x_16,x_00)
            out_70 = self.out_70(x_70)
            out_61 = self.out_61(x_61)
            out_52 = self.out_52(x_52)
            out_43 = self.out_43(x_43)
            out_34 = self.out_34(x_34)
            out_25 = self.out_25(x_25)
            out_16 = self.out_16(x_16)
            out_07 = self.out_07(x_07)

            pred_list = {1:[self.alpha_1, [out_10, out_01]],
                         2:[self.alpha_2, [out_20, out_11, out_02]],
                         3:[self.alpha_3, [out_30, out_21, out_12, out_03]],
                         4:[self.alpha_4, [out_40, out_31, out_22, out_13, out_04]],
                         5:[self.alpha_5, [out_50, out_41, out_32, out_23, out_14, out_05]],
                         6:[self.alpha_6, [out_60, out_51, out_42, out_33, out_24, out_15, out_06]],
                         7:[self.alpha_7, [out_70, out_61, out_52, out_43, out_34, out_25, out_16, out_07]]
                        }
            return pred_list


if __name__ == '__main__':
    def count_your_model(model, x, y):
        # your rule here
        pass

    ADS_1    = AdaBoost_UNet(3, 5, "1", skip_option='scse', filters=64, deep_sup=True, head=None).cuda()
    ADS_2    = AdaBoost_UNet(3, 5, "2", skip_option='scse', filters=64, deep_sup=True, head=None).cuda()
    ADS_3    = AdaBoost_UNet(3, 5, "3", skip_option='scse', filters=64, deep_sup=True, head=None).cuda()
    ADS_4    = AdaBoost_UNet(3, 5, "4", skip_option='scse', filters=64, deep_sup=True, head=None).cuda()
    ADS_1234 = AdaBoost_UNet(3, 5, [1,2,3,4], skip_option='scse', filters=64, deep_sup=True, head=None).cuda()

    NETS = [ADS_1,ADS_2,ADS_3,ADS_4,ADS_1234]
    for net in NETS:
        summary(net, (3,512,512))
        inputs = torch.randn(1, 3, 512, 512).cuda()
        flops, params = profile(net, inputs=(inputs, ), custom_ops={AdaBoost_UNet: count_your_model})
        flops, params = clever_format([flops, params], "%.3f")
        print("flops", flops, "params", params)

    # ADS_4    = AdaBoost_UNet_d_BCSS(3, 5, [1,2,3,4,5,6,7], skip_option='scse', filters=9, deep_sup=True, head=None).cuda()
    # summary(ADS_4,(3,512,512))
