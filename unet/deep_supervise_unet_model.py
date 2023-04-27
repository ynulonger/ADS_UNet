import torch
from thop import *
from .unet_parts import *
from torchsummary import summary


class DS_CNN(nn.Module):
    def __init__(self, n_channels, n_classes, filters=16, bilinear=False, is_train='True'):
        super(DS_CNN, self).__init__()
        self.n_channels = n_channels
        self.is_train=is_train
        if n_classes==1:
            self.n_classes = 2
        else:
            self.n_classes = n_classes

        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, filters)
        self.down1 = Down(filters, 2*filters)
        self.down2 = Down(2*filters, 4*filters)
        self.down3 = Down(4*filters, 8*filters)
        self.down4 = Down(8*filters, 16*filters)
        
        self.up1 = Up(16*filters, 8*filters, bilinear)
        self.up2 = Up(8*filters, 4*filters, bilinear)
        self.up3 = Up(4*filters, 2*filters, bilinear)
        self.up4 = Up(2*filters, filters, bilinear)
        
        self.out1 = Up_2(2*filters, self.n_classes,2)
        self.out2 = Up_2(4*filters, self.n_classes,4)
        self.out3 = Up_2(8*filters, self.n_classes,8)
        self.out4 = Up_2(16*filters, self.n_classes,16)
        self.out5 = Up_2(8*filters, self.n_classes,8)
        self.out6 = Up_2(4*filters, self.n_classes,4)
        self.out7 = Up_2(2*filters, self.n_classes,2)
        self.out8 = OutConv(filters, self.n_classes)

        alpha = np.reshape(np.array([1/8]*8), [1,8])
        self.alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        if self.is_train:
            logits = [self.out1(x2), self.out2(x3), self.out3(x4), self.out4(x5),
                  self.out5(x6), self.out6(x7), self.out7(x8), self.out8(x9)]
            return logits
        else:
            return self.out8(x9)

class DS_UNet_deeper(nn.Module):
    def __init__(self, n_channels, n_classes, filters=16, bilinear=False):
        super(DS_UNet_deeper, self).__init__()
        self.n_channels = n_channels
        if n_classes==1:
            self.n_classes = 2
        else:
            self.n_classes = n_classes

        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, filters)
        self.down1 = Down(filters, 2*filters)
        self.down2 = Down(2*filters, 4*filters)
        self.down3 = Down(4*filters, 8*filters)
        self.down4 = Down(8*filters, 16*filters)
        self.down5 = Down(16*filters, 32*filters)
        self.down6 = Down(32*filters, 64*filters)

        self.up1 = Up(64*filters, 32*filters, bilinear)
        self.up2 = Up(32*filters, 16*filters, bilinear)
        self.up3 = Up(16*filters, 8*filters, bilinear)
        self.up4 = Up(8*filters, 4*filters, bilinear)
        self.up5 = Up(4*filters, 2*filters, bilinear)
        self.up6 = Up(2*filters, filters, bilinear)
        
        self.out1  = OutConv(2*filters, self.n_classes)
        self.out2  = OutConv(4*filters, self.n_classes)
        self.out3  = OutConv(8*filters, self.n_classes)
        self.out4  = OutConv(16*filters, self.n_classes)
        self.out5  = OutConv(32*filters, self.n_classes)
        self.out6  = OutConv(64*filters, self.n_classes)
        self.out7  = OutConv(32*filters, self.n_classes)
        self.out8  = OutConv(16*filters, self.n_classes)
        self.out9  = OutConv(8*filters, self.n_classes)
        self.out10 = OutConv(4*filters, self.n_classes)
        self.out11 = OutConv(2*filters, self.n_classes)
        self.out12 = OutConv(filters, self.n_classes)

        alpha = np.reshape(np.array([1/12]*12), [1,12])
        self.alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        x00 = self.inc(x)
        x10 = self.down1(x00)
        x20 = self.down2(x10)
        x30 = self.down3(x20)
        x40 = self.down4(x30)
        x50 = self.down5(x40)
        x60 = self.down6(x50)
        x51 = self.up1(x60, x50)
        x42 = self.up2(x51, x40)
        x33 = self.up3(x42, x30)
        x24 = self.up4(x33, x20)
        x15 = self.up5(x24, x10)
        x06 = self.up6(x15, x00)

        logits = [self.out1(x10), self.out2(x20), self.out3(x30), self.out4(x40),  self.out5(x50),  self.out6(x60),
                  self.out7(x51), self.out8(x42), self.out9(x33), self.out10(x24), self.out11(x15), self.out12(x06)]

        return logits

class DS_UNet_8(nn.Module):
    def __init__(self, n_channels, n_classes, filters=16, bilinear=False):
        super(DS_UNet_deeper, self).__init__()
        self.n_channels = n_channels
        if n_classes==1:
            self.n_classes = 2
        else:
            self.n_classes = n_classes

        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, filters)
        self.down1 = Down(filters, 2*filters)
        self.down2 = Down(2*filters, 4*filters)
        self.down3 = Down(4*filters, 8*filters)
        self.down4 = Down(8*filters, 16*filters)
        self.down5 = Down(16*filters, 32*filters)
        self.down6 = Down(32*filters, 64*filters)
        self.down7 = Down(64*filters, 128*filters)

        self.up1 = Up(128*filters, 64*filters, bilinear)
        self.up2 = Up(64*filters, 32*filters, bilinear)
        self.up3 = Up(32*filters, 16*filters, bilinear)
        self.up4 = Up(16*filters, 8*filters, bilinear)
        self.up5 = Up(8*filters, 4*filters, bilinear)
        self.up6 = Up(4*filters, 2*filters, bilinear)
        self.up7 = Up(2*filters, filters, bilinear)
        
        self.out1  = OutConv(2*filters, self.n_classes)
        self.out2  = OutConv(4*filters, self.n_classes)
        self.out3  = OutConv(8*filters, self.n_classes)
        self.out4  = OutConv(16*filters, self.n_classes)
        self.out5  = OutConv(32*filters, self.n_classes)
        self.out6  = OutConv(64*filters, self.n_classes)
        self.out7  = OutConv(128*filters, self.n_classes)
        self.out8  = OutConv(64*filters, self.n_classes)
        self.out9  = OutConv(32*filters, self.n_classes)
        self.out10 = OutConv(16*filters, self.n_classes)
        self.out11 = OutConv(8*filters, self.n_classes)
        self.out12 = OutConv(4*filters, self.n_classes)
        self.out13 = OutConv(2*filters, self.n_classes)
        self.out14 = OutConv(filters, self.n_classes)

        alpha = np.reshape(np.array([1/14]*14), [1,14])
        self.alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        x00 = self.inc(x)
        x10 = self.down1(x00)
        x20 = self.down2(x10)
        x30 = self.down3(x20)
        x40 = self.down4(x30)
        x50 = self.down5(x40)
        x60 = self.down6(x50)
        x51 = self.up1(x60, x50)
        x42 = self.up2(x51, x40)
        x33 = self.up3(x42, x30)
        x24 = self.up4(x33, x20)
        x15 = self.up5(x24, x10)
        x06 = self.up6(x15, x00)

        logits = [self.out1(x10), self.out2(x20), self.out3(x30), self.out4(x40),  self.out5(x50),  self.out6(x60),
                  self.out7(x51), self.out8(x42), self.out9(x33), self.out10(x24), self.out11(x15), self.out12(x06)]

        return logits


if __name__ == '__main__':
    def count_your_model(model, x, y):
        # your rule here
        pass

    net = DS_CNN(3,5,filters=64).cuda()
    summary(net, (3,512,512))
    inputs = torch.randn(1, 3, 512, 512).cuda()
    flops, params = profile(net, inputs=(inputs, ), custom_ops={DS_CNN: count_your_model})
    flops, params = clever_format([flops, params], "%.3f")
    print("flops", flops, "params", params)

    net = DS_UNet_1(3,5,filters=64).cuda()
    summary(net, (3,512,512))
    inputs = torch.randn(1, 3, 512, 512).cuda()
    flops, params = profile(net, inputs=(inputs, ), custom_ops={DS_UNet_1: count_your_model})
    flops, params = clever_format([flops, params], "%.3f")
    print("flops", flops, "params", params)
import torch
from thop import *
from .unet_parts import *
from torchsummary import summary


# class DS_CNN(nn.Module):
#     def __init__(self, n_channels, n_classes, filters=16, bilinear=False, is_train='True'):
#         super(DS_CNN, self).__init__()
#         self.n_channels = n_channels
#         self.is_train=is_train
#         if n_classes==1:
#             self.n_classes = 2
#         else:
#             self.n_classes = n_classes

#         self.bilinear = bilinear
#         self.inc = DoubleConv(n_channels, filters)
#         self.down1 = Down(filters, 2*filters)
#         self.down2 = Down(2*filters, 4*filters)
#         self.down3 = Down(4*filters, 8*filters)
#         self.down4 = Down(8*filters, 16*filters)
        
#         self.up1 = Up(16*filters, 8*filters, bilinear)
#         self.up2 = Up(8*filters, 4*filters, bilinear)
#         self.up3 = Up(4*filters, 2*filters, bilinear)
#         self.up4 = Up(2*filters, filters, bilinear)
        
#         self.out1 = Up_2(2*filters, self.n_classes,2)
#         self.out2 = Up_2(4*filters, self.n_classes,4)
#         self.out3 = Up_2(8*filters, self.n_classes,8)
#         self.out4 = Up_2(16*filters, self.n_classes,16)
#         self.out5 = Up_2(8*filters, self.n_classes,8)
#         self.out6 = Up_2(4*filters, self.n_classes,4)
#         self.out7 = Up_2(2*filters, self.n_classes,2)
#         self.out8 = OutConv(filters, self.n_classes)

#         alpha = np.reshape(np.array([1/8]*8), [1,8])
#         self.alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float32, requires_grad=True))

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x6 = self.up1(x5, x4)
#         x7 = self.up2(x6, x3)
#         x8 = self.up3(x7, x2)
#         x9 = self.up4(x8, x1)
#         if self.is_train:
#             logits = [self.out1(x2), self.out2(x3), self.out3(x4), self.out4(x5),
#                   self.out5(x6), self.out6(x7), self.out7(x8), self.out8(x9)]
#             return logits
#         else:
#             return self.out8(x9)


class DS_UNet_1(nn.Module):
    def __init__(self, n_channels, n_classes, filters=16, bilinear=False):
        super(DS_UNet_1, self).__init__()
        self.n_channels = n_channels
        if n_classes==1:
            self.n_classes = 2
        else:
            self.n_classes = n_classes

        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, filters)
        self.down1 = Down(filters, 2*filters)
        self.down2 = Down(2*filters, 4*filters)
        self.down3 = Down(4*filters, 8*filters)
        # factor = 2 if bilinear else 1
        self.down4 = Down(8*filters, 16*filters)
        self.up1 = Up(16*filters, 8*filters, bilinear)
        self.up2 = Up(8*filters, 4*filters, bilinear)
        self.up3 = Up(4*filters, 2*filters, bilinear)
        self.up4 = Up(2*filters, filters, bilinear)
        
        self.out0 = OutConv(filters, self.n_classes)
        self.out1 = OutConv(2*filters, self.n_classes)
        self.out2 = OutConv(4*filters, self.n_classes)
        self.out3 = OutConv(8*filters, self.n_classes)
        self.out4 = OutConv(16*filters, self.n_classes)
        self.out5 = OutConv(8*filters, self.n_classes)
        self.out6 = OutConv(4*filters, self.n_classes)
        self.out7 = OutConv(2*filters, self.n_classes)
        self.out8 = OutConv(filters, self.n_classes)

        alpha = np.reshape(np.array([1/8]*8), [1,8])
        self.alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = [self.out1(x2), self.out2(x3), self.out3(x4), self.out4(x5),
                  self.out5(x6), self.out6(x7), self.out7(x8), self.out8(x9)]
        # logits = [x2, x3, x4, x5, x6, x7, x8, x9]
        
        return logits

class DS_UNet_deeper(nn.Module):
    def __init__(self, n_channels, n_classes, filters=16, bilinear=False):
        super(DS_UNet_deeper, self).__init__()
        self.n_channels = n_channels
        if n_classes==1:
            self.n_classes = 2
        else:
            self.n_classes = n_classes

        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, filters)
        self.down1 = Down(filters, 2*filters)
        self.down2 = Down(2*filters, 4*filters)
        self.down3 = Down(4*filters, 8*filters)
        self.down4 = Down(8*filters, 16*filters)
        self.down5 = Down(16*filters, 32*filters)
        self.down6 = Down(32*filters, 64*filters)

        self.up1 = Up(64*filters, 32*filters, bilinear)
        self.up2 = Up(32*filters, 16*filters, bilinear)
        self.up3 = Up(16*filters, 8*filters, bilinear)
        self.up4 = Up(8*filters, 4*filters, bilinear)
        self.up5 = Up(4*filters, 2*filters, bilinear)
        self.up6 = Up(2*filters, filters, bilinear)
        
        self.out1  = OutConv(2*filters, self.n_classes)
        self.out2  = OutConv(4*filters, self.n_classes)
        self.out3  = OutConv(8*filters, self.n_classes)
        self.out4  = OutConv(16*filters, self.n_classes)
        self.out5  = OutConv(32*filters, self.n_classes)
        self.out6  = OutConv(64*filters, self.n_classes)
        self.out7  = OutConv(32*filters, self.n_classes)
        self.out8  = OutConv(16*filters, self.n_classes)
        self.out9  = OutConv(8*filters, self.n_classes)
        self.out10 = OutConv(4*filters, self.n_classes)
        self.out11 = OutConv(2*filters, self.n_classes)
        self.out12 = OutConv(filters, self.n_classes)

        alpha = np.reshape(np.array([1/12]*12), [1,12])
        self.alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        x00 = self.inc(x)
        x10 = self.down1(x00)
        x20 = self.down2(x10)
        x30 = self.down3(x20)
        x40 = self.down4(x30)
        x50 = self.down5(x40)
        x60 = self.down6(x50)
        x51 = self.up1(x60, x50)
        x42 = self.up2(x51, x40)
        x33 = self.up3(x42, x30)
        x24 = self.up4(x33, x20)
        x15 = self.up5(x24, x10)
        x06 = self.up6(x15, x00)

        logits = [self.out1(x10), self.out2(x20), self.out3(x30), self.out4(x40),  self.out5(x50),  self.out6(x60),
                  self.out7(x51), self.out8(x42), self.out9(x33), self.out10(x24), self.out11(x15), self.out12(x06)]

        return logits


if __name__ == '__main__':
    def count_your_model(model, x, y):
        # your rule here
        pass

    net = DS_CNN(3,5,filters=64).cuda()
    summary(net, (3,512,512))
    inputs = torch.randn(1, 3, 512, 512).cuda()
    flops, params = profile(net, inputs=(inputs, ), custom_ops={DS_CNN: count_your_model})
    flops, params = clever_format([flops, params], "%.3f")
    print("flops", flops, "params", params)

    net = DS_UNet_1(3,5,filters=64).cuda()
    summary(net, (3,512,512))
    inputs = torch.randn(1, 3, 512, 512).cuda()
    flops, params = profile(net, inputs=(inputs, ), custom_ops={DS_UNet_1: count_your_model})
    flops, params = clever_format([flops, params], "%.3f")
    print("flops", flops, "params", params)
