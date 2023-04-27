""" Full assembly of the parts to form the complete network """
from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, filters=16, bilinear=False, flag=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.flag = flag
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
        self.outc = OutConv(filters, self.n_classes)

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
        logits = self.outc(x9)
        if self.flag:
            return [x2, x3, x4, x5, x6, x7, x8, x9]
        else:
            return logits

class CENet(nn.Module):
    def __init__(self, n_channels, n_classes, filters=16, bilinear=False, flag=False):
        super(CENet, self).__init__()
        self.n_channels = n_channels
        self.flag = flag
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
        self.up2 = CENet_Up_1(8*filters, 4*filters)
        self.up3 = CENet_Up_2(4*filters, 2*filters)
        self.up4 = CENet_Up_3(2*filters, filters)
        self.outc = OutConv(filters, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3, x4)
        x8 = self.up3(x7, x2, x3, x4)
        x9 = self.up4(x8, x1, x2, x3, x4)
        logits = self.outc(x9)
        if self.flag:
            return [x1, x2, x3, x4, x5, x6, x7, x8, x9]
        else:
            return logits