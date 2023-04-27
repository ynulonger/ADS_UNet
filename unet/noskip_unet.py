""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, filters=32, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        if n_classes==2:
            self.n_classes = 1
        else:
            self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, filters)
        self.down1 = Down(filters, 2*filters)
        self.down2 = Down(2*filters, 4*filters)
        self.down3 = Down(4*filters, 8*filters)
        factor = 2 if bilinear else 1
        self.down4 = Down(8*filters, 16*filters // factor)
        self.up1 = Up_noskip(16*filters, 8*filters // factor, bilinear)
        self.up2 = Up_noskip(8*filters, 4*filters // factor, bilinear)
        self.up3 = Up_noskip(4*filters, 2*filters // factor, bilinear)
        self.up4 = Up_noskip(2*filters, filters, bilinear)
        self.outc = OutConv(filters, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits
