"""The U-Net.

Channels: 32 -> 64 -> 128 -> 256 -> 512 at the bottleneck, and back down.
7.76 M parameters. Learn those numbers; they get asked about.

Why a U-Net for this task in particular: a cochlea is about four voxels
across. By the bottleneck the image has been pooled from 256x256 down to
16x16, so a four-voxel structure has been reduced to a fraction of one cell
and its position is gone. The skip connections are what carry the fine detail
straight across from the encoder to the decoder at full resolution. Without
them this architecture could find a parotid and would never find a cochlea.
"""
import torch
import torch.nn as nn


def block(cin: int, cout: int) -> nn.Sequential:
    """Two 3x3 convolutions, each followed by normalisation and a ReLU.

    bias=False because BatchNorm immediately subtracts the mean, which throws
    any bias away — carrying one would be parameters that cannot matter.
    """
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    """(B, in_ch, H, W) -> (B, n_classes, H, W) of raw logits.

    Raw logits, not probabilities: cross_entropy and the Dice loss both apply
    their own softmax, and applying it twice quietly flattens the gradients.
    """

    def __init__(self, in_ch: int = 2, n_classes: int = 5, base: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1, self.enc2 = block(in_ch, c1), block(c1, c2)
        self.enc3, self.enc4 = block(c2, c3), block(c3, c4)
        self.bott = block(c4, c5)
        self.pool = nn.MaxPool2d(2)
        self.up4, self.dec4 = nn.ConvTranspose2d(c5, c4, 2, 2), block(c4 * 2, c4)
        self.up3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, 2), block(c3 * 2, c3)
        self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, 2), block(c2 * 2, c2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, 2), block(c1 * 2, c1)
        self.head = nn.Conv2d(c1, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)                                  # 256, c1
        e2 = self.enc2(self.pool(e1))                      # 128, c2
        e3 = self.enc3(self.pool(e2))                      #  64, c3
        e4 = self.enc4(self.pool(e3))                      #  32, c4
        b = self.bott(self.pool(e4))                       #  16, c5
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))    #  32, c4
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))   #  64, c3
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))   # 128, c2
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))   # 256, c1
        return self.head(d1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
