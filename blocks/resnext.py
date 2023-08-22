import torch
import math
import torch.nn as nn
import torch.nn.functional


class ConvResNext(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BottleneckResNext(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, k=3, s=1, p=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvResNext(c1, c_, 1, 1, 0)
        self.cv2 = ConvResNext(c_, c_, k, s, p, g=g)
        self.cv3 = ConvResNext(c_, c2, 1, 1, 0)
        self.add = shortcut

    def forward(self, x):
        input = x
        x = self.cv3(self.cv2(self.cv1(x)))
        if self.add:
            x = input + x
        return x


