# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""
import os
import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox, LoadImages
from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_jupyter, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, smart_inference_mode
from timm.models.layers import DropPath
from blocks import cbam, resnext
from inspect import isfunction


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class CSPResNextBlock(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, g=32, k=3, s=1, p=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(resnext.BottleneckResNext(c_, c_, k, s, p, shortcut, g, e) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class ConvNxt(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = LayerNorm(c2)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottlenext(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvNxt(c1, c_, 1, 1, 0)
        self.cv2 = ConvNxt(c_, c2, 3, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3Next1(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n, dd = False, dw_mid=False, k=3, p=1, shortcut=True, drop_path=0., layer_scale_init_value_csp=0, layer_scale_init_value_in=0, e=0.5): #paper LS = 1e-6
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c2 * e)  # hidden channels
        self.norm = LayerNorm(c1)
        #self.norm = LayerNorm(c1, eps=1e-6,data_format="channels_first" )
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0, bias=False)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0, bias=False)
        if not dd:
            self.m = nn.Sequential(*(Bottlenext(c_, c_, shortcut, g=1, e=1.0) for _ in range(n)))
        else:
            self.m = DoubleDown(c_, c_, n, shortcut, drop_path, layer_scale_init_value_in)
        self.gamma = nn.Parameter(layer_scale_init_value_csp * torch.ones(c_),
                                  requires_grad=True) if layer_scale_init_value_csp > 0 else None
    def forward(self, x):
        x = self.norm(x)
        x_ = self.m(self.cv1(x))    
        if self.gamma is not None:
            x_ = x_.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x_ = self.gamma * x_                #questo potrebbe non avere senso
            x_ = x_.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = torch.cat((x_ ,self.cv2(x)), 1)
        return self.cv3(self.act(x))

'''
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last or channels_first (default).
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
'''

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine=True)

    def forward(self,x):
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x

class Stem(nn.Module):
    def __init__(self,c1,c2,k,s,p):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p)

    def forward(self,x):
        return self.conv(x)


class ConvNextBlock0(nn.Module):  #come paper
    def __init__(self, c1, dim, k=7, p=3, shortcut=True, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        c_ = 4 * dim
        self.act = nn.GELU()
        self.add = shortcut
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=p, groups=dim) # depthwise conv
        #self.norm = LayerNorm(dim, eps=1e-6,data_format="channels_first" )
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, c_, 1, 1, 0) # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Conv2d(c_, dim, 1, 1, 0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.add:
            x = input + self.drop_path(x)

        return x

class ConvNextBlock1(nn.Module):      # dw nel mezzo
    def __init__(self, c1, dim, k = 7, p = 3, shortcut=True, drop_path=0., layer_scale_init_value=0., e = 4):
        super().__init__()
        c_ = e * dim
        self.add = shortcut
        self.pwconv1 = nn.Conv2d(dim, c_, 1, 1, 0)
        self.dwconv = nn.Conv2d(c_, c_, kernel_size=k, padding=p, groups=c_) # depthwise conv
        self.norm = nn.BatchNorm2d(c_)
        #self.norm = LayerNorm(c_, eps=1e-6,data_format="channels_first" )
        self.norm = LayerNorm(c_)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(c_, dim, 1, 1, 0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.add:
            x = input + self.drop_path(x)

        return x

class ConvNextBlock2(nn.Module):
    def __init__(self, c1, dim, k=7, p=3, shortcut=True, drop_path=0., layer_scale_init_value=0):
        super().__init__()
        c_ = int(dim / 4)
        self.add = shortcut
        self.pwconv1 = nn.Conv2d(dim, c_, 1, 1, 0)
        self.pwconv2 = nn.Conv2d(dim, c_, 1, 1, 0)
        self.pwconv3 = nn.Conv2d(dim, c_, 1, 1, 0)
        self.pwconv4 = nn.Conv2d(dim, c_, 1, 1, 0)
        self.dwconv1 = nn.Conv2d(c_, c_, kernel_size=k, padding=p, groups=c_) # depthwise conv
        self.dwconv2 = nn.Conv2d(c_, c_, kernel_size=k, padding=p, groups=c_)
        self.dwconv3 = nn.Conv2d(c_, c_, kernel_size=k, padding=p, groups=c_)
        self.dwconv4 = nn.Conv2d(c_, c_, kernel_size=k, padding=p, groups=c_)
        #self.norm = LayerNorm(dim, eps=1e-6,data_format="channels_first" )
        self.norm = LayerNorm(dim)
        self.pwconv5 = nn.Conv2d(dim, dim, 1, 1, 0) # pointwise/1x1 convs
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((c_)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x1 = self.dwconv1(self.pwconv1(x))
        x2 = self.dwconv2(self.pwconv2(x))
        x3 = self.dwconv3(self.pwconv3(x))
        x4 = self.dwconv4(self.pwconv4(x))
        x = self.pwconv5(self.norm(torch.cat((x1, x2, x3, x4), 1)))
        x = self.act(x)
        x = self.conv(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.add:
            x = input + self.drop_path(x)

        return x

class CBAMConvNextBlock0(nn.Module):
    def __init__(self, c1, dim, k=7, p=3, shortcut=True, drop_path=0., layer_scale_init_value=0):
        super().__init__()
        c_ = 4 * dim
        self.add = shortcut
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=p, groups=dim) # depthwise conv
        #self.norm = LayerNorm(dim, eps=1e-6,data_format="channels_first" )
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, 1, 1, 0) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        #self.pwconv2 = nn.Conv2d(c_, dim, 1, 1, 0)
        self.ChannelGate = cbam.ChannelGate1(dim, 16)
        self.SpatialGate = cbam.SpatialGate()


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.ChannelGate(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.SpatialGate(x)
        x = self.act(x)
        #x = self.pwconv2(x)
        if self.add:
            x = input + x

        return x


class ConvNextBlock3(nn.Module):
    def __init__(self, c1, dim, k=7, p=3, shortcut=True, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        c_ = 4 * dim
        self.drop = nn.Dropout(0.1)
        self.add = shortcut
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=p, groups=dim) # depthwise conv
        #self.norm = LayerNorm(dim, eps=1e-6,data_format="channels_first" )
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, c_, 1, 1, 0) # pointwise/1x1 convs, implemented with linear layers
        #self.act = nn.GELU()
        self.act = nn.SiLU()  #per il SOLO cambio bottleneck
        self.pwconv2 = nn.Conv2d(c_, dim, 1, 1, 0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.drop(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.add:
            x = input + self.drop_path(x)

        return x


class CBAMConvNextBlock1(nn.Module):
    def __init__(self, c1, dim, k=7, p=3, shortcut=True, drop_path=0., layer_scale_init_value=0):
        super().__init__()
        c_ = 4 * dim
        self.add = shortcut
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=p, groups=dim) # depthwise conv
        #self.norm = LayerNorm(dim, eps=1e-6,data_format="channels_first" )
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, c_, 1, 1, 0) # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Conv2d(c_, dim, 1, 1, 0)
        self.act = nn.GELU()
        #self.pwconv2 = nn.Conv2d(c_, dim, 1, 1, 0)
        self.ChannelGate = cbam.ChannelGate(dim, 16, ['avg', 'max'])
        self.SpatialGate = cbam.SpatialGate()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.ChannelGate(x)
        x = self.SpatialGate(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.add:
            x = input + self.drop_path(x)

        return x

class CBAMConvNextBlock2(nn.Module):
    def __init__(self, c1, dim, k=7, p=3, shortcut=True, drop_path=0., layer_scale_init_value=0):
        super().__init__()
        c_ = 4 * dim
        self.add = shortcut
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=p, groups=dim) # depthwise conv
        #self.norm = LayerNorm(dim, eps=1e-6,data_format="channels_first" )
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, c_, 1, 1, 0) # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Conv2d(c_, dim, 1, 1, 0)
        self.act = nn.GELU()
        self.ChannelGate = cbam.ChannelGate(dim, 16)
        self.SpatialGate = cbam.SpatialGate()
        #self.cbam = cbam.CBAM(c_)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.ChannelGate(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.SpatialGate(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.add:
            x = input + self.drop_path(x)

        return x


class CBAMConvNextBlock3(nn.Module): # dw_middle
    def __init__(self, c1, dim, k=7, p=3, shortcut=True, drop_path=0., layer_scale_init_value=0):
        super().__init__()
        c_ = 2 * dim
        self.add = shortcut
        self.dwconv = nn.Conv2d(c_, c_, kernel_size=k, padding=p, groups=c_) # depthwise conv
        self.norm = LayerNorm(c_)
        self.pwconv1 = nn.Conv2d(dim, c_, 1, 1, 0) # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Conv2d(c_, dim, 1, 1, 0)
        self.act = nn.GELU()
        self.ChannelGate = cbam.ChannelGate1(c_, 16)
        self.SpatialGate = cbam.SpatialGate()
        #self.cbam = cbam.CBAM(c_)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.pwconv1(x)
        x = self.SpatialGate(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.ChannelGate(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.add:
            x = input + self.drop_path(x)

        return x

class ConvNextBlock00(nn.Module):  #come paper
    def __init__(self, c1, dim, k=7, p=3, shortcut=True, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        c_ = 4 * dim
        self.act = nn.GELU()
        self.add = shortcut
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=k, padding=p, groups=dim) # depthwise conv
        #self.norm = LayerNorm(dim, eps=1e-6,data_format="channels_first" )
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, c_, 1, 1, 0) # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Conv2d(c_, dim, 1, 1, 0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.CAM = cbam.ChannelGate(dim, 16)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.CAM(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.add:
            x = input + self.drop_path(x)

        return x


class ConvNextBlock(nn.Module):
    def __init__(self, c1, c2, n, k=7, p=3, dw_mid=False, shortcut=True, drop_path=0.1, layer_scale_init_value=1e-6):
        super().__init__()
        if not dw_mid:
            self.m = nn.Sequential(*(ConvNextBlock0(c2, c2, k, p, shortcut, drop_path, layer_scale_init_value) for _ in range(n))) #convnext0 = base,convnext2 = 4dw,CBAM = CBAM xD
        else:
            self.m = nn.Sequential(*(CBAMConvNextBlock3(c2, c2, k, p, shortcut, drop_path, layer_scale_init_value) for _ in range(n)))

    def forward(self,x):
        return self.m(x)


class YoloNextBlock0(nn.Module):
    def __init__(self, c1, dim, k=7, p=3, shortcut=True):
        super().__init__()
        c_ = 4 * dim
        self.add = shortcut
        self.dwconv = Conv(dim, dim, k=k, p=p, g=dim)
        self.pwconv1 = Conv(dim, c_, k=1, s=1, p=0) # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = Conv(dim, c_, k=1, s=1, p=0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.pwconv2(x)
        if self.add:
            x = input + self.drop_path(x)

        return x


class YoloNextBlock(nn.Module):

    def __init__(self, c1, c2, n, k, p, shortcut=True):
        super().__init__()
        self.m = nn.Sequential(*(YoloNextBlock0(c1, c2, k, p, shortcut) for _ in range(n))) #convnext0 = base,convnext2 = 4dw,CBAM = CBAM xD

    def forward(self,x):
        return self.m(x)


class C3Next(nn.Module): # cambia solo il bottleneck che è di ConvNext coi CBAM

    def __init__(self, c1, c2, n=1, shortcut=True, k=7, p=3, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = YoloNextBlock(c_, c_, n, k, p, shortcut)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


#class DownSample(nn.Module):
#    def __init__(self, c1, c2, k, s, p):
#        super().__init__()
#        self.layer = nn.Sequential(
#            LayerNorm(c1, eps=1e-6, data_format="channels_first"),
#            nn.Conv2d(c1, c2, k, s, p),
#        )
#    def forward(self,x):
#        x = self.layer(x)
#        return x


class DownSample(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.layer = nn.Conv2d(c1, c2, k, s, p)
    def forward(self,x):
        x = self.layer(x)
        return x


class SAMdownYOLO(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.layer = nn.Conv2d(c1, c2, k, s, p, bias = False)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.SA = cbam.SpatialGate1(c2)
    def forward(self,x):
        x = self.act(self.SA(self.norm(self.layer(x))))
        return x


class SpatialAttDownSample(nn.Module):   # ho aggiunto la GELU
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.layer = nn.Conv2d(c1, c2, k, s, p)
        self.SA = cbam.SpatialGate1(c2)
        #self.act = nn.GELU()
    def forward(self,x):
        x = self.layer(x)
        x = self.SA(x)
        #x = self.act(x)
        return x


class SAMDropDown(nn.Module):   # ho aggiunto la GELU
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.layer = nn.Conv2d(c1, c2, k, s, p)
        self.SA = cbam.SpatialGate1(c2)
        self.drop = nn.Dropout(0.05)
        # self.act = nn.GELU()

    def forward(self, x):
        x = self.layer(x)
        x = self.SA(x)
        x = self.drop(x)
        # x = self.act(x)
        return x


class ChannelAttDownSample(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.layer = nn.Conv2d(c1, c2, k, s, p)
        self.SA = cbam.ChannelGate1(c2, 16)
    def forward(self,x):
        x = self.layer(x)
        x = self.SA(x)
        return x

'''
class ConvNextBlockCSP(nn.Module):
    def __init__(self, c1, c2, n, shortcut=True, drop_path=0., layer_scale_init_value=1e-6, e=0.5):
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0)
        self.m = ConvNextBlock(c_, c_, n, shortcut, drop_path, layer_scale_init_value)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((c_)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x_ = self.m(self.cv1(x))
        if self.gamma is not None:
            x_ = self.gamma * x_         #questo potrebbe non avere senso
        x = torch.cat((x_ , self.cv2(x)), 1)
        x = self.act(x)
        return self.cv3(x)
'''

class DoubleDown1(nn.Module): #cat

    def __init__(self, c1, dim, shortcut=True, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        c_=dim*2
        self.add = shortcut
        self.dwconv1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, groups=c_) # depthwise conv
        self.dwconv2 = nn.Conv2d(c_, c_, kernel_size=7, padding=3, groups=c_)
        self.norm = LayerNorm(2*c_)
        self.pwconv1 = nn.Conv2d(dim, c_,1,1,0) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(c_*2, dim,1,1,0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.pwconv1(x)
        x = self.act(x)
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x = torch.cat((x1 , x2), 1)
        x = self.norm(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.add:
            x = input + self.drop_path(x)

        return x

class DoubleDown2(nn.Module): #add

    def __init__(self, c1, dim, shortcut=True, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        c_=dim*2
        self.add = shortcut
        self.dwconv1 = nn.Conv2d(c_, c_, kernel_size=3, padding=1, groups=c_) # depthwise conv
        self.dwconv2 = nn.Conv2d(c_, c_, kernel_size=7, padding=3, groups=c_)
        self.norm = LayerNorm(c_)
        self.pwconv1 = nn.Conv2d(dim, c_,1,1,0) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(c_, dim,1,1,0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.pwconv1(x)
        x = self.act(x)
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x = x1 + x2
        x = self.norm(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        if self.add:
            x = input + self.drop_path(x)

        return x

class DoubleDown(nn.Module):
    def __init__(self, c1, c2, n, shortcut=True, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.m = nn.Sequential(*(DoubleDown1(c2, c2, shortcut, drop_path, layer_scale_init_value) for _ in range(n)))

    def forward(self,x):
        return self.m(x)

class ConvNextBlockCSP(nn.Module):
    def __init__(self, c1, c2, n, k=7, p=3, dd = False, dw_mid=False, shortcut=True, drop_path=0., layer_scale_init_value_csp=0, layer_scale_init_value_in=1e-6, e=0.5): #paper LS = 1e-6
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c2 * e)  # hidden channels
        self.norm = LayerNorm(c1)
        #self.norm = LayerNorm(c1, eps=1e-6,data_format="channels_first" )
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0)
        if not dd:
            self.m = ConvNextBlock(c_, c_, n, k, p, dw_mid, shortcut, drop_path, layer_scale_init_value_in)
        else:
            self.m = DoubleDown(c_, c_, n, shortcut, drop_path, layer_scale_init_value_in)
        self.gamma = nn.Parameter(layer_scale_init_value_csp * torch.ones(c_),
                                  requires_grad=True) if layer_scale_init_value_csp > 0 else None
    def forward(self, x):
        x = self.norm(x)
        x_ = self.m(self.cv1(x))    #c'era solo una act dopo il cat
        if self.gamma is not None:
            x_ = x_.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x_ = self.gamma * x_                #questo potrebbe non avere senso
            x_ = x_.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = torch.cat((x_, self.cv2(x)), 1)
        return self.cv3(self.act(x))


class ConvNextCSP0(nn.Module):
    def __init__(self, c1, c2, n, drop_path=0.1, k=7, p=3, shortcut=True, layer_scale_init_value_csp=0, layer_scale_init_value_in=1e-6, e=0.5): #paper LS = 1e-6
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c2 * e)  # hidden channels
        self.norm = LayerNorm(c1)
        self.norm1 = LayerNorm(c1)
        #self.norm2 = LayerNorm(c_)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0)

        self.m = nn.Sequential(*(ConvNextBlock0(c_, c_, k, p, shortcut, drop_path, layer_scale_init_value_in) for _ in range(n)))
        self.gamma = nn.Parameter(layer_scale_init_value_csp * torch.ones(c_),
                                  requires_grad=True) if layer_scale_init_value_csp > 0 else None
    def forward(self, x):
        x = self.norm(x)
        x_ = self.m(self.cv1(x))
        if self.gamma is not None:
            x_ = x_.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x_ = self.gamma * x_                #questo potrebbe non avere senso
            x_ = x_.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.norm1(torch.cat((x_, self.cv2(x)), 1))
        return self.cv3(self.act(x))


class ConvNextCSP1(nn.Module):
    def __init__(self, c1, c2, n, drop_path=0.1, k=7, p=3, shortcut=True, layer_scale_init_value_csp=0, layer_scale_init_value_in=1e-6, e=0.5): #paper LS = 1e-6
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c2 * e)  # hidden channels
        self.norm = LayerNorm(c1)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0)

        self.m = nn.Sequential(*(ConvNextBlock0(c_, c_, k, p, shortcut, drop_path, layer_scale_init_value_in) for _ in range(n)))
        self.gamma = nn.Parameter(layer_scale_init_value_csp * torch.ones(c_),
                                  requires_grad=True) if layer_scale_init_value_csp > 0 else None

    def forward(self, x):
        x = self.norm(x)
        x_ = self.m(self.cv1(x))
        xcsp = self.cv2(x)
        if self.gamma is not None:
            xcsp = xcsp.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            xcsp = self.gamma * xcsp                #questo potrebbe non avere senso
            xcsp = xcsp.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = torch.cat((x_, xcsp), 1)
        return self.cv3(self.act(x))


class ConvNextCSP2(nn.Module):
    def __init__(self, c1, c2, n, drop_path=0.1, k=7, p=3, shortcut=True, layer_scale_init_value_csp=0, layer_scale_init_value_in=1e-6, e=0.5): #paper LS = 1e-6
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c2 * e)  # hidden channels
        self.norm = LayerNorm(c1)
        self.norm1 = LayerNorm(c_)
        self.norm2 = LayerNorm(c_)
        self.drop = nn.Dropout(0.2)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0)

        self.m = nn.Sequential(*(ConvNextBlock0(c_, c_, k, p, shortcut, drop_path, layer_scale_init_value_in) for _ in range(n)))
        self.gamma = nn.Parameter(layer_scale_init_value_csp * torch.ones(c_),
                                  requires_grad=True) if layer_scale_init_value_csp > 0 else None
    def forward(self, x):
        x = self.norm(x)
        x_ = self.norm1(self.m(self.cv1(x)))
        if self.gamma is not None:
            x_ = x_.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x_ = self.gamma * x_                #questo potrebbe non avere senso
            x_ = x_.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = torch.cat((x_, self.norm2(self.drop(self.cv2(x)))), 1)
        return self.cv3(self.act(x))


class ConvNextCSP3(nn.Module):
    def __init__(self, c1, c2, n, drop_path_in=0.1, drop_path_csp=0.2, k=7, p=3, shortcut=True, layer_scale_init_value_in=1e-6, e=0.5): #paper LS = 1e-6
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c2 * e)  # hidden channels
        self.norm = LayerNorm(c1)
        self.norm1 = LayerNorm(c_)
        self.norm2 = LayerNorm(c_)
        self.drop = nn.Dropout(0.2)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0)
        self.drop_path = DropPath(drop_path_csp) if drop_path_csp > 0. else nn.Identity()
        self.m = nn.Sequential(*(ConvNextBlock0(c_, c_, k, p, shortcut, drop_path_in, layer_scale_init_value_in) for _ in range(n)))

    def forward(self, x):
        x = self.norm(x)
        x_ = self.norm1(self.m(self.cv1(x)))
        x = torch.cat((x_, self.norm2(self.drop_path(self.cv2(x)))), 1)
        return self.cv3(self.act(self.drop(x)))


class ConvNextCSP4(nn.Module):
    def __init__(self, c1, c2, n, drop_path_in=0.1, k=7, p=3, shortcut=True, layer_scale_init_value_in=1e-6, e=0.5): #paper LS = 1e-6
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c2 * e)  # hidden channels
        self.norm = LayerNorm(c1)
        self.norm1 = LayerNorm(c_)
        self.norm2 = LayerNorm(c_)
        self.drop = nn.Dropout(0.2)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0)
        self.m = nn.Sequential(*(ConvNextBlock0(c_, c_, k, p, shortcut, drop_path_in, layer_scale_init_value_in) for _ in range(n)))

    def forward(self, x):
        x = self.norm(x)
        x_ = self.norm1(self.m(self.cv1(x)))
        x = torch.cat((x_, self.norm2(self.cv2(x))), 1)
        return self.cv3(self.act(self.drop(x)))


class ConvNextCSP5(nn.Module):
    def __init__(self, c1, c2, n, drop_path=0.1, k=7, p=3, shortcut=True, layer_scale_init_value_csp=0, layer_scale_init_value_in=1e-6, e=0.5): #paper LS = 1e-6
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c2 * e)  # hidden channels
        self.norm = LayerNorm(c1)
        self.norm1 = LayerNorm(c_)
        self.norm2 = LayerNorm(c_)
        self.drop = nn.Dropout(0.2)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0)
        # dw mid
        self.m = nn.Sequential(*(ConvNextBlock0(c_, c_, k, p, shortcut, drop_path, layer_scale_init_value_in) for _ in range(n)))
        self.gamma = nn.Parameter(layer_scale_init_value_csp * torch.ones(c_),
                                  requires_grad=True) if layer_scale_init_value_csp > 0 else None
    def forward(self, x):
        x = self.norm(x)
        x_ = self.norm1(self.m(self.cv1(x)))
        if self.gamma is not None:
            x_ = x_.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x_ = self.gamma * x_                #questo potrebbe non avere senso
            x_ = x_.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = torch.cat((x_, self.norm2(self.drop(self.cv2(x)))), 1)
        return self.cv3(self.act(x))


class ConvNextCSP6(nn.Module):
    def __init__(self, c1, c2, n, drop_path=0.1, k=7, p=3, shortcut=True, layer_scale_init_value_csp=0, layer_scale_init_value_in=1e-6, e=0.5): #paper LS = 1e-6
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c1 * e)  # hidden channels
        self.norm = LayerNorm(c1)
        self.SAM = cbam.SpatialGate1(c_)
        self.CAM = cbam.ChannelGate1(c_,16)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1, 0)
        # dw mid
        self.dwconv = nn.Conv2d(c_, c_, kernel_size=k, padding=p, groups=c_)
        self.m = nn.Sequential(*(ConvNextBlock1(c_, c_, k, p, shortcut, drop_path, layer_scale_init_value_in, e = 2) for _ in range(n)))
        self.gamma = nn.Parameter(layer_scale_init_value_csp * torch.ones(c_),
                                  requires_grad=True) if layer_scale_init_value_csp > 0 else None
    def forward(self, x):

        x = self.SAM(self.cv1(x))
        x_ = self.CAM(self.dwconv(x))
        x = self.m(x)

        if self.gamma is not None:
            x_ = x_.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x_ = self.gamma * x_                #questo potrebbe non avere senso
            x_ = x_.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.norm(torch.cat((x, x_), 1))
        return self.cv3(self.act(x))

class CBAMdown(nn.Module):
    def __init__(self, c1, c2, k, s, p=0):
        super().__init__()
        self.SAM = cbam.SpatialGate1(c2)
        self.CAM = cbam.ChannelGate1(c2,16)
        self.down = nn.Conv2d(c1, c2, k, s, p)
        self.dw = nn.Conv2d(c2, c2, kernel_size=7, padding=3, groups=c2)

    def forward(self, x):
        return self.CAM(self.dw(self.SAM(self.down(x))))



class CSPBlock(nn.Module):
    def __init__(self, c1, c2, n, drop_path=0.1, k=7, p=3, shortcut=True, layer_scale_init_value_in=1e-6, e=0.5): #paper LS = 1e-6
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c1 * e)  # hidden channels
        self.norm = LayerNorm(c_)
        self.cv0 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.m = nn.Sequential(*(ConvNextBlock0(c_, c_, k, p, shortcut, drop_path, layer_scale_init_value_in) for _ in range(n)))

    def forward(self, x):
        input = x
        x_ = self.m(self.cv0(x))
        x = torch.cat((x_, self.cv1(input)), 1)
        return self.act(x)


class CSPFPN(nn.Module):
    def __init__(self, c1, c2, n, drop_path=0.1, k=7, p=3, shortcut=True, layer_scale_init_value_in=1e-6, e=0.5): #paper LS = 1e-6
        super().__init__()
        self.act = nn.GELU()
        c_ = int(c1 * e)  # hidden channels
        self.norm = LayerNorm(c_)
        self.cv0 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.m = nn.Sequential(*(ConvNextBlock0(c_, c_, k, p, shortcut, drop_path, layer_scale_init_value_in) for _ in range(n)))
        self.cv2 = nn.Conv2d(c1, c2, 1, 1, 0)

    def forward(self, x):
        x_ = self.m(self.cv0(x))
        x = torch.cat((x_, self.cv1(x)), 1)
        return self.cv2(self.act(x))




class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class SPPFConvNext(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.act = nn.GELU()
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.act(self.cv1(x))    #questa act non c'era
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DetectMultiBackend_meanfeature(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:  # load metadata dict
                d = json.loads(extra_files['config.txt'],
                               object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                      for k, v in d.items()})
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements('opencv-python>=4.5.4')
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements('openvino')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout('NCHW'))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            executable_network = ie.compile_model(network, device_name='CPU')  # device_name="MYRIAD" for Intel NCS2
            stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
            import tensorflow as tf
            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=''), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, 'rb') as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs='x:0', outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                delegate = {
                    'Linux': 'libedgetpu.so.1',
                    'Darwin': 'libedgetpu.1.dylib',
                    'Windows': 'edgetpu.dll'}[platform.system()]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, 'r') as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode('utf-8'))
                    stride, names = int(meta['stride']), meta['names']
        elif tfjs:  # TF.js
            raise NotImplementedError('ERROR: YOLOv5 TF.js inference is not supported')
        elif paddle:  # PaddlePaddle
            LOGGER.info(f'Loading {w} for PaddlePaddle inference...')
            check_requirements('paddlepaddle-gpu' if cuda else 'paddlepaddle')
            import paddle.inference as pdi
            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix('.pdiparams')
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f'Using {w} as Triton Inference Server...')
            check_requirements('tritonclient[all]')
            from utils.triton import TritonRemoteModel
            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith('tensorflow')
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, meanfeature=False, extract= False, target=0):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize, meanfeature=meanfeature) if augment or visualize or meanfeature else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.executable_network([im]).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    if int8:
                        scale, zero_point = output['quantization']
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if not extract:
            if isinstance(y, (list, tuple)):
                return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
            else:
                return self.from_numpy(y)
        else:

            if isinstance(y, (list, tuple)):
                if len(y)==1:
                    print('1')
                    return self.from_numpy(y[0])
                else:
                    return ([self.from_numpy(x) for x in y])
            else:
                return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ['http', 'grpc']), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True, last_layers=[26,30,34]):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            if not hasattr(self.model, 'heat_maps'):
                self.heat_maps=[]
                self.last_layers = last_layers
            else:
                self.heat_maps=None
                self.last_layers = None


        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:  # load metadata dict
                d = json.loads(extra_files['config.txt'],
                               object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                      for k, v in d.items()})
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements('opencv-python>=4.5.4')
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements('openvino')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout('NCHW'))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            executable_network = ie.compile_model(network, device_name='CPU')  # device_name="MYRIAD" for Intel NCS2
            stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
            import tensorflow as tf
            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=''), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, 'rb') as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs='x:0', outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                delegate = {
                    'Linux': 'libedgetpu.so.1',
                    'Darwin': 'libedgetpu.1.dylib',
                    'Windows': 'edgetpu.dll'}[platform.system()]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, 'r') as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode('utf-8'))
                    stride, names = int(meta['stride']), meta['names']
        elif tfjs:  # TF.js
            raise NotImplementedError('ERROR: YOLOv5 TF.js inference is not supported')
        elif paddle:  # PaddlePaddle
            LOGGER.info(f'Loading {w} for PaddlePaddle inference...')
            check_requirements('paddlepaddle-gpu' if cuda else 'paddlepaddle')
            import paddle.inference as pdi
            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix('.pdiparams')
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f'Using {w} as Triton Inference Server...')
            check_requirements('tritonclient[all]')
            from utils.triton import TritonRemoteModel
            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith('tensorflow')
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, save_features=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            if augment or visualize:
                if save_features:
                    y = self.model(im, augment=augment, visualize=visualize, save_features=save_features,
                                   ext_heat=self.heat_maps, ext_last_layers=self.last_layers)
                else:
                    y = self.model(im, augment=augment, visualize=visualize)
            else:
                if save_features:
                    y = self.model(im, save_features=save_features,  ext_heat=self.heat_maps,
                                   ext_last_layers=self.last_layers)
                else:
                    y = self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.executable_network([im]).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    if int8:
                        scale, zero_point = output['quantization']
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ['http', 'grpc']), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

        if os.path.isdir(ims):
            list_file = sorted(os.listdir(ims))
            y = []
            files = []
            dt = (Profile(), Profile(), Profile())
            shapes = []
            img_dirs = []

            for im in list_file:
                print(f'reading {im}...')
                det = self.run(os.path.join(ims, im), size=size, augment=augment, profile=profile, dt = dt)
                img_dirs.append(os.path.join(ims, im))
                y.append(det.pred[0])
                files.append(im)
                #dt = (dt[0]+det.times[0], dt[1]+det.times[1], dt[2]+det.times[2])
                shapes.append(det.s)

            return Detections(img_dirs,y,files,dt,self.names,shapes,self.model.heat_maps)
        elif isinstance(ims,str) and Path(ims).suffix[1:] in VID_FORMATS:
            if Path(ims).suffix[1:] in VID_FORMATS:
                k = 0
                y = []
                dt = (Profile(), Profile(), Profile())
                shapes = []
                imgs = []
                files = Path(ims).name
                dataset = LoadImages(ims, img_size=None, stride=self.model.stride, auto=self.model.pt, vid_stride=1)
                cap = cv2.VideoCapture(ims)
                for _, im, _, _, _ in dataset:
                    k+=1
                    print(f'looking at frame{k}')
                    det = self.run(im, size=size, augment=augment, profile=profile, dt=dt)
                    imgs.append(det.ims[0])
                    y.append(det.pred[0])
                    # dt = (dt[0]+det.times[0], dt[1]+det.times[1], dt[2]+det.times[2])
                    shapes.append(det.s)

                return Detections(imgs, y, files, dt, self.names, shapes, self.model.heat_maps, is_video=True, cap=cap )

        else:
            return self.run(ims, size=size, augment=augment, profile=profile)

    @smart_inference_mode()
    def run(self, ims, size=640, augment=False, profile=False, dt = None):
        if not dt:
            dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f'image{i}'  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
                files.append(Path(f).with_suffix('.jpg').name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:    #forse modifica qua
                #y = self.model(x, augment=augment)  # forward
                y = self.model(x, augment=augment, save_features = True)

            # Post-process
            with dt[2]:
                y = non_max_suppression(y if self.dmb else y[0],
                                        self.conf,
                                        self.iou,
                                        self.classes,
                                        self.agnostic,
                                        self.multi_label,
                                        max_det=self.max_det)  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape, self.model.heat_maps)

class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None, heat_maps = None, is_video=None, cap = None):
        super().__init__()
        d = pred[0].device  # device
        self.is_video=is_video
        self.vid_cap = cap
        if os.path.isfile(ims[0]): # o tutte o nessuna
            for im_path in ims:
                im = Image.open(requests.get(im_path, stream=True).raw if str(im_path).startswith('http') else im_path)
                im = np.asarray(exif_transpose(im))
                gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d)]  # normalizations
        else:
            gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        if heat_maps:
            self.heat_maps = heat_maps # list of heatmaps x picture (e.g. heat_maps[0] = [l,m,s])
        self.ims = ims  # list of images as numpy arrays (o lista di path to images se gli passi una directory al modello)
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1E3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path(''),
             heat_map=False, save_frames=False):
        s, crops = '', []

        if save and self.is_video:
            save_path = save_dir / self.files
            save_path=str(Path(save_path).with_suffix('.mp4')) # forcing mp4 extension
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(fps,w,h)
            video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))

        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            if os.path.isfile(im):
                im = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im)
                im = np.asarray(exif_transpose(im))
            j = 0
            crop_box = []
            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(', ')
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        j+=1
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop and not heat_map:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        elif crop and heat_map:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None

                            maps = self.heat_maps[i]
                            for indx in range(len(maps)):
                                maps[indx][maps[indx] < 0] = 0
                                maps[indx] = cv2.resize(maps[indx], (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
                            crop_box.append({
                                'name': f'box_{j}',
                                'im': save_one_box(box, im, file=file, save=save),
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'l': save_one_box(box, maps[0], file=file, save=save, gray=True),
                                'm': save_one_box(box, maps[1], file=file, save=save, gray=True),
                                's': save_one_box(box, maps[2], file=file, save=save, gray=True)
                            })

                        else:  # all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
                    if crop and heat_map:
                        crops.append({
                            'name': f'img_{i}',
                            'shape': im.shape,
                            'preds': crop_box
                        })
            else:
                if crop and heat_map:
                    crops.append([])
                s += '(no detections)'
            if save and self.is_video:
                im_v = im.transpose(1, 0, 2).astype(np.uint8)
                print(im_v.shape)
                video_writer.write(im_v)
                if i == self.n - 1:
                    LOGGER.info(f"Saved video to {colorstr('bold', save_dir)}")
            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if save and self.is_video and save_frames:
                f = f'frame{i}.jpg'
                if not os.path.isdir(save_dir / 'frames'):
                    os.mkdir(save_dir / 'frames')
                im.save(save_dir / 'frames' / f)  # save
            if show:
                if is_jupyter():
                    from IPython.display import display
                    display(im)
                else:
                    im.show(self.files[i])
            if save and not self.is_video:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip('\n')
            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops
        if save and self.is_video:
            video_writer.release()
    @TryExcept('Showing images is not supported in this environment')
    def show(self, labels=True):
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False, save_frames=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir, save_frames=save_frames)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False, heat_map=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir, heat_map=heat_map)  # crop results

    def render(self, labels=True):
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def max_per_box(self, save = False, save_path='runs/detect/exp', heat_map=True, thresh=False, medie=False):
        crops = self.crop(save=save, save_dir=save_path, heat_map=heat_map)
        if medie:
            return self.medie(crops)

        if thresh:
            return self.thres(crops)

    def print(self):
        LOGGER.info(self.__str__())

    def medie(self, crops):
        img_coords=[]
        for imgs in crops:
            coords = []
            if imgs: # le immagini senza preds escono come lista vuota
                img_name = imgs['name']
                shape = imgs['shape']
                A_img = shape[0]*shape[1]
                for box in imgs['preds']:
                    box_name = box['name']
                    A_box = box['l'].shape[0] * box['l'].shape[1]
                    x = int(box['box'][0].numpy())
                    y = int(box['box'][1].numpy())
                    ratio= A_box/A_img
                    wl=max(0,259.7403*ratio**3-97.4026*ratio**2+4.1169*ratio+1)
                    wm=-64*ratio**2+16*ratio
                    ws=max(0,-259.7403*ratio**3+97.4026*ratio**2-4.1169*ratio)
                    media= ((wl*box['l'] + wm*box['m'] + ws*box['s'])/(wm+wl+ws)) if ratio < .25 else box['s']

                    coords.append(np.unravel_index(media.argmax(), media.shape))
                    coords[-1] = (coords[-1][0] + y, coords[-1][1] + x)

            img_coords.append(coords)

        return img_coords



    def thres(self, crops):
        d = (729335 - 5525)//3
        th_min = 5525 + d
        th_max = 729335 - d

        img_coords = []

        for imgs in crops:
            coords = []
            if imgs: # le immagini senza preds escono come lista vuota
                img_name = imgs['name']
                shape = imgs['shape']
                A_img = shape[0]*shape[1]
                for box in imgs['preds']:
                    box_name = box['name']
                    A_box = box['l'].shape[0] * box['l'].shape[1]
                    x = int(box['box'][0].numpy())
                    y = int(box['box'][1].numpy())
                    if (A_box*1000)//A_img <= (th_min*1000)//A_img:
                        print(f'{img_name}_{box_name} used large')
                        coords.append(np.unravel_index(box['l'].argmax(), box['l'].shape))
                    elif (A_box*1000)//A_img >= (th_max*1000)//A_img:
                        print(f'{img_name}_{box_name} used small')
                        coords.append(np.unravel_index(box['s'].argmax(), box['s'].shape))
                    else:
                        print(f'{img_name}_{box_name} used medium')
                        coords.append(np.unravel_index(box['m'].argmax(), box['m'].shape))
                    coords[-1] = (coords[-1][0] + y, coords[-1][1] + x)

            img_coords.append(coords)

        return img_coords


    def get_statistics(self, crops):
        Areas = []
        var_x = []
        var_y = []

        for imgs in crops:
            if imgs:
                for box in imgs['preds']:
                    max_l = np.unravel_index(box['l'].argmax(), box['l'].shape)
                    max_m = np.unravel_index(box['m'].argmax(), box['m'].shape)
                    max_s = np.unravel_index(box['s'].argmax(), box['s'].shape)

                    A_box = box['l'].shape[0] * box['l'].shape[1]  # le crop hanno tutte stessa dim
                    # quest'area va rimpicciolita e usata per pesare la media dei 3 cazzi
                    # MAGARI funziona già solo uno scemo thresholding per sceglierne una alla volta
                    Areas.append(A_box)
                    var_x.append(np.sqrt(np.var([max_l[0], max_m[0], max_s[0]])))
                    var_y.append(np.sqrt(np.var([max_l[1], max_m[1], max_s[1]])))

        var_varx = (np.min(var_x), np.max(var_x), np.mean(var_x), np.sqrt(np.var(var_x)))
        var_vary = (np.min(var_y), np.max(var_y), np.mean(var_y), np.sqrt(np.var(var_y)))
        A = (np.min(Areas), np.max(Areas), np.mean(Areas), np.sqrt(np.var(Areas)))
        print('formato (min: [], max: [], mean: [], var: []')
        print('Area: {}'.format(A))
        print('var_x: {}'.format(var_varx))
        print('var_y: {}'.format(var_vary))
    def __len__(self):  # override len(results)
        return self.n

    def __str__(self):  # override print(results)
        return self._run(pprint=True)  # print results

    def __repr__(self):
        return f'YOLOv5 {self.__class__} instance\n' + self.__str__()


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 dropout_p=0.0):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


##########################################
############## FILO ######################
##########################################

class GBottleneck(nn.Module):
    # group bottleneck
    def __init__(self, c1, c2, gw, shortcut=True, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        #g = make_divisible(c2 * gw, 8)
        g = int(gw * 32)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class GC3(nn.Module):
    # CSP Group Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n, gw, shortcut=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(GBottleneck(c_, c_, shortcut, gw, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class HSigmoid(nn.Module):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """

    def forward(self, x):
        return nn.functional.relu6(x + 3.0, inplace=True) / 6.0


def get_activation_layer(activation):
    """
    Create activation layer from string/function.
    Parameters:
    ----------
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    Returns
    -------
    nn.Module
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation == "swish":
            return Swish()
        elif activation == "hswish":
            return HSwish(inplace=True)
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Module))
        return activation


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    approx_sigmoid : bool, default False
        Whether to use approximated sigmoid function.
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    """

    def __init__(self,
                 channels,
                 reduction=16,
                 approx_sigmoid=False,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(SEBlock, self).__init__()
        mid_cannels = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=mid_cannels,
            kernel_size=1,
            bias=True)
        self.activ = get_activation_layer(activation)
        self.conv2 = nn.Conv2d(
            in_channels=mid_cannels,
            out_channels=channels,
            kernel_size=1,
            bias=True)
        self.sigmoid = HSigmoid() if approx_sigmoid else nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x


class SEBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.SEBlock = SEBlock(channels=c_,
                               reduction=16,
                               approx_sigmoid=False,
                               activation=(lambda: nn.ReLU(inplace=True)))

    def forward(self, x):
        return x + self.SEBlock(self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))


class C3SE(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(SEBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        super().__init__()
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.bn(self.c(x)))


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, residual=True, cardinatlity=32):
        super().__init__()
        self.residual = residual
        self.C = cardinatlity

        res_channels = in_channels
        self.c1 = CBL(res_channels, res_channels, 1, 1, 0)
        self.c2 = CBL(res_channels, res_channels, 3, stride, 1, self.C)
        self.c3 = CBL(res_channels, res_channels, 1, 1, 0)
        self.relu = nn.LeakyReLU()
        self.SEBlock = SEBlock(channels=res_channels,
                               reduction=16,
                               approx_sigmoid=False,
                               activation=(lambda: nn.ReLU(inplace=True)))

    def forward(self, x):
        shortcut = x
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.SEBlock(x)
        if self.residual:
            x = self.relu(x + shortcut)  ## modifica la relu qui perché fai la relu della relu
        return x


class Residual_CSP(nn.Module):
    def __init__(self, in_channels, out_channels, n, stride, residual=True):
        super().__init__()
        partial = in_channels // 2
        self.C1 = CBL(in_channels, partial, 1, 1, padding=0)  # prima cbl
        self.C2 = CBL(partial, partial, 1, 1, padding=0)  # secondo cbl
        self.C3 = CBL(in_channels, partial, 1, 1, padding=0)  # cbl che va al concat
        self.C4 = CBL(in_channels, out_channels, 1, 1, padding=0)  # cbl finale
        self.m = nn.Sequential(
            *(Residual(partial, partial, stride, residual, cardinatlity=32) for _ in range(n)))  # da modificare

    def forward(self, x):
        y = x
        x = self.C1(x)
        x = self.m(x)
        x = torch.cat((x, self.C3(y)), 1)
        return self.C4(x)


class maxpool2d(nn.Module):
    def __init__(self, kernel, stride, padding):
        super().__init__()
        self.c = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x):
        return self.c(x)
######################################################
################# FINE FILO ##########################
######################################################
