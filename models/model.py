from .torch_wavelets import DWT_2D, IDWT_2D, DWT_2D3, IDWT_2D3
from typing import Union
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import os
import math
import numpy as np
from functools import partial
from torchvision import models
import matplotlib.pyplot as plt
from mmengine.model import BaseModule
from timm.models.layers import trunc_normal_
from mmseg.utils import OptConfigType
from torch import Tensor
import cv2
from scipy.interpolate import splprep, splev
from torchvision.transforms.functional import gaussian_blur
from models.resnet_x import resnet18
from models.resnet_x import resnet34
from models.resnet_x import resnet50
from mmcv.cnn import ConvModule



class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.stage1 = self.resnet18.conv1
        self.stage2 = self.resnet18.layer1
        self.stage3 = self.resnet18.layer2
        self.stage4 = self.resnet18.layer3
        self.stage5 = self.resnet18.layer4


    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        return x1, x2, x3, x4, x5





class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class MixingBlock(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self.dim = ch_out
        self._convmix = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            nn.PReLU(),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        mixed = torch.stack((x, y), dim=2)
        mixed = torch.reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        mixed = self._convmix(mixed)
        mixed = nn.LayerNorm(mixed.shape[1:]).to('cuda')(mixed)
        return mixed

class MixingBlock3(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self.dim = ch_out
        self._convmix = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            nn.PReLU(),
        )

    def forward(self, x: Tensor, y: Tensor, a: Tensor) -> Tensor:
        mixed = torch.stack((x, y, a), dim=2)
        mixed = torch.reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        mixed = self._convmix(mixed)
        mixed = nn.LayerNorm(mixed.shape[1:]).to('cuda')(mixed)
        return mixed

class MixingBlock4(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self.dim = ch_out
        self._convmix = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            nn.PReLU(),
        )

    def forward(self, x: Tensor, y: Tensor, a: Tensor, b: Tensor) -> Tensor:
        mixed = torch.stack((x, y, a, b), dim=2)
        mixed = torch.reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        mixed = self._convmix(mixed)
        mixed = nn.LayerNorm(mixed.shape[1:]).to('cuda')(mixed)
        return mixed



class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))

        return c_out


class NeighborFeatureAggregation5(nn.Module):
    def __init__(self, in_d=64, out_d=64):
        super(NeighborFeatureAggregation5, self).__init__()
        self.in_d = in_d
        self.mid_d = in_d
        self.out_d = out_d
        self.mix4_ = MixingBlock4(self.mid_d * 4, self.mid_d)
        self.mix3_ = MixingBlock3(self.mid_d * 3, self.mid_d)
        self.mix2_ = MixingBlock(self.mid_d * 2, self.mid_d)

        # scale 2
        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        # scale 3
        self.conv_scale3_c2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d, self.in_d, self.out_d)
        # scale 4
        self.conv_scale4_c2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d, self.in_d, self.out_d)
        # scale 5
        self.conv_scale5_c2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale5_c3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale5_c4 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale5_c5 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d, self.in_d, self.out_d)

    def forward(self, c2, c3, c4, c5):
        # scale 2
        s2 = self.conv_scale2_c2(c2)

        # scale 3
        c2_s3 = self.conv_scale3_c2(c2)
        c2_s3 = F.interpolate(c2_s3, scale_factor=(2, 2), mode='bilinear')

        c3_s3 = self.conv_scale3_c3(c3)

        s3 = self.conv_aggregation_s3(self.mix2_(c2_s3, c3_s3), c3)
        # scale 4
        c2_s4 = self.conv_scale4_c2(c2)
        c2_s4 = F.interpolate(c2_s4, scale_factor=(4, 4), mode='bilinear')

        c3_s4 = self.conv_scale4_c3(c3)
        c3_s4 = F.interpolate(c3_s4, scale_factor=(2, 2), mode='bilinear')

        c4_s4 = self.conv_scale4_c4(c4)

        s4 = self.conv_aggregation_s4(self.mix3_(c2_s4, c3_s4, c4_s4), c4)
        # scale 5
        c2_s5 = self.conv_scale5_c2(c2)
        c2_s5 = F.interpolate(c2_s5, scale_factor=(8, 8), mode='bilinear')

        c3_s5 = self.conv_scale5_c3(c3)
        c3_s5 = F.interpolate(c3_s5, scale_factor=(4, 4), mode='bilinear')

        c4_s5 = self.conv_scale5_c4(c4)
        c4_s5 = F.interpolate(c4_s5, scale_factor=(2, 2), mode='bilinear')

        c5_s5 = self.conv_scale5_c5(c5)

        s5 = self.conv_aggregation_s5(self.mix4_(c2_s5, c3_s5, c4_s5, c5_s5), c5)

        return s2, s3, s4, s5


class NeighborFeatureAggregation6(nn.Module):
    def __init__(self, in_d=64, out_d=64):
        super(NeighborFeatureAggregation6, self).__init__()
        self.in_d = in_d
        self.mid_d = out_d
        self.out_d = out_d
        self.mix4_4 = MixingBlock4(self.mid_d * 4, self.mid_d)
        self.mix3_3 = MixingBlock3(self.mid_d * 3, self.mid_d)
        self.mix2_2 = MixingBlock(self.mid_d * 2, self.mid_d)
        # scale 2
        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        # scale 3
        self.conv_scale3_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d, self.in_d, self.out_d)
        # scale 4
        self.conv_scale4_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d, self.in_d, self.out_d)
        # scale 5
        self.conv_scale5_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale5_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale5_c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale5_c5 = nn.Sequential(
            nn.Conv2d(self.in_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d, self.in_d, self.out_d)

    def forward(self, c2, c3, c4, c5):
        # scale 2
        s2 = self.conv_scale2_c2(c2)

        # scale 3
        c2_s3 = self.conv_scale3_c2(c2)

        c3_s3 = self.conv_scale3_c3(c3)

        s3 = self.conv_aggregation_s3(self.mix2_2(c2_s3, c3_s3), c3)
        # scale 4
        c2_s4 = self.conv_scale4_c2(c2)

        c3_s4 = self.conv_scale4_c3(c3)

        c4_s4 = self.conv_scale4_c4(c4)

        s4 = self.conv_aggregation_s4(self.mix3_3(c2_s4, c3_s4, c4_s4), c4)
        # scale 5
        c2_s5 = self.conv_scale5_c2(c2)

        c3_s5 = self.conv_scale5_c3(c3)

        c4_s5 = self.conv_scale5_c4(c4)

        c5_s5 = self.conv_scale5_c5(c5)

        s5 = self.conv_aggregation_s5(self.mix4_4(c2_s5, c3_s5, c4_s5, c5_s5), c5)

        return s2, s3, s4, s5




class SCL(BaseModule):
    def __init__(self,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)

    def forward(self, Sinput: Tensor, Cinput: Tensor, Binput: Tensor, att) -> Tensor:
        B, C, H, W = Cinput.shape
        Sinput = Sinput.view(B, C, -1)   #t2
        Cinput = Cinput.view(B, C, -1)   #t1
        Binput = Binput.view(B, C, -1)   #change attention map
        att = att.view(B, 1, -1)

        Cinput = Cinput * att #t1
        Sinput = Sinput * att #t2
        q = self.sigmoid(Binput)
        k = self.softmax(Cinput)
        v = self.softmax(Sinput)

        attn1 = k * q
        Sinput1 = attn1 * Sinput + Sinput
        attn2 = v * (1 - q)
        Cinput1 = attn2 * Cinput + Cinput
        out = torch.abs(Cinput1 - Sinput1)
        return out.view(B, -1, H, W)

def conv_shape(x):
    B, N, C = x.shape
    H = int(math.sqrt(N))
    W = int(math.sqrt(N))
    x = x.reshape(B, C, H, W).contiguous()
    return x

class ChangeAttention2(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super().__init__()
        self.dim = dim

        self.x1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.x2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.sigmoid = nn.Sigmoid()
        self.cl = nn.Conv2d(1, dim, 1)

        self.conv_context = nn.Sequential(
            nn.Conv2d(2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        x = self.x1(x.flatten(2).transpose(1, 2))
        y = self.x2(y.flatten(2).transpose(1, 2))
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        attn = F.cosine_similarity(x, y, dim=-1)
        attn1 = self.sigmoid(attn)
        attn2 = 1 - attn1
        attn1 = conv_shape(attn1.unsqueeze(dim=-1))
        attn2 = conv_shape(attn2.unsqueeze(dim=-1))
        context = torch.cat([attn2, attn1], dim=1)
        context = self.conv_context(context)

        return context, attn2






def createPDCFunc(PDC_type):  # 创建像素差卷积函数
    assert PDC_type in ['xc','yc'], 'unknown PDC type: %s' % str(PDC_type)
    if PDC_type == 'xc':  # CPDC,基于x轴的像素差卷积加强中心，垂直

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 or weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            buffer = weights.clone()
            buffer[:, :, [0, 2]] = buffer[:, :, [0, 2]] + buffer[:, :,[6, 8]] - 2 * buffer[:, :,[3, 5]]
            buffer[:, :, [6, 8]] = 0 - (buffer[:, :, [0, 2]] + buffer[:, :, [6, 8]] - 2 * buffer[:, :, [3, 5]])
            buffer[:, :, [1]] = 2 * (buffer[:, :, [1]] + buffer[:, :, [7]] - 2 * buffer[:, :, [4]])
            buffer[:, :, [7]] = 2 * (0 - (buffer[:, :, [1]] + buffer[:, :, [7]] - 2 * buffer[:, :, [4]]))
            buffer[:, :, [3, 4, 5]] = 0
            weights = buffer.view(shape)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    elif PDC_type == 'yc':  # CPDC,基于y轴的像素差卷积加强中心，水平

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 or weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            buffer = weights.clone()
            buffer[:, :, [0, 6]] = buffer[:, :, [0, 6]] + buffer[:, :, [2, 8]] - 2 * buffer[:, :, [1, 7]]
            buffer[:, :, [2, 8]] = 0 - (buffer[:, :, [0, 6]] + buffer[:, :, [2, 8]] - 2 * buffer[:, :, [1, 7]])
            buffer[:, :, [3]] = 2 * (buffer[:, :, [3]] + buffer[:, :, [5]] - 2 * buffer[:, :, [4]])
            buffer[:, :, [5]] = 2 * (0 - (buffer[:, :, [3]] + buffer[:, :, [5]] - 2 * buffer[:, :, [4]]))
            buffer[:, :, [1, 4, 7]] = 0
            weights = buffer.view(shape)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    else:
        print('unknown PDC type: %s' % str(PDC_type))
        return None


class Conv_2d(nn.Module):  # 把之前创建的卷积函数包装成torch卷积api相同的格式
    def __init__(self, pdc_func, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):
        """
        :param pdc_func: 卷积函数
        """
        super(Conv_2d, self).__init__()
        if in_channels % groups != 0:  # depth wise卷积要求通道要能被分组数整除
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation  # 用于控制空洞卷积，默认为1，卷积核尺寸不膨胀
        self.groups = groups
        # print(self.kernel_size)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc_func = createPDCFunc(pdc_func)

    def reset_parameters(self):
        # 凯明初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        #               输入      卷积核权重     偏置                                卷积核膨胀（用于空洞卷积）
        return self.pdc_func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)




class dfconv5(nn.Module):
    def __init__(self, in_channels, out_channels,kz=3,pd=1):
        super(dfconv5, self).__init__()
        self.conv = nn.Sequential(*[Conv_2d(pdc_func='xc', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x

class dfconv6(nn.Module):
    def __init__(self, in_channels, out_channels,kz=3,pd=1):
        super(dfconv6, self).__init__()
        self.conv = nn.Sequential(*[Conv_2d(pdc_func='yc', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x





class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class DenoiseNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 3,
                 do_res: int = False,
                 norm_type: str = 'group',
                 n_groups: int or None = None,
                 ):

        super().__init__()

        self.do_res = do_res
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x, dummy_tensor=None):

        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class LearnableThreshold1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = DenoiseNet(in_channels, in_channels)

    def forward(self, x):
        threshold_map = self.net(x)
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold_map)

class FSR(nn.Module):
    def __init__(self, in_channels):
        super(FSR, self).__init__()
        self.dwt1 = DWT_2D(wave='haar')
        self.idwt1 = IDWT_2D(wave='haar')
        self.dwt2 = DWT_2D(wave='haar')
        self.idwt2 = IDWT_2D(wave='haar')
        self.in_channels = in_channels
        self.conv2d = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.conv2d1 = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=7, stride=1, padding=3)

        self.extract21 = nn.Sequential(nn.Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1),
                                                 stride=(1, 1), bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU(inplace=True))
        self.extract22 = nn.Sequential(nn.Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1),
                                                 stride=(1, 1), bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU(inplace=True))

        # self.Softmax = nn.Softmax()
        self.dfconv1_a = dfconv6(in_channels // 4, in_channels // 4)
        self.dfconv2_a = dfconv5(in_channels // 4, in_channels // 4)

        self.dfconv1_b = dfconv6(in_channels // 4, in_channels // 4)
        self.dfconv2_b = dfconv5(in_channels // 4, in_channels // 4)

        self.dfconv1 = dfconv6(in_channels // 4, in_channels // 4)
        self.dfconv2 = dfconv5(in_channels // 4, in_channels // 4)

        self.deno1 = LearnableThreshold1(64 * 3)
        self.deno2 = LearnableThreshold1(64 * 3)
        self.deno3 = LearnableThreshold1(64 * 3)

    def forward(self, x1, x2):
        # print(x1.size(), x2.size())
        x1_i = self.dwt1(x1)
        x2_i = self.dwt2(x2)
        list11 = torch.split(x1_i, self.in_channels // 4, dim=1)
        list11 = list(list11)
        ll1 = list11[0]
        lh1 = list11[1]   #水平
        hl1 = list11[2]   #垂直
        hh1 = list11[3]

        lh1_1 = self.dfconv1_a(lh1)
        hl1_1 = self.dfconv2_a(hl1)
        h1 = torch.cat([lh1_1, hl1_1, hh1], 1)
        h1 = self.deno1(h1)

        x1 = torch.cat([ll1, h1], 1) + x1_i


        list12 = torch.split(x2_i, self.in_channels // 4, dim=1)
        list12 = list(list12)
        ll2 = list12[0]
        lh2 = list12[1]
        hl2 = list12[2]
        hh2 = list12[3]

        lh2_1 = self.dfconv1_b(lh2)
        hl2_1 = self.dfconv2_b(hl2)
        h2 = torch.cat([lh2_1, hl2_1, hh2], 1)
        h2 = self.deno2(h2)

        x2 = torch.cat([ll2, h2], 1) + x2_i
        out1 = self.sigmoid(self.conv2d(x1))
        out2 = self.sigmoid(self.conv2d(x2))
        out_1 = out1 * x1
        out_2 = out2 * x2

        n_b, n_c, h, w = x1.size()

        ll3 = torch.abs(ll1 - ll2)
        lh3 = torch.abs(lh1 - lh2)
        hl3 = torch.abs(hl1 - hl2)
        hh3 = torch.abs(hh1 - hh2)

        lh3_1 = self.dfconv1(lh3)
        hl3_1 = self.dfconv2(hl3)
        h3 = torch.cat([lh3_1, hl3_1, hh3], 1)
        h3 = self.deno3(h3)

        l = torch.cat([ll3, h3], 1) + torch.cat([ll3, lh3, hl3, hh3], 1)
        out = self.sigmoid(self.conv2d1(l))

        s1 = out * out_1
        s1 = s1 + x1
        x1 = self.extract21(s1)
        out1 = self.idwt1(x1)

        s2 = out * out_2
        s2 = s2 + x2
        x2 = self.extract22(s2)
        out2 = self.idwt2(x2)

        output = torch.abs(out1 - out2)

        return output




class MYNet_2(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d):
        super(MYNet_2, self).__init__()
        self.backbone = resnet18(pretrained=True)

        self.conv_reduce_11 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv_reduce_12 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv_reduce_13 = BasicConv2d(128, 64, 3, 1, 1)
        self.conv_reduce_14 = BasicConv2d(256, 64, 3, 1, 1)
        self.conv_reduce_21 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv_reduce_22 = BasicConv2d(64, 64, 3, 1, 1)
        self.conv_reduce_23 = BasicConv2d(128, 64, 3, 1, 1)
        self.conv_reduce_24 = BasicConv2d(256, 64, 3, 1, 1)

        self.fsr0 = FSR(64 * 4)
        self.fsr1 = FSR(64 * 4)
        self.fsr2 = FSR(64 * 4)
        self.fsr3 = FSR(64 * 4)

        self.changeatt1 = ChangeAttention2(64)
        self.changeatt2 = ChangeAttention2(64)
        self.changeatt3 = ChangeAttention2(64)
        self.changeatt4 = ChangeAttention2(64)

        self.scl1 = SCL()
        self.scl2 = SCL()
        self.scl3 = SCL()
        self.scl4 = SCL()

        self.conv_reduce_1 = BasicConv2d(64 * 2, 64, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(64 * 2, 64, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(64 * 2, 64, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(64 * 2, 64, 3, 1, 1)

        self.nfa1 = NeighborFeatureAggregation5()
        self.nfa = NeighborFeatureAggregation6()

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(128, 64, 3, 1, 1)
        self.decoder_module3 = BasicConv2d(128, 64, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(128, 64, 3, 1, 1)

        self.cls1 = nn.Conv2d(64, 1, kernel_size=1)
        self.cls2 = nn.Conv2d(64, 1, kernel_size=1)
        self.cls3 = nn.Conv2d(64, 1, kernel_size=1)
        self.cls4 = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, x, y):
        x_size = x.size()

        x_1, x_2, x_3, x_4, x_5 = self.backbone.base_forward(x)
        y_1, y_2, y_3, y_4, y_5 = self.backbone.base_forward(y)
        x_1 = self.conv_reduce_11(x_1)
        y_1 = self.conv_reduce_21(y_1)
        x_2 = self.conv_reduce_12(x_2)
        y_2 = self.conv_reduce_22(y_2)
        x_3 = self.conv_reduce_13(x_3)
        y_3 = self.conv_reduce_23(y_3)
        x_4 = self.conv_reduce_14(x_4)
        y_4 = self.conv_reduce_24(y_4)

        l1 = self.fsr0(x_1, y_1)
        l2 = self.fsr1(x_2, y_2)

        mask1, att1 = self.changeatt1(x_1, y_1)
        layer1 = self.scl1(y_1, x_1, mask1, att1)
        mask2, att2 = self.changeatt2(x_2, y_2)
        layer2 = self.scl2(y_2, x_2, mask2, att2)
        mask3, att3 = self.changeatt3(x_3, y_3)
        layer3 = self.scl3(y_3, x_3, mask3, att3)
        mask4, att4 = self.changeatt4(x_4, y_4)
        layer4 = self.scl4(y_4, x_4, mask4, att4)

        layer1 = self.conv_reduce_1(torch.cat([l1, layer1], 1))
        layer2 = self.conv_reduce_2(torch.cat([l2, layer2], 1))

        layer1, layer2, layer3, layer4 = self.nfa(layer1, layer2, layer3, layer4)
        layer4, layer3, layer2, layer1 = self.nfa1(layer4, layer3, layer2, layer1)

        feature3 = self.decoder_module4(torch.cat([self.upsample2x(layer4), layer3], 1))
        feature2 = self.decoder_module3(torch.cat([self.upsample2x(feature3), layer2], 1))
        feature1 = self.decoder_module2(torch.cat([self.upsample2x(feature2), layer1], 1))

        mask_p5 = self.cls4(layer4)
        mask_p4 = self.cls3(feature3)
        mask_p3 = self.cls2(feature2)
        mask_p2 = self.cls1(feature1)

        # change map
        mask_p2 = F.interpolate(mask_p2, x.size()[2:], mode='bilinear', align_corners=True)
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, x.size()[2:], mode='bilinear', align_corners=True)
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, x.size()[2:], mode='bilinear', align_corners=True)
        mask_p4 = torch.sigmoid(mask_p4)
        mask_p5 = F.interpolate(mask_p5, x.size()[2:], mode='bilinear', align_corners=True)
        mask_p5 = torch.sigmoid(mask_p5)

        return mask_p2, mask_p3, mask_p4, mask_p5
