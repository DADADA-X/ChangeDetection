import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.efficientnet import EfficientNet


__all__ = ['VGG16Base', 'ResBase', 'EfficientBase']


class VGG16Base(nn.Module):
    def __init__(self):
        super(VGG16Base, self).__init__()
        self.encoder1 = se_conv_block(4, 64)
        self.encoder2 = se_conv_block(64, 128)
        self.encoder3 = se_conv_block(128, 256)
        self.encoder4 = se_conv_block(256, 512)
        self.encoder5 = se_conv_block(512, 512)

        self.mp = nn.MaxPool2d(kernel_size=2)

        self.decoder5 = deconv_block(512, 512)
        self.decoder4 = nn.Sequential(conv_block(512, 512), deconv_block(512, 256))
        self.decoder3 = nn.Sequential(conv_block(256, 256), deconv_block(256, 128))
        self.decoder2 = nn.Sequential(conv_block(128, 128), deconv_block(128, 64))

        self.out = nn.Sequential(conv_block(64, 64), nn.Conv2d(64, 2, 1))

    def forward(self, t1, t2):
        t1_1 = self.encoder1(t1)
        t1_2 = self.encoder2(self.mp(t1_1))
        t1_3 = self.encoder3(self.mp(t1_2))
        t1_4 = self.encoder4(self.mp(t1_3))
        t1_5 = self.encoder5(self.mp(t1_4))

        t2_1 = self.encoder1(t2)
        t2_2 = self.encoder2(self.mp(t2_1))
        t2_3 = self.encoder3(self.mp(t2_2))
        t2_4 = self.encoder4(self.mp(t2_3))
        t2_5 = self.encoder5(self.mp(t2_4))

        # out = self.decoder5(torch.cat([t1_5, t2_5], dim=1))
        # out = self.decoder4(torch.cat([out, t1_4, t2_4], dim=1))
        # out = self.decoder3(torch.cat([out, t1_3, t2_3], dim=1))
        # out = self.decoder2(torch.cat([out, t1_2, t2_2], dim=1))
        #
        # out = self.out(torch.cat([out, t1_1, t2_1], dim=1))

        out = self.decoder5(t1_5 + t2_5)
        out = self.decoder4(out + t1_4 + t2_4)
        out = self.decoder3(out + t1_3 + t2_3)
        out = self.decoder2(out + t1_2 + t2_2)

        out = self.out(out + t1_1 + t2_1)

        return out


class ResBase(nn.Module):
    def __init__(self, groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(ResBase, self).__init__()
        block = SEBasicBlock
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # deconv layers
        self.decoder4 = self._make_deconv_layer(DecoderBlock, 512 * block.expansion, 256 * block.expansion, stride=2)
        self.decoder3 = self._make_deconv_layer(DecoderBlock, 256 * block.expansion, 128 * block.expansion, stride=2)
        self.decoder2 = self._make_deconv_layer(DecoderBlock, 128 * block.expansion, 64 * block.expansion, stride=2)
        self.decoder1 = self._make_deconv_layer(DecoderBlock, 64 * block.expansion, 64)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,  # TODO intermediate channel
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, block, inplanes, planes, stride=1):
        layers = block(inplanes, planes, stride=stride)
        return layers

    def forward(self, t1, t2):
        # t1 Encoder
        t1 = self.conv1(t1)
        t1 = self.bn1(t1)
        t1 = self.relu(t1)

        t1_1 = self.maxpool(t1)
        t1_1 = self.layer1(t1_1)
        t1_2 = self.layer2(t1_1)
        t1_3 = self.layer3(t1_2)
        t1_4 = self.layer4(t1_3)

        # t2 Encoder
        t2 = self.conv1(t2)
        t2 = self.bn1(t2)
        t2 = self.relu(t2)

        t2_1 = self.maxpool(t2)
        t2_1 = self.layer1(t2_1)
        t2_2 = self.layer2(t2_1)
        t2_3 = self.layer3(t2_2)
        t2_4 = self.layer4(t2_3)

        # decoder
        d4 = self.decoder4(t1_4 + t2_4) + (t1_3 + t2_3)
        d3 = self.decoder3(d4) + (t1_2 + t2_2)
        d2 = self.decoder2(d3) + (t1_1 + t2_1)
        d1 = self.decoder1(d2)

        out = self.deconv(d1)

        return out


class EfficientBase(nn.Module):
    def __init__(self):
        super(EfficientBase, self).__init__()
        self.model_name = 'efficientnet-b0'
        self.encoder = EfficientNet.from_name('efficientnet-b0')
        self.size = [1280, 80, 40, 24, 16]

        # sum
        self.decoder5 = self._make_deconv_layer(DecoderBlock, self.size[0], self.size[1], stride=2)
        self.decoder4 = self._make_deconv_layer(DecoderBlock, self.size[1], self.size[2], stride=2)
        self.decoder3 = self._make_deconv_layer(DecoderBlock, self.size[2], self.size[3], stride=2)
        self.decoder2 = self._make_deconv_layer(DecoderBlock, self.size[3], self.size[4], stride=2)
        self.decoder1 = self._make_deconv_layer(DecoderBlock, self.size[4], self.size[4])

        self.final = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 2, kernel_size=1)
        )

    def _make_deconv_layer(self, block, inplanes, planes, stride=1):
        layers = block(inplanes, planes, stride=stride)
        return layers

    def forward(self, t1, t2):
        t1_1, t1_2, t1_3, t1_4, t1_5 = self.encoder(t1)
        t2_1, t2_2, t2_3, t2_4, t2_5 = self.encoder(t2)

        out = self.decoder5(t1_5 + t2_5) + (t1_4 + t2_4)
        out = self.decoder4(out) + (t1_3 + t2_3)
        out = self.decoder3(out) + (t1_2 + t2_2)
        out = self.decoder2(out) + (t1_1 + t2_1)
        out = self.decoder1(out)

        out = self.final(out)

        return out


#####################################
#            utilities              #
#####################################


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))


def se_conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SELayer(out_channels))


def deconv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.ReLU(inplace=True))


class SELayer(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        u = self.ca(self.avg_pool(x))
        ca_out = x * u

        return ca_out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.se = SELayer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inplanes // 4, inplanes // 4, kernel_size=3, stride=stride, padding=1,
                               output_padding=stride - 1, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 4, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.deconv(x)
        return out


if __name__ == '__main__':
    t1 = torch.randn(1, 4, 256, 256)
    t2 = torch.randn(1, 4, 256, 256)
    model = EfficientBase()
    print(model)
    print(model(t1, t2).shape)
