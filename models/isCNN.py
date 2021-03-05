import torch
import torch.nn as nn
from collections import OrderedDict

from models.efficientnet import EfficientNet


class isCNN(nn.Module):
    def __init__(self, backbone, lr_channels=17, hr_channels=5):
        super(isCNN, self).__init__()

        self.model_name = backbone
        self.lr_backbone = EfficientNet.from_name(backbone)
        self.hr_backbone = EfficientNet.from_name(backbone)

        self.lr_final = nn.Sequential(double_conv(self.n_channels, 64), nn.Conv2d(64, lr_channels, 1))

        self.up_conv1 = up_conv(2*self.n_channels+lr_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)
        self.up_conv_input = up_conv(64, 32)
        self.double_conv_input = double_conv(self.size[4], 32)
        self.hr_final = nn.Conv2d(self.size[5], hr_channels, kernel_size=1)


    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.model_name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 36, 32], 'efficientnet-b1': [592, 296, 152, 80, 36, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 36, 32], 'efficientnet-b3': [608, 304, 160, 88, 36, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 36, 32], 'efficientnet-b5': [640, 320, 168, 88, 36, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 36, 32], 'efficientnet-b7': [672, 336, 176, 96, 36, 32]}
        return size_dict[self.model_name]

    def forward(self, x):
        input_ = x

        _, _, _, _, lr_5 = self.lr_backbone(x)
        hr_1, hr_2, hr_3, hr_4, hr_5 = self.hr_backbone(x)

        lr_out = self.lr_final(lr_5)

        hr_out = torch.cat([hr_5, lr_5, lr_out], dim=1)
        hr_out = self.up_conv1(hr_out)
        hr_out = torch.cat([hr_out, hr_4], dim=1)
        hr_out = self.double_conv1(hr_out)

        hr_out = self.up_conv2(hr_out)
        hr_out = torch.cat([hr_out, hr_3], dim=1)
        hr_out = self.double_conv2(hr_out)

        hr_out = self.up_conv3(hr_out)
        hr_out = torch.cat([hr_out, hr_2], dim=1)
        hr_out = self.double_conv3(hr_out)

        hr_out = self.up_conv4(hr_out)
        hr_out = torch.cat([hr_out, hr_1], dim=1)
        hr_out = self.double_conv4(hr_out)

        hr_out = self.up_conv_input(hr_out)
        hr_out = torch.cat([hr_out, input_], dim=1)
        hr_out = self.double_conv_input(hr_out)

        hr_out = self.hr_final(hr_out)

        return lr_out, hr_out


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


if __name__ == '__main__':
    model = isCNN(backbone='efficientnet-b7')  # b0-b7
    input = torch.randn(2, 4, 256, 256)
    print(model)
    lr_output, hr_output = model(input)
    print(lr_output.shape, hr_output.shape)
