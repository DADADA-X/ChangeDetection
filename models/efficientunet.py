import torch
import torch.nn as nn
from collections import OrderedDict

from models.efficientnet import EfficientNet


class EfficientUnet(nn.Module):
    def __init__(self, backbone, out_channels=2, concat_input=True):
        super().__init__()

        self.model_name = backbone
        self.backbone = EfficientNet.from_name(backbone)
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.model_name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.model_name]

    def forward(self, x):
        input_ = x

        e1, e2, e3, e4, e5 = self.backbone(x)

        x = self.up_conv1(e5)
        x = torch.cat([x, e4], dim=1)  #
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, e3], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, e2], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, e1], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)

        return x


# def get_blocks_to_be_concat(model, x):
#     shapes = set()
#     blocks = OrderedDict()
#     hooks = []
#     count = 0
#
#     def register_hook(module):
#
#         def hook(module, input, output):
#             try:
#                 nonlocal count
#                 if module.name == f'blocks_{count}_output_batch_norm':
#                     count += 1
#                     shape = output.size()[-2:]
#                     if shape not in shapes:
#                         shapes.add(shape)
#                         blocks[module.name] = output
#
#                 elif module.name == 'head_swish':
#                     # when module.name == 'head_swish', it means the program has already got all necessary blocks for
#                     # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
#                     # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
#                     # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
#                     # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
#                     # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
#                     blocks.popitem()
#                     blocks[module.name] = output
#
#             except AttributeError:
#                 pass
#
#         if (
#                 not isinstance(module, nn.Sequential)
#                 and not isinstance(module, nn.ModuleList)
#                 and not (module == model)
#         ):
#             hooks.append(module.register_forward_hook(hook))
#
#     # register hook
#     model.apply(register_hook)
#
#     # make a forward pass to trigger the hooks
#     model(x)
#
#     # remove these hooks
#     for h in hooks:
#         h.remove()
#
#     return blocks


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
    model = EfficientUnet(backbone='efficientnet-b7', out_channels=17, concat_input=True)  # b0-b7
    input = torch.randn(2, 3, 224, 224)
    print(model)
    output = model(input)
    print(output.shape)