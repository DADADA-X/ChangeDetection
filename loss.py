import torch
import torch.nn as nn
from torch.nn import functional as F

import config
import utils


def balanced_ce_loss(pred, gt):
    pred_ = F.log_softmax(pred, dim=1)
    gt_ = gt.squeeze().long()
    if len(gt_.shape) < 3:
        gt_ = gt_.unsqueeze(0)

    weight = torch.Tensor(utils.NLCD_IDX_TRAIN_WEIGHT).cuda()
    nll_loss = nn.NLLLoss(weight=weight, ignore_index=0)
    return nll_loss(pred_, gt_)


def hr_loss(pred, gt):
    """
    high resolution loss: Simplified LABEL SUPER-RESOLUTION NETWORKS
    """

    pred_ = F.log_softmax(pred, dim=1)
    gt_ = to_soft_label(gt)

    loss = -torch.einsum('bchw, bchw->bhw', [pred_, gt_])

    mask = gt != 0
    mask_size = mask.sum()
    loss = (loss * mask).sum()/mask_size

    return loss


def to_soft_label(tensor):
    nlcd_to_reduced_lc_accumulator = torch.Tensor(config.NLCD_IDX_TO_REDUCED_LC_ACCUMULATOR).cuda()
    soft_label = nlcd_to_reduced_lc_accumulator[tensor].permute(0, 3, 1, 2)
    return soft_label


def to_one_hot_var(tensor, num_class):
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    n, c, h, w = tensor.shape
    one_hot = tensor.new(n, num_class, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.to(torch.int64), 1)
    return one_hot


def ddist(prediction, c_interval_center, c_interval_radius):
    return torch.relu(torch.abs(prediction - c_interval_center) - c_interval_radius)


if __name__ == '__main__':
    pred = torch.randn(16, 5, 240, 240)
    gt = torch.randint(0, 17, (16, 240, 240))
    print(hr_loss(pred, gt))
