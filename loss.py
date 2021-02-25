import torch
import torch.nn as nn
from torch.nn import functional as F

import utils


def balanced_ce_loss(pred, gt):
    pred_ = F.log_softmax(pred, dim=1)
    gt_ = gt.squeeze().long()
    if len(gt_.shape) < 3:
        gt_ = gt_.unsqueeze(0)

    weight = torch.Tensor(utils.NLCD_IDX_TRAIN_WEIGHT).cuda()
    nll_loss = nn.NLLLoss(weight=weight, ignore_index=0)
    return nll_loss(pred_, gt_)
