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


def hr_loss(pred, gt):
    """
    high resolution loss:  CrossEntropy
    """
    pass


def sr_loss(pred, gt):
    """
    super resolution loss: LABEL SUPER-RESOLUTION NETWORKS
    """
    nlcd_class_weights, nlcd_means, nlcd_vars = utils.load_nlcd_stats()
    super_res_crit = 0
    mask_size = (torch.sum(gt, axis=[1, 2, 3]) + 10).unsqueeze(-1)  # shape 16x1
    for nlcd_idx in range(nlcd_class_weights.shape[0]):
        c_mask = gt[:, 0, :, :].unsqueeze(1)  # shape 16x1x240x240
        c_mask_size = torch.sum(c_mask, axis=(2, 3)) + 0.000001  # shape 16x1

        c_interval_center = nlcd_means[nlcd_idx]  # shape 5,
        c_interval_radius = nlcd_vars[nlcd_idx]  # shape 5,

        masked_probs = (
                pred * c_mask
        )  # (16x5x240x240) * (16x1x240x240) --> shape (16x5x240x240)

        # Mean mean of predicted distribution todo
        mean = (
                torch.sum(masked_probs, axis=(2, 3)) / c_mask_size
        )  # (16x5) / (16,1) --> shape 16x5

        # Mean var of predicted distribution
        var = torch.sum(masked_probs * (1.0 - masked_probs), axis=(2, 3)) / (
                c_mask_size * c_mask_size
        )  # (16x5) / (16,1) --> shape 16x5

        c_super_res_crit = torch.square(
            ddist(mean, c_interval_center, c_interval_radius)
        )  # calculate numerator of equation 7 in ICLR paper
        c_super_res_crit = c_super_res_crit / (
                var + (c_interval_radius * c_interval_radius) + 0.000001
        )  # calculate denominator
        c_super_res_crit = c_super_res_crit + torch.log(
            var + 0.03
        )  # calculate log term
        c_super_res_crit = (
                c_super_res_crit
                * (c_mask_size / mask_size)
                * nlcd_class_weights[nlcd_idx]
        )  # weight by the fraction of NLCD pixels and the NLCD class weight

        super_res_crit = super_res_crit + c_super_res_crit  # accumulate

    super_res_crit = torch.sum(super_res_crit, axis=1)  # sum superres loss across highres classes
    return super_res_crit


def ddist(prediction, c_interval_center, c_interval_radius):
    return torch.relu(torch.abs(prediction - c_interval_center) - c_interval_radius)


if __name__ == '__main__':
    pred = torch.randn(16, 5, 240, 240)
    gt = torch.randint(0, 2, (16, 17, 240, 240))
    sr_loss(pred, gt)
