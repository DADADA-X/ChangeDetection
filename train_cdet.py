import sys
import os

os.environ[
    "CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"  # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import time
import datetime
import argparse
import copy
import random
import math
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import torch

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config
import utils
from dataloaders.TileDatasets import TileChangeDetectionDataset
from dataloaders.data_agu import *
from models import *
from loss import *

NUM_WORKERS = 4
INIT_LR = 0.001

train_img_t1_dir = Path("/home/data/xyj/competition-data/ChangeDetection/Train/image-2013")
train_img_t2_dir = Path("/home/data/xyj/competition-data/ChangeDetection/Train/image-2017")
train_lb_dir = Path("/home/data/xyj/competition-data/ChangeDetection/Train/labels")
valid_img_t1_dir = Path("/home/data/xyj/competition-data/ChangeDetection/Valid/image-2013")
valid_img_t2_dir = Path("/home/data/xyj/competition-data/ChangeDetection/Valid/image-2017")
valid_lb_dir = Path("/home/data/xyj/competition-data/ChangeDetection/Valid/labels")


parser = argparse.ArgumentParser(description='DFC2021 baseline training script')
parser.add_argument('-o', '--output_dir', type=str, required=True,
                    help='The path to a directory to store model checkpoints.')
parser.add_argument('-m', '--model', type=str, default='VGG16Base')

## Training arguments
# parser.add_argument('--gpu', type=int, default=0, help='The ID of the GPU to use')
parser.add_argument('-g', '--gpu', type=str, default=None, help='The indices of GPUs to enable (default: all)')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size to use for training (default: 32)')
parser.add_argument('-e', '--num_epochs', type=int, default=50, help='Number of epochs to train for (default: 50)')
parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed to pass to numpy and torch (default: 0)')
args = parser.parse_args()


def weights_init(model, seed=7):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()


def main():
    # -------------------
    # Setup
    # -------------------
    assert train_img_t1_dir.exists()
    assert train_img_t2_dir.exists()
    assert train_lb_dir.exists()
    assert valid_img_t1_dir.exists()
    assert valid_img_t2_dir.exists()
    assert valid_lb_dir.exists()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = utils.init_logger(output_dir / 'info.log')

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    device_ids = list(range(n_gpu))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_image_fns_t1 = [str(f) for f in train_img_t1_dir.glob("*.tif")]
    train_image_fns_t2 = [str(f) for f in train_img_t2_dir.glob("*.tif")]
    train_label_fns = [str(f) for f in train_lb_dir.glob("*.tif")]
    train_image_fns_t1.sort()
    train_image_fns_t2.sort()
    train_label_fns.sort()

    train_dataset = TileChangeDetectionDataset(
        image_fns_t1=train_image_fns_t1,
        image_fns_t2=train_image_fns_t2,
        label_fns=train_label_fns,
        transform=transform2,
        data_aug_prob=0.5
    )

    valid_image_fns_t1 = [str(f) for f in valid_img_t1_dir.glob("*.tif")]
    valid_image_fns_t2 = [str(f) for f in valid_img_t2_dir.glob("*.tif")]
    valid_label_fns = [str(f) for f in valid_lb_dir.glob("*.tif")]
    valid_image_fns_t1.sort()
    valid_image_fns_t2.sort()
    valid_label_fns.sort()

    valid_dataset = TileChangeDetectionDataset(
        image_fns_t1=valid_image_fns_t1,
        image_fns_t2=valid_image_fns_t2,
        label_fns=valid_label_fns,
        transform=transform2,
        data_aug_prob=0
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = eval(args.model)()
    weights_init(model, seed=args.seed)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=INIT_LR, amsgrad=True, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()  # todo
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model)))

    # -------------------
    # Model training
    # -------------------
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    best_loss = 1e50
    num_times_lr_dropped = 0

    for epoch in range(args.num_epochs):
        lr = utils.get_lr(optimizer)

        train_loss_epoch, valid_loss_epoch = utils.fit2(
            model,
            device,
            train_dataloader,
            valid_dataloader,
            optimizer,
            criterion,
            epoch,
            logger)

        scheduler.step(valid_loss_epoch)

        if epoch % config.SAVE_PERIOD == 0 and epoch != 0:
            temp_model_fn = output_dir / 'checkpoint-epoch{}.pth'.format(epoch + 1)
            torch.save(model.state_dict(), temp_model_fn)

        if valid_loss_epoch < best_loss:
            logger.info("Saving model_best.pth...")
            temp_model_fn = output_dir / 'model_best.pth'
            torch.save(model.state_dict(), temp_model_fn)
            best_loss = valid_loss_epoch

        if utils.get_lr(optimizer) < lr:
            num_times_lr_dropped += 1
            print("")
            print("Learning rate dropped")
            print("")

        train_loss_total_epochs.append(train_loss_epoch)
        valid_loss_total_epochs.append(valid_loss_epoch)
        epoch_lr.append(lr)

        if num_times_lr_dropped == 4:
            break


if __name__ == "__main__":
    main()
