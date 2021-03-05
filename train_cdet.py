import sys
import os
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt" # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
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

import models
import config
import utils
from dataloaders.StreamingDatasets import StreamingChangeDetTrainDataset, StreamingChangeDetValidDataset
from dataloaders.data_agu import *
from loss import *


NUM_WORKERS = 4
NUM_CHIPS_PER_TILE = config.NUM_CHIPS_PER_TILE_DET
CHIP_SIZE = config.TRAIN_CHIP_SIZE_DET
INIT_LR = 0.001

parser = argparse.ArgumentParser(description='DFC2021 baseline training script')
parser.add_argument('-t1', '--train_fn_t1', type=str, required=True,  help='The path to a train CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('-t2', '--train_fn_t2', type=str, required=True,  help='The path to a train CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('-v1', '--valid_fn_t1', type=str, required=True,  help='The path to a valid CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('-v2', '--valid_fn_t2', type=str, required=True,  help='The path to a valid CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('-o', '--output_dir', type=str, required=True,  help='The path to a directory to store model checkpoints.')
# parser.add_argument('--overwrite', action="store_true",  help='Flag for overwriting `output_dir` if that directory already exists.')
# parser.add_argument('--save_most_recent', action="store_true",  help='Flag for saving the most recent version of the model during training.')
parser.add_argument('-m', '--model', default='VGG16Base',
    choices=(
        'VGG16Base', 'ResBase', 'EfficientBase'
    ),
    help='Model to use'
)

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
    assert os.path.exists(args.train_fn_t1)
    assert os.path.exists(args.train_fn_t2)
    assert os.path.exists(args.valid_fn_t1)
    assert os.path.exists(args.valid_fn_t2)

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

    train_dataframe_t1 = pd.read_csv(args.train_fn_t1)
    train_image_fns_t1 = train_dataframe_t1["image_fn"].values
    train_label_fns_t1 = train_dataframe_t1["label_fn"].values
    train_dataframe_t2 = pd.read_csv(args.train_fn_t2)
    train_image_fns_t2 = train_dataframe_t2["image_fn"].values
    train_label_fns_t2 = train_dataframe_t2["label_fn"].values

    train_dataset = StreamingChangeDetTrainDataset(
        image_fns_t1 = train_image_fns_t1,
        label_fns_t1 = train_label_fns_t1,
        image_fns_t2 = train_image_fns_t2,
        label_fns_t2 = train_label_fns_t2,
        transform=transform2
    )

    valid_dataframe_t1 = pd.read_csv(args.valid_fn_t1)
    valid_image_fns_t1 = valid_dataframe_t1["image_fn"].values
    valid_label_fns_t1 = valid_dataframe_t1["label_fn"].values
    valid_dataframe_t2 = pd.read_csv(args.valid_fn_t2)
    valid_image_fns_t2 = valid_dataframe_t2["image_fn"].values
    valid_label_fns_t2 = valid_dataframe_t2["label_fn"].values

    valid_dataset = StreamingChangeDetValidDataset(
        image_fns_t1=valid_image_fns_t1,
        label_fns_t1=valid_label_fns_t1,
        image_fns_t2=valid_image_fns_t2,
        label_fns_t2=valid_label_fns_t2,
        transform=transform2
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

    num_training_images_per_epoch = int(len(train_image_fns_t1) * NUM_CHIPS_PER_TILE)

    model = eval(args.model)
    weights_init(model, seed=args.seed)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=INIT_LR, amsgrad=True, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()  # todo
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model)))

    #-------------------
    # Model training
    #-------------------
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
            num_training_images_per_epoch,
            optimizer,
            criterion,
            epoch,
            logger)

        scheduler.step(valid_loss_epoch)

        if epoch % config.SAVE_PERIOD == 0 and epoch != 0:
            temp_model_fn = output_dir / 'checkpoint-epoch{}.pth'.format(epoch+1)
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