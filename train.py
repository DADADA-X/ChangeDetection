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
import utils
from dataloaders.StreamingDatasets import StreamingGeospatialDataset, StreamingValidationDataset
from dataloaders.data_agu import *
from loss import *

NUM_WORKERS = 4
NUM_CHIPS_PER_TILE = 100
CHIP_SIZE = 256
INIT_LR = 0.001
save_period = 10

parser = argparse.ArgumentParser(description='DFC2021 baseline training script')
parser.add_argument('--train_fn', type=str, required=True,  help='The path to a train CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('--valid_fn', type=str, required=True,  help='The path to a valid CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('--output_dir', type=str, required=True,  help='The path to a directory to store model checkpoints.')
parser.add_argument('--overwrite', action="store_true",  help='Flag for overwriting `output_dir` if that directory already exists.')
parser.add_argument('--save_most_recent', action="store_true",  help='Flag for saving the most recent version of the model during training.')
parser.add_argument('--model', default='unet',
    choices=(
        'unet',
        'fcn'
    ),
    help='Model to use'
)

## Training arguments
# parser.add_argument('--gpu', type=int, default=0, help='The ID of the GPU to use')
parser.add_argument('--gpu', type=str, default=None, help='The indices of GPUs to enable (default: all)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use for training (default: 32)')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for (default: 50)')
parser.add_argument('--seed', type=int, default=0, help='Random seed to pass to numpy and torch (default: 0)')
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
    # print("Starting DFC2021 baseline training script at %s" % (str(datetime.datetime.now())))
    #-------------------
    # Setup
    #-------------------
    assert os.path.exists(args.train_fn)
    assert os.path.exists(args.valid_fn)

    now_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    # output path
    # output_dir = Path(args.output_dir).parent / time_str / Path(args.output_dir).stem
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = utils.init_logger(output_dir / 'info.log')
    # if os.path.isfile(args.output_dir):
    #     print("A file was passed as `--output_dir`, please pass a directory!")
    #     return
    #
    # if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)):
    #     if args.overwrite:
    #         print("WARNING! The output directory, %s, already exists, we might overwrite data in it!" % (args.output_dir))
    #     else:
    #         print("The output directory, %s, already exists and isn't empty. We don't want to overwrite and existing results, exiting..." % (args.output_dir))
    #         return
    # else:
    #     print("The output directory doesn't exist or is empty.")
    #     os.makedirs(args.output_dir, exist_ok=True)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    device_ids = list(range(n_gpu))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #-------------------
    # Load input data
    #-------------------

    train_dataframe = pd.read_csv(args.train_fn)
    train_image_fns = train_dataframe["image_fn"].values
    train_label_fns = train_dataframe["label_fn"].values
    train_groups = train_dataframe["group"].values
    train_dataset = StreamingGeospatialDataset(
        imagery_fns=train_image_fns, label_fns=train_label_fns, groups=train_groups, chip_size=CHIP_SIZE,
        num_chips_per_tile=NUM_CHIPS_PER_TILE, transform=transform, nodata_check=nodata_check
    )

    valid_dataframe = pd.read_csv(args.valid_fn)
    valid_image_fns = valid_dataframe["image_fn"].values
    valid_label_fns = valid_dataframe["label_fn"].values
    valid_groups = valid_dataframe["group"].values
    valid_dataset = StreamingValidationDataset(
        imagery_fns=valid_image_fns, label_fns=valid_label_fns, groups=valid_groups, chip_size=CHIP_SIZE,
        stride=CHIP_SIZE, transform=transform, nodata_check=nodata_check
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

    num_training_images_per_epoch = int(len(train_image_fns) * NUM_CHIPS_PER_TILE)
    # print("We will be training with %d batches per epoch" % (num_training_batches_per_epoch))

    #-------------------
    # Setup training
    #-------------------
    if args.model == "unet":
        model = models.get_unet()
    elif args.model == "fcn":
        model = models.get_fcn()
    else:
        raise ValueError("Invalid model")

    weights_init(model, seed=args.seed)

    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=INIT_LR, amsgrad=True, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss() # todo
    # criterion = balanced_ce_loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    # factor=0.5, patience=3, min_lr=0.0000001
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model)))

    #-------------------
    # Model training
    #-------------------
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    best_loss = 1e50
    num_times_lr_dropped = 0
    # model_checkpoints = []
    # temp_model_fn = os.path.join(output_dir, "most_recent_model.pt")

    for epoch in range(args.num_epochs):
        lr = utils.get_lr(optimizer)

        train_loss_epoch, valid_loss_epoch = utils.fit(
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

        if epoch % save_period == 0 and epoch != 0:
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


    #-------------------
    # Save everything
    #-------------------
    # save_obj = {
    #     'args': args,
    #     'training_task_losses': training_task_losses,
    #     "checkpoints": model_checkpoints
    # }
    #
    # save_obj_fn = "results.pt"
    # with open(os.path.join(output_dir, save_obj_fn), 'wb') as f:
    #     torch.save(save_obj, f)

if __name__ == "__main__":
    main()