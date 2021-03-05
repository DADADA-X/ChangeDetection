import sys
import time
import json
import logging
import logging.config
import logging.handlers
import numpy as np
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import config


NLCD_IDX_COLORMAP = {
    idx: config.NLCD_CLASS_COLORMAP[c]
    for idx, c in enumerate(config.NLCD_CLASSES)
}


def get_nlcd_class_to_idx_map():
    nlcd_label_to_idx_map = []
    idx = 0
    for i in range(config.NLCD_CLASSES[-1] + 1):
        if i in config.NLCD_CLASSES:
            nlcd_label_to_idx_map.append(idx)
            idx += 1
        else:
            nlcd_label_to_idx_map.append(0)
    nlcd_label_to_idx_map = np.array(nlcd_label_to_idx_map).astype(np.int64)
    return nlcd_label_to_idx_map


NLCD_CLASS_TO_IDX_MAP = get_nlcd_class_to_idx_map()  # I do this computation on import for illustration (this could instead be a length 96 vector that is hardcoded here)


NLCD_IDX_TRAIN_PROB = torch.Tensor(
    [0.00, 14.99, 0.00, 8.10, 4.47, 2.11, 0.81, 0.35, 20.75, 1.97, 8.28, 0.59, 0.45, 9.12, 17.22, 8.36, 2.42])
NLCD_IDX_TRAIN_WEIGHT = (1 / torch.log(1.02 + NLCD_IDX_TRAIN_PROB))


class Timer():
    '''A wrapper class for printing out what is running and how long it took.
    
    Use as:
    ```
    with utils.Timer("running stuff"):
        # do stuff
    ```

    This will output:
    ```
    Starting 'running stuff'
    # any output from 'running stuff'
    Finished 'running stuff' in 12.45 seconds
    ```
    '''

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.tic = float(time.time())
        print("Starting '%s'" % (self.message))

    def __exit__(self, type, value, traceback):
        print("Finished '%s' in %0.4f seconds" % (self.message, time.time() - self.tic))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fit(model, device, train_loader, valid_loader, num_images, optimizer, lr_criterion, hr_criterion, epoch, logger, log_step=1,
        memo=''):
    logger.info('------------ Training Epoch {} ------------'.format(epoch + 1))
    model.train()
    train_epoch_loss = AverageMeter()
    train_iter_loss = AverageMeter()
    # tic = time.time()
    for batch_idx, (data, lr_targets, hr_targets) in enumerate(train_loader):

        # ------------- train -------------- #
        data = data.to(device)
        lr_targets = lr_targets.to(device)
        hr_targets = hr_targets.to(device)

        optimizer.zero_grad()
        lr_out, hr_out = model(data)
        lr_loss = lr_criterion(lr_out, lr_targets)
        hr_loss = hr_criterion(hr_out, hr_targets)
        loss = config.GAMMA * hr_loss + config.ETA * lr_loss
        loss.backward()
        optimizer.step()

        train_epoch_loss.update(loss.item())
        train_iter_loss.update(loss.item())

        if batch_idx % log_step == 0:
            current = batch_idx * train_loader.batch_size
            logger.info('Train Epoch: {}\t [{:2d}/{:2d} ({:2.0f}%)]\t Loss: {:.6f}'.format(
                epoch + 1,
                current,
                num_images,
                current / num_images * 100,
                train_iter_loss.avg))
            train_iter_loss.reset()

    # ------------ valid ------------- #
    logger.info('\nValidating...')
    valid_epoch_loss = AverageMeter()
    # valid_iter_loss = AverageMeter()
    with torch.no_grad():
        with tqdm(total=195 * valid_loader.batch_size, file=sys.stdout) as pbar:
            for batch_idx, (data, lr_targets, hr_targets) in enumerate(valid_loader):
                pbar.update(valid_loader.batch_size)
                data = data.to(device)
                lr_targets = lr_targets.to(device)
                hr_targets = hr_targets.to(device)

                lr_out, hr_out = model(data)
                lr_loss = lr_criterion(lr_out, lr_targets)
                hr_loss = hr_criterion(hr_out, hr_targets)
                loss = config.GAMMA * hr_loss + config.ETA * lr_loss

                # valid_iter_loss.update(loss.item())
                valid_epoch_loss.update(loss.item())

                # if batch_idx % log_step == 0:
                #     current = batch_idx * 64  # todo
                #     logger.info('Valid Epoch: {}\t [{:2d}/{:2d} ({:2.0f}%)]\t Loss: {:.6f}'.format(
                #         epoch + 1,
                #         current,
                #         196 * 64,
                #         current / 196,
                #         valid_iter_loss.avg))
                #     valid_iter_loss.reset()

    train_avg_loss = train_epoch_loss.avg
    valid_avg_loss = valid_epoch_loss.avg

    logger.info('\nTraining Epoch: {}\n Train_Loss: {:.2f}\n Valid_Loss: {:.2f}\n'.format(
        epoch + 1, train_avg_loss, valid_avg_loss))

    return train_avg_loss, valid_avg_loss


def fit2(model, device, train_loader, valid_loader, num_images, optimizer, criterion, epoch, logger, log_step=1,
        memo=''):
    logger.info('------------ Training Epoch {} ------------'.format(epoch + 1))
    model.train()
    train_epoch_loss = AverageMeter()
    train_iter_loss = AverageMeter()
    # tic = time.time()
    for batch_idx, (img1, img2, targets) in enumerate(train_loader):

        # ------------- train -------------- #
        img1 = img1.to(device)
        img2 = img2.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(img1, img2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_epoch_loss.update(loss.item())
        train_iter_loss.update(loss.item())

        if batch_idx % log_step == 0:
            current = batch_idx * train_loader.batch_size
            logger.info('Train Epoch: {}\t [{:2d}/{:2d} ({:2.0f}%)]\t Loss: {:.6f}'.format(
                epoch + 1,
                current,
                num_images,
                current / num_images * 100,
                train_iter_loss.avg))
            train_iter_loss.reset()

    # ------------ valid ------------- #
    logger.info('\nValidating...')
    valid_epoch_loss = AverageMeter()
    # valid_iter_loss = AverageMeter()
    with torch.no_grad():
        with tqdm(total=195 * valid_loader.batch_size, file=sys.stdout) as pbar:  # todo 195
            for batch_idx, (img1, img2, targets) in enumerate(valid_loader):
                pbar.update(valid_loader.batch_size)
                img1 = img1.to(device)
                img2 = img2.to(device)
                targets = targets.to(device)

                outputs = model(img1, img2)
                loss = criterion(outputs, targets)

                # valid_iter_loss.update(loss.item())
                valid_epoch_loss.update(loss.item())

                # if batch_idx % log_step == 0:
                #     current = batch_idx * 64  # todo
                #     logger.info('Valid Epoch: {}\t [{:2d}/{:2d} ({:2.0f}%)]\t Loss: {:.6f}'.format(
                #         epoch + 1,
                #         current,
                #         196 * 64,
                #         current / 196,
                #         valid_iter_loss.avg))
                #     valid_iter_loss.reset()

    train_avg_loss = train_epoch_loss.avg
    valid_avg_loss = valid_epoch_loss.avg

    logger.info('\nTraining Epoch: {}\n Train_Loss: {:.2f}\n Valid_Loss: {:.2f}\n'.format(
        epoch + 1, train_avg_loss, valid_avg_loss))

    return train_avg_loss, valid_avg_loss


def evaluate(model, device, data_loader, num_batches, criterion, epoch, memo=''):
    model.eval()

    losses = []
    tic = time.time()
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader), total=num_batches, file=sys.stdout):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

    avg_loss = np.mean(losses)

    print('[{}] Validation Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}'.format(
        memo, epoch, time.time() - tic, avg_loss), end=""
    )
    print("")

    return [avg_loss]


def score(model, device, data_loader, num_batches):
    model.eval()

    num_classes = model.module.segmentation_head[0].out_channels
    num_samples = len(data_loader.dataset)
    predictions = np.zeros((num_samples, num_classes), dtype=np.float32)
    idx = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            output = F.softmax(model(data))
        batch_size = data.shape[0]
        predictions[idx:idx + batch_size] = output.cpu().numpy()
        idx += batch_size
    return predictions


def init_logger(save_fn):
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.DEBUG)

    simple = logging.Formatter("%(message)s")
    datetime = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(simple)

    info_file_handler = logging.handlers.RotatingFileHandler(save_fn, maxBytes=10485760, backupCount=20,
                                                             encoding='utf-8')
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(datetime)

    logger.addHandler(console_handler)
    logger.addHandler(info_file_handler)

    return logger


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def do_nlcd_means_tuning(nlcd_means):
    nlcd_means[2:, 1] -= 0
    nlcd_means[3:7, 4] += 0.25
    nlcd_means = nlcd_means / np.maximum(0, nlcd_means).sum(axis=1, keepdims=True)
    nlcd_means[0, :] = 0
    nlcd_means[-1, :] = 0
    return nlcd_means

