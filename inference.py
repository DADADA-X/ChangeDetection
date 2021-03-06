import sys
import os

os.environ[
    "CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"  # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import time
import datetime
import argparse
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F

import models
import config
import utils
from dataloaders.TileDatasets import TileInferenceDataset

NUM_WORKERS = 4
CHIP_SIZE = config.INF_CHIP_SIZE
PADDING = config.INF_PADDING
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING

parser = argparse.ArgumentParser(description='DFC2021 model inference script')
parser.add_argument('-i', '--input_fn', type=str, required=True,
                    help='The path to a CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('-m', '--model_fn', type=str, required=True, help='Path to the model file to use.')
parser.add_argument('-o', '--output_dir', type=str, required=True,
                    help='The path to output the model predictions as a GeoTIFF. Will fail if this file already exists.')
# parser.add_argument('--overwrite', action="store_true", help='Flag for overwriting `--output_dir` if that directory already exists.')
parser.add_argument('-g', '--gpu', type=str, default=None, help='The indices of GPUs to enable (default: all)')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size to use during inference.')
# parser.add_argument('--save_soft', action="store_true", help='Flag that enables saving the predicted per class probabilities in addition to the "hard" class predictions.')
parser.add_argument('-bb', '--backbone', default='efficientnet-b0',
                    choices=(
                        'efficientnet-b0',
                        'efficientnet-b1',
                        'efficientnet-b2',
                        'efficientnet-b3',
                        'efficientnet-b4',
                        'efficientnet-b5',
                        'efficientnet-b6',
                        'efficientnet-b7',
                        'efficientnet-b0'
                    ),
                    help='Backbone to use'
                    )

args = parser.parse_args()


def _test_augment(image):
    img = image.transpose(1, 2, 0)
    img90 = np.array(np.rot90(img))
    img1 = np.concatenate([img[None], img90[None]])
    img2 = np.array(img1)[:, ::-1]
    img3 = np.concatenate([img1, img2])
    img4 = np.array(img3)[:, :, ::-1]
    img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)

    return torch.from_numpy(img5)


def _test_augment_pred(pred):
    channel = pred.shape[1]
    pred_out = []
    for c in range(channel):
        pred_temp = pred[:, c, :, :].squeeze()
        pred_temp_1 = pred_temp[:4] + pred_temp[4:, :, ::-1]
        pred_temp_2 = pred_temp_1[:2] + pred_temp_1[2:, ::-1]
        pred_temp_3 = pred_temp_2[0] + np.rot90(pred_temp_2[1])[::-1, ::-1]
        pred_out.append(pred_temp_3.copy() / 8.)

    pred_out = np.array(pred_out)
    if len(pred_out.shape) == 3:
        pred_out = pred_out[np.newaxis, :]

    return pred_out


def _generate_matrix(gt, pred, num_class):
    mask = (gt >= 0) & (gt < num_class)
    label = num_class * gt[mask].astype('int') + pred[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix


def _Class_IOU(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    MIoU = intersection / (union + 1e-8)
    MIoU[np.where(union==0)] = 1
    return MIoU


def mIoU(pred, gt):
    confusion_matrix = _generate_matrix(gt, pred, num_class=config.HR_NCLASSES-1)
    miou = _Class_IOU(confusion_matrix)
    return miou.mean()


def main():
    # -------------------
    # Setup
    # -------------------
    assert os.path.exists(args.input_fn)
    assert os.path.exists(args.model_fn)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    device_ids = list(range(n_gpu))

    model = models.isCNN(args.backbone)

    state_dict = torch.load(args.model_fn)
    for k in list(state_dict.keys()):
        if len(device_ids) == 1 and k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]
        elif len(device_ids) > 1 and (not k.startswith('module.')):
            state_dict['module.' + k] = state_dict[k]
            del state_dict[k]
    model.load_state_dict(state_dict)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # -------------------
    # Run on each line in the input
    # -------------------
    input_dataframe = pd.read_csv(args.input_fn)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values
    groups = input_dataframe["group"].values

    inf_miou = 0

    for image_idx in range(len(image_fns)):
        tic = time.time()
        image_fn = image_fns[image_idx]
        label_fn = label_fns[image_idx]
        group = groups[image_idx]

        print("(%d/%d) Processing %s" % (image_idx + 1, len(image_fns), Path(image_fn).stem), end=" ... ")

        # -------------------
        # Load input and create dataloader
        # -------------------
        def image_transforms(img):
            if group == 0:
                img = (img - config.NAIP_2013_MEANS) / config.NAIP_2013_STDS
            elif group == 1:
                img = (img - config.NAIP_2017_MEANS) / config.NAIP_2017_STDS
            else:
                raise ValueError("group not recognized")
            img = np.rollaxis(img, 2, 0).astype(np.float32)
            img = torch.from_numpy(img)
            return img

        with rasterio.open(image_fn) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        with rasterio.open(label_fn) as f:
            label = utils.NLCD_CLASS_TO_IDX_MAP[f.read().squeeze()]
            label = config.NLCD_IDX_TO_REDUCED_LC_MAP[label]

        dataset = TileInferenceDataset(image_fn, chip_size=CHIP_SIZE, stride=CHIP_STRIDE, transform=image_transforms,
                                       verbose=False)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # -------------------
        # Run model and organize output
        # -------------------

        output = np.zeros((config.HR_NCLASSES, input_height, input_width), dtype=np.float32)
        kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
        kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        for i, (data, coords) in enumerate(dataloader):
            if config.TTA:
                data = _test_augment(data[0].numpy())
            data = data.to(device)
            with torch.no_grad():
                _, hr_out = model(data)
                t_output = F.softmax(hr_out, dim=1).cpu().numpy()
            if config.TTA:
                t_output = _test_augment_pred(t_output)

            for j in range(t_output.shape[0]):
                y, x = coords[j]

                output[:, y:y + CHIP_SIZE, x:x + CHIP_SIZE] += t_output[j] * kernel
                counts[y:y + CHIP_SIZE, x:x + CHIP_SIZE] += kernel

        output = output / counts
        output_hard = output.argmax(axis=0).astype(np.uint8)

        # offline test
        miou = mIoU(output_hard, label)
        inf_miou += miou

        # -------------------
        # Save output
        # -------------------
        output_profile = input_profile.copy()
        output_profile["driver"] = "GTiff"
        output_profile["dtype"] = "uint8"
        output_profile["count"] = 1
        output_profile["nodata"] = 0

        output_fn = image_fn.split("/")[-1]  # something like "546_naip-2013.tif"
        output_fn = output_fn.replace("naip", "predictions")
        output_fn = os.path.join(args.output_dir, output_fn)

        with rasterio.open(output_fn, "w", **output_profile) as f:
            f.write(output_hard, 1)
            f.write_colormap(1, utils.NLCD_IDX_COLORMAP)

        print("finished in %0.4f seconds" % (time.time() - tic))

    print("Mean IoU: {}".format(inf_miou / len(image_fns)))


if __name__ == "__main__":
    main()
