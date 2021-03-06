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
from dataloaders.TileDatasets import TileInferenceCDetDataset
from models import *

NUM_WORKERS = 4
CHIP_SIZE = config.INF_CHIP_SIZE
PADDING = config.INF_PADDING
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING

parser = argparse.ArgumentParser(description='DFC2021 model inference script')
parser.add_argument('-i', '--input_fn', type=str, required=True)
parser.add_argument('-m', '--model_fn', type=str, required=True, help='Path to the model file to use.')
parser.add_argument('-o', '--output_dir', type=str, required=True,
                    help='The path to output the model predictions as a GeoTIFF. Will fail if this file already exists.')
parser.add_argument('-g', '--gpu', type=str, default=None, help='The indices of GPUs to enable (default: all)')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size to use during inference.')
parser.add_argument('-md', '--model', type=str, default='VGG16Base')

args = parser.parse_args()


def _test_augment(image1, image2):
    img = image1.transpose(1, 2, 0)
    img90 = np.array(np.rot90(img))
    img1 = np.concatenate([img[None], img90[None]])
    img2 = np.array(img1)[:, ::-1]
    img3 = np.concatenate([img1, img2])
    img4 = np.array(img3)[:, :, ::-1]
    img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
    image1 = torch.from_numpy(img5)

    img = image2.transpose(1, 2, 0)
    img90 = np.array(np.rot90(img))
    img1 = np.concatenate([img[None], img90[None]])
    img2 = np.array(img1)[:, ::-1]
    img3 = np.concatenate([img1, img2])
    img4 = np.array(img3)[:, :, ::-1]
    img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
    image2 = torch.from_numpy(img5)

    return image1, image2


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


def IoU(pred, gt):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    if union.sum() == 0:
        iou_score = 1
    else:
        iou_score = intersection.sum() / union.sum()
    return iou_score


def main():
    # print("Starting DFC2021 model inference script at %s" % (str(datetime.datetime.now())))

    # -------------------
    # Setup
    # -------------------
    assert Path(args.input_fn).exists()
    assert Path(args.model_fn).exists()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    device_ids = list(range(n_gpu))

    model = eval(args.model)()

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
    grouped_dataframe = input_dataframe.groupby('group')
    input_dataframe_t1 = grouped_dataframe.get_group(0)
    input_dataframe_t2 = grouped_dataframe.get_group(1)
    image_fns_t1 = input_dataframe_t1["image_fn"].values
    label_fns_t1 = input_dataframe_t1["label_fn"].values
    image_fns_t2 = input_dataframe_t2["image_fn"].values
    label_fns_t2 = input_dataframe_t2["label_fn"].values

    inf_iou = 0

    for idx in range(len(image_fns_t1)):
        tic = time.time()
        image_fn_t1 = image_fns_t1[idx]
        label_fn_t1 = label_fns_t1[idx]
        image_fn_t2 = image_fns_t2[idx]
        label_fn_t2 = label_fns_t2[idx]

        print("(%d/%d) Processing image_%s" % (idx + 1, len(image_fns_t1), Path(image_fn_t1).stem.split('_')[0]),
              end=" ... ")

        # -------------------
        # Load input and create dataloader
        # -------------------
        def image_transforms(img_t1, img_t2):
            img_t1 = (img_t1 - config.NAIP_2013_MEANS) / config.NAIP_2013_STDS
            img_t2 = (img_t2 - config.NAIP_2017_MEANS) / config.NAIP_2017_STDS

            img_t1 = np.rollaxis(img_t1, 2, 0).astype(np.float32)
            img_t1 = torch.from_numpy(img_t1)
            img_t2 = np.rollaxis(img_t2, 2, 0).astype(np.float32)
            img_t2 = torch.from_numpy(img_t2)

            return img_t1, img_t2

        with rasterio.open(image_fn_t1) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        with rasterio.open(label_fn_t1) as f:
            label_t1 = utils.NLCD_CLASS_TO_IDX_MAP[f.read().squeeze()]
            label_t1_reduced = config.NLCD_IDX_TO_REDUCED_LC_MAP[label_t1]

        with rasterio.open(label_fn_t2) as f:
            label_t2 = utils.NLCD_CLASS_TO_IDX_MAP[f.read().squeeze()]
            label_t2_reduced = config.NLCD_IDX_TO_REDUCED_LC_MAP[label_t2]

        label = (label_t1_reduced != label_t2_reduced).astype(np.uint8)

        dataset = TileInferenceCDetDataset(
            img_fn_t1=image_fn_t1,
            img_fn_t2=image_fn_t2,
            chip_size=CHIP_SIZE,
            stride=CHIP_STRIDE,
            transform=image_transforms
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # -------------------
        # Run model and organize output
        # -------------------

        output = np.zeros((config.CD_NCLASSES, input_height, input_width), dtype=np.float32)
        kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
        kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        for i, (img1, img2, coords) in enumerate(dataloader):
            # TTA
            img1, img2 = _test_augment(img1[0].numpy(), img2[0].numpy())
            img1 = img1.to(device)
            img2 = img2.to(device)
            with torch.no_grad():
                outputs = model(img1, img2)
                outputs = F.softmax(outputs, dim=1).cpu().numpy()
            t_output = _test_augment_pred(outputs)
            for j in range(t_output.shape[0]):
                y, x = coords[j]

                output[:, y:y + CHIP_SIZE, x:x + CHIP_SIZE] += t_output[j] * kernel
                counts[y:y + CHIP_SIZE, x:x + CHIP_SIZE] += kernel

        output = output / counts
        output_hard = output.argmax(axis=0).astype(np.uint8)
        # iou todo
        iou = IoU(output_hard, label)
        inf_iou += iou
        # -------------------
        # Save output
        # -------------------
        # output_profile = input_profile.copy()
        # output_profile["driver"] = "GTiff"
        # output_profile["dtype"] = "uint8"
        # output_profile["count"] = 1
        # output_profile["nodata"] = 0
        #
        # output_fn = Path(args.output_dir) / "{}_predictions.tif".format(Path(image_fn_t1).stem.split('_')[0])
        #
        # with rasterio.open(output_fn, "w", **output_profile) as f:
        #     f.write(output_hard, 1)
        #     f.write_colormap(1, utils.NLCD_IDX_COLORMAP)

        # -------------------
        # plot output
        # -------------------

        print("finished in %0.4f seconds" % (time.time() - tic))

    print(inf_iou / len(image_fns_t1))


if __name__ == "__main__":
    main()
