import numpy as np
import torch
import cv2

import utils


def transform(img, labels, group, data_aug_prob=0):
    if group == 0:
        img = (img - utils.NAIP_2013_MEANS) / utils.NAIP_2013_STDS
    elif group == 1:
        img = (img - utils.NAIP_2017_MEANS) / utils.NAIP_2017_STDS
    else:
        raise ValueError("group not recognized")

    labels = utils.NLCD_CLASS_TO_IDX_MAP[labels]

    if data_aug_prob > 0:
        data_augment = Compose([
        # RandomHueSaturationValue(data_aug_prob),
        RandomShiftScaleRotate(data_aug_prob),
        RandomHorizontalFlip(data_aug_prob),
        RandomVerticleFlip(data_aug_prob),
        RandomRotate90(data_aug_prob)
    ])
        augmented = data_augment(img, labels)
        img = augmented[0]
        labels = augmented[1].copy()

    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    labels = torch.from_numpy(labels.astype(np.long))

    return img, labels


def nodata_check(img, labels):
    return np.any(labels == 0) or np.any(np.sum(img == 0, axis=2) == 4)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class RandomHueSaturationValue:
    def __init__(self, u=0.):
        self.u = u

    def get_params(self, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15)):
        return {'hue_shift': np.random.uniform(hue_shift_limit[0], hue_shift_limit[1]),
                'sat_shift': np.random.uniform(sat_shift_limit[0], sat_shift_limit[1]),
                'val_shift': np.random.uniform(val_shift_limit[0], val_shift_limit[1])}

    def fix_shift_values(self, img, *args):
        """
        shift values are normally specified in uint, but if your data is float - you need to remap values
        """
        if np.ceil(img.max()) == 1:
            return list(map(lambda x: x / 255, args))
        return args

    def shift_hsv(self, img, hue_shift, sat_shift, val_shift):
        dtype = img.dtype
        maxval = np.max(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
        h, s, v = cv2.split(img)
        h = cv2.add(h, hue_shift)
        h = np.where(h < 0, maxval - h, h)
        h = np.where(h > maxval, h - maxval, h)
        h = h.astype(dtype)
        s = clip(cv2.add(s, sat_shift), dtype, maxval)
        v = clip(cv2.add(v, val_shift), dtype, maxval)
        img = cv2.merge((h, s, v)).astype(dtype)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def __call__(self, image, label):
        if np.random.random() < self.u:
            params = self.get_params()
            hue_shift, sat_shift, val_shift = self.fix_shift_values(image, *params.values())
            image = self.shift_hsv(image, hue_shift, sat_shift, val_shift)
        return image, label


class RandomShiftScaleRotate:
    def __init__(self, u=0.):
        self.u = u

    @staticmethod
    def get_params(image, angle_scope=0, scale_scope=0.1, aspect_scope=0.1, shift_scope=0.1):

        height, width = image.shape[:2]

        # rotate
        angle = np.random.uniform(-angle_scope, angle_scope)
        scale = np.random.uniform(1 - scale_scope, 1 + scale_scope)
        aspect = np.random.uniform(1 - aspect_scope, 1 + aspect_scope)

        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)

        dx = round(np.random.uniform(-shift_scope, shift_scope) * width)
        dy = round(np.random.uniform(-shift_scope, shift_scope) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        return mat

    @staticmethod
    def shift_scale_roate(img, mat, flags):
        height, width = img.shape[:2]

        img = cv2.warpPerspective(img, mat, (width, height), flags=flags,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0,))

        return img

    def __call__(self, image, label):
        if np.random.random() < self.u:
            mat = self.get_params(image)
            image = self.shift_scale_roate(image, mat, cv2.INTER_LINEAR)
            label = self.shift_scale_roate(label, mat, cv2.INTER_NEAREST)
        return image, label


class RandomHorizontalFlip:
    def __init__(self, u=0.):
        self.u = u

    def __call__(self, image, label):
        if np.random.random() < self.u:
            image = np.fliplr(image)
            label = np.fliplr(label)
        return image, label


class RandomVerticleFlip:
    def __init__(self, u=0.):
        self.u = u

    def __call__(self, image, label):
        if np.random.random() < self.u:
            image = np.flipud(image)
            label = np.flipud(label)
        return image, label


class RandomRotate90:
    """Anti-clockwise roation with 90 * k (k = 1, 2, 3)"""
    def __init__(self, u=0.):
        self.u = u

    def __call__(self, image, label):
        if np.random.random() < self.u:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
        return image, label


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)