import numpy as np
import torch
import cv2

import utils
import config


def transform(img, labels, group, data_aug_prob=0):
    if group == 0:
        img = (img - config.NAIP_2013_MEANS) / config.NAIP_2013_STDS
    elif group == 1:
        img = (img - config.NAIP_2017_MEANS) / config.NAIP_2017_STDS
    else:
        raise ValueError("group not recognized")

    labels = utils.NLCD_CLASS_TO_IDX_MAP[labels]

    if data_aug_prob > 0:
        data_augment = Compose([
        # RandomHueSaturationValue(data_aug_prob),
        RandomColorJitter(data_aug_prob),
        RandomShiftScaleRotate(data_aug_prob),
        # RandomScale(data_aug_prob),
        RandomHorizontalFlip(data_aug_prob),
        RandomVerticleFlip(data_aug_prob),
        RandomRotate90(data_aug_prob)
    ])
        augmented = data_augment(img, labels)
        img = augmented[0]
        labels = augmented[1].copy()

    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    h, w = img.shape[1:]
    lr_labels = torch.from_numpy(cv2.resize(labels, (w//32, h//32), interpolation=cv2.INTER_NEAREST).astype(np.long))
    hr_labels = torch.from_numpy(labels.astype(np.long))

    return img, lr_labels, hr_labels


def transform2(img_t1, img_t2, labels, data_aug_prob=0):
    img_t1 = (img_t1 - config.NAIP_2013_MEANS) / config.NAIP_2013_STDS
    img_t2 = (img_t2 - config.NAIP_2017_MEANS) / config.NAIP_2017_STDS

    if data_aug_prob > 0:
        data_augment = Compose2([
        RandomColorJitter2(data_aug_prob),
        RandomShiftScaleRotate2(data_aug_prob),
        RandomHorizontalFlip2(data_aug_prob),
        RandomVerticleFlip2(data_aug_prob),
        RandomRotate902(data_aug_prob)
    ])
        augmented = data_augment(img_t1, img_t2, labels)
        img_t1 = augmented[0]
        img_t2 = augmented[1]
        labels = augmented[2].copy()

    img_t1 = np.rollaxis(img_t1, 2, 0).astype(np.float32)
    img_t1 = torch.from_numpy(img_t1)
    img_t2 = np.rollaxis(img_t2, 2, 0).astype(np.float32)
    img_t2 = torch.from_numpy(img_t2)
    labels = torch.from_numpy(labels.astype(np.long))

    return img_t1, img_t2, labels


def nodata_check(img, labels):
    return np.any(labels == 0) or np.any(np.sum(img == 0, axis=2) == 4)


def nodata_check2(labels):
    return (labels != 0).sum() / labels.size < 0.09


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Compose2:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_1, image_2, label):
        for t in self.transforms:
            image_1, image_2, label = t(image_1, image_2, label)
        return image_1, image_2, label


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


class RandomColorJitter:
    def __init__(self, u=0.):
        self.u = u

    def __call__(self, image, label):
        if np.random.random() < self.u:
            _, _, c = image.shape
            contra_adj = 0.05
            bright_adj = 0.05

            ch_mean = np.mean(image, axis=(0, 1), keepdims=True).astype(np.float32)

            contra_mul = np.random.uniform(1 - contra_adj, 1 + contra_adj, (1, 1, c)).astype(
                np.float32
            )
            bright_mul = np.random.uniform(1 - bright_adj, 1 + bright_adj, (1, 1, c)).astype(
                np.float32
            )

            image = (image - ch_mean) * contra_mul + ch_mean * bright_mul

        return image, label


class RandomColorJitter2:
    def __init__(self, u=0.):
        self.u = u

    def __call__(self, image_1, image_2, label):
        if np.random.random() < self.u:
            _, _, c = image_1.shape
            contra_adj = 0.05
            bright_adj = 0.05

            ch_mean_1 = np.mean(image_1, axis=(0, 1), keepdims=True).astype(np.float32)
            ch_mean_2 = np.mean(image_2, axis=(0, 1), keepdims=True).astype(np.float32)

            contra_mul = np.random.uniform(1 - contra_adj, 1 + contra_adj, (1, 1, c)).astype(
                np.float32
            )
            bright_mul = np.random.uniform(1 - bright_adj, 1 + bright_adj, (1, 1, c)).astype(
                np.float32
            )

            image_1 = (image_1 - ch_mean_1) * contra_mul + ch_mean_1 * bright_mul
            image_2 = (image_2 - ch_mean_2) * contra_mul + ch_mean_2 * bright_mul

        return image_1, image_2, label


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


class RandomShiftScaleRotate2:
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

    def __call__(self, image_1, image_2, label):
        if np.random.random() < self.u:
            mat = self.get_params(image_1)
            image_1 = self.shift_scale_roate(image_1, mat, cv2.INTER_LINEAR)
            image_2 = self.shift_scale_roate(image_2, mat, cv2.INTER_LINEAR)
            label = self.shift_scale_roate(label, mat, cv2.INTER_NEAREST)
        return image_1, image_2, label


class RandomHorizontalFlip:
    def __init__(self, u=0.):
        self.u = u

    def __call__(self, image, label):
        if np.random.random() < self.u:
            image = np.fliplr(image)
            label = np.fliplr(label)
        return image, label


class RandomHorizontalFlip2:
    def __init__(self, u=0.):
        self.u = u

    def __call__(self, image_1, image_2, label):
        if np.random.random() < self.u:
            image_1 = np.fliplr(image_1)
            image_2 = np.fliplr(image_2)
            label = np.fliplr(label)
        return image_1, image_2, label


class RandomVerticleFlip:
    def __init__(self, u=0.):
        self.u = u

    def __call__(self, image, label):
        if np.random.random() < self.u:
            image = np.flipud(image)
            label = np.flipud(label)
        return image, label


class RandomVerticleFlip2:
    def __init__(self, u=0.):
        self.u = u

    def __call__(self, image_1, image_2, label):
        if np.random.random() < self.u:
            image_1 = np.flipud(image_1)
            image_2 = np.flipud(image_2)
            label = np.flipud(label)
        return image_1, image_2, label


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


class RandomRotate902:
    """Anti-clockwise roation with 90 * k (k = 1, 2, 3)"""
    def __init__(self, u=0.):
        self.u = u

    def __call__(self, image_1, image_2, label):
        if np.random.random() < self.u:
            k = np.random.randint(1, 4)
            image_1 = np.rot90(image_1, k)
            image_2 = np.rot90(image_2, k)
            label = np.rot90(label, k)
        return image_1, image_2, label


class RandomScale:
    def __init__(self, u=0.):
        self.u = u

    def __call__(self, image, label):
        if np.random.random() < self.u:
            scale_scope = 0.1  # todo
            scale_h = np.random.uniform(1, 1 + scale_scope)
            scale_w = np.random.uniform(1, 1 + scale_scope)
            height, width = image.shape[:2]
            new_height, new_width = int(height * scale_h), int(width * scale_w)
            image = cv2.resize(image, (new_height, new_width), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (new_height, new_width), interpolation=cv2.INTER_NEAREST)
            if height == new_height and width == new_width:
                return image, label
            elif height == new_height:
                h = 0
                w = np.random.randint(0, new_width - width)
            elif width == new_width:
                h = np.random.randint(0, new_height - height)
                w = 0
            else:
                h = np.random.randint(0, new_height - height)
                w = np.random.randint(0, new_width - width)
            image = image[w:w+width, h:h+height, :]
            label = label[w:w+width, h:h+height]
        return image, label


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


if __name__ == '__main__':
    aug = RandomScale(0.5)
    image = np.ones((256, 256, 3))
    labels = np.ones((256, 256))
    aug(image, labels)