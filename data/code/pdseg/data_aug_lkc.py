# -*- coding: utf-8 -*-
#  @Author: KunchangLi
#  @Date: 2020-10-14 09:56:54
#  @LastEditor: KunchangLi
#  @LastEditTime: 2020-11-07 18:15:04


from albumentations import (
    HorizontalFlip, VerticalFlip, 
    RandomContrast, RandomGamma,
    RandomBrightness, ElasticTransform, 
    GridDistortion, OpticalDistortion,
    ShiftScaleRotate, RandomRotate90,
    RandomResizedCrop, CLAHE, IAASharpen,
    MotionBlur, GaussNoise, MaskDropout,
    GridDropout, ColorJitter,
    RandomBrightnessContrast,
    OneOf, Compose
)

from utils.config import cfg

# 自定义增强
# 调用时确保yaml里的CROP_SIZE和FIX_RESIZE_SIZE一致
# 同时设置MEAN和STD正确归一化

# https://github.com/sneddy/pneumothorax-segmentation/blob/master/unet_pipeline/transforms/train_transforms_complex_1024.json
# 去除resize和归一化，增加水平翻转
def aug_baseline(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        OneOf([
            RandomContrast(),
            RandomGamma(),
            RandomBrightness()
        ], p=0.3),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(),
            OpticalDistortion(distort_limit=2, shift_limit=0.5)
        ], p=0.3),
        ShiftScaleRotate()
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_baseline_randRotate90(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        OneOf([
            RandomContrast(),
            RandomGamma(),
            RandomBrightness()
        ], p=0.3),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(),
            OpticalDistortion(distort_limit=2, shift_limit=0.5)
        ], p=0.3),
        ShiftScaleRotate(rotate_limit=0),
        RandomRotate90()
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_baseline_randCrop(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        OneOf([
            RandomContrast(),
            RandomGamma(),
            RandomBrightness()
        ], p=0.3),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(),
            OpticalDistortion(distort_limit=2, shift_limit=0.5)
        ], p=0.3),
        RandomResizedCrop(width=cfg.TRAIN_CROP_SIZE[0], height=cfg.TRAIN_CROP_SIZE[1], scale=[0.5, 1.0]),
        RandomRotate90()
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_baseline_CLAHE_Sharpen(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        OneOf([
            RandomContrast(),
            RandomGamma(),
            RandomBrightness()
        ], p=0.3),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(),
            OpticalDistortion(distort_limit=2, shift_limit=0.5)
        ], p=0.3),
        ShiftScaleRotate(),
        CLAHE(),
        IAASharpen()
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_baseline_simple(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        ShiftScaleRotate()
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_baseline_simple_motionBlur(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        ShiftScaleRotate(),
        MotionBlur(p=0.1)
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_baseline_simple_gaussNoise(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        ShiftScaleRotate(),
        GaussNoise(p=0.1)
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_baseline_simple_maskDropout(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        ShiftScaleRotate(),
        MaskDropout(mask_fill_value=cfg.DATASET.IGNORE_INDEX, p=0.1)
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_baseline_simple_gridDropout(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        ShiftScaleRotate(),
        GridDropout(mask_fill_value=cfg.DATASET.IGNORE_INDEX, p=0.1)
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_baseline_simple_randBC(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        ShiftScaleRotate(),
        RandomBrightnessContrast(p=0.3)
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_baseline_simple_randCrop(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomResizedCrop(width=cfg.TRAIN_CROP_SIZE[0], height=cfg.TRAIN_CROP_SIZE[1], scale=[0.01, 1.0], p=0.5)
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_baseline_simple_randRotate90(image, mask):
    aug = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(p=0.3)
    ])
    aug_img = aug(image=image, mask=mask)
    return aug_img['image'], aug_img['mask']


def aug_gaussNoise(image, mask):
    if cfg.AUG.RICH_CROP.GAUSSNOISE:
        aug = GaussNoise(p=cfg.AUG.RICH_CROP.GAUSSNOISE_RATIO)
        aug_img = aug(image=image, mask=mask)
        return aug_img['image'], aug_img['mask']
    else:
        return image, mask


def aug_RandomRotate90(image, mask):
    if cfg.AUG.RICH_CROP.RANDOM_ROTATE90 > 0:
        aug = RandomRotate90(p=cfg.AUG.RICH_CROP.RANDOM_ROTATE90)
        aug_img = aug(image=image, mask=mask)
        return aug_img['image'], aug_img['mask']
    else:
        return image, mask