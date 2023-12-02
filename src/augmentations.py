from typing import Union

import albumentations as albu
from albumentations.pytorch import ToTensorV2

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


def get_transforms(
    width: int,
    height: int,
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> TRANSFORM_TYPE:
    transforms = []

    if preprocessing:
        transforms.append(albu.Resize(height=height, width=width))

    if augmentations:
        transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=(-0.05, 0.05),
                        rotate_limit=(-15, 15),
                        interpolation=2,
                        border_mode=0,
                        value=(0, 0, 0),
                        p=0.5,
                ),
                albu.OneOf(
                       [
                        albu.CLAHE(clip_limit=2, p=0.5),
                        albu.RandomBrightnessContrast(
                            brightness_limit=0.1,
                            contrast_limit=0.2,
                            brightness_by_max=True,
                            p=0.5,
                        ),
                    ],
                    p=0.3,
                ),
            ],
        )

    if postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)
