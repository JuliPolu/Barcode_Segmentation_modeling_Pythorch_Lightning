from pathlib import Path
from typing import Optional, Union

import albumentations as albu
import jpeg4py as jpeg
import numpy as np
from torch.utils.data import Dataset

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


class SegmentationDataset(Dataset):
    def __init__(
        self,
        dataframe: np.ndarray,
        image_folder: Path,
        transforms: Optional[TRANSFORM_TYPE] = None,
    ):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):  # noqa: WPS210
        row = self.dataframe[idx]

        # Load image
        image_path = self.image_folder / row[0]
        image = jpeg.JPEG(str(image_path)).decode()

        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x_from, y_from, width, height = int(row[1]), int(row[2]), int(row[3]), int(row[4])  # noqa: WPS221
        mask[y_from : y_from + height, x_from : x_from + width] = 1  # noqa: WPS221
        mask = mask.astype(np.uint8)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
            transformed_mask = transformed_mask.unsqueeze(0)
        else:
            transformed_image = image
            transformed_mask = mask

        return transformed_image, transformed_mask
