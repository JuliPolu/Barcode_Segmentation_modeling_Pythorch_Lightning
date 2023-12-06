import os
from typing import Optional, Union

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]

class SegmentationDataset(Dataset):
    def __init__(self,
        dataframe: pd.DataFrame,
        image_folder: str,
        transforms: Optional[TRANSFORM_TYPE] = None,
    ):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Load image
        image_path = os.path.normpath(os.path.join(self.image_folder, row['filename']))
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x, y, w, h = row['x_from'], row['y_from'], row['width'], row['height']
        mask[y:y+h, x:x+w] = 1
        
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
            

        return transformed_image, transformed_mask
    