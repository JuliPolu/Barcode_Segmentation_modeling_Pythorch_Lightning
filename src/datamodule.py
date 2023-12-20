import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.augmentations import get_transforms
from src.config import DataConfig
from src.dataset import SegmentationDataset
from src.dataset_splitter import split_subsets


class SegmentDM(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self._config = config
        self._train_transforms = get_transforms(
            width=config.width,
            height=config.height,
            encoder=config.encoder_name,
        )  # noqa: WPS221, E501
        self._valid_transforms = get_transforms(
            width=config.width,
            height=config.height,
            encoder=config.encoder_name,
            augmentations=False,
        )
        self._image_folder = Path(self._config.data_path).resolve()

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        split_and_save_datasets(
            Path(self._config.data_path),
            self._config.train_size,
            self._config.test_size,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            df_train = read_df(Path(self._config.data_path), 'train')
            df_valid = read_df(Path(self._config.data_path), 'valid')
            self.train_dataset = SegmentationDataset(
                df_train,
                image_folder=self._image_folder,
                transforms=self._train_transforms,
            )
            self.valid_dataset = SegmentationDataset(
                df_valid,
                image_folder=self._image_folder,
                transforms=self._valid_transforms,
            )

        elif stage == 'test':
            df_test = read_df(Path(self._config.data_path), 'test')
            self.test_dataset = SegmentationDataset(
                df_test,
                image_folder=self._image_folder,
                transforms=self._valid_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


def split_and_save_datasets(data_path: str, train_fraction: float = 0.8) -> None:
    df_path = data_path / 'annotations.tsv'
    df = pd.read_csv(df_path, delimiter='\t')
    logging.info(f'Original dataset: {len(df)}')
    df = df.drop_duplicates()
    df = df.drop(['code'], axis=1)
    logging.info(f'Final dataset: {len(df)}')

    train_df, valid_df, test_df = split_subsets(df, train_fraction=train_fraction)
    logging.info(f'Train dataset: {len(train_df)}')
    logging.info(f'Valid dataset: {len(valid_df)}')
    logging.info(f'Test dataset: {len(test_df)}')

    train_df.to_csv(data_path / 'df_train.csv', index=False)
    valid_df.to_csv(data_path / 'df_valid.csv', index=False)
    test_df.to_csv(data_path / 'df_test.csv', index=False)
    logging.info('Datasets successfully saved!')


def read_df(data_path: Path, mode: str) -> np.ndarray:
    df = pd.read_csv(data_path / f'df_{mode}.csv')
    return df.to_numpy()
