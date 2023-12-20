import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_subsets(
    annotation: pd.DataFrame,
    train_fraction: float = 0.8,
    test_size: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_subset, test_val_subset = train_test_split(annotation, test_size=(1 - train_fraction))

    test_subset, valid_subset = train_test_split(test_val_subset, test_size=test_size)

    logging.info('Splitting of dataset is completed')

    return train_subset, valid_subset, test_subset
