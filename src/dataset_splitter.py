import logging
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def split_subsets(
    annotation: pd.DataFrame,
    train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбиение датасета на train/valid/test."""
    
    train_subset, test_val_subset = train_test_split(
        annotation, 
        test_size=(1 - train_fraction),
        random_state=42
    )

    test_subset, valid_subset = train_test_split(
        test_val_subset, 
        test_size=0.5,
        random_state=100
    )

    logging.info('Splitting of dataset is completed')

    return train_subset, valid_subset, test_subset
