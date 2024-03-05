import os
from typing import Optional
from enum import Enum, auto

import pandas as pd

from utils.data_location import *
from utils.col_names import *


class DatasetVersion(Enum):
    ORIGINAL = auto()
    V1 = auto()
    V2 = auto()
    V3 = auto()
    V4 = auto()


def load_data(
    dataset_version: DatasetVersion,
    filename: str,
    dropna: bool = False,
    drop_duplicates: bool = False,
    features: Optional[list] = None,
) -> pd.DataFrame:
    cols2load = features

    if dataset_version == DatasetVersion.ORIGINAL:
        data_dir = DATA_DIR
    elif dataset_version == DatasetVersion.V1:
        data_dir = DATA_DIR_TR_V1
    elif dataset_version == DatasetVersion.V2:
        data_dir = DATA_DIR_TR_V2
    elif dataset_version == DatasetVersion.V3:
        data_dir = DATA_DIR_TR_V3
    elif dataset_version == DatasetVersion.V4:
        data_dir = DATA_DIR_TR_V4
    else:
        raise RuntimeError()

    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath, usecols=cols2load)

    if drop_duplicates:
        df = df.drop_duplicates()
        assert df.duplicated().all().sum() == 0
    if dropna:
        df = df.dropna(axis=0)
        assert df.isnull().all().sum() == 0

    return df


def get_training_data(
    dataset_version: DatasetVersion,
    dropna: bool = True,
    drop_duplicates: bool = True,
    features: Optional[list] = None,
) -> (pd.DataFrame, pd.DataFrame):
    df = load_data(
        dataset_version=dataset_version,
        filename="train.csv",
        dropna=dropna,
        drop_duplicates=drop_duplicates,
        features=features,
    )

    x_train = df.drop([RATE], axis=1)
    y_train = df[RATE]

    return x_train, y_train


def get_validation_data(
    dataset_version: DatasetVersion,
    features: Optional[list] = None,
) -> (pd.DataFrame, pd.DataFrame):
    df = load_data(
        dataset_version=dataset_version,
        filename="validation.csv",
        dropna=False,
        drop_duplicates=False,
        features=features,
    )

    x_val = df.drop([RATE], axis=1)
    y_val = df[RATE]

    return x_val, y_val


def get_test_data(
    dataset_version: DatasetVersion,
    features: Optional[list] = None,
) -> pd.DataFrame:
    df = load_data(
        dataset_version=dataset_version,
        filename="test.csv",
        dropna=False,
        drop_duplicates=False,
        features=features,
    )

    return df
