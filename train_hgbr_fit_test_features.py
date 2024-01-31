import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from utils.col_names import *
from utils.data_location import *
from utils.data_loader import DatasetVersion, get_training_data, get_validation_data
from utils.loss_functions import mape_loss

DATETIME_VIEW_STR_YHMS = "%a_%d_%b_%Y_%H_%M_%S"
LR = 0.01
LOSS = "gamma"
ITER = 50000  # 50000 (early stopping enabled by default at > 10k as per docs)
MAX_LEAF_NODES = 60


if __name__ == "__main__":
    dataset_version = DatasetVersion.V1
    cols2load = [
        RATE,
        DISTANCE,
        WEIGHT,
        PICKUP_DATE_DAY,
        PICKUP_DATE_WEEK_DAY,
        PICKUP_DATE_YEAR,
        PICKUP_DATE_MONTH,
        PICKUP_DATE_HOUR,
        ORIGIN_ENCODED,
        DESTINATION_ENCODED,
    ] + TRANSPORT_TYPES

    x_train, y_train = get_training_data(
        dataset_version=dataset_version, features=cols2load
    )
    x_val, y_val = get_validation_data(
        dataset_version=dataset_version, features=cols2load
    )

    print(f"Training Shape: {x_train.shape}")

    reg = HistGradientBoostingRegressor(
        loss=LOSS,
        max_iter=ITER,
        learning_rate=LR,
        max_leaf_nodes=MAX_LEAF_NODES,
        random_state=42,
        categorical_features=[ORIGIN_ENCODED, DESTINATION_ENCODED],
    )

    reg.fit(x_train, y_train)

    preds = reg.predict(x_val)

    if dataset_version == DatasetVersion.V2:
        preds = np.floor(preds) / 10000
    else:
        preds = preds.round(4)

    mape = round(mape_loss(y_val, preds), 2)

    save_name = f"HGBR_mape_{mape}_iter_{ITER}_lr_{LR}_loss_{LOSS}_leafs_{MAX_LEAF_NODES}_num_feat_{x_train.shape[1]}_dataset_{dataset_version.value}_{datetime.now().strftime(DATETIME_VIEW_STR_YHMS)}.pkl"
    result = permutation_importance(
        reg, x_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
    )

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=x_train.columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances (train set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    plt.savefig(
        os.path.join(
            VISIALISATIONS_DIR,
            f"feature_importance_count_{x_train.shape[1]}_{save_name}.png",
        )
    )

    print(reg.get_params())
    print(f"## MAPE: {round(mape, 2)}")
