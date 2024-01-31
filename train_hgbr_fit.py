import os

import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime

from sklearn.ensemble import HistGradientBoostingRegressor

from utils.col_names import *
from utils.data_location import *
from utils.data_loader import DatasetVersion, get_training_data, get_validation_data
from utils.loss_functions import mape_loss

DATETIME_VIEW_STR_YHMS = "%a_%d_%b_%Y_%H_%M_%S"
LR = 0.01
LOSS = "gamma"
ITER = 50000  # 50000 (early stopping enabled by default at > 10k as per docs)
MAX_LEAF_NODES = 60
NUM_RUNS = 20

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
        categorical_features=[ORIGIN_ENCODED, DESTINATION_ENCODED],
    )

    best_score = 9999
    scores = []
    for _ in tqdm(range(NUM_RUNS), unit="Run"):
        reg.fit(x_train, y_train)

        preds = reg.predict(x_val)

        if dataset_version == DatasetVersion.V2:
            preds = np.floor(preds) / 10000
        else:
            preds = preds.round(4)

        mape = round(mape_loss(y_val, preds), 2)

        scores.append(mape)
        if mape < best_score:
            best_score = mape
            save_name = f"HGBR_mape_{mape}_iter_{ITER}_lr_{LR}_loss_{LOSS}_leafs_{MAX_LEAF_NODES}_num_feat_{x_train.shape[1]}_dataset_{dataset_version.value}_{datetime.now().strftime(DATETIME_VIEW_STR_YHMS)}.pkl"
            save_path = os.path.join(MODELS_DIR, save_name)
            with open(save_path, "wb") as f:
                pickle.dump(reg, f)

    scores = np.array(scores)
    print(f"Best Score: {best_score}")
    print(f"Worst Score: {np.max(scores)}")
    print(f"Average (std): {np.mean(scores).round(2)}({np.std(scores).round(2)})")
    print(f"Mean Value: {np.sort(scores)[int(np.floor(len(scores)/2))]}")
