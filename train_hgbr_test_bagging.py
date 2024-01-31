import os

from datetime import datetime

from sklearn.ensemble import HistGradientBoostingRegressor, BaggingRegressor

from utils.col_names import *
from utils.data_location import *
from utils.data_loader import (
    DatasetVersion,
    get_training_data,
    get_validation_data,
    get_test_data,
)


DATETIME_VIEW_STR_YHMS = "%a_%d_%b_%Y_%H_%M_%S"
LR = 0.01
LOSS = "gamma"
ITER = 50000
MAX_LEAF_NODES = 60


if __name__ == "__main__":
    dataset_version = DatasetVersion.V1
    cols2load = [
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
        dataset_version=dataset_version, features=cols2load + [RATE]
    )
    x_val, y_val = get_validation_data(
        dataset_version=dataset_version, features=cols2load + [RATE]
    )
    df_test = get_test_data(dataset_version=dataset_version, features=cols2load)

    # Combine train and validation data
    x_train = x_train.append(x_val).reset_index(drop=True)
    y_train = y_train.append(y_val).reset_index(drop=True)

    print(f"Training Shape: {x_train.shape}")

    hgbr = HistGradientBoostingRegressor(
        loss=LOSS,
        max_iter=ITER,
        learning_rate=LR,
        max_leaf_nodes=MAX_LEAF_NODES,
        early_stopping=True,
        categorical_features=[9, 10],
        # categorical_features=[ORIGIN_ENCODED, DESTINATION_ENCODED],
    )

    reg = BaggingRegressor(estimator=hgbr, n_estimators=16, n_jobs=8, verbose=3)
    reg.fit(x_train, y_train)

    df_test["predicted_rate"] = reg.predict(df_test).round(4)

    model_name = f"HGBR_bagging_iter_{ITER}_lr_{LR}_loss_{LOSS}_leafs_{MAX_LEAF_NODES}_num_feat_{x_train.shape[1]}_dataset_{dataset_version.value}_{datetime.now().strftime(DATETIME_VIEW_STR_YHMS)}.pkl"
    df_test.to_csv(
        os.path.join(FINAL_PREDICTIONS_DIR, f"predicted_by_{model_name}.csv"),
        index=False,
    )
