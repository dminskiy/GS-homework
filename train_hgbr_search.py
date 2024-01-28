import os

import pandas as pd
import pickle
from datetime import datetime

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from utils.col_names import *
from utils.data_location import *
from utils.data_loader import DatasetVersion, get_training_data, get_validation_data
from utils.loss_functions import mape_loss

DATETIME_VIEW_STR_YHMS = "%a_%d_%b_%Y_%H_%M_%S"


if __name__ == "__main__":
    dataset_version = DatasetVersion.V1
    cols2load = [
        RATE,
        DISTANCE,
        WEIGHT,
        # PICKUP_DATE_DAY,
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

    param_grid = [
        {
            "loss": ["gamma"],
            "max_iter": [5000, 10000, 50000],
            "learning_rate": [0.01, 0.1, 0.5],
            "max_leaf_nodes": [40, 50, 60],
        },
    ]

    mape_loss_scorer = make_scorer(mape_loss, greater_is_better=False)
    search = GridSearchCV(
        estimator=HistGradientBoostingRegressor(
            categorical_features=[ORIGIN_ENCODED, DESTINATION_ENCODED],
        ),
        param_grid=param_grid,
        scoring=mape_loss_scorer,
        cv=5,
        verbose=3,
    )

    search.fit(x_train, y_train)

    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]

    print(results_df)

    save_name = f"HGBR_Search_num_feat_{x_train.shape[1]}_dataset_{dataset_version.value}_{datetime.now().strftime(DATETIME_VIEW_STR_YHMS)}.pkl"
    save_path = os.path.join(MODELS_DIR, save_name)
    with open(save_path, "wb") as f:
        pickle.dump(search, f)
