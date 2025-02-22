from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

from utils.col_names import *
from utils.data_loader import DatasetVersion, get_training_data, get_validation_data
from utils.loss_functions import mape_loss


if __name__ == "__main__":
    cols2load = None

    x_train, y_train = get_training_data(
        dataset_version=DatasetVersion.V1, features=cols2load
    )
    x_val, y_val = get_validation_data(
        dataset_version=DatasetVersion.V1, features=cols2load
    )

    reg = KNeighborsRegressor(n_neighbors=12, weights="distance", algorithm="brute")
    reg.fit(x_train, y_train)

    preds = reg.predict(x_val)
    preds = preds.round(4)

    mape = mape_loss(y_val, preds)

    print(reg.get_params())
    print(f"MAPE: {round(mape, 2)}")

    # {'algorithm': 'brute', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 12, 'p': 2, 'weights': 'distance'}
    # MAPE: 16.75
