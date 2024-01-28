import xgboost as xgb

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

    xgr = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=10,
        min_child_weight=6,
        gamma=0.4,
        eta=0.1,
        objective="reg:squarederror",
    )

    xgr.fit(x_train, y_train)

    preds = xgr.predict(x_val)
    preds = preds.round(4)

    mape = mape_loss(y_val, preds)

    print(f"MAPE: {round(mape, 2)}")
