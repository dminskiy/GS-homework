from sklearn.svm import LinearSVR

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

    reg = LinearSVR(dual="auto")
    reg.fit(x_train, y_train)

    preds = reg.predict(x_val)
    preds = preds.round(4)

    mape = mape_loss(y_val, preds)

    print(f"MAPE: {round(mape, 2)}")
