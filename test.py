import os
import joblib

from utils.col_names import *
from utils.data_location import *
from utils.data_loader import DatasetVersion, get_validation_data, get_test_data
from utils.loss_functions import mape_loss

DATASET_VERSION = DatasetVersion.V1
FINAL_FEATURES = [
    DISTANCE,
    WEIGHT,
    PICKUP_DATE_WEEK_DAY,
    PICKUP_DATE_YEAR,
    PICKUP_DATE_MONTH,
    PICKUP_DATE_HOUR,
    ORIGIN_ENCODED,
    DESTINATION_ENCODED,
] + TRANSPORT_TYPES


def load_and_validate(model) -> float:

    x_val, y_val = get_validation_data(
        dataset_version=DATASET_VERSION, features=FINAL_FEATURES + [RATE]
    )
    predicted_rates = model.predict(x_val).round(4)

    return round(mape_loss(y_val, predicted_rates), 2)


def generate_final_solution(model, model_name: str):
    # generate and save test predictions
    df_test = get_test_data(dataset_version=DATASET_VERSION, features=FINAL_FEATURES)
    df_test["predicted_rate"] = model.predict(df_test).round(4)
    df_test.to_csv(
        os.path.join(FINAL_PREDICTIONS_DIR, f"predicted_for_{model_name}.csv"),
        index=False,
    )


if __name__ == "__main__":

    for model_name in os.listdir(MODELS_DIR):
        if not model_name.endswith(".pkl"):
            continue

        model_path = os.path.join(MODELS_DIR, model_name)
        model = joblib.load(model_path)

        mape = load_and_validate(model)
        print(f"Accuracy of validation is {mape}%")

        if mape < 9:  # try to reach 9% or less for validation
            generate_final_solution(model, model_name.strip(".pkl"))
            print("'predicted.csv' is generated, please send it to us")
