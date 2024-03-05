import pandas as pd
from utils.logger import setup_logger
from utils.col_names import *
from utils.data_location import *

logger = setup_logger(__name__)


def get_kma_list() -> list:
    cols2read = [ORIGIN, DESTINATION]
    train_df = pd.read_csv(TRAINING_SET_FILE, usecols=cols2read)
    test_df = pd.read_csv(TEST_SET_FILE, usecols=cols2read)
    validation_df = pd.read_csv(VALIDATION_SET_FILE, usecols=cols2read)

    combined_df = pd.concat(
        [train_df, test_df, validation_df], axis=0, ignore_index=True
    )

    assert (
        combined_df.shape[0]
        == train_df.shape[0] + test_df.shape[0] + validation_df.shape[0]
    )
    assert (
        combined_df.shape[1]
        == train_df.shape[1]
        == test_df.shape[1]
        == validation_df.shape[1]
    )

    all_kmas = combined_df[ORIGIN].to_list() + combined_df[DESTINATION].to_list()

    assert len(all_kmas) == combined_df.shape[0] * combined_df.shape[1]

    return list(set(all_kmas))


if __name__ == "__main__":

    all_kmas = get_kma_list()

    for file in DATA_FILES:
        # for file in [TRAINING_SET_FILE]:
        file_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)

        # Split pickup date into days and hours
        df[PICKUP_DATE] = pd.to_datetime(df[PICKUP_DATE])
        df[PICKUP_DATE_HOUR] = df[PICKUP_DATE].dt.hour
        df[PICKUP_DATE_DAY] = df[PICKUP_DATE].dt.day
        df[PICKUP_DATE_WEEK_DAY] = df[PICKUP_DATE].dt.day_of_week
        df[PICKUP_DATE_MONTH] = df[PICKUP_DATE].dt.month
        df[PICKUP_DATE_YEAR] = df[PICKUP_DATE].dt.year

        # Visualise the count of runs per hour in a day & day in a week
        df["cnt"] = len(df[PICKUP_DATE]) * [1]
        df_hours = df.groupby(PICKUP_DATE_HOUR).agg("count")
        df_days = df.groupby(PICKUP_DATE_WEEK_DAY).agg("count")

        # Encode Tranport Type using OneCodeEncoder - low dim, not related
        df = pd.get_dummies(df, columns=[TRANSPORT_TYPE])
        df = pd.get_dummies(df, columns=[ORIGIN])
        df = pd.get_dummies(df, columns=[DESTINATION])

        for kma in all_kmas:
            origina_kma = f"origin_kma_{kma}"
            destination_kma = f"destination_kma_{kma}"
            if origina_kma not in df.columns:
                df[origina_kma] = 0
            if destination_kma not in df.columns:
                df[destination_kma] = 0

        # Drop helper col and originals that have been transformed
        df.drop(["cnt", PICKUP_DATE], axis=1, inplace=True)
        df = df.sort_index(axis=1)
        df.to_csv(os.path.join(DATA_DIR_TR_V4, f"{file_name}.csv"), index=False)

        print(df.info())
        print(df.describe())
