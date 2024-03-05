import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.logger import setup_logger
from utils.col_names import *
from utils.data_location import *

logger = setup_logger(__name__)


if __name__ == "__main__":

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

        le = LabelEncoder()
        df[ORIGIN_ENCODED] = le.fit_transform(df[ORIGIN])
        df[DESTINATION_ENCODED] = le.fit_transform(df[DESTINATION])

        # Drop helper col and originals that have been transformed
        df.drop(["cnt", ORIGIN, DESTINATION, PICKUP_DATE], axis=1, inplace=True)
        df.to_csv(os.path.join(DATA_DIR_TR_V3, f"{file_name}.csv"), index=False)

        print(df.info())
        print(df.head())
