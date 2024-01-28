import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.logger import setup_logger
from utils.col_names import *
from utils.data_location import *

logger = setup_logger(__name__)
random.seed(15637879334)


def get_unique_integer(existing: set) -> int:
    min_val = 1000
    max_val = 9999

    assert len(existing) < max_val - min_val

    while True:
        num = random.randint(min_val, max_val)
        if num not in existing:
            return num


def get_kma_map() -> dict:
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

    unique_kmas = list(set(all_kmas))

    generated_ints = set()
    kma_map = {
        val: get_unique_integer(generated_ints) for _, val in enumerate(unique_kmas)
    }

    return kma_map


if __name__ == "__main__":

    kma_map = get_kma_map()

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

        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 5)
        sns.barplot(data=df_hours, y="cnt", x=PICKUP_DATE_HOUR, ax=axes[0])
        sns.barplot(data=df_days, y="cnt", x=PICKUP_DATE_WEEK_DAY, ax=axes[1])
        plt.title(f"Run Distributions. File: {file}")
        axes[0].set(
            xlabel="Hours Of The Day",
            ylabel="Count",
            title="Plot On Count Across Hours in a Day",
        )
        axes[1].set(
            xlabel="Week Day Number",
            ylabel="Count",
            title="Plot On Count Across Days Of The Week",
        )
        plt.savefig(
            os.path.join(VISIALISATIONS_DIR, f"{file_name}_run_distributions.png")
        )

        # Plot weight per tranport type in each file
        plt.figure()
        sns.boxplot(data=df, y=WEIGHT, x=TRANSPORT_TYPE, orient="v")
        plt.title(f"Weight per Transport Type. File: {file}")
        plt.xlabel("Transport Type")
        plt.ylabel("Weight")
        plt.savefig(
            os.path.join(
                VISIALISATIONS_DIR, f"{file_name}_weigth_per_transport_type.png"
            )
        )

        # Plot weight per tranport type in each file
        plt.figure()
        sns.boxplot(data=df, y=DISTANCE, x=TRANSPORT_TYPE, orient="v")
        plt.title(f"Distance per Transport Type. File: {file}")
        plt.xlabel("Transport Type")
        plt.ylabel("Distance")
        plt.savefig(
            os.path.join(
                VISIALISATIONS_DIR, f"{file_name}_distance_per_transport_type.png"
            )
        )

        # Encode Tranport Type using OneCodeEncoder - low dim, not related
        df = pd.get_dummies(df, columns=[TRANSPORT_TYPE])
        # Encode Origin using label encoding with unique randon numbers in range [1000, 9999]
        df[ORIGIN_ENCODED] = df[ORIGIN].map(kma_map)
        # Encode Destination using label encoding with unique randon numbers in range [1000, 9999]
        df[DESTINATION_ENCODED] = df[DESTINATION].map(kma_map)

        # Drop helper col and originals that have been transformed
        df.drop(["cnt", ORIGIN, DESTINATION, PICKUP_DATE], axis=1, inplace=True)

        # Plot the Corr Matrix
        corr_matrix = df.corr()
        plt.figure(figsize=(15, 15))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True
        )
        plt.title("Correlation Matrix Heatmap")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.savefig(os.path.join(VISIALISATIONS_DIR, f"{file_name}_corr_matrix.png"))

        df.to_csv(os.path.join(DATA_DIR_TR_V1, f"{file_name}.csv"), index=False)
        plt.close()
