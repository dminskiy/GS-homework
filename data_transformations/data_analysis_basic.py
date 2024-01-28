import os
import pandas as pd

from utils.logger import setup_logger
from utils.col_names import *
from utils.data_location import *

logger = setup_logger(__name__)

### DATA Notes
# 1. Train has 80 missing values for the weight
# 2. There's a fair amount of duplicate rows
#   train (3.5%), test (7.42%), val(4.9%)
#   test has nearly double the amount -> how much is rate influenced by the day at pickup? time?
# 3. Train & test have weights upto ~190k, whereas val only goes to 70k -> can val be missleading as a result?
#   Try pushing some large vals to validation set from training

if __name__ == "__main__":

    transport_types = None

    for file in DATA_FILES:
        df = pd.read_csv(os.path.join(file))

        # Get a basic idea of the data
        print("\n\n #############################")
        print(f"\n## FILE: {file}")
        print("\n# Head:")
        print(df.head())
        print("\n# Describe:")
        print(df.describe())
        print("\n# Shape:")
        print(df.shape)
        print("\n# D Types:")
        print(df.dtypes)
        print("\n# Missing Vals (null):")
        print(df.isnull().sum())
        print("\n# Duplicate Vals:")
        print(df.duplicated().sum())
        print("\n# Unique per col:")

        for col in df.columns:
            # Make sure transport types are consistent across all files
            if col == TRANSPORT_TYPE and transport_types is None:
                transport_types = list(df[col].unique()).sort()
            else:
                local_tt = list(df[col].unique()).sort()
                if local_tt != transport_types:
                    logger.warn(
                        f"Transport Types do not match. Saved: {transport_types}. Found: {local_tt}"
                    )

            print(f"Col: {col} ({len(df[col])})")
            print(f"Unique: {len(df[col].unique())}")
