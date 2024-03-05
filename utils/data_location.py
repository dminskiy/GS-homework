import os

DATA_DIR = "dataset"
DATA_DIR_TR_V1 = os.path.join(DATA_DIR, "transformed_v1")
DATA_DIR_TR_V2 = os.path.join(DATA_DIR, "transformed_v2")
DATA_DIR_TR_V3 = os.path.join(DATA_DIR, "transformed_v3")
DATA_DIR_TR_V4 = os.path.join(DATA_DIR, "transformed_v4")

VISIALISATIONS_DIR = "visualisations"
MODELS_DIR = "trained_models"
FINAL_PREDICTIONS_DIR = "final_predictions"

TRAINING_SET_FILE = os.path.join(DATA_DIR, "train.csv")
VALIDATION_SET_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_SET_FILE = os.path.join(DATA_DIR, "test.csv")

DATA_FILES = [TRAINING_SET_FILE, TEST_SET_FILE, VALIDATION_SET_FILE]
