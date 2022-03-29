import os
from interspeechmi.constants import DATA_DIR

ANNO_MI_DATA_DIR = os.path.join(DATA_DIR, "anno_mi")
ANNO_MI_PREPROCESSED_DATA_DIR = os.path.join(ANNO_MI_DATA_DIR, "preprocessed")

for data_dir in [
    ANNO_MI_DATA_DIR, 
    ANNO_MI_PREPROCESSED_DATA_DIR, 
]:
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

###########################################################################

ANNO_MI_URL = "https://github.com/uccollab/AnnoMI/archive/refs/heads/main.zip"
ANNO_MI_DATASET_ZIP_PATH = os.path.join(ANNO_MI_DATA_DIR, "anno_mi_dataset.zip")
ANNO_MI_DATASET_DIR = os.path.join(ANNO_MI_DATA_DIR, "anno_mi_dataset")
ANNO_MI_PATH = os.path.join(ANNO_MI_DATASET_DIR, "dataset.csv")
ANNO_MI_NORMALIZED_PATH = os.path.join(ANNO_MI_PREPROCESSED_DATA_DIR, "dataset.normalized.csv")
ANNO_MI_NORMALIZED_AUGMENTED_PATH = os.path.join(ANNO_MI_PREPROCESSED_DATA_DIR, "dataset.normalized.augmented.csv")

###########################################################################

