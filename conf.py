import os
import numpy as np

### Default for Argparse options: settable by user via command line options
LANGUAGE = "latin"
###

### Parameters settable for users via this config file
MULTIPROCESSING = False
N_PROCESSES = 6 # None # None for using all
USE_GPU = False # GPU often slower than CPU, and runs out of memory for concat representation. but saves CPU availability
WRITE_CSV = True 

N_RUNS = 5 # 10
ART_VIGILANCE=0.25
ART_LEARNING_RATE=2

VIGILANCE_RANGE_STEP = 0.05 #0.02
MAX_VIGILANCE = 0.6

OUTPUT_DIR = "output"

SQUEEZE_INTO_VERBS=True
CONCAT_VERB_FEATURES=True
REMOVE_FEATURES_ALLZERO = True
SET_COMMON_FEATURES_TO_ZERO=False
NGRAMS=3
SAMPLE_FIRST = None # 1000
CELLS_PRESENT_ESTONIAN = ["ind.prs.1sg", "ind.prs.2sg", "ind.prs.3sg", "ind.prs.1pl", "ind.prs.2pl", "ind.prs.3pl"]
###



#### Not to be changed by user
VIGILANCE_RANGE = [x/100 for x in range(0,int(MAX_VIGILANCE*100), int(VIGILANCE_RANGE_STEP*100))]
CONFIG_STRING=f"---squeeze_into_verbs=={SQUEEZE_INTO_VERBS}---Concat_verb_features={CONCAT_VERB_FEATURES}---CommonFeat0={SET_COMMON_FEATURES_TO_ZERO}---Ngram={NGRAMS}"
LABEL_DENSITY = 5
EVAL_INTERVAL=20
# INFLECTION_CLASSES = ["I", "II", "III", "IV", "special"]
# N_INFLECTION_CLASSES = len(INFLECTION_CLASSES)
INITIAL_CLUSTERS = 1

FILLED_MARKERS = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']*16
# EMPTY_SYMBOL = True

currentdir = os.path.abspath("")
DATA_PATH = os.path.join(currentdir, "data")

paths = {
    "latin":
    {"archive_url": "https://zenodo.org/records/4039059/files/v2.0.4.zip",
     "archive_path": "Romance_Verbal_Inflection_Dataset-v2.0.4.zip",
     "file_path": os.path.join(DATA_PATH, "Romance_Verbal_Inflection_Dataset-v2.0.4"),
     "metadata_relative_path": "cldf/Wordlist-metadata.json",
     "conjugation_df_path": 'latin_data_df.csv'},
     "estonian":
    {"archive_url": "https://zenodo.org/records/8392744/files/v1.0.1.zip",
     "archive_path": "estonian-v.1.0.1.zip",
     "file_path": os.path.join(DATA_PATH, "estonian-v.1.0.1"),
     "conjugation_df_path": "estonian_data_df.csv"},
    "portuguese":
    {"archive_url": "https://zenodo.org/records/8392722/files/v2.0.1.zip",
     "archive_path": "portuguese-v.2.0.1.zip",
     "file_path": os.path.join(DATA_PATH, "portuguese-v.2.0.1"),
     "conjugation_df_path": "portuguese_data_df.csv"},
    # "arabic":
    # {"archive_url": "https://zenodo.org/records/10100678/files/aravelex-1.0.zip",
    #  "archive_path": "arabic-v.1.0.zip",
    #  "file_path": os.path.join(DATA_PATH, "arabic-v.1.0"),
    #  "conjugation_df_path": "arabic_conjugation_df.csv"},
}

###
