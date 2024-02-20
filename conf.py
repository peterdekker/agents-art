import os
import numpy as np

### Argparse options: settable by user via command line options. Default can be set here.
LANGUAGE = "latin"
FEATURES_SET = False
SOUNDCLASSES = "none"  #  One of: "none", "asjp" or "sca"
USE_CELLS_PRESENT = False
NGRAMS = 3
VIGILANCE_SINGLE_RUN = 0.5

params = {
    "language": {"default": LANGUAGE, "type": str},
    "features_set": {"default": FEATURES_SET, "type": bool},
    "soundclasses": {"default": SOUNDCLASSES, "type": str},
    "use_cells_present": {"default": USE_CELLS_PRESENT, "type": bool},
    "ngrams": {"default": NGRAMS, "type": int},
    "vigilance_single_run": {"default": VIGILANCE_SINGLE_RUN, "type": float},
}

mode_params = ["plot_data", "single_run", "eval_batches", "eval_vigilances", "eval_intervals", "baseline", "train_test"]

###

### Parameters settable for users via this config file
MULTIPROCESSING = True
N_PROCESSES = 2  # None # None for using all
USE_GPU = False  # GPU often slower than CPU, and runs out of memory for concat representation. but saves CPU availability
WRITE_CSV = True
N_RUNS = 5  # 10

ART_LEARNING_RATE = 2

VIGILANCE_RANGE_STEP = 0.02  # 0.02
MAX_VIGILANCE = 0.3

OUTPUT_DIR = "output"

SQUEEZE_INTO_VERBS = True
REMOVE_FEATURES_ALLZERO = True
SET_COMMON_FEATURES_TO_ZERO = False
SAMPLE_FIRST = None  # 1000
###


### Not to be changed by user
VIGILANCE_RANGE = [
    x/100 for x in range(0, int(MAX_VIGILANCE*100), int(VIGILANCE_RANGE_STEP*100))]
# CONFIG_STRING = f"---squeeze_into_verbs={SQUEEZE_INTO_VERBS}---FEATURES_SET={FEATURES_SET}---CommonFeat0={SET_COMMON_FEATURES_TO_ZERO}---Ngram={NGRAMS}---present={USE_CELLS_PRESENT}"
LABEL_DENSITY = 5
EVAL_INTERVAL = 20
# INFLECTION_CLASSES = ["I", "II", "III", "IV", "special"]
# N_INFLECTION_CLASSES = len(INFLECTION_CLASSES)
INITIAL_CLUSTERS = 1
FILLED_MARKERS = ['o', 'v', '^', '<', '>', '8',
                  's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']*16
# EMPTY_SYMBOL = True

currentdir = os.path.abspath("")
DATA_PATH = os.path.join(currentdir, "data")

paths = {
    "latin":
    {"archive_url": "https://zenodo.org/records/4039059/files/v2.0.4.zip",
     "archive_path": "Romance_Verbal_Inflection_Dataset-v2.0.4.zip",
     "file_path": os.path.join(DATA_PATH, "Romance_Verbal_Inflection_Dataset-v2.0.4"),
     "metadata_relative_path": "cldf/Wordlist-metadata.json",
     "conjugation_df_path": 'latin_data_df.csv',
     "cells_distillation": ["INF", "PRS-IND.1SG", "PRS-IND.2SG", "PRS-IND.3SG", "PRS-IND.3PL"]}, #"IMPERF-IND.3SG",  "PRS-SBJV.3SG"
    "estonian":
    {"archive_url": "https://zenodo.org/records/8392744/files/v1.0.1.zip",
     "archive_path": "estonian-v.1.0.1.zip",
     "file_path": os.path.join(DATA_PATH, "estonian-v.1.0.1"),
     "conjugation_df_path": "estonian_data_df.csv",
     "cells_present": ["ind.prs.1sg", "ind.prs.2sg", "ind.prs.3sg", "ind.prs.1pl", "ind.prs.2pl", "ind.prs.3pl"],
     "cells_distillation": ["sup", "inf", "ind.prs.1sg", "ind.pst.ipfv.1sg", "ind.pst.ipfv.3sg","imp.prs.pers", "ptcp.prs.pers", "ptcp.pst.pers", "ind.prs.impers", "ptcp.pst.impers"]},
    "portuguese":
    {"archive_url": "https://zenodo.org/records/8392722/files/v2.0.1.zip",
     "archive_path": "portuguese-v.2.0.1.zip",
     "file_path": os.path.join(DATA_PATH, "portuguese-v.2.0.1"),
     "conjugation_df_path": "portuguese_data_df.csv",
     "cells_distillation": ["prs.ind.1sg", "prs.ind.3sg", "prs.ind.1pl", "prs.ind.2pl",
                            "prs.ind.3pl", "pst.impf.ind.3sg", "pst.pfv.ind.1sg", "pst.perf.ind.3sg",
                            "fut.ind.3sg", "prs.sbjv.3sg", "prs.sbjv.2pl", "pst.ptcp"]},
    # "arabic":
    # {"archive_url": "https://zenodo.org/records/10100678/files/aravelex-1.0.zip",
    #  "archive_path": "arabic-v.1.0.zip",
    #  "file_path": os.path.join(DATA_PATH, "arabic-v.1.0"),
    #  "conjugation_df_path": "arabic_conjugation_df.csv"},
}

###
