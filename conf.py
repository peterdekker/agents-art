import os
import numpy as np
import pandas as pd
# import matplotlib
import seaborn as sns
import colorcet as cc

pd.set_option('display.max_rows', None)

sns.set_theme(font="Charis SIL Compact", font_scale=1.4)

cmap_categorical = sns.color_palette(cc.glasbey, n_colors=25, as_cmap=True)

### Command line options. Defaults (when no argument is given) can be set via the uppercase variables.
LANGUAGE = "latin"
VIGILANCE_SINGLE_RUN = 0.1
N_RUNS = 10  # User has to set to 1 via command line for train_test mode

params = {
    "language": {"default": LANGUAGE, "type": str},
    "vigilance_single_run": {"default": VIGILANCE_SINGLE_RUN, "type": float},
    "n_runs": {"default": N_RUNS, "type": int},
}



mode_params = ["plot_data_before", "single_run", "eval_batches",
               "eval_vigilances", "eval_intervals", "baseline", "train_test"]

###

# Parameters settable only via this config file
MULTIPROCESSING = True
N_PROCESSES = None # Default: run processes on all processor cores (None), or give a specific number of processes
FEATURES_SET = False # Default: use a concat representation based on all paradigm cells (False). When this variable is True, a set of the features in all paradigm cells is used, leading to a shorter representation.
NGRAMS = 3 # By default, 3-grams are used. Using this variable, the number n of the n-gram (number of phonemes in the n-gram) can be set.
SOUNDCLASSES = "none"  # By default, the phonemes of the wordforms are used ('none'). A soundclass representation can be actived using "asjp" or "sca"
USE_GPU = False  # GPU often slower than CPU, and runs out of memory for concat representation. but saves CPU availability
WRITE_CSV = True
WRITE_TEX = True


ART_LEARNING_RATE = 2

VIGILANCE_RANGE_STEP = 0.005  # 0.02
MAX_VIGILANCE = 0.3 + VIGILANCE_RANGE_STEP

OUTPUT_DIR = "output"

REMOVE_FEATURES_ALLZERO = True
SET_COMMON_FEATURES_TO_ZERO = False
SAMPLE_FIRST = None  # 1000
###


# Not to be changed by user
MAX_CLUSTERS_PORTUGUESE = None  # disabled
MIN_DATAPOINTS_CLASS_PORTUGUESE = 10

VIGILANCE_RANGE = [
    x/1000 for x in range(0, int(MAX_VIGILANCE*1000), int(VIGILANCE_RANGE_STEP*1000))]
LABEL_DENSITY = 5
EVAL_INTERVAL = 20
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
     "cells_distillation": ["IMPERF-IND.3SG", "INF", "IMP.2SG", "PRS-IND.1SG", "PRS-IND.2SG", "PRS-IND.3SG", "PRS-IND.3PL", "PRS-SBJV.3SG", "GER"]},
    "estonian":
    {"archive_url": "https://zenodo.org/records/10692800/files/eesthetic-v1.0.3.zip",
     "archive_path": "estonian-v.1.0.3.zip",
     "file_path": os.path.join(DATA_PATH, "estonian-v.1.0.3"),
     "conjugation_df_path": "estonian_data_df.csv",
     "cells_distillation": ["inf", "imp.prs.2pl", "imp.prs.pers", "ger", "ptcp.pst.pers",
                            "ind.prs.1sg", "cond.prs.pers", "imp.prs.2sg",
                            "sup", "ptcp.prs.pers", "quot.prs.pers", "ind.pst.ipfv.1sg",
                            "ind.prs.impers", "ind.pst.ipfv.impers"
                            ]},
    "portuguese":
    {"archive_url": "https://zenodo.org/records/8392722/files/v2.0.1.zip",
     "archive_path": "portuguese-v.2.0.1.zip",
     "file_path": os.path.join(DATA_PATH, "portuguese-v.2.0.1"),
     "conjugation_df_path": "portuguese_data_df.csv",
     "cells_distillation": ["prs.ind.1sg", "prs.ind.3sg", "prs.ind.1pl", "prs.ind.2pl",
                            "prs.ind.3pl", "pst.impf.ind.3sg", "pst.pfv.ind.1sg", "pst.perf.ind.3sg",
                            "fut.ind.3sg", "prs.sbjv.3sg", "prs.sbjv.2pl", "pst.ptcp"]},
}
