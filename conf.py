import numpy as np

LANGUAGE = "Italic_Latino-Faliscan_Latin"

LATIN_CONJUGATION_DF_FILE = 'latin_conjugation_df.csv'

SAMPLE_FIRST = None # 1000
BYTEPAIR_ENCODING = False # TODO: still under construction
N_RUNS = 10

ART_VIGILANCE=0.25
ART_LEARNING_RATE=2
LABEL_DENSITY = 5
VIGILANCE_RANGE = np.arange(0.0,1.02,0.02)

INFLECTION_CLASSES = ["I", "II", "III", "IV", "special"]
N_INFLECTION_CLASSES = len(INFLECTION_CLASSES)
INITIAL_CLUSTERS = 1

FILLED_MARKERS = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']*16
EMPTY_SYMBOL = True

OUTPUT_DIR = "output"

CONCAT_VERB_FEATURES=False
SET_COMMON_FEATURES_TO_ZERO=False
USE_ONLY_3PL=False
SQUEEZE_INTO_VERBS=True
NGRAMS=3

CONFIG_STRING=f"--use_only_3PL={USE_ONLY_3PL}---squeeze_into_verbs=={SQUEEZE_INTO_VERBS}---Concat_verb_features={CONCAT_VERB_FEATURES}---CommonFeat0={SET_COMMON_FEATURES_TO_ZERO}---Ngram={NGRAMS}"