LANGUAGE = "Italic_Latino-Faliscan_Latin"

LATIN_CONJUGATION_DF_FILE = 'latin_conjugation_df.csv'

SAMPLE_FIRST = None # 1000
BYTEPAIR_ENCODING = False # TODO: still under construction
N_RUNS = 2

ART_VIGILANCE=0.2
ART_LEARNING_RATE=2
LABEL_DENSITY = 5

INFLECTION_CLASSES = ["I", "II", "III", "IV", "special"]
N_INFLECTION_CLASSES = len(INFLECTION_CLASSES)
MAX_CLUSTERS = 20

FILLED_MARKERS = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']*16
EMPTY_SYMBOL = True

OUTPUT_DIR = "output"