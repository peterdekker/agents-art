LANGUAGE = "Italic_Latino-Faliscan_Latin"

LABEL_DENSITY=5
N_CLUSTERS=10

SAMPLE_FIRST = None # 1000
BYTEPAIR_ENCODING = False # TODO: still under construction
N_RUNS = 10

ART_VIGILANCE=0.8
ART_LEARNING_RATE=0.1

INFLECTION_CLASSES = ["I", "II", "III", "IV", "special"]
N_INFLECTION_CLASSES = len(INFLECTION_CLASSES)

FILLED_MARKERS = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']*16
EMPTY_SYMBOL = True

OUTPUT_DIR = "output"