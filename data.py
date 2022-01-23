import numpy as np
import io
import time
from pycldf.dataset import Dataset
import os
import pathlib
import requests
import shutil
import pandas as pd

np.random.seed(11)
currentdir = os.path.abspath("")

DATA_ARCHIVE_PATH = "Romance_Verbal_Inflection_Dataset-v2.0.4.tar.gz"
DATA_ARCHIVE_URL = "https://gitlab.com/sbeniamine/Romance_Verbal_Inflection_Dataset/-/archive/v2.0.4/Romance_Verbal_Inflection_Dataset-v2.0.4.tar.gz"
DATA_PATH = os.path.join(currentdir, "Romance_Verbal_Inflection_Dataset-v2.0.4") # Directory after unpacking archive
METADATA_PATH = os.path.join(DATA_PATH, "cldf/Wordlist-metadata.json")

def download_if_needed(archive_path, archive_url, file_path, label):
    if not os.path.exists(file_path):
        # Create parent dirs
        #p = pathlib.Path(file_path)
        #p.parent.mkdir(parents=True, exist_ok=True)
        with open(archive_path, 'wb') as f:
            print(f"Downloading {label} from {archive_url}")
            try:
                r = requests.get(archive_url, allow_redirects=True)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                raise SystemExit(e)
            # Write downloaded content to file
            f.write(r.content)
            if archive_path.endswith(".tar.gz"):
                print("Unpacking archive.")
                shutil.unpack_archive(archive_path, currentdir)


def load_romance_dataset():
    download_if_needed(DATA_ARCHIVE_PATH, DATA_ARCHIVE_URL, DATA_PATH, "romance")
    print("Loading data...")
    dataset = Dataset.from_metadata(METADATA_PATH)
    forms_df = pd.DataFrame(dataset["FormTable"])
    cognates_df = pd.DataFrame(dataset["CognatesetTable"])
    lects_df = pd.DataFrame(dataset["LanguageTable"])
    print("Loaded data.")
    return forms_df, cognates_df, lects_df

def map_int(string_list):
    return [int(s) for s in string_list]

def flatten(t):
    return [item for sublist in t for item in sublist]

def load_vectors(fname, read_first_words=10000, sample_words=1000):
    # t0 = time.time()
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, dim = map(int, fin.readline().split())
    file_data = []
    words = []
    sample_indices = np.random.choice(np.arange(read_first_words), sample_words, replace=False)
    for i,line in enumerate(fin):
        if i >= read_first_words:
            break
        if i in sample_indices: # Maybe inefficient if sample_words is high
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            words.append(word)
            bin_values = [f"{int(decimal_value):064b}" for decimal_value in tokens[1:]]
            file_data.append(bin_values)
    data_flatten = "".join(flatten(file_data))
    data_int = map_int(list(data_flatten))
    assert len(data_int)== dim*sample_words
    data_array = np.array(data_int).reshape(sample_words, dim)
    # Split array into n_words. This also joins the groups of 4 longs that represent one vector together
    # t1 = time.time()
    # print(t1-t0)
    return data_array, words, dim

# def load_vectors(fname):
#     t0 = time.time()
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n_words, d = map(int, fin.readline().split())
#     data = []
#     words = []
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         word = tokens[0]
#         words.append(word)
#         bin_values = [map_int(list(f"{int(decimal_value):064b}")) for decimal_value in tokens[1:]]
#         data.append(bin_values)
#     data_array = np.array(data)
#     print(data_array.shape)
#     t1 = time.time()
#     print(t1-t0)
#     return data

# TODO: also load real-valued vectors, to compare