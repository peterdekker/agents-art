import numpy as np
import io
import time
from pycldf.dataset import Dataset
import os
import pathlib
import requests
import shutil
import pandas as pd
# from bpe import Encoder

from conf import INFLECTION_CLASSES

# np.random.seed(11)
currentdir = os.path.abspath("")
DATA_PATH = os.path.join(currentdir, "data")

# ROMANCE_ARCHIVE_PATH = "Romance_Verbal_Inflection_Dataset-v2.0.4.tar.gz"
# ROMANCE_ARCHIVE_URL = "https://zenodo.org/records/4039059/files/v2.0.4.zip"
# # Directory after unpacking archive
# ROMANCE_FILE_PATH = os.path.join(
#     DATA_PATH, "Romance_Verbal_Inflection_Dataset-v2.0.4")
# ROMANCE_METADATA_PATH = os.path.join(
#     ROMANCE_PATH, "cldf/Wordlist-metadata.json")

# PORTUGUESE_ARCHIVE_URL = "https://zenodo.org/records/8392722/files/v2.0.1.zip"
# ESTONIAN_ARCHIVE_URL = "https://zenodo.org/records/8392744/files/v1.0.1.zip"
# ARABIC_ARCHIVE_URL = "https://zenodo.org/records/10100678/files/aravelex-1.0.zip"


paths = {
    "latin":
    {"archive_url": "https://zenodo.org/records/4039059/files/v2.0.4.zip",
     "archive_path": "Romance_Verbal_Inflection_Dataset-v2.0.4.zip",
     "file_path": os.path.join(DATA_PATH, "Romance_Verbal_Inflection_Dataset-v2.0.4"),
     "metadata_relative_path": "cldf/Wordlist-metadata.json"},
     "estonian":
    {"archive_url": "https://zenodo.org/records/8392744/files/v1.0.1.zip"},
    "portuguese":
    {"archive_url": "https://zenodo.org/records/8392722/files/v2.0.1.zip"},
    "arabic":
    {"archive_url": "https://zenodo.org/records/10100678/files/aravelex-1.0.zip"},
}
############ Methods Romance dataset ##########


def download_if_needed(paths_lang, label):
    if not os.path.exists(paths_lang["file_path"]):
        # Create parent dirs
        # p = pathlib.Path(data_path)
        # p.parent.mkdir(parents=True, exist_ok=True)
        with open(paths_lang["archive_path"], 'wb') as f:
            print(f'Downloading {label} from {paths_lang["archive_url"]}')
            try:
                r = requests.get(paths_lang["archive_url"], allow_redirects=True)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                raise SystemExit(e)
            # Write downloaded content to file
            f.write(r.content)
            # if not paths_lang["archive_path"].endswith(".zip"):
            #     raise ValueError("Archive path does not end in .zip.")
            print("Unpacking archive.")
            os.makedirs(paths_lang["file_path"], exist_ok=True)
            shutil.unpack_archive(paths_lang["archive_path"], extract_dir=paths_lang["file_path"])


def load_romance_dataset():
    download_if_needed(paths["latin"], "Romance Verbal Inflection Dataset")
    print("Loading data...")
    dataset = Dataset.from_metadata(os.path.join(paths["latin"]["file_path"], paths["latin"]["metadata_relative_path"]))
    forms_df = pd.DataFrame(dataset["FormTable"])
    cognates_df = pd.DataFrame(dataset["CognatesetTable"])
    lects_df = pd.DataFrame(dataset["LanguageTable"])
    print("Loaded data.")
    return forms_df, cognates_df, lects_df


def filter_romance_empty_multicog(forms_df):
    # Filter out empty entries and entry with more than one cognate class
    forms_df_nonempty = forms_df[~forms_df["Form"].isin(["Ø", "?"])]
    forms_df_1cognate = forms_df_nonempty[forms_df_nonempty["Cognateset_ID"].apply(
        len) == 1].copy()
    forms_df_1cognate["Cognateset_ID_first"] = forms_df_1cognate["Cognateset_ID"].apply(
        lambda x: x[0])
    return forms_df_1cognate


def filter_romance_inflections(forms_df_1cognate, cognates_df):
    # Filter and keep only entries that have Latin inflection class
    forms_df_merge = forms_df_1cognate.merge(
        right=cognates_df, left_on="Cognateset_ID_first", right_on="ID")
    latin_conjugation_df = forms_df_merge[~forms_df_merge["Latin_Conjugation"].isnull(
    )]
    return latin_conjugation_df


# Diacritics are counted as separate character


def get_sound_inventory(forms):
    sound_inventory = list(set(list("".join(forms))))
    max_form_len = max([len(x) for x in forms])
    return sound_inventory, max_form_len

def get_existing_sound_Ngrams(forms, Ngrams):  # NOT FINISHED!!
    sound_inventory = list(set(list("".join(forms))))
    Ngram_list=[]
    form_values=forms.values

    for i1 in range(0, len(form_values)):
        for i2 in range(0, len(form_values[i1])-(Ngrams-1)):
            Ngram=form_values[i1][i2:i2+Ngrams]
            if Ngram not in Ngram_list:
                Ngram_list.append(Ngram)

    return Ngram_list


def create_onehot_inflections(inflections):
    n_inflections = len(inflections)
    inflection_inventory = list(set(inflections))
    inflection_inventory_size = len(inflection_inventory)
    array = np.zeros(shape=(n_inflections, inflection_inventory_size))
    for infl_row, inflection in enumerate(inflections):
        hot_index = inflection_inventory.index(inflection)
        array[infl_row, hot_index] = 1
    return array, inflection_inventory

def create_onehot_forms_from_Ngrams(forms, Ngrams, empty_symbol=True, pool_verb_features=False):
    Ngram_list = get_existing_sound_Ngrams(forms, Ngrams)
    n_forms = len(forms)

    n_Ngrams = len(Ngram_list)
    # print(sounds)
    # print(f"n_forms: {n_forms}")
    # print(f"n_sounds: {n_sounds}")
    # print(f"max_form_len: {max_form_len}")
    array = np.zeros(shape=(n_forms, n_Ngrams))

    for form_row, form in enumerate(forms):
        for char_position in range(0, len(form)-(Ngrams-1)):
            current_Ngram=form[char_position:char_position+Ngrams]
            index=Ngram_list.index(current_Ngram)
            array[form_row, index] = 1

    return array, Ngram_list


def create_onehot_forms(forms, empty_symbol=True):
    sounds, max_form_len = get_sound_inventory(forms)
    n_forms = len(forms)
    if empty_symbol:
        sounds.append(".")
    n_sounds = len(sounds)
    # print(sounds)
    # print(f"n_forms: {n_forms}")
    # print(f"n_sounds: {n_sounds}")
    # print(f"max_form_len: {max_form_len}")
    array = np.zeros(shape=(n_forms, n_sounds * max_form_len))
    for form_row, form in enumerate(forms):
        # print(form)
        form_len = len(form)
        for char_position in range(max_form_len):
            if char_position < form_len:
                char = form[char_position]
                char_hot_index = sounds.index(char)
                array[form_row, char_position*n_sounds+char_hot_index] = 1
                # print(f"Char position {char_position} within form: {char}. Hot index: {char_hot_index}. Index: {form_row, char_position*n_sounds+char_hot_index}")
            else:  # Char_position >= form_len; so word shorter than the longest word
                if empty_symbol:
                    # Create symbol for these positions
                    empty_hot_index = sounds.index(".")
                    array[form_row, char_position*n_sounds+empty_hot_index] = 1
                    # print(f"Char position {char_position} OUTSIDE form. Hot index: {empty_hot_index}. Index: {form_row, char_position*n_sounds+empty_hot_index}")
                # Else: just leave the 0000s for empty symbol
    return array, sounds

# def create_bytepair_forms(forms):
#     encoder = Encoder(200, pct_bpe=0.88)
#     print(forms)
#     encoder.fit(forms)
#     print([encoder.tokenize(form) for form in forms])
#     print(list(encoder.transform(forms)))
#     # TODO: finish bytepair encoding. Possibly train on all languages
#     # "{0:b}".format() -> map(int)
#     # and set max bitstring length by max value
#     # and try to get max word lengths almost equal by setting parameters right: Set parameters right to get quite even word lengths: https://arxiv.org/abs/1508.07909


def create_language_dataset(df, language, Ngrams=2, empty_symbol=True, language_column="Language_ID", form_column="Form", inflection_column="Latin_Conjugation", cogid_column="Cognateset_ID_first", sample_first=None, use_only_present=True, use_only_3PL=False, squeeze_into_verbs=True, concat_verb_features=True, set_common_features_to_zero=False):
    df_language = df[df[language_column] ==
                     language]
    if sample_first:
        df_language = df_language.head(sample_first)

    if use_only_present:
        if use_only_3PL:
            df_used=df_language[df_language['Cell'].str.contains(
                "'PRS-IND', '3PL'")]
        else:
            df_used=df_language[df_language['Cell'].str.contains("'PRS-IND'")]

            df_used.to_csv('only_used_Latin_stuff.csv')
    forms = df_used[form_column]
    inflections = df_used[inflection_column]
    cogids = df_used[cogid_column]
    person_tags=df_used.Cell.str[-5:-2]
    unique_person_tags=person_tags.unique()
    # array(['1SG', '2SG', '3SG', ..., '1PL', '2PL', '3PL']
    person_tags=person_tags.values
    unique_verbs=cogids.unique()

    forms_encoded, bigram_inventory = create_onehot_forms_from_Ngrams(
        forms, Ngrams, empty_symbol, concat_verb_features)

    if squeeze_into_verbs:
        if concat_verb_features:
            # Make new pooled bigram inventory with person tags
            pooled_bigram_inventory=[]
            for p in range(len(unique_person_tags)):
                pooled_bigram_inventory=np.append(pooled_bigram_inventory, ([
                                                  i + '_'+unique_person_tags[p] for i in bigram_inventory]), axis=0)

            pooled_forms_encoded=np.empty(
                (0, len(unique_person_tags)*forms_encoded.shape[1]))
            pooled_inflections=[]
            # array(['1SG', '2SG', '3SG', '1PL', '2PL', '3PL'],
            pool_order=unique_person_tags
            for i in range(0, len(unique_verbs)):

                pooled_forms_encoded_for_verb=[]
                indices_for_verb =np.where(cogids == unique_verbs[i])[0]
                # Save inflection for this pooled verb, from the first position
                pooled_inflections.append(
                    inflections.values[indices_for_verb[0]])
                # Not necessarily in the same order as wanted
                person_tags_for_verb=person_tags[indices_for_verb]
                for p in range(len(pool_order)):

                    if pool_order[p] in person_tags_for_verb:
                        index=np.where(pool_order[p] == person_tags_for_verb)
                        index_in_forms_encoded=indices_for_verb[index[0][0]]
                        pooled_forms_encoded_for_verb=np.append(
                            pooled_forms_encoded_for_verb, forms_encoded[index_in_forms_encoded], axis=0)
                    else:
                        pooled_forms_encoded_for_verb=np.append(
                            pooled_forms_encoded_for_verb, np.zeros((forms_encoded.shape[1])), axis=0)

                if set_common_features_to_zero:
                    temp=np.reshape(pooled_forms_encoded_for_verb,
                                    (len(pool_order), forms_encoded.shape[1]))
                    S=sum(temp)
                    # If the same gram was activated in all person tenses, their sum here is 6
                    temp[:, S == 6] =0
                    pooled_forms_encoded_for_verb=np.reshape(
                        temp, (len(pool_order)*forms_encoded.shape[1]))

                pooled_forms_encoded=np.append(
                    pooled_forms_encoded, pooled_forms_encoded_for_verb[None, :], axis=0)
            inflections = pooled_inflections

            forms_encoded=pooled_forms_encoded
            bigram_inventory=pooled_bigram_inventory
        # Squeeze all persons into a single bigram vector, without concatenating (One hot of existing bigrams over all person tenses)
        else:
               # Make new pooled bigram inventory with person tags

            pooled_forms_encoded=np.empty((0, forms_encoded.shape[1]))
            pooled_inflections=[]
            # array(['1SG', '2SG', '3SG', '1PL', '2PL', '3PL'],
            pool_order=unique_person_tags
            for i in range(0, len(unique_verbs)):

                pooled_forms_encoded_for_verb=np.zeros(
                    (1, len(bigram_inventory)))
                indices_for_verb =np.where(cogids == unique_verbs[i])[0]
                # Save inflection for this pooled verb, from the first position
                pooled_inflections.append(
                    inflections.values[indices_for_verb[0]])
                # Not necessarily in the same order as wanted
                person_tags_for_verb=person_tags[indices_for_verb]
                for p in range(len(pool_order)):

                    if pool_order[p] in person_tags_for_verb:
                        index=np.where(pool_order[p] == person_tags_for_verb)
                        index_in_forms_encoded=indices_for_verb[index[0][0]]
                        pooled_forms_encoded_for_verb=pooled_forms_encoded_for_verb + \
                            forms_encoded[index_in_forms_encoded]

                if set_common_features_to_zero:
                    # If the same gram was activated in all person tenses, their sum here is 6
                    pooled_forms_encoded_for_verb[pooled_forms_encoded_for_verb == 6] =0

                pooled_forms_encoded_for_verb=np.clip(
                    pooled_forms_encoded_for_verb, 0, 1)
                pooled_forms_encoded=np.append(
                    pooled_forms_encoded, pooled_forms_encoded_for_verb, axis=0)
            inflections = pooled_inflections

            forms_encoded=pooled_forms_encoded


    inflections_onehot, inflection_inventory = create_onehot_inflections(
        inflections)
    return forms_encoded, inflections_onehot, list(forms), list(inflections), list(cogids), bigram_inventory

############### Methods binarized word embedings dataset #########


# def map_int(string_list):
#     return [int(s) for s in string_list]


# def flatten(t):
#     return [item for sublist in t for item in sublist]


# def load_vectors(fname, read_first_words=10000, sample_words=1000):
#     # t0 = time.time()
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, dim = map(int, fin.readline().split())
#     file_data = []
#     words = []
#     sample_indices = np.random.choice(
#         np.arange(read_first_words), sample_words, replace=False)
#     for i, line in enumerate(fin):
#         if i >= read_first_words:
#             break
#         if i in sample_indices:  # Maybe inefficient if sample_words is high
#             tokens = line.rstrip().split(' ')
#             word = tokens[0]
#             words.append(word)
#             bin_values = [
#                 f"{int(decimal_value):064b}" for decimal_value in tokens[1:]]
#             file_data.append(bin_values)
#     data_flatten = "".join(flatten(file_data))
#     data_int = map_int(list(data_flatten))
#     assert len(data_int) == dim*sample_words
#     data_array = np.array(data_int).reshape(sample_words, dim)
#     # Split array into n_words. This also joins the groups of 4 longs that represent one vector together
#     # t1 = time.time()
#     # print(t1-t0)
#     return data_array, words, dim

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
