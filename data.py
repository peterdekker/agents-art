import numpy as np
from pycldf.dataset import Dataset
import os
import requests
import shutil
import pandas as pd
from lingpy import ipa2tokens
# from bpe import Encoder

from conf import paths, WRITE_CSV

# np.random.seed(11)


############ Methods Romance dataset ##########


def download_if_needed(paths_lang, label):
    if not os.path.exists(paths_lang["file_path"]):
        # Create parent dirs
        # p = pathlib.Path(data_path)
        # p.parent.mkdir(parents=True, exist_ok=True)
        with open(paths_lang["archive_path"], 'wb') as f:
            print(f'Downloading {label} from {paths_lang["archive_url"]}')
            try:
                r = requests.get(
                    paths_lang["archive_url"], allow_redirects=True)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                raise SystemExit(e)
            # Write downloaded content to file
            f.write(r.content)
            # if not paths_lang["archive_path"].endswith(".zip"):
            #     raise ValueError("Archive path does not end in .zip.")
            print("Unpacking archive.")
            os.makedirs(paths_lang["file_path"], exist_ok=True)
            shutil.unpack_archive(
                paths_lang["archive_path"], extract_dir=paths_lang["file_path"])


def load_romance_dataset(conjugation_df_path, only_latin):
    download_if_needed(paths["latin"], "Romance Verbal Inflection Dataset")
    print("Loading data...")
    dataset = Dataset.from_metadata(os.path.join(
        paths["latin"]["file_path"], paths["latin"]["metadata_relative_path"]))
    forms_df = pd.DataFrame(dataset["FormTable"])
    cognates_df = pd.DataFrame(dataset["CognatesetTable"])
    # lects_df = pd.DataFrame(dataset["LanguageTable"])

    # Filter data
    forms_df_1cognate = filter_romance_empty_multicog(forms_df)
    # Filter on Latin inflection classes + merge forms and cognates table
    conjugation_df = merge_filter_romance_inflections(
        forms_df_1cognate, cognates_df)

    conjugation_df["Cell"] = conjugation_df["Cell"].apply(
        lambda tense_person_list: ".".join(tense_person_list))
    
    # Tokenize forms using Lingpy
    conjugation_df["Form_tokenized"] = conjugation_df["Form"].apply(lambda f: " ".join(ipa2tokens(f, merge_vowels=False, merge_geminates=False)))

    if only_latin:
        conjugation_df = conjugation_df[conjugation_df["Language_ID"]
                                        == "Italic_Latino-Faliscan_Latin"]

    # Write table to file, so it can be read by our script later
    conjugation_df.to_csv(conjugation_df_path)
    print("Done loading data from archive, wrote to csv.")


def load_paralex_dataset(language, conjugation_df_path):
    download_if_needed(paths[language], language)
    file_path = paths[language]["file_path"]
    language_filename = language #"std_modern_arabic" if language == "arabic" else language
    paradigms_df = pd.read_csv(os.path.join(
        file_path, f"{language_filename}_paradigms.csv"), low_memory=False)
    lexemes_df = pd.read_csv(os.path.join(
        file_path, f"{language_filename}_lexemes.csv"), low_memory=False)
    paradigms_lexemes_merged = paradigms_df.merge(
        right=lexemes_df, left_on="lexeme", right_on="lexeme_id")

    # Filter on only verbs (dataset also contains other POS)
    df_verbs = paradigms_lexemes_merged[paradigms_lexemes_merged["POS"] == "verb"]

    # Write table to file, so it can be read by our script later
    df_verbs.to_csv(conjugation_df_path)
    print("Done loading data from archive, wrote to csv.")


def filter_romance_empty_multicog(forms_df):
    # Filter out empty entries and entry with more than one cognate class
    forms_df_nonempty = forms_df[~forms_df["Form"].isin(["Ø", "?"])]
    forms_df_1cognate = forms_df_nonempty[forms_df_nonempty["Cognateset_ID"].apply(
        len) == 1].copy()
    forms_df_1cognate["Cognateset_ID_first"] = forms_df_1cognate["Cognateset_ID"].apply(
        lambda x: x[0])
    return forms_df_1cognate


def merge_filter_romance_inflections(forms_df_1cognate, cognates_df):
    # Filter and keep only entries that have Latin inflection class
    forms_df_merge = forms_df_1cognate.merge(
        right=cognates_df, left_on="Cognateset_ID_first", right_on="ID")
    latin_conjugation_df = forms_df_merge[~forms_df_merge["Latin_Conjugation"].isnull(
    )]
    return latin_conjugation_df



def get_existing_sound_Ngrams(forms_tokenized, Ngrams):
    # sound_inventory = list(set(list("".join(forms))))
    Ngram_list = []
    for form in forms_tokenized:
        for token_ix in range(0, len(form)-(Ngrams-1)):
            # tuple can be deduplicated using set
            Ngram = tuple(form[token_ix:token_ix+Ngrams])
            #if Ngram not in Ngram_list:
            Ngram_list.append(Ngram)
    Ngram_list = list(set(Ngram_list))
    return Ngram_list


# empty_symbol=True, pool_verb_features=False
def create_onehot_forms_from_Ngrams(forms_list, Ngrams, tokenize_form_spaces):
    if tokenize_form_spaces:
        forms_tokenized = [f.split(" ") for f in forms_list]
    else:
        # If not space-tokenized, tokenize by character: form string becomes list of characters, so format is interoperable
        forms_tokenized = [list(f) for f in forms_list]
    n_forms = len(forms_tokenized)
    assert n_forms == len(forms_list)

    Ngram_list = get_existing_sound_Ngrams(forms_tokenized, Ngrams)
    n_Ngrams = len(Ngram_list)
    Ngram_list_indexes = {ngram: idx for idx, ngram in enumerate(Ngram_list)}
    array = np.zeros(shape=(n_forms, n_Ngrams))

    for form_row, form in enumerate(forms_tokenized):
        for token_ix in range(0, len(form)-(Ngrams-1)):
            # convert to tuple, because Ngram_list contains tuples
            current_Ngram = tuple(form[token_ix:token_ix+Ngrams])
            index = Ngram_list_indexes[current_Ngram]
            # index = Ngram_list.index(current_Ngram)
            array[form_row, index] = 1
    return array, Ngram_list


def create_language_dataset(df_language, language, data_format, use_only_present, Ngrams, sample_first,  use_only_3PL, squeeze_into_verbs, concat_verb_features, set_common_features_to_zero, remove_features_allzero):
    if data_format == "paralex":
        form_column = "phon_form"
        inflection_column = "inflection_class"
        lexeme_column = "lexeme_id"
        cell_column = "cell"
        if language == "portuguese":
            tag_present = "prs.ind"
        else:
            tag_present = "ind.prs"
        tag_present_3pl = f"{tag_present}.3pl"
        tokenize_form_spaces = True
    else:  # data_format=="romance"
        form_column = "Form_tokenized" # form_column = "Form"
        inflection_column = "Latin_Conjugation"
        lexeme_column = "Cognateset_ID_first"
        cell_column = "Cell"
        tag_present = "PRS-IND"
        tag_present_3pl = f"{tag_present}.3PL"  # "'PRS-IND', '3PL'"
        tokenize_form_spaces = True
    
    # Keep only first form for a cell (Estonian has doubles)
    df_language = df_language.groupby([lexeme_column,cell_column], as_index=False).first()

    if sample_first:
        df_language = df_language.head(sample_first)

    if use_only_present:
        df_used = df_language[df_language[cell_column].str.contains(
            tag_present_3pl if use_only_3PL else tag_present)]
        if WRITE_CSV:
            df_used.to_csv(f'only_used_{language}_stuff.csv')
    else:
        df_used = df_language
    
    # Reset index, so it starts from 0, and we can use these indices to distinguish items
    df_used = df_used.reset_index()

    # All these variables have length and follow order of df_used
    forms = df_used[form_column]
    forms_list = list(forms)
    inflections = df_used[inflection_column]
    lexemes = df_used[lexeme_column]
    cells = df_used[cell_column]
    # if data_format == "paralex":
    #     cells = df_used[cell_column].str[-3:]
    # else:  # romance
    #     cells = df_used[cell_column].str[-3] #-5:-2]
    assert len(forms) == len(forms_list) == len(inflections) == len(
        lexemes) == len(cells) == len(df_used)

    # Variables representing unique set (not length df_used)
    unique_cells_ordered = cells.unique()
    lexemes_unique = lexemes.unique()
    lexemes_unique.sort()
    n_lexemes_unique = len(lexemes_unique)
    # Inflection classes calculated based on only used dataset (possibly not full dataset)
    inflection_classes = list(inflections.unique())

    forms_encoded, ngram_inventory = create_onehot_forms_from_Ngrams(
        forms_list, Ngrams, tokenize_form_spaces)
    assert forms_encoded.shape[0] == len(df_used)
    assert forms_encoded.shape[1] == len(ngram_inventory)
    orig_ngram_inventory = ngram_inventory


    if squeeze_into_verbs:
        # Make new pooled ngram inventory with person tags
        if concat_verb_features:

            # Pooled ngram inventory only used for plotting, not in encoding processing
            pooled_ngram_inventory = ["".join(
                ngram) + '_'+unique_cell for unique_cell in unique_cells_ordered for ngram in ngram_inventory]
            ngram_inventory = pooled_ngram_inventory

            pooled_forms_encoded = np.empty(
                (n_lexemes_unique, len(unique_cells_ordered)*forms_encoded.shape[1]))
        else:  # Set representation
            pooled_forms_encoded = np.empty(
                (n_lexemes_unique, forms_encoded.shape[1]))
        pooled_inflections = []
        #for lexeme_ix, lexeme_unique in enumerate(lexemes_unique):
        for lexeme_ix, lexeme_unique in enumerate(lexemes_unique):
            rows_lexeme_unique = df_used[df_used[lexeme_column]==lexeme_unique]
            if concat_verb_features:
                pooled_forms_encoded_for_verb = [[]]  # add dimension
            else:
                pooled_forms_encoded_for_verb = np.zeros(
                    (1, len(ngram_inventory)))
            indices_for_verb = list(rows_lexeme_unique.index) #np.where(lexemes == lexeme_unique)[0]

            # Save inflection for this pooled verb, from the first position
            pooled_inflections.append(
                rows_lexeme_unique[inflection_column].values[0]) # inflections.values[indices_for_verb[0]]
            # Not necessarily in the same order as wanted
            cells_for_verb = list(rows_lexeme_unique[cell_column]) # cells_np[indices_for_verb]
            for unique_cell in unique_cells_ordered:
                if unique_cell in cells_for_verb:
                    # index = np.where(
                    #     unique_cell == cells_for_verb)
                    index = cells_for_verb.index(unique_cell)
                    index_in_forms_encoded = indices_for_verb[index] # indices_for_verb[index[0][0]]
                    if concat_verb_features:
                        pooled_forms_encoded_for_verb = np.append(
                            pooled_forms_encoded_for_verb, forms_encoded[None, index_in_forms_encoded], axis=1)
                    else:
                        pooled_forms_encoded_for_verb = pooled_forms_encoded_for_verb + \
                            forms_encoded[index_in_forms_encoded]
                else:
                    if concat_verb_features:
                        pooled_forms_encoded_for_verb = np.append(
                            pooled_forms_encoded_for_verb, np.zeros((forms_encoded.shape[1])), axis=0)

            if set_common_features_to_zero:
                print("This should not be executed when set_common_features_to_zero is not on.")
                # NOTE: This functionality assumes number of paradigm cells is 6: only works for Latin
                # NOTE: Not tested anymore after changing data processing code
                if concat_verb_features:
                    temp = np.reshape(pooled_forms_encoded_for_verb,
                                      (len(unique_cells_ordered), forms_encoded.shape[1]))
                    S = sum(temp)
                    # If the same gram was activated in all person tenses, their sum here is 6
                    temp[:, S == 6] = 0
                    pooled_forms_encoded_for_verb = np.reshape(
                        temp, (len(unique_cells_ordered)*forms_encoded.shape[1]))
                else:
                    pooled_forms_encoded_for_verb[pooled_forms_encoded_for_verb == 6] = 0

            # keeps only one activated n-gram, even though it may occur in several forms (for set representation, does nothing for concat)
            if not concat_verb_features:
                pooled_forms_encoded_for_verb = np.clip(
                    pooled_forms_encoded_for_verb, 0, 1)
            pooled_forms_encoded[lexeme_ix,:] = pooled_forms_encoded_for_verb
        inflections = pooled_inflections
        forms_encoded = pooled_forms_encoded
    
    # Remove features for which whole column is 0:
    # mostly relevant in concat mode, where some ngrams do not occur for some cells
    # in set mode (or if not squeezing into verbs) there should not be all zeros features
    if remove_features_allzero: 
        features_all_zero = np.all(forms_encoded == 0, axis=0)
        # print(features_all_zero.sum())

    # print_diagnostic_encoding(form_column, lexeme_column,
    #                           df_used, lexemes_unique, forms_encoded, ngram_inventory)
    print(f"{language}. unique lexemes: {n_lexemes_unique}. Inflection classes: {len(inflection_classes)}. Ngrams: {len(orig_ngram_inventory)}. Cells: {len(unique_cells_ordered)}. Features: {forms_encoded.shape[1]}")

    return forms_encoded, forms_list, list(inflections), inflection_classes, list(lexemes), ngram_inventory


def print_diagnostic_encoding(form_column, lexeme_column, df_used, lexemes_unique, forms_encoded, ngram_inventory):
    encoding_diagnostic_df = pd.DataFrame(
        data=forms_encoded, index=lexemes_unique, columns=ngram_inventory)
    for lexeme, lexeme_row in encoding_diagnostic_df.iterrows():
        activated_ngrams = ["".join(ngram)
                            for ngram in lexeme_row[lexeme_row == 1].index]
        used_forms = list(
            df_used[df_used[lexeme_column] == lexeme][form_column])
        print(f"{lexeme}: {activated_ngrams}")
        print(f"Forms used for encoding: {used_forms}\n")

# def get_sound_inventory(forms):
#     sound_inventory = list(set(list("".join(forms))))
#     max_form_len = max([len(x) for x in forms])
#     return sound_inventory, max_form_len

# def create_onehot_forms(forms, empty_symbol=True):
#     sounds, max_form_len = get_sound_inventory(forms)
#     n_forms = len(forms)
#     if empty_symbol:
#         sounds.append(".")
#     n_sounds = len(sounds)
#     # print(sounds)
#     # print(f"n_forms: {n_forms}")
#     # print(f"n_sounds: {n_sounds}")
#     # print(f"max_form_len: {max_form_len}")
#     array = np.zeros(shape=(n_forms, n_sounds * max_form_len))
#     for form_row, form in enumerate(forms):
#         # print(form)
#         form_len = len(form)
#         for char_position in range(max_form_len):
#             if char_position < form_len:
#                 char = form[char_position]
#                 char_hot_index = sounds.index(char)
#                 array[form_row, char_position*n_sounds+char_hot_index] = 1
#                 # print(f"Char position {char_position} within form: {char}. Hot index: {char_hot_index}. Index: {form_row, char_position*n_sounds+char_hot_index}")
#             else:  # Char_position >= form_len; so word shorter than the longest word
#                 if empty_symbol:
#                     # Create symbol for these positions
#                     empty_hot_index = sounds.index(".")
#                     array[form_row, char_position*n_sounds+empty_hot_index] = 1
#                     # print(f"Char position {char_position} OUTSIDE form. Hot index: {empty_hot_index}. Index: {form_row, char_position*n_sounds+empty_hot_index}")
#                 # Else: just leave the 0000s for empty symbol
#     return array, sounds

# def create_onehot_inflections(inflections):
#     n_inflections = len(inflections)
#     inflection_inventory = list(set(inflections))
#     inflection_inventory_size = len(inflection_inventory)
#     array = np.zeros(shape=(n_inflections, inflection_inventory_size))
#     for infl_row, inflection in enumerate(inflections):
#         hot_index = inflection_inventory.index(inflection)
#         array[infl_row, hot_index] = 1
#     return array, inflection_inventory

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
