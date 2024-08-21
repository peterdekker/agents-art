import numpy as np
from pycldf.dataset import Dataset
import os
import requests
import shutil
import pandas as pd
from lingpy import ipa2tokens, tokens2class

from conf import paths, WRITE_CSV


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


def load_romance_dataset(conjugation_df_path):
    download_if_needed(paths["latin"], "Romance Verbal Inflection Dataset")
    print("Loading data...")
    dataset = Dataset.from_metadata(os.path.join(
        paths["latin"]["file_path"], paths["latin"]["metadata_relative_path"]))
    forms_df = pd.DataFrame(dataset["FormTable"])
    cognates_df = pd.DataFrame(dataset["CognatesetTable"])
    # lects_df = pd.DataFrame(dataset["LanguageTable"])

    # Only use Latin
    forms_df = forms_df[forms_df["Language_ID"]
                        == "Italic_Latino-Faliscan_Latin"]

    # Filter data
    forms_df_1cognate = filter_romance_empty_multicog(forms_df)
    # Filter on Latin inflection classes + merge forms and cognates table
    conjugation_df = merge_filter_romance_inflections(
        forms_df_1cognate, cognates_df)

    conjugation_df["Cell"] = conjugation_df["Cell"].apply(
        lambda tense_person_list: ".".join(tense_person_list))

    # Tokenize forms using Lingpy
    conjugation_df["Form_tokenized"] = conjugation_df["Form"].apply(
        lambda f: " ".join(ipa2tokens(f, merge_vowels=False, merge_geminates=False)))

    # for form in conjugation_df["Form_tokenized"]:
    #     form_split = form.split(" ")
    #     print(form_split)
    #     print(tokens2class(form_split, model="sca"))
    conjugation_df["form_sca"] = conjugation_df["Form_tokenized"].apply(
        lambda f: " ".join(tokens2class(f.split(" "), model="sca")))
    conjugation_df["form_dolgo"] = conjugation_df["Form_tokenized"].apply(
        lambda f: " ".join(tokens2class(f.split(" "), model="dolgo")))
    conjugation_df["form_asjp"] = conjugation_df["Form_tokenized"].apply(
        lambda f: " ".join(tokens2class(f.split(" "), model="asjp")))

    # Write table to file, so it can be read by our script later
    conjugation_df.to_csv(conjugation_df_path)
    print("Done loading data from archive, wrote to csv.")


def load_paralex_dataset(language, conjugation_df_path):
    download_if_needed(paths[language], language)
    file_path = paths[language]["file_path"]
    # "std_modern_arabic" if language == "arabic" else language
    language_filename = language
    paradigms_df = pd.read_csv(os.path.join(
        file_path, f"{language_filename}_paradigms.csv"), low_memory=False)
    lexemes_df = pd.read_csv(os.path.join(
        file_path, f"{language_filename}_lexemes.csv"), low_memory=False)
    paradigms_lexemes_merged = paradigms_df.merge(
        right=lexemes_df, left_on="lexeme", right_on="lexeme_id")

    # Filter on only verbs (dataset also contains other POS)
    conjugation_df = paradigms_lexemes_merged[paradigms_lexemes_merged["POS"] == "verb"].copy(
    )

    conjugation_df["form_sca"] = conjugation_df["phon_form"].apply(
        lambda f: " ".join(tokens2class(f.split(" "), model="sca")))
    conjugation_df["form_dolgo"] = conjugation_df["phon_form"].apply(
        lambda f: " ".join(tokens2class(f.split(" "), model="dolgo")))
    conjugation_df["form_asjp"] = conjugation_df["phon_form"].apply(
        lambda f: " ".join(tokens2class(f.split(" "), model="asjp")))

    # Write table to file, so it can be read by our script later
    conjugation_df.to_csv(conjugation_df_path)
    print("Done loading data from archive, wrote to csv.")


def filter_romance_empty_multicog(forms_df):
    # NOTE: For Latin, Ø exists, but ? not and multiple cognate classes per lexeme also not
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
    ngram_inventory = []
    for form in forms_tokenized:
        for token_ix in range(0, len(form)-(Ngrams-1)):
            # tuple can be deduplicated using set
            Ngram = tuple(form[token_ix:token_ix+Ngrams])
            # if Ngram not in Ngram_list:
            ngram_inventory.append(Ngram)
    ngram_inventory = list(set(ngram_inventory))
    return ngram_inventory


# empty_symbol=True, pool_verb_features=False
def create_onehot_forms_from_Ngrams(forms_list, Ngrams, tokenized_form_spaces):
    if tokenized_form_spaces:
        forms_tokenized = [f.split(" ") for f in forms_list]
    else:
        # If not space-tokenized, tokenize by character: form string becomes list of characters, so format is interoperable
        forms_tokenized = [list(f) for f in forms_list]
    n_forms = len(forms_tokenized)
    assert n_forms == len(forms_list)

    ngram_inventory = get_existing_sound_Ngrams(forms_tokenized, Ngrams)
    n_Ngrams = len(ngram_inventory)
    Ngram_list_indexes = {ngram: idx for idx,
                          ngram in enumerate(ngram_inventory)}
    array = np.zeros(shape=(n_forms, n_Ngrams))

    for form_row, form in enumerate(forms_tokenized):
        for token_ix in range(0, len(form)-(Ngrams-1)):
            # convert to tuple, because Ngram_list contains tuples
            current_Ngram = tuple(form[token_ix:token_ix+Ngrams])
            index = Ngram_list_indexes[current_Ngram]
            # index = Ngram_list.index(current_Ngram)
            array[form_row, index] = 1
    return array, np.array(ngram_inventory)


def create_language_dataset(df_language, language, Ngrams, sample_first, features_set, set_common_features_to_zero, remove_features_allzero, soundclasses):
    if language == "portuguese" or language == "estonian":
        form_column = f"form_{soundclasses}" if soundclasses != "none" else "phon_form"
        inflection_column = "inflection_class"
        lexeme_column = "lexeme_id"
        cell_column = "cell"
        if language == "portuguese":
            tag_present = "prs.ind"
        else:
            tag_present = "ind.prs"
        # tag_present_3pl = f"{tag_present}.3pl"
        tokenized_form_spaces = True
    else:  # data_format=="romance"
        # form_column = "Form"
        form_column = f"form_{soundclasses}" if soundclasses != "none" else "Form_tokenized"
        inflection_column = "Latin_Conjugation"
        lexeme_column = "Cognateset_ID_first"
        cell_column = "Cell"
        tag_present = "PRS-IND"
        # tag_present_3pl = f"{tag_present}.3PL"  # "'PRS-IND', '3PL'"
        tokenized_form_spaces = True

    # Keep only first form for a cell (Estonian has doubles)
    df_language = df_language.groupby(
        [lexeme_column, cell_column], as_index=False).first()

    if sample_first:
        df_language = df_language.head(sample_first)

    # Use cells from distillation
    df_used = df_language[df_language[cell_column].isin(
        paths[language]["cells_distillation"])]
    if WRITE_CSV:
        df_used.to_csv(f'only_used_{language}_stuff.csv')

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
    n_cells_unique = len(unique_cells_ordered)
    lexemes_unique = lexemes.unique()
    lexemes_unique.sort()
    n_lexemes_unique = len(lexemes_unique)
    # Inflection classes calculated based on only used dataset (possibly not full dataset)
    inflection_classes = list(inflections.unique())

    forms_encoded, ngram_inventory = create_onehot_forms_from_Ngrams(
        forms_list, Ngrams, tokenized_form_spaces)
    assert forms_encoded.shape[0] == len(df_used)
    assert forms_encoded.shape[1] == len(ngram_inventory)
    orig_ngram_inventory = ngram_inventory

    # Begin squeeze into verbs
    # Make new pooled ngram inventory with person tags
    if not features_set:  # concat

        # Pooled ngram inventory only used for plotting, not in encoding processing
        pooled_ngram_inventory = np.array(["".join(
            ngram) + '_'+unique_cell for unique_cell in unique_cells_ordered for ngram in ngram_inventory])
        ngram_inventory = pooled_ngram_inventory

        pooled_forms_encoded = np.zeros(
            (n_lexemes_unique, n_cells_unique*forms_encoded.shape[1]))
    else:  # Set representation
        pooled_forms_encoded = np.zeros(
            (n_lexemes_unique, forms_encoded.shape[1]))
    pooled_inflections = []
    # for lexeme_ix, lexeme_unique in enumerate(lexemes_unique):
    used_lexeme_indices = []
    for lexeme_ix, lexeme_unique in enumerate(lexemes_unique):
        rows_lexeme_unique = df_used[df_used[lexeme_column] == lexeme_unique]
        # Not necessarily in the same order as wanted
        # cells_np[indices_for_verb]
        cells_for_verb = list(rows_lexeme_unique[cell_column])
        if len(cells_for_verb) < len(unique_cells_ordered):
            # Skip whole lexeme if one of the cells is not there
            print(
                f"Skipping lexeme {lexeme_unique} because it does not include the following desired cells: {set(unique_cells_ordered).difference(set(cells_for_verb))}")
            continue
        # Keep track of which lexemes we use, because of lexeme dropping when not all cells are there
        used_lexeme_indices.append(lexeme_ix)

        if not features_set:  # concat
            pooled_forms_encoded_for_verb = [[]]  # add dimension
        else:
            pooled_forms_encoded_for_verb = np.zeros(
                (1, len(ngram_inventory)))
        # np.where(lexemes == lexeme_unique)[0]
        indices_for_verb = list(rows_lexeme_unique.index)

        # Save inflection for this pooled verb, from the first position
        pooled_inflections.append(
            rows_lexeme_unique[inflection_column].values[0])  # inflections.values[indices_for_verb[0]]
        for unique_cell in unique_cells_ordered:
            # if unique_cell in cells_for_verb:
            # index = np.where(
            #     unique_cell == cells_for_verb)
            index = cells_for_verb.index(unique_cell)
            # indices_for_verb[index[0][0]]
            index_in_forms_encoded = indices_for_verb[index]
            if not features_set:  # concat
                pooled_forms_encoded_for_verb = np.append(
                    pooled_forms_encoded_for_verb, forms_encoded[None, index_in_forms_encoded], axis=1)
            else:
                pooled_forms_encoded_for_verb = pooled_forms_encoded_for_verb + \
                    forms_encoded[index_in_forms_encoded]
            # else:
            #     if not features_set: #concat
            #         pooled_forms_encoded_for_verb = np.append(
            #             pooled_forms_encoded_for_verb, np.zeros((forms_encoded.shape[1])), axis=0)

        if set_common_features_to_zero:
            print(
                "This should not be executed when set_common_features_to_zero is not on.")
            # NOTE: This functionality assumes number of paradigm cells is 6: only works for Latin
            # NOTE: Not tested anymore after changing data processing code
            if not features_set:  # concat
                temp = np.reshape(pooled_forms_encoded_for_verb,
                                  (n_cells_unique, forms_encoded.shape[1]))
                S = sum(temp)
                # If the same gram was activated in all person tenses, their sum here is 6
                temp[:, S == 6] = 0
                pooled_forms_encoded_for_verb = np.reshape(
                    temp, (n_cells_unique*forms_encoded.shape[1]))
            else:
                pooled_forms_encoded_for_verb[pooled_forms_encoded_for_verb == 6] = 0

        # keeps only one activated n-gram, even though it may occur in several forms (for set representation, does nothing for concat)
        if features_set:
            pooled_forms_encoded_for_verb = np.clip(
                pooled_forms_encoded_for_verb, 0, 1)
        pooled_forms_encoded[lexeme_ix, :] = pooled_forms_encoded_for_verb
    inflections = pooled_inflections
    forms_encoded = pooled_forms_encoded
    # End Squeeze into verbs

    # For lexemes, make sure to return only data of used lexemes, because we drop lexemes
    # To return forms: list(forms[used_lexeme_indices])
    lexemes_used = list(lexemes[used_lexeme_indices])
    lexemes_unique_used = lexemes_unique[used_lexeme_indices]
    forms_encoded = forms_encoded[used_lexeme_indices]

    # Remove features for which whole column is 0:
    # mostly relevant in concat mode, where some ngrams do not occur for some cells
    # in set mode (or if not squeezing into verbs) there should not be all zeros features
    if remove_features_allzero:
        print(f"Features before remove 0: {forms_encoded.shape[1]}")
        features_all_zero = np.all(forms_encoded == 0, axis=0)
        forms_encoded = forms_encoded[:, ~features_all_zero]
        ngram_inventory = ngram_inventory[~features_all_zero]

    print(
        f"{language}. unique lexemes: {len(lexemes_unique_used)}. Inflection classes: {len(inflection_classes)}. Ngrams: {len(orig_ngram_inventory)}. Cells: {n_cells_unique}:{unique_cells_ordered}. Features: {forms_encoded.shape[1]}.")
    # NOTE: Normalized counts of inflection classes are based on lexemes before dropping entries that do not contain certain cells
    print(df_used.drop_duplicates(subset=lexeme_column)[
          inflection_column].value_counts(normalize=True))
    # print_diagnostic_encoding(form_column, lexeme_column,
    #                           df_used, lexemes_unique_used, forms_encoded, ngram_inventory)
    assert len(forms_encoded) == len(lexemes_used) == len(inflections)
    return forms_encoded, list(inflections), inflection_classes, lexemes_used, ngram_inventory


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
