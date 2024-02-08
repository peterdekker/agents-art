
from conf import OUTPUT_DIR, LANGUAGE, SAMPLE_FIRST, N_RUNS, CONCAT_VERB_FEATURES, USE_ONLY_3PL, CONFIG_STRING, SQUEEZE_INTO_VERBS, NGRAMS, SET_COMMON_FEATURES_TO_ZERO, REMOVE_FEATURES_ALLZERO, VIGILANCE_RANGE, paths
from model import art, majority_baseline, random_baseline, kmeans_cluster_baseline
import pandas as pd
import os
import argparse
import plot
import data
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description='Command line ART model.')
    parser.add_argument('--plot_data', action='store_true')
    parser.add_argument('--single_run', action='store_true')
    parser.add_argument('--eval_batches', action='store_true')
    parser.add_argument('--eval_vigilances', action='store_true')
    parser.add_argument('--eval_intervals', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--language', type=str, default=LANGUAGE)
    args = parser.parse_args()

    language = args.language

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conjugation_df_path = paths[language]["conjugation_df_path"]
    if language == "latin":
        if not os.path.exists(conjugation_df_path):
            # Load data
            data.load_romance_dataset(conjugation_df_path, only_latin=True)
        data_format = "romance"
        use_only_present_lang = True  # for Latin, inflection classes define only present
    elif language == "estonian" or language == "portuguese":
        if not os.path.exists(conjugation_df_path):
            data.load_paralex_dataset(language, conjugation_df_path)
        data_format = "paralex"
        # Estonian and Portuguese: inflection classes define all forms. But for now: only use present
        use_only_present_lang = True

    # First time script is run for a language, we write and then immediately read from CSV file.
    df_language = pd.read_csv(
        conjugation_df_path, index_col=0, low_memory=False)

    forms_onehot, forms, inflections, inflection_classes, _, ngram_inventory = data.create_language_dataset(df_language, language, data_format=data_format, use_only_present=use_only_present_lang, Ngrams=NGRAMS,
                                                                                                                 sample_first=SAMPLE_FIRST,  use_only_3PL=USE_ONLY_3PL, squeeze_into_verbs=SQUEEZE_INTO_VERBS, concat_verb_features=CONCAT_VERB_FEATURES, set_common_features_to_zero=SET_COMMON_FEATURES_TO_ZERO, remove_features_allzero=REMOVE_FEATURES_ALLZERO)
    if args.plot_data:
        # Plot data before running model
        df, pca = plot.fit_pca(forms_onehot)
        plot.plot_data(df, labels=None, clusters=inflections,
                       micro_clusters=None, file_label=f"pca-art-data_ngram_hamming_original_MCA_-{language}_{CONFIG_STRING}", show=False)
    
    if args.single_run:
        # print(f"Full data shuffle, {N_RUNS} runs")
        art(forms_onehot, forms, ngram_inventory, inflections, inflection_classes, language, n_runs=1, shuffle_data=True, repeat_dataset=True, data_plot=False)

    if args.eval_batches:
        print(f"Full data shuffle, {N_RUNS} runs:")
        art(forms_onehot, forms, ngram_inventory, inflections, inflection_classes,
            None, language, n_runs=N_RUNS, shuffle_data=True)
        print(f"Repeat dataset shuffle, {N_RUNS} runs:")
        art(forms_onehot, forms, ngram_inventory, inflections, inflection_classes, None,
            language, n_runs=N_RUNS, repeat_dataset=True, shuffle_data=True)
        print(f"batch 10 shuffle, {N_RUNS} runs:")
        art(forms_onehot, forms, ngram_inventory, inflections, inflection_classes,
            None, language, batch_size=10, n_runs=N_RUNS, shuffle_data=True)
        print(f"batch 50 shuffle, {N_RUNS} runs:")
        art(forms_onehot, forms, ngram_inventory, inflections, inflection_classes,
            None, language, batch_size=50, n_runs=N_RUNS, shuffle_data=True)

    if args.eval_intervals:
        print(f"Full data shuffle, {N_RUNS} runs:")
        art(forms_onehot, forms, ngram_inventory, inflections, inflection_classes, None, LANGUAGE,
            n_runs=N_RUNS, shuffle_data=True, repeat_dataset=True, eval_intervals=True)

    if args.eval_vigilances:
        art(forms_onehot, forms, ngram_inventory, inflections, inflection_classes, language,
            n_runs=N_RUNS, shuffle_data=True, repeat_dataset=True, vigilances=VIGILANCE_RANGE)

    if args.baseline:
        # print("Full data shuffle, n runs")
        # art_one(forms_onehot, inflections, cogids, language, n_runs=N_RUNS, shuffle_data=True)
        print("Majority baseline:")
        # TODO: Infer #classes from data
        majority_baseline(inflections, n_inflection_classes=5)

        print("Random baseline:")
        # TODO: Infer #classes from data
        random_baseline(inflections, n_inflection_classes=5)

        # print("Agg clustering baseline:")
        # agg_cluster_baseline(forms_onehot, inflections, n_inflection_classes=5) # TODO: Infer #classes from data

        print("Kmeans clustering baseline:")
        # TODO: Infer #classes from data
        kmeans_cluster_baseline(
            forms_onehot, inflections, n_inflection_classes=5)

        print("Comparison to inflection classes:")
        print("Token count:")
        print(df_language["Latin_Conjugation"].value_counts(normalize=True))
        # print("Type count")
        # print(df_language.drop_duplicates(subset="Cognateset_ID_first")["Latin_Conjugation"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
