
from conf import OUTPUT_DIR, LANGUAGE, EMPTY_SYMBOL,  SAMPLE_FIRST, N_RUNS, CONCAT_VERB_FEATURES, USE_ONLY_3PL, CONFIG_STRING, SQUEEZE_INTO_VERBS, NGRAMS, SET_COMMON_FEATURES_TO_ZERO, VIGILANCE_RANGE, paths
from model import art, majority_baseline, random_baseline, kmeans_cluster_baseline
import pandas as pd
import os
import argparse
import numpy as np
import plot
import data
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description='Command line ART model.')
    parser.add_argument('--single_run_plotdata', action='store_true')
    parser.add_argument('--eval_batches', action='store_true')
    parser.add_argument('--eval_vigilances', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--language', type=str, default=LANGUAGE)
    args = parser.parse_args()

    language = args.language

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conjugation_df_path = paths[language]["conjugation_df_path"]
    if language=="latin":
        if not os.path.exists(conjugation_df_path):
            # Load data
            conjugation_df = data.load_romance_dataset()
            
            latin_conjugation_df = conjugation_df[conjugation_df["Language_ID"] == "Italic_Latino-Faliscan_Latin"]
            latin_conjugation_df.to_csv(conjugation_df_path)
            # First time script is run, we write and then immediately read from CSV file. This makes Cell column right Python object

        df_language = pd.read_csv(conjugation_df_path, index_col=0)
        # Create dataset only for Latin
        
        forms_onehot, _, forms, inflections, cogids, bigram_inventory = data.create_language_dataset(df_language, Ngrams=NGRAMS, empty_symbol=EMPTY_SYMBOL, form_column="Form", inflection_column="Latin_Conjugation", cogid_column="Cognateset_ID_first",
                                                                                                                    sample_first=SAMPLE_FIRST, use_only_3PL=USE_ONLY_3PL, squeeze_into_verbs=SQUEEZE_INTO_VERBS, concat_verb_features=CONCAT_VERB_FEATURES, set_common_features_to_zero=SET_COMMON_FEATURES_TO_ZERO)
    elif language=="estonian":
        if not os.path.exists(conjugation_df_path):
            data.load_paralex_dataset()

    if args.single_run_plotdata:
        # Plot data before running model
        df, pca = plot.fit_pca(forms_onehot)
        # plot.plot_data(df, labels=None, clusters=inflectionss,
        # micro_clusters=cogids, file_label=f"pca-art-data_bigram_hamming_original_MCA_-{LANGUAGE_ROMANCE_DATASET}", show=False)
        plot.plot_data(df, labels=None, clusters=inflections,
                       micro_clusters=None, file_label=f"pca-art-data_bigram_hamming_original_MCA_-{language}_{CONFIG_STRING}", show=False)
        # print(f"Full data shuffle, {N_RUNS} runs")
        art(forms_onehot, forms, bigram_inventory, inflections, cogids, pca,
            language, n_runs=1, shuffle_data=True, repeat_dataset=True, data_plot=True)

    if args.eval_batches:
        print(f"Full data shuffle, {N_RUNS} runs:")
        art(forms_onehot, forms, bigram_inventory, inflections,
            cogids, None, language, n_runs=N_RUNS, shuffle_data=True)
        print(f"Repeat dataset shuffle, {N_RUNS} runs:")
        art(forms_onehot, forms, bigram_inventory, inflections, cogids, None,
            language, n_runs=N_RUNS, repeat_dataset=True, shuffle_data=True)
        print(f"batch 10 shuffle, {N_RUNS} runs:")
        art(forms_onehot, forms, bigram_inventory, inflections, cogids,
            None, language, batch_size=10, n_runs=N_RUNS, shuffle_data=True)
        print(f"batch 50 shuffle, {N_RUNS} runs:")
        art(forms_onehot, forms, bigram_inventory, inflections, cogids,
            None, language, batch_size=50, n_runs=N_RUNS, shuffle_data=True)

    if args.eval_vigilances:
        art(forms_onehot, forms, bigram_inventory, inflections, cogids, None, language,
            n_runs=N_RUNS, shuffle_data=True, repeat_dataset=True, vigilances=VIGILANCE_RANGE)

    if args.baseline:
        # print("Full data shuffle, n runs")
        # art_one(forms_onehot, inflections, cogids, language, n_runs=N_RUNS, shuffle_data=True)
        print("Majority baseline:")
        majority_baseline(inflections)

        print("Random baseline:")
        random_baseline(inflections)

        # print("Agg clustering baseline:")
        # agg_cluster_baseline(forms_onehot, inflections)

        print("Kmeans clustering baseline:")
        kmeans_cluster_baseline(forms_onehot, inflections)

        print("Comparison to inflection classes:")
        language_df = latin_conjugation_df[latin_conjugation_df["Language_ID"]
                                           == LANGUAGE_ROMANCE_DATASET]
        print("Token count:")
        print(language_df["Latin_Conjugation"].value_counts(normalize=True))
        # print("Type count")
        # print(language_df.drop_duplicates(subset="Cognateset_ID_first")["Latin_Conjugation"].value_counts(normalize=True))

    # if iterated_run:
    #   forms_inflections_onehot = np.concatenate((forms_onehot, inflections_onehot), axis=1)
    #     if plot_data_before:
    #         score = evaluation.plot_data(forms_inflections_onehot, labels=None, clusters=inflections, micro_clusters=cogids, file_label=f"inflections-{language}")
    #         print (f"Silhouette score, data before run (with inflection class): {score}")
    #     inflections_empty = np.zeros(inflections_onehot.shape)
    #     forms_empty_inflections_onehot = np.concatenate((forms_onehot, inflections_empty), axis=1)

    #     for bs in [10,20,50,100,200,500]:
    #         print(f"Batch size: {bs}")
    #         art_iterated(forms_empty_inflections_onehot, n_runs=20, n_timesteps=500, batch_size_iterated=bs, inflections_gold=inflections, cogids=cogids, vigilances=[0.25, 0.5, 0.75] )


if __name__ == "__main__":
    main()
