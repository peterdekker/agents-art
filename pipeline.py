
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import data
import plot
import numpy as np
import argparse
import os
import pandas as pd
from model import art, majority_baseline, random_baseline
from conf import OUTPUT_DIR, LANGUAGE, EMPTY_SYMBOL, BYTEPAIR_ENCODING, SAMPLE_FIRST, N_RUNS, LATIN_CONJUGATION_DF_FILE





def main():
    parser = argparse.ArgumentParser(description='Command line ART model.')
    parser.add_argument('--single_run_plotdata', action='store_true')
    parser.add_argument('--eval_batches', action='store_true')
    parser.add_argument('--eval_vigilances', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if os.path.exists(LATIN_CONJUGATION_DF_FILE):
        latin_conjugation_df = pd.read_csv (LATIN_CONJUGATION_DF_FILE)
    else:
        #Load data
        forms_df, cognates_df, lects_df = data.load_romance_dataset()
        # Filter data
        forms_df_1cognate = data.filter_romance_empty_multicog(forms_df)
        # Filter on Latin inflection classes
        latin_conjugation_df = data.filter_romance_inflections(forms_df_1cognate, cognates_df)
        latin_conjugation_df.to_csv(LATIN_CONJUGATION_DF_FILE)

    # Create dataset per LANGUAGE  
    forms_onehot, inflections_onehot, forms, inflections, cogids, bigram_inventory = data.create_language_dataset(latin_conjugation_df, LANGUAGE, empty_symbol=EMPTY_SYMBOL, encoding="bytepair" if BYTEPAIR_ENCODING else "onehot", sample_first=SAMPLE_FIRST)
    
    # if args.single_run_plotdata:
        # Plot data before running model
    df, pca = plot.fit_pca(forms_onehot)
    # plot.plot_data(df, labels=None, clusters=inflections,
                            # micro_clusters=cogids, file_label=f"pca-art-data_bigram_hamming_original_MCA_-{LANGUAGE}", show=False)
    plot.plot_data(df, labels=None, clusters=inflections,
                            micro_clusters=None, file_label=f"pca-art-data_bigram_hamming_original_MCA_-{LANGUAGE}", show=False)  
    # print(f"Full data shuffle, {N_RUNS} runs")
    art(forms_onehot, forms, bigram_inventory,inflections, cogids, pca,LANGUAGE, n_runs=1, shuffle_data=True, data_plot=True)
    
    if args.eval_batches:
        print(f"Full data shuffle, {N_RUNS} runs:")
        art(forms_onehot, inflections, cogids, LANGUAGE, n_runs=N_RUNS, shuffle_data=True)
        print(f"Repeat dataset shuffle, {N_RUNS} runs:")
        art(forms_onehot, inflections, cogids, LANGUAGE, n_runs=N_RUNS, repeat_dataset=True, shuffle_data=True)
        print(f"batch 10 shuffle, {N_RUNS} runs:")
        art(forms_onehot, inflections, cogids, LANGUAGE, batch_size=10, n_runs=N_RUNS, shuffle_data=True)
        print(f"batch 50 shuffle, {N_RUNS} runs:")
        art(forms_onehot, inflections, cogids, LANGUAGE, batch_size=50, n_runs=N_RUNS, shuffle_data=True)
        print(f"batch 1000 shuffle, {N_RUNS} runs:")
        art(forms_onehot, inflections, cogids, LANGUAGE, batch_size=1000, n_runs=N_RUNS, shuffle_data=True)

    # if args.eval_vigilances:
    # art(forms_onehot, forms, bigram_inventory, inflections, cogids, LANGUAGE, n_runs=N_RUNS, shuffle_data=True, vigilances = np.arange(0.0,1.05,0.05))
    
    if args.baseline:
        # print("Full data shuffle, n runs")
        # art_one(forms_onehot, inflections, cogids, LANGUAGE, n_runs=N_RUNS, shuffle_data=True)
        print("Majority baseline:")
        majority_baseline(inflections)

        print("Random baseline:")
        random_baseline(inflections)

        print("Comparison to inflection classes:")
        language_df = latin_conjugation_df[latin_conjugation_df["Language_ID"]==LANGUAGE]
        print("Token count:")
        print(language_df["Latin_Conjugation"].value_counts(normalize=True))
        # print("Type count")
        # print(language_df.drop_duplicates(subset="Cognateset_ID_first")["Latin_Conjugation"].value_counts(normalize=True))

    # if iterated_run:
    #   forms_inflections_onehot = np.concatenate((forms_onehot, inflections_onehot), axis=1)
        # if plot_data_before:
        #     score = evaluation.plot_data(forms_inflections_onehot, labels=None, clusters=inflections, micro_clusters=cogids, file_label=f"inflections-{LANGUAGE}")
        #     print (f"Silhouette score, data before run (with inflection class): {score}")
    #     inflections_empty = np.zeros(inflections_onehot.shape)
    #     forms_empty_inflections_onehot = np.concatenate((forms_onehot, inflections_empty), axis=1)

    #     for bs in [10,20,50,100,200,500]:
    #         print(f"Batch size: {bs}")
    #         art_iterated(forms_empty_inflections_onehot, n_runs=20, n_timesteps=500, batch_size_iterated=bs, inflections_gold=inflections, cogids=cogids, vigilances=[0.25, 0.5, 0.75] )



if __name__ == "__main__":
    main()