
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import data
import evaluation
import numpy as np
import os
from model import art_one, art_iterated
from conf import OUTPUT_DIR, LANGUAGE, EMPTY_SYMBOL, BYTEPAIR_ENCODING, SAMPLE_FIRST, N_RUNS


 # Operation modes
single_run_eval_batches = True
single_run_eval_vigilances = False



def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Load data
    forms_df, cognates_df, lects_df = data.load_romance_dataset()
    
    # Filter data
    forms_df_1cognate = data.filter_romance_empty_multicog(forms_df)

    # Filter on Latin inflection classes
    latin_conjugation_df = data.filter_romance_inflections(forms_df_1cognate, cognates_df)

    # Create dataset per LANGUAGE
    forms_onehot, inflections_onehot, forms, inflections, cogids = data.create_language_dataset(latin_conjugation_df, LANGUAGE, empty_symbol=EMPTY_SYMBOL, encoding="bytepair" if BYTEPAIR_ENCODING else "onehot", sample_first=SAMPLE_FIRST)
    forms_inflections_onehot = np.concatenate((forms_onehot, inflections_onehot), axis=1)

    
    if single_run_eval_batches:
        print("Full data")
        art_one(forms_onehot, inflections, cogids, LANGUAGE)
        print("Full data shuffle, n runs")
        art_one(forms_onehot, inflections, cogids, LANGUAGE, n_runs=N_RUNS, shuffle_data=True)
        print("Repeat dataset")
        art_one(forms_onehot, inflections, cogids, LANGUAGE, repeat_dataset=True)
        print("Repeat dataset shuffle, n runs")
        art_one(forms_onehot, inflections, cogids, LANGUAGE, n_runs=N_RUNS, repeat_dataset=True, shuffle_data=True)
        print("batch 10")
        art_one(forms_onehot, inflections, cogids, LANGUAGE, batch_size=10)
        print("batch 10 shuffle, n runs")
        art_one(forms_onehot, inflections, cogids, LANGUAGE, batch_size=10, n_runs=N_RUNS, shuffle_data=True)
        print("batch 50")
        art_one(forms_onehot, inflections, cogids, LANGUAGE, batch_size=50)
        print("batch 50 shuffle, n runs")
        art_one(forms_onehot, inflections, cogids, LANGUAGE, batch_size=50, n_runs=N_RUNS, shuffle_data=True)
        print("batch 1000")
        art_one(forms_onehot, inflections, cogids, LANGUAGE, batch_size=1000)
        print("batch 1000 shuffle, n runs")
        art_one(forms_onehot, inflections, cogids, LANGUAGE, batch_size=1000, n_runs=N_RUNS, shuffle_data=True)

        # Next: check scores for individual batches

    if single_run_eval_vigilances:
        # One run
        #art_one(forms_onehot, inflections, cogids, LANGUAGE, vigilances = np.arange(0,1.05,0.05))
        # n runs with shuffle
        art_one(forms_onehot, inflections, cogids, LANGUAGE, n_runs=N_RUNS, shuffle_data=True, vigilances = np.arange(0,1.05,0.05))

    # if iterated_run:
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