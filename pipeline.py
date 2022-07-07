
# Checked 16-06-22: same as notebook
import data
import evaluation
import numpy as np
from model import art_one, art_iterated

plot_data_before = False

single_run_eval_batches = True
single_run_eval_vigilances = False
iterated_run = False

#language = "Italic_Latino-Faliscan_Latin"
language = "French_Modern_Standard"

def main():
    # Load data
    forms_df,cognates_df,lects_df = data.load_romance_dataset()
    
    # Filter data
    forms_df_1cognate = data.filter_romance_empty_multicog(forms_df)

    # Filter on Latin inflection classes
    latin_conjugation_df = data.filter_romance_inflections(forms_df_1cognate, cognates_df)

    # Create dataset per language
    forms_onehot, inflections_onehot, forms, inflections, cogids = data.create_language_dataset(latin_conjugation_df, language, empty_symbol=True, sample_first=1000)
    forms_inflections_onehot = np.concatenate((forms_onehot, inflections_onehot), axis=1)

    
    if plot_data_before:
        score = evaluation.plot_data(forms_inflections_onehot, labels=None, clusters=inflections, micro_clusters=cogids, file_label=f"inflections-{language}")
        print (f"Silhouette score, data before run (with inflection class): {score}")
    
    if single_run_eval_batches:
        print("Full data")
        art_one(forms_onehot, inflections, cogids, language)
        print("Full data shuffle")
        art_one(forms_onehot, inflections, cogids, language, shuffle_data=True)
        print("Repeat dataset")
        art_one(forms_onehot, inflections, cogids, language, repeat_dataset=True)
        print("Repeat dataset shuffle")
        art_one(forms_onehot, inflections, cogids, language, repeat_dataset=True, shuffle_data=True)
        print("batch 10")
        art_one(forms_onehot, inflections, cogids, language, batch_size=10)
        print("batch 50")
        art_one(forms_onehot, inflections, cogids, language, batch_size=50)
        print("random batch 10")
        art_one(forms_onehot, inflections, cogids, language, batch_size=10, shuffle_data=True)
        print("random batch 50")
        art_one(forms_onehot, inflections, cogids, language, batch_size=50, shuffle_data=True)

        # Next: evaluating on avg of batches, instead of doing full run after. Evaluate full dataset run on small batches?
        # Avg over multiple runs, then show statistics

    if single_run_eval_vigilances:
        art_one(forms_onehot, inflections, cogids, language, vigilances = np.arange(0,1.05,0.05))

    if iterated_run:
        inflections_empty = np.zeros(inflections_onehot.shape)
        forms_empty_inflections_onehot = np.concatenate((forms_onehot, inflections_empty), axis=1)

        for bs in [10,20,50,100,200,500]:
            print(f"Batch size: {bs}")
            art_iterated(forms_empty_inflections_onehot, n_runs=20, n_timesteps=500, batch_size_iterated=bs, inflections_gold=inflections, cogids=cogids, vigilances=[0.25, 0.5, 0.75] )



if __name__ == "__main__":
    main()