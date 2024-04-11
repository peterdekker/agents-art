
from conf import OUTPUT_DIR, SAMPLE_FIRST, SET_COMMON_FEATURES_TO_ZERO, REMOVE_FEATURES_ALLZERO, VIGILANCE_RANGE, paths, params, mode_params
from model import art, majority_baseline, random_baseline, kmeans_cluster_baseline
import pandas as pd
import os
import argparse
import plot
import data
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Command line ART model.')
    for mode_param in mode_params:
        parser.add_argument(f'--{mode_param}', action='store_true')
    for param_name in params:
        param = params[param_name]
        if param["type"]==bool:
            parser.add_argument(f"--{param_name}", default=param["default"], action='store_true')
        else:
            parser.add_argument(f"--{param_name}", type=param["type"], default=param["default"])
    args = parser.parse_args()
    print(args)

    config_string = "train_test" if args.train_test else ""
    config_string += "".join([f"-{param_name}={param_value}" for param_name,param_value in vars(args).items() if param_name not in mode_params and param_name != "language" and (args.single_run or param_name!="vigilance_single_run")])
    language = args.language

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conjugation_df_path = paths[language]["conjugation_df_path"]
    if language == "latin":
        if not os.path.exists(conjugation_df_path):
            # Load data
            data.load_romance_dataset(conjugation_df_path)
    elif language == "estonian" or language == "portuguese":
        if not os.path.exists(conjugation_df_path):
            data.load_paralex_dataset(language, conjugation_df_path)

    # First time script is run for a language, we write and then immediately read from CSV file.
    df_language = pd.read_csv(
        conjugation_df_path, index_col=0, low_memory=False)
    forms_onehot, inflections, inflection_classes, _, ngram_inventory = data.create_language_dataset(df_language, language, Ngrams=args.ngrams,
                                                                                                                 sample_first=SAMPLE_FIRST, features_set=args.features_set, set_common_features_to_zero=SET_COMMON_FEATURES_TO_ZERO, remove_features_allzero=REMOVE_FEATURES_ALLZERO, soundclasses=args.soundclasses, use_present=args.use_present)
    if args.plot_data:
        # Plot data before running model
        df, pca = plot.fit_pca(forms_onehot)
        plot.plot_data(df, labels=None, clusters=inflections,
                       micro_clusters=None, file_label=f"pca-art-data_ngram_hamming_original_MCA_-{language}_{config_string}", show=False)
    
    if args.single_run:
        art(forms_onehot, ngram_inventory, inflections, inflection_classes, language, config_string, n_runs=1, vigilances=[args.vigilance_single_run], shuffle_data=True, repeat_dataset=True, data_plot=True, train_test=args.train_test, eval_intervals=args.eval_intervals)
    
    if args.eval_vigilances:
        art(forms_onehot, ngram_inventory, inflections, inflection_classes, language,config_string, 
            n_runs=args.n_runs, shuffle_data=True, repeat_dataset=True, vigilances=VIGILANCE_RANGE, train_test=args.train_test, eval_intervals=args.eval_intervals)
    
    if args.eval_batches:
        print(f"Full data shuffle, {args.n_runs} runs:")
        art(forms_onehot, ngram_inventory, inflections, inflection_classes,
            None, language, config_string, n_runs=args.n_runs, shuffle_data=True)
        print(f"Repeat dataset shuffle, {args.n_runs} runs:")
        art(forms_onehot, ngram_inventory, inflections, inflection_classes, None,
            language, config_string, n_runs=args.n_runs, repeat_dataset=True, shuffle_data=True)
        print(f"batch 10 shuffle, {args.n_runs} runs:")
        art(forms_onehot, ngram_inventory, inflections, inflection_classes,
            None, language,config_string,  batch_size=10, n_runs=args.n_runs, shuffle_data=True)
        print(f"batch 50 shuffle, {args.n_runs} runs:")
        art(forms_onehot, ngram_inventory, inflections, inflection_classes,
            None, language,config_string,  batch_size=50, n_runs=args.n_runs, shuffle_data=True)

    # if args.eval_intervals:
    #     art(forms_onehot, forms, ngram_inventory, inflections, inflection_classes, language,config_string, 
    #         n_runs=args.n_runs, shuffle_data=True, repeat_dataset=True, eval_intervals=True)


    if args.baseline:
        print("Baselines:")
        # print("Full data shuffle, n runs")
        # art_one(forms_onehot, inflections, cogids, language, n_runs=args.n_runs, shuffle_data=True)
        records_baseline = []
        records_baseline.append(majority_baseline(inflections))
        records_baseline.append(random_baseline(inflections, n_inflection_classes=len(inflection_classes)))

        # print("Agg clustering baseline:")
        # agg_cluster_baseline(forms_onehot, inflections, n_inflection_classes=5) # TODO: Infer #classes from data

        records_baseline.append(kmeans_cluster_baseline(
            forms_onehot, inflections, n_inflection_classes=len(inflection_classes)))
        
        print(pd.DataFrame(records_baseline).set_index("method"))


if __name__ == "__main__":
    main()
