### Script to generate the evaluation plots of performance per vigilance from csv files
### Required input files in directory: csv files generated by evaluation.py using the eval_vigilances option, both with and without train_test, for the 3 languages
# Creates different width plots than main script

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# sns.set_theme(font="Charis SIL Compact", font_scale=1.4)
# plt.rcParams['figure.figsize'] = [7,5.6]
# Set to (7,5.6) for normal
sns.set_theme(font="Charis SIL Compact", font_scale=1.4)
plt.rcParams['figure.dpi'] = 200
NORMAL_PLOT_SIZE = (7,5.6)
TRAIN_TEST_PLOT_SIZE = (5.25,5.6)

CSV_DIR = "output"
OUTPUT_DIR_NEW = "output_new"
os.makedirs(OUTPUT_DIR_NEW, exist_ok=True)

dfs = []
for train_test in [False, True]:
    for language in ["latin", "portuguese", "estonian"]:
        config_string = f"{'train_test' if train_test else ''}-features_set=False-soundclasses=none-use_present=False-ngrams=3-n_runs={'1' if train_test else '10'}"
        df_results = pd.read_csv(os.path.join(CSV_DIR, f"results-{language}-{config_string}.csv"), index_col=0)
        # df_lang = df_lang[df_lang["model"]=="ART1"]

        # df_melt_scores = pd.melt(df_total, id_vars=["vigilance", "run", "batch", "fold_id", "mode", "model", "language"], value_vars=[
        #                                 "ARI"], var_name="metric", value_name="score")
        # print(df_melt_scores)
        # ax_scores = sns.lineplot(data=df_melt_scores, x="vigilance",
        #                     y="score", hue="language", style="mode")
        # ax_scores.set_ylim(top=1)
        # plt.savefig(os.path.join(OUTPUT_DIR_NEW, f"{filename}.pdf"))
        # plt.clf()

        ## Scores plot
        df_melt_scores = pd.melt(df_results, id_vars=["vigilance", "run", "batch", "fold_id", "mode", "model"], value_vars=[
                                "ARI", "AMI"], var_name="metric", value_name="score")
        
        if train_test:
            plt.figure(figsize=TRAIN_TEST_PLOT_SIZE)
            # train_test: Only use ART score (no baseline) and only report ARI
            df_melt_scores = df_melt_scores[df_melt_scores["model"].isin(["ART1"])]
            df_melt_scores = df_melt_scores[df_melt_scores["metric"].isin(["ARI"])]
            ax_scores = sns.lineplot(data=df_melt_scores, x="vigilance",
                    y="score", hue="mode")
        else:
            plt.figure(figsize=NORMAL_PLOT_SIZE)
            df_melt_scores = df_melt_scores[df_melt_scores["model"].isin(["ART1", "k-means baseline"])]
            ax_scores = sns.lineplot(data=df_melt_scores, x="vigilance",
                        y="score", hue="model", style="metric")
        ax_scores.set_ylim(top=1)
        plt.savefig(os.path.join(
            OUTPUT_DIR_NEW, f"scores-{language}-{config_string}.pdf"))
        plt.clf()

        ## N_clusters plot
        df_melt_n_clusters = pd.melt(df_results, id_vars=["vigilance", "run", "batch", "fold_id", "mode", "model"], value_vars=[
                                "n_clusters"], value_name="# clusters")
        df_melt_n_clusters = df_melt_n_clusters[df_melt_n_clusters["model"].isin(["ART1", "real inflection classes"])]
        # Rename 'model' to 'clustering', because 'real inflection classes' are not really a model
        df_melt_n_clusters = df_melt_n_clusters.rename(columns={"model":"clustering"})
        if train_test:
            pass
            # No nclusters plot in train-test
        else:
            plt.figure(figsize=NORMAL_PLOT_SIZE)
            sns.lineplot(data=df_melt_n_clusters, x="vigilance",
                        y="# clusters", hue="clustering")
            plt.savefig(os.path.join(
                OUTPUT_DIR_NEW, f"nclusters-{language}-{config_string}.pdf"))
            plt.clf()