from conf import ART_VIGILANCE, ART_LEARNING_RATE, INFLECTION_CLASSES, N_INFLECTION_CLASSES, OUTPUT_DIR
import plot
from neupy.algorithms import ART1
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

import numpy as np

def random_baseline(inflections_gold):
    base = np.random.choice(range(0,5),len(inflections_gold))
    rand, adj_rand, min_cluster_size, max_cluster_size = eval_results(base, inflections_gold)
    print(f" - Random baseline. Rand: {rand}. Adj_rand: {adj_rand}")

def majority_baseline(inflections_gold):
    base = np.zeros(len(inflections_gold))
    rand, adj_rand, min_cluster_size, max_cluster_size = eval_results(base, inflections_gold)
    print(f" - Majority baseline. Rand: {rand}. Adj_rand: {adj_rand}")


def art(data_onehot, inflections_gold, cogids, language, n_runs=1, vigilances=[ART_VIGILANCE], repeat_dataset=False, batch_size=None, shuffle_data=False, data_plot=False, show=False):
    if cogids is not None:
        cogids = np.array(cogids)
    records = []
    eval_vigilances = False
    ################# Make evaluation random, to test if model is doing something
    #np.random.shuffle(inflections_gold)
    #################
    if len(vigilances) > 1:
        eval_vigilances = True
    
    for vig in vigilances:
        if eval_vigilances:
            print(f"Vigilance: {vig}")
        
        for r in range(n_runs):
            artnet = ART1(
                step=ART_LEARNING_RATE,
                rho=vig,
                n_clusters=N_INFLECTION_CLASSES,
            )
            # Make copy of data, because we will possibly shuffle
            input_data = data_onehot.copy()
            clusters_gold = np.array(inflections_gold) #this is also copy

            # If batching off, use full dataset
            len_data = len(input_data)
            if not batch_size:
                batch_size = len_data
            n_batches = len_data//batch_size

            for rep in range(10 if repeat_dataset else 1):
                if shuffle_data:
                    # Makes taking batches sampling without replacement
                    shf = np.random.permutation(len(input_data))
                    input_data = input_data[shf]
                    clusters_gold = clusters_gold[shf]
                for b in range(n_batches):
                    batch = np.arange(b*batch_size, (b+1)*batch_size)
                    clusters_art_batch = artnet.train(input_data[batch])
                    ri_batch, ari_batch, nmi_batch, ami_batch, min_cluster_size_batch, max_cluster_size_batch = eval_results(
                        clusters_art_batch, clusters_gold[batch])
                    records.append(
                    {"vigilance": vig, "run": r, "batch": rep*n_batches+b,
                     "ri": ri_batch, "ari": ari_batch, "nmi": nmi_batch, "ami": ami_batch,
                     "min_cluster_size": min_cluster_size_batch, "max_cluster_size": max_cluster_size_batch})

            if data_plot:
                # Use result from last batch to plot TODO: think about this
                plot.plot_data(data_onehot, labels=None, clusters=clusters_art_batch,
                                     micro_clusters=cogids[batch], file_label=f"pca-art-vig{vig}-run{r}-{language}", show=show)
            
    df_results = pd.DataFrame(records)
    print(df_results.groupby("vigilance")["ri", "ari", "nmi", "ami", "min_cluster_size", "max_cluster_size"].mean())


    # Only create vigilance plot when comparing multiple vigilances
    if eval_vigilances:
        # Plot results
        df_melt_scores = pd.melt(df_results, id_vars=["vigilance", "run", "batch"], value_vars=["ri","ari", "nmi", "ami"], var_name="metric", value_name="score")
        df_melt_clusters = pd.melt(df_results, id_vars=["vigilance", "run", "batch"], value_vars=["min_cluster_size","max_cluster_size"], var_name="metric", value_name="size")
        sns.lineplot(data=df_melt_scores, x="vigilance",
                     y="score", hue="metric")
        plt.savefig(os.path.join(OUTPUT_DIR, f"scores-art-end-{language}.pdf"))
        if show:
            plt.show()
        plt.clf()

        sns.lineplot(data=df_melt_clusters, x="vigilance",
                     y="size", hue="metric")
        plt.savefig(os.path.join(
            OUTPUT_DIR, f"clusters-art-end-{language}.pdf"))
        if show:
            plt.show()
        plt.clf()
        df_results.to_csv(
            "clusters-scores-art-end.tex", sep="&", line_terminator="\\\\\n")


def eval_results(results, inflections_gold):
    # Calculate scores
    # silhouette = silhouette_score(X=data_onehot[batch], labels=clusters_art, metric="hamming")
    rand = rand_score(inflections_gold, results)
    adj_rand = adjusted_rand_score(inflections_gold, results)
    norm_mutual_info = normalized_mutual_info_score(inflections_gold, results)
    adj_mutual_info = adjusted_mutual_info_score(inflections_gold, results)
    cluster_sizes = np.bincount(np.array(results, dtype=int))
    min_cluster_size = np.min(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    return rand, adj_rand, norm_mutual_info, adj_mutual_info, min_cluster_size, max_cluster_size



# def art_iterated(data_onehot, n_runs, n_timesteps, batch_size_iterated, inflections_gold, cogids, language, vigilances=[ART_VIGILANCE], data_plot=False):
#     inflections_gold = np.array(inflections_gold)
#     if cogids is not None:
#         cogids = np.array(cogids)
#     records_end_scores = []
#     records_end_clusters = []
#     records_course_scores = []
#     records_course_clusters = []
#     for vig in vigilances:
#         print(f" - Vigilance: {vig}")
#         for r in range(n_runs):
#             print(f" -- Run: {r}")
#             # Initialize original data for new run
#             input_next_gen = data_onehot.copy()
#             for i in range(n_timesteps):
#                 batch = np.random.choice(
#                     len(input_next_gen), batch_size_iterated, replace=False)

#                 artnet = ART1(
#                     step=ART_LEARNING_RATE,
#                     rho=vig,
#                     n_clusters=N_INFLECTION_CLASSES,
#                 )
#                 clusters_art = artnet.predict(input_next_gen[batch])

#                 # Calculate scores
#                 silhouette = silhouette_score(
#                     X=data_onehot[batch], labels=clusters_art, metric="hamming")
#                 rand = rand_score(inflections_gold[batch], clusters_art)
#                 adj_rand = adjusted_rand_score(
#                     inflections_gold[batch], clusters_art)
#                 cluster_sizes = np.bincount(np.array(clusters_art, dtype=int))
#                 min_cluster_size = np.min(cluster_sizes)
#                 max_cluster_size = np.max(cluster_sizes)

#                 # Transfer information to next generation
#                 clusters_art_onehot, _ = data.create_onehot_inflections(
#                     clusters_art)
#                 # Replace last columns, represeting inflection class, by one-hot vector of inferred inflection classes
#                 input_next_gen[batch, -
#                                N_INFLECTION_CLASSES:] = clusters_art_onehot
#                 records_course_scores.append(
#                     {"run": r, "timestep": i, "vigilance": vig, "metric": "silhouette", "score": silhouette})
#                 #records_course_scores.append({"run": r, "timestep": i, "vigilance": vig, "metric": "rand", "score": rand})
#                 records_course_scores.append(
#                     {"run": r, "timestep": i, "vigilance": vig, "metric": "adj_rand", "score": adj_rand})
#                 records_course_clusters.append(
#                     {"run": r, "timestep": i, "vigilance": vig, "metric": "min_cluster_size", "n_forms": min_cluster_size})
#                 records_course_clusters.append(
#                     {"run": r, "timestep": i, "vigilance": vig, "metric": "max_cluster_size", "n_forms": max_cluster_size})
#             if data_plot:
#                 evaluation.plot_data(data_onehot[batch], labels=None, clusters=clusters_art,
#                                      micro_clusters=cogids[batch], file_label=f"art-{n_timesteps}-end-vig{vig}-{language}")

#     # Plot results
#     print("Plotting graphs.")
#     df_course_scores = pd.DataFrame.from_records(records_course_scores)
#     sns.lineplot(data=df_course_scores, x="timestep",
#                  y="score", hue="metric", style="vigilance")
#     plt.savefig(os.path.join(
#         OUTPUT_DIR, f"scores-art-course-batch{batch_size_iterated}-{language}.pdf"))
#     plt.clf()

#     df_course_clusters = pd.DataFrame.from_records(records_course_clusters)
#     sns.lineplot(data=df_course_clusters, x="timestep",
#                  y="n_forms", hue="metric", style="vigilance")
#     plt.savefig(os.path.join(
#         OUTPUT_DIR, f"clusters-art-course-batch{batch_size_iterated}-{language}.pdf"))
#     plt.clf()
#     print("Done plotting.")
