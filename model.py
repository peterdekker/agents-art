from conf import ART_VIGILANCE, ART_LEARNING_RATE, INFLECTION_CLASSES, N_INFLECTION_CLASSES, OUTPUT_DIR
import plot
from neupy.algorithms import ART1
from sklearn.metrics import rand_score, adjusted_rand_score
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


def art(data_onehot, inflections_gold, cogids, language, n_runs=1, vigilances=[ART_VIGILANCE], repeat_dataset=False, batch_size=None, shuffle_data=False, eval_full_run=False, data_plot=False, show=False):
    if cogids is not None:
        cogids = np.array(cogids)
    records_end_scores = []
    records_end_clusters = []
    eval_vigilances = False
    if len(vigilances) > 1:
        eval_vigilances = True
    for vig in vigilances:
        if eval_vigilances:
            print(f"Vigilance: {vig}")
        if eval_full_run:
            rand_full_runs = []
            adj_rand_full_runs = []
        rand_avg_batches_runs = []
        adj_rand_avg_batches_runs = []
        rand_avg_batches_tail_runs = []
        adj_rand_avg_batches_tail_runs = []

        for r in range(n_runs):
            artnet = ART1(
                step=ART_LEARNING_RATE,
                rho=vig,
                n_clusters=N_INFLECTION_CLASSES,
            )
            # Make copy of data, because we will possibly shuffle
            input_data = data_onehot.copy()
            clusters_gold = np.array(inflections_gold) #this is also copy
            len_data = len(input_data)

            # If batching off, use full dataset
            if not batch_size:
                batch_size = len_data
            rand_batches = []
            adj_rand_batches = []
            rand_batches_tail = []
            adj_rand_batches_tail = []
            n_batches = len_data//batch_size
            # calculate tail to evaluate on last 10% of batches
            tail = n_batches - n_batches//10 - 1
            for rep in range(10 if repeat_dataset else 1):
                if shuffle_data:
                    # Makes taking batches sampling without replacement
                    shf = np.random.permutation(len(input_data))
                    input_data = input_data[shf]
                    clusters_gold = clusters_gold[shf]
                for b in range(n_batches):
                    batch = np.arange(b*batch_size, (b+1)*batch_size)
                    clusters_art_batch = artnet.train(input_data[batch])
                    rand_batch, adj_rand_batch, min_cluster_size_batch, max_cluster_size_batch = eval_results(
                        clusters_art_batch, clusters_gold[batch])
                    rand_batches.append(rand_batch)
                    adj_rand_batches.append(adj_rand_batch)
                    if b >= tail:
                        rand_batches_tail.append(rand_batch)
                        adj_rand_batches_tail.append(adj_rand_batch)
                print(rand_batches)
            # Keep numbers per run, to be able to give spread of results
            rand_avg_batches_runs.append(np.mean(rand_batches))
            adj_rand_avg_batches_runs.append(np.mean(adj_rand_batches))
            rand_avg_batches_tail_runs.append(np.mean(rand_batches_tail))
            adj_rand_avg_batches_tail_runs.append(np.mean(adj_rand_batches_tail))

            if eval_full_run:
                # Evaluate once more on full dataset
                clusters_art_full = artnet.train(input_data)
                rand_full, adj_rand_full, min_cluster_size_full, max_cluster_size_full = eval_results(
                    clusters_art_full, clusters_gold)
                rand_full_runs.append(rand_full)
                adj_rand_full_runs.append(adj_rand_full)

            if data_plot:
                plot.plot_data(data_onehot, labels=None, clusters=clusters_art_batch,
                                     micro_clusters=cogids[batch], file_label=f"pca-art-vig{vig}-run{r}-{language}", show=show)
            
            if eval_vigilances:
                # Take values from last batch, because in general, when evaluating vigilance, we dont work with batches
                records_end_scores.append(
                    {"vigilance": vig, "run": r, "metric": "rand", "score": rand_batch}) # rand
                records_end_scores.append(
                    {"vigilance": vig, "run": r, "metric": "adj_rand", "score": adj_rand_batch}) # adj_rand
                records_end_clusters.append(
                    {"vigilance": vig, "run": r, "metric": "min_cluster_size", "n_forms": min_cluster_size_batch}) # min_cluster_size
                records_end_clusters.append(
                    {"vigilance": vig, "run": r, "metric": "max_cluster_size", "n_forms": max_cluster_size_batch}) # max_cluster_size
        rand_avg_batches_mean = np.mean(rand_avg_batches_runs)
        adj_rand_avg_batches_mean = np.mean(adj_rand_avg_batches_runs)
        print(f" - Avg of batches (batch size {batch_size}). Rand: {rand_avg_batches_mean}. Adj_rand: {adj_rand_avg_batches_mean}")

        rand_avg_batches_tail_mean = np.mean(rand_avg_batches_tail_runs)
        adj_rand_avg_batches_tail_mean = np.mean(adj_rand_avg_batches_tail_runs)
        print(f" - Avg of last 10% batches (batch size {batch_size}). Rand: {rand_avg_batches_tail_mean}. Adj_rand: {adj_rand_avg_batches_tail_mean}")

        if eval_full_run:
            rand_full_mean = np.mean(rand_full_runs)
            adj_rand_full_mean = np.mean(adj_rand_full_runs)
            print(f" - Evaluate on extra full run. Rand: {rand_full_mean}. Adj_rand: {adj_rand_full_mean}")

    # Only create vigilance plot when comparing multiple vigilances
    if eval_vigilances:
        # Plot results
        df_end_scores = pd.DataFrame.from_records(records_end_scores)
        sns.lineplot(data=df_end_scores, x="vigilance",
                     y="score", hue="metric")
        plt.savefig(os.path.join(OUTPUT_DIR, f"scores-art-end-{language}.pdf"))
        if show:
            plt.show()
        plt.clf()
        df_end_scores.reset_index().pivot_table(index="vigilance", columns="metric", values="score", aggfunc='mean').to_csv(
            "scores-art-end.tex", sep="&", line_terminator="\\\\\n")

        df_end_clusters = pd.DataFrame.from_records(records_end_clusters)
        sns.lineplot(data=df_end_clusters, x="vigilance",
                     y="n_forms", hue="metric")
        plt.savefig(os.path.join(
            OUTPUT_DIR, f"clusters-art-end-{language}.pdf"))
        if show:
            plt.show()
        plt.clf()
        df_end_clusters.reset_index().pivot_table(index="vigilance", columns="metric", values="n_forms", aggfunc='mean').to_csv(
            "clusters-art-end.tex", sep="&", line_terminator="\\\\\n")


def eval_results(results, inflections_gold):
    # Calculate scores
    # silhouette = silhouette_score(X=data_onehot[batch], labels=clusters_art, metric="hamming")
    rand = rand_score(inflections_gold, results)
    adj_rand = adjusted_rand_score(inflections_gold, results)
    cluster_sizes = np.bincount(np.array(results, dtype=int))
    min_cluster_size = np.min(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    return rand, adj_rand, min_cluster_size, max_cluster_size



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
