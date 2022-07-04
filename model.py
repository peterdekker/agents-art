from conf import ART_VIGILANCE, ART_LEARNING_RATE, INFLECTION_CLASSES, N_INFLECTION_CLASSES, OUTPUT_DIR
from neupy.algorithms import ART1
from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

import evaluation
import data



def art_one(data_onehot, inflections_gold, cogids, vigilances=[ART_VIGILANCE], data_plot=False):
    art_iterated_no(data_onehot, n_runs = 1, inflections_gold=inflections_gold, cogids=cogids, vigilances=vigilances, data_plot=data_plot)

def art_iterated_no(data_onehot, n_runs, inflections_gold, cogids, vigilances=[ART_VIGILANCE], data_plot=False):
    inflections_gold = np.array(inflections_gold)
    if cogids is not None:
        cogids = np.array(cogids)
    records_end_scores = []
    records_end_clusters = []
    for vig in vigilances:
        print(f" - Vigilance: {vig}")
        for r in range(n_runs):
            print(f" -- Run: {r}")
            # Initialize original data for new run
            input_next_gen = data_onehot.copy()
            # Use whole dataset
            batch = np.arange(len(input_next_gen))

            artnet = ART1(
                step=ART_LEARNING_RATE,
                rho=vig,
                n_clusters=N_INFLECTION_CLASSES,
            )
            clusters_art = artnet.predict(input_next_gen[batch])

            # Calculate scores
            silhouette = silhouette_score(X=data_onehot[batch], labels=clusters_art, metric="hamming")
            rand = rand_score(inflections_gold[batch], clusters_art)
            adj_rand = adjusted_rand_score(inflections_gold[batch], clusters_art)
            cluster_sizes = np.bincount(np.array(clusters_art,dtype=int))
            min_cluster_size = np.min(cluster_sizes)
            max_cluster_size = np.max(cluster_sizes)

            if data_plot:
                evaluation.plot_data(data_onehot[batch], labels=None, clusters=clusters_art, micro_clusters=cogids[batch], file_label=f"data-art-{n_timesteps}-end-vig{vig}")
            records_end_scores.append({"vigilance": vig, "metric": "silhouette", "score": silhouette})
            records_end_scores.append({"vigilance": vig, "metric": "rand", "score": rand})
            records_end_scores.append({"vigilance": vig, "metric": "adj_rand", "score": adj_rand})
            records_end_clusters.append({"vigilance": vig, "metric": "min_cluster_size", "n_forms": min_cluster_size})
            records_end_clusters.append({"vigilance": vig, "metric": "max_cluster_size", "n_forms": max_cluster_size})

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Plot results
    df_end_scores = pd.DataFrame.from_records(records_end_scores)
    df_end_scores.pivot(index="vigilance", columns="metric", values="score").to_csv("scores-art-end.tex", sep="&", line_terminator = "\\\\\n")
    sns.lineplot(data=df_end_scores, x="vigilance", y = "score", hue="metric")
    plt.savefig(os.path.join(OUTPUT_DIR, f"scores-art-end.pdf"))
    plt.clf()
    
    df_end_clusters = pd.DataFrame.from_records(records_end_clusters)
    df_end_clusters.pivot(index="vigilance", columns="metric", values="n_forms").to_csv("clusters-art-end.tex", sep="&", line_terminator = "\\\\\n")
    sns.lineplot(data=df_end_clusters, x="vigilance", y = "n_forms", hue="metric")
    plt.savefig(os.path.join(OUTPUT_DIR, f"clusters-art-end.pdf"))
    plt.clf()

def art_iterated(data_onehot, n_runs, n_timesteps, batch_size, inflections_gold, cogids, vigilances=[ART_VIGILANCE], data_plot=False):
    inflections_gold = np.array(inflections_gold)
    if cogids is not None:
        cogids = np.array(cogids)
    records_end_scores = []
    records_end_clusters = []
    records_course_scores = []
    records_course_clusters = []
    for vig in vigilances:
        print(f" - Vigilance: {vig}")
        for r in range(n_runs):
            print(f" -- Run: {r}")
            # Initialize original data for new run
            input_next_gen = data_onehot.copy()
            for i in range(n_timesteps):
                batch = np.random.choice(len(input_next_gen), batch_size, replace=False)

                artnet = ART1(
                    step=ART_LEARNING_RATE,
                    rho=vig,
                    n_clusters=N_INFLECTION_CLASSES,
                )
                clusters_art = artnet.predict(input_next_gen[batch])

                # Calculate scores
                silhouette = silhouette_score(X=data_onehot[batch], labels=clusters_art, metric="hamming")
                rand = rand_score(inflections_gold[batch], clusters_art)
                adj_rand = adjusted_rand_score(inflections_gold[batch], clusters_art)
                cluster_sizes = np.bincount(np.array(clusters_art,dtype=int))
                min_cluster_size = np.min(cluster_sizes)
                max_cluster_size = np.max(cluster_sizes)

                # Transfer information to next generation
                clusters_art_onehot, _ = data.create_onehot_inflections(clusters_art)
                # Replace last columns, represeting inflection class, by one-hot vector of inferred inflection classes
                input_next_gen[batch,-N_INFLECTION_CLASSES:] = clusters_art_onehot
                records_course_scores.append({"run": r, "timestep": i, "vigilance": vig, "metric": "silhouette", "score": silhouette})
                #records_course_scores.append({"run": r, "timestep": i, "vigilance": vig, "metric": "rand", "score": rand})
                records_course_scores.append({"run": r, "timestep": i, "vigilance": vig, "metric": "adj_rand", "score": adj_rand})
                records_course_clusters.append({"run": r, "timestep": i, "vigilance": vig, "metric": "min_cluster_size", "n_forms": min_cluster_size})
                records_course_clusters.append({"run": r, "timestep": i, "vigilance": vig, "metric": "max_cluster_size", "n_forms": max_cluster_size})
            if data_plot:
                evaluation.plot_data(data_onehot[batch], labels=None, clusters=clusters_art, micro_clusters=cogids[batch], file_label=f"data-art-{n_timesteps}-end-vig{vig}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Plot results
    print("Plotting graphs.")
    df_course_scores = pd.DataFrame.from_records(records_course_scores)
    sns.lineplot(data=df_course_scores, x="timestep", y = "score", hue="metric", style="vigilance")
    plt.savefig(os.path.join(OUTPUT_DIR, f"scores-art-course-batch{batch_size}.pdf"))
    plt.clf()
    
    df_course_clusters = pd.DataFrame.from_records(records_course_clusters)
    sns.lineplot(data=df_course_clusters, x="timestep", y = "n_forms", hue="metric", style="vigilance")
    plt.savefig(os.path.join(OUTPUT_DIR, f"clusters-art-course-batch{batch_size}.pdf"))
    plt.clf()
    print("Done plotting.")