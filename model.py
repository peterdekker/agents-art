from conf import ART_VIGILANCE, ART_LEARNING_RATE, OUTPUT_DIR, INITIAL_CLUSTERS, CONFIG_STRING, VIGILANCE_RANGE, EVAL_INTERVAL, MULTIPROCESSING, WRITE_CSV, N_PROCESSES
import plot
from art import ART1
from sklearn import cluster
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
from multiprocessing import Pool



def random_baseline(inflections_gold, n_inflection_classes):
    base = np.random.choice(
        range(0, n_inflection_classes), len(inflections_gold))
    rand, adj_rand, norm_mutual_info, adj_mutual_info, min_cluster_size, max_cluster_size = eval_results(
        base, inflections_gold)
    print(
        f" - Random baseline. RI: {rand}. ARI: {adj_rand} NMI: {norm_mutual_info}. AMI: {adj_mutual_info}")


def majority_baseline(inflections_gold):
    base = np.zeros(len(inflections_gold))
    rand, adj_rand, norm_mutual_info, adj_mutual_info, min_cluster_size, max_cluster_size = eval_results(
        base, inflections_gold)
    print(
        f" - Random baseline. RI: {rand}. ARI: {adj_rand} NMI: {norm_mutual_info}. AMI: {adj_mutual_info}")


def agg_cluster_baseline(data_onehot, inflections_gold, n_inflection_classes):
    agg_labels = cluster.AgglomerativeClustering(
        n_clusters=n_inflection_classes, affinity="manhattan", linkage="average").fit_predict(data_onehot)
    rand, adj_rand, norm_mutual_info, adj_mutual_info, min_cluster_size, max_cluster_size = eval_results(
        agg_labels, inflections_gold)
    print(
        f" - Agg clustering baseline. RI: {rand}. ARI: {adj_rand} NMI: {norm_mutual_info}. AMI: {adj_mutual_info}")


def kmeans_cluster_baseline(data_onehot, inflections_gold, n_inflection_classes):
    kmeans_labels = cluster.KMeans(
        n_clusters=n_inflection_classes).fit_predict(data_onehot)
    # print(cluster.KMeans.cluster_centers_)
    rand, adj_rand, norm_mutual_info, adj_mutual_info, min_cluster_size, max_cluster_size = eval_results(
        kmeans_labels, inflections_gold)
    print(
        f" - Kmeans clustering baseline. RI: {rand}. ARI: {adj_rand} NMI: {norm_mutual_info}. AMI: {adj_mutual_info}")



def art(data_onehot, forms, ngram_inventory, inflections_gold, inflection_classes, pca, language, n_runs=1, vigilances=[ART_VIGILANCE], repeat_dataset=False, batch_size=None, shuffle_data=False, data_plot=False, show=False, eval_intervals=False):
    eval_vigilances = False
    # np.random.shuffle(inflections_gold) # Make evaluation random, to test if model is doing something
    if len(vigilances) > 1:
        eval_vigilances = True

    if MULTIPROCESSING:
        if eval_intervals:
            raise ValueError("eval_intervals is not possible in multiprocessing mode.")
        # Param settings: only vig and r are variable
        param_settings = [(data_onehot, forms, ngram_inventory, inflections_gold, inflection_classes, pca, language, repeat_dataset, batch_size, shuffle_data, data_plot, show, eval_intervals, vig, r) for vig in vigilances for r in range(n_runs)]
        with Pool(processes=N_PROCESSES) as pool:
            records_listlist = pool.starmap(art_run_parallel_wrapper, param_settings) # take only first return value
    else: # If multiprocessing is off, this allows to do eval_intervals, which is done once per vigilance
        records_listlist = []
        for vig in vigilances:
            ari_per_interval_per_run = []
            for r in range(n_runs):
                records_batches, plottedIndices_batches, ari_per_interval_batches = art_run_parallel(data_onehot, forms, ngram_inventory, inflections_gold, inflection_classes, pca, language, repeat_dataset, batch_size, shuffle_data, data_plot, show, eval_intervals, vig, r)
                records_listlist.append(records_batches)
                ari_per_interval_per_run.append(ari_per_interval_batches)

            if eval_intervals:
                plot.plot_intervals(ari_per_interval_per_run, plottedIndices_batches,
                                    file_label=f"pca-art-vig{vig}-run{r}-{language}_protos_{CONFIG_STRING}", show=show)
    
    records = list(itertools.chain.from_iterable(records_listlist))
    df_results = pd.DataFrame(records)
    if WRITE_CSV:
        df_results.to_csv(os.path.join(
            OUTPUT_DIR, f"histogram_per_vigilance-{language}_{CONFIG_STRING}_out.csv"))
    print(df_results.groupby("vigilance")[["ri", "ari", "nmi", "ami", "min_cluster_size", "max_cluster_size", "n_clusters"]].mean()) # 
    df_results_small = df_results[["vigilance", "run", "cluster_population",
                                   "category_ngrams", "cluster_inflection_stats", "ari", "batch"]]
    if WRITE_CSV:
        df_results_small.to_csv(os.path.join(
            OUTPUT_DIR, f"cluster_stats_{CONFIG_STRING}.csv"))

    # Only create vigilance plot when comparing multiple vigilances
    if eval_vigilances:
        # Plot results

        df_melt_scores = pd.melt(df_results, id_vars=["vigilance", "run", "batch"], value_vars=[
                                 "ari"], var_name="metric", value_name="score")
        # df_melt_scores = pd.melt(df_results, id_vars=["vigilance", "run", "batch"], value_vars=["ri","ari", "nmi", "ami"], var_name="metric", value_name="score")
        df_melt_clusters = pd.melt(df_results, id_vars=["vigilance", "run", "batch"], value_vars=[
                                   "min_cluster_size", "max_cluster_size"], var_name="metric", value_name="size")
        
        # df_melt_ci = pd.melt(df_results, id_vars=["vigilance"], value_vars=[
        #                      "cluster_population"], var_name="metric", value_name="N_in_cluster")
        # from matplotlib import cm
        # x = df_melt_ci["vigilance"].values
        # xrep=np.repeat(x, 50)
        # y = np.linspace(1, 50,50)
        # yrep=np.tile(y,len(x))
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # # x, y = np.meshgrid(x, y)
        # z=df_melt_ci.N_in_cluster._values
        # z=np.stack(z, axis=0)
        # X, Y = np.meshgrid(x, y)
        # ax.plot_surface(X, Y, np.transpose(z),cmap=cm.coolwarm)
        # plt.savefig(os.path.join(OUTPUT_DIR, f"histogram_per_vigilance-{language}.pdf"))

        sns.lineplot(data=df_melt_scores, x="vigilance",
                     y="score", hue="metric")
        # This is obtained from one baseline run
        # rep_kmeans_ARI = np.ones((1, len(VIGILANCE_RANGE)))*0.782
        # rep_kmeans_AMI=np.ones((1,len(VIGILANCE_RANGE)))*0.835 #This is obtained from one baseline run
        # sns.lineplot(x=VIGILANCE_RANGE,
        #              y=rep_kmeans_ARI[0], dashes=True, hue=1)
        # plt.legend(labels=["ARI", "_ss", "AMI", "_ss",
        #            "Baseline ARI (kmeans)", "Baseline AMI (kmeans)"])
        plt.savefig(os.path.join(
            OUTPUT_DIR, f"scores-art-end-{language}-{CONFIG_STRING}.pdf"))
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
        if WRITE_CSV:
            df_results.to_csv(
                "clusters-scores-art-end.tex", sep="&", lineterminator="\\\\\n")

def art_run_parallel_wrapper(*args):
    return art_run_parallel(*args)[0]

def art_run_parallel(data_onehot, forms, ngram_inventory, inflections_gold, inflection_classes, pca, language, repeat_dataset, batch_size, shuffle_data, data_plot, show, eval_intervals, vig, r):
    print(f"Vigilance: {vig}. Run: {r}.")
    artnet = ART1(
                step=ART_LEARNING_RATE,
                rho=vig,
                n_clusters=INITIAL_CLUSTERS,
            )
    # Make copy of data, because we will possibly shuffle
    input_data = data_onehot.copy()
    clusters_gold = np.array(inflections_gold)  # this is also copy

    # If batching off, use full dataset
    len_data = len(input_data)
    if not batch_size:
        batch_size = len_data
    n_batches = len_data//batch_size
    plottedIndices_batches = []
    ari_per_interval_batches = []
    records_batches = []
    for rep in range(2 if repeat_dataset else 1):
        if shuffle_data:
            # Makes taking batches sampling without replacement
            shf = np.random.permutation(len(input_data))
            input_data = input_data[shf]
            F = np.array(forms)
            F = F[shf]
            clusters_gold = clusters_gold[shf]
        for b in range(n_batches):
            # TODO: batches right now (without replacement)
            # do exactly the same as processing the whole dataset at once.
            # Experiment with sampling with replacement.
            batch = np.arange(b*batch_size, (b+1)*batch_size)
            clusters_art_batch, prototypes, incrementalClasses, incrementalIndices = artnet.train(
                        input_data[batch], EVAL_INTERVAL)

            N_found_clusters = len(prototypes)
            clusters_gold_batch = clusters_gold[batch]
                    # print(clusters_art_batch)
            ri_batch, ari_batch, nmi_batch, ami_batch, min_cluster_size_batch, max_cluster_size_batch = eval_results(
                        clusters_art_batch, clusters_gold_batch)

            if eval_intervals:
                if rep == 0:
                    N_evals = len(incrementalClasses)

                    for i in range(0, N_evals):
                        Nsamples = len(incrementalClasses[i])
                        ri_batch, ari_batch, nmi_batch, ami_batch, min_cluster_size_batch, max_cluster_size_batch = eval_results(
                                    incrementalClasses[i], clusters_gold_batch[0:Nsamples])
                        ari_per_interval_batches.append(ari_batch)
                        plottedIndices_batches.append(incrementalIndices[i])
                else:
                    ari_per_interval_batches.append(ari_batch)
                    plottedIndices_batches.append(
                                incrementalIndices[-1]+plottedIndices_batches[-1])
            histo = np.histogram(clusters_art_batch, bins=np.arange(0, N_found_clusters+1))[0]
            order = np.flip(np.argsort(histo))
            cluster_population = histo[order]
            prototypes = prototypes[order, :]
            S = np.sum(prototypes, axis=0)
            ngram_inventory = np.array(ngram_inventory)
            always_activated_features = np.argwhere(
                        S == N_found_clusters)

            category_ngrams = []
            for p in range(0, N_found_clusters):
                ones = np.nonzero(prototypes[p, :])[0]
                cluster_ngrams = []
                for i in ones:
                    if i in always_activated_features:
                        # If always activated feature, add in position 0 -> clearer for barplot
                        cluster_ngrams.insert(0, ngram_inventory[i])
                    else:
                        cluster_ngrams.append(ngram_inventory[i])
                category_ngrams.append(cluster_ngrams)

            clusters_gold_int = []
            ORDER = np.array(inflection_classes)
            for i in range(0, len(clusters_gold_batch)):
                clusters_gold_int.append(
                            np.where(ORDER == clusters_gold_batch[i])[0][0])
            # Number of clusters (rows) that are not unused (unused=all 1s)
            n_used_clusters = np.sum(1-np.all(prototypes, axis=1))

            # This counts how many of each gold-standard words per each inflection class is clustered in each of the clusters coming from ART
            # Eg. 0th row being [2,3,0,4,6] would mean that cluster 0 (coming out from ART) includes 2 words from inflection class 'I', 3 words from 'II', and so on
            cluster_inflection_stats = np.zeros(
                        (n_used_clusters, len(inflection_classes)))
            for i in range(0, len(clusters_gold_int)):
                cluster_inflection_stats[int(
                            clusters_art_batch[i]), clusters_gold_int[i]] += 1
            row_sums = cluster_inflection_stats.sum(axis=1)

            # With multiple batches/repeats, it's possible that on the next batch, no input samples are set into a category created on a previous batch. In this case the new category will be empty.
            # This is done to avoid division by zero.
            row_sums[np.where(row_sums == 0)] = 1

            # Here the counts are changed in percentages
            cluster_inflection_stats_percent = cluster_inflection_stats / \
                        row_sums[:, np.newaxis]
            records_batches.append(
                        {"vigilance": vig, "run": r, "batch": rep*n_batches+b,
                         "ri": ri_batch, "ari": ari_batch, "nmi": nmi_batch, "ami": ami_batch,
                         "ari_per_interval": ari_per_interval_batches,
                         "ari_per_interval_indices":  plottedIndices_batches,
                         "cluster_population": cluster_population,
                         "category_ngrams": category_ngrams,
                         "prototypes": prototypes,
                         "cluster_inflection_stats": cluster_inflection_stats,
                         "cluster_inflection_stats_percent": cluster_inflection_stats_percent,
                         "min_cluster_size": min_cluster_size_batch, "max_cluster_size": max_cluster_size_batch, "n_clusters": N_found_clusters})

    if data_plot:
        # Use result from last batch to plot TODO: think about this
        df = plot.transform_using_fitted_pca(prototypes, pca)
        # plot.plot_data(df, labels=None, clusters=range(0,20),
        #  micro_clusters=cogids[batch], file_label=f"pca-art-vig{vig}-run{r}-{language}", show=show)
        df.columns = ['dim1', 'dim2']
        prototype_based_new_coords = []
        for i in range(0, len(clusters_gold)):
            prototype_N = int(clusters_art_batch[i])
            matching_prototype_coord = df.values[prototype_N]
            with_noise = matching_prototype_coord + \
                        np.random.randn(2)*0.02
            prototype_based_new_coords.append(with_noise)

        df2 = pd.DataFrame(prototype_based_new_coords)
        df2.columns = ['dim1', 'dim2']
        plot.plot_data(df2, labels=None, clusters=clusters_gold, prototypes=df,
                               file_label=f"pca-art-vig{vig}-run{r}-{language}_protos_{CONFIG_STRING}", show=show)
        plot.plot_barchart(cluster_inflection_stats, inflection_classes,  # category_ngrams, always_activated_ngrams,
                                   file_label=f"pca-art-vig{vig}-run{r}-{language}_protos_{CONFIG_STRING}", show=show)
                           
    return records_batches,plottedIndices_batches,ari_per_interval_batches


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


