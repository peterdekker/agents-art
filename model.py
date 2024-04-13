from conf import ART_LEARNING_RATE, OUTPUT_DIR, INITIAL_CLUSTERS, EVAL_INTERVAL, MULTIPROCESSING, WRITE_CSV, WRITE_TEX, N_PROCESSES
import plot
from art import ART1
from sklearn import cluster
from sklearn.model_selection import KFold
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
from multiprocessing import Pool

sns.set(font="Charis SIL Compact")


def random_baseline(inflections_gold, n_inflection_classes):
    base = np.random.choice(
        range(0, n_inflection_classes), len(inflections_gold))
    ri, ari, nmi, ami, min_cluster_size, max_cluster_size = eval_results(
        base, inflections_gold)
    n_clusters = len(list(set(base)))
    return {"method": "random", "RI": ri, "ARI": ari, "NMI": nmi, "AMI": ami,
                         "min_cluster_size": min_cluster_size, "max_cluster_size": max_cluster_size, "clusters ART": n_clusters}


def majority_baseline(inflections_gold):
    base = np.zeros(len(inflections_gold))
    ri, ari, nmi, ami, min_cluster_size, max_cluster_size = eval_results(
        base, inflections_gold)
    n_clusters = len(list(set(base)))
    return {"method": "majority", "RI": ri, "ARI": ari, "NMI": nmi, "AMI": ami,
                         "min_cluster_size": min_cluster_size, "max_cluster_size": max_cluster_size, "clusters ART": n_clusters}


def agg_cluster_baseline(data_onehot, inflections_gold, n_inflection_classes):
    agg_labels = cluster.AgglomerativeClustering(
        n_clusters=n_inflection_classes, affinity="manhattan", linkage="average").fit_predict(data_onehot)
    ri, ari, nmi, ami, min_cluster_size, max_cluster_size = eval_results(
        agg_labels, inflections_gold)
    n_clusters = len(list(set(agg_labels)))
    return {"method": "agg", "RI": ri, "ARI": ari, "NMI": nmi, "AMI": ami,
                         "min_cluster_size": min_cluster_size, "max_cluster_size": max_cluster_size, "clusters ART": n_clusters}


def kmeans_cluster_baseline(data_onehot, inflections_gold, n_inflection_classes):
    kmeans_labels = cluster.KMeans(
        n_clusters=n_inflection_classes).fit_predict(data_onehot)
    # print(cluster.KMeans.cluster_centers_)
    ri, ari, nmi, ami, min_cluster_size, max_cluster_size = eval_results(
        kmeans_labels, inflections_gold)
    n_clusters = len(list(set(kmeans_labels)))
    return {"method": "kmeans", "RI": ri, "ARI": ari, "NMI": nmi, "AMI": ami,
                         "min_cluster_size": min_cluster_size, "max_cluster_size": max_cluster_size, "clusters ART": n_clusters}


def art(data_onehot, ngram_inventory, inflections_gold, inflection_classes, language, config_string, n_runs=1, vigilances=[], repeat_dataset=False, batch_size=None, shuffle_data=False, data_plot=False, show=False, eval_intervals=False, train_test=False):
    eval_vigilances = False
    # np.random.shuffle(inflections_gold) # Make evaluation random, to test if model is doing something
    if len(vigilances) > 1:
        eval_vigilances = True
    
    pca = None
    if data_plot:
        _, pca = plot.fit_pca(data_onehot)

    if train_test:
        kf = KFold(n_splits=10, shuffle=True)
        train_test_splits = list(kf.split(data_onehot))
    else:
        train_test_splits = [(None,None)]


    if MULTIPROCESSING:
        if eval_intervals:
            raise ValueError("eval_intervals is not possible in multiprocessing mode.")
        param_settings = [(data_onehot, ngram_inventory, inflections_gold, inflection_classes, pca, language, repeat_dataset, batch_size, shuffle_data, data_plot, show, eval_intervals, train_test, config_string, vig, r, fold_id, train_ix, test_ix) for vig in vigilances for r in range(n_runs) for fold_id, (train_ix, test_ix) in enumerate(train_test_splits)]
        with Pool(processes=N_PROCESSES) as pool:
            records_listlist = pool.starmap(art_run_parallel_wrapper, param_settings) # take only first return value
    else: # If multiprocessing is off, this allows to do eval_intervals, which is done once per vigilance
        records_listlist = []
        for vig in vigilances:
            ari_per_interval_per_run = []
            for r in range(n_runs):
                for fold_id, (train_ix, test_ix) in enumerate(train_test_splits):
                    records_batches, plottedIndices_batches, ari_per_interval_batches = art_run_parallel(data_onehot, ngram_inventory, inflections_gold, inflection_classes, pca, language, repeat_dataset, batch_size, shuffle_data, data_plot, show, eval_intervals, train_test, config_string, vig, r, fold_id, train_ix, test_ix)
                    records_listlist.append(records_batches)
                    ari_per_interval_per_run.append(ari_per_interval_batches)

            if eval_intervals:
                if train_test:
                    print("Line shows #datapoints in whole dataset, not #datapoints in train fold.")
                plot.plot_intervals(ari_per_interval_per_run, plottedIndices_batches, len(data_onehot),
                                    file_label=f"intervals-vig{vig}-{language}_{config_string}", show=show)
    
    records = list(itertools.chain.from_iterable(records_listlist))
    df_results = pd.DataFrame(records)
    # Add ground truth number of inflection classes as extra column, as baseline
    df_results["inflection classes"] = len(inflection_classes)
    print(df_results.groupby(["vigilance", "mode"], sort=False)[["RI", "ARI", "NMI", "AMI", "min_cluster_size", "max_cluster_size", "clusters ART", "inflection classes"]].mean()) # 

    # Only create vigilance plot when comparing multiple vigilances
    if eval_vigilances:
        # Plot results

        df_melt_scores = pd.melt(df_results, id_vars=["vigilance", "run", "batch", "fold_id", "mode"], value_vars=[
                                 "ARI", "AMI"], var_name="metric", value_name="score")
        # df_melt_clusters = pd.melt(df_results, id_vars=["vigilance", "run", "batch", "fold_id", "mode"], value_vars=[
        #                            "min_cluster_size", "max_cluster_size"], var_name="metric", value_name="size")
        df_melt_n_clusters = pd.melt(df_results, id_vars=["vigilance", "run", "batch", "fold_id", "mode"], value_vars=[
                                   "clusters ART", "inflection classes"], var_name="metric", value_name="# clusters")
        
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

        ## Scores plot
        if train_test:
            ax_scores = sns.lineplot(data=df_melt_scores, x="vigilance",
                     y="score", hue="metric", style="mode")
        else:
            ax_scores = sns.lineplot(data=df_melt_scores, x="vigilance",
                        y="score", hue="metric")
        ax_scores.set_ylim(0, 1)
        # This is obtained from one baseline run
        # rep_kmeans_ARI = np.ones((1, len(VIGILANCE_RANGE)))*0.782
        # rep_kmeans_AMI=np.ones((1,len(VIGILANCE_RANGE)))*0.835 #This is obtained from one baseline run
        # sns.lineplot(x=VIGILANCE_RANGE,
        #              y=rep_kmeans_ARI[0], dashes=True, hue=1)
        # plt.legend(labels=["ARI", "_ss", "AMI", "_ss",
        #            "Baseline ARI (kmeans)", "Baseline AMI (kmeans)"])
        plt.savefig(os.path.join(
            OUTPUT_DIR, f"scores-{language}-{config_string}.pdf"))
        if show:
            plt.show()
        plt.clf()

        ## Cluster sizes plot
        # if train_test:
        #     sns.lineplot(data=df_melt_clusters, x="vigilance",
        #                 y="size", hue="metric", style="mode")
        # else:
        #     sns.lineplot(data=df_melt_clusters, x="vigilance",
        #                 y="size", hue="metric")
        # plt.savefig(os.path.join(
        #     OUTPUT_DIR, f"clusters-{language}-{config_string}.pdf"))
        # if show:
        #     plt.show()
        # plt.clf()
        
        ## N_clusters plot
        if train_test:
            sns.lineplot(data=df_melt_n_clusters, x="vigilance",
                        y="# clusters", hue="metric", style="mode")
        else:
            sns.lineplot(data=df_melt_n_clusters, x="vigilance",
                        y="# clusters", hue="metric")
        # plt.axhline(y=len(inflection_classes)) # plot number of real inflection classes
        plt.savefig(os.path.join(
            OUTPUT_DIR, f"nclusters-{language}-{config_string}.pdf"))
        if show:
            plt.show()
        plt.clf()

        if WRITE_CSV:
            df_results.to_csv(os.path.join(
            OUTPUT_DIR, 
                f"results-{language}-{config_string}.csv"))
        if WRITE_TEX:
            df_results.to_csv(os.path.join(
            OUTPUT_DIR, 
                f"results-{language}-{config_string}.tex"), sep="&", lineterminator="\\\\\n")

def art_run_parallel_wrapper(*args):
    return art_run_parallel(*args)[0]

def art_run_parallel(data_onehot, ngram_inventory, inflections_gold, inflection_classes, pca, language, repeat_dataset, batch_size_given, shuffle_data, data_plot, show, eval_intervals, train_test, config_string, vig, r, fold_id, train_ix, test_ix):
    print(f"Vigilance: {vig}. Run: {r}.{' Split id '+ str(fold_id) if fold_id is not None else ''}")
    artnet = ART1(
                step=ART_LEARNING_RATE,
                rho=vig,
                n_clusters=INITIAL_CLUSTERS,
            )
    
    plottedIndices_batches = []
    ari_per_interval_batches = []
    records_batches = []

    # Make copy of data, because we will possibly shuffle
    data_onehot_full = data_onehot.copy()
    clusters_gold_full = np.array(inflections_gold)  # this is also copy


    # In train/test mode, shuffle the data
    if train_test:
        # For train-test mode, shuffle and extra time at beginning, because k-folds are in order
        if shuffle_data:
            # Makes taking batches sampling without replacement
            shf = np.random.permutation(len(data_onehot_full))
            data_onehot_full = data_onehot_full[shf]
            clusters_gold_full = clusters_gold_full[shf]
            # print(f"Data original (len {len(data_onehot_full)}): {np.count_nonzero(data_onehot_full[3,:])}")
    
    modes = ["train","test"] if train_test else ["train"]
    for mode in modes:
        if train_test:
            if mode=="train":
                input_data = data_onehot_full[train_ix]
                clusters_gold = clusters_gold_full[train_ix]
            elif mode=="test":
                input_data = data_onehot_full[test_ix]
                clusters_gold = clusters_gold_full[test_ix]
        else:
            input_data = data_onehot_full
            clusters_gold = clusters_gold_full
        # print(f"Data {mode} (len {len(input_data)}): {np.count_nonzero(input_data[3,:])}")


        # If batching off, use full dataset (or: full train fold in train-test mode)
        len_data = len(input_data)
        batch_size = batch_size_given if batch_size_given else len_data
        n_batches = len_data//batch_size
        n_reps = 2 if repeat_dataset and mode=="train" else 1
        for rep in range(n_reps):
            if shuffle_data:
                # Shuffle before every repetition, so data gets presented again in different order.
                # Makes taking batches sampling without replacement
                shf = np.random.permutation(len(input_data))
                input_data = input_data[shf]
                # F = np.array(forms)
                # F = F[shf]
                clusters_gold = clusters_gold[shf]
            for b in range(n_batches):
                # NOTE: batches right now (without replacement)
                # do exactly the same as processing the whole dataset at once.
                # Experiment with sampling with replacement.
                batch = np.arange(b*batch_size, (b+1)*batch_size)
                if mode=="train":
                    clusters_art_batch, prototypes, incrementalClasses, incrementalIndices = artnet.train(
                                input_data[batch], EVAL_INTERVAL)
                elif mode=="test":
                    # clusters_art_batch, prototypes, incrementalClasses, incrementalIndices = artnet.train(
                    #             input_data[batch], EVAL_INTERVAL)
                    clusters_art_batch, prototypes = artnet.test(
                                input_data[batch], only_bottom_up=True)
                
                N_found_clusters = len(prototypes)
                clusters_gold_batch = clusters_gold[batch]
                ri_batch, ari_batch, nmi_batch, ami_batch, min_cluster_size_batch, max_cluster_size_batch = eval_results(
                            clusters_art_batch, clusters_gold_batch)

                if eval_intervals and mode=="train": # for test, eval_intervals is not useful because nothing is learned
                    if rep == 0:
                        N_evals = len(incrementalClasses)
                        for i in range(0, N_evals):
                            Nsamples = len(incrementalClasses[i])
                            _, ari_interval, _, _, _, _ = eval_results(
                                        incrementalClasses[i], clusters_gold_batch[0:Nsamples])
                            ari_per_interval_batches.append(ari_interval)
                            plottedIndices_batches.append(incrementalIndices[i])
                    else:
                        ari_per_interval_batches.append(ari_batch)
                        plottedIndices_batches.append(
                                    incrementalIndices[-1]+plottedIndices_batches[-1])
                # Only do results analysis and save record for last repetition
                if rep==n_reps-1:
                    histo = np.histogram(clusters_art_batch, bins=np.arange(0, N_found_clusters+1))[0]
                    order = np.flip(np.argsort(histo))
                    cluster_population = histo[order]
                    prototypes = prototypes[order, :]
                    S = np.sum(prototypes, axis=0)
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
                    # cluster_inflection_stats_percent = cluster_inflection_stats / \
                    #             row_sums[:, np.newaxis]
                    records_batches.append(
                                {"vigilance": vig, "run": r, "fold_id": fold_id, "batch": b, "mode": mode,
                                "RI": ri_batch, "ARI": ari_batch, "NMI": nmi_batch, "AMI": ami_batch,
                                #"ari_per_interval": ari_per_interval_batches,
                                #"ari_per_interval_indices":  plottedIndices_batches,
                                #"cluster_population": cluster_population,
                                #"category_ngrams": category_ngrams,
                                #"prototypes": prototypes,
                                #"cluster_inflection_stats": cluster_inflection_stats,
                                #"cluster_inflection_stats_percent": cluster_inflection_stats_percent,
                                "min_cluster_size": min_cluster_size_batch, "max_cluster_size": max_cluster_size_batch, "clusters ART": N_found_clusters})

    if data_plot and not train_test:
        # NOTE: Result from last batch is used for plot
        # NOTE: At the moment not working with train-test, question is then whether train or test phase should be plotted.
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
                               file_label=f"{language}-vig{vig}-run{r}_{config_string}", show=show)
        plot.plot_barchart(cluster_inflection_stats, inflection_classes,  # category_ngrams, always_activated_ngrams,
                                   file_label=f"{language}-vig{vig}-run{r}_{config_string}", show=show)
    
    # print(f"Vigilance: {vig}. Run: {r}. Finished.")
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


