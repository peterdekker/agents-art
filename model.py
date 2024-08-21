from conf import ART_LEARNING_RATE, OUTPUT_DIR, INITIAL_CLUSTERS, EVAL_INTERVAL, MULTIPROCESSING, WRITE_CSV, WRITE_TEX, N_PROCESSES, MIN_DATAPOINTS_CLASS_PORTUGUESE, MAX_CLUSTERS_PORTUGUESE
import plot
from art import ART1
from sklearn import cluster
from sklearn.model_selection import KFold
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np
import numpy as np
import seaborn as sns  # from conf import sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
from multiprocessing import Pool
from collections import defaultdict


def random_baseline(inflections_gold, n_inflection_classes):
    base = np.random.choice(
        range(0, n_inflection_classes), len(inflections_gold))
    ri, ari, nmi, ami, min_cluster_size, max_cluster_size = eval_results(
        base, inflections_gold)
    n_clusters = len(list(set(base)))
    return {"model": "random", "RI": ri, "ARI": ari, "NMI": nmi, "AMI": ami,
            "min_cluster_size": min_cluster_size, "max_cluster_size": max_cluster_size, "n_clusters": n_clusters}


def majority_baseline(inflections_gold):
    base = np.zeros(len(inflections_gold))
    ri, ari, nmi, ami, min_cluster_size, max_cluster_size = eval_results(
        base, inflections_gold)
    n_clusters = len(list(set(base)))
    return {"model": "majority", "RI": ri, "ARI": ari, "NMI": nmi, "AMI": ami,
            "min_cluster_size": min_cluster_size, "max_cluster_size": max_cluster_size, "n_clusters": n_clusters}


def agg_cluster_baseline(data_onehot, inflections_gold, n_inflection_classes):
    agg_labels = cluster.AgglomerativeClustering(
        n_clusters=n_inflection_classes, affinity="manhattan", linkage="average").fit_predict(data_onehot)
    ri, ari, nmi, ami, min_cluster_size, max_cluster_size = eval_results(
        agg_labels, inflections_gold)
    n_clusters = len(list(set(agg_labels)))
    return {"model": "agg", "RI": ri, "ARI": ari, "NMI": nmi, "AMI": ami,
            "min_cluster_size": min_cluster_size, "max_cluster_size": max_cluster_size, "n_clusters": n_clusters}


def kmeans_cluster_baseline(data_onehot, inflections_gold, n_inflection_classes):
    kmeans_labels = cluster.KMeans(
        n_clusters=n_inflection_classes).fit_predict(data_onehot)
    ri, ari, nmi, ami, min_cluster_size, max_cluster_size = eval_results(
        kmeans_labels, inflections_gold)
    n_clusters = len(list(set(kmeans_labels)))
    return {"model": "k-means baseline", "RI": ri, "ARI": ari, "NMI": nmi, "AMI": ami,
            "min_cluster_size": min_cluster_size, "max_cluster_size": max_cluster_size, "n_clusters": n_clusters}


def art(data_onehot, ngram_inventory, inflections_gold, inflection_classes, language, config_string, n_runs=1, vigilances=[], repeat_dataset=False, batch_size=None, shuffle_data=False, visualise_clusters=False, show=False, eval_intervals=False, train_test=False):
    eval_vigilances = False
    if len(vigilances) > 1:
        eval_vigilances = True

    pca = None

    if train_test:
        kf = KFold(n_splits=10, shuffle=True)
        train_test_splits = list(kf.split(data_onehot))
    else:
        train_test_splits = [(None, None)]

    if MULTIPROCESSING:
        if eval_intervals:
            raise ValueError(
                "eval_intervals is not possible in multiprocessing mode.")
        param_settings = [(data_onehot, ngram_inventory, inflections_gold, inflection_classes, pca, language, repeat_dataset, batch_size, shuffle_data, visualise_clusters, show, eval_intervals,
                           train_test, config_string, vig, r, fold_id, train_ix, test_ix) for vig in vigilances for r in range(n_runs) for fold_id, (train_ix, test_ix) in enumerate(train_test_splits)]
        with Pool(processes=N_PROCESSES) as pool:
            # take only first return value
            records_listlist = pool.starmap(
                art_run_parallel_wrapper, param_settings)
    else:  # If multiprocessing is off, this allows to do eval_intervals, which is done once per vigilance
        records_listlist = []
        for vig in vigilances:
            ari_per_interval_per_run = []
            for r in range(n_runs):
                for fold_id, (train_ix, test_ix) in enumerate(train_test_splits):
                    records_batches, plottedIndices_batches, ari_per_interval_batches = art_run_parallel(
                        data_onehot, ngram_inventory, inflections_gold, inflection_classes, pca, language, repeat_dataset, batch_size, shuffle_data, visualise_clusters, show, eval_intervals, train_test, config_string, vig, r, fold_id, train_ix, test_ix)
                    records_listlist.append(records_batches)
                    ari_per_interval_per_run.append(ari_per_interval_batches)

            if eval_intervals:
                if train_test:
                    print(
                        "Line shows #datapoints in whole dataset, not #datapoints in train fold.")
                plot.plot_intervals(ari_per_interval_per_run, plottedIndices_batches, len(data_onehot),
                                    file_label=f"intervals-vig{vig}-{language}_{config_string}", show=show)

    records = list(itertools.chain.from_iterable(records_listlist))

    # Add extra records, same for every vigilance, for k-means baseline clustering baseline, and for real number of inflection classes
    kmeans_record = kmeans_cluster_baseline(
        data_onehot, inflections_gold, len(inflection_classes))
    for vig in vigilances:
        records.append({**{"vigilance": vig}, **kmeans_record})
        records.append({"vigilance": vig, "model": "real inflection classes",
                       "n_clusters": len(inflection_classes)})

    # Convert records to dataframe
    df_results = pd.DataFrame(records)

    # Add ground truth number of inflection classes as extra column, as baseline
    print(df_results.groupby(["vigilance", "mode", "model"], sort=False)[
          ["RI", "ARI", "NMI", "AMI", "min_cluster_size", "max_cluster_size", "n_clusters"]].mean())

    # Only create vigilance plot when comparing multiple vigilances
    if eval_vigilances:
        # Scores plot
        df_melt_scores = pd.melt(df_results, id_vars=["vigilance", "run", "batch", "fold_id", "mode", "model"], value_vars=[
                                 "ARI", "AMI"], var_name="metric", value_name="score")

        if train_test:
            # train_test: Only use ART score (no baseline) and only report ARI
            df_melt_scores = df_melt_scores[df_melt_scores["model"].isin([
                                                                         "ART1"])]
            df_melt_scores = df_melt_scores[df_melt_scores["metric"].isin([
                                                                          "ARI"])]
            ax_scores = sns.lineplot(data=df_melt_scores, x="vigilance",
                                     y="score", hue="mode")
        else:
            df_melt_scores = df_melt_scores[df_melt_scores["model"].isin(
                ["ART1", "k-means baseline"])]
            ax_scores = sns.lineplot(data=df_melt_scores, x="vigilance",
                                     y="score", hue="model", style="metric")
        ax_scores.set_ylim(top=1)
        plt.savefig(os.path.join(
            OUTPUT_DIR, f"scores-{language}-{config_string}.pdf"))
        if show:
            plt.show()
        plt.clf()

        # N_clusters plot
        df_melt_n_clusters = pd.melt(df_results, id_vars=["vigilance", "run", "batch", "fold_id", "mode", "model"], value_vars=[
            "n_clusters"], value_name="# clusters")
        df_melt_n_clusters = df_melt_n_clusters[df_melt_n_clusters["model"].isin(
            ["ART1", "real inflection classes"])]
        # Rename 'model' to 'clustering', because 'real inflection classes' are not really a model
        df_melt_n_clusters = df_melt_n_clusters.rename(
            columns={"model": "clustering"})
        if train_test:
            pass
        else:
            sns.lineplot(data=df_melt_n_clusters, x="vigilance",
                         y="# clusters", hue="clustering")
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
            df_results.to_latex(os.path.join(
                OUTPUT_DIR,
                f"results-{language}-{config_string}.tex"), index=False)


def art_run_parallel_wrapper(*args):
    return art_run_parallel(*args)[0]


def art_run_parallel(data_onehot, ngram_inventory, inflections_gold, inflection_classes, pca, language, repeat_dataset, batch_size_given, shuffle_data, visualise_clusters, show, eval_intervals, train_test, config_string, vig, r, fold_id, train_ix, test_ix):
    print(
        f"Vigilance: {vig}. Run: {r}.{' Split id '+ str(fold_id) if fold_id is not None else ''}")
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

    modes = ["train", "test"] if train_test else ["train"]
    for mode in modes:
        if train_test:
            if mode == "train":
                input_data = data_onehot_full[train_ix]
                clusters_gold = clusters_gold_full[train_ix]
            elif mode == "test":
                input_data = data_onehot_full[test_ix]
                clusters_gold = clusters_gold_full[test_ix]
        else:
            input_data = data_onehot_full
            clusters_gold = clusters_gold_full

        # If batching off, use full dataset (or: full train fold in train-test mode)
        len_data = len(input_data)
        batch_size = batch_size_given if batch_size_given else len_data
        n_batches = len_data//batch_size
        n_reps = 2 if repeat_dataset and mode == "train" else 1
        for rep in range(n_reps):
            if shuffle_data:
                # Shuffle before every repetition, so data gets presented again in different order.
                # Makes taking batches sampling without replacement
                shf = np.random.permutation(len(input_data))
                input_data = input_data[shf]
                clusters_gold = clusters_gold[shf]
            for b in range(n_batches):
                # NOTE: batches right now (without replacement)
                # do exactly the same as processing the whole dataset at once.
                batch = np.arange(b*batch_size, (b+1)*batch_size)
                if mode == "train":
                    clusters_art_batch, prototypes, incrementalClasses, incrementalIndices = artnet.train(
                        input_data[batch], EVAL_INTERVAL)
                elif mode == "test":
                    clusters_art_batch, prototypes = artnet.test(
                        input_data[batch], only_bottom_up=True)

                N_found_clusters = len(prototypes)
                clusters_gold_batch = clusters_gold[batch]
                ri_batch, ari_batch, nmi_batch, ami_batch, min_cluster_size_batch, max_cluster_size_batch = eval_results(
                    clusters_art_batch, clusters_gold_batch)

                if eval_intervals and mode == "train":  # for test, eval_intervals is not useful because nothing is learned
                    if rep == 0:
                        N_evals = len(incrementalClasses)
                        for i in range(0, N_evals):
                            Nsamples = len(incrementalClasses[i])
                            _, ari_interval, _, _, _, _ = eval_results(
                                incrementalClasses[i], clusters_gold_batch[0:Nsamples])
                            ari_per_interval_batches.append(ari_interval)
                            plottedIndices_batches.append(
                                incrementalIndices[i])
                    else:
                        ari_per_interval_batches.append(ari_batch)
                        plottedIndices_batches.append(
                            incrementalIndices[-1]+plottedIndices_batches[-1])
                # Only do results analysis and save record for last repetition
                if rep == n_reps-1:
                    histo = np.histogram(
                        clusters_art_batch, bins=np.arange(0, N_found_clusters+1))[0]
                    order = np.flip(np.argsort(histo))
                    # cluster_population = histo[order]
                    prototypes = prototypes[order, :]
                    S = np.sum(prototypes, axis=0)
                    always_activated_features = np.argwhere(
                        S == N_found_clusters)
                    # category_ngrams is ordered from biggest cluster to smallest, because prototypes has been ordered
                    category_ngrams = []
                    for p in range(0, N_found_clusters):
                        ones = np.nonzero(prototypes[p, :])[0]
                        cluster_ngrams = []
                        for i in ones:
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
                    # Eg. 0th row being [2,3,0,4,6] would mean that cluster 0 (coming out from ART) includes 2 words from inflection class 0, 3 words from IC 1, and so on
                    cluster_inflection_stats = np.zeros(
                        (n_used_clusters, len(inflection_classes)))
                    for i in range(0, len(clusters_gold_int)):
                        cluster_inflection_stats[int(
                            clusters_art_batch[i]), clusters_gold_int[i]] += 1
                    row_sums = cluster_inflection_stats.sum(axis=1)

                    # With multiple batches/repeats, it's possible that on the next batch, no input samples are set into a category created on a previous batch. In this case the new category will be empty.
                    # This is done to avoid division by zero.
                    row_sums[np.where(row_sums == 0)] = 1

                    records_batches.append(
                        {"model": "ART1", "vigilance": vig, "run": r, "fold_id": fold_id, "batch": b, "mode": mode,
                         "RI": ri_batch, "ARI": ari_batch, "NMI": nmi_batch, "AMI": ami_batch,
                         "min_cluster_size": min_cluster_size_batch, "max_cluster_size": max_cluster_size_batch, "n_clusters": N_found_clusters})

    if visualise_clusters and not train_test:
        # NOTE: At the moment not working with train-test, question is then whether train or test phase should be plotted.
        sums_ci_stats = np.sum(cluster_inflection_stats, axis=1)
        # Get indeces from largest cluster to smallest (minus reverses default ascending sorting order)
        order_ci_stats = np.argsort(-sums_ci_stats)
        orderedStats = cluster_inflection_stats[order_ci_stats]
        plot.plot_barchart(orderedStats, inflection_classes, max_clusters=MAX_CLUSTERS_PORTUGUESE if language == "portuguese" else None, min_datapoints_class=MIN_DATAPOINTS_CLASS_PORTUGUESE if language == "portuguese" else None,  # category_ngrams, always_activated_ngrams,
                           file_label=f"{language}-vig{vig}-run{r}_{config_string}", show=show)

        if WRITE_TEX:
            # Write ngrams for clusters to TeX file
            write_table_ngrams(inflection_classes, category_ngrams,
                               orderedStats, language, config_string)

    # print(f"Vigilance: {vig}. Run: {r}. Finished.")
    return records_batches, plottedIndices_batches, ari_per_interval_batches


def write_table_ngrams(inflection_classes, category_ngrams, orderedStats, language, config_string):
    ngrams_records = []
    print(category_ngrams)

    # barHeights counts how big are the clusters, i.e. how many times each n-gram occurs in a cluster
    barHeights = np.sum(orderedStats, axis=1)
    # All features used in all clusters as a list
    all_used_ngrams = list(set(x for l in category_ngrams for x in l))
    # counts per cluster, for all possible features
    sums_per_cluster = np.zeros([len(category_ngrams), len(all_used_ngrams)])

    # Do the counts
    for i, ngrams_per_cluster1 in enumerate(category_ngrams):
        for j, ngram in enumerate(ngrams_per_cluster1):
            sums_per_cluster[i, all_used_ngrams.index(ngram)] = barHeights[i]

    total_per_feature = sum(sums_per_cluster)
    proportions = sums_per_cluster/total_per_feature

    # What proportion of each ngram is in the given cluster. 1 means that the feature is unique to the given cluster
    for cluster_id in range(len(category_ngrams)):
        ngrams_per_cell = defaultdict(list)
        # sort proportions from highest to lowest
        ind_sorted_proportions = np.argsort(-proportions[cluster_id])
        sorted_proportions = -np.sort(-proportions[cluster_id])

        ngrams_shown = 0  # How many features are shown for this cluster by now
        for ind, proportion in enumerate(sorted_proportions):
            # Which ngram is being dealt with now
            ngram_person = all_used_ngrams[ind_sorted_proportions[ind]]
            if proportion > 0:
                ngram_person_split = ngram_person.split("_")
                assert len(ngram_person_split) == 2
                ngram = ngram_person_split[0]
                person = ngram_person_split[1]
                total = int(total_per_feature[ind_sorted_proportions[ind]])
                ngrams_per_cell[person].append(
                    f"\\textit{{{ngram}}} ({proportion:.3f} | {total})")
                ngrams_shown += 1
        ngrams_per_cell_tex_list = [
            f'\\textsc{{{p.lower()}}}: {", ".join(n)}' for p, n in ngrams_per_cell.items()]
        ngrams_per_cell_tex = "\\newline".join(ngrams_per_cell_tex_list)
        if len(ngrams_per_cell_tex) == 0:
            ngrams_per_cell_tex = "--"
        ix_majority_class = np.argmax(orderedStats[cluster_id])
        majority_class_name = inflection_classes[ix_majority_class]
        record = {"cluster": cluster_id, "distinctive n-grams": ngrams_per_cell_tex,
                  "majority class": majority_class_name}
        ngrams_records.append(record)
    ngrams_df = pd.DataFrame(ngrams_records)
    ngrams_df.to_latex(os.path.join(
        OUTPUT_DIR,
        f"ngrams-{language}-{config_string}.tex"), index=False, escape=False, column_format="lp{0.7\linewidth}p{0.2\linewidth}")


def eval_results(results, inflections_gold):
    # Calculate scores
    rand = rand_score(inflections_gold, results)
    adj_rand = adjusted_rand_score(inflections_gold, results)
    norm_mutual_info = normalized_mutual_info_score(inflections_gold, results)
    adj_mutual_info = adjusted_mutual_info_score(inflections_gold, results)
    cluster_sizes = np.bincount(np.array(results, dtype=int))
    min_cluster_size = np.min(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    return rand, adj_rand, norm_mutual_info, adj_mutual_info, min_cluster_size, max_cluster_size
