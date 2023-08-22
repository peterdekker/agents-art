from conf import ART_VIGILANCE, ART_LEARNING_RATE, INFLECTION_CLASSES, N_INFLECTION_CLASSES, OUTPUT_DIR, MAX_CLUSTERS
import plot
from art import ART1
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


def art(data_onehot, forms, bigram_inventory, inflections_gold, cogids, pca, language, n_runs=1, vigilances=[ART_VIGILANCE], repeat_dataset=False, batch_size=None, shuffle_data=False, data_plot=False, show=False):
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
                n_clusters=MAX_CLUSTERS,
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
                    F=np.array(forms)
                    F=F[shf]
                    clusters_gold = clusters_gold[shf]
                for b in range(n_batches):
                    batch = np.arange(b*batch_size, (b+1)*batch_size)
                    clusters_art_batch, prototypes = artnet.train(input_data[batch], F[batch])
                    clusters_gold_batch = clusters_gold[batch]

                    ri_batch, ari_batch, nmi_batch, ami_batch, min_cluster_size_batch, max_cluster_size_batch = eval_results(
                        clusters_art_batch, clusters_gold_batch)
                        

                    histo=np.histogram(clusters_art_batch, bins=list(np.arange(0,21)))[0]
                    order=np.flip(np.argsort(histo))
                    cluster_population=histo[order]
                    prototypes=prototypes[order,:]
                    category_bigrams=[]
                    for p in range(0,MAX_CLUSTERS):
                        ones=np.nonzero(prototypes[p,:])[0]
                        ones=list(ones)
                        cluster_bigrams=[]
                        for i in ones:
                            cluster_bigrams.append(bigram_inventory[i])
                        category_bigrams.append(cluster_bigrams)

                    clusters_gold_int=[]
                    ORDER=np.array(INFLECTION_CLASSES)
                    for i in range(0,len(clusters_gold_batch)):
                        clusters_gold_int.append(np.where(ORDER==clusters_gold_batch[i])[0][0])
                    # Number of clusters (rows) that are not unused (unused=all 1s)
                    n_used_clusters = np.sum(1-np.all(prototypes,axis=1))

                    # This counts how many of each gold-standard words per each inflection class is clustered in each of the clusters coming from ART
                    # Eg. 0th row being [2,3,0,4,6] would mean that cluster 0 (coming out from ART) includes 2 words from inflection class 'I', 3 words from 'II', and so on
                    cluster_inflection_stats=np.zeros((n_used_clusters,N_INFLECTION_CLASSES))
                    for i in range(0,len(clusters_gold_int)):
                        cluster_inflection_stats[int(clusters_art_batch[i]),clusters_gold_int[i]]+=1
                    row_sums = cluster_inflection_stats.sum(axis=1)
                    
                    #With multiple runs, it's possible that on the next run, no input samples are set into a category created on a previous run. In this case the new category will be empty.
                    #This is done to avoid division by zero.
                    row_sums[np.where(row_sums==0)]=1

                    # Here the counts are changed in percentages
                    cluster_inflection_stats = cluster_inflection_stats / row_sums[:, np.newaxis]
                    records.append(
                    {"vigilance": vig, "run": r, "batch": rep*n_batches+b,
                     "ri": ri_batch, "ari": ari_batch, "nmi": nmi_batch, "ami": ami_batch,
                     "cluster_population": cluster_population,
                     "category_bigrams": category_bigrams,
                     "prototypes": prototypes,
                     "cluster_inflection_stats":cluster_inflection_stats,
                     "min_cluster_size": min_cluster_size_batch, "max_cluster_size": max_cluster_size_batch})



            if data_plot:
                # Use result from last batch to plot TODO: think about this
                df = plot.transform_using_fitted_pca(prototypes, pca)
                # plot.plot_data(df, labels=None, clusters=range(0,20),
                                    #  micro_clusters=cogids[batch], file_label=f"pca-art-vig{vig}-run{r}-{language}", show=show)
                df.columns=['dim1', 'dim2']
                prototype_based_new_coords=[]
                for i in range(0,len(clusters_gold)):
                    prototype_N=int(clusters_art_batch[i])
                    matching_prototype_coord=df.values[prototype_N]
                    with_noise = matching_prototype_coord+np.random.randn(2)*0.02
                    prototype_based_new_coords.append(with_noise)
                
                df2=pd.DataFrame(prototype_based_new_coords)
                df2.columns=['dim1', 'dim2']
                plot.plot_data(df2, labels=None, clusters=clusters_gold, prototypes=df,
                        file_label=f"pca-art-vig{vig}-run{r}-{language}_protos", show=show)
                
            
    df_results = pd.DataFrame(records)
    df_results.to_csv(os.path.join(OUTPUT_DIR, f"histogram_per_vigilance-{language}_out.csv"))
    print(df_results.groupby("vigilance")[["ri", "ari", "nmi", "ami", "min_cluster_size", "max_cluster_size"]].mean())
    df_results_small=df_results[["vigilance", "run", "cluster_population","category_bigrams","cluster_inflection_stats"]]
    df_results_small.to_csv(os.path.join(OUTPUT_DIR, f"cluster_stats.csv"))
    

    # Only create vigilance plot when comparing multiple vigilances
    if eval_vigilances:
        # Plot results

        df_melt_scores = pd.melt(df_results, id_vars=["vigilance", "run", "batch"], value_vars=["ri","ari", "nmi", "ami"], var_name="metric", value_name="score")
        df_melt_clusters = pd.melt(df_results, id_vars=["vigilance", "run", "batch"], value_vars=["min_cluster_size","max_cluster_size"], var_name="metric", value_name="size")
        df_melt_ci = pd.melt(df_results, id_vars=["vigilance"], value_vars=["cluster_population"], var_name="metric", value_name="N_in_cluster")

        from matplotlib import cm

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
            "clusters-scores-art-end.tex", sep="&", lineterminator="\\\\\n")


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
