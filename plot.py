# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from conf import LABEL_DENSITY, FILLED_MARKERS, OUTPUT_DIR
import numpy as np
import matplotlib
# from matplotlib.lines import Line2D
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler
# import prince

import os


plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.format'] = "pdf"

def fit_pca(data_bin):
    pca = PCA(n_components=2)
    # X=pd.DataFrame(data=data_bin)
    pca = pca.fit(data_bin)
    data_red = pca.transform(data_bin)
    df=pd.DataFrame(data_red)
    df.columns=['dim1', 'dim2']
    return df, pca

def transform_using_fitted_pca(data_bin, pca):
    # X=pd.DataFrame(data=data_bin)
    data_red = pca.transform(data_bin)
    df=pd.DataFrame(data_red)
    # df=data_red
    df.columns=['dim1', 'dim2']
    return df

def plot_data(df, clusters, labels=None, micro_clusters=None, sample_points = None, file_label=None, prototypes=None, show=False):
    print("Start plotting...")
    # data_std = StandardScaler().fit_transform(data_bin)

    # plot(data_std, clusters, labels=labels, micro_clusters=micro_clusters, sample_points=sample_points, file_label=file_label, show=show)
    plot_heikki(df, clusters, labels=labels, micro_clusters=micro_clusters, sample_points=sample_points, prototypes=prototypes, file_label=file_label, show=show)
    print("End plotting.")
    #score = silhouette_score(X=data_bin, labels=clusters, metric="hamming")
    # return score

def plot_intervals(ari_intervals, incrementalIndices, n_datapoints, file_label=None, show=True):
    matplotlib.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots()
    stackedResults=np.array(ari_intervals)
    # cols=list(map(str, incrementalIndices))
    df = pd.DataFrame(data=stackedResults, columns=incrementalIndices)
    df = df.melt(var_name="Number of input paradigms", value_name="ARI")

    sns.lineplot(data=df,x="Number of input paradigms", y="ARI")
    plt.ylim(0)
    ax.axvline(x=n_datapoints, linewidth=2, color='orange', ls=':')
    if file_label:
        plt.savefig(os.path.join(OUTPUT_DIR,f"{file_label}.pdf"))
    if show:
        plt.show()

def plot_barchart(cluster_inflection_stats, inflection_classes, #category_ngrams, always_activated_ngrams,
                        file_label=None, show=False):
    sums=np.sum(cluster_inflection_stats,axis=1)
    order=np.argsort(-sums) #Get indeces from largest cluster to smallest (minus reverses default ascending sorting order)
    orderedStats=cluster_inflection_stats[order]
    matplotlib.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots()
    
    
    
    bottom = np.zeros(orderedStats.shape[0])
    xCoords=list(range(0,orderedStats.shape[0]))

    for i in range(len(inflection_classes)):
        members=orderedStats[:,i]
        p = ax.bar(xCoords, members, 0.4, bottom=bottom, label=inflection_classes[i],alpha=0.5)
        bottom += members
    ax.legend()
    ax.set_xticks([])
    
    # for bar in range(orderedStats.shape[0]):
    #     for ngram in range(len(category_ngrams[bar])):
    #         if category_ngrams[bar][ngram] in always_activated_ngrams:
    #             #Is not unique feature
    #             plt.text(bar-0.30, ngram*6+5, category_ngrams[bar][ngram], fontsize=20)
    #         else:
    #             #Is unique feature
    #             plt.text(bar-0.30, ngram*6+5, category_ngrams[bar][ngram], fontsize=20, weight='bold')
    
    # category_ngrams
    # plt.figure(figsize=(10,6))
    if show:
        plt.show()
        
    if file_label:
        plt.savefig(os.path.join(OUTPUT_DIR,f"barchart-{file_label}.pdf"),bbox_inches='tight')

def plot_heikki(df, clusters, labels=None, micro_clusters = None, file_label=None, sample_points=None, prototypes=None, show=False):
    # assert len(clusters) == data_bin.shape[0]
    # alg = TSNE(n_components=2, metric="hamming", init="pca", learning_rate="auto")
    # data_red = alg.fit_transform(data_standardized)
    # clusters=np.sort(clusters)
    df["clusters"] = [str(a) for a in clusters]
    df = df.sort_values(by=["clusters"])
    if labels is not None:
        df["labels"] = labels
    if micro_clusters is not None:
        df["micro_clusters"] = micro_clusters
    if sample_points is not None:
        samples = np.random.choice(len(clusters), sample_points, replace=False)
        df = df.loc[samples].reset_index(drop=True)
    if micro_clusters is not None: # THis would in practice be used for lemmas=cognate ids
        micro_clusters_uniq = df["micro_clusters"].unique()
        marker_list = FILLED_MARKERS[:len(micro_clusters_uniq)]
        red_plot = sns.scatterplot(data = df, x="dim1", y="dim2", hue="clusters", style="micro_clusters", palette="hls", markers=marker_list, size=1, legend="full")
    else:
        red_plot = sns.scatterplot(data = df, x="dim1", y="dim2", hue="clusters", palette="hls", size=1, legend="full")
        # red_plot = plt.pie(data = d f, labels = labels, colors = colors,
    red_plot.set(xlabel=None)
    red_plot.set(ylabel=None)
    red_plot.set(xticklabels=[])
    red_plot.set(yticklabels=[])
    if labels:
        for i in range(0,len(df.labels), LABEL_DENSITY):
            red_plot.text(df.dim1[i]+0.01, df.dim2[i], 
            df.labels[i], horizontalalignment='left', 
            size=6, color='black')
    
    if prototypes is not None:
        red_plot = sns.scatterplot(data = prototypes, x="dim1", y="dim2", c='k', size=5, legend="full")
    
    if file_label:
        plt.savefig(os.path.join(OUTPUT_DIR,f"data-{file_label}.pdf"))
    if show:
        plt.show()
    plt.clf()
