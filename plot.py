# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns  # from conf import sns
import matplotlib.pyplot as plt
import pandas as pd
from conf import LABEL_DENSITY, FILLED_MARKERS, OUTPUT_DIR, cmap_categorical
import numpy as np
import matplotlib

import os

# sns.set(font="Charis SIL Compact")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.format'] = "pdf"


def fit_pca(data_bin):
    pca = PCA(n_components=2)
    # X=pd.DataFrame(data=data_bin)
    pca = pca.fit(data_bin)
    data_red = pca.transform(data_bin)
    df = pd.DataFrame(data_red)
    df.columns = ['dim1', 'dim2']
    return df, pca


def transform_using_fitted_pca(data_bin, pca):
    data_red = pca.transform(data_bin)
    df = pd.DataFrame(data_red)
    df.columns = ['dim1', 'dim2']
    return df


def plot_data(df, clusters, labels=None, micro_clusters=None, sample_points=None, file_label=None, prototypes=None, show=False):
    print("Start plotting...")
    plot_h(df, clusters, labels=labels, micro_clusters=micro_clusters,
           sample_points=sample_points, prototypes=prototypes, file_label=file_label, show=show)
    print("End plotting.")


def plot_intervals(ari_intervals, incrementalIndices, n_datapoints, file_label=None, show=True):
    # matplotlib.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots()
    stackedResults = np.array(ari_intervals)
    df = pd.DataFrame(data=stackedResults, columns=incrementalIndices)
    df = df.melt(var_name="Number of input paradigms", value_name="ARI")

    sns.lineplot(data=df, x="Number of input paradigms", y="ARI")
    plt.ylim(0)
    ax.axvline(x=n_datapoints, linewidth=2, color='orange', ls=':')
    if file_label:
        plt.savefig(os.path.join(OUTPUT_DIR, f"{file_label}.pdf"))
    if show:
        plt.show()


def plot_barchart(orderedStats, inflection_classes, max_clusters=None, min_datapoints_class=None,  # category_ngrams, always_activated_ngrams,
                  file_label=None, show=False):
    # Only show first n clusters (bars), if this variable is active
    orderedStatsUsed = orderedStats
    if max_clusters is not None and len(orderedStats) > max_clusters:
        orderedStatsUsed = orderedStats[:max_clusters, :]

    # Font scale for the barplot smaller than the default 1.4, so labels fit better
    sns.set_theme(font="Charis SIL Compact", font_scale=1)
    fig, ax = plt.subplots()

    bottom = np.zeros(orderedStatsUsed.shape[0])
    xCoords = list(range(0, orderedStatsUsed.shape[0]))
    for i in range(len(inflection_classes)):
        members = orderedStatsUsed[:, i]
        if min_datapoints_class is not None and members.sum() < min_datapoints_class:
            # Skip this inflection class in bargraph if it has less than minimum datapoints
            print(
                f"Skipping inflection class {inflection_classes[i]} in bar graph, number of datapoints: {members.sum()}")
            continue
        p = ax.bar(xCoords, members, 0.4, bottom=bottom,
                   label=inflection_classes[i], alpha=0.5, color=cmap_categorical[i])
        bottom += members
    ax.legend()
    ax.set_xticks(xCoords)

    if show:
        plt.show()

    if file_label:
        plt.savefig(os.path.join(
            OUTPUT_DIR, f"bar-{file_label}.pdf"), bbox_inches='tight')


def plot_h(df, clusters, labels=None, micro_clusters=None, file_label=None, sample_points=None, prototypes=None, show=False):
    df["clusters"] = [str(a) for a in clusters]
    df = df.sort_values(by=["clusters"])
    if labels is not None:
        df["labels"] = labels
    if micro_clusters is not None:
        df["micro_clusters"] = micro_clusters
    if sample_points is not None:
        samples = np.random.choice(len(clusters), sample_points, replace=False)
        df = df.loc[samples].reset_index(drop=True)
    if micro_clusters is not None:  # THis would in practice be used for lemmas=cognate ids
        micro_clusters_uniq = df["micro_clusters"].unique()
        marker_list = FILLED_MARKERS[:len(micro_clusters_uniq)]
        red_plot = sns.scatterplot(data=df, x="dim1", y="dim2", hue="clusters",
                                   style="micro_clusters", palette="hls", markers=marker_list, size=1, legend="full")
    else:
        red_plot = sns.scatterplot(
            data=df, x="dim1", y="dim2", hue="clusters", palette="hls", size=1, legend="full")
    red_plot.set(xlabel=None)
    red_plot.set(ylabel=None)
    red_plot.set(xticklabels=[])
    red_plot.set(yticklabels=[])
    if labels:
        for i in range(0, len(df.labels), LABEL_DENSITY):
            red_plot.text(df.dim1[i]+0.01, df.dim2[i],
                          df.labels[i], horizontalalignment='left',
                          size=6, color='black')

    if prototypes is not None:
        red_plot = sns.scatterplot(
            data=prototypes, x="dim1", y="dim2", c='k', size=5, legend="full")

    if file_label:
        plt.savefig(os.path.join(OUTPUT_DIR, f"data-{file_label}.pdf"))
    if show:
        plt.show()
    plt.clf()
