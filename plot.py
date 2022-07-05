from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

import os

from conf import LABEL_DENSITY, INFLECTION_CLASSES, FILLED_MARKERS, OUTPUT_DIR

plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.format'] = "pdf"



def plot(data_standardized, clusters, labels=None, micro_clusters = None, file_label=None, sample_points=None, show=False):
    assert len(clusters) == data_standardized.shape[0]
    alg = TSNE(n_components=2, metric="hamming", init="pca", learning_rate="auto", square_distances=True)
    data_red = alg.fit_transform(data_standardized)
    df = pd.DataFrame(data=data_red, columns=['dim1', 'dim2'])
    df["clusters"] = clusters
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
        red_plot = sns.scatterplot(data = df, x="dim1", y="dim2", hue="clusters", style="micro_clusters", hue_order=INFLECTION_CLASSES, palette="hls", markers=marker_list, size=1, legend=False)
    else:
        red_plot = sns.scatterplot(data = df, x="dim1", y="dim2", hue="clusters", hue_order=INFLECTION_CLASSES, palette="hls", size=1, legend=False)
    red_plot.set(xlabel=None)
    red_plot.set(ylabel=None)
    red_plot.set(xticklabels=[])
    red_plot.set(yticklabels=[])
    if labels:
        for i in range(0,len(df.labels), LABEL_DENSITY):
            red_plot.text(df.dim1[i]+0.01, df.dim2[i], 
            df.labels[i], horizontalalignment='left', 
            size=6, color='black')
    if file_label:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        plt.savefig(os.path.join(OUTPUT_DIR,f"data-{file_label}.pdf"))
    if show:
        plt.show()
    plt.clf()
