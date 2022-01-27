from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from conf import LABEL_DENSITY

plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.format'] = "pdf"



def plot(data_standardized, labels, clusters, file_label=None, sample_points=None, use_labels=True):
    assert len(clusters) == data_standardized.shape[0]
    alg = TSNE(n_components=2)
    data_red = alg.fit_transform(data_standardized)
    df = pd.DataFrame(data=data_red, columns=['pc1', 'pc2'])
    df["labels"] = labels
    df["clusters"] = clusters
    if sample_points:
        samples = np.random.choice(len(clusters), sample_points, replace=False)
        df = df.loc[samples].reset_index(drop=True)
    pca_plot = sns.scatterplot(data = df, x="pc1", y="pc2", hue="clusters", legend = False, palette="hls", size=1)
    if use_labels:
        for i in range(0,len(df.labels), LABEL_DENSITY):
            pca_plot.text(df.pc1[i]+0.01, df.pc2[i], 
            df.labels[i], horizontalalignment='left', 
            size=6, color='black')
    plt.savefig(f"{file_label}-pca.pdf")
    #os.path.join(output_dir, f"{variable_param}-end-sb.{IMG_FORMAT}")
