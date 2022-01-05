from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from conf import LABEL_DENSITY

plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.format'] = "pdf"



def plot(data_standardized, labels, clusters, file_label):
    n_labels = len(labels)
    assert n_labels == len(clusters) == data_standardized.shape[0]
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_standardized)
    df = pd.DataFrame(data=data_pca, columns=['pc1', 'pc2'])
    df["labels"] = labels
    df["clusters"] = clusters
    pca_plot = sns.scatterplot(data = df, x="pc1", y="pc2", hue="clusters", legend = False, palette="hls", size=1)
    for i in range(0,n_labels, LABEL_DENSITY):
        pca_plot.text(df.pc1[i]+0.01, df.pc2[i], 
        labels[i], horizontalalignment='left', 
        size=6, color='black')
    #plt.savefig(f"{file_label}-pca.pdf")
    #os.path.join(output_dir, f"{variable_param}-end-sb.{IMG_FORMAT}")
