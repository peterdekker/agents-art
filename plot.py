from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.format'] = "pdf"


def plot(vectors, labels, clusters):
    n_labels = len(labels)
    assert n_labels == len(clusters) == vectors.shape[0]
    data_standardized = StandardScaler().fit_transform(vectors)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_standardized)
    df = pd.DataFrame(data=data_pca, columns=['pc1', 'pc2'])
    df["labels"] = labels
    df["clusters"] = clusters
    pca_plot = sns.scatterplot(data = df, x="pc1", y="pc2", hue="clusters", legend = False, palette="hls")
    for i in range(0,n_labels):
        pca_plot.text(df.pc1[i]+0.01, df.pc2[i], 
        labels[i], horizontalalignment='left', 
        size='small', color='black')
    plt.savefig("pca.pdf")
    #os.path.join(output_dir, f"{variable_param}-end-sb.{IMG_FORMAT}")
