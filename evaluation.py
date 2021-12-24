import plot
from sklearn.metrics import silhouette_score

def evaluate_model(model, data_bin, words, model_name):
    if model_name == "ART":
        clusters = model.predict(data_bin)
    else:
        clusters = model.fit_predict(data_bin)
    print(silhouette_score(X=data_bin, labels=clusters, metric="hamming"))
    plot.plot(data_bin, words, clusters, model_name)